// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! A `SubHeap`: an independently-budgeted partition of the reservation.
//!
//! Each sub-heap owns a disjoint slice of the locked arena (PartitionAlloc-style
//! isolation: one sub-heap's exhaustion can't starve another). Within its slice
//! it lays out per-CPU slabs (pointer stacks per size class) and a central free
//! list per class. Because everything is reserved up front, the "slow path"
//! (per-CPU stack empty/full) never calls the kernel — it just moves pointers
//! between the per-CPU stack and the central list, both already-resident.
//!
//! Phase 1 scope: single-thread-correct alloc/dealloc on the locked baseline,
//! central free lists carved from the slice, per-sub-heap byte budget. Remote
//! free, the supervisor, and the rseq fast path arrive in later phases.

use crate::sizeclass::{self};
use crate::sys::Reservation;
use crate::rseq::slab::{ClassLoc, CpuStack, Fast, Header, SlabLayout};
use core::sync::atomic::Ordering;
use std::ptr::NonNull;
use std::sync::Mutex;

/// Exhaustion policy for a sub-heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnExhaust {
    /// Return `None` from `alloc` (fallible front-end).
    None,
    /// Abort the process (global-allocator hard floor). Implemented by the
    /// front-end, not here; the core always returns `None` and lets the caller
    /// decide.
    Abort,
}

/// Central free list for one `(home shard, size class)`: an **intrusive stack**
/// of free objects behind a tiny spin lock. Each free object's first 8 bytes hold
/// the `next` pointer (every class is >= 8 bytes), so the stack needs **zero heap
/// allocation** of its own — critical when toccata is the global allocator,
/// where a `Vec`-backed list would re-enter `alloc` and deadlock.
///
/// This is the cross-thread meeting point of the producer/consumer path. Sharding
/// it by a *stable per-thread home* (a producer always refills from, and the
/// consumers of its objects always deposit into, the same shard) keeps each
/// `(shard, class)` lock essentially uncontended across independent pairs.
///
/// We keep the spin lock rather than a lock-free Treiber stack deliberately: the
/// lock is uncontended across pairs, and *within* a pair the single producer
/// (popping) and its consumers (pushing) share this one structure — a measured
/// head-to-head showed a Treiber `pop` CAS-loop *retries* under that
/// producer/consumer race and ends up slower than a spin lock that serializes the
/// two cleanly (64B prod/cons 24 ns lock-free vs 19 ns locked on Graviton). The
/// lock's critical region is a handful of instructions and never sleeps. Batched
/// deposit/refill (`push_chain` / multi-`pop` under one acquire) amortize it.
/// Central free pool for one `(home shard, size class)`: the head of an
/// **active-span list** (spans owning ≥1 free object of this class), guarded by a
/// tiny spin lock. The free objects themselves are tracked **out of band** in
/// per-span bitmaps ([`super::meta::SpanBitmaps`]) — a cross-thread free SETS its
/// slot's bit (no read of the object) and a refill SCANS bits + computes addresses
/// (`span_base + slot*osz`, no read), so handing memory between a producer and
/// consumer core never touches the (cold, cross-core) freed object — eliminating
/// the serial cache-miss chain an intrusive free-list incurs, and returning
/// objects in ascending-address order (good producer-reuse locality). This is the
/// jemalloc/tcmalloc slab-bitmap model.
///
/// Sharded by a *stable per-thread home* so independent producer/consumer pairs
/// hit disjoint locks. All fields (head + every owned span's bitmap/link) are
/// touched only by this lock's holder, so they are plain `UnsafeCell`.
struct CentralList {
    /// Active-span list head (span index, or `NO_SPAN` if no free objects here).
    head: core::cell::UnsafeCell<u32>,
    lock: core::sync::atomic::AtomicU32,
    /// **Reuse hint**: the span a cross-thread free most recently deposited into
    /// (relaxed, written lock-free by the consumer's deposit, `NO_SPAN` = none). The
    /// producer's refill scans this span *first* so it re-hands-out the buffers the
    /// consumer most recently returned — the hottest (still-cached) objects —
    /// shrinking the reuse distance that otherwise marches linearly through a span
    /// and spills the working set to DRAM (toccata's residual LLC-miss gap to
    /// jemalloc at the memset-heavy larger prod/cons sizes). Purely an ordering hint:
    /// correctness never depends on it (the active-list walk still covers every span).
    reuse_hint: core::sync::atomic::AtomicU32,
}

// SAFETY: head + the owned spans' bitmaps are only touched under `lock`.
unsafe impl Send for CentralList {}
unsafe impl Sync for CentralList {}

impl CentralList {
    const fn new() -> Self {
        Self {
            head: core::cell::UnsafeCell::new(super::meta::NO_SPAN),
            lock: core::sync::atomic::AtomicU32::new(0),
            reuse_hint: core::sync::atomic::AtomicU32::new(super::meta::NO_SPAN),
        }
    }

    #[inline]
    fn lock(&self) -> CentralGuard<'_> {
        while self
            .lock
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            while self.lock.load(Ordering::Relaxed) != 0 {
                core::hint::spin_loop();
            }
        }
        CentralGuard(self)
    }

    /// Acquire the lock only if uncontended. Used by the background reclaim sweep so
    /// it **never blocks a producer**: if a refill/drain holds this cell's lock, the
    /// sweep skips the cell and revisits it next tick. This keeps the supervisor off
    /// the producer/consumer critical path (a blocking acquire here spiked prod/cons
    /// latency — the sweep would serialize behind every refill).
    #[inline]
    fn try_lock(&self) -> Option<CentralGuard<'_>> {
        self.lock
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| CentralGuard(self))
    }
}

struct CentralGuard<'a>(&'a CentralList);

impl Drop for CentralGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.lock.store(0, Ordering::Release);
    }
}

/// A shared bump cursor over the sub-heap's object arena. Hands out `osz`-sized,
/// `MIN_ALIGN`-aligned chunks to whichever class asks first. Never frees back to
/// the arena (returned objects go to the per-class [`CentralList`] free list);
/// the budget meter bounds total live bytes regardless.
struct BumpArena {
    /// Bump cursor (next free byte), advanced by a lock-free CAS. Using an atomic
    /// rather than a `Mutex` means carving never `futex`-sleeps — important under
    /// the producer/consumer mix where several threads carve fresh runs for
    /// different classes at once (a blocking mutex there caused scheduling
    /// perturbation / run-to-run variance), and it keeps carving on the
    /// never-stall path (no kernel blocking).
    cursor: core::sync::atomic::AtomicUsize,
    end: usize,
    /// Arena base — the **span-grid origin**. The span table, per-span bitmaps, and
    /// `LargeAllocator::span_index`/`span_ptr` all index spans **relative to this**
    /// (`span = (ptr - base) >> SPAN_BITS`). `arena_base` is only page-aligned, not
    /// necessarily 64 KiB span-aligned, so `carve_run` must align large runs to
    /// `base + k*SPAN_BYTES` (this grid), NOT to absolute address span boundaries —
    /// otherwise a large run's pointer wouldn't round-trip through `span_index`/
    /// `span_ptr` and the coalescing pool would hand back a shifted, overlapping run.
    base: usize,
}

// SAFETY: the cursor walks the owned reservation via atomics.
unsafe impl Send for BumpArena {}
unsafe impl Sync for BumpArena {}

impl BumpArena {
    /// Claim a fresh run of up to `num_spans` **whole spans** for `class`, homed on
    /// `home_cpu`, assign their ownership in `spans`, and return the span-aligned
    /// `[start, run_end)`. The caller tiles objects within the run and marks their
    /// bitmap slots (the arena has no bitmap access). Returns `(start, start)` when
    /// the arena is drained.
    ///
    /// Carving is **always span-granular**: the cursor starts span-aligned (at
    /// `arena_base`, the span grid origin) and only ever advances by whole spans, so
    /// every carved object can be tiled entirely within one span (or, for classes
    /// larger than a span, within a whole-span group) — no object ever straddles a
    /// span boundary. That is what keeps the bitmap's `slot ↔ address` mapping
    /// (`slot = (ptr - span_base) / osz`, `addr = span_base + slot*osz`) exact for
    /// every size class. At true exhaustion the tail (< 1 span) is left unused.
    fn carve_spans(
        &self,
        num_spans: usize,
        class: u16,
        home_cpu: u16,
        spans: &super::meta::SpanTable,
    ) -> (usize, usize) {
        let span_bytes = super::meta::SPAN_BYTES;
        // Lock-free claim: CAS the cursor forward by whole spans (clamped to the
        // whole spans remaining). Each class carves whole spans, never sharing a
        // span with another class — what makes the span table's ptr→class/home
        // lookup and the per-span bitmap sound.
        let mut start = self.cursor.load(Ordering::Relaxed);
        let run_end = loop {
            let avail_spans = self.end.saturating_sub(start) / span_bytes;
            if avail_spans == 0 {
                return (start, start); // arena drained (no whole span left)
            }
            let take_spans = num_spans.min(avail_spans);
            let end = start + take_spans * span_bytes;
            match self.cursor.compare_exchange_weak(start, end, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break end,
                Err(s) => start = s,
            }
        };
        spans.assign_range(start as *const u8, run_end - start, class, home_cpu);
        (start, run_end)
    }

    /// High-water of bytes ever carved from the arena. The cursor only advances
    /// (carving never frees back — returned objects go to per-class free lists),
    /// so this is monotonic: toccata's true touched footprint, the analog of
    /// process RSS for an allocator that doesn't `mlock` its budget up front.
    /// `budget` is the arena's span (the arena ends `budget` bytes past its base).
    fn carved_bytes(&self, budget: usize) -> usize {
        let base = self.end - budget;
        self.cursor.load(Ordering::Relaxed).saturating_sub(base)
    }

    /// Carve a single span-aligned run of exactly `run_bytes` (a multiple of
    /// SPAN_BYTES) for a large allocation. Returns null if the arena is drained.
    fn carve_run(&self, run_bytes: usize) -> Option<*mut u8> {
        let span_bytes = super::meta::SPAN_BYTES;
        debug_assert_eq!(run_bytes % span_bytes, 0);
        // Lock-free: align up to a span boundary **relative to the arena base** (the
        // span-grid origin) and CAS the cursor forward. Aligning to the arena-
        // relative grid — not absolute address boundaries — is what keeps a large
        // run's pointer round-tripping exactly through `LargeAllocator::span_index`/
        // `span_ptr` (`span = (ptr - base) >> SPAN_BITS`), so the coalescing SpanPool
        // hands the same address back and never produces a shifted, overlapping run.
        // (`base` is only page-aligned, so absolute alignment would diverge from the
        // index grid whenever `base % SPAN_BYTES != 0`.)
        let mut cur = self.cursor.load(Ordering::Relaxed);
        loop {
            let rel = cur - self.base;
            let start = self.base + rel.div_ceil(span_bytes) * span_bytes;
            let end = start.checked_add(run_bytes)?;
            if end > self.end {
                return None;
            }
            match self.cursor.compare_exchange_weak(cur, end, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Some(start as *mut u8),
                Err(c) => cur = c,
            }
        }
    }
}

/// The span-table class marker for large (oversize) allocations — distinct from
/// any real size class. Large objects are span-aligned runs carved from the
/// arena; their exact size is recorded in [`LargeAllocator`].
const LARGE_CLASS: u16 = u16::MAX - 1;

/// Cache-line-padded per-CPU accounting cell. The hot path bumps only its own
/// CPU's cell (relaxed add), so 64 cores don't contend a single line — the
/// design's per-CPU lease accounting (§8.3). The exact total is the sum across
/// cells, computed only on a `live_*` query (rare). The hard budget limit is
/// already enforced structurally by the bounded arena, so the hot path needs no
/// global limit-check atomic at all.
/// Each cell splits into a **local** part (written only by the owning CPU's
/// fast path with a plain non-atomic add — no RMW round-trip, the snmalloc/
/// tcmalloc approach) and a **remote** part (written by cross-CPU frees with an
/// atomic). A query sums `local + remote` across all cells. The local part isn't
/// atomic because, in the common case, only the thread currently on CPU `i`
/// touches cell `i`'s local part; a migration mid-add can at worst lose/double a
/// single delta in the *instrumentation* (not memory state), and the
/// supervisor's periodic reconcile corrects drift. This removes the per-op
/// atomic that was toccata's gap to jemalloc.
#[repr(C, align(128))]
struct CpuCounter {
    local_bytes: core::cell::UnsafeCell<i64>,
    local_objs: core::cell::UnsafeCell<i64>,
    remote_bytes: core::sync::atomic::AtomicI64,
    remote_objs: core::sync::atomic::AtomicI64,
}

impl CpuCounter {
    const fn new() -> Self {
        Self {
            local_bytes: core::cell::UnsafeCell::new(0),
            local_objs: core::cell::UnsafeCell::new(0),
            remote_bytes: core::sync::atomic::AtomicI64::new(0),
            remote_objs: core::sync::atomic::AtomicI64::new(0),
        }
    }
}

/// Per-CPU counters for a sub-heap, indexed by CPU.
struct PerCpuCounters {
    cells: crate::SysBoxSlice<CpuCounter>,
}

// SAFETY: the non-atomic `local_*` cells are only written by the owning CPU's
// fast path; reads (queries) use a relaxed volatile and tolerate transient skew.
unsafe impl Sync for PerCpuCounters {}

impl PerCpuCounters {
    fn new(num_cpus: u32) -> Self {
        Self { cells: crate::sys_boxed_slice(num_cpus.max(1) as usize, |_| CpuCounter::new()) }
    }

    #[inline]
    fn cell(&self, cpu: u32) -> &CpuCounter {
        &self.cells[(cpu as usize).min(self.cells.len() - 1)]
    }

    /// Local fast-path update (owning CPU only): a plain non-atomic add. No RMW.
    #[inline]
    fn add_local(&self, cpu: u32, bytes: i64, objs: i64) {
        let c = self.cell(cpu);
        // SAFETY: only the thread currently on `cpu` updates `cpu`'s local cell.
        unsafe {
            let b = c.local_bytes.get();
            *b = (*b).wrapping_add(bytes);
            let o = c.local_objs.get();
            *o = (*o).wrapping_add(objs);
        }
    }

    /// Cross-CPU update (remote free charging the home CPU): atomic.
    #[inline]
    fn add_remote(&self, cpu: u32, bytes: i64, objs: i64) {
        let c = self.cell(cpu);
        c.remote_bytes.fetch_add(bytes, Ordering::Relaxed);
        c.remote_objs.fetch_add(objs, Ordering::Relaxed);
    }

    fn total_bytes(&self) -> usize {
        self.cells
            .iter()
            .map(|c| unsafe { core::ptr::read_volatile(c.local_bytes.get()) }
                + c.remote_bytes.load(Ordering::Relaxed))
            .sum::<i64>()
            .max(0) as usize
    }
    fn total_objs(&self) -> u64 {
        self.cells
            .iter()
            .map(|c| unsafe { core::ptr::read_volatile(c.local_objs.get()) }
                + c.remote_objs.load(Ordering::Relaxed))
            .sum::<i64>()
            .max(0) as u64
    }
}

/// Allocator for requests larger than `MAX_SMALL`. Carves whole-span runs from
/// the same bump arena (so everything stays in the one mlock'd reservation).
///
/// Recycling has two tiers, both keyed by span-count and built once at init
/// (**zero per-alloc heap allocation**, safe as the global allocator):
/// 1. an **exact-count L0 cache** ([`LargeInner::free_by_spans`]) probed first —
///    a freed N-span run is re-handed-out for the next N-span request with no
///    work, which preserves toccata's best-in-class stable-size behavior
///    (`frag match` 0.98×);
/// 2. a [`SpanPool`](crate::spanpool::SpanPool) coalescing free-run index used on
///    an L0 miss — it merges address-adjacent free runs and splits a larger run
///    to serve a smaller request, so a *drifting* large size no longer carves
///    fresh arena every generation (`frag shift` 5.99× → ~1×).
///
/// Mutex-guarded; oversize allocs are rare and off the small-object hot path.
struct LargeAllocator {
    inner: Mutex<LargeInner>,
    /// Span-indexed: for the span at the start of a live run, its span-count.
    /// 0 = not a live-run head. Built once; indexed by `(ptr-arena)>>SPAN_BITS`.
    /// `swap(0, AcqRel)` on free also serializes a run's alloc vs free.
    run_spans: crate::SysBoxSlice<core::sync::atomic::AtomicU32>,
    arena_base: usize,
}

struct LargeInner {
    /// The single source of truth for free runs: an address-ordered, coalescing
    /// free-run index over the span grid ([`SpanPool`](crate::spanpool::SpanPool)).
    /// Its per-span-count buckets give **exact-fit-first** for free (an N-span
    /// request is served from the exact-N bucket with no split when one exists —
    /// preserving `frag match`), while a drifting size falls through to the next
    /// larger bucket and splits, and frees coalesce address-adjacent runs. Pure
    /// span-index arithmetic — never reads a freed run's own memory.
    pool: crate::spanpool::SpanPool,
}

// SAFETY: all access is mutex-guarded; pointers are into the owned reservation.
unsafe impl Send for LargeAllocator {}
unsafe impl Sync for LargeAllocator {}

impl LargeAllocator {
    fn new(arena_base: *mut u8, arena_bytes: usize) -> Self {
        let num_spans = arena_bytes.div_ceil(super::meta::SPAN_BYTES).max(1);
        let run_spans =
            crate::sys_boxed_slice(num_spans, |_| core::sync::atomic::AtomicU32::new(0));
        // Coalescing free-run index over the arena's span grid (built once).
        let pool = crate::spanpool::SpanPool::new(num_spans);
        Self {
            inner: Mutex::new(LargeInner { pool }),
            run_spans,
            arena_base: arena_base as usize,
        }
    }

    #[inline]
    fn span_index(&self, ptr: *const u8) -> usize {
        (ptr as usize - self.arena_base) >> super::meta::SPAN_BITS
    }

    /// Inverse of [`span_index`](Self::span_index): the base pointer of span
    /// `idx` in this arena. Used to turn a [`SpanPool`] head index back into a
    /// run pointer.
    #[inline]
    fn span_ptr(&self, idx: u32) -> *mut u8 {
        (self.arena_base + ((idx as usize) << super::meta::SPAN_BITS)) as *mut u8
    }
}

/// Per-class refill/drain batch size (how many pointers move between a per-CPU
/// stack and the central list per slow-path event). Mirrors tcmalloc num_to_move.
fn batch_for(class: usize) -> usize {
    let sz = sizeclass::size_of_class(class);
    match sz {
        0..=128 => 64,
        129..=512 => 32,
        513..=2048 => 16,
        2049..=8192 => 8,
        _ => 4,
    }
}

/// Per-(CPU, class) L2 slab capacity, **tiered by object size** (tcmalloc/jemalloc
/// shape: a smaller per-CPU cache for larger objects). A flat cap pins up to `cap`
/// whole spans per class in L2 even after the app stops using that class — for the
/// largest classes (one object per span or more) that is `cap` stranded spans,
/// which dominated the `frag crossclass` residual. Shrinking the cache for big
/// objects bounds that pinning (e.g. 64 KiB class: 24 spans, not 256) while leaving
/// the small-object cache large where the per-op cache-hit rate matters for speed.
/// `base` is the configured ceiling (the small-class cap); larger classes scale it
/// down. Capped below `base` so an explicit small `cap_per_class` is still honored.
fn cap_for_class(class: usize, base: u32) -> u32 {
    let sz = sizeclass::size_of_class(class);
    let tier = match sz {
        0..=512 => base,            // small: full cache (hot-path hit rate)
        513..=4096 => base / 2,     // medium
        4097..=32768 => base / 4,   // large-ish
        _ => 24,                    // 48/64/96/128/192/256 KiB: a couple-dozen spans
    };
    tier.min(base).max(8)
}

/// Process-global counter handing out dense, stable per-thread shard ids.
static NEXT_SHARD_ID: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// A **stable per-thread** shard id, assigned once (round-robin via the global
/// counter) and cached in a `const`-init, `Drop`-free TLS cell. The cell shape is
/// the recursion-safe one toccata's global allocator already relies on (direct
/// native-TLS load, no lazy-init guard, no `__cxa_thread_atexit` registration, no
/// allocation), so reading it never re-enters the allocator.
///
/// This is the load-bearing fix for the producer/consumer path: object "home" is
/// keyed on this thread shard, **not** the current CPU. A CPU home is sprayed by
/// the scheduler — a thread that migrates strands its returned memory on the old
/// CPU and is forced to carve fresh arena under a global lock (toccata's measured
/// prod/cons collapse). A thread shard is migration-invariant: a producer's freed
/// objects always flow back to the one meeting point that same producer refills
/// from, so independent producer/consumer pairs never collide — the property that
/// makes jemalloc/tcmalloc/mimalloc scale flat on this workload.
#[inline]
fn thread_shard_id() -> u32 {
    thread_local! {
        static SHARD_ID: core::cell::Cell<u32> = const { core::cell::Cell::new(u32::MAX) };
    }
    SHARD_ID.with(|c| {
        let v = c.get();
        if v != u32::MAX {
            return v;
        }
        let id = NEXT_SHARD_ID.fetch_add(1, Ordering::Relaxed);
        c.set(id);
        id
    })
}

/// A sub-heap. Created by the registry/builder during `configure()`; handed to
/// consumers as a `&'static SubHeap` (or via a `Copy` handle in later phases).
pub struct SubHeap {
    name: &'static str,
    on_exhaust: OnExhaust,

    /// The per-CPU slab layout for this sub-heap (pointer stacks).
    slab: SlabLayout,
    /// Central free lists, **sharded by home**: a flat `num_shards * NUM_CLASSES`
    /// array indexed by `shard * NUM_CLASSES + class` (see [`SubHeap::central`]).
    ///
    /// This replaces the single global-per-class list that serialized every
    /// producer/consumer pair on one spinlock. Object home is a *stable per-thread*
    /// shard (not the current CPU — see [`thread_shard_id`]); a producer refills
    /// from `central[its_shard][class]` and the consumers of its objects deposit
    /// there, so each pair meets on a distinct lock and independent pairs never
    /// contend. This is jemalloc's per-arena-bin / tcmalloc's transfer-cache shape:
    /// a *plentiful* meeting point keyed on a migration-invariant owner.
    central: crate::SysBoxSlice<CentralList>,
    /// Out-of-band per-span free bitmaps backing the central pools (the jemalloc/
    /// tcmalloc slab-bitmap model — see [`CentralList`]).
    bitmaps: super::meta::SpanBitmaps,
    /// Number of home shards (`central` has this many rows). Defaults to a small
    /// multiple of the CPU count so producer/consumer pairs rarely collide.
    num_shards: u32,
    /// Shared bump arena fresh objects are carved from on demand.
    arena: BumpArena,
    /// Span ownership table: recover (class, home_shard) from any object pointer.
    spans: super::meta::SpanTable,
    /// Allocator for requests larger than `MAX_SMALL`.
    large: LargeAllocator,

    /// Hard byte budget for this sub-heap. Enforced structurally by the bounded
    /// arena (carve returns empty at the limit), so the hot path needs no global
    /// limit-check atomic.
    budget_bytes: usize,
    /// Per-CPU live bytes/objects counters (Layer-1 instrumentation). Bumped on
    /// the owning CPU's padded cell on the hot path — no global atomic contention.
    counters: PerCpuCounters,

    /// Keeps the backing reservation slice alive for the sub-heap's lifetime.
    /// `None` for sub-heaps that borrow a shared reservation (set by the owner).
    _reservation: Option<Reservation>,
}

// SAFETY: all interior mutability is atomic or mutex-guarded; pointers are into
// the owned reservation.
unsafe impl Send for SubHeap {}
unsafe impl Sync for SubHeap {}

impl SubHeap {
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }
    #[inline]
    pub fn on_exhaust(&self) -> OnExhaust {
        self.on_exhaust
    }
    #[inline]
    pub fn live_bytes(&self) -> usize {
        self.counters.total_bytes()
    }
    #[inline]
    pub fn live_objects(&self) -> u64 {
        self.counters.total_objs()
    }
    #[inline]
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Bytes ever carved from the bump arena — the monotonic high-water that only
    /// grows as fresh spans/runs are claimed and **never retreats** when objects
    /// are freed (frees go to per-class lists / the large free-by-span stacks, not
    /// back to the arena). Because toccata `mlock`s and populates its whole budget
    /// at init, process RSS is pinned at the budget and cannot reveal
    /// fragmentation; this internal high-water is toccata's faithful analog of the
    /// "peak touched footprint" that shows up as peak RSS in a `madvise`-based
    /// allocator. Compare against [`live_bytes`](Self::live_bytes): the ratio
    /// `carved / live` is toccata's fragmentation / stranding factor.
    #[inline]
    pub fn carved_bytes(&self) -> usize {
        self.arena.carved_bytes(self.budget_bytes)
    }

    /// The central free list for `(shard, class)`. `shard` is a home shard id
    /// (clamped into range); `class` a size class. Sharding the central list by
    /// home is what lets independent producer/consumer pairs avoid one shared lock.
    #[inline]
    fn central(&self, shard: u32, class: usize) -> &CentralList {
        let s = (shard % self.num_shards) as usize;
        &self.central[s * sizeclass::NUM_CLASSES + class]
    }

    /// **Lock-free** deposit of one freed object into its home shard's bitmap pool:
    /// set its slot's free bit with a single atomic `fetch_or` — no central lock and
    /// no read of the object. The owning producer reclaims it on its next refill
    /// (which clears the bit under the lock). This removes the producer/consumer
    /// collision on the central spin lock that made the steady-state cross-thread
    /// latency bimodal: a consumer flushing freed buffers never blocks on, or bounces
    /// the cache line of, the producer's refill.
    ///
    /// Correctness without a lock rests on two invariants: (1) the object's span was
    /// carved — and therefore linked onto the active list — before the object was
    /// ever handed out, so the bit's span is already reachable from the producer's
    /// walk; (2) spans are **never unlinked** (`on_list` is monotonic), so a deposit
    /// never needs to (re)link and can never race an unlink into a stranded free.
    ///
    /// # Safety
    /// `obj` is a freed `osz`-class object whose span is owned by this sub-heap.
    #[inline]
    unsafe fn central_deposit(&self, obj: *mut u8, osz: usize) {
        let span = self.bitmaps.span_of(obj);
        let slot = ((obj as usize - self.bitmaps.span_base(span)) / osz) as u32;
        self.bitmaps.set(span, slot);
    }

    // NOTE: a per-thread tcache-GC flush (jemalloc low-water style) that pushed idle
    // magazine/L2 objects straight to the central bitmap was prototyped to fully close
    // the largest-class `frag crossclass` residual, but it lived on the cold refill
    // path and regressed prod/cons (it flushed an active producer's cache). Dropped in
    // favor of the size-tiered L2 cap (`cap_for_class`) + the background supervisor
    // reclaim, which match the field without touching the refill path. If revisited,
    // gate the flush strictly on a class being idle across a full GC period (a
    // refilled-since-gc flag), never on a count heuristic; deposit straight to the
    // central bitmap (not L2) so emptied spans become cross-class reclaimable.

    /// Fill `out` with up to `out.len()` free objects of size `osz` from central
    /// `c`'s active spans — walking the active-span list, scanning each span's
    /// bitmap words for set (free) bits, clearing only the bits it hands out, and
    /// computing each address as `span_base + slot*osz` (no object read). Returns
    /// the count written (objects come out in ascending-address order within a span).
    ///
    /// Spans are **never unlinked** here: once carved a span stays on the active
    /// list for the life of the sub-heap (its object count converges to the
    /// application's live working set, exactly like a jemalloc slab). This is what
    /// lets [`central_deposit`] be lock-free — a deposit only ever SETS a bit and
    /// never has to (re)link, so it can't race the taker into a stranded free.
    /// Clearing exactly the handed-out bits (`clear_taken`, a `fetch_and`) rather
    /// than swapping the whole word preserves any free a consumer deposits
    /// concurrently into the same word.
    ///
    /// # Safety
    /// Caller holds `c`'s lock (`_g`) — serializes multiple producers sharing a
    /// shard against each other and against the carver; deposits run lock-free.
    #[inline]
    unsafe fn central_take(
        &self,
        c: &CentralList,
        _g: &CentralGuard<'_>,
        out: &mut [*mut u8],
        osz: usize,
        class: usize,
    ) -> usize {
        // A span holds `slots_per_span` tiled objects (see `central_carve`); their
        // bits live in the first `nwords` bitmap words. Scanning the full 128-word
        // capacity would waste time on larger classes (e.g. 1.5 KiB → 1 word). The
        // `.max(1)` is essential for classes larger than a span (`osz > SPAN_BYTES`,
        // i.e. 128/192/256 KiB), which occupy slot 0 of one span: without it
        // `SPAN_BYTES/osz` floors to 0, the scan reads no words, and those frees are
        // never reclaimed (the arena then drains to a false budget exhaustion).
        let slots_per_span = (super::meta::SPAN_BYTES / osz).max(1);
        let nwords = (slots_per_span + 63) / 64;
        let mut n = 0;

        // Reuse hint: for **large classes** (few objects per span, so a thread's live
        // set spans MANY spans and the linear active-list march has a long reuse
        // distance that spills the working set to DRAM), scan the span the consumer
        // most recently deposited into FIRST — re-handing-out the freshest (cache-hot)
        // returns to keep reuse close. Skipped for small classes: there a single span
        // already holds the whole working set, so the hint adds only a redundant
        // first-scan (it measurably regressed 64 B). Pure ordering; the list walk
        // below still covers every span (re-scanning the hint span there is harmless).
        let hint = if slots_per_span <= 64 {
            let h = c.reuse_hint.load(Ordering::Relaxed);
            // Re-validate the hinted span's CLASS before scanning it. The hint is
            // written lock-free on the deposit path (`dealloc_class_batch`) with the
            // freeing thread's class parameter and is read here only under this lock;
            // but the cross-class reclaimer can re-tag a fully-free span to a
            // different class (and a deposit preempted mid-flush can even re-publish a
            // now-stale hint *after* a reclaim). Scanning a re-tagged span with this
            // class's `osz` would hand out the other class's slots → a cross-class
            // double-hand-out. The span table's class is published with `Release` by
            // the reclaimer's `assign` under this same lock and read with `Acquire`
            // here (`lookup`), so a re-tagged span is always observed with its new
            // class and skipped. (Without reclaim this check never fires — the hint
            // can only point at a span this very class owns.)
            if h != super::meta::NO_SPAN
                && (cfg!(toccata_reclaim_negative_control)
                    || self.spans.lookup(self.bitmaps.span_base(h) as *const u8)
                        .is_some_and(|(hc, _)| hc as usize == class))
            {
                n = self.scan_span(h, out, n, nwords, osz);
                h
            } else {
                super::meta::NO_SPAN
            }
        } else {
            super::meta::NO_SPAN
        };

        let mut span = *c.head.get();
        while n < out.len() && span != super::meta::NO_SPAN {
            if span != hint {
                n = self.scan_span(span, out, n, nwords, osz);
            }
            span = self.bitmaps.next(span);
        }
        n
    }

    /// Hand out free objects from one span into `out[n..]`, returning the new count.
    /// `fold_free_word` first merges any consumer deposits (free_words → the private
    /// alloc_words) and returns the resulting alloc word; bits are then handed out +
    /// cleared in alloc_words with no atomics. In steady state the fold reads 0
    /// (consumer hasn't touched this word since the last refill) so there is no
    /// cross-core line bounce — the producer cycles its own private line.
    ///
    /// # Safety
    /// Caller holds the owning central lock.
    #[inline]
    unsafe fn scan_span(
        &self,
        span: u32,
        out: &mut [*mut u8],
        mut n: usize,
        nwords: usize,
        osz: usize,
    ) -> usize {
        let base = self.bitmaps.span_base(span);
        for w in 0..nwords {
            if n == out.len() {
                break;
            }
            let mut word = self.bitmaps.fold_free_word(span, w);
            if word == 0 {
                continue;
            }
            let mut taken = 0u64;
            while word != 0 && n < out.len() {
                let bit = word.trailing_zeros();
                let slot = w * 64 + bit as usize;
                out[n] = (base + slot * osz) as *mut u8;
                n += 1;
                let m = 1u64 << bit;
                taken |= m;
                word &= !m;
            }
            self.bitmaps.alloc_clear(span, w, taken);
        }
        n
    }

    /// Carve a fresh run for `(my_shard, class)` from the arena, **tiling each span
    /// independently** (so no object straddles a span boundary), mark every carved
    /// object's slot free in the bitmap, and link its span(s) onto `c`'s active
    /// list. Returns the number of objects made available (0 = arena drained).
    ///
    /// Per-span tiling is what keeps the bitmap's `slot ↔ address` mapping exact for
    /// every size class: within span `s`, object `i` lives at `span_base(s) + i*osz`
    /// for `i in 0..floor(SPAN_BYTES/osz)`, and the bounded tail (`SPAN_BYTES mod
    /// osz` bytes) is left unused. Power-of-two classes tile a 64 KiB span with zero
    /// waste; the few non-divisor classes (e.g. 1.5 KiB, 48 KiB) waste < one object
    /// per span. Classes **larger than a span** (128/192/256 KiB) get one object per
    /// `ceil(osz/SPAN_BYTES)`-span group, occupying slot 0 of the group's first span.
    ///
    /// Claim `num_spans` whole spans for `(class, home)`, **reusing a reclaimed run
    /// from the [`SpanPool`](crate::spanpool::SpanPool) if one is available** before
    /// advancing the monotonic bump cursor. This is what lets a span freed by one
    /// size class be re-carved by another (the `frag crossclass` fix): the supervisor
    /// retires fully-free spans into the pool ([`reclaim_empty_spans`]), and a later
    /// carve of any class draws them here instead of growing `carved_bytes`.
    ///
    /// The pool lives behind the `LargeAllocator` futex mutex, which can sleep — but
    /// this runs under the (never-sleeping) central spin lock, so we **`try_lock`**
    /// the pool and fall straight through to the bump cursor on contention rather
    /// than risk a futex wait while holding the spin lock (preserving never-stall).
    /// On a pool hit we assign the run's span range to `(class, home)` ourselves
    /// (the pool handed back an untagged `LARGE_CLASS` run). Returns the byte range
    /// `[start, end)`, or `(x, x)` when neither pool nor arena can satisfy it.
    ///
    /// [`reclaim_empty_spans`]: Self::reclaim_empty_spans
    #[inline]
    fn carve_or_reuse_spans(&self, num_spans: usize, class: u16, home: u16) -> (usize, usize) {
        // Probe the reclaimed-span pool first (non-blocking): a span freed by another
        // (now-idle) size class and retired into the pool by the supervisor sweep is
        // reused here instead of advancing the monotonic bump cursor (the `frag
        // crossclass` fix). `try_lock` so the central-lock holder never futex-waits on
        // the large mutex.
        if let Some(r) = self.try_pool_alloc(num_spans, class, home) {
            return r;
        }
        // Pool miss: advance the bump cursor. (We deliberately do NOT run a
        // synchronous cross-class reclaim here: this is called under a central spin
        // lock, and reclaim's pool-filing takes the large futex mutex — blocking on it
        // while holding a spin lock would violate never-stall. The background
        // supervisor does the reclaim off the datapath instead.)
        self.arena.carve_spans(num_spans, class, home, &self.spans)
    }

    /// Non-blocking attempt to satisfy a `num_spans` carve from the reclaimed-span
    /// [`SpanPool`]. Returns the assigned `[start, end)` byte range on a hit. Uses
    /// `try_lock` so the central-lock holder never futex-waits on the large mutex.
    #[inline]
    fn try_pool_alloc(&self, num_spans: usize, class: u16, home: u16) -> Option<(usize, usize)> {
        let mut inner = self.large.inner.try_lock().ok()?;
        let head_idx = inner.pool.alloc(num_spans as u32)?;
        drop(inner);
        let start = self.large.span_ptr(head_idx) as usize;
        let end = start + num_spans * super::meta::SPAN_BYTES;
        self.spans.assign_range(start as *const u8, end - start, class, home);
        Some((start, end))
    }

    /// # Safety
    /// Caller holds `c`'s lock (`_g`); `c == self.central(my_shard, class)`.
    #[inline]
    unsafe fn central_carve(
        &self,
        c: &CentralList,
        _g: &CentralGuard<'_>,
        class: usize,
        my_shard: u32,
        batch: usize,
    ) -> usize {
        let osz0 = sizeclass::size_of_class(class);
        let osz = (osz0 + (sizeclass::MIN_ALIGN - 1)) & !(sizeclass::MIN_ALIGN - 1);
        let span_bytes = super::meta::SPAN_BYTES;

        if osz > span_bytes {
            // Class larger than a span: one object per whole-span group, at slot 0.
            let group_spans = osz.div_ceil(span_bytes);
            let (start, run_end) =
                self.carve_or_reuse_spans(group_spans, class as u16, my_shard as u16);
            if run_end <= start {
                return 0;
            }
            // A single carve claims exactly one group (we ask for `group_spans`),
            // so there is exactly one object here, at the group's first span slot 0.
            // Fresh slots go straight into the producer-private alloc_words.
            let span = self.bitmaps.span_of(start as *const u8);
            self.bitmaps.alloc_set_mask(span, 0, 1);
            if !self.bitmaps.is_on_list(span) {
                let head = *c.head.get();
                *c.head.get() = self.bitmaps.link_front(span, head);
            }
            return 1;
        }

        // Class fits within a span: claim enough whole spans to hold ~`batch`
        // objects (at least one span), then tile each span independently. Fresh
        // slots go straight into the producer-private alloc_words (set whole words
        // at once: a full word = all 64 slots free, then a partial last word).
        let per_span = span_bytes / osz;
        let want_spans = batch.div_ceil(per_span).max(1);
        let (start, run_end) =
            self.carve_or_reuse_spans(want_spans, class as u16, my_shard as u16);
        if run_end <= start {
            return 0;
        }
        let nwords = per_span / 64;
        let rem = per_span % 64;
        let mut carved = 0usize;
        let mut span_base = start;
        while span_base < run_end {
            let span = self.bitmaps.span_of(span_base as *const u8);
            for w in 0..nwords {
                self.bitmaps.alloc_set_mask(span, w, u64::MAX);
            }
            if rem != 0 {
                self.bitmaps.alloc_set_mask(span, nwords, (1u64 << rem) - 1);
            }
            carved += per_span;
            if !self.bitmaps.is_on_list(span) {
                let head = *c.head.get();
                *c.head.get() = self.bitmaps.link_front(span, head);
            }
            span_base += span_bytes;
        }
        carved
    }

    /// Test seam: this thread's home shard (`thread_shard_id() % num_shards`). Lets
    /// a regression test target the same central cell `refill_batch` will read.
    #[doc(hidden)]
    pub fn test_thread_shard(&self) -> u32 {
        thread_shard_id() % self.num_shards
    }

    /// Test seam: the span index containing an in-arena `ptr`.
    #[doc(hidden)]
    pub fn test_span_of(&self, ptr: *const u8) -> u32 {
        self.bitmaps.span_of(ptr)
    }

    /// Test seam: force `central(shard, class).reuse_hint` to span index `span`.
    /// Simulates the cross-thread deposit path (`dealloc_class_batch`,
    /// subheap.rs:883) publishing a reuse hint — including the "window 2" case where
    /// a preempted depositor re-publishes a now-stale hint *after* a reclaim has
    /// re-tagged that span. Used to deterministically exercise the `central_take`
    /// class-revalidation that closes the stale-hint cross-class double-hand-out.
    #[doc(hidden)]
    pub fn test_set_reuse_hint(&self, shard: u32, class: usize, span: u32) {
        self.central(shard, class).reuse_hint.store(span, Ordering::Relaxed);
    }

    /// The `[base, len)` of this sub-heap's backing reservation, for tests that
    /// verify toccata issues no kernel memory syscall against its own pool
    /// after seal. Returns `(0, 0)` if the sub-heap borrows a shared reservation.
    #[doc(hidden)]
    pub fn reservation_range(&self) -> (usize, usize) {
        match &self._reservation {
            Some(r) => (r.as_ptr() as usize, r.len()),
            None => (0, 0),
        }
    }

    /// Allocate `class`-sized storage. Returns `None` if the budget is exhausted
    /// or (transiently) the central list is empty. Never blocks beyond the brief
    /// per-CPU / central locks; never calls the kernel.
    #[inline]
    pub fn alloc_class(&self, class: usize) -> Option<NonNull<u8>> {
        let obj_size = sizeclass::size_of_class(class) as i64;

        // No global budget atomic on the hot path: the bounded arena enforces the
        // limit structurally (refill's carve returns empty at exhaustion). The
        // rseq asm reads the CPU itself and returns it, so we skip the upfront
        // current_cpu() read (current_fast) and account on the returned CPU.
        let stack = CpuStack::current_fast(&self.slab);
        let (ptr, cpu) = match stack.pop_on(class) {
            (Fast::Ok(p), cpu) => (p, cpu),
            (Fast::NeedsSlow, _) => {
                let p = self.refill(&stack, class, thread_shard_id())?;
                (p, stack.cpu())
            }
        };
        #[cfg(not(toccata_no_accounting))]
        self.counters.add_local(cpu, obj_size, 1);
        let _ = (obj_size, cpu);
        Some(ptr)
    }

    /// Free a single `class`-sized object back to this sub-heap.
    ///
    /// Routes by the object's **home shard** (recovered from span metadata): if it
    /// matches the freeing thread's shard — the single-thread / same-owner case —
    /// the object goes to the local L2 rseq slab (cheap, no lock). Otherwise it is
    /// deposited into the home shard's central list `central[home][class]`, where
    /// the owning producer refills from. Because home is a *stable thread shard*
    /// (not the current CPU), the deposit always reaches the one meeting point the
    /// producer draws from, regardless of which CPU either thread is scheduled on.
    ///
    /// # Safety
    /// `ptr` must have come from `alloc_class(class)` on this sub-heap and not be
    /// freed already.
    #[inline]
    pub unsafe fn dealloc_class(&self, ptr: NonNull<u8>, class: usize) {
        let home = self.spans.home_relaxed(ptr.as_ptr());
        let obj_size = sizeclass::size_of_class(class) as i64;
        let my_shard = thread_shard_id() % self.num_shards;

        let stack = CpuStack::current(&self.slab);
        let cur_cpu = stack.cpu();
        if home == u16::MAX || (home as u32) % self.num_shards == my_shard {
            // Local / same-owner (or unrecognized home): push to this thread's L2.
            match stack.push(class, ptr) {
                Fast::Ok(()) => {}
                Fast::NeedsSlow => self.drain(&stack, class, ptr, my_shard),
            }
        } else {
            // Cross-owner: set the object's free bit in the home shard's bitmap
            // pool with a single lock-free `fetch_or` (no read of the object, no
            // central lock). The owning producer reclaims it on its next refill.
            // SAFETY: ptr is a freed `class` object owned by this sub-heap.
            self.central_deposit(ptr.as_ptr(), obj_size as usize);
        }
        // Charge the freeing CPU's own cell, non-atomically, whether local or
        // remote (see `dealloc_class_batch` for why this is sound and cheaper than
        // the per-remote-free atomic `add_remote`).
        #[cfg(not(toccata_no_accounting))]
        self.counters.add_local(cur_cpu, -obj_size, -1);
        let _ = (obj_size, cur_cpu);
    }

    /// Free a batch of same-class objects (an L1 magazine flush). Objects whose
    /// home shard is this thread's shard go to the local L2 slab; objects homed on
    /// another shard have their free bit set in that shard's bitmap pool **lock-free
    /// and coalesced per bitmap word** — no central lock at all. A consumer flushing
    /// a magazine full of one producer's buffers gets them in near-contiguous address
    /// order, so their slots cluster into one or two bitmap words; OR-ing each word's
    /// bits together and depositing per-word (`set_mask`) turns ~32 atomic RMWs into
    /// ~2, minimizing traffic on the per-span bitmap line the producer's refill also
    /// touches. The consumer never blocks on or contends a lock with the producer;
    /// the producer reclaims the bits on its next refill (under its own lock, single
    /// taker). This is what removes the producer/consumer collision that made the
    /// steady-state cross-thread latency bimodal. Independent pairs hit disjoint
    /// shards, so there is no cross-pair contention either.
    ///
    /// # Safety
    /// Every `ptr` in `ptrs` must have come from `alloc_class(class)` on this
    /// sub-heap and not be freed already.
    #[cold]
    pub unsafe fn dealloc_class_batch(&self, ptrs: &[*mut u8], class: usize) {
        let obj_size = sizeclass::size_of_class(class) as i64;
        let my_shard = thread_shard_id() % self.num_shards;
        let stack = CpuStack::current(&self.slab);
        let cur_cpu = stack.cpu();

        // Deposit remote objects' free bits into their home shard's bitmap pool,
        // coalescing consecutive frees that share a `(span, word)` into one masked
        // `fetch_or` (the common case: a flush is near-contiguous addresses). Local
        // objects go to this thread's L2 slab.
        let osz = obj_size as usize;
        let mut remote_len: i64 = 0;
        let mut local_len: i64 = 0;
        // Pending coalesced deposit: bits accumulated for `(cur_span, cur_word)`.
        let mut cur_span = super::meta::NO_SPAN;
        let mut cur_word = 0usize;
        let mut cur_mask = 0u64;

        for &p in ptrs {
            // Just the home field (no Option / no class unpack — class is known).
            // `u16::MAX` (out-of-range sentinel) is treated as local, always safe.
            let home = self.spans.home_relaxed(p);
            let hs = (home as u32) % self.num_shards;
            if home == u16::MAX || hs == my_shard {
                // SAFETY: p is a freed object of `class` owned by this sub-heap.
                let nn = NonNull::new_unchecked(p);
                match stack.push(class, nn) {
                    Fast::Ok(()) => {}
                    Fast::NeedsSlow => self.drain(&stack, class, nn, my_shard),
                }
                local_len += 1;
            } else {
                // Coalesce into the pending word if it matches; else flush + restart.
                let (span, word, bit) = self.bitmaps.locate(p, osz);
                if span != cur_span || word != cur_word {
                    if cur_mask != 0 {
                        self.bitmaps.set_mask(cur_span, cur_word, cur_mask);
                    }
                    cur_span = span;
                    cur_word = word;
                    cur_mask = 0;
                }
                cur_mask |= 1u64 << bit;
                remote_len += 1;
            }
        }
        // Flush the final pending word.
        if cur_mask != 0 {
            self.bitmaps.set_mask(cur_span, cur_word, cur_mask);
        }
        // Point the home shard's reuse hint at the last span we deposited into, so
        // the owning producer's next refill re-hands-out these freshly-returned
        // (cache-hot) buffers first. A magazine flush shares one home shard, so one
        // hint write covers the batch; `cur_span` is the most recent deposit.
        if cur_span != super::meta::NO_SPAN {
            let home = (self.spans.home_relaxed(self.bitmaps.span_base(cur_span) as *const u8)
                as u32)
                % self.num_shards;
            self.central(home, class).reuse_hint.store(cur_span, Ordering::Relaxed);
        }

        // Charge the whole batch (local + remote) to the freeing CPU's own cell
        // with a single NON-ATOMIC add. The live meter is a sum across all cells,
        // so it only needs each delta counted once *somewhere*; charging the
        // freeing thread's own cell avoids the per-remote-free `fetch_add` atomic
        // that profiling flagged on this hot path. Migration can at worst skew the
        // instrumentation by one delta (already an accepted tradeoff, like the
        // local fast path), never the memory state.
        #[cfg(not(toccata_no_accounting))]
        {
            let total = local_len + remote_len;
            if total > 0 {
                self.counters.add_local(cur_cpu, -obj_size * total, -total);
            }
        }
        let _ = (obj_size, cur_cpu, local_len, remote_len);
    }

    /// Supervisor seize support: set/clear this sub-heap's per-CPU stop flag.
    /// Between `set_stopped(cpu, true)` + a rseq-abort membarrier and
    /// `set_stopped(cpu, false)`, the supervisor may mutate that CPU's slab
    /// headers directly via [`SubHeap::rebalance_capacity`]; concurrent writers
    /// that abort fall to the locked slow path, see the flag, and route around.
    #[doc(hidden)]
    pub fn set_cpu_stopped(&self, cpu: u32, stopped: bool) {
        self.slab.set_stopped(cpu, stopped);
    }

    /// Move `count` free objects of `class` from CPU `cpu`'s slab into the
    /// central list, shrinking that CPU's cached capacity. MUST be called only
    /// while `cpu` is seized (stop flag set + rseq-abort membarrier issued), so
    /// no writer is in that CPU's rseq section. Returns how many were moved.
    ///
    /// # Safety
    /// Caller holds the seize for `cpu`.
    #[doc(hidden)]
    pub unsafe fn rebalance_capacity(&self, cpu: u32, class: usize, count: u32) -> u32 {
        let hdr = self.slab.header_mut(cpu, class);
        let cur = (*hdr).current;
        let take = count.min(cur);
        if take == 0 {
            return 0;
        }
        // Pop `take` pointers off the top of the seized CPU's stack and park each
        // in its home shard's central list (so a later owner refill finds it). We
        // read slots directly since the CPU is stopped.
        let osz = sizeclass::size_of_class(class);
        for _ in 0..take {
            let new_cur = (*hdr).current - 1;
            let slot = self.slab.slot_ptr(cpu, class, new_cur);
            let obj = *slot;
            (*hdr).current = new_cur;
            if !obj.is_null() {
                // SAFETY: obj is a free `class` slot from the seized CPU's stack;
                // deposit its free bit into its home shard's bitmap pool (lock-free).
                self.central_deposit(obj, osz);
            }
        }
        take
    }

    /// Supervisor entry point: reclaim cross-thread frees that may be stranded.
    ///
    /// With the lock-free sharded central design a cross-thread free is *already*
    /// deposited onto the home shard's bitmap pool — there is no separate per-CPU
    /// remote queue to drain, and objects on an idle shard are still globally
    /// reachable (any thread hashing to that shard pops them on refill). So this is
    /// a no-op for *object* relocation; the real reclamation work — returning whole
    /// fully-free spans across size classes — is [`reclaim_empty_spans`].
    /// Returns the number of objects relocated (always 0).
    ///
    /// [`reclaim_empty_spans`]: Self::reclaim_empty_spans
    pub fn reclaim_stranded_remote(&self) -> usize {
        0
    }

    /// **Cross-class empty-span reclaim** — the fix for budget stranding when one
    /// size class is freed and another grows (`frag crossclass`). A small-class span
    /// is write-once tagged to its class and, once carved, never returned to the
    /// arena; so freeing every object of class A then allocating class B strands A's
    /// spans (the bump cursor only advances). This sweep walks each `(shard, class)`
    /// active list and, for every span that is **provably fully free**, untags it and
    /// returns it to the shared [`SpanPool`](crate::spanpool::SpanPool) as a free run,
    /// where a later carve of *any* class (or a large allocation) reuses it.
    ///
    /// Run off the datapath by the supervisor. `max_per_shard_class` caps how many
    /// spans are reclaimed per central-lock acquisition so the (cold, uncontended)
    /// central spin lock is never held long enough to perturb the latency tail; the
    /// lock is dropped between `(shard, class)` cells. Returns spans reclaimed.
    ///
    /// ## Safety protocol (no magazine quiesce / epoch needed)
    ///
    /// All per-class bitmap/list work happens under the owning `(shard, class)`
    /// central lock — the same lock a producer's refill takes — so reclaim and
    /// allocation never overlap for a cell. For each candidate span S of class A:
    /// 1. [`is_fully_free`](super::meta::SpanBitmaps::is_fully_free) (fold every word,
    ///    popcount == slots, re-read `free_words` with Acquire). A live or
    ///    magazine-cached object keeps its `alloc_words` bit clear, so any referenced
    ///    object aborts the reclaim of S — there is no in-flight deposit to race.
    /// 2. [`unlink_active`](super::meta::SpanBitmaps::unlink_active) from A's list and
    ///    clear A's `reuse_hint` if it pointed at S.
    /// 3. `assign(S -> LARGE_CLASS)` (Release) + `reset_bitmaps`, then file S into the
    ///    SpanPool under the large mutex (lock order: central first, dropped, then
    ///    large — never both held).
    ///
    /// The one cross-cutting hazard — a deposit preempted mid-`dealloc_class_batch`
    /// re-publishing S into A's stale `reuse_hint` *after* reclaim — is closed on the
    /// READ side: [`central_take`](Self::central_take) re-validates the hint span's
    /// class (Acquire `lookup` vs the Release `assign` here) before scanning it, so a
    /// re-tagged span is never scanned under the wrong class.
    pub fn reclaim_empty_spans(&self, max_per_lock: usize) -> usize {
        let cap = max_per_lock.max(1);
        let mut reclaimed = 0usize;
        for shard in 0..self.num_shards {
            for class in 1..sizeclass::NUM_CLASSES {
                // Drain this cell across MULTIPLE bounded lock acquisitions: each
                // `reclaim_cell` call reclaims at most `cap` spans then DROPS the lock
                // (so the central spin lock is never held long enough to perturb the
                // tail), and we re-acquire to continue until the cell yields no more.
                // A freed-then-abandoned class (the `frag crossclass` case) can hold
                // thousands of empty spans in ONE cell; capping per-acquisition but
                // looping lets a sweep reclaim them all without a long lock hold. The
                // walk restarts from the head each acquisition — fine, since reclaimed
                // spans are unlinked, so progress is monotonic.
                loop {
                    let got = unsafe { self.reclaim_cell(shard, class, cap) };
                    reclaimed += got;
                    if got < cap {
                        break; // cell exhausted (fewer than a full batch reclaimable)
                    }
                }
            }
        }
        reclaimed
    }

    /// Reclaim fully-free spans from one `(shard, class)` cell. Up to `cap` spans,
    /// then the central lock is released (the caller re-acquires to continue).
    ///
    /// # Safety
    /// Internal; takes the cell's central lock itself.
    unsafe fn reclaim_cell(&self, shard: u32, class: usize, cap: usize) -> usize {
        if cap == 0 {
            return 0;
        }
        let osz0 = sizeclass::size_of_class(class);
        let osz = (osz0 + (sizeclass::MIN_ALIGN - 1)) & !(sizeclass::MIN_ALIGN - 1);
        let span_bytes = super::meta::SPAN_BYTES;
        // Slots tiled per span for this class, and how many whole spans one object
        // occupies (≥2 only for classes larger than a span, e.g. 128/192/256 KiB).
        let slots_per_span = (span_bytes / osz).max(1);
        let group_spans = osz.div_ceil(span_bytes).max(1) as u32;
        let c = self.central(shard, class);

        // Collect reclaimed (head_span, group_spans) under the central lock; file
        // them into the SpanPool AFTER dropping it (lock order central -> large).
        let mut taken: [u32; 32] = [super::meta::NO_SPAN; 32];
        let cap = cap.min(taken.len());
        let mut n = 0usize;

        {
            // try_lock, not lock: the sweep must never block a producer's refill/
            // drain. If the cell is busy, skip it this tick (revisited next sweep).
            let Some(_g) = c.try_lock() else { return 0 };
            // Walk the singly-linked active list, tracking the predecessor so we can
            // unlink in O(1). A reclaimed span is removed from the list; otherwise we
            // advance. `head` is owned by this lock.
            let mut prev = super::meta::NO_SPAN;
            let mut span = *c.head.get();
            while span != super::meta::NO_SPAN && n < cap {
                let next = self.bitmaps.next(span);
                if self.bitmaps.is_fully_free(span, slots_per_span) {
                    // Unlink S (fixing head/prev), clear a stale reuse hint, retile.
                    let new_head = self.bitmaps.unlink_active(span, prev, *c.head.get());
                    *c.head.get() = new_head;
                    if c.reuse_hint.load(Ordering::Relaxed) == span {
                        c.reuse_hint.store(super::meta::NO_SPAN, Ordering::Relaxed);
                    }
                    // Re-tag to LARGE_CLASS (Release) so a stale hint re-publish is
                    // caught by central_take's class check, and dealloc routing won't
                    // treat a future large run handed from here as this small class.
                    self.spans.assign_range(
                        self.bitmaps.span_base(span) as *const u8,
                        (group_spans as usize) * span_bytes,
                        LARGE_CLASS,
                        0,
                    );
                    self.bitmaps.reset_bitmaps(span, slots_per_span.div_ceil(64));
                    taken[n] = span;
                    n += 1;
                    // `prev` stays; the list now skips `span` so its predecessor's
                    // successor is `next`.
                } else {
                    prev = span;
                }
                span = next;
            }
        } // central lock dropped here

        if n == 0 {
            return 0;
        }
        // File the reclaimed runs into the SpanPool under the large mutex. These
        // spans are now LARGE_CLASS, fully retired from the small class, and will be
        // handed out by `alloc_large` or a future small carve that probes the pool.
        let mut inner = self.large.inner.lock().unwrap();
        for &span in &taken[..n] {
            let idx = self.large.span_index(self.bitmaps.span_base(span) as *const u8) as u32;
            // run_spans head marker stays 0 until alloc_large hands it out; the pool
            // owns the free run by index.
            inner.pool.free(idx, group_spans);
        }
        drop(inner);
        n
    }

    /// Free an object by pointer alone, recovering its size class from the span
    /// table. This is the entry point used by the global allocator and by
    /// `Box`/`Vec` drop, which only know the pointer (the passed `Layout` is
    /// advisory). Returns `false` if the pointer is not from
    /// this sub-heap's arena (caller should route elsewhere).
    ///
    /// # Safety
    /// `ptr` must have come from this sub-heap and not be freed already.
    #[inline]
    pub unsafe fn dealloc_by_ptr(&self, ptr: NonNull<u8>) -> bool {
        match self.spans.lookup(ptr.as_ptr()) {
            Some((class, _home)) if class == LARGE_CLASS => {
                self.dealloc_large(ptr);
                true
            }
            Some((class, _home)) => {
                // dealloc_class re-reads the home shard and routes (local L2 vs
                // home shard's central list) itself.
                self.dealloc_class(ptr, class as usize);
                true
            }
            None => false,
        }
    }

    /// Allocate `size` bytes, choosing the small-class or large path. The single
    /// entry point for the global allocator / Box / Vec.
    #[inline]
    pub fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        match sizeclass::class_for(size) {
            Some(class) => self.alloc_class(class),
            None => self.alloc_large(size),
        }
    }

    /// Large (oversize) allocation: carve or recycle a span-aligned run. Rare,
    /// mutex-guarded, off the small-object hot path. Zero heap allocation.
    #[cold]
    pub fn alloc_large(&self, size: usize) -> Option<NonNull<u8>> {
        let span_bytes = super::meta::SPAN_BYTES;
        let span_count = size.div_ceil(span_bytes).max(1);
        let run_bytes = span_count * span_bytes;

        let mut inner = self.large.inner.lock().unwrap();
        // Recycle a free run via the coalescing pool: an exact-span-count run is
        // returned with no split (preserving `frag match`); otherwise a larger
        // run is split and the tail remainder re-filed. Pure index arithmetic.
        let ptr = if let Some(head_idx) = inner.pool.alloc(span_count as u32) {
            self.large.span_ptr(head_idx)
        } else {
            // Pool has no run large enough — carve a fresh span-aligned run.
            drop(inner); // don't hold the large lock across the arena lock
            let run = self.arena.carve_run(run_bytes)?;
            // Re-acquire to record the run.
            inner = self.large.inner.lock().unwrap();
            let _ = &mut inner; // keep the lock alive symmetrically
            run
        };
        // Record the run's span-count in the span-indexed array and tag the span
        // table so dealloc_by_ptr routes here.
        let idx = self.large.span_index(ptr);
        self.large.run_spans[idx].store(span_count as u32, Ordering::Release);
        self.spans.assign_range(ptr, run_bytes, LARGE_CLASS, 0);
        drop(inner);
        let cpu = crate::rseq::current_cpu().unwrap_or(0).min(self.counters.cells.len() as u32 - 1);
        self.counters.add_remote(cpu, run_bytes as i64, 1);
        Some(unsafe { NonNull::new_unchecked(ptr) })
    }

    /// Free a large run by pointer, recycling it for its span-count.
    #[cold]
    unsafe fn dealloc_large(&self, ptr: NonNull<u8>) {
        let idx = self.large.span_index(ptr.as_ptr());
        let span_count = self.large.run_spans[idx].swap(0, Ordering::AcqRel) as usize;
        debug_assert!(span_count > 0, "dealloc_large on a non-run pointer");
        if span_count == 0 {
            return;
        }
        let run_bytes = span_count * super::meta::SPAN_BYTES;
        let idx = idx as u32;
        let mut inner = self.large.inner.lock().unwrap();
        // Return the run to the coalescing pool, which merges it with any
        // address-adjacent free runs. No write to the freed run's own (cold)
        // memory — the pool tracks everything out-of-band by span index.
        inner.pool.free(idx, span_count as u32);
        drop(inner);
        let cpu = crate::rseq::current_cpu().unwrap_or(0).min(self.counters.cells.len() as u32 - 1);
        self.counters.add_remote(cpu, -(run_bytes as i64), -1);
    }

    /// Whether `ptr` lies within this sub-heap's arena.
    #[inline]
    pub fn owns(&self, ptr: NonNull<u8>) -> bool {
        self.spans.contains(ptr.as_ptr())
    }

    /// Recover the size class for a pointer from this sub-heap, if owned.
    #[inline]
    pub fn class_of(&self, ptr: NonNull<u8>) -> Option<usize> {
        self.spans.lookup(ptr.as_ptr()).map(|(c, _)| c as usize)
    }

    /// Slow path: per-CPU stack empty. Pull a batch from **this thread's home
    /// shard's** central list into the stack and return one. The objects waiting
    /// there are exactly what consumers deposited for this producer (same home
    /// shard) plus what it carved itself — so the producer reuses returned memory
    /// without a separate reclaim step or per-object cross-core hop. Purely
    /// userspace pointer movement.
    #[cold]
    fn refill(&self, stack: &CpuStack<'_>, class: usize, my_shard: u32) -> Option<NonNull<u8>> {
        let batch = batch_for(class);
        let osz = sizeclass::size_of_class(class);
        let c = self.central(my_shard, class);
        let g = c.lock();
        // Take a batch from the bitmap pool (scans set bits, computes addresses —
        // no cold-object reads); carve fresh + retry if the pool is empty.
        let mut buf: [*mut u8; 64] = [core::ptr::null_mut(); 64];
        let want = batch.min(64);
        // SAFETY: lock held throughout.
        let mut got = unsafe { self.central_take(c, &g, &mut buf[..want], osz, class) };
        if got == 0 {
            if unsafe { self.central_carve(c, &g, class, my_shard, batch) } == 0 {
                return None; // arena (and thus budget) truly drained
            }
            got = unsafe { self.central_take(c, &g, &mut buf[..want], osz, class) };
            if got == 0 {
                return None;
            }
        }
        drop(g);
        // Hand out the first; push the rest onto the per-CPU L2 stack.
        let handed_out = buf[0];
        for &p in &buf[1..got] {
            // SAFETY: p is a fresh/recycled `class` object.
            let nn = unsafe { NonNull::new_unchecked(p) };
            if let Fast::NeedsSlow = stack.push(class, nn) {
                // L2 full: return the leftover to the bitmap pool (lock-free).
                unsafe { self.central_deposit(p, osz) };
            }
        }
        Some(unsafe { NonNull::new_unchecked(handed_out) })
    }

    /// Fill `out` with up to `out.len()` objects of `class` pulled **directly from
    /// this thread's home-shard central list** (carving fresh from the arena if it
    /// is empty), and return how many were written. This is the L1 magazine's
    /// refill path: it bypasses the per-CPU L2 slab entirely, so on the
    /// producer/consumer workload a producer's refill is `central → magazine` in
    /// one hop instead of `central → L2 push → L2 pop → magazine`, eliminating an
    /// L2 round-trip per object (the structural cost that kept the producer side
    /// above jemalloc). The objects are exactly what consumers deposited for this
    /// shard plus freshly-carved ones.
    ///
    /// # Safety
    /// `out` is a writable slice; returned pointers are live `class`-sized objects.
    #[cold]
    pub unsafe fn refill_batch(&self, class: usize, out: &mut [*mut u8]) -> usize {
        let my_shard = thread_shard_id() % self.num_shards;
        let osz = sizeclass::size_of_class(class);
        let c = self.central(my_shard, class);
        // One lock acquisition. `central_take` fills the magazine from the bitmap
        // pool by scanning set bits + computing addresses (no cold-object reads,
        // ascending-address order); carve to top up when the pool is dry.
        let g = c.lock();
        let mut n = self.central_take(c, &g, out, osz, class);
        while n < out.len() {
            if self.central_carve(c, &g, class, my_shard, batch_for(class)) == 0 {
                break; // arena (and thus budget) truly drained
            }
            let got = self.central_take(c, &g, &mut out[n..], osz, class);
            if got == 0 {
                break;
            }
            n += got;
        }
        drop(g);
        #[cfg(not(toccata_no_accounting))]
        if n > 0 {
            // Charge as local: these objects are now live, homed on this shard.
            let cpu = CpuStack::current(&self.slab).cpu();
            self.counters.add_local(cpu, sizeclass::size_of_class(class) as i64 * n as i64, n as i64);
        }
        n
    }

    /// Slow path: per-CPU stack full. Drain a batch to this thread's home shard's
    /// central list, then stash `ptr`. The drained objects are this thread's own
    /// (homed on `my_shard`), so they go back to the same shard it refills from.
    /// Purely userspace pointer movement; zero heap allocation.
    #[cold]
    fn drain(&self, stack: &CpuStack<'_>, class: usize, ptr: NonNull<u8>, _my_shard: u32) {
        let batch = batch_for(class);
        let osz = sizeclass::size_of_class(class);
        // Move up to half a batch from the L2 stack into this shard's bitmap pool
        // (set each slot's free bit with a lock-free `fetch_or` — no object write
        // beyond the slot, no link chain, no lock). These are this thread's own
        // objects, carved here, so their spans are already linked.
        for _ in 0..(batch / 2).max(1) {
            match stack.pop(class) {
                // SAFETY: p is a free `class` slot homed on `_my_shard`.
                Fast::Ok(p) => unsafe { self.central_deposit(p.as_ptr(), osz) },
                Fast::NeedsSlow => break,
            }
        }
        // Now push the freed ptr; if still full (pathological), deposit it too.
        if let Fast::NeedsSlow = stack.push(class, ptr) {
            // SAFETY: ptr is a freed `class` object homed here.
            unsafe { self.central_deposit(ptr.as_ptr(), osz) };
        }
    }
}

/// Builder that carves a reservation slice into a `SubHeap`. Used by the
/// registry during `configure()`.
pub struct SubHeapBuilder {
    name: &'static str,
    on_exhaust: OnExhaust,
    budget_bytes: usize,
    num_cpus: u32,
    /// Number of central-list home shards. `None` = derive from `num_cpus`.
    num_shards: Option<u32>,
    /// Per-(CPU,class) stack capacity.
    cap_per_class: u32,
}

/// How many home shards to use per CPU when not overridden. Mirrors jemalloc's
/// `narenas = 4 * ncpu`: enough that independent producer/consumer pairs land on
/// distinct central-list shards and don't serialize on one lock, while the
/// per-shard central metadata stays a tiny fraction of the budget. The home is a
/// stable per-thread id, so shard count is about pair-collision probability, not
/// core count.
const SHARDS_PER_CPU: u32 = 4;

/// Upper bound on shard count: the span table packs `home` into a `u16`.
const MAX_SHARDS: u32 = u16::MAX as u32;

impl SubHeapBuilder {
    pub fn new(name: &'static str, budget_bytes: usize) -> Self {
        Self {
            name,
            on_exhaust: OnExhaust::None,
            budget_bytes,
            num_cpus: default_num_cpus(),
            num_shards: None,
            cap_per_class: 256,
        }
    }

    pub fn on_exhaust(mut self, p: OnExhaust) -> Self {
        self.on_exhaust = p;
        self
    }
    pub fn cap_per_class(mut self, c: u32) -> Self {
        self.cap_per_class = c;
        self
    }
    /// Override the number of per-CPU shards. Defaults to the machine's CPU
    /// count. Fewer shards = less slab metadata (smaller locked reservation) at
    /// the cost of more cross-shard contention; useful for tests and for sizing
    /// the mlock footprint under a tight `RLIMIT_MEMLOCK`.
    pub fn num_cpus(mut self, n: u32) -> Self {
        self.num_cpus = n.max(1);
        self
    }
    /// Override the number of central-list home shards (default `4 * num_cpus`,
    /// clamped to `[1, 65535]`). Sharding the central free list by a stable
    /// per-thread home is what lets independent producer/consumer pairs avoid one
    /// shared lock. More shards = better isolation, slightly more central
    /// metadata + a touch more steady-state RSS (objects parked across more
    /// lists).
    pub fn num_shards(mut self, n: u32) -> Self {
        self.num_shards = Some(n.clamp(1, MAX_SHARDS));
        self
    }

    /// Build a standalone sub-heap with its own reservation (Phase 1 / tests).
    /// Pre-populates every class's central list to fill the budget.
    pub fn build_standalone(self) -> Result<SubHeap, crate::sys::ReserveError> {
        let num_classes = sizeclass::NUM_CLASSES;
        let num_shards =
            self.num_shards.unwrap_or_else(|| (self.num_cpus * SHARDS_PER_CPU).clamp(1, MAX_SHARDS));

        // --- compute per-CPU block geometry ---
        // Block layout: [ Header[num_classes] | lock[num_classes] | slots... ]
        let headers_bytes = num_classes * core::mem::size_of::<Header>();
        let locks_bytes = num_classes * 4;
        let mut off = (headers_bytes + locks_bytes + 7) & !7;
        // System-backed (never global) — this is configure-time metadata.
        let mut classes_loc: crate::SysVec<ClassLoc> =
            allocator_api2::vec::Vec::with_capacity_in(num_classes, crate::Sys);
        for c in 0..num_classes {
            classes_loc.push(ClassLoc {
                header_off: (c * core::mem::size_of::<Header>()) as u32,
                lock_off: (headers_bytes + c * 4) as u32,
                slots_off: off as u32,
            });
            // Size-tiered L2 capacity: a smaller per-CPU cache for larger objects so
            // an abandoned large class pins few spans (the `frag crossclass` residual).
            off += cap_for_class(c, self.cap_per_class) as usize * 8;
        }
        let block_bytes = off;
        let shift = (usize::BITS - (block_bytes.max(1) - 1).leading_zeros()) as u32;
        let stride = 1usize << shift;
        let slab_bytes = stride * self.num_cpus as usize;

        // --- object storage: enough to fill the budget across classes ---
        // Phase 1 keeps it simple: reserve slab metadata + a flat object arena
        // sized to the budget, and carve objects per class lazily into central.
        let total = slab_bytes + self.budget_bytes;
        let reservation = Reservation::reserve(total)?;
        let base = reservation.base();

        // Initialize per-CPU headers (capacity per class).
        for cpu in 0..self.num_cpus as usize {
            let blk = unsafe { base.as_ptr().add(cpu * stride) };
            for c in 0..num_classes {
                let hdr = unsafe { blk.add(c * core::mem::size_of::<Header>()) as *mut Header };
                unsafe {
                    (*hdr).current = 0;
                    (*hdr).capacity = cap_for_class(c, self.cap_per_class);
                }
                // locks already zero (free) from mmap/zeroed reservation.
            }
        }

        let classes_loc: &'static [ClassLoc] =
            allocator_api2::boxed::Box::leak(classes_loc.into_boxed_slice());
        let slab = unsafe { SlabLayout::new(base, self.num_cpus, shift, classes_loc) };

        // The object arena follows the slab metadata. Fresh objects are carved
        // from it on demand by whichever class needs them; the byte budget (not
        // the arena split) is the real limit. We size the arena to the budget
        // (rounded down by alignment) so it only drains at true exhaustion.
        let arena_base = unsafe { base.as_ptr().add(slab_bytes) };
        let arena = BumpArena {
            cursor: core::sync::atomic::AtomicUsize::new(arena_base as usize),
            end: arena_base as usize + self.budget_bytes,
            base: arena_base as usize,
        };
        let spans = super::meta::SpanTable::new(arena_base, self.budget_bytes);
        let bitmaps = super::meta::SpanBitmaps::new(arena_base, self.budget_bytes);
        let large = LargeAllocator::new(arena_base, self.budget_bytes);
        // Central pools sharded by home: `num_shards` rows of `num_classes`.
        let central =
            crate::sys_boxed_slice(num_shards as usize * num_classes, |_| CentralList::new());

        Ok(SubHeap {
            name: self.name,
            on_exhaust: self.on_exhaust,
            slab,
            central,
            bitmaps,
            num_shards,
            arena,
            spans,
            large,
            budget_bytes: self.budget_bytes,
            counters: PerCpuCounters::new(self.num_cpus),
            _reservation: Some(reservation),
        })
    }
}

fn default_num_cpus() -> u32 {
    #[cfg(target_os = "linux")]
    {
        let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_CONF) };
        if n > 0 {
            return n as u32;
        }
    }
    std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4)
}
