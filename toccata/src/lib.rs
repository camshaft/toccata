// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! `toccata` — toccata as a process-wide `#[global_allocator]`.
//!
//! This is the application entry point: `install!` (or `configure`) wires
//! toccata's never-stall pool in as the process allocator. Libraries that want
//! toccata's primitives without imposing a global allocator depend on
//! `toccata-core` instead.
//!
//! ```ignore
//! #[global_allocator]
//! static GLOBAL: toccata::Toccata = toccata::Toccata::new();
//!
//! fn main() {
//!     toccata::configure(8 << 30); // 8 GiB budget, reserved up front
//!     // ...
//! }
//! ```
//!
//! ## Backpressure = abort, not panic
//!
//! Unwinding out of `GlobalAlloc::alloc` is UB, and the panic machinery itself
//! allocates. So "panic-as-backpressure" is a non-unwinding `#[cold]` handler
//! that writes a diagnostic with raw `write(2)` and calls `abort()`. A fast,
//! clean crash that a supervisor restarts beats a multi-second reclaim stall.
//!
//! ## Before `configure()`: delegate to the system allocator
//!
//! A `#[global_allocator]` must serve allocations that happen before `main`
//! runs `configure()` (static initializers, std startup, the config parse, and
//! toccata's own `configure()`-time bookkeeping). Rather than a special
//! bootstrap arena with a per-op "are we bootstrapping?" check, toccata simply
//! **delegates to the system allocator** until the pool exists. After
//! `configure()`, small allocations route to the locked pool.
//!
//! `dealloc` needs no bootstrap bookkeeping: it does a single **pool range
//! check** (`base <= ptr < base+len`) — which the span lookup needs anyway — and
//! routes in-pool pointers to toccata, everything else (pre-configure
//! allocations, over-aligned spill) to `System`. One branch, no separate
//! bootstrap state.

use toccata_core::{sizeclass, OnExhaust, SubHeap, SubHeapBuilder, ThreadCache};
use core::alloc::{GlobalAlloc, Layout};
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::alloc::System;
use std::ptr::NonNull;

/// The configured main sub-heap, published once by `configure()`. Null until
/// then (pre-configure allocations go to [`System`]).
static MAIN: AtomicPtr<SubHeap> = AtomicPtr::new(core::ptr::null_mut());

/// Cached pool address range, published by `configure()` so `dealloc` can route
/// by range. `POOL_BASE` holds the arena start; `POOL_END` holds `base + len`
/// (precomputed so `in_pool` is two loads + a `base<=p && p<end` range test, with
/// no add on the hot path). Both are 0 until configured (`base==end==0` ⇒ the
/// range is empty ⇒ every pointer routes to System, the correct pre-configure
/// behavior). Kept as two statics rather than one struct because the codegen
/// places each behind one `adrp`+`ldr` either way; the precomputed `end` saves
/// the hot-path `add`.
static POOL_BASE: AtomicUsize = AtomicUsize::new(0);
static POOL_END: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn in_pool(ptr: *mut u8) -> bool {
    let a = ptr as usize;
    // Relaxed loads: published once at configure before any in-pool pointer can
    // exist, so the values are visible to any thread that holds such a pointer.
    let base = POOL_BASE.load(Ordering::Relaxed);
    let end = POOL_END.load(Ordering::Relaxed);
    a >= base && a < end
}

/// A snapshot of the global allocator's byte accounting. `None` from
/// [`stats`] until [`configure`] has run.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Bytes currently live (allocated and not yet freed).
    pub live_bytes: usize,
    /// Live object count.
    pub live_objects: u64,
    /// High-water of bytes ever carved from the bump arena — monotonic, never
    /// retreats on free. This is toccata's faithful "peak touched footprint":
    /// because the whole budget is `mlock`'d up front, process RSS is pinned at
    /// the budget and cannot show fragmentation, but `carved_bytes` grows only as
    /// fresh spans/large-runs are claimed. The ratio `carved_bytes / live_bytes`
    /// is the fragmentation / stranding factor — the number a fragmentation
    /// benchmark wants, directly comparable to peak RSS of a `madvise`-based
    /// allocator.
    pub carved_bytes: usize,
    /// Total reserved budget.
    pub budget_bytes: usize,
}

/// Snapshot the global allocator's accounting, or `None` if `configure` has not
/// run yet. Intended for benchmarks and observability, not the hot path.
pub fn stats() -> Option<Stats> {
    let m = MAIN.load(Ordering::Acquire);
    // SAFETY: MAIN, once non-null, points at a leaked 'static SubHeap.
    let sub = unsafe { m.as_ref()? };
    Some(Stats {
        live_bytes: sub.live_bytes(),
        live_objects: sub.live_objects(),
        carved_bytes: sub.carved_bytes(),
        budget_bytes: sub.budget_bytes(),
    })
}

/// Configure toccata's global allocator with a total byte budget, reserved and
/// `mlock`'d up front. Best invoked from a constructor before `main` via the
/// [`install!`] macro, so the pool exists before the first allocation. Panics at
/// init (before traffic — acceptable) if the reservation fails.
pub fn configure(total_bytes: usize) {
    configure_with(SubHeapBuilder::new("global", total_bytes).on_exhaust(OnExhaust::Abort));
}

/// Configure with a custom builder (e.g. fewer shards / a tuned cap).
pub fn configure_with(builder: SubHeapBuilder) {
    if MAIN.load(Ordering::Acquire).is_null() {
        let sub = builder
            .build_standalone()
            .unwrap_or_else(|e| panic!("toccata: global reservation failed at init: {e}"));
        let (base, len) = sub.reservation_range();
        // Leak via the SYSTEM allocator (toccata_core::SysBox), NOT a global
        // `Box` — this runs at init and must not route through toccata itself.
        let leaked: &'static SubHeap = toccata_core::SysBox::leak(toccata_core::sys_box(sub));
        let leaked_ptr = leaked as *const SubHeap as *mut SubHeap;
        // Only the first configure wins; a racing/second one leaks its sub-heap
        // (negligible; configure runs once at init).
        if MAIN
            .compare_exchange(core::ptr::null_mut(), leaked_ptr, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Publish the pool range for dealloc routing. The pool is already
            // reserved + populated + locked; from here toccata issues no
            // OS-memory syscall against it (structural never-stall — no flag).
            // Store END last with Release: a thread that observes a non-zero END
            // (the upper bound it tests against) has also observed BASE.
            POOL_BASE.store(base, Ordering::Relaxed);
            POOL_END.store(base + len, Ordering::Release);
            // Optionally start the background reclaim supervisor (the `frag
            // crossclass`/stranding fix). Gated behind both the `supervisor` cargo
            // feature AND the `TOCCATA_SUPERVISOR=1` env var so the default global
            // build is byte-for-byte unaffected. Reclaim-only: it takes central spin
            // locks and issues NO kernel call / membarrier, so it preserves
            // never-stall. The handle is leaked (lives for the process).
            #[cfg(feature = "supervisor")]
            maybe_start_supervisor(leaked);
        }
    }
}

/// Start the reclaim supervisor if `TOCCATA_SUPERVISOR=1`. Leaks the handle so it
/// runs for the process lifetime. Reclaim-only (no seize/membarrier) — never-stall
/// safe. Optionally `TOCCATA_SUPERVISOR_MS` sets the sweep interval (default 50ms).
#[cfg(feature = "supervisor")]
fn maybe_start_supervisor(sh: &'static SubHeap) {
    use toccata_core::supervisor::Supervisor;
    if std::env::var("TOCCATA_SUPERVISOR").as_deref() != Ok("1") {
        return;
    }
    let mut b = Supervisor::builder().manage(sh);
    if let Some(ms) = std::env::var("TOCCATA_SUPERVISOR_MS").ok().and_then(|s| s.parse::<u64>().ok()) {
        b = b.interval(core::time::Duration::from_millis(ms));
    }
    // Leak the handle: dropping it would stop+join the thread, but the supervisor
    // must outlive `configure` for the whole process.
    core::mem::forget(b.spawn());
}

/// Install toccata as the process `#[global_allocator]` **and** reserve its pool
/// in a constructor that runs before `main` — so the pool exists before the
/// first allocation and the hot path needs no "configured yet?" check.
///
/// `$cfg` is a `fn() -> usize` returning the total byte budget. It runs at
/// constructor time and may do anything (read env vars, parse a config file,
/// compute a fraction of system RAM); its own allocations go to the system
/// allocator, never recursing into toccata.
///
/// ```ignore
/// fn heap_budget() -> usize {
///     std::env::var("TOCCATA_HEAP_MB").ok()
///         .and_then(|s| s.parse::<usize>().ok())
///         .map(|mb| mb << 20)
///         .unwrap_or(8 << 30) // 8 GiB default
/// }
/// toccata::install!(heap_budget);
///
/// fn main() { /* pool already live; alloc hot path is check-free */ }
/// ```
#[macro_export]
macro_rules! install {
    ($cfg:expr) => {
        #[global_allocator]
        static TOCCATA_GLOBAL: $crate::Toccata = $crate::Toccata::new();

        #[$crate::ctor::ctor]
        fn __toccata_install() {
            let total: usize = ($cfg)();
            $crate::configure(total);
        }
    };
}

// Re-export `ctor` so the `install!` macro can reference `$crate::ctor`.
#[doc(hidden)]
pub use ctor;

// Re-export the full primitive surface so an application that already depends on
// `toccata` (for the global allocator) can also build a `FramePool` etc. without
// adding a separate `toccata-core` dependency. Libraries that want *only* the
// primitives (no global allocator) should depend on `toccata-core` directly.
pub use toccata_core as primitives;
pub use toccata_core::{
    buf_frame_pool, frame_pool, typed_frame_pool, Frame, FrameBuf, FrameCache, FrameMeta, FrameMut,
    FramePool, FramePoolError, FrameSource, Owned, Reclaim, RefCount, Region, Require, Reservation,
    ReserveOpts, Shared, TypedPool,
};

/// The toccata global allocator handle (zero-sized; state is in statics).
pub struct Toccata;

impl Toccata {
    pub const fn new() -> Self {
        Toccata
    }
}

impl Default for Toccata {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Thread-local L1 cache ----
//
// Holds this thread's `&'static SubHeap` (resolved once — no per-op `MAIN.load`)
// and L1 magazines. Stored behind Rust's `thread_local!`, but only as a
// `Cell<*mut Tls>` with a `const` initializer and no `Drop` (see `mod tls`
// below). That shape sidesteps both dynamic-TLS hazards on the alloc path: the
// `const` init means the access is a direct thread-pointer-relative load with no
// lazy-init guard (no `__tls_get_addr` malloc), and a `Drop`-free `*mut Tls`
// registers no `__cxa_thread_atexit` dtor (which would also malloc) — both would
// recurse into this allocator. The `Tls` storage itself is `calloc`'d off the
// hot path and its thread-exit cleanup runs via a dedicated `pthread_key` whose
// `setspecific` happens once at init. We use the built-in `thread_local!` here
// rather than `toccata_core::sys::pthread_local` because the native TLS load is
// measurably faster than `pthread_getspecific` (see commit 97c2392); the
// disassembly was verified to contain none of those three symbols on the hot
// path. `PthreadLocal` remains for consumers that can't use this pattern (e.g.
// the old-glibc rseq self-register fallback in `toccata_core::rseq`).
//
// `Tls::heap` is `None` until this thread first allocates after `configure()`;
// before that, the global allocator delegates to `System` directly (no TLS
// access at all), so pre-`configure()` allocations never even create the TLS.
struct Tls {
    heap: Option<&'static SubHeap>,
    cache: ThreadCache,
}

impl Tls {
    fn new() -> Self {
        Self { heap: None, cache: ThreadCache::new() }
    }

    /// Resolve (and cache) this thread's sub-heap. `None` if not configured yet.
    #[inline]
    fn heap(&mut self) -> Option<&'static SubHeap> {
        if self.heap.is_none() {
            let m = MAIN.load(Ordering::Acquire);
            if !m.is_null() {
                // SAFETY: MAIN, once non-null, points at a leaked 'static SubHeap.
                self.heap = Some(unsafe { &*m });
            }
        }
        self.heap
    }

    /// Flush all L1 magazines back to L2 (called on thread exit) so cached
    /// objects return to the shared pool rather than being stranded in a dead
    /// thread's cache.
    #[cfg(target_os = "linux")]
    fn flush(&mut self) {
        if let Some(sub) = self.heap {
            self.cache.flush_all(sub);
        }
    }
}

// Fast, recursion-safe thread-local for the per-thread cache.
//
// This matches what Rust's own fast TLS does on STABLE: a `thread_local!` with a
// `const {}` initializer holding a type with NO `Drop` compiles to direct
// native-TLS access (thread-pointer-relative load) with no lazy-init guard and —
// critically — no `__cxa_thread_atexit` registration (which would `malloc` and
// recurse into this allocator). `#[thread_local]` the *attribute* is nightly;
// the `thread_local!` *macro* with const init is the stable equivalent.
//
// We hold only a `*mut Tls` pointer here. The `Tls` storage is `calloc`'d (libc,
// not toccata) on first access. Thread-exit cleanup (flush magazines, free
// storage) runs via a process `pthread_key` whose `pthread_setspecific` is done
// once at init — never on the hot path. So the hot path is a single TLS load.
#[cfg(target_os = "linux")]
mod tls {
    use super::Tls;
    use core::cell::Cell;
    use std::sync::Once;

    thread_local! {
        // No Drop on `*mut Tls` => no atexit malloc; const init => fast native TLS.
        static TLS_PTR: Cell<*mut Tls> = const { Cell::new(core::ptr::null_mut()) };
    }

    // A pthread key whose ONLY purpose is the thread-exit destructor.
    static KEY_ONCE: Once = Once::new();
    static mut KEY: libc::pthread_key_t = 0;

    unsafe extern "C" fn on_thread_exit(ptr: *mut libc::c_void) {
        if ptr.is_null() {
            return;
        }
        let t = ptr as *mut Tls;
        // Flush magazines back to L2 so cached memory returns to the pool.
        (*t).flush();
        core::ptr::drop_in_place(t);
        libc::free(ptr);
    }

    fn key() -> libc::pthread_key_t {
        KEY_ONCE.call_once(|| unsafe {
            let mut k: libc::pthread_key_t = 0;
            if libc::pthread_key_create(&mut k, Some(on_thread_exit)) == 0 {
                KEY = k;
            }
        });
        unsafe { KEY }
    }

    #[inline]
    pub fn with<R>(f: impl FnOnce(&mut Tls) -> R) -> R {
        let p = TLS_PTR.with(|c| c.get());
        // Hot path: a live cached `Tls`. The cold first-touch (`init`) and the
        // calloc-failure fallback are both `#[inline(never)]`, so neither's large
        // (`size_of::<Tls>()` ≈ 12 KiB) stack frame inflates this caller — which
        // would otherwise force a multi-page stack-probe prologue on every
        // alloc/dealloc.
        let t = if p.is_null() { init() } else { p };
        if t.is_null() {
            return with_transient(f);
        }
        // SAFETY: `t` is this thread's live, initialized `Tls`.
        f(unsafe { &mut *t })
    }

    /// calloc failed: operate on a transient, uncached `Tls`. Out-of-line so its
    /// ~12 KiB stack frame never sizes the hot `with` caller.
    #[cold]
    #[inline(never)]
    fn with_transient<R>(f: impl FnOnce(&mut Tls) -> R) -> R {
        let mut tmp = Tls::new();
        f(&mut tmp)
    }

    #[cold]
    #[inline(never)]
    fn init() -> *mut Tls {
        // Storage from libc (never toccata). calloc zeroes; write the real init.
        // This frame holds a ~12 KiB `Tls` temporary, but `init` is
        // `#[inline(never)]` so that only sizes *this* cold frame, not the hot
        // `with` caller. (Once-per-thread; off the hot path.)
        let mem = unsafe { libc::calloc(1, core::mem::size_of::<Tls>().max(1)) } as *mut Tls;
        if mem.is_null() {
            return core::ptr::null_mut();
        }
        unsafe {
            core::ptr::write(mem, Tls::new());
            TLS_PTR.with(|c| c.set(mem));
            // Register for thread-exit cleanup (off the hot path, once per thread).
            let k = key();
            let _ = libc::pthread_setspecific(k, mem as *const libc::c_void);
        }
        mem
    }
}

#[cfg(target_os = "linux")]
#[inline]
fn with_tls<R>(f: impl FnOnce(&mut Tls) -> R) -> R {
    tls::with(f)
}

// Non-Linux dev fallback: a plain thread_local (these platforms don't run the
// global allocator in production; correctness only).
#[cfg(not(target_os = "linux"))]
#[inline]
fn with_tls<R>(f: impl FnOnce(&mut Tls) -> R) -> R {
    thread_local! { static T: core::cell::RefCell<Tls> = core::cell::RefCell::new(Tls::new()); }
    T.with(|t| f(&mut t.borrow_mut()))
}

unsafe impl GlobalAlloc for Toccata {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == 0 {
            return layout.align() as *mut u8; // dangling, aligned (std contract)
        }
        let align = layout.align();
        let need = layout.size().max(align);

        // ---- Steady-state hot path: TLS + magazine pop. NO per-op POOL_LEN /
        // configured check — a magazine hit needs nothing else. The "are we
        // configured?" question only arises on a magazine MISS (the cold refill),
        // which is also where pre-configure routing to System lives. ----
        if need <= sizeclass::MAX_SMALL && align <= sizeclass::MIN_ALIGN {
            let class = sizeclass::class_of(need).unwrap();
            // LARGE multi-cache-line requests (≥ 1 KiB) take an out-of-line path that
            // also software-prefetches-for-write a buffer a few allocations ahead
            // (the caller almost always memsets the whole buffer next; on the cross-
            // thread path it's DRAM-cold, so those stores stall — a PMU sweep showed
            // jemalloc's HW prefetcher hides this and toccata's didn't). It MUST be
            // `#[inline(never)]`: its magazine-pop closure would otherwise inflate
            // this hot `alloc` frame and re-trigger the multi-page stack-probe
            // prologue that regresses the small-object / single-thread path (the
            // 12 KiB-Tls-frame hazard). Out-of-line, the small path's codegen is
            // untouched.
            if need >= 1024 {
                return alloc_large_smallclass(class, layout);
            }
            return with_tls(|t| {
                if let Some(p) = t.cache.try_pop(class) {
                    return p.as_ptr(); // hot: pure array pop
                }
                // Miss: resolve L2 (or System if not yet configured) and refill.
                match t.heap() {
                    Some(sub) => match t.cache.refill(sub, class) {
                        Some(p) => p.as_ptr(),
                        None => oom_backpressure(layout, "global sub-heap budget exhausted"),
                    },
                    None => System.alloc(layout),
                }
            });
        }

        // Over-aligned / large: cold path (no magazine).
        with_tls(|t| {
            let Some(sub) = t.heap() else { return System.alloc(layout) };
            let ptr = if need > sizeclass::MAX_SMALL {
                if align > toccata_core::meta::SPAN_BYTES {
                    oom_backpressure(layout, "alignment exceeds span size");
                }
                sub.alloc_large(need)
            } else {
                match sizeclass::aligned_class(need, align) {
                    Some(c) => sub.alloc_class(c),
                    None => sub.alloc_large(need),
                }
            };
            match ptr {
                Some(p) => p.as_ptr(),
                None => oom_backpressure(layout, "global sub-heap budget exhausted"),
            }
        })
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // The single routing check: is this pointer in toccata's pool? If not,
        // it's a System allocation (pre-configure or over-aligned spill) — the
        // span lookup would need this range check anyway, so it's not extra work.
        if !in_pool(ptr) {
            System.dealloc(ptr, layout);
            return;
        }
        let nn = NonNull::new_unchecked(ptr);
        let align = layout.align();
        let need = layout.size().max(align);

        // Small + default-aligned frees go to the L1 magazine (TLS + array push).
        if need <= sizeclass::MAX_SMALL && align <= sizeclass::MIN_ALIGN {
            let class = sizeclass::class_of(need).unwrap();
            with_tls(|t| match t.heap() {
                Some(sub) => t.cache.free(sub, nn, class),
                None => {
                    if let Some(m) = NonNull::new(MAIN.load(Ordering::Acquire)) {
                        let _ = (*m.as_ptr()).dealloc_by_ptr(nn);
                    }
                }
            });
            return;
        }

        // Over-aligned / large: recover the class from span metadata directly.
        with_tls(|t| {
            if let Some(sub) = t.heap() {
                let _ = sub.dealloc_by_ptr(nn);
            } else if let Some(m) = NonNull::new(MAIN.load(Ordering::Acquire)) {
                let _ = (*m.as_ptr()).dealloc_by_ptr(nn);
            }
        });
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = self.alloc(layout);
        if !p.is_null() {
            // Fresh arena pages are zero, but returned-to-pool slots are dirty,
            // so we must zero explicitly (R: alloc_zeroed correctness).
            core::ptr::write_bytes(p, 0, layout.size());
        }
        p
    }
}

/// Out-of-line allocation path for **large small-classes** (≥ 1 KiB, ≤ MAX_SMALL).
/// Identical to the hot small-class path but pops via `alloc_prefetch`, which
/// software-prefetches-for-write a buffer a few allocations ahead so the caller's
/// imminent `memset` of the (cross-core, DRAM-cold) buffer doesn't stall on
/// read-for-ownership. `#[inline(never)]` so its closure's frame never inflates the
/// hot `alloc` and re-triggers the stack-probe prologue on the small-object path.
#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn alloc_large_smallclass(class: usize, layout: Layout) -> *mut u8 {
    let osz = sizeclass::size_of_class(class);
    with_tls(|t| {
        if let Some(p) = t.cache.alloc_prefetch(class, osz) {
            return p.as_ptr();
        }
        match t.heap() {
            Some(sub) => match t.cache.refill(sub, class) {
                Some(p) => p.as_ptr(),
                None => oom_backpressure(layout, "global sub-heap budget exhausted"),
            },
            None => System.alloc(layout),
        }
    })
}

#[cfg(not(target_os = "linux"))]
#[inline(never)]
unsafe fn alloc_large_smallclass(class: usize, layout: Layout) -> *mut u8 {
    with_tls(|t| match t.heap() {
        Some(sub) => match t.cache.alloc(sub, class) {
            Some(p) => p.as_ptr(),
            None => oom_backpressure(layout, "global sub-heap budget exhausted"),
        },
        None => System.alloc(layout),
    })
}

/// The backpressure handler: NEVER unwinds (that would be UB out of GlobalAlloc)
/// and NEVER allocates (the panic machinery would). Writes a raw diagnostic and
/// aborts. This abort *is* the hard backpressure floor, fired at a budget we
/// chose — below any kernel limit — so we never become the cgroup-throttled task.
#[cold]
#[inline(never)]
fn oom_backpressure(layout: Layout, why: &str) -> ! {
    // Raw write(2) — no formatting machinery (it allocates). Helpers live in
    // toccata-sys::diag so this path and Reservation's degradation notice share
    // one no-alloc implementation.
    use toccata_core::sys::diag;
    let _ = diag::write_stderr(b"toccata: allocation failed (backpressure abort): ");
    let _ = diag::write_stderr(why.as_bytes());
    let _ = diag::write_stderr(b" [size=");
    let mut buf = [0u8; 20];
    let _ = diag::write_stderr(diag::usize_to_dec(layout.size(), &mut buf));
    let _ = diag::write_stderr(b"]\n");
    std::process::abort();
}

#[cfg(test)]
mod tests;
