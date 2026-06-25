// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! `FramePool<M>` — a fixed-size-frame allocator with out-of-band per-frame
//! metadata.
//!
//! A region (toccata's own [`Reservation`], or a caller-supplied mmap such as an
//! AF_XDP UMEM) is pre-divided at construction into `n_frames` equal,
//! power-of-two-sized frames. Allocation hands out a whole frame; free returns it.
//! Because frames are homogeneous this is far simpler than [`SubHeap`](crate::SubHeap):
//! no size classes, no bump arena, no large path, and — because a frame is not
//! "homed" on any CPU — no return-to-owner remote-free queue. Any CPU may free any
//! frame to its own per-CPU cache.
//!
//! ## Out-of-band metadata (the distinguishing feature)
//!
//! Each frame has a slot in a parallel `M[n_frames]` array, indexed by
//! `frame_index = (ptr - base) >> frame_shift`. Nothing about a frame's identity,
//! lifecycle, or free-list linkage lives **in the frame body** — which is
//! mandatory for UMEM, where the kernel DMAs a received packet over the entire
//! frame. `M` is user-chosen:
//! * `M = ()` — a plain pool; [`FrameMeta::on_free`] always reclaims (zero cost).
//! * `M = `[`RefCount`] — a refcounted pool; the last release reclaims. This is
//!   the building block for `Owned`/`Shared`-style handles.
//! * a user type implementing [`FrameMeta`] — e.g. an AF_XDP ring-ownership state.
//!
//! ## Never-stall
//!
//! Frames live in a per-CPU rseq slab (atomic-free fast path on Linux
//! x86_64/aarch64, locked baseline elsewhere) backed by a central spin-locked
//! stack. Both are preallocated; no path ever calls the kernel. The free-list
//! linkage is held **out of band** (the slab's pointer slots and the central
//! array), never threaded through a frame body — so a frame ceded to the kernel
//! and later reclaimed by offset is sound.

use crate::{
    rseq::slab::{ClassLoc, CpuStack, Fast, Header, SlabLayout},
    sys::{Reservation, ReserveError, ReserveOpts},
    sys_boxed_slice, Sys, SysBoxSlice,
};
use core::{
    cell::UnsafeCell,
    sync::atomic::{fence, AtomicU32, AtomicUsize, Ordering},
};
use std::ptr::NonNull;

/// How many frames move between a per-CPU slab and the central stack per
/// slow-path event. Mirrors tcmalloc's `num_to_move`.
const BATCH: usize = 32;
/// Per-CPU slab capacity (frames cached per CPU). Bounded; the rest live central.
const CAP_PER_CPU: u32 = 256;

// ---------------------------------------------------------------------------
// Frame metadata policy
// ---------------------------------------------------------------------------

/// What [`FramePool::free`] should do after a frame's metadata processes a release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reclaim {
    /// Return the frame to the pool (it is now unused).
    Free,
    /// Keep the frame out of the pool (other references remain).
    Keep,
}

/// Per-frame metadata + reclaim policy. Held in the pool's out-of-band `M[]`
/// array, never in the frame body. `Default` provides the initial (idle) value
/// for every slot at construction.
pub trait FrameMeta: Default {
    /// Called by [`FramePool::alloc`] when a fresh frame is handed out — reset the
    /// metadata to its "one live allocation" state.
    fn on_alloc(&self);
    /// Called by [`FramePool::free`] to release one reference. Returns whether the
    /// frame should now be reclaimed.
    fn on_free(&self) -> Reclaim;
}

/// The trivial policy: every frame has exactly one owner, so a free always
/// reclaims. Zero-sized and zero-cost (the calls inline away).
impl FrameMeta for () {
    #[inline(always)]
    fn on_alloc(&self) {}
    #[inline(always)]
    fn on_free(&self) -> Reclaim {
        Reclaim::Free
    }
}

/// An atomic reference count usable as a [`FrameMeta`]: `alloc` sets it to 1,
/// [`retain`](Self::retain) bumps it (a clone), and `free` releases one — the
/// frame is reclaimed only when the count reaches zero. The discipline matches
/// `Arc`: `Release` on every drop, an `Acquire` fence on the last.
#[derive(Debug, Default)]
pub struct RefCount(AtomicUsize);

impl RefCount {
    /// Take an additional reference (a clone). Sound only while at least one
    /// reference is already held (so the count cannot spuriously reach zero).
    #[inline]
    pub fn retain(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    /// Release one reference. Returns [`Reclaim::Free`] iff this was the last one
    /// (with the `Acquire` fence already applied, so the caller may now safely
    /// destroy the contents), else [`Reclaim::Keep`]. This is the primitive the
    /// typed [`Owned`]/[`Shared`] handles call so they can run a destructor before
    /// returning the frame.
    #[inline]
    pub fn release(&self) -> Reclaim {
        if self.0.fetch_sub(1, Ordering::Release) != 1 {
            return Reclaim::Keep;
        }
        fence(Ordering::Acquire);
        Reclaim::Free
    }
    /// Current strong count (instrumentation / tests).
    #[inline]
    pub fn count(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

impl FrameMeta for RefCount {
    #[inline]
    fn on_alloc(&self) {
        self.0.store(1, Ordering::Relaxed);
    }
    #[inline]
    fn on_free(&self) -> Reclaim {
        self.release()
    }
}

// ---------------------------------------------------------------------------
// Region
// ---------------------------------------------------------------------------

/// The backing memory a [`FramePool`] divides into frames.
pub enum Region {
    /// toccata owns the region (reserved + populated + locked): the never-stall
    /// path. Unmapped on drop.
    Owned(Reservation),
    /// A caller-supplied region toccata does **not** own (e.g. an AF_XDP UMEM the
    /// kernel registered). Never unmapped by the pool, never `mlock`'d by it — so
    /// the borrowed variant carries **no** anti-stall guarantee on its own.
    Borrowed { base: NonNull<u8>, len: usize },
}

impl Region {
    #[inline]
    fn base(&self) -> *mut u8 {
        match self {
            Region::Owned(r) => r.as_ptr(),
            Region::Borrowed { base, .. } => base.as_ptr(),
        }
    }
    #[inline]
    fn len(&self) -> usize {
        match self {
            Region::Owned(r) => r.len(),
            Region::Borrowed { len, .. } => *len,
        }
    }
}

// ---------------------------------------------------------------------------
// Central spin-locked stack of free frame pointers (out of band)
// ---------------------------------------------------------------------------

/// A bounded LIFO of free frame pointers, guarded by a tiny spin lock. Storage is
/// system-backed and preallocated to `n_frames` (every frame can be resident at
/// once), so push never reallocates and no path calls the kernel. Pointers are
/// stored here, **not** threaded through frame bodies — UMEM-safe.
struct Central {
    lock: AtomicU32,
    inner: UnsafeCell<CentralInner>,
}

struct CentralInner {
    slots: SysBoxSlice<*mut u8>,
    top: usize,
}

// SAFETY: `inner` is only touched under `lock`; the pointers are into the region.
unsafe impl Send for Central {}
unsafe impl Sync for Central {}

impl Central {
    fn new(n_frames: usize) -> Self {
        Self {
            lock: AtomicU32::new(0),
            inner: UnsafeCell::new(CentralInner {
                slots: sys_boxed_slice(n_frames, |_| core::ptr::null_mut()),
                top: 0,
            }),
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
}

struct CentralGuard<'a>(&'a Central);

impl CentralGuard<'_> {
    #[inline]
    fn push(&self, p: *mut u8) {
        // SAFETY: lock held; `top < n_frames` because we never hold more frames
        // than exist (push only frames previously popped or seeded once at init).
        let inner = unsafe { &mut *self.0.inner.get() };
        debug_assert!(inner.top < inner.slots.len(), "central stack overflow");
        inner.slots[inner.top] = p;
        inner.top += 1;
    }

    #[inline]
    fn pop(&self) -> Option<*mut u8> {
        // SAFETY: lock held.
        let inner = unsafe { &mut *self.0.inner.get() };
        if inner.top == 0 {
            return None;
        }
        inner.top -= 1;
        Some(inner.slots[inner.top])
    }
}

impl Drop for CentralGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.lock.store(0, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Frame handle
// ---------------------------------------------------------------------------

/// A handle to one allocated frame: just its base pointer. `Copy` — ownership is
/// tracked by the caller (and, for refcounted `M`, by the metadata), not by this
/// handle. Resolve its index / metadata / UMEM offset through the owning pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    ptr: NonNull<u8>,
}

impl Frame {
    /// The frame's base pointer.
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
    /// The frame's base pointer as a [`NonNull`].
    #[inline]
    pub fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }
}

// ---------------------------------------------------------------------------
// FramePool
// ---------------------------------------------------------------------------

/// Errors constructing a [`FramePool`].
#[derive(Debug)]
pub enum FramePoolError {
    /// `frame_size` was not a power of two (required for shift-based indexing).
    FrameSizeNotPow2(usize),
    /// The region was too small to hold even one frame.
    RegionTooSmall { len: usize, frame_size: usize },
    /// The backing reservation failed.
    Reserve(ReserveError),
}

impl std::fmt::Display for FramePoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FramePoolError::FrameSizeNotPow2(s) => {
                write!(f, "frame_size {s} must be a power of two")
            }
            FramePoolError::RegionTooSmall { len, frame_size } => {
                write!(f, "region of {len} bytes holds no {frame_size}-byte frame")
            }
            FramePoolError::Reserve(e) => write!(f, "frame pool reservation failed: {e}"),
        }
    }
}

impl std::error::Error for FramePoolError {}

/// A fixed-size-frame allocator with out-of-band per-frame metadata `M`.
pub struct FramePool<M = ()> {
    /// Base of the frame region (frame 0). Frames are `frame_size` apart.
    frames_base: *mut u8,
    /// log2(frame_size); `frame_index = (ptr - base) >> frame_shift`.
    frame_shift: u32,
    frame_size: usize,
    n_frames: usize,

    /// Per-CPU rseq slab (single class) — the L2 fast path.
    slab: SlabLayout,
    /// Central free stack — bulk storage + cross-CPU rebalance.
    central: Central,
    /// Out-of-band per-frame metadata, indexed by frame index.
    meta: SysBoxSlice<M>,

    /// Backing for the slab metadata (kept alive for the pool's lifetime; the
    /// `SlabLayout` holds raw pointers into it).
    _slab_buf: SysBoxSlice<u64>,
    /// The frame region itself (kept alive; `Owned` unmaps on drop).
    _region: Region,
}

// SAFETY: interior mutability is via the rseq slab (per-CPU / locked), the
// spin-locked central stack, and atomic metadata. Pointers are into the region.
unsafe impl<M: Send + Sync> Send for FramePool<M> {}
unsafe impl<M: Send + Sync> Sync for FramePool<M> {}

impl<M: FrameMeta + Send + Sync> FramePool<M> {
    /// Build a pool over a **caller-supplied** region (e.g. an AF_XDP UMEM). The
    /// pool does not own, lock, or unmap it. `frame_size` must be a power of two
    /// and `base` must be `frame_size`-aligned.
    ///
    /// # Safety
    /// `[base, base+len)` must be valid, writable, and outlive the pool. The
    /// caller must not hand the same region to two pools.
    pub unsafe fn over(
        base: NonNull<u8>,
        len: usize,
        frame_size: usize,
    ) -> Result<Self, FramePoolError> {
        Self::build(Region::Borrowed { base, len }, frame_size)
    }

    /// Build a pool that **owns** its region: toccata reserves + populates +
    /// locks `n_frames * frame_size` bytes per `opts` (default never-stall). This
    /// is the path that carries the anti-stall guarantee. `frame_size` must be a
    /// power of two.
    pub fn with_reservation(
        n_frames: usize,
        frame_size: usize,
        opts: ReserveOpts,
    ) -> Result<Self, FramePoolError> {
        if !frame_size.is_power_of_two() {
            return Err(FramePoolError::FrameSizeNotPow2(frame_size));
        }
        let mut opts = opts;
        opts.len = n_frames
            .checked_mul(frame_size)
            .expect("n_frames * frame_size overflow");
        let reservation = Reservation::reserve_with(opts).map_err(FramePoolError::Reserve)?;
        Self::build(Region::Owned(reservation), frame_size)
    }

    fn build(region: Region, frame_size: usize) -> Result<Self, FramePoolError> {
        if !frame_size.is_power_of_two() {
            return Err(FramePoolError::FrameSizeNotPow2(frame_size));
        }
        let base = region.base();
        let len = region.len();
        let n_frames = len / frame_size;
        if n_frames == 0 {
            return Err(FramePoolError::RegionTooSmall { len, frame_size });
        }
        let frame_shift = frame_size.trailing_zeros();

        // --- build the single-class per-CPU slab in a system-backed buffer ---
        // Block layout per CPU: [ Header(8) | lock(u32) | pad | slots(cap*8) ].
        let num_cpus = default_num_cpus();
        let header_off: u32 = 0;
        let lock_off: u32 = core::mem::size_of::<Header>() as u32; // 8
        let slots_off: u32 = (lock_off + 4 + 7) & !7; // 8-align after the lock => 16
        let block_bytes = slots_off as usize + CAP_PER_CPU as usize * 8;
        let shift = (usize::BITS - (block_bytes.max(1) - 1).leading_zeros()) as u32;
        let stride = 1usize << shift;
        let slab_bytes = stride * num_cpus as usize;

        // 8-aligned zeroed buffer (cast u64 slice -> u8 base); guarantees the
        // Header/lock/slot loads inside each block are aligned.
        let slab_buf: SysBoxSlice<u64> = sys_boxed_slice(slab_bytes.div_ceil(8), |_| 0u64);
        let slab_base = NonNull::new(slab_buf.as_ptr() as *mut u8).expect("slab buffer non-null");

        // Initialize each CPU block's header (current=0, capacity=CAP_PER_CPU).
        for cpu in 0..num_cpus as usize {
            let blk = unsafe { slab_base.as_ptr().add(cpu * stride) };
            let hdr = blk as *mut Header;
            unsafe {
                (*hdr).current = 0;
                (*hdr).capacity = CAP_PER_CPU;
            }
            // lock + slots already zero from the zeroed buffer.
        }

        let classes: &'static [ClassLoc] =
            allocator_api2::boxed::Box::leak(allocator_api2::boxed::Box::new_in(
                [ClassLoc {
                    header_off,
                    slots_off,
                    lock_off,
                }],
                Sys,
            ));
        // SAFETY: slab_base points at `slab_bytes` of live, correctly-carved,
        // 8-aligned storage held by `slab_buf` for the pool's lifetime.
        let slab = unsafe { SlabLayout::new(slab_base, num_cpus, shift, classes) };

        // --- metadata array + central stack, seeded with every frame ---
        let meta = sys_boxed_slice(n_frames, |_| M::default());
        let central = Central::new(n_frames);
        {
            let g = central.lock();
            for i in 0..n_frames {
                // SAFETY: frame i lies fully within [base, base+len).
                g.push(unsafe { base.add(i * frame_size) });
            }
        }

        Ok(Self {
            frames_base: base,
            frame_shift,
            frame_size,
            n_frames,
            slab,
            central,
            meta,
            _slab_buf: slab_buf,
            _region: region,
        })
    }

    /// Number of frames in the pool.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.n_frames
    }
    /// The fixed frame size in bytes.
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    #[inline]
    fn index_of(&self, ptr: *const u8) -> usize {
        (ptr as usize - self.frames_base as usize) >> self.frame_shift
    }

    /// The frame's index in `[0, capacity)`.
    #[inline]
    pub fn frame_index(&self, f: &Frame) -> usize {
        self.index_of(f.ptr.as_ptr())
    }

    /// This frame's out-of-band metadata.
    #[inline]
    pub fn meta(&self, f: &Frame) -> &M {
        &self.meta[self.index_of(f.ptr.as_ptr())]
    }

    /// The frame's offset from the region base — the address a ring (AF_XDP
    /// FILL/COMPLETION/RX/TX) speaks.
    #[inline]
    pub fn into_addr(&self, f: Frame) -> u64 {
        (f.ptr.as_ptr() as usize - self.frames_base as usize) as u64
    }

    /// Reconstitute a [`Frame`] from a region offset returned by a ring. The
    /// frame's metadata is intact (out of band) even if the kernel DMA'd over the
    /// frame body.
    ///
    /// # Safety
    /// `addr` must be a frame offset this pool previously handed out via
    /// [`into_addr`](Self::into_addr) (a multiple of `frame_size`, in range), and
    /// the frame must be logically owned by the caller again.
    #[inline]
    pub unsafe fn from_addr(&self, addr: u64) -> Frame {
        debug_assert!(
            (addr as usize) < self.n_frames * self.frame_size,
            "addr out of range"
        );
        debug_assert_eq!(addr as usize % self.frame_size, 0, "addr not frame-aligned");
        Frame {
            ptr: NonNull::new_unchecked(self.frames_base.add(addr as usize)),
        }
    }

    /// Allocate a frame, or `None` if the pool is exhausted. The handle-free path:
    /// hits the per-CPU rseq slab directly (no L1 magazine). For the fastest path,
    /// take a [`cache`](Self::cache) and use [`FrameCache::alloc`]. Never blocks
    /// beyond a brief per-CPU/central lock; never calls the kernel.
    #[inline]
    pub fn alloc(&self) -> Option<Frame> {
        let stack = CpuStack::current_fast(&self.slab);
        let ptr = self.pop_l2(&stack)?;
        // Initialize the frame's metadata for a fresh allocation (no-op for M=()).
        self.meta[self.index_of(ptr.as_ptr())].on_alloc();
        Some(Frame { ptr })
    }

    /// Release one reference to a frame. With `M = ()` this always returns the
    /// frame to the pool; with a refcounted `M` it returns only on the last
    /// reference (per [`FrameMeta::on_free`]).
    ///
    /// # Safety
    /// `f` must have come from this pool and the caller must hold a reference
    /// being released (no double-free).
    #[inline]
    pub unsafe fn free(&self, f: Frame) {
        if self.meta[self.index_of(f.ptr.as_ptr())].on_free() == Reclaim::Keep {
            return;
        }
        let stack = CpuStack::current(&self.slab);
        self.push_l2(&stack, f.ptr);
    }

    /// Return a frame to the pool **unconditionally**, with NO metadata processing
    /// (`on_free` is not called). The typed [`Owned`]/[`Shared`] handles use this:
    /// they own the refcount decision and run the value's destructor themselves,
    /// then hand back the now-dead frame. For a plain `M = ()` pool `recycle` and
    /// `free` are equivalent.
    ///
    /// # Safety
    /// `f` came from this pool, is logically dead (no live references, destructor
    /// already run), and is not recycled twice.
    #[inline]
    pub unsafe fn recycle(&self, f: Frame) {
        let stack = CpuStack::current(&self.slab);
        self.push_l2(&stack, f.ptr);
    }

    /// Allocate up to `out.len()` frames into `out`, returning how many were
    /// actually obtained (fewer than requested only at exhaustion). Reuses one
    /// per-CPU stack binding across the batch — the bulk path for refilling an
    /// AF_XDP **fill** ring. Each returned frame has had `on_alloc` applied.
    pub fn alloc_n(&self, out: &mut [Frame]) -> usize {
        let stack = CpuStack::current_fast(&self.slab);
        let mut n = 0;
        while n < out.len() {
            match self.pop_l2(&stack) {
                Some(ptr) => {
                    self.meta[self.index_of(ptr.as_ptr())].on_alloc();
                    out[n] = Frame { ptr };
                    n += 1;
                }
                None => break,
            }
        }
        n
    }

    /// Release a batch of frames, reusing one per-CPU stack binding — the bulk
    /// path for draining an AF_XDP **completion** ring. Each frame runs its
    /// metadata `on_free`; only those that reach zero references are reclaimed.
    ///
    /// # Safety
    /// Every frame came from this pool and the caller holds a reference being
    /// released for each (no double-free).
    pub unsafe fn free_n(&self, frames: &[Frame]) {
        let stack = CpuStack::current(&self.slab);
        for f in frames {
            if self.meta[self.index_of(f.ptr.as_ptr())].on_free() == Reclaim::Free {
                self.push_l2(&stack, f.ptr);
            }
        }
    }

    /// A thread-local L1 magazine in front of this pool's per-CPU slab — the
    /// fastest path (a magazine pop/push, ~5 instructions, amortizing the rseq
    /// prologue over a batch). Each thread/queue holds its own; on drop the
    /// magazine flushes its frames back to the pool.
    #[inline]
    pub fn cache(&self) -> FrameCache<'_, M> {
        FrameCache {
            pool: self,
            len: 0,
            ptrs: [core::ptr::null_mut(); FRAME_MAG_CAP],
        }
    }

    /// Raw L2 frame acquire: the per-CPU rseq slab (refilling from central on a
    /// miss). Moves a frame pointer out of the free pool with **no** metadata
    /// processing — the magazine / `alloc` wrapper applies `on_alloc`.
    #[inline]
    fn pop_l2(&self, stack: &CpuStack<'_>) -> Option<NonNull<u8>> {
        match stack.pop_on(0) {
            (Fast::Ok(p), _) => Some(p),
            (Fast::NeedsSlow, _) => self.refill(stack),
        }
    }

    /// Raw L2 frame release: push a (truly-free) frame pointer back to the per-CPU
    /// rseq slab (draining to central on overflow). No metadata processing.
    #[inline]
    unsafe fn push_l2(&self, stack: &CpuStack<'_>, p: NonNull<u8>) {
        match stack.push(0, p) {
            Fast::Ok(()) => {}
            Fast::NeedsSlow => self.drain(stack, p),
        }
    }

    /// Slow path: per-CPU slab empty. Pull a batch from central into the slab and
    /// return one frame. Purely userspace pointer movement.
    #[cold]
    fn refill(&self, stack: &CpuStack<'_>) -> Option<NonNull<u8>> {
        let central = self.central.lock();
        let handed = central.pop()?;
        for _ in 1..BATCH {
            let Some(p) = central.pop() else { break };
            // SAFETY: p is a frame pointer from this pool.
            let nn = unsafe { NonNull::new_unchecked(p) };
            if let Fast::NeedsSlow = stack.push(0, nn) {
                central.push(p); // slab filled; return to central
                break;
            }
        }
        // SAFETY: handed is a frame pointer from this pool.
        Some(unsafe { NonNull::new_unchecked(handed) })
    }

    /// Slow path: per-CPU slab full. Drain half a batch to central, then stash the
    /// freed frame. Purely userspace pointer movement.
    #[cold]
    fn drain(&self, stack: &CpuStack<'_>, ptr: NonNull<u8>) {
        let central = self.central.lock();
        for _ in 0..(BATCH / 2).max(1) {
            match stack.pop(0) {
                Fast::Ok(p) => central.push(p.as_ptr()),
                Fast::NeedsSlow => break,
            }
        }
        if let Fast::NeedsSlow = stack.push(0, ptr) {
            central.push(ptr.as_ptr());
        }
    }
}

/// L1 magazine capacity (frames cached per thread/queue). The deliberate,
/// bounded per-thread memory that buys the magazine-class hot path.
pub const FRAME_MAG_CAP: usize = 32;

/// A thread-local (or per-queue) L1 magazine in front of a [`FramePool`]'s
/// per-CPU slab — the fast path. Holds up to [`FRAME_MAG_CAP`] free frame pointers
/// inline; `alloc` pops one (refilling a batch from L2 on a miss) and `free` pushes
/// one (draining a batch to L2 on overflow). Borrows the pool; not `Sync` (one
/// owner thread). On drop, all cached frames flush back to the pool.
///
/// The magazine stores **free** frame pointers out of band (here, in the inline
/// array — never in a frame body), so it is UMEM-safe. Metadata (`on_alloc` /
/// `on_free`) is applied at the L1 boundary so the per-op fast path that hits the
/// magazine still maintains refcounts correctly.
pub struct FrameCache<'p, M: FrameMeta + Send + Sync = ()> {
    pool: &'p FramePool<M>,
    len: usize,
    ptrs: [*mut u8; FRAME_MAG_CAP],
}

impl<'p, M: FrameMeta + Send + Sync> FrameCache<'p, M> {
    /// Allocate a frame from the L1 magazine, refilling a batch from the pool's L2
    /// slab on a miss. Returns `None` only if the whole pool is exhausted.
    #[inline]
    pub fn alloc(&mut self) -> Option<Frame> {
        let ptr = if self.len > 0 {
            self.len -= 1;
            // SAFETY: a non-empty magazine slot holds a live free frame pointer.
            unsafe { NonNull::new_unchecked(self.ptrs[self.len]) }
        } else {
            self.refill()?
        };
        self.pool.meta[self.pool.index_of(ptr.as_ptr())].on_alloc();
        Some(Frame { ptr })
    }

    /// Release one reference to a frame. On the last reference (per
    /// [`FrameMeta::on_free`]) the frame returns to the L1 magazine (draining a
    /// batch to L2 on overflow); otherwise it is kept.
    ///
    /// # Safety
    /// `f` must have come from this cache's pool and the caller holds a reference
    /// being released (no double-free).
    #[inline]
    pub unsafe fn free(&mut self, f: Frame) {
        if self.pool.meta[self.pool.index_of(f.ptr.as_ptr())].on_free() == Reclaim::Keep {
            return;
        }
        self.recycle(f);
    }

    /// Return a frame to the magazine **unconditionally** (no `on_free`). The typed
    /// handles call this after deciding to reclaim and running the destructor.
    ///
    /// # Safety
    /// As [`FramePool::recycle`]: `f` is from this pool, logically dead, not
    /// recycled twice.
    #[inline]
    pub unsafe fn recycle(&mut self, f: Frame) {
        if self.len < FRAME_MAG_CAP {
            self.ptrs[self.len] = f.ptr.as_ptr();
            self.len += 1;
            return;
        }
        self.drain_then_push(f.ptr);
    }

    /// Cold: magazine empty — pull up to half a magazine from L2 and return one.
    #[cold]
    fn refill(&mut self) -> Option<NonNull<u8>> {
        let stack = CpuStack::current_fast(&self.pool.slab);
        let want = FRAME_MAG_CAP / 2;
        while self.len < want {
            match self.pool.pop_l2(&stack) {
                Some(p) => {
                    self.ptrs[self.len] = p.as_ptr();
                    self.len += 1;
                }
                None => break,
            }
        }
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: just populated.
        Some(unsafe { NonNull::new_unchecked(self.ptrs[self.len]) })
    }

    /// Cold: magazine full — flush half back to L2, then stash the freed frame.
    #[cold]
    unsafe fn drain_then_push(&mut self, ptr: NonNull<u8>) {
        let stack = CpuStack::current(&self.pool.slab);
        let drop_n = FRAME_MAG_CAP / 2;
        for _ in 0..drop_n {
            self.len -= 1;
            let p = NonNull::new_unchecked(self.ptrs[self.len]);
            self.pool.push_l2(&stack, p);
        }
        self.ptrs[self.len] = ptr.as_ptr();
        self.len += 1;
    }
}

impl<M: FrameMeta + Send + Sync> Drop for FrameCache<'_, M> {
    fn drop(&mut self) {
        // Flush cached free frames back to the pool's L2 so they aren't stranded
        // in a dead cache. These are already-free frames (metadata processed at
        // free time), so no `on_free` here.
        let stack = CpuStack::current(&self.pool.slab);
        while self.len > 0 {
            self.len -= 1;
            // SAFETY: live free frame pointer from this pool.
            unsafe {
                self.pool
                    .push_l2(&stack, NonNull::new_unchecked(self.ptrs[self.len]))
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Typed handle layer: Owned<P> / Shared<P> over a TypedPool
// ---------------------------------------------------------------------------

/// A concrete typed frame pool (generated by [`typed_frame_pool!`](crate)). It
/// ties a value type [`Item`](Self::Item) to a [`FramePool<RefCount>`] whose
/// frames are sized to hold one `Item`, and routes the typed handles' fast path
/// through the pool's per-thread magazine.
///
/// Implemented by a zero-sized marker type, so [`Owned`]/[`Shared`] over it are
/// pointer-sized and dispatch statically — the pool identity is in the *type*,
/// like `Box`'s allocator. Generated, not hand-implemented; the methods are the
/// seam the handle logic (written once, below) builds on.
///
/// # Safety
/// An implementor must guarantee `alloc_frame` returns frames at least
/// `size_of::<Item>()` bytes and suitably aligned, that `meta`/`recycle` operate
/// on this pool's frames, and that `meta` returns the [`RefCount`] for the frame
/// (set to 1 by `alloc_frame`).
///
/// The shared seam every pool-backed handle ([`Owned`]/[`Shared`], and the
/// buffer handles [`FrameMut`]/[`FrameBuf`]) is written over: frame
/// alloc/recycle, the out-of-band [`RefCount`], and the byte capacity of a frame.
/// Implemented by the zero-sized markers the pool macros generate.
///
/// # Safety
/// An implementor must guarantee `alloc_frame` returns frames of at least
/// `frame_capacity()` writable bytes (suitably aligned), that `recycle_frame`/
/// `refcount` operate on this pool's frames, and that `refcount` returns the
/// [`RefCount`] for the frame (set to 1 by `alloc_frame`).
pub unsafe trait FrameSource: 'static {
    /// Allocate a frame (refcount initialized to 1), or `None` when the pool is
    /// exhausted / not configured. Hits the per-thread magazine.
    fn alloc_frame() -> Option<Frame>;
    /// Return a now-dead frame to the pool (no `on_free`; the handle has already
    /// released the refcount and dropped any contents). Hits the per-thread magazine.
    ///
    /// # Safety
    /// `f` is from this pool, logically dead, not recycled twice.
    unsafe fn recycle_frame(f: Frame);
    /// The frame's reference count.
    ///
    /// # Safety
    /// `f` is a live frame from this pool.
    unsafe fn refcount(f: Frame) -> &'static RefCount;
    /// Writable bytes per frame.
    fn frame_capacity() -> usize;
}

/// A concrete typed frame pool: a [`FrameSource`] whose frames each hold one value
/// of type [`Item`](Self::Item) (the [`Owned`]/[`Shared`] story).
///
/// # Safety
/// Implementors must back this with a real [`FrameSource`] whose frames are each
/// sized and aligned for one [`Item`](Self::Item) at frame offset 0, so the
/// `Owned`/`Shared` handles can treat the frame body as that value.
pub unsafe trait TypedPool: FrameSource {
    /// The value stored one-per-frame.
    type Item: 'static;
}

/// The in-frame value of a typed pool `P`: just the `Item` at frame offset 0. The
/// refcount is out of band (in the pool's `RefCount` metadata), so the frame body
/// is exactly the value — and returning the frame (overwriting the body) can never
/// corrupt routing or the count.
#[inline]
unsafe fn value_ptr<P: TypedPool>(f: Frame) -> *mut P::Item {
    f.as_ptr() as *mut P::Item
}

/// A unique, owning handle to a pooled `P::Item`. Frees its frame (running the
/// value's destructor) on drop. `Deref`/`DerefMut` to the value; convert to a
/// shared [`Shared`] with [`into_shared`](Self::into_shared).
pub struct Owned<P: TypedPool> {
    frame: Frame,
    _pool: core::marker::PhantomData<P>,
}

// SAFETY: Owned is the unique owner; sending it moves sole ownership. Sound iff
// the value is Send (it crosses threads); the frame's recycle routes through the
// pool's cross-thread-safe magazine/L2.
unsafe impl<P: TypedPool> Send for Owned<P> where P::Item: Send {}
unsafe impl<P: TypedPool> Sync for Owned<P> where P::Item: Sync {}

impl<P: TypedPool> Owned<P> {
    /// Allocate a frame and move `value` into it. `None` if the pool is exhausted
    /// (fallible by design — the fallible front-end pattern).
    #[inline]
    pub fn new(value: P::Item) -> Option<Self> {
        let frame = P::alloc_frame()?;
        // SAFETY: a fresh frame is uninitialized storage of >= size_of::<Item>().
        unsafe { core::ptr::write(value_ptr::<P>(frame), value) };
        Some(Self {
            frame,
            _pool: core::marker::PhantomData,
        })
    }

    /// Promote this unique handle into a refcounted [`Shared`]. Zero-cost: the
    /// frame's refcount is already 1 (set at alloc), which is exactly a `Shared`'s
    /// initial count.
    #[inline]
    pub fn into_shared(self) -> Shared<P> {
        let frame = self.frame;
        core::mem::forget(self); // don't run Owned's Drop (would free the frame)
        Shared {
            frame,
            _pool: core::marker::PhantomData,
        }
    }

    /// This frame's index in the backing pool (instrumentation / addressing).
    #[inline]
    pub fn frame(&self) -> Frame {
        self.frame
    }
}

impl<P: TypedPool> core::ops::Deref for Owned<P> {
    type Target = P::Item;
    #[inline]
    fn deref(&self) -> &P::Item {
        // SAFETY: live frame, unique owner.
        unsafe { &*value_ptr::<P>(self.frame) }
    }
}

impl<P: TypedPool> core::ops::DerefMut for Owned<P> {
    #[inline]
    fn deref_mut(&mut self) -> &mut P::Item {
        // SAFETY: unique owner => exclusive access.
        unsafe { &mut *value_ptr::<P>(self.frame) }
    }
}

impl<P: TypedPool> Drop for Owned<P> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: unique owner. Run the value's destructor, then return the frame.
        // (refcount is 1; we own it, so no atomic dance is needed — but we still
        // release it to leave the count at 0 for the recycled frame.)
        unsafe {
            core::ptr::drop_in_place(value_ptr::<P>(self.frame));
            let _ = P::refcount(self.frame).release();
            P::recycle_frame(self.frame);
        }
    }
}

/// A shared, reference-counted handle to a pooled `P::Item` (Arc-like). Cloning
/// bumps the count; the last drop runs the value's destructor and returns the
/// frame. Cross-thread safe; the slot return routes to the owning pool.
pub struct Shared<P: TypedPool> {
    frame: Frame,
    _pool: core::marker::PhantomData<P>,
}

// SAFETY: refcounted; the frame return is atomic and routes to the owner. Sound
// iff Item is Send+Sync (clones observed cross-thread), matching Arc<T>.
unsafe impl<P: TypedPool> Send for Shared<P> where P::Item: Send + Sync {}
unsafe impl<P: TypedPool> Sync for Shared<P> where P::Item: Send + Sync {}

impl<P: TypedPool> Shared<P> {
    /// Allocate a frame and move `value` into it as a shared handle (count 1).
    #[inline]
    pub fn new(value: P::Item) -> Option<Self> {
        Owned::<P>::new(value).map(Owned::into_shared)
    }

    /// Current strong count (instrumentation / tests).
    #[inline]
    pub fn strong_count(&self) -> usize {
        // SAFETY: live frame while this handle exists.
        unsafe { P::refcount(self.frame).count() }
    }

    /// This frame's handle (instrumentation / addressing).
    #[inline]
    pub fn frame(&self) -> Frame {
        self.frame
    }
}

impl<P: TypedPool> Clone for Shared<P> {
    #[inline]
    fn clone(&self) -> Self {
        // Relaxed add: holding a live handle proves the count is >= 1 and rising,
        // so no concurrent drop can spuriously reach zero (descriptor.rs discipline).
        unsafe { P::refcount(self.frame).retain() };
        Shared {
            frame: self.frame,
            _pool: core::marker::PhantomData,
        }
    }
}

impl<P: TypedPool> core::ops::Deref for Shared<P> {
    type Target = P::Item;
    #[inline]
    fn deref(&self) -> &P::Item {
        // SAFETY: live while at least this handle exists; shared => &.
        unsafe { &*value_ptr::<P>(self.frame) }
    }
}

impl<P: TypedPool> Drop for Shared<P> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: release one ref; on the last (Acquire fence applied by release),
        // run the destructor and return the frame. Identity is recovered from the
        // frame, never the body, so the free is safe as it overwrites the slot.
        unsafe {
            if P::refcount(self.frame).release() == Reclaim::Free {
                core::ptr::drop_in_place(value_ptr::<P>(self.frame));
                P::recycle_frame(self.frame);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Split-mutable byte buffers: FrameMut / FrameBuf (the BytesMut model)
// ---------------------------------------------------------------------------
//
// This is the split receive-descriptor shape (`Unfilled`/`Filled`):
// shared OWNERSHIP of a frame via the out-of-band refcount, but PARTITIONED
// EXCLUSIVE access to disjoint byte windows. Unlike `Shared<T>` (Arc-shaped: many
// readers of the *whole* value), a `FrameBuf` can be `split_to` into two handles
// that each `payload_mut()` their own non-overlapping `[offset, offset+len)` —
// sound because each `&mut [u8]` is built from the raw frame pointer + disjoint
// (offset,len), so two live `&mut` never alias. The refcount frees the frame only
// when the last window drops.

/// A unique, writable handle to a freshly-allocated frame's bytes — the write
/// target (`descriptor.rs`'s `Unfilled`). Exclusive, so `&mut [u8]` over the whole
/// capacity is sound. Call [`freeze`](Self::freeze) to publish `len` filled bytes
/// as a splittable [`FrameBuf`]; drop to return the frame unused.
pub struct FrameMut<P: FrameSource> {
    frame: Frame,
    cap: u16,
    _pool: core::marker::PhantomData<P>,
}

// SAFETY: unique owner; the frame return routes through the pool's cross-thread
// magazine/L2. Bytes are Send/Sync-neutral (no interior type).
unsafe impl<P: FrameSource> Send for FrameMut<P> {}
unsafe impl<P: FrameSource> Sync for FrameMut<P> {}

impl<P: FrameSource> FrameMut<P> {
    /// Allocate a fresh writable frame (refcount 1), or `None` if exhausted.
    #[inline]
    pub fn new() -> Option<Self> {
        let frame = P::alloc_frame()?;
        let cap = P::frame_capacity().min(u16::MAX as usize) as u16;
        Some(Self {
            frame,
            cap,
            _pool: core::marker::PhantomData,
        })
    }

    /// The writable capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap as usize
    }

    /// The full writable buffer (exclusive).
    #[inline]
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: exclusive owner of the whole frame; cap <= frame capacity.
        unsafe { core::slice::from_raw_parts_mut(self.frame.as_ptr(), self.cap as usize) }
    }

    /// The frame handle (e.g. to recover a UMEM offset via the pool).
    #[inline]
    pub fn frame(&self) -> Frame {
        self.frame
    }

    /// Publish the first `len` bytes as a shared, splittable [`FrameBuf`]. Consumes
    /// the unique handle; the refcount (already 1) becomes the buffer's count.
    #[inline]
    pub fn freeze(self, len: u16) -> FrameBuf<P> {
        let len = len.min(self.cap);
        let frame = self.frame;
        core::mem::forget(self); // don't run FrameMut's Drop (would recycle)
        FrameBuf {
            frame,
            offset: 0,
            len,
            _pool: core::marker::PhantomData,
        }
    }
}

impl<P: FrameSource> Drop for FrameMut<P> {
    #[inline]
    fn drop(&mut self) {
        // Never filled: release the lone ref and return the frame.
        unsafe {
            let _ = P::refcount(self.frame).release();
            P::recycle_frame(self.frame);
        }
    }
}

/// A filled, shareable, **splittable** byte buffer over a frame
/// (`descriptor.rs`'s `Filled`). Holds a `[offset, offset+len)` window; clones via
/// [`split_to`](Self::split_to) carve disjoint windows that each own a reference
/// to the same frame. The frame returns to the pool when the last window drops.
pub struct FrameBuf<P: FrameSource> {
    frame: Frame,
    offset: u16,
    len: u16,
    _pool: core::marker::PhantomData<P>,
}

// SAFETY: refcounted shared ownership; windows are disjoint so `&mut` never alias;
// the frame return is atomic and routes to the owner.
unsafe impl<P: FrameSource> Send for FrameBuf<P> {}
unsafe impl<P: FrameSource> Sync for FrameBuf<P> {}

impl<P: FrameSource> FrameBuf<P> {
    /// The window length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }
    /// Whether the window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// This window's bytes (shared).
    #[inline]
    pub fn payload(&self) -> &[u8] {
        // SAFETY: filled via FrameMut; this handle owns [offset, offset+len).
        unsafe {
            core::slice::from_raw_parts(
                self.frame.as_ptr().add(self.offset as usize),
                self.len as usize,
            )
        }
    }

    /// This window's bytes (exclusive). Sound because `split_to` guarantees no two
    /// live `FrameBuf` windows over a frame overlap.
    #[inline]
    pub fn payload_mut(&mut self) -> &mut [u8] {
        // SAFETY: this handle exclusively owns [offset, offset+len) (disjoint from
        // every other window by construction).
        unsafe {
            core::slice::from_raw_parts_mut(
                self.frame.as_ptr().add(self.offset as usize),
                self.len as usize,
            )
        }
    }

    /// The frame handle (e.g. to recover a UMEM offset via the pool).
    #[inline]
    pub fn frame(&self) -> Frame {
        self.frame
    }

    /// Current shared count over this frame (instrumentation / tests).
    #[inline]
    pub fn ref_count(&self) -> usize {
        // SAFETY: live frame while this handle exists.
        unsafe { P::refcount(self.frame).count() }
    }

    /// Split the buffer in two at `at`. Afterwards `self` holds `[at, len)` and the
    /// returned buffer holds `[0, at)` — disjoint, each independently mutable, both
    /// over the same frame (one extra reference). `O(1)`.
    ///
    /// # Panics
    /// If `at > len`.
    #[must_use = "if you don't need the other half, use advance"]
    #[inline]
    pub fn split_to(&mut self, at: u16) -> Self {
        assert!(at <= self.len, "split_to past end");
        let head_offset = self.offset;
        self.offset += at;
        self.len -= at;
        // Take one more reference to the frame; even a zero-length head can still
        // exist as a handle, matching descriptor.rs.
        // SAFETY: we hold a live reference, so the count is >= 1 and rising.
        unsafe { P::refcount(self.frame).retain() };
        Self {
            frame: self.frame,
            offset: head_offset,
            len: at,
            _pool: core::marker::PhantomData,
        }
    }

    /// Keep the first `len` bytes, drop the rest (no refcount change).
    #[inline]
    pub fn truncate(&mut self, len: u16) {
        self.len = len.min(self.len);
    }

    /// Advance the window start by `len`, dropping those bytes (no refcount change).
    ///
    /// # Panics
    /// If `len > self.len()`.
    #[inline]
    pub fn advance(&mut self, len: u16) {
        assert!(len <= self.len, "advance past end");
        self.offset += len;
        self.len -= len;
    }
}

impl<P: FrameSource> core::ops::Deref for FrameBuf<P> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.payload()
    }
}

impl<P: FrameSource> Drop for FrameBuf<P> {
    #[inline]
    fn drop(&mut self) {
        // Release one window's reference; the last one returns the frame. No value
        // destructor — these are raw bytes.
        unsafe {
            if P::refcount(self.frame).release() == Reclaim::Free {
                P::recycle_frame(self.frame);
            }
        }
    }
}

/// Declare a **named, configured-once frame pool** with a per-thread native-TLS
/// magazine — the product-facing analogue of [`install!`](crate) for the global
/// allocator. Each invocation defines a distinct zero-sized type, so it gets its
/// own `thread_local!` magazine (a single thread-pointer-relative load on the hot
/// path) and its own thread-exit flush, exactly like the global allocator's TLS.
///
/// ```ignore
/// toccata_core::frame_pool!(pub PacketPool);                 // M = ()
/// toccata_core::frame_pool!(pub RcPool, meta = toccata_core::RefCount);
///
/// PacketPool::configure(1 << 20, 2048, ReserveOpts::new(0))?; // reserve+lock once
/// let f = PacketPool::alloc().expect("frame");                // TLS magazine pop
/// unsafe { PacketPool::free(f) };                             // TLS magazine push
/// ```
///
/// `configure`/`over` build the backing [`FramePool`] once and leak it as
/// `&'static` (system-backed, never through the global allocator). `alloc`/`free`
/// hit the calling thread's magazine; a miss/overflow refills/drains a batch from
/// the pool's per-CPU rseq slab.
#[macro_export]
macro_rules! frame_pool {
    ($v:vis $name:ident) => { $crate::frame_pool!($v $name, meta = ()); };
    ($v:vis $name:ident, meta = $meta:ty) => {
        $v struct $name;

        const _: () = {
            type Pool = $crate::FramePool<$meta>;
            static POOL: ::core::sync::atomic::AtomicPtr<Pool> =
                ::core::sync::atomic::AtomicPtr::new(::core::ptr::null_mut());

            fn pool() -> ::core::option::Option<&'static Pool> {
                let p = POOL.load(::core::sync::atomic::Ordering::Acquire);
                if p.is_null() { ::core::option::Option::None }
                // SAFETY: once non-null, POOL points at a leaked 'static Pool.
                else { ::core::option::Option::Some(unsafe { &*p }) }
            }

            fn install(built: Pool) -> ::core::result::Result<(), $crate::FramePoolError> {
                // Leak via the SYSTEM allocator, never the global one (this is
                // init-time and must not recurse if toccata is the allocator).
                let leaked: &'static Pool = $crate::SysBox::leak($crate::sys_box(built));
                let ptr = leaked as *const Pool as *mut Pool;
                // First configure wins; a racing second leaks its pool (negligible).
                let _ = POOL.compare_exchange(
                    ::core::ptr::null_mut(), ptr,
                    ::core::sync::atomic::Ordering::AcqRel,
                    ::core::sync::atomic::Ordering::Acquire,
                );
                ::core::result::Result::Ok(())
            }

            // Per-thread magazine, stored via the native-TLS pattern: a Drop-free
            // `*mut` cell (no atexit malloc) + calloc'd storage + a pthread_key
            // dtor that flushes on thread exit. One key per pool type.
            #[cfg(target_os = "linux")]
            mod tls {
                use ::core::cell::Cell;
                type Pool = $crate::FramePool<$meta>;
                type Cache = $crate::FrameCache<'static, $meta>;
                ::std::thread_local! {
                    static PTR: Cell<*mut Cache> = const { Cell::new(::core::ptr::null_mut()) };
                }
                static KEY_ONCE: ::std::sync::Once = ::std::sync::Once::new();
                static mut KEY: ::libc::pthread_key_t = 0;
                unsafe extern "C" fn on_exit(p: *mut ::libc::c_void) {
                    if p.is_null() { return; }
                    let c = p as *mut Cache;
                    ::core::ptr::drop_in_place(c); // FrameCache::drop flushes to L2
                    ::libc::free(p);
                }
                fn key() -> ::libc::pthread_key_t {
                    KEY_ONCE.call_once(|| unsafe {
                        let mut k: ::libc::pthread_key_t = 0;
                        if ::libc::pthread_key_create(&mut k, Some(on_exit)) == 0 { KEY = k; }
                    });
                    unsafe { KEY }
                }
                #[inline]
                pub fn with<R>(pool: &'static Pool, f: impl FnOnce(&mut Cache) -> R) -> R {
                    let p = PTR.with(|c| c.get());
                    let c = if p.is_null() { init(pool) } else { p };
                    if c.is_null() {
                        let mut tmp = pool.cache(); // calloc failed: uncached
                        return f(&mut tmp);
                    }
                    // SAFETY: `c` is this thread's live, initialized cache.
                    f(unsafe { &mut *c })
                }
                #[cold]
                fn init(pool: &'static Pool) -> *mut Cache {
                    let mem = unsafe {
                        ::libc::calloc(1, ::core::mem::size_of::<Cache>().max(1))
                    } as *mut Cache;
                    if mem.is_null() { return ::core::ptr::null_mut(); }
                    unsafe {
                        ::core::ptr::write(mem, pool.cache());
                        PTR.with(|c| c.set(mem));
                        let _ = ::libc::pthread_setspecific(key(), mem as *const ::libc::c_void);
                    }
                    mem
                }
            }
            #[cfg(not(target_os = "linux"))]
            mod tls {
                type Pool = $crate::FramePool<$meta>;
                type Cache = $crate::FrameCache<'static, $meta>;
                ::std::thread_local! {
                    static CACHE: ::core::cell::RefCell<::core::option::Option<Cache>> =
                        const { ::core::cell::RefCell::new(::core::option::Option::None) };
                }
                #[inline]
                pub fn with<R>(pool: &'static Pool, f: impl FnOnce(&mut Cache) -> R) -> R {
                    CACHE.with(|c| {
                        let mut b = c.borrow_mut();
                        if b.is_none() { *b = ::core::option::Option::Some(pool.cache()); }
                        f(b.as_mut().unwrap())
                    })
                }
            }

            // A given pool typically uses one constructor (configure OR
            // configure_over) and a subset of the accessors; the rest are public
            // API, not dead.
            #[allow(dead_code)]
            impl $name {
                /// Build + own the backing pool: `n_frames` frames of `frame_size`
                /// bytes each (reserve+populate+lock once per `opts`), then publish
                /// it. Idempotent: a second call is a no-op. (A `typed_frame_pool!`
                /// adds a `configure(n, opts)` that computes the size from the item
                /// type and forwards here.)
                $v fn configure_sized(
                    n_frames: usize,
                    frame_size: usize,
                    opts: $crate::ReserveOpts,
                ) -> ::core::result::Result<(), $crate::FramePoolError> {
                    if pool().is_some() { return ::core::result::Result::Ok(()); }
                    install(Pool::with_reservation(n_frames, frame_size, opts)?)
                }

                /// Publish a pool over a **caller-supplied** region (e.g. an
                /// AF_XDP UMEM). See [`FramePool::over`] for safety.
                ///
                /// # Safety
                /// As [`FramePool::over`].
                $v unsafe fn configure_over(
                    base: ::core::ptr::NonNull<u8>,
                    len: usize,
                    frame_size: usize,
                ) -> ::core::result::Result<(), $crate::FramePoolError> {
                    if pool().is_some() { return ::core::result::Result::Ok(()); }
                    install(Pool::over(base, len, frame_size)?)
                }

                /// The backing pool, or `None` before `configure`. Use for
                /// `meta`/`into_addr`/`from_addr`/`frame_index`.
                #[inline]
                $v fn pool() -> ::core::option::Option<&'static Pool> { pool() }

                /// Allocate a frame from this thread's magazine. `None` before
                /// `configure` or when the pool is exhausted.
                #[inline]
                $v fn alloc() -> ::core::option::Option<$crate::Frame> {
                    tls::with(pool()?, |c| c.alloc())
                }

                /// Release one reference to a frame (magazine push; reclaim per the
                /// metadata policy).
                ///
                /// # Safety
                /// `f` came from this pool and the caller holds a reference being
                /// released (no double-free).
                #[inline]
                $v unsafe fn free(f: $crate::Frame) {
                    if let ::core::option::Option::Some(p) = pool() {
                        tls::with(p, |c| c.free(f));
                    }
                }

                /// Return a frame unconditionally (no `on_free`); the typed-handle
                /// reclaim path. See [`FramePool::recycle`].
                ///
                /// # Safety
                /// As [`FramePool::recycle`].
                #[inline]
                $v unsafe fn recycle(f: $crate::Frame) {
                    if let ::core::option::Option::Some(p) = pool() {
                        tls::with(p, |c| c.recycle(f));
                    }
                }
            }
        };
    };
}

/// Internal: implement [`FrameSource`](crate::FrameSource) for a marker type
/// `$name` previously declared by [`frame_pool!`](crate)`(.., meta = RefCount)`.
/// Used by [`typed_frame_pool!`](crate) and [`buf_frame_pool!`](crate); not part
/// of the public API.
#[doc(hidden)]
#[macro_export]
macro_rules! __frame_source_impl {
    ($name:ident) => {
        const _: () = {
            // SAFETY: the generated pool's frames carry a per-frame RefCount (set
            // to 1 by alloc); alloc/recycle/pool operate on this pool's frames; the
            // leaked 'static pool outlives every handle.
            unsafe impl $crate::FrameSource for $name {
                #[inline]
                fn alloc_frame() -> ::core::option::Option<$crate::Frame> {
                    $name::alloc()
                }
                #[inline]
                unsafe fn recycle_frame(f: $crate::Frame) {
                    $name::recycle(f)
                }
                #[inline]
                unsafe fn refcount(f: $crate::Frame) -> &'static $crate::RefCount {
                    let p = $name::pool().expect("frame pool used before configure");
                    // Extend to 'static: the leaked pool outlives all handles.
                    ::core::mem::transmute::<&$crate::RefCount, &'static $crate::RefCount>(
                        p.meta(&f),
                    )
                }
                #[inline]
                fn frame_capacity() -> usize {
                    $name::pool().map_or(0, |p| p.frame_size())
                }
            }
        };
    };
}

/// Declare a **typed** frame pool: a named pool whose frames each hold one value
/// of type `$item`, plus `Owned`/`Shared` handle aliases bound to it. This is the
/// `Pool<T>` / object-recycler replacement — every frame is exactly `size_of::<$item>()`
/// (no size-class rounding), the refcount is out of band, and the handles get the
/// per-thread native-TLS magazine.
///
/// ```ignore
/// toccata_core::typed_frame_pool!(pub Packets, Packet);  // defines Packets,
///                                                         // Packets::Owned, ::Shared
/// Packets::configure(1 << 20, ReserveOpts::new(0))?;       // n_frames; size is implied
/// let p: Packets::Owned = Packets::owned(Packet::default()).unwrap();
/// let s: Packets::Shared = p.into_shared();
/// let s2 = s.clone();                                      // refcount bump
/// // last drop runs Packet::drop and returns the frame
/// ```
#[macro_export]
macro_rules! typed_frame_pool {
    ($v:vis $name:ident, $item:ty) => {
        $crate::frame_pool!($v $name, meta = $crate::RefCount);

        $crate::__frame_source_impl!($name);
        const _: () = {
            // SAFETY: `configure` sizes frames to hold one `$item`; the FrameSource
            // impl operates on this pool's frames with the per-frame RefCount.
            unsafe impl $crate::TypedPool for $name {
                type Item = $item;
            }
        };

        #[allow(dead_code)]
        impl $name {
            /// Allocate a unique [`Owned`](crate::Owned) handle holding `value`.
            #[inline]
            $v fn owned(value: $item) -> ::core::option::Option<$crate::Owned<$name>> {
                $crate::Owned::<$name>::new(value)
            }
            /// Allocate a shared [`Shared`](crate::Shared) handle holding `value`.
            #[inline]
            $v fn shared(value: $item) -> ::core::option::Option<$crate::Shared<$name>> {
                $crate::Shared::<$name>::new(value)
            }
        }

        #[allow(dead_code)]
        impl $name {
            /// Configure the typed pool: `n_frames` frames each sized to hold one
            /// `$item`. Frame size is the value size rounded up to a power of two
            /// (>= 8). Reserve+populate+lock per `opts`, once.
            $v fn configure(
                n_frames: usize,
                opts: $crate::ReserveOpts,
            ) -> ::core::result::Result<(), $crate::FramePoolError> {
                let sz = ::core::mem::size_of::<$item>().max(8).next_power_of_two();
                // Disambiguate from the inherent `frame_pool!` configure(n, size, opts)
                // by calling it with the computed size.
                <$name>::configure_sized(n_frames, sz, opts)
            }
        }
    };
}

/// Declare a **byte-buffer** frame pool: a named pool of fixed-size byte frames
/// with the split-mutable [`FrameMut`]/[`FrameBuf`] handles (the `BytesMut`
/// `Unfilled`/`Filled` model). Each frame is `frame_size` bytes; the
/// refcount is out of band so a `FrameBuf` can `split_to` disjoint mutable windows
/// that share one frame.
///
/// ```ignore
/// toccata_core::buf_frame_pool!(pub Packets);
/// Packets::configure(1 << 20, 2048, ReserveOpts::new(0))?;     // n_frames, frame_size
/// let mut w = Packets::mut_buf().unwrap();                     // FrameMut (writer)
/// let n = recv_into(w.bytes_mut());                            // fill it
/// let mut buf = w.freeze(n as u16);                            // FrameBuf (filled)
/// let head = buf.split_to(20);                                 // disjoint windows
/// // head + buf each payload_mut() their own region; frame frees on last drop
/// ```
#[macro_export]
macro_rules! buf_frame_pool {
    ($v:vis $name:ident) => {
        $crate::frame_pool!($v $name, meta = $crate::RefCount);
        $crate::__frame_source_impl!($name);

        #[allow(dead_code)]
        impl $name {
            /// Configure: `n_frames` byte frames of `frame_size` bytes each
            /// (reserve+populate+lock per `opts`), once. `frame_size` must be a
            /// power of two.
            $v fn configure(
                n_frames: usize,
                frame_size: usize,
                opts: $crate::ReserveOpts,
            ) -> ::core::result::Result<(), $crate::FramePoolError> {
                <$name>::configure_sized(n_frames, frame_size, opts)
            }

            /// Allocate a fresh writable buffer ([`FrameMut`](crate::FrameMut)),
            /// or `None` if the pool is exhausted.
            #[inline]
            $v fn mut_buf() -> ::core::option::Option<$crate::FrameMut<$name>> {
                $crate::FrameMut::<$name>::new()
            }
        }
    };
}

fn default_num_cpus() -> u32 {
    #[cfg(target_os = "linux")]
    {
        let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_CONF) };
        if n > 0 {
            return n as u32;
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests;
