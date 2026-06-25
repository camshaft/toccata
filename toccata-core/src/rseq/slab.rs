// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Per-CPU pointer-stack slab.
//!
//! Each logical CPU owns a block of per-size-class stacks of *object pointers*
//! (8-byte slots). `push` (free) and `pop` (alloc) operate on the stack for the
//! CPU the calling thread is currently running on. Two fast-path implementations
//! share this geometry:
//!
//! * **Baseline (`CpuStack::*_locked`)** — a per-(CPU,class) lightweight lock,
//!   uncontended in the common case (only same-CPU preemption contends, and the
//!   critical region is a few instructions that never sleep). This is exactly
//!   the librseq mempool model (`refs/librseq/src/rseq-mempool.c:1065,1172` take
//!   a `pthread_mutex` per pool). Always correct, portable, fully testable on
//!   macOS and Linux. toccata ships on this first.
//!
//! * **RSEQ fast path** — replaces the lock with a restartable sequence whose
//!   single committing store needs no atomic. Compiled in automatically on Linux
//!   x86_64/aarch64 (falls back to the locked path at runtime if the kernel lacks
//!   rseq). This is the optimization the whole design is built around. See
//!   [`crate::rseq::asm`].
//!
//! The geometry (offsets per class) is computed by the owner (toccata-core) and
//! passed in; this module is geometry + synchronization only.

use crate::rseq::abi;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU32, Ordering};

/// Per-class location within a CPU block, in bytes from the block base.
#[derive(Clone, Copy, Debug)]
pub struct ClassLoc {
    /// Offset of this class's [`Header`].
    pub header_off: u32,
    /// Offset of slot 0 of this class's pointer array.
    pub slots_off: u32,
    /// Lightweight lock guarding this class's stack on this CPU (baseline path).
    /// One per (CPU, class); see [`SlabLayout::lock`].
    pub lock_off: u32,
}

/// Packed per-class stack header: `current` live count (top of stack),
/// `capacity` ceiling. A single store of `current` commits a push/pop. Kept to 8
/// bytes so the rseq fast path can commit it in one word-store.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub current: u32,
    pub capacity: u32,
}

/// Shared, immutable slab geometry. All CPUs index the same description.
pub struct SlabLayout {
    base: NonNull<u8>,
    num_cpus: u32,
    shift: u32,
    classes: &'static [ClassLoc],
    /// Per-CPU "stopped" flags for the supervisor seize protocol.
    /// When `stopped[cpu] != 0`, that CPU's slab is being mutated by the
    /// supervisor; a writer that finds it set on its slow path must not touch the
    /// slab (it routes to the central list / remote queue instead). Boxed and
    /// leaked for a `'static` lifetime; one cache-line-padded flag per CPU.
    stopped: &'static [StopFlag],
}

/// log2 of `size_of::<StopFlag>()` (128-byte aligned) — the stride the rseq asm
/// uses to index `stopped[cpu]`. Must match `StopFlag`'s alignment/size.
pub const STOP_SHIFT: u32 = 7;

/// Cache-line-padded stop flag.
#[repr(C, align(128))]
pub struct StopFlag(AtomicU32);

impl StopFlag {
    const fn new() -> Self {
        Self(AtomicU32::new(0))
    }
}

// SAFETY: SlabLayout is a read-only description of a region whose interior
// mutability is mediated by per-CPU locks / rseq. It is Send+Sync by design.
unsafe impl Send for SlabLayout {}
unsafe impl Sync for SlabLayout {}

impl SlabLayout {
    /// # Safety
    /// `base` must point at a live region of at least `num_cpus << shift` bytes,
    /// carved per `classes`, and outlive this `SlabLayout`.
    pub unsafe fn new(
        base: NonNull<u8>,
        num_cpus: u32,
        shift: u32,
        classes: &'static [ClassLoc],
    ) -> Self {
        // Allocate the per-CPU stop flags via the SYSTEM allocator, never the
        // global one — this runs at configure-time and must not route through
        // toccata (which may be the global allocator). Leaked for 'static.
        let mut v: crate::SysVec<StopFlag> = allocator_api2::vec::Vec::new_in(crate::Sys);
        v.reserve_exact(num_cpus.max(1) as usize);
        for _ in 0..num_cpus.max(1) {
            v.push(StopFlag::new());
        }
        let stopped: &'static [StopFlag] = allocator_api2::vec::Vec::leak(v);
        Self { base, num_cpus, shift, classes, stopped }
    }

    /// Set/clear a CPU's stop flag (supervisor only). `seq_cst` so it orders with
    /// the membarrier fence the supervisor issues around it.
    #[inline]
    pub fn set_stopped(&self, cpu: u32, stopped: bool) {
        self.stopped[(cpu as usize).min(self.stopped.len() - 1)]
            .0
            .store(stopped as u32, Ordering::SeqCst);
    }

    #[inline]
    fn is_stopped(&self, cpu: u32) -> bool {
        self.stopped[(cpu as usize).min(self.stopped.len() - 1)].0.load(Ordering::Acquire) != 0
    }

    /// Direct access to a class header for the supervisor's seize-time mutation.
    /// # Safety
    /// Caller must hold the seize (StopCpu flag set + rseq-abort membarrier
    /// issued) so no writer is concurrently in that CPU's rseq section.
    #[inline]
    pub unsafe fn header_mut(&self, cpu: u32, class: usize) -> *mut Header {
        self.header(cpu, class)
    }

    /// Direct access to slot `i` of a class on a CPU (supervisor seize-time only).
    /// # Safety
    /// As [`SlabLayout::header_mut`].
    #[inline]
    pub unsafe fn slot_ptr(&self, cpu: u32, class: usize, i: u32) -> *mut *mut u8 {
        self.slots(cpu, class).add(i as usize)
    }

    #[inline]
    pub fn num_cpus(&self) -> u32 {
        self.num_cpus
    }
    #[inline]
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    #[inline]
    fn block(&self, cpu: u32) -> *mut u8 {
        debug_assert!(cpu < self.num_cpus);
        // SAFETY: cpu bounds-checked by callers via current_cpu()/clamp.
        unsafe { self.base.as_ptr().add((cpu as usize) << self.shift) }
    }

    #[inline]
    fn header(&self, cpu: u32, class: usize) -> *mut Header {
        unsafe { self.block(cpu).add(self.classes[class].header_off as usize) as *mut Header }
    }

    #[inline]
    fn slots(&self, cpu: u32, class: usize) -> *mut *mut u8 {
        unsafe { self.block(cpu).add(self.classes[class].slots_off as usize) as *mut *mut u8 }
    }

    #[inline]
    fn lock(&self, cpu: u32, class: usize) -> &AtomicU32 {
        unsafe { &*(self.block(cpu).add(self.classes[class].lock_off as usize) as *const AtomicU32) }
    }
}

/// Outcome of a fast-path attempt.
pub enum Fast<T> {
    /// Committed with this value.
    Ok(T),
    /// Stack empty (pop) or full (push); caller must refill/drain via the slow path.
    NeedsSlow,
}

/// A handle to one CPU's view of the slab. Obtained via [`CpuStack::current`],
/// which reads the running CPU. The `cpu` is a snapshot; the locked path
/// re-validates nothing (the lock makes same-CPU races safe), while the rseq
/// path re-checks the committed CPU inside its section.
pub struct CpuStack<'a> {
    layout: &'a SlabLayout,
    cpu: u32,
}

impl<'a> CpuStack<'a> {
    /// Bind to the CPU the calling thread is on. Falls back to CPU 0 clamp when
    /// rseq is unavailable (still correct under the lock; just less local).
    #[inline]
    pub fn current(layout: &'a SlabLayout) -> Self {
        let cpu = match abi::current_cpu() {
            Some(c) if c < layout.num_cpus => c,
            // Unregistered or out of range: hash the thread onto a shard so load
            // still spreads. Correctness holds because the lock guards the stack.
            _ => fallback_shard(layout.num_cpus),
        };
        Self { layout, cpu }
    }

    /// Bind without an upfront `current_cpu()` read. When the rseq fast path is
    /// compiled in, the asm reads the CPU itself and returns it via `pop_on`/
    /// `push_on`, so the upfront read is pure overhead — we defer to a sentinel
    /// and only resolve a real CPU if we fall to the locked path. This shaves a
    /// TLS read off every fast-path alloc/free.
    #[inline]
    pub fn current_fast(layout: &'a SlabLayout) -> Self {
        #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // cpu is resolved by the asm; use an out-of-range sentinel that the
            // locked fallback re-resolves if ever reached.
            return Self { layout, cpu: u32::MAX };
        }
        #[allow(unreachable_code)]
        Self::current(layout)
    }

    /// Resolve the snapshot CPU, computing it lazily if `current_fast` deferred.
    #[inline]
    fn resolved_cpu(&self) -> u32 {
        if self.cpu == u32::MAX {
            match abi::current_cpu() {
                Some(c) if c < self.layout.num_cpus => c,
                _ => fallback_shard(self.layout.num_cpus),
            }
        } else {
            self.cpu
        }
    }

    /// The bound CPU, resolving the deferred sentinel from `current_fast` if
    /// needed. Used by slow paths (refill/drain/accounting) where a real CPU id
    /// is required.
    #[inline]
    pub fn cpu(&self) -> u32 {
        self.resolved_cpu()
    }

    /// Pop one pointer from this CPU's stack for `class` (baseline locked path).
    #[inline]
    pub fn pop_locked(&self, class: usize) -> Fast<NonNull<u8>> {
        let cpu = self.resolved_cpu();
        // If the supervisor has seized this CPU's slab, don't touch it — let the
        // caller fall through to the central list (seize protocol).
        if self.layout.is_stopped(cpu) {
            return Fast::NeedsSlow;
        }
        let lock = self.layout.lock(cpu, class);
        let _guard = SpinGuard::acquire(lock);
        let hdr = self.layout.header(cpu, class);
        // SAFETY: lock held; single owner of this (cpu,class) stack.
        unsafe {
            let cur = (*hdr).current;
            if cur == 0 {
                return Fast::NeedsSlow;
            }
            let slot = self.layout.slots(cpu, class).add((cur - 1) as usize);
            let obj = *slot;
            (*hdr).current = cur - 1;
            match NonNull::new(obj) {
                Some(p) => Fast::Ok(p),
                // A null in a live slot is a bug; treat as empty rather than UB.
                None => Fast::NeedsSlow,
            }
        }
    }

    /// Push one pointer onto this CPU's stack for `class` (baseline locked path).
    /// Returns `NeedsSlow` if the stack is at capacity (caller drains a batch).
    #[inline]
    pub fn push_locked(&self, class: usize, obj: NonNull<u8>) -> Fast<()> {
        let cpu = self.resolved_cpu();
        if self.layout.is_stopped(cpu) {
            return Fast::NeedsSlow;
        }
        let lock = self.layout.lock(cpu, class);
        let _guard = SpinGuard::acquire(lock);
        let hdr = self.layout.header(cpu, class);
        unsafe {
            let cur = (*hdr).current;
            let cap = (*hdr).capacity;
            if cur >= cap {
                return Fast::NeedsSlow;
            }
            let slot = self.layout.slots(cpu, class).add(cur as usize);
            *slot = obj.as_ptr();
            (*hdr).current = cur + 1;
            Fast::Ok(())
        }
    }

    /// Pop one pointer for `class`, using the atomic-free RSEQ fast path on Linux
    /// x86_64/aarch64 and falling back to the locked path on abort/unavailability.
    /// This is the public hot-path entry.
    #[inline]
    pub fn pop(&self, class: usize) -> Fast<NonNull<u8>> {
        #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let loc = self.layout.classes[class];
            // SAFETY: layout describes a live region; loc offsets are in-bounds.
            let r = unsafe {
                crate::rseq::asm::pop(
                    self.layout.base.as_ptr(),
                    self.layout.shift,
                    self.layout.num_cpus,
                    loc.header_off,
                    loc.slots_off,
                    self.layout.stopped.as_ptr() as *const u8,
                    STOP_SHIFT,
                )
            };
            match r {
                crate::rseq::asm::RseqResult::Ok(p, _cpu) => {
                    return match NonNull::new(p) {
                        Some(nn) => Fast::Ok(nn),
                        None => Fast::NeedsSlow,
                    };
                }
                crate::rseq::asm::RseqResult::NeedsSlow => return Fast::NeedsSlow,
                // Fallback: the rseq path bailed (migration storm / unregistered).
                // Use the locked path on the CPU the rseq layer last saw.
                crate::rseq::asm::RseqResult::Fallback => {}
            }
        }
        self.pop_locked(class)
    }

    /// Like [`CpuStack::pop`] but also returns the CPU the op committed on (or
    /// the snapshot CPU on the locked path), so the caller can do per-CPU
    /// accounting without a separate `current_cpu()` read. This is the hot-path
    /// entry the allocator uses.
    #[inline]
    pub fn pop_on(&self, class: usize) -> (Fast<NonNull<u8>>, u32) {
        #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let loc = self.layout.classes[class];
            let r = unsafe {
                crate::rseq::asm::pop(
                    self.layout.base.as_ptr(),
                    self.layout.shift,
                    self.layout.num_cpus,
                    loc.header_off,
                    loc.slots_off,
                    self.layout.stopped.as_ptr() as *const u8,
                    STOP_SHIFT,
                )
            };
            match r {
                crate::rseq::asm::RseqResult::Ok(p, cpu) => {
                    return match NonNull::new(p) {
                        Some(nn) => (Fast::Ok(nn), cpu),
                        None => (Fast::NeedsSlow, cpu),
                    };
                }
                crate::rseq::asm::RseqResult::NeedsSlow => return (Fast::NeedsSlow, self.resolved_cpu()),
                crate::rseq::asm::RseqResult::Fallback => {}
            }
        }
        (self.pop_locked(class), self.resolved_cpu())
    }

    /// Push one pointer for `class`, RSEQ fast path when compiled in.
    #[inline]
    pub fn push(&self, class: usize, obj: NonNull<u8>) -> Fast<()> {
        #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let loc = self.layout.classes[class];
            // SAFETY: as `pop`.
            let r = unsafe {
                crate::rseq::asm::push(
                    self.layout.base.as_ptr(),
                    self.layout.shift,
                    self.layout.num_cpus,
                    loc.header_off,
                    loc.slots_off,
                    obj.as_ptr(),
                    self.layout.stopped.as_ptr() as *const u8,
                    STOP_SHIFT,
                )
            };
            match r {
                crate::rseq::asm::RseqResult::Ok(_, _) => return Fast::Ok(()),
                crate::rseq::asm::RseqResult::NeedsSlow => return Fast::NeedsSlow,
                crate::rseq::asm::RseqResult::Fallback => {}
            }
        }
        self.push_locked(class, obj)
    }

    /// Like [`CpuStack::push`] but also returns the committed CPU for accounting.
    #[inline]
    pub fn push_on(&self, class: usize, obj: NonNull<u8>) -> (Fast<()>, u32) {
        #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let loc = self.layout.classes[class];
            let r = unsafe {
                crate::rseq::asm::push(
                    self.layout.base.as_ptr(),
                    self.layout.shift,
                    self.layout.num_cpus,
                    loc.header_off,
                    loc.slots_off,
                    obj.as_ptr(),
                    self.layout.stopped.as_ptr() as *const u8,
                    STOP_SHIFT,
                )
            };
            match r {
                crate::rseq::asm::RseqResult::Ok(_, cpu) => return (Fast::Ok(()), cpu),
                crate::rseq::asm::RseqResult::NeedsSlow => return (Fast::NeedsSlow, self.resolved_cpu()),
                crate::rseq::asm::RseqResult::Fallback => {}
            }
        }
        (self.push_locked(class, obj), self.resolved_cpu())
    }
}

/// Hash the current thread onto a shard when rseq can't give a CPU id. Cheap,
/// deterministic per thread, spreads load across blocks.
#[inline]
fn fallback_shard(num_cpus: u32) -> u32 {
    // Thread id via address of a thread-local; stable per thread.
    thread_local! { static ANCHOR: u8 = const { 0 }; }
    let addr = ANCHOR.with(|a| a as *const u8 as usize);
    ((addr >> 6) as u32) % num_cpus.max(1)
}

/// A minimal test-and-set spin lock over an `AtomicU32` embedded in the slab.
/// Uncontended in the common case (per-CPU); the held region is a few
/// instructions and never sleeps/syscalls. 0 = free, 1 = held.
struct SpinGuard<'a>(&'a AtomicU32);

impl<'a> SpinGuard<'a> {
    #[inline]
    fn acquire(lock: &'a AtomicU32) -> Self {
        // Fast path: single CAS. Contended path: spin with a hint. Bounded in
        // practice by the few-instruction critical region.
        while lock
            .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            while lock.load(Ordering::Relaxed) != 0 {
                core::hint::spin_loop();
            }
        }
        Self(lock)
    }
}

impl Drop for SpinGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.store(0, Ordering::Release);
    }
}
