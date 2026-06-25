// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! System-allocator primitives + the OS-memory [`Reservation`].
//!
//! Two concerns live here, both below the rest of the allocator:
//!
//! * **System-allocator bypass** — let toccata's internals allocate **without
//!   ever touching the `#[global_allocator]`** (which is toccata itself, so
//!   routing internal allocations through it would recurse):
//!   * [`Sys`] — an `allocator-api2::Allocator` that calls the libc/system
//!     allocator directly. Use it as the `A` parameter on `allocator-api2`'s
//!     `Box`/`Vec`/`HashMap` for *every* internal toccata data structure (the
//!     boxed `SubHeap`, setup vectors, per-frame metadata arrays). Convenience
//!     aliases [`SysBox`], [`SysVec`] make the intent obvious.
//!   * [`pthread_local::PthreadLocal`] — allocation-free thread-local storage
//!     (raw `pthread_getspecific`, storage via `libc::calloc`), safe to touch
//!     from inside the allocator. All-Unix.
//!   * [`diag`] — allocation-free raw-`write(2)` diagnostics, for paths that run
//!     inside the allocator and must not allocate.
//! * **The OS-memory [`reserve`]ation** — [`Reservation`] / [`ReserveOpts`]: the
//!   up-front `mmap(MAP_POPULATE)` + `mlock2` region that backs every pool. A
//!   pure primitive: it knows nothing about any process-global "sealed" state.

pub mod diag;
pub mod pthread_local;
pub mod reserve;

pub use reserve::{
    memlock_limit, Backing, HugePages, Require, Reservation, ReserveError, ReserveOpts,
};

use allocator_api2::alloc::{AllocError, Allocator};
use core::alloc::{GlobalAlloc, Layout};
use core::ptr::NonNull;

/// An `allocator-api2::Allocator` that delegates to the process's *system*
/// allocator (`std::alloc::System` = libc `malloc`/`free`), bypassing whatever
/// `#[global_allocator]` is installed. Zero-sized.
#[derive(Clone, Copy, Default, Debug)]
pub struct Sys;

// SAFETY: delegates to std::alloc::System, which satisfies the Allocator
// contract; the handle is a ZST so cloning never invalidates blocks.
unsafe impl Allocator for Sys {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(
                NonNull::new(layout.align() as *mut u8).ok_or(AllocError)?,
                0,
            ));
        }
        // SAFETY: non-zero layout.
        let p = unsafe { std::alloc::System.alloc(layout) };
        let p = NonNull::new(p).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(p, layout.size()))
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return self.allocate(layout);
        }
        let p = unsafe { std::alloc::System.alloc_zeroed(layout) };
        let p = NonNull::new(p).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(p, layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            std::alloc::System.dealloc(ptr.as_ptr(), layout);
        }
    }
}

/// A `Box` whose allocation is explicitly system-backed (never toccata).
pub type SysBox<T> = allocator_api2::boxed::Box<T, Sys>;
/// A `Vec` whose allocation is explicitly system-backed (never toccata).
pub type SysVec<T> = allocator_api2::vec::Vec<T, Sys>;
/// A boxed slice whose allocation is explicitly system-backed (never toccata).
/// Use for fixed-size per-sub-heap metadata arrays built at configure time.
pub type SysBoxSlice<T> = allocator_api2::boxed::Box<[T], Sys>;

/// Collect `n` items produced by `f(i)` into a system-backed boxed slice. The
/// allocation goes to the system allocator, never recursing through toccata.
#[inline]
pub fn sys_boxed_slice<T>(n: usize, mut f: impl FnMut(usize) -> T) -> SysBoxSlice<T> {
    let mut v: SysVec<T> = allocator_api2::vec::Vec::with_capacity_in(n, Sys);
    for i in 0..n {
        v.push(f(i));
    }
    v.into_boxed_slice()
}

/// Allocate a `SysBox<T>` (system-backed). Infallible like `Box::new`; aborts on
/// system OOM via `allocator-api2` (acceptable: this is toccata's own
/// init-time bookkeeping, not the app's fallible path).
#[inline]
pub fn sys_box<T>(value: T) -> SysBox<T> {
    allocator_api2::boxed::Box::new_in(value, Sys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sys_box_roundtrips() {
        let b = sys_box(12345u64);
        assert_eq!(*b, 12345);
    }

    #[test]
    fn sys_vec_grows() {
        let mut v: SysVec<u32> = allocator_api2::vec::Vec::new_in(Sys);
        for i in 0..1000 {
            v.push(i);
        }
        assert_eq!(v.len(), 1000);
        assert_eq!(v.iter().sum::<u32>(), (0..1000u32).sum());
    }

    #[cfg(unix)]
    #[test]
    fn pthread_local_caches_per_thread() {
        static TL: pthread_local::PthreadLocal<u64> = pthread_local::PthreadLocal::new(|| 7);
        TL.with(|v| {
            assert_eq!(*v, 7);
            *v = 42;
        });
        TL.with(|v| assert_eq!(*v, 42)); // cached on this thread
        std::thread::spawn(|| {
            TL.with(|v| assert_eq!(*v, 7)); // fresh on another thread
        })
        .join()
        .unwrap();
    }
}
