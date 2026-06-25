// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! `toccata-core` — the toccata allocator's reusable primitive surface.
//!
//! This is the crate a library depends on when it wants
//! toccata's building blocks without imposing a process-global allocator. The
//! application-facing `#[global_allocator]` lives in the separate `toccata`
//! crate, which re-exports this surface.
//!
//! Layers, low to high:
//! * [`sys`] — the OS-memory [`Reservation`] primitive + the system-allocator
//!   bypass (`Sys`/`SysBox`/`pthread_local`/`diag`) so internals never recurse.
//! * [`rseq`] — restartable-sequence per-CPU slab + membarrier (asm fast path
//!   behind the `rseq` feature).
//! * [`sizeclass`], [`meta`], [`magazine`], [`subheap`] — size classes, span
//!   metadata, the L1 thread-cache magazine, and the independently-budgeted
//!   [`SubHeap`].
//! * [`frame`] — fixed-size frame pools (`FramePool`), with typed
//!   `Owned`/`Shared` handles and split-mutable `FrameMut`/`FrameBuf` byte buffers.
//! * [`supervisor`] (feature) — background reclaim/rebalance.
//! * [`metrics`] (feature) — per-`SubHeap` live-usage query.
//!
//! The never-stall guarantee is structural — reserve+populate+lock up front,
//! then never issue a growth syscall against the region — not a global flag.

pub mod frame;
pub mod magazine;
pub mod meta;
pub mod registry;
pub mod rseq;
pub mod sizeclass;
pub mod spanpool;
pub mod subheap;
pub mod sys;

// NOTE: the old typed `pool` module (`Pool<T>`/`Owned`/`Shared`) has been removed
// — superseded by `frame`'s `FramePool` + `typed_frame_pool!` (`Owned`/`Shared`)
// and `buf_frame_pool!` (`FrameMut`/`FrameBuf`), which are faster (magazine-fronted,
// out-of-band refcount) and support split-mutable byte buffers.

#[cfg(feature = "metrics")]
pub mod metrics;
#[cfg(feature = "supervisor")]
pub mod supervisor;

// Re-exported so the `register_subheap!` macro can refer to `$crate::linkme`.
pub use linkme;

pub use magazine::{ThreadCache, MAG_CAP};
// System-allocator primitives (the `sys` module), re-exported at the crate root
// so internal `crate::Sys` / `crate::SysBox` / `crate::sys_box` / `crate::SysVec`
// / `crate::SysBoxSlice` / `crate::sys_boxed_slice` shortcuts resolve, and so
// downstream crates can name them as `toccata_core::Sys`, etc.
pub use sys::{pthread_local, sys_box, sys_boxed_slice, Sys, SysBox, SysBoxSlice, SysVec};

pub use frame::{
    Frame, FrameBuf, FrameCache, FrameMeta, FrameMut, FramePool, FramePoolError, FrameSource,
    Owned, Reclaim, RefCount, Region, Shared, TypedPool, FRAME_MAG_CAP,
};
pub use registry::{configure, subheap, Budgets, ConfigureError, SubHeapRegistration};
pub use subheap::{OnExhaust, SubHeap, SubHeapBuilder};
pub use sys::reserve::{Backing, HugePages, Require, Reservation, ReserveError, ReserveOpts};

#[cfg(test)]
mod tests;
