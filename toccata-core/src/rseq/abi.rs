// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! The kernel `struct rseq` ABI and per-thread registration.
//!
//! This file holds the platform-agnostic ABI types and constants; the
//! OS-specific registration/lookup lives in a per-platform submodule
//! ([`linux`]). The public surface ([`rseq`], [`current_cpu`]) is re-exported
//! from the active platform module, with a stub for everything else.
//!
//! Linux details (in [`linux`]): on a glibc that owns rseq registration
//! (>= 2.35) we *reuse* the kernel area glibc already registered, located via
//! `__rseq_offset` plus the thread pointer; otherwise we self-register via
//! `SYS_rseq` per thread and unregister on thread death.

#![allow(clippy::missing_safety_doc)]

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{current_cpu, rseq};

/// The kernel `struct rseq`. `repr(C, align(32))` matches the kernel ABI.
///
/// Only the first four fields are read on the hot path. The real kernel struct
/// is larger (it has `node_id`, `mm_cid`, padding) but a registration declares
/// its own size; we register exactly this size when self-registering, and when
/// reusing glibc's area we only *read* these offsets, never assume the size.
#[repr(C, align(32))]
pub struct Rseq {
    /// Optimistic CPU hint, always a valid CPU id once registered.
    pub cpu_id_start: u32,
    /// The committed CPU id; `u32::MAX`-ish sentinel until first update or on
    /// registration failure. The fast path double-checks this against
    /// `cpu_id_start` (tcmalloc's two-field trick).
    pub cpu_id: u32,
    /// Pointer to the active `struct rseq_cs` critical-section descriptor.
    /// Set by user code to arm a section; cleared by the kernel on abort.
    pub rseq_cs: u64,
    /// Per-thread flags (the historical inhibit flags are deprecated).
    pub flags: u32,
}

impl Rseq {
    pub const fn zeroed() -> Self {
        Self { cpu_id_start: 0, cpu_id: 0, rseq_cs: 0, flags: 0 }
    }
}

/// The glibc-ABI signature constant, per architecture. This is fixed by glibc;
/// the kernel checks the 4 bytes immediately before the abort landing pad match
/// it, which prevents abort-driven control-flow hijacking.
#[cfg(target_arch = "x86_64")]
pub const RSEQ_SIG: u32 = 0x53053053;
#[cfg(target_arch = "aarch64")]
pub const RSEQ_SIG: u32 = 0xd428bc00;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const RSEQ_SIG: u32 = 0;

/// Sentinel written into `cpu_id_start` when registration fails, so the fast
/// path's `cpu_id_start >= num_cpus` bounds check bails to the fallback.
pub const CPU_UNREGISTERED: u32 = u32::MAX;

// ---- Non-Linux stub (toccata is a Linux allocator; this keeps it building on
// macOS for development, always reporting "unavailable"). ----

#[cfg(not(target_os = "linux"))]
#[inline]
pub fn current_cpu() -> Option<u32> {
    None
}
