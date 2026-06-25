// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! `membarrier(2)` wrappers for the supervisor's slab-seize protocol.
//!
//! The fast path pays zero atomics; the supervisor instead uses membarrier to
//! (a) order writers' stores and (b) — with the RSEQ variant — abort any
//! in-flight restartable section on the target CPU so it can safely mutate that
//! CPU's slab. The plain `PRIVATE_EXPEDITED` command only *orders* memory; it
//! does NOT abort rseq sections, so a multi-class slab seize requires the
//! stronger `PRIVATE_EXPEDITED_RSEQ` (bit 7) + `CMD_FLAG_CPU` (bit 0), verified
//! against tcmalloc `internal/percpu.cc:350-354`.
//!
//! The platform-agnostic [`MembarrierCaps`] lives here; the OS-specific syscall
//! wrappers are in a per-platform submodule ([`linux`]) and re-exported, with a
//! stub for everything else.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{expedited, expedited_rseq_cpu, register};

/// Capabilities detected once at startup.
#[derive(Clone, Copy, Debug, Default)]
pub struct MembarrierCaps {
    pub private_expedited: bool,
    pub private_expedited_rseq: bool,
}

// ---- Non-Linux stubs ----
#[cfg(not(target_os = "linux"))]
pub fn register() -> MembarrierCaps {
    MembarrierCaps::default()
}
#[cfg(not(target_os = "linux"))]
pub fn expedited() -> bool {
    false
}
#[cfg(not(target_os = "linux"))]
pub fn expedited_rseq_cpu(_cpu: u32) -> bool {
    false
}
