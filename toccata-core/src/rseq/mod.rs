// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Restartable-sequence per-CPU primitives — Layer 0 of toccata.
//!
//! * [`abi`] — the kernel `struct rseq` ABI, per-thread registration (reusing
//!   glibc's area when present), and [`abi::current_cpu`].
//! * [`membarrier`] — `membarrier(2)` wrappers, including the RSEQ-variant the
//!   supervisor needs to seize a live per-CPU slab.
//! * [`slab`] — the per-CPU pointer-stack slab with a portable, always-correct
//!   locked baseline fast path. The atomic-free RSEQ fast path ([`asm`]) is
//!   compiled in **automatically** on every Linux x86_64/aarch64 build (no
//!   feature, no `--cfg`). If the running kernel lacks rseq, registration fails
//!   and the path falls back to the locked baseline at runtime — so it is always
//!   safe to compile in.
//!
//! `abi`/`membarrier`/`slab` compile on every platform — `subheap` and the
//! `supervisor` need them — and `asm` adds the inline-asm fast path on the
//! supported targets only.
//!
//! toccata targets Linux. On other platforms (used for development) the rseq
//! and membarrier calls degrade to "unavailable", and the slab's locked baseline
//! remains fully correct.

pub mod abi;
#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod asm;
pub mod membarrier;
pub mod slab;

#[cfg(test)]
mod tests;

pub use abi::{current_cpu, Rseq};
pub use membarrier::MembarrierCaps;
pub use slab::{ClassLoc, CpuStack, Fast, Header, SlabLayout};

/// Whether the atomic-free RSEQ fast path is compiled in: automatic on Linux
/// x86_64/aarch64. When false (other platforms), the locked baseline is used.
/// Note: even when true, an old kernel without rseq still falls back at runtime.
pub const RSEQ_FASTPATH: bool =
    cfg!(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")));
