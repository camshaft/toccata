// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Linux `membarrier(2)` syscall wrappers.

use super::MembarrierCaps;

// From include/uapi/linux/membarrier.h (verified against kernel v6.6):
//   PRIVATE_EXPEDITED               = 1 << 3
//   REGISTER_PRIVATE_EXPEDITED      = 1 << 4
//   PRIVATE_EXPEDITED_RSEQ          = 1 << 7
//   REGISTER_PRIVATE_EXPEDITED_RSEQ = 1 << 8   (NOT 1<<6 — that's SYNC_CORE!)
//   FLAG_CPU                        = 1 << 0
const CMD_REGISTER_PRIVATE_EXPEDITED: libc::c_int = 1 << 4;
const CMD_PRIVATE_EXPEDITED: libc::c_int = 1 << 3;
const CMD_PRIVATE_EXPEDITED_RSEQ: libc::c_int = 1 << 7;
const CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ: libc::c_int = 1 << 8;
const CMD_FLAG_CPU: libc::c_int = 1 << 0;

#[inline]
fn membarrier(cmd: libc::c_int, flags: libc::c_int, cpu: libc::c_int) -> i64 {
    unsafe { libc::syscall(libc::SYS_membarrier, cmd, flags, cpu) }
}

/// Register for the membarrier commands toccata's supervisor needs. Must be
/// called once (per process) before any `expedited_*` call. Returns the
/// capabilities actually obtained; missing capabilities force the supervisor
/// into a fallback (affinity-shuffle) or no-steal mode rather than using the
/// insufficient bare `PRIVATE_EXPEDITED`.
pub fn register() -> MembarrierCaps {
    MembarrierCaps {
        private_expedited: membarrier(CMD_REGISTER_PRIVATE_EXPEDITED, 0, 0) == 0,
        private_expedited_rseq: membarrier(CMD_REGISTER_PRIVATE_EXPEDITED_RSEQ, 0, 0) == 0,
    }
}

/// Global memory-ordering fence across the process's threads (orders stores;
/// does NOT abort rseq sections). Used to publish writer stores before the
/// supervisor reads stolen state.
#[inline]
pub fn expedited() -> bool {
    membarrier(CMD_PRIVATE_EXPEDITED, 0, 0) == 0
}

/// Abort any in-flight restartable sequence on `cpu` AND fence its memory. This
/// is what makes a multi-class slab seize sound. Requires the RSEQ variant.
#[inline]
pub fn expedited_rseq_cpu(cpu: u32) -> bool {
    membarrier(CMD_PRIVATE_EXPEDITED_RSEQ, CMD_FLAG_CPU, cpu as libc::c_int) == 0
}
