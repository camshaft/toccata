// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! The atomic-free RSEQ fast path: per-CPU pointer-stack `pop` (alloc) and
//! `push` (free) as restartable sequences with a single committing store.
//!
//! The `pop`/`push` critical sections are modeled on tcmalloc's
//! `percpu_tcmalloc.h`. The committing instruction is always
//! a single relaxed store of the stack's `current` index, placed last.
//!
//! Slab layout (per CPU, matching `crate::rseq::slab`): block at
//! `base + (cpu << shift)`. Within a block, a class has a `Header{current:u32,
//! capacity:u32}` at `header_off` and a pointer-slot array at `slots_off`; slot
//! `i` is at `block + slots_off + i*8`.
//!
//! Compiled in automatically on Linux x86_64/aarch64; on other targets these are
//! absent and the slab uses its portable locked baseline. Even where compiled, an
//! old kernel without rseq makes registration fail and the slab falls back at
//! runtime.

#![cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]

use crate::rseq::abi::{self, Rseq, RSEQ_SIG};

/// Outcome of an rseq fast-path attempt.
pub enum RseqResult {
    /// Committed on this CPU: popped this object / pushed successfully (value
    /// carries the popped pointer for pop, or null for push). The `u32` is the
    /// CPU the commit happened on — returned so callers can do per-CPU
    /// accounting without a second `current_cpu()` read.
    Ok(*mut u8, u32),
    /// Stack empty (pop) or full (push) — caller must run the slow path.
    NeedsSlow,
    /// rseq unusable or repeatedly aborted — caller must use the locked fallback.
    Fallback,
}

/// Pop one object pointer from the current CPU's stack for the class whose
/// `Header` is at `header_off` and slots at `slots_off`, within blocks of stride
/// `1<<shift` based at `base`. `num_cpus` bounds the CPU index.
///
/// # Safety
/// The layout args must describe a live, correctly-sized per-CPU region.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn pop(
    base: *mut u8,
    shift: u32,
    num_cpus: u32,
    header_off: u32,
    slots_off: u32,
    stop_base: *const u8,
    stop_shift: u32,
) -> RseqResult {
    let rseq_ptr = abi::rseq().as_ptr();
    let obj: *mut u8;
    let mut status: u64;
    let cpu_out: u64; // 0 ok, 1 needs_slow, 2 fallback (incl. seized)
    core::arch::asm!(
        ".pushsection __rseq_cs, \"aw\"",
        ".balign 32",
        "9:",
        ".long 0",
        ".long 0",
        ".quad 2f",
        ".quad (6f-2f)",
        ".quad 7f",
        ".popsection",
        "b 7f",
        ".long {sig}",
        "7:",
        // cpu = rseq.cpu_id_start; bail to fallback if out of range
        "ldr {cpu:w}, [{rseq}, #{cpu_start_off}]",
        "cmp {cpu:w}, {ncpus:w}",
        "b.hs 92f",
        // bounded retry
        "subs {loops}, {loops}, #1",
        "b.eq 92f",
        // arm the critical section
        "adrp {tmp}, 9b",
        "add {tmp}, {tmp}, :lo12:9b",
        "str {tmp}, [{rseq}, #{cs_off}]",
        "2:", // ---- CRITICAL SECTION START ----
        // re-check committed cpu == hint
        "ldr {tmp:w}, [{rseq}, #{cpu_id_off}]",
        "cmp {cpu:w}, {tmp:w}",
        "b.ne 7b",
        // STOP-FLAG CHECK (seize protocol): if the supervisor has stopped this
        // CPU, bail to the slow path. Set before the seize membarrier, so any
        // section that survives/starts past the fence sees it. stop[cpu] at
        // stop_base + (cpu << stop_shift).
        "lsl {tmp}, {cpu}, {stop_shift}",
        "ldr {tmp2:w}, [{stop_base}, {tmp}]",
        "cbnz {tmp2:w}, 92f",
        // block = base + (cpu << shift)
        "lsl {blk}, {cpu}, {shift}",
        "add {blk}, {base}, {blk}",
        // hdr = block + header_off ; cur = hdr.current (u32 at offset 0)
        "add {hdr}, {blk}, {header_off}",
        "ldr {cur:w}, [{hdr}]",
        "cbz {cur:w}, 90f",                 // empty -> needs_slow
        "sub {cur:w}, {cur:w}, #1",
        // obj = slots[cur] where slots = block + slots_off
        "add {slots}, {blk}, {slots_off}",
        "ldr {obj}, [{slots}, {cur}, lsl #3]",
        // *** SINGLE COMMIT STORE: hdr.current = cur ***
        "str {cur:w}, [{hdr}]",
        "6:", // ---- post_commit ----
        "str xzr, [{rseq}, #{cs_off}]",
        "mov {status}, #0",
        "b 8f",
        "90:",                              // empty
        "str xzr, [{rseq}, #{cs_off}]",
        "mov {status}, #1",
        "b 8f",
        "92:",                              // fallback / seized
        "str xzr, [{rseq}, #{cs_off}]",
        "mov {status}, #2",
        "mov {obj}, xzr",
        "8:",
        rseq = in(reg) rseq_ptr,
        base = in(reg) base,
        ncpus = in(reg) num_cpus,
        shift = in(reg) shift as u64,
        header_off = in(reg) header_off as u64,
        slots_off = in(reg) slots_off as u64,
        stop_base = in(reg) stop_base,
        stop_shift = in(reg) stop_shift as u64,
        loops = inout(reg) 5u64 => _,
        cpu = out(reg) cpu_out,
        blk = out(reg) _,
        hdr = out(reg) _,
        cur = out(reg) _,
        slots = out(reg) _,
        tmp = out(reg) _,
        tmp2 = out(reg) _,
        obj = out(reg) obj,
        status = out(reg) status,
        sig = const RSEQ_SIG,
        cpu_start_off = const core::mem::offset_of!(Rseq, cpu_id_start),
        cpu_id_off = const core::mem::offset_of!(Rseq, cpu_id),
        cs_off = const core::mem::offset_of!(Rseq, rseq_cs),
        options(nostack),
    );
    match status {
        0 => RseqResult::Ok(obj, cpu_out as u32),
        1 => RseqResult::NeedsSlow,
        _ => RseqResult::Fallback,
    }
}

/// Push one object pointer onto the current CPU's stack for the class. Commits
/// by storing `current+1` after writing the slot. Returns `NeedsSlow` if full.
///
/// # Safety
/// As [`pop`]. `obj` must be a valid pointer to store.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)] // layout is passed positionally to keep the asm shim flat
pub unsafe fn push(
    base: *mut u8,
    shift: u32,
    num_cpus: u32,
    header_off: u32,
    slots_off: u32,
    obj: *mut u8,
    stop_base: *const u8,
    stop_shift: u32,
) -> RseqResult {
    let rseq_ptr = abi::rseq().as_ptr();
    let mut status: u64;
    let cpu_out: u64;
    core::arch::asm!(
        ".pushsection __rseq_cs, \"aw\"",
        ".balign 32",
        "9:",
        ".long 0",
        ".long 0",
        ".quad 2f",
        ".quad (6f-2f)",
        ".quad 7f",
        ".popsection",
        "b 7f",
        ".long {sig}",
        "7:",
        "ldr {cpu:w}, [{rseq}, #{cpu_start_off}]",
        "cmp {cpu:w}, {ncpus:w}",
        "b.hs 92f",
        "subs {loops}, {loops}, #1",
        "b.eq 92f",
        "adrp {tmp}, 9b",
        "add {tmp}, {tmp}, :lo12:9b",
        "str {tmp}, [{rseq}, #{cs_off}]",
        "2:",
        "ldr {tmp:w}, [{rseq}, #{cpu_id_off}]",
        "cmp {cpu:w}, {tmp:w}",
        "b.ne 7b",
        // STOP-FLAG CHECK (seize protocol)
        "lsl {tmp}, {cpu}, {stop_shift}",
        "ldr {cap:w}, [{stop_base}, {tmp}]",
        "cbnz {cap:w}, 92f",
        "lsl {blk}, {cpu}, {shift}",
        "add {blk}, {base}, {blk}",
        "add {hdr}, {blk}, {header_off}",
        // cur = hdr.current ; cap = hdr.capacity (u32 at offset 4)
        "ldr {cur:w}, [{hdr}]",
        "ldr {cap:w}, [{hdr}, #4]",
        "cmp {cur:w}, {cap:w}",
        "b.hs 90f",                         // full -> needs_slow
        // slots[cur] = obj  (this scratch write is harmless if aborted: it's
        // above the committed top until the commit store advances current)
        "add {slots}, {blk}, {slots_off}",
        "str {obj}, [{slots}, {cur}, lsl #3]",
        "add {cur:w}, {cur:w}, #1",
        // *** SINGLE COMMIT STORE: hdr.current = cur+1 ***
        "str {cur:w}, [{hdr}]",
        "6:",
        "str xzr, [{rseq}, #{cs_off}]",
        "mov {status}, #0",
        "b 8f",
        "90:",
        "str xzr, [{rseq}, #{cs_off}]",
        "mov {status}, #1",
        "b 8f",
        "92:",
        "str xzr, [{rseq}, #{cs_off}]",
        "mov {status}, #2",
        "8:",
        rseq = in(reg) rseq_ptr,
        base = in(reg) base,
        ncpus = in(reg) num_cpus,
        shift = in(reg) shift as u64,
        header_off = in(reg) header_off as u64,
        slots_off = in(reg) slots_off as u64,
        obj = in(reg) obj,
        stop_base = in(reg) stop_base,
        stop_shift = in(reg) stop_shift as u64,
        loops = inout(reg) 5u64 => _,
        cpu = out(reg) cpu_out,
        blk = out(reg) _,
        hdr = out(reg) _,
        cur = out(reg) _,
        cap = out(reg) _,
        slots = out(reg) _,
        tmp = out(reg) _,
        status = out(reg) status,
        sig = const RSEQ_SIG,
        cpu_start_off = const core::mem::offset_of!(Rseq, cpu_id_start),
        cpu_id_off = const core::mem::offset_of!(Rseq, cpu_id),
        cs_off = const core::mem::offset_of!(Rseq, rseq_cs),
        options(nostack),
    );
    match status {
        0 => RseqResult::Ok(core::ptr::null_mut(), cpu_out as u32),
        1 => RseqResult::NeedsSlow,
        _ => RseqResult::Fallback,
    }
}

// ---- x86_64 ----

/// As the aarch64 [`pop`]: pop one object pointer from the current CPU's stack.
///
/// # Safety
/// The layout args must describe a live, correctly-sized per-CPU region.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn pop(
    base: *mut u8,
    shift: u32,
    num_cpus: u32,
    header_off: u32,
    slots_off: u32,
    stop_base: *const u8,
    stop_shift: u32,
) -> RseqResult {
    let rseq_ptr = abi::rseq().as_ptr();
    let obj: *mut u8;
    let mut status: u64;
    let cpu_out: u64;
    core::arch::asm!(
        ".pushsection __rseq_cs, \"aw\"",
        ".balign 32",
        "99:",
        ".long 0",
        ".long 0",
        ".quad 2f",
        ".quad (6f-2f)",
        ".quad 7f",
        ".popsection",
        "jmp 7f",
        ".long {sig}",
        "7:",
        "mov {cpu:e}, [{rseq}+{cpu_start_off}]",
        "cmp {cpu:e}, {ncpus:e}",
        "jae 92f",
        "dec {loops}",
        "jz 92f",
        "lea {tmp}, [rip+99b]",
        "mov [{rseq}+{cs_off}], {tmp}",
        "2:",
        "cmp {cpu:e}, [{rseq}+{cpu_id_off}]",
        "jne 7b",
        // STOP-FLAG CHECK (seize protocol): stop[cpu] at stop_base+(cpu<<stop_shift)
        "mov {tmp}, {cpu}",
        "shlx {tmp}, {tmp}, {stop_shift}",
        "mov {cur:e}, [{stop_base}+{tmp}]",
        "test {cur:e}, {cur:e}",
        "jnz 92f",
        // block = base + (cpu << shift)
        "mov {blk}, {cpu}",
        "shlx {blk}, {blk}, {shiftreg}",
        "add {blk}, {base}",
        // cur = [blk+header_off]
        "mov {cur:e}, [{blk}+{header_off}]",
        "test {cur:e}, {cur:e}",
        "jz 90f",
        "dec {cur:e}",
        // obj = [blk + slots_off + cur*8]
        "mov {obj}, [{blk}+{slots_off}+{cur}*8]",
        // *** SINGLE COMMIT STORE ***
        "mov [{blk}+{header_off}], {cur:e}",
        "6:",
        "mov qword ptr [{rseq}+{cs_off}], 0",
        "xor {status:e}, {status:e}",
        "jmp 8f",
        "90:",
        "mov qword ptr [{rseq}+{cs_off}], 0",
        "mov {status}, 1",
        "jmp 8f",
        "92:",
        "mov qword ptr [{rseq}+{cs_off}], 0",
        "mov {status}, 2",
        "xor {obj:e}, {obj:e}",
        "8:",
        rseq = in(reg) rseq_ptr,
        base = in(reg) base,
        ncpus = in(reg) num_cpus,
        shiftreg = in(reg) shift as u64,
        header_off = in(reg) header_off as u64,
        slots_off = in(reg) slots_off as u64,
        stop_base = in(reg) stop_base,
        stop_shift = in(reg) stop_shift as u64,
        loops = inout(reg) 5u64 => _,
        cpu = out(reg) cpu_out,
        blk = out(reg) _,
        cur = out(reg) _,
        tmp = out(reg) _,
        obj = out(reg) obj,
        status = out(reg) status,
        sig = const RSEQ_SIG,
        cpu_start_off = const core::mem::offset_of!(Rseq, cpu_id_start),
        cpu_id_off = const core::mem::offset_of!(Rseq, cpu_id),
        cs_off = const core::mem::offset_of!(Rseq, rseq_cs),
        options(nostack),
    );
    match status {
        0 => RseqResult::Ok(obj, cpu_out as u32),
        1 => RseqResult::NeedsSlow,
        _ => RseqResult::Fallback,
    }
}

/// As the aarch64 [`push`]: push one object pointer onto the current CPU's stack.
///
/// # Safety
/// As [`pop`]. `obj` must be a valid pointer to store.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)] // layout is passed positionally to keep the asm shim flat
pub unsafe fn push(
    base: *mut u8,
    shift: u32,
    num_cpus: u32,
    header_off: u32,
    slots_off: u32,
    obj: *mut u8,
    stop_base: *const u8,
    stop_shift: u32,
) -> RseqResult {
    let rseq_ptr = abi::rseq().as_ptr();
    let mut status: u64;
    let cpu_out: u64;
    core::arch::asm!(
        ".pushsection __rseq_cs, \"aw\"",
        ".balign 32",
        "99:",
        ".long 0",
        ".long 0",
        ".quad 2f",
        ".quad (6f-2f)",
        ".quad 7f",
        ".popsection",
        "jmp 7f",
        ".long {sig}",
        "7:",
        "mov {cpu:e}, [{rseq}+{cpu_start_off}]",
        "cmp {cpu:e}, {ncpus:e}",
        "jae 92f",
        "dec {loops}",
        "jz 92f",
        "lea {tmp}, [rip+99b]",
        "mov [{rseq}+{cs_off}], {tmp}",
        "2:",
        "cmp {cpu:e}, [{rseq}+{cpu_id_off}]",
        "jne 7b",
        // STOP-FLAG CHECK (seize protocol)
        "mov {tmp}, {cpu}",
        "shlx {tmp}, {tmp}, {stop_shift}",
        "mov {cap:e}, [{stop_base}+{tmp}]",
        "test {cap:e}, {cap:e}",
        "jnz 92f",
        "mov {blk}, {cpu}",
        "shlx {blk}, {blk}, {shiftreg}",
        "add {blk}, {base}",
        "mov {cur:e}, [{blk}+{header_off}]",
        "mov {cap:e}, [{blk}+{header_off}+4]",
        "cmp {cur:e}, {cap:e}",
        "jae 90f",
        "mov [{blk}+{slots_off}+{cur}*8], {obj}",
        "inc {cur:e}",
        // *** SINGLE COMMIT STORE ***
        "mov [{blk}+{header_off}], {cur:e}",
        "6:",
        "mov qword ptr [{rseq}+{cs_off}], 0",
        "xor {status:e}, {status:e}",
        "jmp 8f",
        "90:",
        "mov qword ptr [{rseq}+{cs_off}], 0",
        "mov {status}, 1",
        "jmp 8f",
        "92:",
        "mov qword ptr [{rseq}+{cs_off}], 0",
        "mov {status}, 2",
        "8:",
        rseq = in(reg) rseq_ptr,
        base = in(reg) base,
        ncpus = in(reg) num_cpus,
        shiftreg = in(reg) shift as u64,
        header_off = in(reg) header_off as u64,
        slots_off = in(reg) slots_off as u64,
        obj = in(reg) obj,
        stop_base = in(reg) stop_base,
        stop_shift = in(reg) stop_shift as u64,
        loops = inout(reg) 5u64 => _,
        cpu = out(reg) cpu_out,
        blk = out(reg) _,
        cur = out(reg) _,
        cap = out(reg) _,
        tmp = out(reg) _,
        status = out(reg) status,
        sig = const RSEQ_SIG,
        cpu_start_off = const core::mem::offset_of!(Rseq, cpu_id_start),
        cpu_id_off = const core::mem::offset_of!(Rseq, cpu_id),
        cs_off = const core::mem::offset_of!(Rseq, rseq_cs),
        options(nostack),
    );
    match status {
        0 => RseqResult::Ok(core::ptr::null_mut(), cpu_out as u32),
        1 => RseqResult::NeedsSlow,
        _ => RseqResult::Fallback,
    }
}
