// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Linux rseq registration and per-thread area lookup.
//!
//! Common path (glibc >= 2.35 owns rseq): reuse glibc's auto-registered area,
//! located via `__rseq_offset` + the thread pointer — allocation-free, so it's
//! safe to call from inside the global allocator. Fallback (older glibc):
//! self-register via `SYS_rseq` per thread, stored in pthread-local storage, and
//! unregister on thread death.

use super::{Rseq, CPU_UNREGISTERED, RSEQ_SIG};
use core::sync::atomic::{AtomicI64, Ordering};
use std::ptr::NonNull;

/// Owns a self-registered `Rseq` (old-glibc fallback) and unregisters it on
/// thread death. Stored in pthread-local storage, so the `Box` here is the only
/// allocation — done off the hot path during the cold fallback init.
struct RseqStorage {
    slot: crate::SysBox<Rseq>,
    registered: bool,
}

impl RseqStorage {
    /// Register a fresh rseq area for the calling thread (the `init` for the
    /// pthread-local). On failure, the area's `cpu_id_start` is poisoned so the
    /// fast path bails to the lock-free global fallback. The `Rseq` box uses the
    /// SYSTEM allocator, never toccata (this runs inside the allocator).
    fn register() -> Self {
        let mut storage = RseqStorage {
            slot: crate::sys_box(Rseq::zeroed()),
            registered: false,
        };
        let ptr = &raw mut *storage.slot;
        match sys_rseq(ptr, 0) {
            Ok(()) => storage.registered = true,
            Err(_) => unsafe { (*ptr).cpu_id_start = CPU_UNREGISTERED },
        }
        storage
    }

    #[inline]
    fn ptr(&self) -> NonNull<Rseq> {
        NonNull::new((&raw const *self.slot) as *mut Rseq).unwrap()
    }
}

impl Drop for RseqStorage {
    fn drop(&mut self) {
        let addr = self.ptr();
        if !self.registered {
            return;
        }
        // Unregister before the Box frees, so the kernel doesn't later write
        // into freed (possibly reused) memory.
        let _ = sys_rseq(addr.as_ptr(), 1 /* RSEQ_FLAGS_UNREGISTER */);
    }
}

/// Process-global cache of the glibc `__rseq_offset` (thread-pointer-relative
/// location of glibc's auto-registered rseq area). Resolved once. `i64::MIN` =
/// not-yet-resolved; `i64::MAX` = glibc rseq unavailable (use self-register
/// fallback). Any other value is the offset.
static RSEQ_OFFSET: AtomicI64 = AtomicI64::new(i64::MIN);

/// Returns this thread's `Rseq` area. **Allocation-free on the common path**: it
/// computes `thread_pointer() + __rseq_offset` directly from a register and a
/// process-global cached offset — no `thread_local!`, so it's safe to call from
/// inside the global allocator (no `__tls_get_addr` malloc, no dtor
/// registration). Only the old-glibc self-register fallback uses pthread TLS.
#[inline]
pub fn rseq() -> NonNull<Rseq> {
    let off = RSEQ_OFFSET.load(Ordering::Relaxed);
    if off != i64::MIN && off != i64::MAX {
        // Common path: glibc owns rseq; compute the area directly. No alloc.
        let p = thread_pointer()
            .wrapping_offset(off as isize)
            .cast::<Rseq>();
        // SAFETY: glibc guarantees a registered rseq area at this offset for
        // every thread once __rseq_offset is published.
        return unsafe { NonNull::new_unchecked(p) };
    }
    rseq_resolve(off)
}

#[cold]
fn rseq_resolve(off: i64) -> NonNull<Rseq> {
    if off == i64::MIN {
        // First call in the process: try to locate glibc's __rseq_offset.
        match libc_rseq_offset() {
            Some(o) if o != i64::MIN && o != i64::MAX => {
                RSEQ_OFFSET.store(o, Ordering::Relaxed);
                let p = thread_pointer().wrapping_offset(o as isize).cast::<Rseq>();
                return unsafe { NonNull::new_unchecked(p) };
            }
            _ => RSEQ_OFFSET.store(i64::MAX, Ordering::Relaxed),
        }
    }
    // glibc rseq unavailable: self-register per thread via pthread-local storage.
    self_registered_rseq()
}

/// glibc-owned rseq area: read `__rseq_offset` (a plain `ptrdiff_t` symbol).
fn libc_rseq_offset() -> Option<i64> {
    let _size = dlsym(c"__rseq_size").ok()?;
    let offset = dlsym(c"__rseq_offset").ok()?.cast::<libc::ptrdiff_t>();
    let _flags = dlsym(c"__rseq_flags").ok()?;
    Some(unsafe { offset.read() } as i64)
}

/// Old-glibc fallback: register an rseq area per thread, stored in pthread-local
/// storage (allocation via libc, not toccata) so it's safe on the alloc path
/// and gets unregistered on thread exit.
#[cold]
fn self_registered_rseq() -> NonNull<Rseq> {
    use crate::sys::pthread_local::PthreadLocal;
    static SELF_RSEQ: PthreadLocal<RseqStorage> = PthreadLocal::new(RseqStorage::register);
    SELF_RSEQ.with(|s| s.ptr())
}

fn dlsym(symbol: &std::ffi::CStr) -> std::io::Result<*mut core::ffi::c_void> {
    unsafe {
        let _ = libc::dlerror();
        let addr = libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr());
        if let Some(err) = NonNull::new(libc::dlerror()) {
            let msg = std::ffi::CStr::from_ptr(err.as_ptr())
                .to_string_lossy()
                .into_owned();
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, msg));
        }
        Ok(addr)
    }
}

#[inline]
fn thread_pointer() -> *mut core::ffi::c_void {
    let tp: *mut core::ffi::c_void;
    unsafe {
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!("mov {}, fs:0", out(reg) tp, options(nostack, pure, readonly));
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!("mrs {}, tpidr_el0", out(reg) tp, options(nostack, pure, nomem));
    }
    tp
}

fn sys_rseq(ptr: *mut Rseq, flags: i32) -> std::io::Result<()> {
    let ret = unsafe {
        libc::syscall(
            libc::SYS_rseq,
            ptr,
            core::mem::size_of::<Rseq>() as u32,
            flags,
            RSEQ_SIG,
        )
    };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Read the current CPU id (committed field). Returns `None` if rseq is
/// unavailable/unregistered on this thread.
#[inline]
pub fn current_cpu() -> Option<u32> {
    let ptr = rseq();
    let id = unsafe { (*ptr.as_ptr()).cpu_id_start };
    (id != CPU_UNREGISTERED).then_some(id)
}
