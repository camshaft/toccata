// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Allocation-free thread-local storage for use *inside* the allocator.
//!
//! Rust's `thread_local!` is unsafe to touch on the allocation hot path:
//!
//! * In the dynamic-TLS model (cdylib / cross-DSO), the first access on a thread
//!   calls `__tls_get_addr`, which lazily `malloc`s the TLS block — re-entering
//!   the allocator. (In a fully-static `local-exec` binary it happens to be
//!   allocation-free, which is why naive `thread_local!` *seems* to work — until
//!   toccata is built as a shared object.)
//! * A `thread_local!` whose type is `Drop` registers a destructor via
//!   `__cxa_thread_atexit` on first access, which also allocates.
//!
//! [`PthreadLocal`] sidesteps both:
//! access is a raw `pthread_getspecific` (never allocates); the per-thread
//! storage is one `posix_memalign` (libc's heap, *not* toccata, correctly
//! aligned for `T`) done on first use; and the destructor is registered once via
//! `pthread_key_create`. So the allocator can safely keep per-thread caches
//! without recursing into itself.

//! All Unix: pthread keys are POSIX, so this works on Linux, macOS, the BSDs.
#![cfg(unix)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

/// A thread-local `T`, safe to access from within the global allocator.
///
/// `T` must be initializable from a `fn() -> T` and is dropped (via the
/// registered pthread destructor) when the thread exits.
pub struct PthreadLocal<T: 'static> {
    key: UnsafeCell<libc::pthread_key_t>,
    once: Once,
    init: fn() -> T,
    /// Set if key creation failed; access then returns the fallback (see `with`).
    failed: AtomicBool,
}

// SAFETY: the key is created once under `Once`; per-thread storage is private to
// each thread. `T: Send` because each thread owns its own instance.
unsafe impl<T: 'static + Send> Sync for PthreadLocal<T> {}

unsafe extern "C" fn dtor<T: 'static>(ptr: *mut libc::c_void) {
    if ptr.is_null() {
        return;
    }
    // Run T's destructor, then free the libc-allocated storage.
    let typed = ptr as *mut T;
    core::ptr::drop_in_place(typed);
    libc::free(ptr);
}

impl<T: 'static> PthreadLocal<T> {
    pub const fn new(init: fn() -> T) -> Self {
        Self {
            key: UnsafeCell::new(0),
            once: Once::new(),
            init,
            failed: AtomicBool::new(false),
        }
    }

    #[inline]
    fn key(&self) -> Option<libc::pthread_key_t> {
        self.once.call_once(|| {
            let mut key: libc::pthread_key_t = 0;
            let rc = unsafe { libc::pthread_key_create(&mut key, Some(dtor::<T>)) };
            if rc != 0 {
                self.failed.store(true, Ordering::Release);
            } else {
                // SAFETY: written once, under Once, before any reader observes
                // `failed == false`.
                unsafe { *self.key.get() = key };
            }
        });
        if self.failed.load(Ordering::Acquire) {
            None
        } else {
            Some(unsafe { *self.key.get() })
        }
    }

    /// Run `f` with this thread's `T`, creating it (one `posix_memalign`) on
    /// first access. The access itself (`pthread_getspecific`) never allocates.
    ///
    /// **Aborts** (never unwinds — this runs on the allocator's own path, where
    /// unwinding is UB and the panic machinery itself allocates) if thread-local
    /// storage cannot be established: either `pthread_key_create` failed (e.g.
    /// `PTHREAD_KEYS_MAX` exhausted) or the per-thread allocation failed. Failing
    /// hard is deliberate — a transient fallback `T` would dangle for any `f`
    /// that returns a borrow of it (e.g. a raw pointer into the slot, exactly
    /// what the rseq self-register consumer does).
    ///
    /// Callers on the alloc path must ensure `init` and `f` themselves don't
    /// recurse into the allocator.
    #[inline]
    pub fn with<R>(&'static self, f: impl FnOnce(&mut T) -> R) -> R {
        let Some(key) = self.key() else {
            abort_no_tls();
        };

        // Fast path: pthread_getspecific is allocation-free.
        let existing = unsafe { libc::pthread_getspecific(key) } as *mut T;
        let ptr = if existing.is_null() {
            self.init_slot(key)
        } else {
            existing
        };
        if ptr.is_null() {
            abort_no_tls();
        }
        // SAFETY: `ptr` is this thread's live, initialized `T`.
        f(unsafe { &mut *ptr })
    }

    #[cold]
    fn init_slot(&self, key: libc::pthread_key_t) -> *mut T {
        // Allocate storage from libc's heap (NOT toccata) so this never recurses
        // into the allocator. `posix_memalign` (not `calloc`) so the block is
        // correctly aligned even when `align_of::<T>()` exceeds `max_align_t`;
        // `ptr::write` fully initializes it, so no zeroing is needed. Freed with
        // plain `libc::free` in `dtor`, which is valid for `posix_memalign` blocks.
        let align = core::mem::align_of::<T>().max(core::mem::size_of::<*mut libc::c_void>());
        let size = core::mem::size_of::<T>().max(1);
        let mut mem: *mut libc::c_void = core::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut mem, align, size) };
        if rc != 0 || mem.is_null() {
            return core::ptr::null_mut();
        }
        let mem = mem as *mut T;
        unsafe {
            core::ptr::write(mem, (self.init)());
            // Register for destruction on thread exit; failure just means no dtor.
            let _ = libc::pthread_setspecific(key, mem as *const libc::c_void);
        }
        mem
    }
}

/// No thread-local storage could be established. We must not fabricate a
/// transient `T` (its address would dangle once `with` returns, and the rseq
/// consumer returns exactly such a pointer), and we must not unwind (this runs
/// on the allocator path). So abort with a raw diagnostic — no formatting
/// machinery, which would allocate.
#[cold]
#[inline(never)]
fn abort_no_tls() -> ! {
    // SAFETY: write(2) to stderr is async-signal-safe and allocation-free.
    unsafe {
        let msg = b"toccata-sys: thread-local storage unavailable (pthread key/alloc failed); aborting\n";
        libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
    }
    std::process::abort();
}
