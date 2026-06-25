// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Unit tests exercising the `GlobalAlloc` impl directly (not installed as the
//! process allocator — that's covered by an integration binary). We avoid
//! calling the global `configure()` here (it builds a process-wide pool);
//! instead we test the pre-configure (System-delegating) path and trait logic.

use super::*;

#[test]
fn pre_configure_delegates_to_system() {
    // Before configure(), POOL_LEN is 0, so alloc delegates to the system
    // allocator. The returned pointer is usable and is NOT in toccata's pool.
    let a = Toccata;
    let layout = Layout::from_size_align(64, 8).unwrap();
    let p = unsafe { a.alloc(layout) };
    assert!(!p.is_null());
    assert!(!in_pool(p), "pre-configure alloc must come from System, not the pool");
    unsafe {
        std::ptr::write_bytes(p, 0xEE, 64);
        assert_eq!(*p, 0xEE);
        // dealloc routes by range -> System; must round-trip without crashing.
        a.dealloc(p, layout);
    }
}

#[test]
fn zero_sized_alloc_is_dangling_aligned() {
    let a = Toccata;
    let layout = Layout::from_size_align(0, 16).unwrap();
    let p = unsafe { a.alloc(layout) };
    assert_eq!(p as usize, 16, "ZST alloc returns align as dangling pointer");
    unsafe { a.dealloc(p, layout) }; // no-op, must not crash
}

#[test]
fn pre_configure_alloc_is_aligned() {
    let a = Toccata;
    for &align in &[1usize, 8, 16, 64, 4096] {
        let layout = Layout::from_size_align(align.max(1), align).unwrap();
        let p = unsafe { a.alloc(layout) };
        assert!(!p.is_null());
        assert_eq!(p as usize % align, 0, "System-delegated alloc must honor align {align}");
        unsafe { a.dealloc(p, layout) };
    }
}

// The no-alloc decimal formatter now lives in `toccata_core::sys::diag` and is
// tested there (`diag::tests::dec_formats`); the duplicate test that lived here
// was removed when the helper was lifted down into toccata-core.
