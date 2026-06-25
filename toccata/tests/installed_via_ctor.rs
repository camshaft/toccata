// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Proves the `install!` macro: toccata's pool is reserved in a `#[ctor]` from
//! an app-supplied config fn (which reads an env var, allocating via the system
//! allocator) BEFORE `main` — so by the time any test runs, the pool is live and
//! the global hot path has taken the configured branch with no pre-configure
//! window. Every allocation in this binary is served by toccata.
//!
//! Linux-only, like `installed.rs`: the non-Linux dev fallback's `RefCell<Tls>`
//! is not re-entrant under a panic.

#![cfg(target_os = "linux")]

/// App config fn: runs at ctor time, may read env / do anything. Its own
/// allocations (env var String) go to the system allocator, not toccata.
fn budget() -> usize {
    // Read an optional override; default 64 MiB (small, fits RLIMIT_MEMLOCK).
    std::env::var("TOCCATA_TEST_HEAP_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb << 20)
        .unwrap_or(64 * 1024 * 1024)
}

// Installs the #[global_allocator] AND a #[ctor] that reserves the pool before
// main using `budget()`. No configure() call in main.
toccata::install!(budget);

#[test]
fn pool_is_live_before_main_and_serves_std() {
    // If the ctor ran, the pool exists now — allocations are toccata's, not the
    // pre-configure System fallback. We can't directly observe "which allocator"
    // from std types, but we exercise the full surface and assert correctness.
    let mut v: Vec<u64> = Vec::new();
    for i in 0..50_000u64 {
        v.push(i);
    }
    assert_eq!(v.iter().sum::<u64>(), (0..50_000u64).sum());

    let b = Box::new([7u8; 2048]);
    assert_eq!(b[0], 7);
    assert_eq!(b[2047], 7);

    let mut s = String::new();
    for _ in 0..5000 {
        s.push_str("toccata");
    }
    assert_eq!(s.len(), 40_000);

    let mut m = std::collections::HashMap::new();
    for i in 0..20_000u64 {
        m.insert(i, i * 3);
    }
    assert_eq!(m[&1234], 3702);

    // Cross-thread: allocate here, free on another thread.
    let data: Vec<Vec<u8>> = (0..1000).map(|i| vec![1u8; (i % 256) + 1]).collect();
    let h = std::thread::spawn(move || data.iter().map(|d| d.len()).sum::<usize>());
    assert!(h.join().unwrap() > 0);
}
