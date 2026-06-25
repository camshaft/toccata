// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! The real north-star proof: install toccata as the process `#[global_allocator]`
//! and run actual std collections through it. Every `Box`/`Vec`/`String`/`HashMap`
//! allocation in this test binary is served by toccata.
//!
//! Linux-only: it exercises the production native-TLS hot path + the mlock'd
//! reservation. (The non-Linux dev fallback uses a `RefCell<Tls>` that is not
//! re-entrant under a panic, so this whole-binary global-allocator test only runs
//! on Linux.) Budget kept modest so it fits under RLIMIT_MEMLOCK when the harness
//! raises it.

#![cfg(target_os = "linux")]

use toccata_core::{OnExhaust, SubHeapBuilder};

#[global_allocator]
static GLOBAL: toccata::Toccata = toccata::Toccata::new();

#[test]
fn std_collections_run_on_toccata() {
    // Configure the global heap once. Use few shards to keep the mlock small.
    toccata::configure_with(
        SubHeapBuilder::new("global", 64 * 1024 * 1024)
            .num_cpus(4)
            .on_exhaust(OnExhaust::Abort),
    );

    // Vec growth.
    let mut v: Vec<u64> = Vec::new();
    for i in 0..100_000u64 {
        v.push(i);
    }
    assert_eq!(v.iter().sum::<u64>(), (0..100_000u64).sum());

    // Box.
    let b = Box::new([42u8; 4096]);
    assert_eq!(b[0], 42);
    assert_eq!(b[4095], 42);

    // String.
    let mut s = String::new();
    for _ in 0..10_000 {
        s.push_str("toccata");
    }
    assert_eq!(s.len(), 70_000); // "toccata" is 7 bytes × 10_000

    // HashMap (BTreeMap-free; exercises many small allocations + frees).
    let mut m = std::collections::HashMap::new();
    for i in 0..50_000u64 {
        m.insert(i, i * 2);
    }
    assert_eq!(m.len(), 50_000);
    assert_eq!(m[&12345], 24690);

    // Churn: drop everything, allocate again — exercises free + reuse.
    drop(v);
    drop(m);
    drop(s);
    let mut v2: Vec<String> = Vec::new();
    for i in 0..10_000 {
        v2.push(format!("item-{i}"));
    }
    assert_eq!(v2.len(), 10_000);
    assert_eq!(v2[42], "item-42");
}

#[test]
fn nested_and_cross_thread_allocations() {
    // configure() is idempotent; safe if the other test ran first.
    toccata::configure_with(
        SubHeapBuilder::new("global", 64 * 1024 * 1024)
            .num_cpus(4)
            .on_exhaust(OnExhaust::Abort),
    );

    // Allocate on one thread, free on another (the networking pattern).
    let handles: Vec<_> = (0..8)
        .map(|t| {
            std::thread::spawn(move || {
                let data: Vec<Vec<u8>> = (0..1000).map(|i| vec![t as u8; (i % 512) + 1]).collect();
                data // moved out, dropped by the joiner thread
            })
        })
        .collect();
    let mut total = 0usize;
    for h in handles {
        let data = h.join().unwrap();
        total += data.iter().map(|d| d.len()).sum::<usize>();
        // `data` dropped here, on this thread — cross-thread free.
    }
    assert!(total > 0);
}
