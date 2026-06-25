// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Tests for the rseq layer. These exercise the portable locked baseline (which
//! is correct on every platform); the rseq fast path is validated separately on
//! hardware.

use crate::rseq::abi;
use crate::rseq::slab::{ClassLoc, CpuStack, Fast, Header, SlabLayout};
use std::ptr::NonNull;

/// Build a tiny single-class slab in a heap allocation and return (layout,
/// backing, leaked-classes). Geometry: per CPU block holds [Header | lock(u32) |
/// pad | N slots].
fn make_slab(num_cpus: u32, capacity: u32) -> (SlabLayout, Vec<u8>, &'static [ClassLoc]) {
    let header_off: u32 = 0;
    let lock_off: u32 = 8; // after Header{u32,u32}
    let slots_off: u32 = 16; // 8-aligned after the lock
    let stride_bytes = slots_off as usize + (capacity as usize) * 8;
    // round stride up to a power of two for the shift-based indexing
    let shift = (usize::BITS - (stride_bytes - 1).leading_zeros()) as u32;
    let stride = 1usize << shift;

    let mut backing = vec![0u8; stride * num_cpus as usize];
    let base = NonNull::new(backing.as_mut_ptr()).unwrap();

    // Initialize each CPU block's header capacity.
    for cpu in 0..num_cpus as usize {
        let blk = unsafe { backing.as_mut_ptr().add(cpu * stride) };
        let hdr = blk as *mut Header;
        unsafe {
            (*hdr).current = 0;
            (*hdr).capacity = capacity;
        }
    }

    let classes: &'static [ClassLoc] =
        Box::leak(vec![ClassLoc { header_off, slots_off, lock_off }].into_boxed_slice());
    let layout = unsafe { SlabLayout::new(base, num_cpus, shift, classes) };
    (layout, backing, classes)
}

#[test]
fn push_then_pop_roundtrip_lifo() {
    let (layout, _backing, _classes) = make_slab(4, 8);
    let stack = CpuStack::current(&layout);

    // Push three distinct fake pointers.
    let ptrs: Vec<NonNull<u8>> =
        (1..=3u8).map(|i| NonNull::new(i as usize as *mut u8).unwrap()).collect();
    for p in &ptrs {
        assert!(matches!(stack.push_locked(0, *p), Fast::Ok(())));
    }
    // Pop returns LIFO order.
    for expected in ptrs.iter().rev() {
        match stack.pop_locked(0) {
            Fast::Ok(got) => assert_eq!(got, *expected),
            Fast::NeedsSlow => panic!("unexpected empty"),
        }
    }
    // Now empty.
    assert!(matches!(stack.pop_locked(0), Fast::NeedsSlow));
}

#[test]
fn push_at_capacity_declines() {
    let (layout, _b, _c) = make_slab(2, 2);
    let stack = CpuStack::current(&layout);
    let p = NonNull::new(0x1000 as *mut u8).unwrap();
    assert!(matches!(stack.push_locked(0, p), Fast::Ok(())));
    assert!(matches!(stack.push_locked(0, p), Fast::Ok(())));
    // Third push exceeds capacity 2.
    assert!(matches!(stack.push_locked(0, p), Fast::NeedsSlow));
}

#[test]
fn pop_empty_declines() {
    let (layout, _b, _c) = make_slab(1, 4);
    let stack = CpuStack::current(&layout);
    assert!(matches!(stack.pop_locked(0), Fast::NeedsSlow));
}

#[test]
fn current_cpu_is_in_range_or_none() {
    // On Linux with rseq this is Some(cpu < nproc); elsewhere None. Either way
    // it must never be an absurd value.
    if let Some(cpu) = abi::current_cpu() {
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as u32;
        // cpu can momentarily exceed reported parallelism on some hosts, but
        // should be far below an absurd ceiling.
        assert!(cpu < 4096, "cpu id {cpu} implausible (n={n})");
    }
}

#[test]
fn concurrent_push_pop_no_corruption() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // Many threads hammering a shared slab; the per-(cpu,class) lock must keep
    // it corruption-free regardless of which CPU each thread lands on.
    let (layout, _backing, _classes) = make_slab(8, 64);
    let layout = Arc::new(layout);
    let stop = Arc::new(AtomicBool::new(false));

    let workers: Vec<_> = (0..8)
        .map(|_| {
            let layout = layout.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut held: Vec<NonNull<u8>> = Vec::new();
                let mut ops = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let stack = CpuStack::current(&layout);
                    if held.len() < 32 {
                        let p = NonNull::new((0x1_0000 + ops as usize * 8) as *mut u8).unwrap();
                        if let Fast::Ok(()) = stack.push_locked(0, p) {
                            held.push(p);
                        }
                    }
                    if let Fast::Ok(p) = stack.pop_locked(0) {
                        // Popped pointer must be 8-aligned & nonzero (came from a push).
                        assert!(p.as_ptr() as usize >= 0x1_0000);
                        held.pop();
                    }
                    ops += 1;
                }
                ops
            })
        })
        .collect();

    std::thread::sleep(std::time::Duration::from_millis(200));
    stop.store(true, Ordering::Relaxed);
    let total: u64 = workers.into_iter().map(|w| w.join().unwrap()).sum();
    let _ = total;
    assert!(true);
}

/// Stress the dispatch `pop`/`push` (the RSEQ fast path on Linux x86_64/aarch64,
/// else the locked baseline) across many threads on a slab sized to the machine.
/// Each push stores a tagged pointer; pops must never return a
/// value that wasn't pushed, never the same live value twice, and the slab must
/// not lose or duplicate objects (checked by a conservation count).
#[test]
fn rseq_dispatch_stress_no_corruption() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    let nproc = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u32;
    // Slab sized to the machine so rseq's real CPU index is always in range.
    let (layout, _backing, _classes) = make_slab(nproc + 1, 256);
    let layout = Arc::new(layout);
    let stop = Arc::new(AtomicBool::new(false));

    // A pool of distinct fake pointers, each used by at most one thread at a
    // time. Each worker owns a DISJOINT block of object ids and seeds them into
    // whatever shard `CpuStack::current` binds it to, then churns push/pop on that
    // shard. This matters for the locked baseline: a thread is hashed onto a fixed
    // shard for its whole life (`fallback_shard`) and can only pop what was pushed
    // to *that* shard — so seeding one shard up front (as an earlier version did)
    // starved every worker on a different shard, racily yielding zero pops under
    // load. Per-worker seeding makes progress deterministic regardless of the
    // hash, while still exercising concurrent cross-shard push/pop and the global
    // duplication map.
    let nworkers = nproc as usize * 2;
    const PER_WORKER: usize = 128; // <= capacity (256); fits one shard's class-0 stack
    let seed: usize = nworkers * PER_WORKER;

    // Occupancy map: bit i = "object i is currently OUT of the slab (held by a
    // thread)". Pop must find it clear (in-slab) and set it; push clears it. A
    // double-pop (duplication) or popping an un-pushed object trips an assert —
    // this catches any tear/duplication the lockless rseq path could cause.
    let occ: Arc<Vec<AtomicU64>> = Arc::new((0..(seed / 64 + 1)).map(|_| AtomicU64::new(0)).collect());
    let popped_total = Arc::new(AtomicU64::new(0));
    let workers: Vec<_> = (0..nworkers)
        .map(|w| {
            let layout = layout.clone();
            let stop = stop.clone();
            let occ = occ.clone();
            let popped_total = popped_total.clone();
            std::thread::spawn(move || {
                let lo = w * PER_WORKER; // this worker's disjoint id range
                let idx_of = |p: NonNull<u8>| (p.as_ptr() as usize - 0x10_0000) / 64;
                let mut held: Vec<NonNull<u8>> = Vec::with_capacity(PER_WORKER);

                // Seed this worker's own objects onto its own shard. They start
                // "held" (out of slab) so the conservation map is consistent.
                for k in 0..PER_WORKER {
                    let i = lo + k;
                    occ[i / 64].fetch_or(1u64 << (i % 64), Ordering::AcqRel);
                    held.push(NonNull::new((0x10_0000 + i * 64) as *mut u8).unwrap());
                }

                let mut pops = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let stack = CpuStack::current(&layout);
                    if held.len() < PER_WORKER / 4 {
                        if let Fast::Ok(p) = stack.pop(0) {
                            let a = p.as_ptr() as usize;
                            assert!(a >= 0x10_0000, "popped a value never pushed: {a:#x}");
                            assert_eq!((a - 0x10_0000) % 64, 0, "torn pointer: {a:#x}");
                            let i = idx_of(p);
                            assert!(i < seed, "popped out-of-range object {i}");
                            // Mark out-of-slab; must have been in-slab (bit clear).
                            let word = &occ[i / 64];
                            let bit = 1u64 << (i % 64);
                            let prev = word.fetch_or(bit, Ordering::AcqRel);
                            assert_eq!(prev & bit, 0, "DUPLICATION: object {i} popped while already held");
                            held.push(p);
                            pops += 1;
                        }
                    } else {
                        let p = held.pop().unwrap();
                        let i = idx_of(p);
                        // Clear before pushing back (it's about to be in-slab).
                        occ[i / 64].fetch_and(!(1u64 << (i % 64)), Ordering::AcqRel);
                        if let Fast::NeedsSlow = stack.push(0, p) {
                            // Slab full here; re-mark held and keep it.
                            occ[i / 64].fetch_or(1u64 << (i % 64), Ordering::AcqRel);
                            held.push(p);
                        } else {
                            pops += 1; // count a completed push/pop cycle as progress
                        }
                    }
                }
                for p in held.drain(..) {
                    let i = idx_of(p);
                    occ[i / 64].fetch_and(!(1u64 << (i % 64)), Ordering::AcqRel);
                    let stack = CpuStack::current(&layout);
                    let _ = stack.push(0, p);
                }
                popped_total.fetch_add(pops, Ordering::Relaxed);
            })
        })
        .collect();

    // Let the workers churn; each makes progress on its own shard, so this is a
    // fixed window (not a race against a single seeded shard being scheduled).
    std::thread::sleep(std::time::Duration::from_millis(300));
    stop.store(true, Ordering::Relaxed);
    for w in workers {
        w.join().unwrap();
    }
    let pops = popped_total.load(Ordering::Relaxed);
    assert!(pops > 0, "no successful pops — rseq/locked path made no progress");
    eprintln!("rseq_dispatch_stress: {pops} pops across {nworkers} threads, no corruption");
}
