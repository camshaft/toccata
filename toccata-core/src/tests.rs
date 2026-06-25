// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Phase 1 correctness suite for the single-thread-correct allocator.

use crate::{
    sizeclass::{self},
    subheap::{OnExhaust, SubHeapBuilder},
};
use std::{collections::HashSet, ptr::NonNull};

/// A small sub-heap for tests. Budget kept well under the default 8 MiB
/// `RLIMIT_MEMLOCK` so the reservation succeeds without raising limits. Note the
/// reservation also needs slab metadata (~num_cpus * block) on top of the
/// budget, so we stay conservative.
fn test_subheap() -> crate::SubHeap {
    test_subheap_bytes(1024 * 1024)
}

fn test_subheap_bytes(bytes: usize) -> crate::SubHeap {
    // 4 shards + small caps keep the locked reservation tiny so many tests can
    // run in parallel under a small (8 MiB) RLIMIT_MEMLOCK default.
    SubHeapBuilder::new("test", bytes)
        .num_cpus(4)
        .cap_per_class(16)
        .build_standalone()
        .expect("reservation should succeed for a small budget under RLIMIT_MEMLOCK")
}

#[test]
fn alloc_dealloc_roundtrip() {
    let sh = test_subheap();
    let class = sizeclass::class_for(64).unwrap();
    let p = sh.alloc_class(class).expect("alloc");
    // Writable within the class size.
    unsafe {
        std::ptr::write_bytes(p.as_ptr(), 0xAB, sizeclass::size_of_class(class));
        assert_eq!(*p.as_ptr(), 0xAB);
        sh.dealloc_class(p, class);
    }
}

#[test]
fn allocations_are_distinct_and_aligned() {
    let sh = test_subheap();
    let class = sizeclass::class_for(128).unwrap();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut held = Vec::new();
    for _ in 0..1000 {
        let p = sh.alloc_class(class).expect("alloc");
        let addr = p.as_ptr() as usize;
        assert_eq!(addr % crate::sizeclass::MIN_ALIGN, 0, "must be 8-aligned");
        assert!(
            seen.insert(addr),
            "addresses must be distinct while live: {addr:#x}"
        );
        held.push(p);
    }
    for p in held {
        unsafe { sh.dealloc_class(p, class) };
    }
}

#[test]
fn live_bytes_tracks_alloc_and_free() {
    let sh = test_subheap();
    let class = sizeclass::class_for(256).unwrap();
    let osz = sizeclass::size_of_class(class);
    assert_eq!(sh.live_bytes(), 0);
    let a = sh.alloc_class(class).unwrap();
    let b = sh.alloc_class(class).unwrap();
    assert_eq!(sh.live_bytes(), 2 * osz);
    assert_eq!(sh.live_objects(), 2);
    unsafe { sh.dealloc_class(a, class) };
    assert_eq!(sh.live_bytes(), osz);
    unsafe { sh.dealloc_class(b, class) };
    assert_eq!(sh.live_bytes(), 0);
    assert_eq!(sh.live_objects(), 0);
}

#[test]
fn budget_exhaustion_returns_none() {
    // Tiny budget; allocate until None, then confirm freeing lets us alloc again.
    let sh = SubHeapBuilder::new("tiny", 256 * 1024)
        .num_cpus(4)
        .cap_per_class(16)
        .on_exhaust(OnExhaust::None)
        .build_standalone()
        .unwrap();
    let class = sizeclass::class_for(4096).unwrap();
    let mut held = Vec::new();
    while let Some(p) = sh.alloc_class(class) {
        held.push(p);
        if held.len() > 100_000 {
            panic!("budget never exhausted — accounting broken");
        }
    }
    assert!(!held.is_empty(), "should have allocated at least some");
    // Free one; a subsequent alloc must now succeed.
    let p = held.pop().unwrap();
    unsafe { sh.dealloc_class(p, class) };
    let again = sh.alloc_class(class);
    assert!(again.is_some(), "alloc should succeed after a free");
    if let Some(p) = again {
        held.push(p);
    }
    for p in held {
        unsafe { sh.dealloc_class(p, class) };
    }
}

#[test]
fn many_classes_independent() {
    let sh = test_subheap();
    // Allocate one of several classes; each returns a distinct, correctly-sized
    // region we can fully write.
    let sizes = [8usize, 64, 256, 1500, 4096, 16384];
    let mut held: Vec<(NonNull<u8>, usize)> = Vec::new();
    for &s in &sizes {
        let class = sizeclass::class_for(s).unwrap();
        let p = sh.alloc_class(class).unwrap();
        unsafe { std::ptr::write_bytes(p.as_ptr(), 0xCD, sizeclass::size_of_class(class)) };
        held.push((p, class));
    }
    for (p, class) in held {
        unsafe { sh.dealloc_class(p, class) };
    }
}

#[test]
fn cross_thread_free_returns_to_owner() {
    // Allocate on the main thread; free on spawned threads (the networking
    // producer/consumer pattern). The remote-free queue must route objects back
    // to their home CPU and the owner must reclaim them — no leak, no corruption.
    use std::sync::Arc;
    // Size the slab to the machine so cross-thread frees land on real, distinct
    // home CPUs and exercise the remote-free queue (not the fallback shard).
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) as u32;
    let sh = Arc::new(
        SubHeapBuilder::new("xthread", 8 * 1024 * 1024)
            .num_cpus(nproc)
            .cap_per_class(64)
            .build_standalone()
            .expect("reserve"),
    );
    let class = sizeclass::class_for(256).unwrap();

    for _round in 0..20 {
        // Allocate a batch on this thread.
        let ptrs: Vec<usize> = (0..500)
            .filter_map(|_| sh.alloc_class(class).map(|p| p.as_ptr() as usize))
            .collect();
        assert!(!ptrs.is_empty());

        // Free them across several other threads (cross-CPU frees).
        let chunks: Vec<Vec<usize>> = ptrs.chunks(125).map(|c| c.to_vec()).collect();
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let sh = sh.clone();
                std::thread::spawn(move || {
                    for addr in chunk {
                        let p = NonNull::new(addr as *mut u8).unwrap();
                        unsafe { sh.dealloc_by_ptr(p) };
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Reclaim: allocate again on this thread, which drains remote queues.
        let reclaimed: Vec<_> = (0..500).filter_map(|_| sh.alloc_class(class)).collect();
        assert!(!reclaimed.is_empty(), "should reuse remote-freed memory");
        for p in reclaimed {
            unsafe { sh.dealloc_class(p, class) };
        }
    }
    // Drain everything: live count should settle to ~0 after a final reclaim pass.
    for _ in 0..10 {
        let v: Vec<_> = (0..500).filter_map(|_| sh.alloc_class(class)).collect();
        for p in v {
            unsafe { sh.dealloc_class(p, class) };
        }
    }
}

#[test]
fn large_and_nondivisor_classes_roundtrip_cross_thread() {
    // Regression: the per-span free bitmap must round-trip `slot ↔ address` for
    // EVERY class, including (a) classes whose size does not evenly tile a 64 KiB
    // span (1.5 KiB, 48 KiB, 96 KiB) and (b) classes LARGER than a span (128/192/
    // 256 KiB). Two historical bugs lived here: objects straddling a span boundary
    // (aliased addresses) and a zero-width bitmap scan for `osz > SPAN_BYTES`
    // (stranded frees → false budget exhaustion). Both only manifest on the
    // cross-thread-free (bitmap) path with these awkward sizes — exactly what
    // `installed.rs` first exercised. Allocate on the main thread, free on others,
    // then reclaim, asserting distinctness (no straddle) and full reuse (no strand).
    use std::sync::Arc;
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) as u32;
    // Budget large enough to hold the working set of the biggest class with room
    // to carve fresh + reclaim across rounds.
    let sh = Arc::new(
        SubHeapBuilder::new("awkward", 64 * 1024 * 1024)
            .num_cpus(nproc)
            .cap_per_class(64)
            .build_standalone()
            .expect("reserve"),
    );
    // 1500→1536 (non-divisor), 49152 (¾ span), 98304 (1.5 span), 131072 (2 spans),
    // 196608 (3 spans), 262144 (4 spans = MAX_SMALL).
    for &sz in &[1500usize, 49152, 98304, 131072, 196608, 262144] {
        let class = sizeclass::class_for(sz).unwrap();
        let osz = sizeclass::size_of_class(class);
        for _round in 0..4 {
            // Allocate a batch on this thread and fully write each (catches any
            // overlap with a neighbor: a straddling carve would alias regions).
            let n = 24;
            let ptrs: Vec<usize> = (0..n)
                .filter_map(|_| {
                    sh.alloc_class(class).map(|p| {
                        unsafe { std::ptr::write_bytes(p.as_ptr(), 0xE7, osz) };
                        p.as_ptr() as usize
                    })
                })
                .collect();
            assert_eq!(
                ptrs.len(),
                n,
                "{sz}B: all allocs should succeed within budget"
            );
            // Distinct live regions, and no two objects overlap given their size.
            let mut sorted = ptrs.clone();
            sorted.sort_unstable();
            for w in sorted.windows(2) {
                assert!(
                    w[1] - w[0] >= osz,
                    "{sz}B: objects overlap (straddle/alias): {:#x}+{osz} > {:#x}",
                    w[0],
                    w[1]
                );
            }
            // Free them all on other threads (the bitmap deposit path).
            let chunks: Vec<Vec<usize>> = ptrs.chunks(6).map(|c| c.to_vec()).collect();
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|chunk| {
                    let sh = sh.clone();
                    std::thread::spawn(move || {
                        for addr in chunk {
                            let p = NonNull::new(addr as *mut u8).unwrap();
                            unsafe { sh.dealloc_by_ptr(p) };
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        }
        // After the rounds, the owner must be able to reclaim the cross-thread
        // frees (no stranding): a fresh batch must still allocate.
        let reclaimed: Vec<_> = (0..24).filter_map(|_| sh.alloc_class(class)).collect();
        assert!(
            !reclaimed.is_empty(),
            "{sz}B: cross-thread frees must be reclaimable"
        );
        for p in reclaimed {
            unsafe { sh.dealloc_class(p, class) };
        }
    }
}

#[test]
fn large_drift_coalesces_and_caps_carved() {
    // Stage 1 (SpanPool): the `frag shift` scenario distilled. Each generation
    // allocates a large (>MAX_SMALL) run whose span-count GROWS by one, frees it,
    // then allocates the next, larger one. The old exact-span-count recycler could
    // never reuse a freed N-run for an (N+1)-run, so it carved fresh every
    // generation and `carved_bytes` grew as the SUM of all generations (the 5.99×
    // blow-up). With address-coalescing recycling, each freed run merges back into
    // one big free region that the next (larger) request is split out of, so the
    // high-water plateaus near the LARGEST single generation, not the sum.
    let span = crate::meta::SPAN_BYTES;
    let sh = test_subheap_bytes(256 * span); // 16 MiB, ample headroom
    let gens = 12usize;
    let mut carved_after_gen = Vec::new();
    let mut sum_no_coalesce = 0usize; // what carved WOULD be without reuse
    for g in 0..gens {
        let run_spans = 5 + g; // drift up: 5,6,7,...,16 spans
        let run_bytes = run_spans * span; // all > MAX_SMALL (256 KiB = 4 spans)
        assert!(run_bytes > sizeclass::MAX_SMALL, "must hit the large path");
        sum_no_coalesce += run_spans;
        let p = sh.alloc(run_bytes).expect("large alloc within budget");
        unsafe {
            *p.as_ptr() = g as u8;
            *p.as_ptr().add(run_bytes - 1) = g as u8;
            assert_eq!(*p.as_ptr(), g as u8, "large run not writable end-to-end");
        }
        // Free it back to the coalescing pool before the next, larger generation.
        unsafe { sh.dealloc_by_ptr(p) };
        carved_after_gen.push(sh.carved_bytes() / span); // in spans
    }
    let last = *carved_after_gen.last().unwrap();
    let peak_single = 5 + gens - 1; // largest generation, in spans
                                    // This single-buffer-per-generation pattern is a HARDER case than the real
                                    // `frag shift` bench (which holds many buffers ≈ a full generation live and
                                    // frees them together, building one big coalesced region that every later
                                    // same-size generation fits inside — carved there plateaus near 1×). Even
                                    // here the high-water must be a SMALL multiple of the largest generation, not
                                    // the sum: the freed region is reused (merged + split), only growing when a
                                    // request first exceeds the contiguous free region available at that moment.
    assert!(
        last <= peak_single * 2,
        "carved {last} spans should stay within ~2× the peak single gen \
         {peak_single}, not grow toward the no-coalescing sum {sum_no_coalesce}: \
         {carved_after_gen:?}"
    );
    // Decisive: dramatically below the no-coalescing baseline (the old exact-count
    // recycler carved the full sum). Observed ~5.5× better; require ≥ 2×.
    assert!(
        last * 2 < sum_no_coalesce,
        "coalescing ineffective: carved {last} spans vs no-coalesce sum {sum_no_coalesce}"
    );
}

#[test]
fn large_alloc_free_never_overlaps_under_random_churn() {
    // Double-handout invariant on the REAL alloc_large/dealloc_large path (not
    // just the SpanPool unit test): a randomized stream of large alloc/free with
    // drifting span-counts must never hand out two overlapping live runs.
    let span = crate::meta::SPAN_BYTES;
    let sh = test_subheap_bytes(96 * span);
    let mut rng: u64 = 0xDEAD_BEEF_0BAD_F00D;
    let step = |s: &mut u64| {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    };
    // live: (start_addr, len)
    let mut live: Vec<(usize, usize)> = Vec::new();
    for _ in 0..20_000 {
        let do_alloc = live.is_empty() || (step(&mut rng) & 1 == 0);
        if do_alloc {
            // 5..20 spans, always > MAX_SMALL.
            let run_spans = 5 + (step(&mut rng) % 16) as usize;
            let run_bytes = run_spans * span;
            if let Some(p) = sh.alloc(run_bytes) {
                let start = p.as_ptr() as usize;
                // No overlap with any live run.
                for &(ls, ll) in &live {
                    let overlap = start < ls + ll && ls < start + run_bytes;
                    assert!(!overlap, "large double hand-out");
                }
                // Write a per-run sentinel to its first byte.
                unsafe { *p.as_ptr() = 0x5A };
                live.push((start, run_bytes));
            }
        } else {
            let k = (step(&mut rng) as usize) % live.len();
            let (start, _len) = live.swap_remove(k);
            // Sentinel must survive (no one else wrote our run).
            unsafe {
                assert_eq!(*(start as *const u8), 0x5A, "large run torn by overlap");
                sh.dealloc_by_ptr(NonNull::new(start as *mut u8).unwrap());
            }
        }
    }
}

#[test]
fn crossclass_reclaim_reuses_spans_and_caps_carved() {
    // The `frag crossclass` fix: free every object of class A, reclaim its now-empty
    // spans, then allocate class B. Without reclaim, B carves fresh and carved climbs
    // by ~one class-worth per transition (6× stranding). With reclaim, B reuses A's
    // retired spans, so carved plateaus.
    let span = crate::meta::SPAN_BYTES;
    let sh = test_subheap_bytes(64 * span); // 4 MiB
                                            // ~4× apart so each is its own class & carves its own spans (the bench's set).
    let classes = [64usize, 256, 1024, 4096, 16384];
    let per_class_bytes = 8 * span; // ~512 KiB live per class step
    let mut carved_after = Vec::new();
    for (step, &size) in classes.iter().enumerate() {
        let class = sizeclass::class_for(size).unwrap();
        let osz = sizeclass::size_of_class(class);
        let count = per_class_bytes / osz;
        let mut held = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(p) = sh.alloc_class(class) {
                // Write a per-step sentinel over the whole object — a cross-class
                // double-hand-out (B reusing a span still owned by A) would tear this.
                unsafe { std::ptr::write_bytes(p.as_ptr(), (0x40 + step) as u8, osz) };
                held.push(p);
            }
        }
        assert!(!held.is_empty(), "step {step} ({size}B) should allocate");
        // Verify the sentinel survived (no overlap with the previous class's still
        // -reclaiming spans).
        for p in &held {
            unsafe {
                assert_eq!(*p.as_ptr(), (0x40 + step) as u8, "step {step}: object torn");
            }
        }
        // Free everything of this class.
        for p in held {
            unsafe {
                sh.dealloc_class(p, class);
            }
        }
        // Reclaim the now-empty spans so the NEXT class can reuse them. Cap high so
        // the whole class is swept in this synchronous call (test, not supervisor).
        let got = sh.reclaim_empty_spans(usize::MAX);
        carved_after.push(sh.carved_bytes() / span);
        let _ = got;
    }
    // Decisive: carved must NOT grow ~one-class-per-step. With reclaim+reuse the
    // high-water plateaus near the largest single class's footprint, not the sum.
    let last = *carved_after.last().unwrap();
    let max_step_spans = per_class_bytes / span + 2; // one class-worth + slack
    assert!(
        last <= max_step_spans * 2,
        "carved kept climbing across classes (reclaim not reused): {carved_after:?} spans"
    );
}

#[test]
fn crossclass_reclaim_stale_hint_no_double_handout() {
    // DETERMINISTIC regression for the stale-`reuse_hint` cross-class double-hand-out
    // the adversarial panel found (and which a probabilistic single-process churn
    // cannot reliably reproduce — that's what the deferred loom pass is for). We drive
    // the exact window via test seams:
    //   1. allocate + free a class-A batch so A owns a span S (A's osz >= 1024 so the
    //      reuse_hint path in central_take is active: slots_per_span <= 64),
    //   2. reclaim S (retag -> LARGE, park in SpanPool),
    //   3. carve class B by allocating B until it reuses S and tiles it for osz_B,
    //   4. FORCE central[shard][A].reuse_hint = S (simulating a deposit preempted
    //      mid-flush re-publishing a now-stale hint AFTER reclaim — "window 2"),
    //   5. refill class A: central_take reads the stale hint S. With the class
    //      re-validation it SKIPS S (now LARGE/B); without it (negative control) it
    //      would scan S with osz_A and hand out B-tiled slots as A objects.
    // Assert: no A object ever lands inside a live B object's span, and B's sentinels
    // are never torn.
    use std::sync::Arc;
    let span = crate::meta::SPAN_BYTES;
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) as u32;
    let sh = Arc::new(
        SubHeapBuilder::new("stalehint", 64 * span)
            .num_cpus(nproc)
            .cap_per_class(64)
            .build_standalone()
            .expect("reserve"),
    );
    let class_a = sizeclass::class_for(1024).unwrap(); // osz 1024, slots/span = 64
    let class_b = sizeclass::class_for(4096).unwrap(); // osz 4096, slots/span = 16
    let osz_a = sizeclass::size_of_class(class_a);
    let osz_b = sizeclass::size_of_class(class_b);
    let shard = sh.test_thread_shard();

    // (1) Allocate a full span's worth of A on this thread, then free them all on a
    // SPAWNED thread. A's home shard is this thread's shard; the freeing thread is a
    // different shard, so each free is a cross-shard CENTRAL bitmap deposit (sets the
    // free bit) rather than an L2-cached push — so the spans become fully-free *in
    // central*, which is what `reclaim_empty_spans` (and the safety check) require.
    let a_count = (span / osz_a) * 2; // ~2 spans of A
    let a_ptrs: Vec<usize> = (0..a_count)
        .filter_map(|_| sh.alloc_class(class_a).map(|p| p.as_ptr() as usize))
        .collect();
    assert!(!a_ptrs.is_empty());
    // Track candidate span S by INDEX (address masking can underflow vs arena base).
    let s_addr = *a_ptrs.iter().min().unwrap();
    let s_span = sh.test_span_of(s_addr as *const u8);
    let _ = span;
    {
        let sh2 = sh.clone();
        let ptrs = a_ptrs.clone();
        std::thread::spawn(move || {
            for a in ptrs {
                unsafe { sh2.dealloc_class(NonNull::new(a as *mut u8).unwrap(), class_a) };
            }
        })
        .join()
        .unwrap();
    }

    // (2) Reclaim: S (and the other now-empty A spans) retag -> LARGE, into the pool.
    let reclaimed = sh.reclaim_empty_spans(usize::MAX);
    assert!(
        reclaimed > 0,
        "no spans reclaimed — A's frees didn't reach central as fully-free"
    );

    // (3) Carve B until it reuses the reclaimed spans (the pool hands lowest-address
    // runs first, so B should land on S's region). Hold them live + sentineled.
    let b_count = (span / osz_b) * 2;
    let mut b_live: Vec<(usize, u8)> = Vec::new();
    for i in 0..b_count {
        if let Some(p) = sh.alloc_class(class_b) {
            let sentinel = (0x80 + (i & 0x3f)) as u8;
            unsafe { std::ptr::write_bytes(p.as_ptr(), sentinel, osz_b) };
            b_live.push((p.as_ptr() as usize, sentinel));
        }
    }
    // Confirm B actually reused S's span (else the test wouldn't exercise the hazard).
    let b_in_s: Vec<usize> = b_live
        .iter()
        .filter(|&&(a, _)| sh.test_span_of(a as *const u8) == s_span)
        .map(|&(a, _)| a)
        .collect();
    assert!(
        !b_in_s.is_empty(),
        "B did not reuse the reclaimed span S — test precondition unmet"
    );

    // Free HALF of S's B objects cross-thread (so S has FREE B-slots for the stale-hint
    // scan to hand out) while keeping the other half LIVE (so a bogus A hand-out from
    // S provably collides with a live B object). The freed B slots' bits are set in S's
    // bitmap — exactly what a stale-hint class-A scan would wrongly hand out as A.
    let b_free: Vec<usize> = b_in_s
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, &a)| a)
        .collect();
    // Remove b_free from b_live (keep the odd-indexed ones live) and free them.
    b_live.retain(|&(a, _)| !b_free.contains(&a));
    {
        let sh2 = sh.clone();
        let frees = b_free.clone();
        std::thread::spawn(move || {
            for a in frees {
                unsafe { sh2.dealloc_class(NonNull::new(a as *mut u8).unwrap(), class_b) };
            }
        })
        .join()
        .unwrap();
    }

    // (4) Re-publish the stale hint: central[shard][A].reuse_hint = S (now B-tiled,
    // with some B slots free and some B objects still live).
    sh.test_set_reuse_hint(shard, class_a, s_span);

    // (5) Refill class A. central_take reads the stale hint. Allocate a batch of A and
    // check NONE of them lands inside S (B's span) — that would be a double hand-out.
    let mut a_new = Vec::new();
    for _ in 0..a_count {
        if let Some(p) = sh.alloc_class(class_a) {
            let addr = p.as_ptr() as usize;
            // The load-bearing assertion: no A object may be carved out of S, which is
            // now owned by B. (Stale hint + no revalidation => exactly this.)
            assert!(
                sh.test_span_of(addr as *const u8) != s_span,
                "cross-class double hand-out: class-A object {addr:#x} landed in B-owned span {s_span}"
            );
            unsafe { std::ptr::write_bytes(p.as_ptr(), 0x11, osz_a) };
            a_new.push(addr);
        }
    }
    // B's objects must be intact — no A write tore them.
    for &(addr, sentinel) in &b_live {
        unsafe {
            assert_eq!(
                *(addr as *const u8),
                sentinel,
                "B object {addr:#x} torn by an A hand-out"
            );
        }
    }

    for a in a_new {
        unsafe { sh.dealloc_class(NonNull::new(a as *mut u8).unwrap(), class_a) };
    }
    for (addr, _) in b_live {
        unsafe { sh.dealloc_class(NonNull::new(addr as *mut u8).unwrap(), class_b) };
    }
}

#[test]
fn crossclass_reclaim_no_double_handout_cross_thread() {
    // Differential double-hand-out detector (the adversarial panel's gate) for the
    // stale-`reuse_hint` hazard. That hazard is ONLY reachable via the CROSS-shard
    // deposit path: `reuse_hint` is written in `dealloc_class_batch`
    // (subheap.rs:883), reached only when the freeing thread's shard differs from the
    // object's home shard. A single-thread free is same-shard → local L2 → never
    // writes the hint, so a single-threaded test cannot reach the bug (verified: the
    // `toccata_reclaim_negative_control` cfg, which disables the central_take class
    // re-validation, does NOT fail single-threaded). So we drive it cross-thread:
    //   1. main (shard M) allocates a class-A batch (homed on M),
    //   2. a spawned thread (a different shard) frees the whole batch → cross-shard
    //      deposits set A's spans free AND write central[M][A].reuse_hint = S,
    //   3. main reclaims: S is fully free → retagged LARGE, parked in the SpanPool,
    //      but central[M][A].reuse_hint STILL == S (stale),
    //   4. main carves class B (different osz) which REUSES S from the pool and tiles
    //      it for B, then main refills class A → central_take reads the stale hint S.
    // Without the class re-validation in central_take, step 4 scans the B-tiled S with
    // A's osz and hands out B's slots as A objects (a cross-class double hand-out).
    // The detector below asserts no address is ever simultaneously live in two
    // classes and no sentinel is ever torn.
    use std::{collections::HashMap, sync::Arc};
    let span = crate::meta::SPAN_BYTES;
    // Many shards so the spawned freer lands on a different shard than main (M).
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) as u32;
    let sh = Arc::new(
        SubHeapBuilder::new("xclass", 64 * span)
            .num_cpus(nproc)
            .cap_per_class(64)
            .build_standalone()
            .expect("reserve"),
    );
    // Distinct osz so a B-tiled span handed out as A (or vice versa) tears a sentinel.
    let classes = [64usize, 256, 1024, 4096, 16384];
    let per_step = 6 * span; // whole-span-multiples of live, so spans go fully empty
                             // addr -> (class, sentinel) for everything currently live (across all classes).
    let mut live: HashMap<usize, (usize, u8)> = HashMap::new();

    for round in 0..40 {
        let class = sizeclass::class_for(classes[round % classes.len()]).unwrap();
        let osz = sizeclass::size_of_class(class);
        let count = per_step / osz;
        // (1) allocate on main, sentinel each, register — checking no double hand-out.
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(p) = sh.alloc_class(class) {
                let addr = p.as_ptr() as usize;
                let sentinel = (0x40 + round) as u8;
                assert!(
                    !live.contains_key(&addr),
                    "double hand-out at {addr:#x}: already live as {:?} (now class {class})",
                    live.get(&addr)
                );
                unsafe { std::ptr::write_bytes(p.as_ptr(), sentinel, osz) };
                live.insert(addr, (class, sentinel));
                batch.push(addr);
            }
        }
        // Re-verify every live object's sentinel: a fresh B-carve reusing a reclaimed
        // span that some class still believes it owns would have torn one.
        for (&addr, &(_c, sentinel)) in &live {
            unsafe {
                assert_eq!(*(addr as *const u8), sentinel, "tear at {addr:#x}");
            }
        }
        // (2) free this batch on ANOTHER thread (cross-shard → central deposit +
        // reuse_hint write). Free the whole batch so its spans become fully free.
        for &addr in &batch {
            live.remove(&addr);
        }
        {
            let sh2 = sh.clone();
            let b = batch.clone();
            std::thread::spawn(move || {
                for addr in b {
                    unsafe { sh2.dealloc_class(NonNull::new(addr as *mut u8).unwrap(), class) };
                }
            })
            .join()
            .unwrap();
        }
        // (3) reclaim the now-empty spans (retag to LARGE, park in SpanPool).
        sh.reclaim_empty_spans(usize::MAX);
        // Loop: the NEXT round carves a different class, reusing these spans (4), and
        // also refills the just-freed class on main, reading any stale hint.
    }
    // Final: drain + integrity.
    for (&addr, &(_c, sentinel)) in &live {
        unsafe {
            assert_eq!(*(addr as *const u8), sentinel, "final tear at {addr:#x}");
        }
    }
    for (addr, (lclass, _)) in live {
        unsafe { sh.dealloc_class(NonNull::new(addr as *mut u8).unwrap(), lclass) };
    }
}

#[test]
fn churn_does_not_leak_or_corrupt() {
    let sh = test_subheap();
    let class = sizeclass::class_for(512).unwrap();
    // Repeated alloc/free cycles must keep live_bytes returning to zero and never
    // hand out overlapping live regions.
    for _round in 0..50 {
        let mut held = Vec::new();
        for _ in 0..200 {
            if let Some(p) = sh.alloc_class(class) {
                held.push(p);
            }
        }
        // Verify distinctness within the round.
        let mut seen = HashSet::new();
        for p in &held {
            assert!(seen.insert(p.as_ptr() as usize), "overlap within live set");
        }
        for p in held {
            unsafe { sh.dealloc_class(p, class) };
        }
        assert_eq!(sh.live_bytes(), 0, "leak after round");
    }
}

// NOTE: the seal invariant (no OS-memory syscall after init) is validated by a
// syscall-trace integration test on Linux (`tests/no_syscalls_after_seal.rs`),
// not here — calling the global `seal()` from a unit test would poison sibling
// tests that still need to `reserve()`.

#[test]
fn dealloc_by_ptr_recovers_class() {
    // The global-allocator / Box-drop path frees by pointer alone; the span
    // table must recover the class so we don't need the Layout.
    let sh = test_subheap();
    let sizes = [8usize, 100, 1500, 4096];
    let mut held = Vec::new();
    for &s in &sizes {
        let class = sizeclass::class_for(s).unwrap();
        let p = sh.alloc_class(class).unwrap();
        assert_eq!(
            sh.class_of(p),
            Some(class),
            "span table must know the class for {s}B"
        );
        assert!(sh.owns(p), "sub-heap must own its own pointer");
        held.push(p);
    }
    let before = sh.live_objects();
    assert_eq!(before, sizes.len() as u64);
    for p in held {
        // Free by pointer alone — no class passed.
        assert!(
            unsafe { sh.dealloc_by_ptr(p) },
            "dealloc_by_ptr should succeed for owned ptr"
        );
    }
    assert_eq!(sh.live_objects(), 0);
    assert_eq!(sh.live_bytes(), 0);
}

#[test]
fn does_not_own_foreign_pointer() {
    let sh = test_subheap();
    let stack_var = 42u8;
    let foreign = NonNull::from(&stack_var);
    assert!(!sh.owns(foreign.cast()));
    assert_eq!(sh.class_of(foreign.cast()), None);
    assert!(!unsafe { sh.dealloc_by_ptr(foreign.cast()) });
}

/// R2: an over-budget request that exceeds `RLIMIT_MEMLOCK` must fail loudly and
/// deterministically at init with an actionable error — never silently, never a
/// runtime stall. (Linux only; the dev stub has no lock limit.)
#[cfg(target_os = "linux")]
#[test]
fn over_memlock_budget_fails_loudly() {
    let limit = crate::sys::memlock_limit();
    if limit == u64::MAX {
        eprintln!("SKIP: RLIMIT_MEMLOCK is unlimited on this host");
        return;
    }
    // Ask for more than the limit; reservation must Err with a clear message.
    let over = (limit as usize).saturating_add(64 * 1024 * 1024);
    let res = SubHeapBuilder::new("over", over).build_standalone();
    match res {
        Err(crate::ReserveError::RlimitTooLow { .. }) | Err(crate::ReserveError::Mlock { .. }) => {}
        Err(other) => panic!("expected a memlock-related error, got: {other}"),
        Ok(_) => panic!("expected reservation over RLIMIT_MEMLOCK to fail"),
    }
    // And the error message must tell the operator how to fix it.
    let msg = match SubHeapBuilder::new("over", over).build_standalone() {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected reservation over RLIMIT_MEMLOCK to fail"),
    };
    assert!(
        msg.contains("RLIMIT_MEMLOCK"),
        "error must mention RLIMIT_MEMLOCK: {msg}"
    );
}
