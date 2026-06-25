// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Shared benchmark workloads, allocator-agnostic.
//!
//! The workloads here allocate through the **process** `#[global_allocator]` —
//! whatever the running binary installed. Each `src/bin/<name>.rs` shim installs
//! exactly one allocator and calls [`run`], so the allocator under test is chosen
//! by *which binary you run*, with no cargo features and no way to install two at
//! once.
//!
//! CLI (forwarded by every shim): `<bin> <workload> [args...]`
//! * `throughput` — single-thread, multi-thread, producer/consumer, and
//!   alloc-latency-distribution micro-suite.
//! * `latency-tail [secs] [churn_mib]` — a latency-critical "networking" thread
//!   vs. RSS pressure from background churners; reports the tail and the
//!   >1ms/>10ms/>100ms stall counts.
//! * `micro [iters]` — a tight fixed-size alloc/free loop for `perf`.
//! * `frag <mode> [args...]` — fragmentation / budget-stranding: reports
//!   footprint/live (the "frag factor"). toccata's footprint is its arena
//!   carved-high-water (RSS is pinned by the mlock'd budget); the others use
//!   process RSS. Modes:
//!   * `shift` — large-object size drift (adversarial).
//!   * `match` — stable large sizes (control).
//!   * `crossclass` — small-object cross-size-class stranding (free one class,
//!     allocate another).
//!   * `mixed` — seeded steady-state churn over a skewed small+large size mix
//!     (realistic traffic).
//!
//! Defaults run all of `throughput` when no workload is given.

use std::time::{Duration, Instant};

/// Entry point every binary shim calls after installing its allocator.
/// `allocator` is the human-readable name for report headers.
pub fn run(allocator: &str) {
    let mut args = std::env::args().skip(1);
    let workload = args.next().unwrap_or_else(|| "throughput".to_string());
    let rest: Vec<String> = args.collect();
    match workload.as_str() {
        "throughput" => throughput(allocator),
        "single" => single(allocator),
        "multi" => multi(allocator, &rest),
        "mpsc" => {
            let reps = std::env::var("TOCCATA_BENCH_REPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(7);
            let pairs = rest.first().and_then(|s| s.parse().ok()).unwrap_or(8);
            let mut s = Vec::with_capacity(reps);
            for _ in 0..reps {
                s.push(bench_mpsc_once(pairs));
            }
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "=== mpsc x{pairs}: {allocator} ({reps} reps) ===  min {:>5.2}  median {:>5.2}  max {:>5.2} ns/op",
                s[0], s[reps / 2], s[reps - 1]
            );
        }
        "latency-tail" => latency_tail(allocator, &rest),
        "micro" => micro(allocator, &rest),
        "framepool" => framepool(allocator, &rest),
        "prodcons" => prodcons(allocator, &rest),
        "frag" => frag(allocator, &rest),
        other => {
            eprintln!(
                "unknown workload {other:?}; expected throughput | single | multi | latency-tail | micro | framepool | prodcons | frag"
            );
            std::process::exit(2);
        }
    }
}

/// Isolated single-thread `churn`, repeated for a stable median — a clean
/// profiling + attribution target (the `throughput` suite interleaves four
/// sub-benches, which pollutes a `perf record`).
fn single(allocator: &str) {
    let rounds = 5_000_000;
    let reps = std::env::var("TOCCATA_BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let _ = churn(100_000, 256); // warm up
    let mut samples = Vec::with_capacity(reps);
    let mut cs = 0u64;
    for _ in 0..reps {
        let start = Instant::now();
        cs = churn(rounds, 1024);
        samples.push(start.elapsed().as_nanos() as f64 / rounds as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "=== single: {allocator} ({reps} reps) ===  min {:>5.2}  median {:>5.2}  max {:>5.2} ns/op   (checksum {cs})",
        samples[0], samples[reps / 2], samples[reps - 1]
    );
}

/// Isolated multi-thread `churn`, repeated for a stable median.
fn multi(allocator: &str, args: &[String]) {
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let threads: usize = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(nproc.min(16));
    let rounds = 2_000_000;
    let reps = std::env::var("TOCCATA_BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let mut samples = Vec::with_capacity(reps);
    let mut cs = 0u64;
    for _ in 0..reps {
        let start = Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|_| std::thread::spawn(move || churn(rounds, 512)))
            .collect();
        cs = 0;
        for h in handles {
            cs = cs.wrapping_add(h.join().unwrap());
        }
        samples.push(start.elapsed().as_nanos() as f64 / (rounds * threads) as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "=== multi x{threads}: {allocator} ({reps} reps) ===  min {:>5.2}  median {:>5.2}  max {:>5.2} ns/op   (checksum {cs})",
        samples[0], samples[reps / 2], samples[reps - 1]
    );
}

// ---------------------------------------------------------------------------
// throughput micro-suite (formerly benches/allocators.rs)
// ---------------------------------------------------------------------------

/// Sizes spanning toccata's small classes, weighted toward small,
/// networking-like sizes (no oversize — that's a separate concern).
const SIZES: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 1500, 2048, 4096];

#[inline(never)]
fn churn(rounds: usize, live: usize) -> u64 {
    // Keep `live` allocations outstanding, cycling through sizes; return a
    // checksum so the optimizer can't elide the work.
    let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(live);
    let mut checksum = 0u64;
    let mut i = 0usize;
    for r in 0..rounds {
        let size = SIZES[(r ^ i) % SIZES.len()];
        let mut v = vec![0u8; size];
        v[0] = (r & 0xff) as u8;
        v[size - 1] = (i & 0xff) as u8;
        checksum = checksum
            .wrapping_add(v[0] as u64)
            .wrapping_add(v[size - 1] as u64);
        if bufs.len() < live {
            bufs.push(v);
        } else {
            bufs[i % live] = v; // drop the old one (free) and store the new
        }
        i = i.wrapping_add(1);
    }
    checksum
}

fn throughput(allocator: &str) {
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!("=== allocator benchmark: {allocator} ({nproc} cpus) ===");
    let _ = churn(100_000, 256); // warm up

    bench_single_thread();
    bench_multi_thread(nproc.min(16));
    bench_producer_consumer((nproc / 2).clamp(1, 8));
    bench_latency_distribution();

    // Keep the process alive a beat so any background purge threads (jemalloc)
    // would have a chance to run — relevant when comparing steady-state RSS.
    std::thread::sleep(Duration::from_millis(50));
}

fn bench_single_thread() {
    let rounds = 5_000_000;
    let start = Instant::now();
    let cs = churn(rounds, 1024);
    let elapsed = start.elapsed();
    let per = elapsed.as_nanos() as f64 / rounds as f64;
    println!(
        "  single-thread:   {rounds} ops in {:>7.1}ms  =>  {per:>5.1} ns/op   (checksum {cs})",
        elapsed.as_secs_f64() * 1e3
    );
}

fn bench_multi_thread(threads: usize) {
    let rounds = 2_000_000;
    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|_| std::thread::spawn(move || churn(rounds, 512)))
        .collect();
    let mut cs = 0u64;
    for h in handles {
        cs = cs.wrapping_add(h.join().unwrap());
    }
    let elapsed = start.elapsed();
    let total = rounds * threads;
    let per = elapsed.as_nanos() as f64 / total as f64;
    println!(
        "  multi-thread x{threads}: {total} ops in {:>7.1}ms  =>  {per:>5.1} ns/op   (checksum {cs})",
        elapsed.as_secs_f64() * 1e3
    );
}

fn bench_producer_consumer(pairs: usize) {
    let per = bench_mpsc_once(pairs);
    let total = 1_000_000 * pairs;
    println!("  prod/cons x{pairs}:  {total} ops  =>  {per:>5.1} ns/op");
}

/// One trial of the `std::mpsc` producer/consumer workload: producers allocate
/// buffers and hand them to consumers via a channel; consumers free them. Returns
/// ns/op. (The channel itself allocates a block per ~N sends, so this measures two
/// interleaved cross-thread free streams — the Vec buffer and the channel block.)
fn bench_mpsc_once(pairs: usize) -> f64 {
    use std::sync::mpsc;
    let per_producer = 1_000_000;
    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..pairs {
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let prod = std::thread::spawn(move || {
            for r in 0..per_producer {
                let size = SIZES[r % SIZES.len()];
                let mut v = vec![0u8; size];
                v[0] = 1;
                if tx.send(v).is_err() {
                    break;
                }
            }
        });
        let cons = std::thread::spawn(move || {
            let mut n = 0u64;
            while let Ok(v) = rx.recv() {
                n += v[0] as u64; // touch then drop (free on this thread)
            }
            n
        });
        handles.push((prod, cons));
    }
    for (p, c) in handles {
        p.join().unwrap();
        c.join().unwrap();
    }
    start.elapsed().as_nanos() as f64 / (per_producer * pairs) as f64
}

fn bench_latency_distribution() {
    // Measure individual allocation latency and report the tail. This is the
    // headline metric for toccata: not mean throughput but the WORST case.
    let samples = 2_000_000;
    let mut latencies: Vec<u32> = Vec::with_capacity(samples);
    let mut sink: Vec<Vec<u8>> = Vec::with_capacity(1024);
    for r in 0..samples {
        let size = SIZES[r % SIZES.len()];
        let t = Instant::now();
        let v = vec![0u8; size];
        let ns = t.elapsed().as_nanos() as u32;
        latencies.push(ns);
        if sink.len() < 1024 {
            sink.push(v);
        } else {
            sink[r % 1024] = v;
        }
    }
    latencies.sort_unstable();
    let pct = |p: f64| latencies[((samples as f64 * p) as usize).min(samples - 1)];
    println!(
        "  alloc latency:   p50={:>4}ns  p99={:>5}ns  p999={:>6}ns  p9999={:>7}ns  max={:>8}ns",
        pct(0.50),
        pct(0.99),
        pct(0.999),
        pct(0.9999),
        latencies[samples - 1]
    );
}

// ---------------------------------------------------------------------------
// latency-tail under RSS pressure (formerly bin/latency_tail.rs)
// ---------------------------------------------------------------------------

fn latency_tail(allocator: &str, args: &[String]) {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    let secs: u64 = args.first().and_then(|s| s.parse().ok()).unwrap_or(5);
    // Total churn working-set in MiB. Keep near (just under/at) the cgroup's
    // memory.high so the kernel throttles rather than OOM-kills.
    let churn_mib: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let stop = Arc::new(AtomicBool::new(false));

    // Background "churners" that hold a steady working set and continuously
    // rewrite it, pushing RSS against the limit and forcing a decay/purge
    // allocator to madvise+refault (the condition that triggers cgroup throttle).
    let mut bg = Vec::new();
    let per_thread_mib = (churn_mib / 4).max(1);
    for _ in 0..4 {
        let stop = stop.clone();
        bg.push(std::thread::spawn(move || {
            let mut blocks: Vec<Vec<u8>> = (0..per_thread_mib)
                .map(|_| vec![7u8; 1024 * 1024])
                .collect();
            let mut i = 0usize;
            while !stop.load(Ordering::Relaxed) {
                // Replace one block (free + fresh alloc) and touch it — drives
                // purge/refault under pressure without unbounded growth.
                let idx = i % blocks.len();
                blocks[idx] = vec![(i & 0xff) as u8; 1024 * 1024];
                i = i.wrapping_add(1);
            }
            blocks.len()
        }));
    }

    // The latency-critical "networking" thread: small alloc/free every iteration.
    // Latencies go into a FIXED log-bucketed histogram (O(1) memory) so the
    // measurement never competes with the allocator's budget.
    let mut hist = Histogram::new();
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut sink: Vec<Vec<u8>> = Vec::with_capacity(256);
    let mut iters = 0u64;
    while Instant::now() < deadline {
        let t = Instant::now();
        let v = vec![0u8; 1500]; // an MTU-sized packet buffer
        let ns = t.elapsed().as_nanos() as u64;
        hist.record(ns);
        if sink.len() < 256 {
            sink.push(v);
        } else {
            sink[(iters as usize) % 256] = v;
        }
        iters += 1;
    }
    stop.store(true, Ordering::Relaxed);
    let mut held = 0usize;
    for h in bg {
        held += h.join().unwrap_or(0);
    }
    let _ = held;

    let n = hist.count;
    println!("=== latency_tail: {allocator} ({n} samples over {secs}s) ===");
    println!(
        "  p50={:>4}ns p99={:>5}ns p999={:>6}ns p9999={:>8}ns p99999={:>9}ns max={:>10}ns",
        hist.pct(0.5),
        hist.pct(0.99),
        hist.pct(0.999),
        hist.pct(0.9999),
        hist.pct(0.99999),
        hist.max
    );
    let over = |thresh_ns: u64| hist.count_over(thresh_ns);
    println!(
        "  stalls: >1ms={}  >10ms={}  >100ms={}  (a never-stall allocator keeps these at 0)",
        over(1_000_000),
        over(10_000_000),
        over(100_000_000)
    );
}

/// A fixed-memory log-bucketed latency histogram. Bucket `i` covers
/// `[2^i, 2^(i+1))` nanoseconds; 64 buckets span 1ns .. ~584 years. O(1) memory,
/// so recording is allocation-free and never competes with the allocator.
struct Histogram {
    buckets: [u64; 64],
    count: u64,
    max: u64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            buckets: [0; 64],
            count: 0,
            max: 0,
        }
    }

    #[inline]
    fn record(&mut self, ns: u64) {
        let b = if ns == 0 {
            0
        } else {
            63 - ns.leading_zeros() as usize
        };
        self.buckets[b] += 1;
        self.count += 1;
        if ns > self.max {
            self.max = ns;
        }
    }

    /// Approximate percentile (returns the lower edge of the containing bucket).
    fn pct(&self, p: f64) -> u64 {
        let target = (self.count as f64 * p) as u64;
        let mut acc = 0u64;
        for (i, &c) in self.buckets.iter().enumerate() {
            acc += c;
            if acc >= target {
                return 1u64 << i;
            }
        }
        self.max
    }

    /// Count of samples strictly greater than `thresh_ns`.
    fn count_over(&self, thresh_ns: u64) -> u64 {
        let mut n = 0u64;
        for (i, &c) in self.buckets.iter().enumerate() {
            if (1u64 << i).saturating_mul(2).saturating_sub(1) > thresh_ns {
                n += c;
            }
        }
        n
    }
}

// ---------------------------------------------------------------------------
// prodcons: clean producer/consumer attribution (no mpsc confound)
// ---------------------------------------------------------------------------

/// A fixed-capacity SPSC ring of raw boxed buffers. Unlike `std::mpsc` (which
/// heap-allocates a node *per send* through the allocator under test — see the
/// `bench_producer_consumer` confound), this ring is allocated **once** up front,
/// so the only allocator traffic the benchmark measures is the producer's buffer
/// `alloc` and the consumer's buffer `free`: the pure one-way memory-flow pattern
/// (born on producer CPU, dies on consumer CPU) that is toccata's target.
///
/// One producer, one consumer per ring. Backpressure is a bounded spin (the
/// producer waits when full, the consumer waits when empty) — no allocation, no
/// blocking primitive that could allocate.
/// One published buffer: a raw `(ptr, len)`. Carrying the raw parts (not a
/// `Box<Vec<u8>>`) keeps the measured allocator traffic to **exactly one**
/// alloc/free per op — the buffer itself — instead of two (a `Box<Vec>` adds a
/// 24-byte cross-thread allocation per op that dilutes the signal for every
/// allocator). The slot packs ptr+len into two atomics; the consumer rebuilds the
/// `Vec` to free it on its thread.
#[repr(C)]
struct Slot {
    ptr: std::sync::atomic::AtomicPtr<u8>,
    len: std::sync::atomic::AtomicUsize,
}

struct SpscRing {
    slots: Box<[Slot]>,
    /// Next slot the producer will write (monotonic; index = head % cap).
    head: std::sync::atomic::AtomicUsize,
    /// Next slot the consumer will read.
    tail: std::sync::atomic::AtomicUsize,
    cap: usize,
}

impl SpscRing {
    fn new(cap: usize) -> Self {
        let cap = cap.next_power_of_two();
        let slots = (0..cap)
            .map(|_| Slot {
                ptr: std::sync::atomic::AtomicPtr::new(core::ptr::null_mut()),
                len: std::sync::atomic::AtomicUsize::new(0),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots,
            head: std::sync::atomic::AtomicUsize::new(0),
            tail: std::sync::atomic::AtomicUsize::new(0),
            cap,
        }
    }

    /// Producer: hand a buffer (its raw parts) to the consumer. Spins while full.
    /// Takes ownership of `v` and forgets it; the consumer reconstitutes + frees.
    #[inline]
    fn push(&self, v: Vec<u8>) {
        use std::sync::atomic::Ordering;
        let head = self.head.load(Ordering::Relaxed);
        loop {
            if head.wrapping_sub(self.tail.load(Ordering::Acquire)) < self.cap {
                break;
            }
            core::hint::spin_loop();
        }
        let mut v = core::mem::ManuallyDrop::new(v);
        let (ptr, len) = (v.as_mut_ptr(), v.len());
        debug_assert_eq!(v.len(), v.capacity()); // vec![0u8; n] ⇒ len == cap
        let slot = &self.slots[head & (self.cap - 1)];
        slot.len.store(len, Ordering::Relaxed);
        slot.ptr.store(ptr, Ordering::Release);
        self.head.store(head.wrapping_add(1), Ordering::Release);
    }

    /// Consumer: take the next buffer as a `Vec` (to free on this thread), or
    /// `None` once the producer is done AND the ring is drained.
    #[inline]
    fn pop(&self, producer_done: &std::sync::atomic::AtomicBool) -> Option<Vec<u8>> {
        use std::sync::atomic::Ordering;
        let tail = self.tail.load(Ordering::Relaxed);
        loop {
            if tail != self.head.load(Ordering::Acquire) {
                break;
            }
            if producer_done.load(Ordering::Acquire) && tail == self.head.load(Ordering::Acquire) {
                return None;
            }
            core::hint::spin_loop();
        }
        let slot = &self.slots[tail & (self.cap - 1)];
        let ptr = slot.ptr.swap(core::ptr::null_mut(), Ordering::Acquire);
        let len = slot.len.load(Ordering::Relaxed);
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        // SAFETY: producer published these raw parts from a `vec![0u8; len]`
        // (len == capacity); we now own them and rebuild the exact Vec to free it.
        Some(unsafe { Vec::from_raw_parts(ptr, len, len) })
    }
}

/// Run one timed producer/consumer trial and return (ns/op, freed). Each pair gets
/// its own pre-allocated SPSC ring, so the measured allocator traffic is exactly:
/// producer allocs a buffer, consumer (on a different CPU) frees it. Threads are
/// left unpinned — the same scheduler freedom jemalloc runs under, so the
/// comparison stays apples-to-apples.
fn prodcons_trial(pairs: usize, fixed_size: usize, per_producer: usize) -> (f64, u64) {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    const RING_CAP: usize = 1024;

    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..pairs {
        let ring = Arc::new(SpscRing::new(RING_CAP));
        let done = Arc::new(AtomicBool::new(false));
        let prod = {
            let ring = ring.clone();
            let done = done.clone();
            std::thread::spawn(move || {
                for r in 0..per_producer {
                    let size = if fixed_size != 0 {
                        fixed_size
                    } else {
                        SIZES[r % SIZES.len()]
                    };
                    let mut v = vec![0u8; size];
                    v[0] = 1;
                    ring.push(v);
                }
                done.store(true, Ordering::Release);
            })
        };
        let cons = {
            let ring = ring.clone();
            let done = done.clone();
            std::thread::spawn(move || {
                let mut n = 0u64;
                while let Some(v) = ring.pop(&done) {
                    n += v[0] as u64; // touch, then drop (free on this CPU)
                }
                n
            })
        };
        handles.push((prod, cons));
    }
    let mut freed = 0u64;
    for (p, c) in handles {
        p.join().unwrap();
        freed += c.join().unwrap();
    }
    let elapsed = start.elapsed();
    let total = per_producer * pairs;
    (elapsed.as_nanos() as f64 / total as f64, freed)
}

/// Clean producer/consumer attribution workload. `prodcons [pairs] [size] [millions]`.
///
/// `pairs`    — number of producer→consumer pairs (default min(nproc/2, 8)).
/// `size`     — fixed buffer size in bytes, or 0 to cycle through `SIZES` (default 0).
/// `millions` — buffers per producer, in millions (default 1).
///
/// Runs `TOCCATA_BENCH_REPS` trials (default 7) and reports min/median/max ns/op,
/// because a single trial on a busy 64-cpu box is too noisy to attribute a
/// single-digit-ns allocator change. Threads are unpinned (as jemalloc runs).
fn prodcons(allocator: &str, args: &[String]) {
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let pairs: usize = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or((nproc / 2).clamp(1, 8));
    let fixed_size: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let per_producer: usize = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        * 1_000_000;
    let reps: usize = std::env::var("TOCCATA_BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    let mut samples = Vec::with_capacity(reps);
    let mut freed = 0u64;
    for _ in 0..reps {
        let (per, f) = prodcons_trial(pairs, fixed_size, per_producer);
        samples.push(per);
        freed = f;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[reps / 2];
    let size_label = if fixed_size != 0 {
        format!("{fixed_size}B")
    } else {
        "mixed".to_string()
    };
    println!(
        "=== prodcons: {allocator} (x{pairs} pairs, {size_label}, {reps} reps) ===  \
         min {:>6.2}  median {:>6.2}  max {:>6.2} ns/op   (freed {freed})",
        samples[0],
        median,
        samples[reps - 1]
    );
}

// ---------------------------------------------------------------------------
// frag: external-fragmentation / budget-stranding comparison
// ---------------------------------------------------------------------------

/// Current process resident set size (bytes), read from `/proc/self/statm`
/// (field 2 = resident pages). Returns 0 off Linux or on read failure — the
/// `frag` workload is Linux-only in practice (so are rseq and the cgroup tests).
fn rss_now_bytes() -> usize {
    let s = match std::fs::read_to_string("/proc/self/statm") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let resident_pages: usize = s
        .split_whitespace()
        .nth(1)
        .and_then(|f| f.parse().ok())
        .unwrap_or(0);
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(0) as usize;
    resident_pages * page
}

/// Peak RSS ever reached (bytes), from `getrusage(RUSAGE_SELF).ru_maxrss`. On
/// Linux `ru_maxrss` is in KiB; this normalizes to bytes.
fn peak_rss_bytes() -> usize {
    let mut ru: libc::rusage = unsafe { core::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return 0;
    }
    (ru.ru_maxrss.max(0) as usize) * 1024
}

/// The footprint of the allocator under test, in bytes — the number a
/// fragmentation benchmark compares against the live ("useful") bytes.
///
/// Two regimes, because the allocators measure footprint differently:
/// * **toccata** `mlock`s and populates its *whole* budget at init, so process
///   RSS is pinned at the budget and says nothing about fragmentation. Its honest
///   footprint is the bump-arena high-water (`carved_bytes`): bytes ever claimed
///   from the arena, monotonic, never returned on free. This is the exact analog
///   of "pages an allocator has had to touch", which for the others *is* their RSS.
/// * **everyone else** (jemalloc/mimalloc/snmalloc/system): current process RSS.
///
/// Detection is structural: only the `toccata` binary calls `configure`, so
/// `toccata::stats()` is `Some` there and `None` for every other binary (which
/// link toccata for `FramePool` but never install it as the global allocator).
fn footprint_bytes() -> usize {
    match toccata::stats() {
        Some(s) => s.carved_bytes,
        None => rss_now_bytes(),
    }
}

/// Print which footprint metric this binary reports (toccata's carved-high-water
/// vs the others' process RSS), so every `frag` mode's output is self-documenting.
fn print_footprint_source() {
    if let Some(s) = toccata::stats() {
        println!(
            "  footprint source: toccata carved-high-water (RSS is pinned at the \
             {} MiB mlock'd budget)",
            s.budget_bytes / (1024 * 1024)
        );
    } else {
        println!("  footprint source: process RSS (/proc/self/statm)");
    }
}

/// Abort-safety guard for the comparative `frag` modes. toccata's global heap
/// runs `OnExhaust::Abort`, so an allocation that would exceed the budget aborts
/// the whole process (`oom_backpressure → std::process::abort()`) mid-benchmark
/// instead of letting us report. When running under toccata (`stats()` is
/// `Some`), this returns `true` — after printing a graceful stop notice — if
/// carving `about_to_alloc` more bytes would push the monotonic carved high-water
/// past 95% of the budget. For the other allocators `stats()` is `None`, so it
/// always returns `false`: they grow RSS until the OS (not an abort) intervenes,
/// which is exactly the behavior the benchmark wants to observe.
fn frag_budget_guard(about_to_alloc: usize, at: &str) -> bool {
    if let Some(s) = toccata::stats() {
        let projected = s.carved_bytes + about_to_alloc;
        let ceiling = s.budget_bytes / 100 * 95;
        if projected > ceiling {
            println!(
                "  [stopped at {at}] projected carved {:.0} MiB would exceed 95% of the \
                 {} MiB mlock'd budget;",
                projected as f64 / (1024.0 * 1024.0),
                s.budget_bytes / (1024 * 1024),
            );
            println!(
                "  raise TOCCATA_BENCH_MB to push further (toccata aborts on true exhaustion)."
            );
            return true;
        }
    }
    false
}

/// Touch one byte of every page (plus the last byte) so the kernel faults the
/// range in and RSS reflects real residency. `vec![0u8; n]` uses `alloc_zeroed`,
/// which can hand back lazily-zeroed pages that are not resident until written —
/// so the comparison allocators would under-report RSS without this. toccata has
/// already populated its whole budget, so this is a no-op for it (but harmless).
#[inline]
fn touch_pages(v: &mut [u8], tag: u8) {
    let mut off = 0;
    while off < v.len() {
        v[off] = tag;
        off += 4096;
    }
    if let Some(last) = v.last_mut() {
        *last = tag;
    }
}

/// A tiny deterministic xorshift64 PRNG (seeded, reproducible — no external crate
/// and no `Math.random`-style nondeterminism). Used only by the `mixed` mode.
#[inline]
fn xorshift64(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

/// Fragmentation / budget-stranding benchmark with several modes. toccata never
/// relocates memory and (for the comparison allocators) the metric is process RSS;
/// for toccata it is the bump-arena carved high-water (RSS is pinned by the
/// `mlock`'d budget). The reported `frag` factor = footprint / live makes each
/// allocator's stranding explicit and directly comparable.
///
/// `frag <mode> [args...]`
/// * `shift  [target_mib] [generations] [base_spans]` — large-object size *drift*
///   (adversarial): each generation's buffers grow by one span, a span-count
///   toccata's exact-count `LargeAllocator` recycler has never freed, so it
///   carves fresh every generation while a coalescing allocator stays ~1x.
/// * `match  [target_mib] [generations] [base_spans]` — control: stable large
///   sizes recycle exactly; toccata should hold ~1x. Proves the blow-up is the
///   size drift, not large allocations per se.
/// * `crossclass [target_mib] [settle_ms]` — small-object cross-size-class
///   stranding: free all of one size class, then allocate another. toccata tags
///   each span with its first class and never re-tags/returns it, so each freed
///   class strands its spans; coalescing allocators purge+remap.
/// * `mixed  [ops_millions] [live_mib] [seed]` — a realistic, non-adversarial,
///   seeded steady-state churn over a skewed size mix (small + large), holding a
///   bounded live set; reports the frag factor of "real traffic".
fn frag(allocator: &str, args: &[String]) {
    let mode = args.first().map(|s| s.as_str()).unwrap_or("shift");
    match mode {
        "shift" => frag_drift(allocator, args, true),
        "match" => frag_drift(allocator, args, false),
        "crossclass" => frag_crossclass(allocator, args),
        "mixed" => frag_mixed(allocator, args),
        other => {
            eprintln!(
                "frag: unknown mode {other:?}; expected `shift` | `match` | `crossclass` | `mixed`"
            );
            std::process::exit(2);
        }
    }
}

/// `shift`/`match` large-object drift workload (see [`frag`] for the mode docs).
/// Each generation frees its whole set, then allocates the next — `shift` drifts
/// the per-buffer span-count up each generation, `match` keeps it fixed.
fn frag_drift(allocator: &str, args: &[String], shift: bool) {
    let span = toccata::primitives::meta::SPAN_BYTES; // 64 KiB
    let mode = if shift { "shift" } else { "match" };
    let target_mib: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let generations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let base_spans: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let target = target_mib * 1024 * 1024;

    println!(
        "=== frag: {allocator} (mode={mode}, {target_mib} MiB/gen, {generations} gens, \
         base {base_spans} spans, span={} KiB) ===",
        span / 1024
    );
    print_footprint_source();

    // Baseline footprint after a warmup, so the reported numbers are deltas
    // attributable to this workload rather than the runtime's fixed overhead.
    let mut warm: Vec<Vec<u8>> = (0..16).map(|_| vec![0u8; base_spans * span]).collect();
    for v in warm.iter_mut() {
        v[0] = 1;
    }
    drop(warm);
    let base_footprint = footprint_bytes();

    println!("  gen   spans   live(MiB)   footprint(MiB)   frag(x)   reused");
    let mut held: Vec<Vec<u8>> = Vec::new();
    let mut peak_frag = 0.0f64;
    let mut last_live = 0usize;
    let mut last_footprint = 0usize;
    for g in 0..generations {
        // Free the previous generation entirely → its runs land on toccata's
        // per-span-count free stacks (or are madvise'd back by the others).
        let prev_footprint = footprint_bytes();
        held.clear();

        let spans = if shift { base_spans + g } else { base_spans };
        let bytes = spans * span;
        // Hold ~`target` bytes live this generation; shrink the count as the
        // per-buffer size grows so live stays ≈ constant across generations.
        let count = (target / bytes).max(1);
        // Don't let toccata's Abort policy kill the process under stranding.
        if frag_budget_guard(count * bytes, &format!("gen {g}")) {
            break;
        }
        held.reserve(count);
        for i in 0..count {
            let mut v = vec![0u8; bytes];
            touch_pages(&mut v, (i ^ g) as u8);
            held.push(v);
        }

        let live = count * bytes;
        let footprint = footprint_bytes().saturating_sub(base_footprint);
        let frag = footprint as f64 / live as f64;
        // "reused" = did this generation's allocations come from recycled space
        // rather than growing the footprint? True when the footprint did not climb.
        let reused = footprint_bytes() <= prev_footprint;
        peak_frag = peak_frag.max(frag);
        last_live = live;
        last_footprint = footprint;
        println!(
            "  {g:>3}   {spans:>5}   {:>9.1}   {:>14.1}   {frag:>7.2}   {}",
            live as f64 / (1024.0 * 1024.0),
            footprint as f64 / (1024.0 * 1024.0),
            if reused { "yes" } else { "no" },
        );
    }

    println!(
        "  ---\n  peak frag factor: {peak_frag:.2}x   (final live {:.0} MiB, footprint {:.0} MiB, peak RSS {:.0} MiB)",
        last_live as f64 / (1024.0 * 1024.0),
        last_footprint as f64 / (1024.0 * 1024.0),
        peak_rss_bytes() as f64 / (1024.0 * 1024.0),
    );
    if shift {
        println!(
            "  interpretation: a coalescing allocator holds ~1x here; toccata's exact-\n  \
             span-count large recycler cannot reuse drifting sizes, so carved climbs ~{generations}x.",
        );
    } else {
        println!(
            "  interpretation: stable large sizes recycle exactly — toccata should hold ~1x,\n  \
             matching the coalescing allocators (this is the control for the `shift` run).",
        );
    }
    // Keep the held set alive until here so the final measurement is valid, and
    // give jemalloc's background purge a beat before the process exits.
    drop(held);
    std::thread::sleep(Duration::from_millis(50));
}

/// `crossclass` — small-object cross-size-class stranding (see [`frag`]).
///
/// Walks a sequence of distinct small size classes. For each: free every object
/// of the previous class, then allocate ~`target` MiB of the new class. toccata
/// tags a span with the class it was first carved for and **never** re-tags it or
/// returns it to the arena ([`SpanTable::assign_range`], the bump cursor only
/// advances), so each freed class permanently strands its spans — carved climbs
/// by ~one class-worth per transition. jemalloc/mimalloc/snmalloc purge the freed
/// class and remap the pages for the next, so their RSS stays ~1x. The `settle_ms`
/// sleep before each footprint read lets the decay-based allocators run their
/// background purge so the comparison is fair.
fn frag_crossclass(allocator: &str, args: &[String]) {
    let target_mib: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let settle_ms: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let target = target_mib * 1024 * 1024;
    // Distinct small size classes (each well under the 256 KiB large threshold),
    // ~4x apart so each maps to its own toccata size class and carves its own
    // spans. A span carved for one can never serve another.
    const CLASSES: &[usize] = &[64, 256, 1024, 4096, 16384, 65536];

    println!(
        "=== frag: {allocator} (mode=crossclass, {target_mib} MiB/class, settle {settle_ms}ms, \
         classes {CLASSES:?}) ==="
    );
    print_footprint_source();

    // Pre-reserve the holding Vec for the worst case (the smallest class → most
    // objects) BEFORE the baseline, so its own backing store is a fixed overhead
    // folded into the baseline rather than a per-step confound.
    let smallest = CLASSES.iter().copied().min().unwrap();
    let max_count = (target / smallest).max(1);
    let mut held: Vec<Vec<u8>> = Vec::with_capacity(max_count);

    // Warm up + baseline on the first class.
    for _ in 0..1024 {
        let mut v = vec![0u8; CLASSES[0]];
        v[0] = 1;
        held.push(v);
    }
    held.clear();
    std::thread::sleep(Duration::from_millis(settle_ms));
    let base_footprint = footprint_bytes();

    println!("  step   size(B)   live(MiB)   footprint(MiB)   frag(x)   reused");
    let mut peak_frag = 0.0f64;
    for (step, &size) in CLASSES.iter().enumerate() {
        let prev_footprint = footprint_bytes();
        // Free every object of the previous class before allocating the new one.
        held.clear();

        let count = (target / size).max(1);
        if frag_budget_guard(count * size, &format!("step {step} ({size}B)")) {
            break;
        }
        for i in 0..count {
            let mut v = vec![0u8; size];
            v[0] = (i & 0xff) as u8;
            v[size - 1] = (step & 0xff) as u8;
            held.push(v);
        }
        // Let decay-based allocators purge the freed previous class.
        std::thread::sleep(Duration::from_millis(settle_ms));

        let live = count * size;
        let footprint = footprint_bytes().saturating_sub(base_footprint);
        let frag = footprint as f64 / live as f64;
        let reused = footprint_bytes() <= prev_footprint;
        peak_frag = peak_frag.max(frag);
        println!(
            "  {step:>4}   {size:>7}   {:>9.1}   {:>14.1}   {frag:>7.2}   {}",
            live as f64 / (1024.0 * 1024.0),
            footprint as f64 / (1024.0 * 1024.0),
            if reused { "yes" } else { "no" },
        );
    }
    println!(
        "  ---\n  peak frag factor: {peak_frag:.2}x   (peak RSS {:.0} MiB)",
        peak_rss_bytes() as f64 / (1024.0 * 1024.0),
    );
    println!(
        "  interpretation: toccata tags each span with its first class and never re-tags or\n  \
         returns it, so each freed class strands its spans — carved climbs ~1 class-worth per\n  \
         transition. Coalescing allocators purge+remap, so footprint should stay ~1x.",
    );
    drop(held);
    std::thread::sleep(Duration::from_millis(50));
}

/// `mixed` — realistic, non-adversarial, seeded steady-state churn (see [`frag`]).
///
/// A deterministic xorshift64 stream drives a skewed size distribution spanning
/// small and large classes (≈80% 16–512 B, ≈18% 1–16 KiB, ≈2% 256 KiB–1 MiB, the
/// large path). A bounded live set (`live_mib`) is maintained: each op evicts
/// random held buffers until the new one fits, then allocates and stores it —
/// steady-state alloc/free traffic like a long-running server. This produces a
/// credible "real traffic" frag factor to set against the adversarial `shift`
/// worst case. Reproducible: same `seed` ⇒ same run.
fn frag_mixed(allocator: &str, args: &[String]) {
    let ops_millions: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let live_mib: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let seed: u64 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .filter(|&s| s != 0)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let ops = ops_millions * 1_000_000;
    let live_budget = live_mib * 1024 * 1024;

    println!(
        "=== frag: {allocator} (mode=mixed, {ops_millions}M ops, {live_mib} MiB live set, \
         seed {seed:#x}) ==="
    );
    print_footprint_source();
    println!("  size mix: ~80% 16-512B, ~18% 1-16KiB, ~2% 256KiB-1MiB (large path)");

    let mut rng = seed;
    // Skewed size draw: cheap bucketed distribution over the PRNG stream.
    let draw_size = |rng: &mut u64| -> usize {
        let pct = xorshift64(rng) % 100;
        if pct < 80 {
            16 + (xorshift64(rng) as usize % (512 - 16))
        } else if pct < 98 {
            1024 + (xorshift64(rng) as usize % (16 * 1024 - 1024))
        } else {
            256 * 1024 + (xorshift64(rng) as usize % (1024 * 1024 - 256 * 1024))
        }
    };

    let base_footprint = footprint_bytes();
    let mut pool: Vec<Vec<u8>> = Vec::new();
    let mut live = 0usize;
    let mut peak_frag = 0.0f64;
    let mut peak_footprint = 0usize;
    let sample_every = (ops / 10).max(1);
    println!("  ops(M)   live(MiB)   footprint(MiB)   frag(x)");
    let mut stopped_early = false;
    for op in 0..ops {
        let size = draw_size(&mut rng);
        // Evict random held buffers until the newcomer fits the live budget.
        while live + size > live_budget && !pool.is_empty() {
            let victim = xorshift64(&mut rng) as usize % pool.len();
            live -= pool[victim].len();
            pool.swap_remove(victim); // drops → frees on this thread
        }
        // Guard toccata's Abort policy: stranding can push carved past budget
        // even though the live set is bounded.
        if frag_budget_guard(size, &format!("op {op}")) {
            stopped_early = true;
            break;
        }
        let mut v = vec![0u8; size];
        touch_pages(&mut v, (op & 0xff) as u8);
        live += size;
        pool.push(v);

        if op % sample_every == 0 {
            let footprint = footprint_bytes().saturating_sub(base_footprint);
            let frag = footprint as f64 / live.max(1) as f64;
            // Only fold a sample into the headline peak once the live set has
            // filled (≥ half the budget). The early samples ratio a real
            // footprint against a near-empty `live`, which is a startup artifact
            // (a meaningless 30x), not steady-state fragmentation.
            let warm = live * 2 >= live_budget;
            if warm {
                peak_frag = peak_frag.max(frag);
                peak_footprint = peak_footprint.max(footprint);
            }
            println!(
                "  {:>6}   {:>9.1}   {:>14.1}   {frag:>7.2}{}",
                op / 1_000_000,
                live as f64 / (1024.0 * 1024.0),
                footprint as f64 / (1024.0 * 1024.0),
                if warm { "" } else { "  (warmup)" },
            );
        }
    }
    let final_footprint = footprint_bytes().saturating_sub(base_footprint);
    peak_footprint = peak_footprint.max(final_footprint);
    peak_frag = peak_frag.max(final_footprint as f64 / live.max(1) as f64);
    println!(
        "  ---\n  peak frag factor: {peak_frag:.2}x   (final live {:.0} MiB, peak footprint {:.0} MiB, peak RSS {:.0} MiB){}",
        live as f64 / (1024.0 * 1024.0),
        peak_footprint as f64 / (1024.0 * 1024.0),
        peak_rss_bytes() as f64 / (1024.0 * 1024.0),
        if stopped_early { "  [stopped early]" } else { "" },
    );
    println!(
        "  interpretation: a bounded live set under skewed real-world sizes — toccata's frag\n  \
         here reflects how much the large-path size variety strands vs the coalescing allocators.",
    );
    drop(pool);
    std::thread::sleep(Duration::from_millis(50));
}

// ---------------------------------------------------------------------------
// micro hot-path loop (formerly bin/micro.rs)
// ---------------------------------------------------------------------------

fn micro(allocator: &str, args: &[String]) {
    let iters: u64 = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000_000);
    let mut sink = 0u64;
    let start = Instant::now();
    // Fixed 64-byte alloc/free, the cleanest hot path.
    for i in 0..iters {
        let b: Box<u64> = Box::new(i);
        sink = sink.wrapping_add(*b);
        core::hint::black_box(&sink);
        // `b` drops here (free).
    }
    let per = start.elapsed().as_nanos() as f64 / iters as f64;
    println!("=== micro: {allocator} === {iters} ops  =>  {per:.2} ns/op  (sink {sink})");
}

// ---------------------------------------------------------------------------
// framepool acceptance bar: dedicated FramePool vs the global Box path
// ---------------------------------------------------------------------------

/// The doc's acceptance bar (OQ#1): a dedicated fixed-size `FramePool` must BEAT
/// the general global-allocator path for a fixed-size alloc/free loop — otherwise
/// the primitive isn't worth exposing. Both loops do the same shape (acquire a
/// 64-byte slot, touch it, release); one through `Box` (the process allocator),
/// one through `FramePool`. The interesting comparison is under the `toccata`
/// binary, where the global path is toccata's own 9.7 ns `Box`.
fn framepool(allocator: &str, args: &[String]) {
    use toccata::{FramePool, Require, ReserveOpts};

    let iters: u64 = args
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000_000);
    const FRAME: usize = 64;
    const N: usize = 1 << 20; // 1Mi frames (64 MiB)

    println!("=== framepool vs global Box ({allocator}, frame={FRAME}B) ===");

    // Global Box<[u8;64]> path (whatever allocator this binary installed).
    {
        let mut sink = 0u64;
        let start = Instant::now();
        for i in 0..iters {
            let b: Box<[u8; FRAME]> = Box::new([0u8; FRAME]);
            sink = sink.wrapping_add(b[0] as u64).wrapping_add(i & 1);
            std::hint::black_box(&b);
        }
        let per = start.elapsed().as_nanos() as f64 / iters as f64;
        println!("  global Box<[u8;{FRAME}]>:  {per:>5.1} ns/op   (sink {sink})");
    }

    let opts = ReserveOpts::new(0).lock(Require::TRY);
    let pool: FramePool = FramePool::with_reservation(N, FRAME, opts).expect("frame pool");

    // Handle-free path: pool.alloc()/free() straight to the per-CPU rseq slab (no
    // L1 magazine). Expected to trail the global path, which has a magazine.
    {
        let mut sink = 0u64;
        let start = Instant::now();
        for i in 0..iters {
            let f = pool.alloc().expect("frame");
            unsafe { *f.as_ptr() = (i & 0xff) as u8 };
            sink = sink.wrapping_add(unsafe { *f.as_ptr() } as u64);
            std::hint::black_box(&f);
            unsafe { pool.free(f) };
        }
        let per = start.elapsed().as_nanos() as f64 / iters as f64;
        println!("  pool.alloc/free (no L1):  {per:>5.1} ns/op   (sink {sink})");
    }

    // L1 magazine path: pool.cache() — the fast path, the acceptance-bar number.
    {
        let mut cache = pool.cache();
        let mut sink = 0u64;
        let start = Instant::now();
        for i in 0..iters {
            let f = cache.alloc().expect("frame");
            unsafe { *f.as_ptr() = (i & 0xff) as u8 };
            sink = sink.wrapping_add(unsafe { *f.as_ptr() } as u64);
            std::hint::black_box(&f);
            unsafe { cache.free(f) };
        }
        let per = start.elapsed().as_nanos() as f64 / iters as f64;
        println!(
            "  cache.alloc/free (L1):    {per:>5.1} ns/op   (sink {sink})   <- acceptance bar"
        );
    }
}

/// toccata's pool budget in bytes, from `TOCCATA_BENCH_MB` (default 2 GiB). Read
/// by the `toccata` binary shim before `configure`. Lives here so all shims agree.
pub fn toccata_budget_bytes() -> usize {
    std::env::var("TOCCATA_BENCH_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048)
        * 1024
        * 1024
}
