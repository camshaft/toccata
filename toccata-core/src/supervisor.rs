// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! `toccata-supervisor` — the single background management thread (constraint 7).
//!
//! Callers never self-manage backpressure or reclaim; the supervisor does it off
//! the datapath. Its jobs:
//!
//! * **Reclaim stranded remote frees.** A cross-CPU free queues the object on
//!   its home CPU; if that CPU goes idle and never allocates again, the memory
//!   would be stranded. The supervisor periodically sweeps every sub-heap's
//!   remote queues into the central free lists (via
//!   [`SubHeap::reclaim_stranded_remote`]) so it's reusable by any CPU. This
//!   touches only mutex-guarded central lists — never a per-CPU rseq slab — so it
//!   needs neither to run on the owner CPU nor to seize it.
//!
//! * **(Future) capacity rebalancing.** Growing/shrinking a live per-CPU slab's
//!   class capacities requires mutating that CPU's slab while its owner might be
//!   mid-rseq-section. That needs the StopCpu + `MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ`
//!   seize protocol — exposed here as [`Supervisor::seize_cpu`] but not yet
//!   driven by an automatic policy.
//!
//! The supervisor never blocks a worker: it only takes central-list locks
//! (uncontended in steady state) and issues membarrier IPIs (rate-limited).

use crate::SubHeap;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

/// A registered sub-heap the supervisor manages. Held by `&'static` reference
/// because sub-heaps live for the process (registry-owned / leaked).
type Managed = &'static SubHeap;

/// How many fully-free spans the empty-span reclaim sweep may retire per central-
/// lock acquisition. Bounds worst-case lock-hold time so the (cold, uncontended)
/// central spin lock is never held long enough to perturb the latency tail; the
/// lock is DROPPED after each batch and `reclaim_empty_spans` re-acquires to drain
/// the rest of a cell, so a class with thousands of stranded empty spans (the
/// `frag crossclass` case) is fully reclaimed within one sweep without a long hold.
/// 32 spans = a handful of `is_fully_free` popcounts under the lock — microseconds.
const RECLAIM_CAP_PER_CELL: usize = 32;

/// Handle to the running supervisor. Dropping it signals the thread to stop and
/// joins it.
pub struct Supervisor {
    stop: Arc<AtomicBool>,
    stats: Arc<Stats>,
    handle: Option<std::thread::JoinHandle<()>>,
    caps: crate::rseq::MembarrierCaps,
}

/// Cumulative supervisor activity (for the metrics/dashboard layer).
#[derive(Default)]
pub struct Stats {
    pub sweeps: AtomicU64,
    pub reclaimed_objects: AtomicU64,
}

/// Builder for the supervisor.
pub struct SupervisorBuilder {
    subheaps: Vec<Managed>,
    interval: Duration,
}

impl SupervisorBuilder {
    pub fn new() -> Self {
        // 10ms default: fast enough that cross-class empty-span reclaim keeps up with
        // a workload that frees one class and immediately grows another (the `frag
        // crossclass` shape) — at 50ms the reclaim lagged the churn and footprint
        // stranded; 10ms tracks it while keeping the sweep's CPU cost negligible
        // (each sweep is a bounded walk of the active lists, lock dropped between
        // batches). Override via `interval` / `TOCCATA_SUPERVISOR_MS`.
        Self {
            subheaps: Vec::new(),
            interval: Duration::from_millis(10),
        }
    }

    /// Register a sub-heap to be swept for stranded remote frees.
    pub fn manage(mut self, sh: &'static SubHeap) -> Self {
        self.subheaps.push(sh);
        self
    }

    /// How often to sweep (default 50ms). The sweep is cheap (one swap per
    /// non-empty per-CPU queue) so this can be frequent.
    pub fn interval(mut self, d: Duration) -> Self {
        self.interval = d;
        self
    }

    /// Spawn the supervisor thread. Registers for membarrier (for the future
    /// seize path) once.
    pub fn spawn(self) -> Supervisor {
        let caps = crate::rseq::membarrier::register();
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::default());
        let subheaps = self.subheaps;
        let interval = self.interval;

        let handle = {
            let stop = stop.clone();
            let stats = stats.clone();
            std::thread::Builder::new()
                .name("toccata-supervisor".into())
                .spawn(move || {
                    // Adaptive cadence (jemalloc/mimalloc decay shape): when a sweep
                    // reclaims spans, stranding is actively happening (e.g. a workload
                    // freeing one size class and growing another) and the monotonic
                    // carved high-water would otherwise lock in a transition spike — so
                    // poll FAST to keep pace. When a sweep reclaims nothing (steady
                    // state), back off toward `interval` so the supervisor burns no CPU
                    // and never perturbs the tail. `FAST_MS` is the busy floor.
                    const FAST_MS: u64 = 1;
                    let idle = interval;
                    let fast = Duration::from_millis(FAST_MS).min(interval);
                    let mut nap = idle;
                    while !stop.load(Ordering::Relaxed) {
                        let mut reclaimed = 0u64;
                        for sh in &subheaps {
                            reclaimed += sh.reclaim_stranded_remote() as u64;
                            // Cross-class empty-span reclaim: return fully-free spans
                            // to the shared SpanPool so another class can reuse them
                            // (the `frag crossclass` fix). Bounded per (shard,class)
                            // cell so the central spin lock is never held long enough
                            // to perturb the latency tail; the lock is dropped between
                            // cells. Takes no kernel call and issues no membarrier, so
                            // it is safe on the never-stall path.
                            reclaimed += sh.reclaim_empty_spans(RECLAIM_CAP_PER_CELL) as u64;
                        }
                        stats.sweeps.fetch_add(1, Ordering::Relaxed);
                        stats
                            .reclaimed_objects
                            .fetch_add(reclaimed, Ordering::Relaxed);
                        // Reclaimed something -> stay hot; nothing -> relax to idle.
                        nap = if reclaimed > 0 { fast } else { idle };
                        std::thread::sleep(nap);
                    }
                    let _ = nap;
                    // Final sweep on shutdown so nothing is left stranded.
                    for sh in &subheaps {
                        sh.reclaim_stranded_remote();
                        sh.reclaim_empty_spans(RECLAIM_CAP_PER_CELL);
                    }
                })
                .expect("spawn supervisor thread")
        };

        Supervisor {
            stop,
            stats,
            handle: Some(handle),
            caps,
        }
    }
}

impl Default for SupervisorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Supervisor {
    pub fn builder() -> SupervisorBuilder {
        SupervisorBuilder::new()
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Whether the kernel supports the RSEQ-variant membarrier needed to safely
    /// seize a live multi-class slab. Without it, automatic capacity rebalancing
    /// must fall back to a no-seize mode.
    pub fn can_seize(&self) -> bool {
        self.caps.private_expedited_rseq
    }

    /// Seize CPU `cpu`'s slab in `sh` for safe direct mutation,
    /// returning a guard that releases it on drop. The full StopCpu protocol:
    ///
    /// 1. Set the per-CPU stop flag (seq_cst). New slow-path writers will see it
    ///    and route around the slab.
    /// 2. Issue `MEMBARRIER_CMD_PRIVATE_EXPEDITED_RSEQ` targeting `cpu`: this
    ///    aborts any writer currently in that CPU's rseq section and fences, so
    ///    after it returns no writer is mid-commit on that slab. The aborted
    ///    writer retries, falls to its locked slow path, sees the stop flag, and
    ///    routes to the central list instead of the slab.
    /// 3. The caller may now mutate `cpu`'s headers via `sh.rebalance_capacity`.
    ///
    /// Returns `None` if the RSEQ-variant membarrier is unavailable (no seize —
    /// caller must skip the rebalance rather than risk a race).
    pub fn seize_cpu<'s>(&self, sh: &'s SubHeap, cpu: u32) -> Option<SeizeGuard<'s>> {
        if !self.caps.private_expedited_rseq {
            return None;
        }
        sh.set_cpu_stopped(cpu, true);
        // Abort in-flight rseq sections on `cpu` and fence. If this somehow
        // fails, undo the flag and bail.
        if !crate::rseq::membarrier::expedited_rseq_cpu(cpu) {
            sh.set_cpu_stopped(cpu, false);
            return None;
        }
        Some(SeizeGuard { sh, cpu })
    }
}

/// Holds a CPU's slab seized; clears the stop flag on drop. While held, the
/// owner may call `sh.rebalance_capacity(cpu, ..)` safely.
pub struct SeizeGuard<'s> {
    sh: &'s SubHeap,
    cpu: u32,
}

impl SeizeGuard<'_> {
    pub fn cpu(&self) -> u32 {
        self.cpu
    }

    /// Move `count` free objects of `class` off the seized CPU's slab into the
    /// central list. Safe because the seize guarantees no concurrent writer.
    pub fn rebalance_capacity(&self, class: usize, count: u32) -> u32 {
        // SAFETY: we hold the seize (stop flag + rseq-abort membarrier issued).
        unsafe { self.sh.rebalance_capacity(self.cpu, class, count) }
    }
}

impl Drop for SeizeGuard<'_> {
    fn drop(&mut self) {
        self.sh.set_cpu_stopped(self.cpu, false);
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sizeclass, SubHeapBuilder};
    use std::ptr::NonNull;

    #[test]
    fn supervisor_reclaims_stranded_remote_frees() {
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4) as u32;
        let sh: &'static SubHeap = Box::leak(Box::new(
            SubHeapBuilder::new("sup", 8 * 1024 * 1024)
                .num_cpus(nproc)
                .cap_per_class(64)
                .build_standalone()
                .expect("reserve"),
        ));
        let class = sizeclass::class_for(256).unwrap();

        let sup = Supervisor::builder()
            .manage(sh)
            .interval(Duration::from_millis(5))
            .spawn();

        // Allocate here, free on another thread (queues to home CPUs), then let
        // the supervisor sweep — afterward the memory must be reusable.
        for _ in 0..10 {
            let ptrs: Vec<usize> = (0..400)
                .filter_map(|_| sh.alloc_class(class).map(|p| p.as_ptr() as usize))
                .collect();
            let h = std::thread::spawn(move || {
                for a in ptrs {
                    unsafe { sh.dealloc_by_ptr(NonNull::new(a as *mut u8).unwrap()) };
                }
            });
            h.join().unwrap();
            std::thread::sleep(Duration::from_millis(15)); // let the sweep run
        }

        // The supervisor should have run sweeps and reclaimed objects.
        assert!(
            sup.stats().sweeps.load(Ordering::Relaxed) > 0,
            "supervisor never swept"
        );
        // Memory is reusable (allocate a fresh batch).
        let v: Vec<_> = (0..400).filter_map(|_| sh.alloc_class(class)).collect();
        assert!(!v.is_empty());
        drop(sup);
    }

    #[test]
    fn supervisor_reclaims_empty_spans_cross_class() {
        // End-to-end: the background supervisor must return fully-free spans of one
        // class so a DIFFERENT class can reuse them — capping carved growth (the
        // `frag crossclass` fix) without corruption. Allocate class A across many
        // spans, free them all cross-thread (so the frees land in central as
        // fully-free, the reclaim trigger), let the supervisor sweep, then allocate
        // class B and assert carved did not climb by a whole second class-worth.
        let span = crate::meta::SPAN_BYTES;
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4) as u32;
        let sh: &'static SubHeap = Box::leak(Box::new(
            SubHeapBuilder::new("sup_xclass", 64 * span)
                .num_cpus(nproc)
                .cap_per_class(64)
                .build_standalone()
                .expect("reserve"),
        ));
        let class_a = sizeclass::class_for(1024).unwrap();
        let class_b = sizeclass::class_for(4096).unwrap();
        let osz_a = sizeclass::size_of_class(class_a);
        let osz_b = sizeclass::size_of_class(class_b);

        let sup = Supervisor::builder()
            .manage(sh)
            .interval(Duration::from_millis(2))
            .spawn();

        // Allocate ~6 spans of A, free them all on another thread.
        let a_count = (span / osz_a) * 6;
        let a: Vec<usize> = (0..a_count)
            .filter_map(|_| sh.alloc_class(class_a).map(|p| p.as_ptr() as usize))
            .collect();
        {
            let a2 = a.clone();
            std::thread::spawn(move || {
                for p in a2 {
                    unsafe { sh.dealloc_class(NonNull::new(p as *mut u8).unwrap(), class_a) };
                }
            })
            .join()
            .unwrap();
        }
        let carved_after_a = sh.carved_bytes();

        // Give the supervisor time to sweep the now-empty A spans into the pool.
        let mut swept = false;
        for _ in 0..200 {
            if sup.stats().sweeps.load(Ordering::Relaxed) > 1 {
                swept = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(swept, "supervisor never swept");
        std::thread::sleep(Duration::from_millis(20)); // a couple more sweeps

        // Now allocate ~6 spans of B. With reclaim, B reuses A's retired spans, so
        // carved must NOT grow by a whole ~6-span class-worth.
        let b_count = (span / osz_b) * 6;
        let b: Vec<usize> = (0..b_count)
            .filter_map(|_| {
                sh.alloc_class(class_b).map(|p| {
                    // Write the whole object — a reused-but-not-retagged span would tear.
                    unsafe { std::ptr::write_bytes(p.as_ptr(), 0x5B, osz_b) };
                    p.as_ptr() as usize
                })
            })
            .collect();
        assert!(!b.is_empty());
        let carved_after_b = sh.carved_bytes();
        // B should have largely reused A's spans: carved grows by far less than a
        // full 6-span class-worth (which would be the no-reclaim behavior).
        assert!(
            carved_after_b < carved_after_a + 6 * span,
            "carved climbed a full class-worth ({} -> {} spans); reclaim not reused",
            carved_after_a / span,
            carved_after_b / span
        );
        // Integrity: every B object intact.
        for &p in &b {
            unsafe { assert_eq!(*(p as *const u8), 0x5B, "B object {p:#x} corrupted") };
        }
        for p in b {
            unsafe { sh.dealloc_class(NonNull::new(p as *mut u8).unwrap(), class_b) };
        }
        drop(sup);
    }

    #[test]
    fn seize_and_rebalance_capacity() {
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4) as u32;
        let sh: &'static SubHeap = Box::leak(Box::new(
            SubHeapBuilder::new("seize", 8 * 1024 * 1024)
                .num_cpus(nproc)
                .cap_per_class(64)
                .build_standalone()
                .expect("reserve"),
        ));
        let class = sizeclass::class_for(128).unwrap();
        let sup = Supervisor::builder().spawn();

        if !sup.can_seize() {
            eprintln!("SKIP: PRIVATE_EXPEDITED_RSEQ unavailable on this kernel");
            return;
        }

        // Warm CPU 0's slab: alloc then free a batch so its stack has cached
        // objects (free returns them to the local stack).
        let ptrs: Vec<_> = (0..50).filter_map(|_| sh.alloc_class(class)).collect();
        for p in ptrs {
            unsafe { sh.dealloc_class(p, class) };
        }

        // Seize every CPU in turn and drain its cached capacity to central.
        let mut moved_total = 0u32;
        for cpu in 0..nproc {
            if let Some(g) = sup.seize_cpu(sh, cpu) {
                assert_eq!(g.cpu(), cpu);
                moved_total += g.rebalance_capacity(class, 1000);
                // guard drops here -> stop flag cleared
            }
        }
        eprintln!("seize+rebalance moved {moved_total} objects to central");

        // The sub-heap must still work after seize/release: alloc succeeds
        // (drawing the rebalanced objects back from central).
        let after: Vec<_> = (0..50).filter_map(|_| sh.alloc_class(class)).collect();
        assert!(!after.is_empty(), "alloc must work after seize/rebalance");
        for p in after {
            unsafe { sh.dealloc_class(p, class) };
        }
        drop(sup);
    }

    #[test]
    fn seize_under_concurrent_writers_is_safe() {
        // The real test: seize a CPU while many threads hammer alloc/free. The
        // seize must never corrupt the slab and writers must make progress.
        let nproc = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4) as u32;
        let sh: &'static SubHeap = Box::leak(Box::new(
            SubHeapBuilder::new("seize_race", 16 * 1024 * 1024)
                .num_cpus(nproc)
                .cap_per_class(128)
                .build_standalone()
                .expect("reserve"),
        ));
        let class = sizeclass::class_for(256).unwrap();
        let sup = Supervisor::builder().spawn();
        if !sup.can_seize() {
            eprintln!("SKIP: no rseq membarrier");
            return;
        }

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let workers: Vec<_> = (0..nproc * 2)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut held = Vec::with_capacity(16);
                    while !stop.load(Ordering::Relaxed) {
                        if held.len() < 16 {
                            if let Some(p) = sh.alloc_class(class) {
                                // Write the whole object to catch any tear.
                                unsafe { std::ptr::write_bytes(p.as_ptr(), 0x5A, 256) };
                                held.push(p);
                            }
                        } else {
                            let p = held.pop().unwrap();
                            unsafe {
                                assert_eq!(*p.as_ptr(), 0x5A, "object torn by a concurrent seize");
                                sh.dealloc_class(p, class);
                            }
                        }
                    }
                    for p in held {
                        unsafe { sh.dealloc_class(p, class) };
                    }
                })
            })
            .collect();

        // Hammer seize on all CPUs repeatedly while workers run.
        for _ in 0..200 {
            for cpu in 0..nproc {
                if let Some(g) = sup.seize_cpu(sh, cpu) {
                    g.rebalance_capacity(class, 8);
                }
            }
        }
        stop.store(true, Ordering::Relaxed);
        for w in workers {
            w.join().unwrap();
        }
        drop(sup);
    }
}
