// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Per-`SubHeap` live-usage query. Answers "memory
//! is at 90% — but **why**" for a sub-heap you hold a reference to.
//!
//! **Layer 1 (always on, near-zero cost):** live bytes/objects per sub-heap,
//! maintained by [`SubHeap`](crate::SubHeap)'s per-CPU padded counters (one
//! relaxed add on the alloc/free path, no global atomic). [`usage`] reads them
//! lock-free (relaxed loads), so it never perturbs the datapath.
//!
//! There is no fleet-wide `snapshot()` anymore: with the sub-heap registry
//! removed, toccata no longer enumerates every sub-heap. An application that
//! wants a fleet view holds the `&SubHeap`s it built and calls [`usage`] on each.

use crate::SubHeap;

/// Live usage of one sub-heap.
#[derive(Clone, Debug)]
pub struct SubHeapUsage {
    pub name: &'static str,
    pub live_bytes: usize,
    pub live_objects: u64,
    pub budget_bytes: usize,
}

impl SubHeapUsage {
    /// Fraction of the sub-heap's budget currently live (0.0..=1.0+).
    pub fn utilization(&self) -> f64 {
        if self.budget_bytes == 0 {
            0.0
        } else {
            self.live_bytes as f64 / self.budget_bytes as f64
        }
    }
}

/// Query a single sub-heap's live usage. Lock-free relaxed loads; safe and cheap
/// to call from the supervisor on a timer.
pub fn usage(sh: &SubHeap) -> SubHeapUsage {
    SubHeapUsage {
        name: sh.name(),
        live_bytes: sh.live_bytes(),
        live_objects: sh.live_objects(),
        budget_bytes: sh.budget_bytes(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sizeclass, SubHeapBuilder};

    #[test]
    fn usage_reports_live_then_zero() {
        let sh = SubHeapBuilder::new("metrics_test", 4 * 1024 * 1024)
            .num_cpus(4)
            .build_standalone()
            .expect("reserve");

        let class = sizeclass::class_for(1000).unwrap();
        let osz = sizeclass::size_of_class(class);
        let ptrs: Vec<_> = (0..100).filter_map(|_| sh.alloc_class(class)).collect();
        assert!(!ptrs.is_empty());

        let u = usage(&sh);
        assert_eq!(u.live_bytes, ptrs.len() * osz, "Layer-1 must report live bytes");
        assert_eq!(u.live_objects, ptrs.len() as u64);
        assert!(u.utilization() > 0.0 && u.utilization() < 1.0);

        for p in ptrs {
            unsafe { sh.dealloc_class(p, class) };
        }
        // After freeing, live objects drop back to zero.
        assert_eq!(usage(&sh).live_objects, 0);
    }
}
