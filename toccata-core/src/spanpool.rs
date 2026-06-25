// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! `SpanPool` — an address-coalescing free-run index over the span grid.
//!
//! The large allocator ([`crate::subheap::LargeAllocator`]) carves whole-span
//! runs from the bump arena and, on free, must be able to **recycle** them for
//! *any* span-count — not only the exact count originally requested. The old
//! design kept one intrusive free stack per exact span-count and never merged
//! adjacent runs, so a freed N-span run could satisfy only an N-span request and
//! the bump cursor marched forward forever under size drift (`frag shift` →
//! 5.99×). `SpanPool` replaces that with a segregated-fit free-run index that
//! **coalesces address-adjacent free runs** and **splits** a larger run to serve
//! a smaller request, all in bounded time.
//!
//! ## Why span-count keyed, not byte/TLSF-rounded
//!
//! Buckets are indexed by the **exact** span-count (`bucket c` holds every free
//! run of exactly `c` spans). A hierarchical bitmap finds the lowest non-empty
//! bucket `≥ n` in bounded time, so an *exact* request (`bucket[n]` non-empty)
//! is served from bucket `n` with **no split** — this is what preserves
//! toccata's best-in-class `frag match` (0.98×): a stable large size recycles
//! byte-for-byte as before. Only a *drifting* size falls through to the next
//! larger bucket and splits. There is **no power-of-two allocation rounding**
//! (the reason a buddy allocator was rejected), so no internal-fragmentation
//! regression.
//!
//! ## No syscalls, no freed-object reads
//!
//! `SpanPool` operates purely on **span indices** (`0..num_spans`) and integer
//! counts. All metadata lives in out-of-band [`crate::SysBoxSlice`] arrays
//! allocated once at sub-heap build time (system allocator, never toccata), so
//! a free/alloc never touches the kernel and never reads or writes a freed run's
//! own (resident, possibly cold) memory. The caller serializes access (the
//! existing `LargeAllocator` mutex).
//!
//! ## Coalescing via boundary tags
//!
//! Each free run records its count at both its **head** span and its **tail**
//! span (`head_count` / `tail_count`), plus a doubly-linked stack per bucket
//! (`next`/`prev`, keyed at the head) for O(1) unlink from the middle when a
//! neighbor is consumed by a merge. On free of `[H, H+C)`: the right neighbor is
//! a free-run head at `H+C`; the left neighbor is a free-run tail at `H-1` (whose
//! head is `H-1 - tail_count + 1`). Each side merges at most once (we always
//! coalesce on insert, so no two free runs are ever adjacent), bounding the work.

use crate::meta::NO_SPAN;
use crate::SysBoxSlice;

/// A segregated-fit, address-coalescing index of free span-runs over
/// `[0, num_spans)`. Single-threaded; the caller holds the large-allocator lock.
pub struct SpanPool {
    num_spans: u32,
    /// `bucket_head[c]` = head span index of the free-run stack for runs of
    /// exactly `c` spans (`NO_SPAN` = empty). Length `num_spans + 1`; index 0
    /// unused (a run has ≥ 1 span).
    bucket_head: SysBoxSlice<u32>,
    /// Per-span (at a free-run **head**): next/prev run in the same bucket's
    /// stack (`NO_SPAN` terminator). Doubly linked so a coalesce can unlink a
    /// neighbor that is not the stack head in O(1).
    next: SysBoxSlice<u32>,
    prev: SysBoxSlice<u32>,
    /// Boundary tags: count stored at a free run's head span (`head_count`) and
    /// tail span (`tail_count`); 0 means "not a free-run head/tail". A 1-span run
    /// sets both at the same index.
    head_count: SysBoxSlice<u32>,
    tail_count: SysBoxSlice<u32>,
    /// Two-level occupancy bitmap over bucket indices `0..=num_spans`: `l0` bit
    /// `c` is set iff `bucket_head[c] != NO_SPAN`; `l1` bit `i` is set iff
    /// `l0[i] != 0`. Lets [`find_first_set_ge`](Self::find_first_set_ge) locate
    /// the lowest non-empty bucket `≥ n` in bounded time.
    l0: SysBoxSlice<u64>,
    l1: SysBoxSlice<u64>,
}

impl SpanPool {
    /// Build an empty pool covering `num_spans` spans. All metadata is
    /// system-allocated once here and never grows.
    pub fn new(num_spans: usize) -> Self {
        let n = num_spans.max(1);
        let nbuckets = n + 1; // counts 1..=n
        let l0_words = nbuckets.div_ceil(64).max(1);
        let l1_words = l0_words.div_ceil(64).max(1);
        Self {
            num_spans: n as u32,
            bucket_head: crate::sys_boxed_slice(nbuckets, |_| NO_SPAN),
            next: crate::sys_boxed_slice(n, |_| NO_SPAN),
            prev: crate::sys_boxed_slice(n, |_| NO_SPAN),
            head_count: crate::sys_boxed_slice(n, |_| 0u32),
            tail_count: crate::sys_boxed_slice(n, |_| 0u32),
            l0: crate::sys_boxed_slice(l0_words, |_| 0u64),
            l1: crate::sys_boxed_slice(l1_words, |_| 0u64),
        }
    }

    // --- occupancy bitmap -------------------------------------------------

    #[inline]
    fn bit_set(&mut self, c: u32) {
        let w = (c / 64) as usize;
        self.l0[w] |= 1u64 << (c % 64);
        self.l1[w / 64] |= 1u64 << (w % 64);
    }

    #[inline]
    fn bit_clear(&mut self, c: u32) {
        let w = (c / 64) as usize;
        self.l0[w] &= !(1u64 << (c % 64));
        if self.l0[w] == 0 {
            self.l1[w / 64] &= !(1u64 << (w % 64));
        }
    }

    /// Lowest bucket index `≥ c` whose bit is set, or `None`. Bounded: at most
    /// one masked read per level plus a short scan of `l1` words.
    #[inline]
    fn find_first_set_ge(&self, c: u32) -> Option<u32> {
        if c as usize >= self.l0.len() * 64 {
            return None;
        }
        let w = (c / 64) as usize;
        let b = c % 64;
        // Bits ≥ b in the starting l0 word.
        let masked = self.l0[w] & (u64::MAX << b);
        if masked != 0 {
            return Some((w as u32) * 64 + masked.trailing_zeros());
        }
        // Find the next non-empty l0 word at an index > w, using l1 as summary.
        let lw = w / 64;
        // Remaining l1 bits above position (w % 64) within l1[lw].
        let wb = (w % 64) as u32;
        if wb < 63 {
            let m = self.l1[lw] & (u64::MAX << (wb + 1));
            if m != 0 {
                let nw = lw * 64 + m.trailing_zeros() as usize;
                return Some((nw as u32) * 64 + self.l0[nw].trailing_zeros());
            }
        }
        for li in (lw + 1)..self.l1.len() {
            if self.l1[li] != 0 {
                let nw = li * 64 + self.l1[li].trailing_zeros() as usize;
                return Some((nw as u32) * 64 + self.l0[nw].trailing_zeros());
            }
        }
        None
    }

    // --- bucket stack (doubly linked, keyed at run head) ------------------

    /// Link free run `head` (count `count`) onto the front of its bucket stack
    /// and stamp its boundary tags + occupancy bit.
    #[inline]
    fn link(&mut self, head: u32, count: u32) {
        debug_assert!(count >= 1 && head + count <= self.num_spans);
        let h = head as usize;
        let old = self.bucket_head[count as usize];
        self.next[h] = old;
        self.prev[h] = NO_SPAN;
        if old != NO_SPAN {
            self.prev[old as usize] = head;
        }
        self.bucket_head[count as usize] = head;
        self.head_count[h] = count;
        self.tail_count[(head + count - 1) as usize] = count;
        self.bit_set(count);
    }

    /// Unlink free run `head` (count `count`) from its bucket stack and clear its
    /// boundary tags (+ occupancy bit if the bucket empties).
    #[inline]
    fn unlink(&mut self, head: u32, count: u32) {
        let h = head as usize;
        let p = self.prev[h];
        let n = self.next[h];
        if p != NO_SPAN {
            self.next[p as usize] = n;
        } else {
            // `head` was the stack head.
            self.bucket_head[count as usize] = n;
            if n == NO_SPAN {
                self.bit_clear(count);
            }
        }
        if n != NO_SPAN {
            self.prev[n as usize] = p;
        }
        self.head_count[h] = 0;
        self.tail_count[(head + count - 1) as usize] = 0;
        self.next[h] = NO_SPAN;
        self.prev[h] = NO_SPAN;
    }

    // --- public API -------------------------------------------------------

    /// Return a free run of `n` spans (its head span index), or `None` if no free
    /// run is large enough (the caller then carves fresh from the arena).
    ///
    /// An exact-count bucket is found first (no split); otherwise the lowest
    /// larger bucket is split and the tail remainder re-inserted as its own free
    /// run. O(levels) — bounded, on the cold large path only.
    pub fn alloc(&mut self, n: u32) -> Option<u32> {
        if n == 0 || n > self.num_spans {
            return None;
        }
        let m = self.find_first_set_ge(n)?;
        let head = self.bucket_head[m as usize];
        debug_assert!(head != NO_SPAN);
        self.unlink(head, m);
        if m > n {
            // Split: keep [head, head+n), return [head+n, head+m) to the pool.
            self.link(head + n, m - n);
        }
        Some(head)
    }

    /// Return run `[head, head+count)` to the pool, coalescing with any
    /// address-adjacent free runs (≤ 1 on each side) before filing it.
    pub fn free(&mut self, head: u32, count: u32) {
        debug_assert!(count >= 1 && head + count <= self.num_spans);
        let mut h = head;
        let mut c = count;

        // Right neighbor: a free-run head exactly at the end of this run.
        let right = h + c;
        if right < self.num_spans {
            let rc = self.head_count[right as usize];
            if rc != 0 {
                self.unlink(right, rc);
                c += rc;
            }
        }
        // Left neighbor: a free-run tail exactly one span before this run.
        if h > 0 {
            let lc = self.tail_count[(h - 1) as usize];
            if lc != 0 {
                let lhead = h - lc;
                self.unlink(lhead, lc);
                h = lhead;
                c += lc;
            }
        }
        self.link(h, c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn exact_match_does_not_split() {
        let mut p = SpanPool::new(100);
        p.free(0, 5);
        // Exact request returns the whole run, no remainder re-inserted.
        assert_eq!(p.alloc(5), Some(0));
        assert_eq!(p.alloc(1), None, "pool should be empty after exact take");
    }

    #[test]
    fn inexact_splits_tail() {
        let mut p = SpanPool::new(100);
        p.free(10, 8);
        // Request 3 of an 8-run: get the head, remainder [13,18) (5 spans) stays.
        assert_eq!(p.alloc(3), Some(10));
        assert_eq!(p.alloc(5), Some(13), "remainder must be reusable");
        assert_eq!(p.alloc(1), None);
    }

    #[test]
    fn coalesce_right_left_and_both() {
        let mut p = SpanPool::new(100);
        // Two adjacent runs coalesce into one on the second free.
        p.free(0, 4);
        p.free(4, 4);
        assert_eq!(p.alloc(8), Some(0), "adjacent frees must merge to 8");

        // Free middle, then left, then right — both-sides coalesce.
        p.free(20, 2); // middle
        p.free(18, 2); // left-adjacent
        p.free(22, 2); // right-adjacent -> all merge to [18,24)
        assert_eq!(p.alloc(6), Some(18), "three adjacent frees merge to 6");
    }

    #[test]
    fn no_coalesce_across_gap() {
        let mut p = SpanPool::new(100);
        p.free(0, 2);
        p.free(3, 2); // gap at span 2
        assert_eq!(p.alloc(4), None, "non-adjacent runs must not merge");
        assert_eq!(p.alloc(2), Some(3), "LIFO: most recent first");
        assert_eq!(p.alloc(2), Some(0));
    }

    #[test]
    fn find_first_set_spans_word_boundary() {
        // Force buckets past a 64-count word boundary to exercise the l1 scan.
        let mut p = SpanPool::new(500);
        p.free(0, 200); // bucket 200 (l0 word 3)
        assert_eq!(p.find_first_set_ge(1), Some(200));
        assert_eq!(p.find_first_set_ge(200), Some(200));
        assert_eq!(p.find_first_set_ge(201), None);
        // Add a smaller bucket in an earlier word.
        p.free(300, 5); // bucket 5 (l0 word 0)
        assert_eq!(p.find_first_set_ge(1), Some(5));
        assert_eq!(p.find_first_set_ge(6), Some(200));
    }

    /// The load-bearing invariant: across a long random alloc/free stream, every
    /// span index is in exactly one of {free, live} — no double hand-out, no
    /// lost span, and coalescing/splitting never overlaps runs.
    #[test]
    fn double_handout_invariant_random_stress() {
        let num_spans = 256u32;
        let mut p = SpanPool::new(num_spans as usize);
        // Seed: the whole arena is one free run.
        p.free(0, num_spans);

        // live: head -> count. Deterministic xorshift so failures reproduce.
        let mut live: BTreeMap<u32, u32> = BTreeMap::new();
        let mut rng: u64 = 0x1234_5678_9abc_def1;
        let step = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };

        for _ in 0..200_000 {
            let do_alloc = live.is_empty() || (step(&mut rng) & 1 == 0);
            if do_alloc {
                let n = 1 + (step(&mut rng) % 12) as u32;
                if let Some(head) = p.alloc(n) {
                    // Must not overlap any live run.
                    assert!(head + n <= num_spans, "run out of bounds");
                    for (&lh, &lc) in &live {
                        let overlap = head < lh + lc && lh < head + n;
                        assert!(!overlap, "double hand-out: [{head},{}) vs [{lh},{})", head + n, lh + lc);
                    }
                    live.insert(head, n);
                }
            } else {
                // Free a random live run.
                let k = (step(&mut rng) as usize) % live.len();
                let &head = live.keys().nth(k).unwrap();
                let count = live.remove(&head).unwrap();
                p.free(head, count);
            }
        }

        // Drain everything live, then the entire arena must reform as one run.
        let live_total: u32 = live.values().sum();
        for (head, count) in std::mem::take(&mut live) {
            p.free(head, count);
        }
        // The whole arena is free again -> a single num_spans run is allocatable.
        assert_eq!(p.alloc(num_spans), Some(0), "full coalesce back to one run failed (live was {live_total})");
    }
}
