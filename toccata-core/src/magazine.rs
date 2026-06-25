// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Thread-local L1 magazine cache — the layer that closes the single-thread
//! latency gap with jemalloc while keeping toccata's never-stall tail.
//!
//! The per-CPU RSEQ slab (L2) is migration-safe and bounds memory by core count,
//! but its commit prologue is ~12 instructions. jemalloc is fast single-threaded
//! because it reaches a per-*thread* cache with one TLS access and pops an array
//! slot (~5 instructions). This [`ThreadCache`] is exactly that: a small,
//! fixed-size per-(thread, size-class) magazine held inline in TLS (no heap
//! allocation for the cache itself). The hot path is a TLS read + array pop/push;
//! a miss/overflow refills/drains a batch from the L2 slab (amortized rseq cost).
//!
//! Memory cost: `MAG_CAP * NUM_CLASSES` pointers per (thread, sub-heap) — a few
//! KB, bounded, and flushed back to L2 on thread exit. This is the deliberate,
//! bounded reintroduction of per-thread memory that buys jemalloc-class speed.
//!
//! Accounting note: the per-op magazine pop/push do **no** budget accounting; the
//! batch refill/drain (which call `alloc_class`/`dealloc_class`) do. So the live
//! meter counts "objects in magazines or handed out" = objects out of the L2/
//! central pool — a consistent, conservative definition — and the hot path is
//! accounting-free.

use crate::{sizeclass::NUM_CLASSES, SubHeap};
use std::ptr::NonNull;

/// Prefetch the first line of `p` for **writing** (read-for-ownership). Never faults
/// (a hint); a no-op on non-aarch64.
#[inline(always)]
fn prefetch_for_write(p: *mut u8) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `prfm` is a side-effect-free hint that never faults, even on a null/bad
    // address; `p` is a live free-object pointer from the magazine.
    unsafe {
        core::arch::asm!("prfm pstl1keep, [{0}]", in(reg) p, options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = p;
}

/// Magazine capacity per size class. Larger = fewer L2 trips (faster) but more
/// per-thread memory. 32 matches `FramePool` and tcmalloc's small-class
/// `num_to_move`; it halves L2-trip frequency on both producer and consumer sides
/// of the prod/cons path vs 16, for a few KB more per (thread, sub-heap).
pub const MAG_CAP: usize = 32;

/// A per-(thread, class) stack of free object pointers.
struct Magazine {
    len: u32,
    ptrs: [*mut u8; MAG_CAP],
}

impl Magazine {
    const fn new() -> Self {
        Self {
            len: 0,
            ptrs: [core::ptr::null_mut(); MAG_CAP],
        }
    }
}

/// A thread's L1 cache for one sub-heap: one [`Magazine`] per size class, held
/// inline (const-initialized, zero heap allocation). Not `Sync` — it's accessed
/// only by its owning thread.
pub struct ThreadCache {
    mags: [Magazine; NUM_CLASSES],
}

// SAFETY: a ThreadCache is owned exclusively by one thread (held in that
// thread's thread-local `Tls`) and never accessed concurrently. The `*mut u8`
// slots are free-object pointers into the Send+Sync reservation. Marking it
// Send lets it live in the thread-local `Tls`; it is never actually moved across
// threads while in use — the dtor runs on the owning thread at exit.
unsafe impl Send for ThreadCache {}

impl Default for ThreadCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadCache {
    pub const fn new() -> Self {
        Self {
            mags: [const { Magazine::new() }; NUM_CLASSES],
        }
    }

    /// The pure L1 hot path: pop from the magazine for `class` with **no** other
    /// state touched — no `sub`, no configured-check, no atomic. Returns `None`
    /// on a magazine miss (the caller then resolves the L2 sub-heap and refills,
    /// which is also where the pre-configure "not ready" case is handled). This
    /// is what lets the global allocator's steady-state hot path be just
    /// `TLS -> try_pop` with no bootstrap check.
    #[inline]
    pub fn try_pop(&mut self, class: usize) -> Option<NonNull<u8>> {
        let m = &mut self.mags[class];
        if m.len > 0 {
            m.len -= 1;
            // SAFETY: a non-empty magazine slot holds a live free pointer.
            return Some(unsafe { NonNull::new_unchecked(m.ptrs[m.len as usize]) });
        }
        None
    }

    /// Allocate from the L1 magazine for `class`, refilling a batch from `sub`'s
    /// L2 slab on miss. Convenience for callers that always have `sub`.
    #[inline]
    pub fn alloc(&mut self, sub: &SubHeap, class: usize) -> Option<NonNull<u8>> {
        if let Some(p) = self.try_pop(class) {
            return Some(p);
        }
        self.refill(sub, class)
    }

    /// Like [`try_pop`](Self::try_pop) but, on a hit, also software-prefetches-for-
    /// write the buffer to be handed out `PREFETCH_AHEAD` pops from now. For **large**
    /// (multi-cache-line) classes only — the global allocator gates on request size
    /// and routes here out-of-line, so the small-object `try_pop` stays untouched.
    /// The producer drains its whole magazine before refilling, so the slot
    /// `PREFETCH_AHEAD` below the one returned now is the buffer it will `memset` in
    /// ~that many allocations: far enough that the read-for-ownership of its cold
    /// (cross-core / DRAM-evicted) first line overlaps the intervening work, close
    /// enough not to be evicted first. Software stand-in for the HW prefetch a PMU
    /// sweep showed jemalloc gets on this path and toccata didn't.
    #[inline]
    pub fn alloc_prefetch(&mut self, class: usize, osz: usize) -> Option<NonNull<u8>> {
        let m = &mut self.mags[class];
        if m.len > 0 {
            m.len -= 1;
            let top = m.len as usize;
            let p = m.ptrs[top];
            // Software-pipeline the cross-core buffer fetch across two distances. The
            // caller memsets the WHOLE object (24 lines at 1.5 KiB), all DRAM-cold on
            // the cross-thread path. Prefetching too many lines too far ahead just
            // gets them EVICTED by the intervening buffers' memset traffic before use
            // (measured: 512 B-ahead was worse than 256 B). So:
            //  - FAR (3 pops ahead): warm the first 2 lines only — long lead for the
            //    line the memset hits first, cheap enough to survive eviction.
            //  - NEAR (1 pop ahead): warm the next 4 lines — short lead, won't be
            //    evicted, covers the rest of the buffer's leading edge that the HW
            //    prefetcher then extends into a forward stream.
            if top >= 3 {
                let far = m.ptrs[top - 3];
                prefetch_for_write(far);
                if osz > 64 {
                    prefetch_for_write(far.wrapping_add(64));
                }
            }
            if top >= 1 {
                let near = m.ptrs[top - 1];
                let mut off = 0usize;
                while off < osz && off < 256 {
                    prefetch_for_write(near.wrapping_add(off));
                    off += 64;
                }
            }
            // SAFETY: a non-empty magazine slot holds a live free pointer.
            return Some(unsafe { NonNull::new_unchecked(p) });
        }
        None
    }

    /// Cold miss path: refill a batch and return one. Public so the global
    /// allocator can call it after a [`try_pop`](Self::try_pop) miss (once it has
    /// resolved the L2 sub-heap).
    ///
    /// Pulls a batch **directly from the home-shard central list** via
    /// [`SubHeap::refill_batch`], bypassing the per-CPU L2 slab. On the
    /// producer/consumer path this is the difference between `central → magazine`
    /// (one hop) and `central → L2 push → L2 pop → magazine` (an L2 round-trip per
    /// object) — the structural cost that kept producers above jemalloc. The L2
    /// slab still fronts same-thread alloc/free churn via the magazine's own
    /// drain/refill balance; it just isn't a mandatory intermediate on every miss.
    #[cold]
    pub fn refill(&mut self, sub: &SubHeap, class: usize) -> Option<NonNull<u8>> {
        // Pull a FULL magazine (not half) in one batched, lock-amortized hop. A
        // pure producer (alloc-only — the prod/cons producer) drains its magazine
        // to empty and refills; pulling MAG_CAP instead of MAG_CAP/2 halves how
        // often it touches the shared central lock, which profiling showed is the
        // contended atom on the small-object prod/cons path. The drain side still
        // flushes MAG_CAP/2, so a mixed alloc/free thread oscillates in the upper
        // half of the magazine rather than refilling from zero.
        let want = MAG_CAP;
        let m = &mut self.mags[class];
        let base = m.len as usize;
        // SAFETY: filling the contiguous free tail of the magazine array; `base +
        // want <= MAG_CAP` because refill is only called when the magazine missed
        // (empty for this class), so base is 0 here, and want = MAG_CAP/2.
        let got = unsafe { sub.refill_batch(class, &mut m.ptrs[base..base + want]) };
        // `refill_batch` fills the slice in ASCENDING address order, but `try_pop` is
        // LIFO (pops the top). Without this reverse the producer would hand out — and
        // the caller would memset — successive buffers at DESCENDING addresses, a
        // backward store stream the Neoverse L2 prefetcher trains on poorly (it locks
        // onto ascending runs). Reversing the just-filled slice makes LIFO pops climb
        // in address order, so a run of `vec![0u8; n]` is a clean forward stream the
        // HW prefetcher covers — the fix for toccata's residual cross-thread LLC-miss
        // gap to jemalloc at multi-cache-line sizes. Cold path, so no hot-path cost.
        m.ptrs[base..base + got].reverse();
        m.len += got as u32;
        if m.len > 0 {
            m.len -= 1;
            Some(unsafe { NonNull::new_unchecked(m.ptrs[m.len as usize]) })
        } else {
            None
        }
    }

    /// The pure L1 free hot path: push to the magazine for `class` with no other
    /// state touched. Returns `false` if the magazine is full (caller then
    /// drains to L2). Lets the global allocator free with `TLS -> try_push` and
    /// no per-op routing check.
    ///
    /// # Safety
    /// `ptr` is a live object of size class `class`.
    #[inline]
    pub unsafe fn try_push(&mut self, ptr: NonNull<u8>, class: usize) -> bool {
        let m = &mut self.mags[class];
        if (m.len as usize) < MAG_CAP {
            m.ptrs[m.len as usize] = ptr.as_ptr();
            m.len += 1;
            return true;
        }
        false
    }

    /// Free into the L1 magazine for `class`, draining a batch to L2 on overflow.
    ///
    /// # Safety
    /// `ptr` came from `sub` and is of size class `class`.
    #[inline]
    pub unsafe fn free(&mut self, sub: &SubHeap, ptr: NonNull<u8>, class: usize) {
        if self.try_push(ptr, class) {
            return;
        }
        self.drain_then_push(sub, ptr, class)
    }

    #[cold]
    unsafe fn drain_then_push(&mut self, sub: &SubHeap, ptr: NonNull<u8>, class: usize) {
        // Flush half the magazine back to L2 to make room, in ONE batched call so
        // cross-CPU frees (the producer/consumer case: a consumer's magazine full
        // of buffers a producer allocated) return to the owner in a single CAS per
        // home CPU rather than one-CAS-per-object. The top `drop_n` slots are the
        // oldest; draining them keeps the hot (most-recently-freed) objects in L1.
        let drop_n = MAG_CAP * 3 / 4;
        let m = &mut self.mags[class];
        let new_len = m.len as usize - drop_n;
        // The batch is the bottom `drop_n` slots; keep the top `new_len` in place.
        sub.dealloc_class_batch(&m.ptrs[..drop_n], class);
        // Compact the survivors down to the base of the array.
        m.ptrs.copy_within(drop_n..m.len as usize, 0);
        m.len = new_len as u32;
        m.ptrs[m.len as usize] = ptr.as_ptr();
        m.len += 1;
    }

    /// Flush every magazine back to L2. Called on thread exit so cached memory
    /// returns to the pool rather than leaking with the thread.
    pub fn flush_all(&mut self, sub: &SubHeap) {
        for class in 0..NUM_CLASSES {
            let m = &mut self.mags[class];
            while m.len > 0 {
                m.len -= 1;
                let p = m.ptrs[m.len as usize];
                // SAFETY: live free pointer of this class.
                unsafe { sub.dealloc_class(NonNull::new_unchecked(p), class) };
            }
        }
    }
}
