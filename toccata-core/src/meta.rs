// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Span metadata: recover `(size class, home CPU)` from any object pointer.
//!
//! The object arena is carved into fixed-size **spans**, each dedicated to one
//! size class and "homed" on the CPU that first carved it. A dense single-level
//! table maps `span_index = (ptr - arena_base) >> SPAN_BITS` to a [`SpanMeta`].
//! Recovering metadata from a pointer is one shift + one indexed load — the
//! design's `slab_meta` (§5.4), without snmalloc's multi-level radix.
//!
//! This is what lets `dealloc(ptr)` work without a `Layout` (the global
//! allocator / Box / Vec case) and lets a cross-CPU free route back to the
//! object's home CPU.

use core::sync::atomic::{AtomicU32, Ordering};

/// log2 of the span size in bytes. 64 KiB spans: large enough that the table is
/// small (one `SpanMeta` per 64 KiB) yet fine enough that even the largest
/// small-class objects (256 KiB) span only a handful of entries.
pub const SPAN_BITS: u32 = 16;
pub const SPAN_BYTES: usize = 1 << SPAN_BITS;

/// Free-slot bits per span and `u64` words to hold them. A span holds at most
/// `SPAN_BYTES / MIN_ALIGN` (= 65536/8 = 8192) min-size objects, so 8192 bits =
/// 128 `u64` words covers any class.
pub const SPAN_SLOTS: usize = SPAN_BYTES / 8;
pub const SPAN_BITMAP_WORDS: usize = SPAN_SLOTS / 64;

/// Out-of-band per-span free bitmaps — the jemalloc/tcmalloc free-slot model that
/// avoids the intrusive free-list's cold cross-core cache-miss chain on
/// producer/consumer hand-off. A cross-thread free SETS the bit for its slot (slot
/// = `(ptr - span_base) / osz`, pure arithmetic, no read of the freed object); a
/// refill SCANS a span's set bits and computes each object's address
/// (`span_base + slot*osz`, again no read), handing them back in ascending-address
/// order (good reuse locality). One bitmap per span; a span belongs to exactly one
/// `(shard, class)` (the carve assigns it), so no per-(shard,class) duplication.
///
/// `next_active` threads the spans that currently hold ≥1 free object into a
/// per-`(shard,class)` singly-linked list (by span index), so a refill knows which
/// spans to scan without walking the whole arena. `NO_SPAN` = list terminator. A
/// parallel `on_list` byte records membership (so the carve/deposit can tell
/// "already linked?" without it being ambiguous with the terminator).
pub struct SpanBitmaps {
    base: usize,
    /// **Producer-private** free bits (1 = a free slot the owner can hand out),
    /// `num_spans * SPAN_BITMAP_WORDS`. Touched **only under the owning (shard,class)
    /// central lock**, so it is plain (`UnsafeCell<u64>`, non-atomic): the producer
    /// scans + clears it with ordinary loads/stores — no atomic RMW, no acquire
    /// fence on the hot refill. In steady state the producer re-hands-out slots it
    /// already holds here without touching the consumer's line at all.
    alloc_words: crate::SysBoxSlice<core::cell::UnsafeCell<u64>>,
    /// **Consumer-written** recently-freed bits (1 = freed cross-thread, not yet
    /// folded into `alloc_words`). A cross-thread free deposits with a lock-free
    /// `fetch_or` (no central lock, no object read). The owner *folds* this into
    /// `alloc_words` at the start of a refill (`swap(0)` per non-empty word, then a
    /// plain OR), so a deposit landing after the swap simply shows up on the next
    /// refill (no lost free). Keeping the consumer's write set (`free_words`) on a
    /// **different cache line** from the producer's hand-out set (`alloc_words`) is
    /// what stops the producer↔consumer line ping-pong that kept the cross-thread
    /// latency above jemalloc — the two cores no longer fight over one bitmap word.
    free_words: crate::SysBoxSlice<core::sync::atomic::AtomicU64>,
    /// Per-span active-list link (span index, or `NO_SPAN`). Written only under the
    /// owning central lock (at carve/link time).
    next_active: crate::SysBoxSlice<core::cell::UnsafeCell<u32>>,
    /// Per-span "currently on its (shard,class) active list" flag. Written under
    /// the central lock; read relaxed by the lock-free deposit.
    on_list: crate::SysBoxSlice<core::sync::atomic::AtomicBool>,
}

/// Active-list terminator.
pub const NO_SPAN: u32 = u32::MAX;

// SAFETY: every access is mediated by the owning (shard,class) central spin lock;
// a span's bitmap/link is only touched by that lock's holder.
unsafe impl Send for SpanBitmaps {}
unsafe impl Sync for SpanBitmaps {}

impl SpanBitmaps {
    pub fn new(base: *mut u8, len: usize) -> Self {
        let num_spans = len.div_ceil(SPAN_BYTES);
        Self {
            base: base as usize,
            alloc_words: crate::sys_boxed_slice(num_spans * SPAN_BITMAP_WORDS, |_| {
                core::cell::UnsafeCell::new(0)
            }),
            free_words: crate::sys_boxed_slice(num_spans * SPAN_BITMAP_WORDS, |_| {
                core::sync::atomic::AtomicU64::new(0)
            }),
            next_active: crate::sys_boxed_slice(num_spans, |_| {
                core::cell::UnsafeCell::new(NO_SPAN)
            }),
            on_list: crate::sys_boxed_slice(num_spans, |_| {
                core::sync::atomic::AtomicBool::new(false)
            }),
        }
    }

    /// Span index containing `ptr` (no bounds check — caller guarantees in-arena).
    #[inline]
    pub fn span_of(&self, ptr: *const u8) -> u32 {
        ((ptr as usize - self.base) >> SPAN_BITS) as u32
    }

    /// Base address of span `idx`.
    #[inline]
    pub fn span_base(&self, idx: u32) -> usize {
        self.base + ((idx as usize) << SPAN_BITS)
    }

    /// Locate object `ptr` of size `osz` as `(span, word, bit)`: its span index, the
    /// bitmap word holding its slot, and the bit within that word. Pure arithmetic,
    /// no object read. Lets a batched free group objects that share a `(span, word)`
    /// and deposit their bits with one masked `fetch_or` (see [`set_mask`](Self::set_mask)).
    #[inline]
    pub fn locate(&self, ptr: *const u8, osz: usize) -> (u32, usize, u32) {
        let span = self.span_of(ptr);
        let slot = (ptr as usize - self.span_base(span)) / osz;
        (span, slot / 64, (slot % 64) as u32)
    }

    /// Whether span `idx` is currently on its central's active list. Relaxed read
    /// (the lock-free deposit consults it; correctness doesn't depend on freshness
    /// because spans stay linked once carved — see `set`).
    #[inline]
    pub fn is_on_list(&self, idx: u32) -> bool {
        self.on_list[idx as usize].load(Ordering::Relaxed)
    }

    /// The next active span after `idx` (or `NO_SPAN`).
    ///
    /// # Safety
    /// Caller holds the owning central lock.
    #[inline]
    pub unsafe fn next(&self, idx: u32) -> u32 {
        *self.next_active[idx as usize].get()
    }

    /// Push span `idx` onto the front of an active list whose current head is
    /// `head`; returns the new head (`idx`). Marks `idx` on-list.
    ///
    /// # Safety
    /// Caller holds the owning central lock; `idx` not already on a list.
    #[inline]
    pub unsafe fn link_front(&self, idx: u32, head: u32) -> u32 {
        *self.next_active[idx as usize].get() = head;
        self.on_list[idx as usize].store(true, Ordering::Relaxed);
        idx
    }

    /// **Lock-free** cross-thread deposit: set the free bit for `slot` of span `idx`
    /// in the **consumer-written** `free_words` with one `fetch_or`. No central lock,
    /// no object read. The owner folds this into its private `alloc_words` on its next
    /// refill ([`fold_free_word`](Self::fold_free_word)); a `fetch_or` landing after
    /// that fold simply shows up on the following refill (no lost free). Spans stay
    /// linked from carve onward so a deposit never (re)links. `Release` so the freed
    /// object's contents are visible before the producer can re-hand-out the slot.
    #[inline]
    pub fn set(&self, idx: u32, slot: u32) {
        let word = &self.free_words[(idx as usize) * SPAN_BITMAP_WORDS + (slot as usize / 64)];
        word.fetch_or(1u64 << (slot % 64), Ordering::Release);
    }

    /// Like [`set`](Self::set) but deposits a whole **mask** of free bits into one
    /// `free_words` word with a single `fetch_or`. A batched cross-thread free (a
    /// magazine flush) hands objects back in near-contiguous address order, so most
    /// of a 32-object batch falls in one or two bitmap words; OR-ing their bits and
    /// depositing per-word turns ~32 atomic RMWs into ~2. `word` is the word index in
    /// `0..SPAN_BITMAP_WORDS`; `Release` as in [`set`](Self::set).
    #[inline]
    pub fn set_mask(&self, idx: u32, word: usize, mask: u64) {
        self.free_words[(idx as usize) * SPAN_BITMAP_WORDS + word]
            .fetch_or(mask, Ordering::Release);
    }

    /// Fold any consumer-deposited free bits for word `w` of span `idx` from
    /// `free_words` into the producer-private `alloc_words`, and return the resulting
    /// `alloc_words` word (the bits the owner may now hand out). One `swap(0,
    /// Acquire)` drains the consumer's deposits (the Acquire pairs with the deposit's
    /// Release); the producer then ORs them into its private word with a plain store.
    /// After this the producer reads/clears `alloc_words[w]` with no atomics at all.
    /// In steady state, when the consumer hasn't deposited into this word since the
    /// last fold, the `swap` reads 0 and the producer keeps re-handing-out from its
    /// own line without bouncing the consumer's.
    ///
    /// # Safety
    /// Caller holds the owning central lock (single folder/taker per (shard,class)).
    #[inline]
    pub unsafe fn fold_free_word(&self, idx: u32, w: usize) -> u64 {
        let i = (idx as usize) * SPAN_BITMAP_WORDS + w;
        let cell = self.alloc_words[i].get();
        // Peek with a relaxed LOAD first: when the consumer hasn't deposited into
        // this word since the last fold (the common case when the producer is
        // keeping up), the `free_words` line stays in shared state — no `swap` store
        // to pull it exclusive and bounce it off the consumer's core. Only when
        // there is something to fold do we pay the `swap` RMW (Acquire pairs with
        // the deposit's Release).
        if self.free_words[i].load(Ordering::Relaxed) == 0 {
            return *cell;
        }
        let incoming = self.free_words[i].swap(0, Ordering::Acquire);
        let merged = *cell | incoming;
        *cell = merged;
        merged
    }

    /// Set free bits `mask` directly in the producer-private `alloc_words` word `w`
    /// of span `idx` (used by the carver, which runs under the central lock and owns
    /// fresh slots outright — they need no fold). Plain non-atomic OR.
    ///
    /// # Safety
    /// Caller holds the owning central lock.
    #[inline]
    pub unsafe fn alloc_set_mask(&self, idx: u32, w: usize, mask: u64) {
        let cell = self.alloc_words[(idx as usize) * SPAN_BITMAP_WORDS + w].get();
        *cell |= mask;
    }

    /// Clear exactly the bits in `taken` from the producer-private `alloc_words` word
    /// `w` of span `idx` (the slots a refill just handed out). Plain non-atomic store
    /// — `alloc_words` is touched only under the central lock, so no RMW is needed.
    ///
    /// # Safety
    /// Caller holds the owning central lock.
    #[inline]
    pub unsafe fn alloc_clear(&self, idx: u32, w: usize, taken: u64) {
        let cell = self.alloc_words[(idx as usize) * SPAN_BITMAP_WORDS + w].get();
        *cell &= !taken;
    }

    /// Whether span `idx` is **fully free** for a class whose tiling has
    /// `slots_per_span` slots: every one of those slots is currently a free bit the
    /// owner could hand out (none live, none magazine-cached, none in-flight). This
    /// is the emptiness gate the cross-class reclaimer checks before re-tagging a
    /// span (the A1-A3 protocol):
    ///
    /// * **A1 fold** every used word (`fold_free_word`'s `swap(0, Acquire)`), draining
    ///   any consumer deposits into the producer-private `alloc_words` — the Acquire
    ///   pairs with each deposit's Release so a freed object's contents are visible.
    /// * **A2 popcount**: if the folded `alloc_words` free-bit count over the
    ///   `slots_per_span` slots is not the full count, a live / magazine-cached object
    ///   exists (its bit is clear) → not fully free.
    /// * **A3 re-read** `free_words` with `Acquire`: a deposit that committed between
    ///   A1's relaxed peek and now would show here → not fully free (be conservative).
    ///
    /// Correctness rests on the bitmap polarity invariant: a slot's `alloc_words` bit
    /// is SET only while the slot is free-and-handoutable; it is CLEARed on hand-out
    /// ([`alloc_clear`](Self::alloc_clear)) and only re-SET by folding a `Release`
    /// deposit of *that very object* — which the holder cannot issue while still
    /// holding it. So any referenced object forces a non-full popcount here.
    ///
    /// # Safety
    /// Caller holds the owning `(shard, class)` central lock (single folder/taker).
    pub unsafe fn is_fully_free(&self, idx: u32, slots_per_span: usize) -> bool {
        let nwords = slots_per_span.div_ceil(64);
        let mut free_count = 0u32;
        for w in 0..nwords {
            // A1: fold consumer deposits into the private alloc word, then count.
            let word = self.fold_free_word(idx, w);
            free_count += word.count_ones();
        }
        if free_count as usize != slots_per_span {
            return false;
        }
        // A3: a deposit that landed after A1's peek but before now would be visible
        // here. `Acquire` pairs with the deposit's `Release`. Any set bit means an
        // object was just freed cross-thread into this span → treat as not-yet-empty.
        for w in 0..nwords {
            let i = (idx as usize) * SPAN_BITMAP_WORDS + w;
            if self.free_words[i].load(Ordering::Acquire) != 0 {
                return false;
            }
        }
        true
    }

    /// Unlink span `idx` from a `(shard, class)` active list given its predecessor
    /// (`prev`, or `NO_SPAN` if `idx` is the head). Returns the (possibly new) head —
    /// equal to `idx`'s old successor when `idx` was the head, else unchanged `head`.
    /// Marks `idx` off-list. The active list is singly linked, so the caller (a
    /// supervisor sweep already walking the list) supplies the predecessor — no scan.
    ///
    /// Unlinking is sound **only** for a span proven [`is_fully_free`](Self::is_fully_free)
    /// under the same lock: a fully-free span has no future deposits (you can't free
    /// an object you don't hold, and none are held), so removing it from the active
    /// list strands no free. This is the controlled exception to the otherwise-
    /// monotonic `on_list` invariant the lock-free deposit relies on.
    ///
    /// # Safety
    /// Caller holds the owning central lock; `prev` is `idx`'s predecessor on this
    /// list (or `NO_SPAN`); `head` is this list's current head.
    pub unsafe fn unlink_active(&self, idx: u32, prev: u32, head: u32) -> u32 {
        let succ = *self.next_active[idx as usize].get();
        let new_head = if prev == NO_SPAN {
            debug_assert_eq!(head, idx, "unlink: head-case but head != idx");
            succ
        } else {
            *self.next_active[prev as usize].get() = succ;
            head
        };
        *self.next_active[idx as usize].get() = NO_SPAN;
        self.on_list[idx as usize].store(false, Ordering::Relaxed);
        new_head
    }

    /// Reset span `idx`'s bitmaps to empty (all slots allocated / no free bits in
    /// either `alloc_words` or `free_words`), in preparation for re-tiling under a new
    /// size class or for returning the span to the span pool. Clears the first
    /// `clear_words` words of both arrays.
    ///
    /// # Safety
    /// Caller holds the owning central lock and has proven the span fully free (so no
    /// concurrent deposit can be targeting it).
    pub unsafe fn reset_bitmaps(&self, idx: u32, clear_words: usize) {
        let base = (idx as usize) * SPAN_BITMAP_WORDS;
        for w in 0..clear_words {
            *self.alloc_words[base + w].get() = 0;
            self.free_words[base + w].store(0, Ordering::Relaxed);
        }
    }
}

/// Per-span metadata. Packed into two atomics so it can be published by the
/// carver and read lock-free on the free path.
///
/// `class` is the size class id (0 = not yet assigned / not a span we own).
/// `home_cpu` is the CPU that carved the span; cross-CPU frees route here.
#[repr(C)]
pub struct SpanMeta {
    /// `class << 16 | home_cpu`, or 0 if unassigned. Single word so a carve
    /// publishes atomically and a free reads atomically.
    packed: AtomicU32,
}

impl SpanMeta {
    const fn new() -> Self {
        Self {
            packed: AtomicU32::new(0),
        }
    }

    #[inline]
    fn pack(class: u16, home_cpu: u16) -> u32 {
        ((class as u32) << 16) | (home_cpu as u32)
    }

    #[inline]
    pub fn assign(&self, class: u16, home_cpu: u16) {
        self.packed
            .store(Self::pack(class, home_cpu), Ordering::Release);
    }

    /// Returns `(class, home_cpu)` or `None` if unassigned.
    #[inline]
    pub fn load(&self) -> Option<(u16, u16)> {
        self.unpack(self.packed.load(Ordering::Acquire))
    }

    /// Like [`load`](Self::load) but a **relaxed** read. Sound on any free path:
    /// the caller already holds a live pointer into the span, so the carve that
    /// published this entry happens-before the caller obtained the pointer
    /// (through the original `alloc` and whatever channel handed the pointer over)
    /// — the metadata is already visible without re-synchronizing. Dropping the
    /// `Acquire` removes a barrier (`ldar` on aarch64) from every per-object free
    /// lookup, which dominates the batched producer/consumer free path.
    #[inline]
    pub fn load_relaxed(&self) -> Option<(u16, u16)> {
        self.unpack(self.packed.load(Ordering::Relaxed))
    }

    /// The raw packed word, relaxed (no `Option`/unpack). For the hot batched-free
    /// path that only needs the home field.
    #[inline]
    pub fn load_packed_relaxed(&self) -> u32 {
        self.packed.load(Ordering::Relaxed)
    }

    #[inline]
    fn unpack(&self, v: u32) -> Option<(u16, u16)> {
        if v == 0 {
            return None;
        }
        Some(((v >> 16) as u16, (v & 0xffff) as u16))
    }
}

/// The span table for one sub-heap's arena. `base` is the arena start; entries
/// cover `[base, base + num_spans * SPAN_BYTES)`.
pub struct SpanTable {
    base: usize,
    len: usize,
    entries: crate::SysBoxSlice<SpanMeta>,
}

// SAFETY: entries use atomics; base/len are immutable after construction.
unsafe impl Send for SpanTable {}
unsafe impl Sync for SpanTable {}

impl SpanTable {
    /// Build a table covering `[base, base+len)`. The entries array is allocated
    /// via the SYSTEM allocator (configure-time metadata; never recurses through
    /// toccata's global allocator).
    pub fn new(base: *mut u8, len: usize) -> Self {
        let num_spans = len.div_ceil(SPAN_BYTES);
        let entries = crate::sys_boxed_slice(num_spans, |_| SpanMeta::new());
        Self {
            base: base as usize,
            len,
            entries,
        }
    }

    #[inline]
    fn index_of(&self, ptr: *const u8) -> Option<usize> {
        let addr = ptr as usize;
        if addr < self.base || addr >= self.base + self.len {
            return None;
        }
        Some((addr - self.base) >> SPAN_BITS)
    }

    /// Assign every span overlapping `[ptr, ptr+bytes)` to `(class, home_cpu)`.
    /// Called by the carver when a span (or run) is dedicated to a class.
    pub fn assign_range(&self, ptr: *const u8, bytes: usize, class: u16, home_cpu: u16) {
        let start = match self.index_of(ptr) {
            Some(i) => i,
            None => return,
        };
        let end_addr = ptr as usize + bytes - 1;
        let end = ((end_addr - self.base) >> SPAN_BITS).min(self.entries.len() - 1);
        for e in &self.entries[start..=end] {
            e.assign(class, home_cpu);
        }
    }

    /// Recover `(class, home_cpu)` for an object pointer, or `None` if the
    /// pointer is outside this arena / unassigned.
    #[inline]
    pub fn lookup(&self, ptr: *const u8) -> Option<(u16, u16)> {
        let idx = self.index_of(ptr)?;
        self.entries[idx].load()
    }

    /// Relaxed variant of [`lookup`](Self::lookup) for the free hot path; see
    /// [`SpanMeta::load_relaxed`] for why a relaxed read is sound there.
    #[inline]
    pub fn lookup_relaxed(&self, ptr: *const u8) -> Option<(u16, u16)> {
        let idx = self.index_of(ptr)?;
        self.entries[idx].load_relaxed()
    }

    /// Just the home field (low 16 bits) for an in-pool pointer, relaxed. The hot
    /// batched-free path already knows the object is in-pool and knows its class,
    /// so it needs neither the bounds-`Option` nor the class unpack — only the home
    /// shard to route the deposit. One shift + one relaxed load + one mask, no
    /// branch. Returns `u16::MAX` if the pointer is out of range (caller treats an
    /// unrecognized home as "local", which is always safe).
    ///
    /// # Safety
    /// `ptr` should be an in-pool object pointer (the caller's `in_pool`/magazine
    /// invariant); an out-of-range pointer returns the sentinel rather than UB.
    #[inline]
    pub unsafe fn home_relaxed(&self, ptr: *const u8) -> u16 {
        let addr = ptr as usize;
        if addr < self.base || addr >= self.base + self.len {
            return u16::MAX;
        }
        let idx = (addr - self.base) >> SPAN_BITS;
        (self.entries[idx].load_packed_relaxed() & 0xffff) as u16
    }

    /// Whether `ptr` falls within this arena's address range.
    #[inline]
    pub fn contains(&self, ptr: *const u8) -> bool {
        let addr = ptr as usize;
        addr >= self.base && addr < self.base + self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_and_lookup_roundtrip() {
        // Fake a 1 MiB arena at a page-aligned dummy base; we never deref.
        let base = 0x1_0000_0000usize as *mut u8;
        let table = SpanTable::new(base, 1 << 20);
        // Assign a class-7 span homed on CPU 3 at offset 0.
        table.assign_range(base, SPAN_BYTES, 7, 3);
        assert_eq!(table.lookup(base), Some((7, 3)));
        assert_eq!(
            table.lookup(unsafe { base.add(SPAN_BYTES - 1) }),
            Some((7, 3))
        );
        // The next span is unassigned.
        assert_eq!(table.lookup(unsafe { base.add(SPAN_BYTES) }), None);
        // Out of range.
        assert_eq!(table.lookup((base as usize - 1) as *const u8), None);
    }

    #[test]
    fn multi_span_range() {
        let base = 0x2_0000_0000usize as *mut u8;
        let table = SpanTable::new(base, 4 * SPAN_BYTES);
        // A large object spanning 2.5 spans.
        table.assign_range(base, SPAN_BYTES * 2 + 100, 40, 1);
        assert_eq!(table.lookup(base), Some((40, 1)));
        assert_eq!(
            table.lookup(unsafe { base.add(SPAN_BYTES * 2) }),
            Some((40, 1))
        );
    }
}
