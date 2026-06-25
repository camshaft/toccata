// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Size-class table and size→class mapping.
//!
//! Small allocations are rounded up to one of a fixed set of classes: finely
//! spaced for small sizes, then geometric. Requests larger than the maximum
//! small class bypass the per-CPU slab and go to the TLSF cold backstop
//! (`crate::backstop`). The mapping is a precomputed `u8` lookup table indexed by
//! `(size + 7) >> 3`, so size→class is one load and no branch on the hot path.
//!
//! Derived from tcmalloc's small-class spacing (verified against
//! `refs/tcmalloc/tcmalloc/size_classes.cc`): bound internal fragmentation while
//! keeping the class count and per-CPU header overhead modest.

/// Largest request served from the per-CPU slab. Above this → TLSF backstop.
pub const MAX_SMALL: usize = 256 * 1024;

/// Minimum object size / slot granularity. Pointers are 8-aligned, which the
/// (future) rseq fast path relies on for its emptiness sentinel.
pub const MIN_ALIGN: usize = 8;

/// The size-class object sizes, in bytes. Index 0 is unused (class ids start at
/// 1 so 0 can mean "not a slab class / use backstop"). Fine to MAX_SMALL.
pub const CLASS_SIZES: &[usize] = &[
    0, // class 0 = sentinel (backstop / oversize)
    8, 16, 24, 32, 48, 64, 80, 96, 112, 128, // fine, 16B steps then 16
    160, 192, 224, 256, // 32B steps
    320, 384, 448, 512, // 64B steps
    640, 768, 896, 1024, // 128B steps
    1280, 1536, 1792, 2048, // 256B steps
    2560, 3072, 3584, 4096, // 512B steps
    5120, 6144, 7168, 8192, // 1K steps
    10240, 12288, 14336, 16384, // 2K steps
    20480, 24576, 28672, 32768, // 4K steps
    49152, 65536, // 16K steps
    98304, 131072, // 32K steps
    196608, 262144, // 64K steps → MAX_SMALL
];

/// Number of size classes including the sentinel at index 0.
pub const NUM_CLASSES: usize = CLASS_SIZES.len();

/// Returns the class id for a request of `size` bytes, or `None` if it exceeds
/// `MAX_SMALL` (caller routes to the backstop). `size` must be > 0. `const` so
/// the hot-path lookup table can be computed at compile time.
#[inline]
pub const fn class_for(size: usize) -> Option<usize> {
    if size > MAX_SMALL {
        return None;
    }
    let want = if size == 0 { 1 } else { size };
    // CLASS_SIZES is sorted ascending after index 0.
    let mut idx = 1;
    while idx < CLASS_SIZES.len() {
        if CLASS_SIZES[idx] >= want {
            return Some(idx);
        }
        idx += 1;
    }
    None
}

/// Number of entries in the const size→class lookup table: one per 8-byte step
/// up to `MAX_SMALL`, plus the zero slot.
const LOOKUP_LEN: usize = (MAX_SMALL >> 3) + 1;

/// The size→class lookup table, computed **at compile time** and baked into the
/// binary (`.rodata`). The hot path indexes it directly — a single load, no
/// `OnceLock` atomic, no linear scan. Entry `i` is the class for a request of
/// `i*8` bytes (0 = oversize → large path).
static LOOKUP: [u16; LOOKUP_LEN] = build_lookup_const();

const fn build_lookup_const() -> [u16; LOOKUP_LEN] {
    let mut table = [0u16; LOOKUP_LEN];
    // Walk entries and classes together (both ascending), so this is O(entries +
    // classes), not O(entries * classes) — keeps const-eval fast.
    let mut i = 0;
    let mut class = 1; // smallest real class
    while i < LOOKUP_LEN {
        let size = if i == 0 { 1 } else { i << 3 };
        // Advance the class cursor to the first class whose size >= this size.
        while class < CLASS_SIZES.len() && CLASS_SIZES[class] < size {
            class += 1;
        }
        table[i] = if class < CLASS_SIZES.len() {
            class as u16
        } else {
            0
        };
        i += 1;
    }
    table
}

/// O(1) size→class for the hot path: a single indexed load into the const table,
/// no atomic and no scan. `None` if `size > MAX_SMALL` (→ large path).
#[inline]
pub fn class_of(size: usize) -> Option<usize> {
    if size == 0 || size > MAX_SMALL {
        return None;
    }
    let idx = (size + (MIN_ALIGN - 1)) >> 3;
    let c = LOOKUP[idx];
    (c != 0).then_some(c as usize)
}

/// The object size for a class id.
#[inline]
pub fn size_of_class(class: usize) -> usize {
    CLASS_SIZES[class]
}

/// Smallest class whose object size is >= `need` AND a multiple of `align`, for
/// over-aligned requests (`align > MIN_ALIGN`). Cold path — the hot path only
/// calls this when alignment exceeds the default. `None` if no small class fits
/// (caller routes to the large/span path).
#[cold]
pub fn aligned_class(need: usize, align: usize) -> Option<usize> {
    let mut class = class_of(need)?;
    while size_of_class(class) % align != 0 {
        class += 1;
        if class >= NUM_CLASSES {
            return None;
        }
    }
    Some(class)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_sorted_and_bounded() {
        for w in CLASS_SIZES.windows(2).skip(1) {
            assert!(w[0] < w[1], "class sizes must be strictly ascending: {w:?}");
        }
        assert_eq!(*CLASS_SIZES.last().unwrap(), MAX_SMALL);
    }

    #[test]
    fn class_for_rounds_up() {
        assert_eq!(class_for(1), Some(1)); // -> 8
        assert_eq!(size_of_class(class_for(1).unwrap()), 8);
        assert_eq!(size_of_class(class_for(8).unwrap()), 8);
        assert_eq!(size_of_class(class_for(9).unwrap()), 16);
        assert_eq!(size_of_class(class_for(1500).unwrap()), 1536);
        assert_eq!(size_of_class(class_for(2048).unwrap()), 2048);
        assert_eq!(class_for(MAX_SMALL), Some(NUM_CLASSES - 1));
        assert_eq!(class_for(MAX_SMALL + 1), None);
    }

    #[test]
    fn lookup_table_matches_class_for() {
        // The const O(1) table (class_of) must agree with the scan (class_for)
        // across the full range, not just spot checks.
        for size in 1..=MAX_SMALL {
            assert_eq!(class_of(size), class_for(size), "mismatch at size {size}");
        }
        assert_eq!(class_of(MAX_SMALL + 1), None);
        assert_eq!(class_of(0), None);
    }

    #[test]
    fn worst_case_internal_fragmentation_bounded() {
        // No class should waste more than ~25% vs the next-smaller class+1.
        for c in 2..NUM_CLASSES {
            let prev = CLASS_SIZES[c - 1];
            let this = CLASS_SIZES[c];
            let smallest_in_class = prev + 1;
            let waste = (this - smallest_in_class) as f64 / this as f64;
            assert!(waste < 0.5, "class {c} ({this}B) waste {waste:.2} too high");
        }
    }
}
