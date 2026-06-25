// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! The OS-memory reservation primitive.
//!
//! A [`Reservation`] is an up-front `mmap(MAP_POPULATE)` + `mlock2`'d region: the
//! whole anti-stall thesis in one object. Once locked and pre-faulted, the region
//! is inert to kernel reclaim (`refs/kernel/mm/madvise.c:560` excludes
//! `VM_LOCKED`), so no later page fault, purge, or cgroup-reclaim penalty can
//! stall a hot path that touches it.
//!
//! The anti-stall guarantee is **structural**: reserve + populate + lock once,
//! then never issue a growth/reclaim syscall against the region again. That is a
//! property of how the higher layers *use* the region (they carve it and never
//! re-`mmap`), not something this primitive enforces with a global flag — a
//! single process can hold many independent reservations (the global heap, a
//! per-component UMEM `FramePool`), built at different times. `Reservation` is
//! therefore a pure primitive with no process-global "sealed" state.

use std::ptr::NonNull;

/// Errors from the one-shot reservation. All are init-time and fatal by design
/// (the user chose "fail loudly at boot, never stall at runtime").
#[derive(Debug)]
pub enum ReserveError {
    /// `mmap` failed (e.g. address space or overcommit limits).
    Mmap(std::io::Error),
    /// `mlock2` failed — almost always `RLIMIT_MEMLOCK` too low (see R2). The
    /// message includes the actionable fix. Only returned when `lock` was
    /// [`Require::Required`]; a [`Require::BestEffort`] lock failure proceeds
    /// unlocked instead (observable via [`Reservation::is_locked`]).
    Mlock { source: std::io::Error, limit: u64, requested: usize },
    /// `getrlimit(RLIMIT_MEMLOCK)` says the lock cannot possibly succeed; we
    /// pre-flight this to fail with a clear message before even mapping. As
    /// `Mlock`, only a `Required` lock surfaces this.
    RlimitTooLow { limit: u64, requested: usize },
    /// The [`ReserveOpts`] were invalid (e.g. a non-power-of-two huge-page size,
    /// or a backing mode not yet implemented on this platform).
    InvalidConfig(&'static str),
}

impl std::fmt::Display for ReserveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReserveError::Mmap(e) => write!(f, "toccata reservation mmap failed: {e}"),
            ReserveError::Mlock { source, limit, requested } => write!(
                f,
                "toccata reservation mlock2 failed: {source}. Requested {requested} bytes but \
                 RLIMIT_MEMLOCK is {limit}. Raise it (ulimit -l / LimitMEMLOCK=infinity in the \
                 unit / a higher container memlock limit), or lower toccata's configured budget."
            ),
            ReserveError::RlimitTooLow { limit, requested } => write!(
                f,
                "toccata cannot lock {requested} bytes: RLIMIT_MEMLOCK is only {limit}. Raise \
                 RLIMIT_MEMLOCK (ulimit -l unlimited / LimitMEMLOCK=infinity) or lower the budget."
            ),
            ReserveError::InvalidConfig(why) => {
                write!(f, "toccata reservation config invalid: {why}")
            }
        }
    }
}

impl std::error::Error for ReserveError {}

/// A capability that can be **demanded**, **attempted**, or **skipped**.
///
/// Used for [`ReserveOpts`] knobs that can fail at map time *and* whose absence is
/// still correct, just degraded (locking, huge pages). A [`Require::BestEffort`]
/// attempt that fails falls back gracefully and records the effective state on the
/// [`Reservation`] (see [`Reservation::is_locked`] / [`Reservation::huge_page_size`])
/// rather than failing the whole reservation; a [`Require::Required`] failure is
/// fatal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Require<T> {
    /// Do not attempt it.
    Off,
    /// Try; on failure, fall back gracefully and record what happened.
    BestEffort(T),
    /// Must succeed, or [`Reservation::reserve_with`] errors.
    Required(T),
}

impl<T> Require<T> {
    /// The payload, if this is `BestEffort` or `Required`.
    #[inline]
    pub fn value(&self) -> Option<&T> {
        match self {
            Require::Off => None,
            Require::BestEffort(t) | Require::Required(t) => Some(t),
        }
    }
    #[inline]
    fn is_required(&self) -> bool {
        matches!(self, Require::Required(_))
    }
    // Only consulted by the Linux reserve path; the dev stub ignores lock/huge knobs.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    #[inline]
    fn is_off(&self) -> bool {
        matches!(self, Require::Off)
    }
}

impl Require<()> {
    /// Demand the capability (fatal on failure). The default for `lock`.
    pub const ON: Self = Require::Required(());
    /// Attempt the capability; proceed degraded on failure.
    pub const TRY: Self = Require::BestEffort(());
    /// Skip the capability.
    pub const OFF: Self = Require::Off;
}

/// How a [`Reservation`]'s pages are backed.
#[derive(Clone, Debug, Default)]
pub enum Backing {
    /// Private anonymous pages — the default, and the owned-UMEM path.
    #[default]
    Anonymous,
    // FUTURE (P2, block-lru parity — not built yet): a named /dev/shm segment for
    // cross-process sharing. Designed-in as a variant so adding it is non-breaking.
    // Shared { name: String, create: bool },
}

/// A huge-page request. Mirrors block-lru's `HugePages`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HugePages {
    /// The system default huge-page size (`MAP_HUGETLB`, no size bits).
    Default,
    /// A specific huge-page size in bytes; **must be a power of two**.
    Size(usize),
}

/// Options for building a [`Reservation`].
///
/// The defaults reproduce toccata's never-stall thesis **exactly**: anonymous,
/// populated, locked, dontdump. A caller relaxes individual knobs (dev/non-Linux)
/// or adds huge pages (owned UMEM). Construct with [`ReserveOpts::new`] and chain
/// the builder methods.
#[derive(Clone, Debug)]
pub struct ReserveOpts {
    /// Bytes to reserve (rounded up to the page / huge-page size as needed).
    pub len: usize,
    /// Page backing. Default [`Backing::Anonymous`].
    pub backing: Backing,
    /// `mlock2` + the `RLIMIT_MEMLOCK` pre-flight. Default [`Require::ON`] — this
    /// is the thesis; relaxing it forfeits never-stall.
    pub lock: Require<()>,
    /// `MAP_HUGETLB`. Default [`Require::Off`].
    pub huge_pages: Require<HugePages>,
    /// `MAP_POPULATE`: pre-fault every page at map time. Default `true`. (A kernel
    /// hint that never fails the mmap, so it is a bool, not a [`Require`].)
    pub populate: bool,
    /// `MADV_DONTDUMP`: keep the locked pool out of core dumps. Default `true`.
    /// (Already best-effort internally — its `madvise` return is ignored.)
    pub dontdump: bool,
}

impl ReserveOpts {
    /// Reserve `len` bytes with toccata's default never-stall configuration:
    /// anonymous, populated, locked (required), dontdump, no huge pages.
    #[inline]
    pub fn new(len: usize) -> Self {
        Self {
            len,
            backing: Backing::Anonymous,
            lock: Require::ON,
            huge_pages: Require::Off,
            populate: true,
            dontdump: true,
        }
    }

    /// Request huge pages (required).
    #[inline]
    pub fn huge_pages(mut self, h: HugePages) -> Self {
        self.huge_pages = Require::Required(h);
        self
    }
    /// Request huge pages, falling back to base pages if unavailable.
    #[inline]
    pub fn huge_pages_best_effort(mut self, h: HugePages) -> Self {
        self.huge_pages = Require::BestEffort(h);
        self
    }
    /// Override the lock mode (`Require::ON` / `TRY` / `OFF`).
    #[inline]
    pub fn lock(mut self, mode: Require<()>) -> Self {
        self.lock = mode;
        self
    }
    /// Toggle `MAP_POPULATE`.
    #[inline]
    pub fn populate(mut self, yes: bool) -> Self {
        self.populate = yes;
        self
    }
    /// Toggle `MADV_DONTDUMP`.
    #[inline]
    pub fn dontdump(mut self, yes: bool) -> Self {
        self.dontdump = yes;
        self
    }
}

/// A reserved, populated, (best-effort) locked region. Owns the mapping for
/// program lifetime; dropping it `munmap`s (only ever at shutdown — never on a hot
/// path).
///
/// The **effective** state — whether the lock actually took, and the huge-page
/// size actually obtained — is recorded at build time and queryable via
/// [`is_locked`](Self::is_locked) / [`huge_page_size`](Self::huge_page_size). A
/// caller that truly needs the never-stall guarantee should
/// `assert!(res.is_locked())`; a [`Require::BestEffort`] knob can silently degrade
/// otherwise.
pub struct Reservation {
    base: NonNull<u8>,
    len: usize,
    /// Whether `mlock2` actually succeeded (false if `lock` was `Off`, or a
    /// `BestEffort` lock that failed).
    locked: bool,
    /// The huge-page size in bytes actually mapped, or `None` for base pages
    /// (including a `BestEffort` huge-page request that fell back).
    huge_page_size: Option<usize>,
}

// SAFETY: the reservation is a raw region whose interior synchronization is
// handled by the slab/pool layers above. The handle itself is just (ptr, len).
unsafe impl Send for Reservation {}
unsafe impl Sync for Reservation {}

impl Reservation {
    /// Reserve `len` bytes with toccata's default never-stall configuration
    /// (anonymous, populated, locked, dontdump). Shorthand for
    /// [`reserve_with`](Self::reserve_with)`(ReserveOpts::new(len))`.
    #[inline]
    pub fn reserve(len: usize) -> Result<Self, ReserveError> {
        Self::reserve_with(ReserveOpts::new(len))
    }

    /// Reserve a region per [`ReserveOpts`]: resolve huge pages, pre-flight
    /// `RLIMIT_MEMLOCK` (against the huge-page-rounded length), then one
    /// `mmap` (+ `MAP_POPULATE`/`MAP_HUGETLB`) + `mlock2` + `MADV_DONTDUMP`.
    ///
    /// `Require::BestEffort` knobs that fail proceed degraded (the effective state
    /// is queryable via [`is_locked`](Self::is_locked) /
    /// [`huge_page_size`](Self::huge_page_size)) and emit a no-alloc raw-`write(2)`
    /// notice; `Require::Required` knobs are fatal.
    #[cfg(target_os = "linux")]
    pub fn reserve_with(opts: ReserveOpts) -> Result<Self, ReserveError> {
        assert!(opts.len > 0, "reservation length must be > 0");

        // --- resolve huge pages BEFORE the memlock pre-flight, so the lock charge
        // is computed against the length we actually map (rounded up to the
        // huge-page size). ---
        let mut map_len = opts.len;
        let mut hp_flags = 0i32;
        let mut hp_size: Option<usize> = None;
        if !opts.huge_pages.is_off() {
            let h = *opts.huge_pages.value().unwrap();
            match resolve_huge_pages(h) {
                Ok((flags, size)) => {
                    hp_flags = flags;
                    hp_size = Some(size);
                    map_len = opts.len.div_ceil(size) * size;
                }
                Err(why) => {
                    // A non-pow2 size is always a config bug; an unavailable
                    // default size is fatal only if the request was Required.
                    if opts.huge_pages.is_required() {
                        return Err(ReserveError::InvalidConfig(why));
                    }
                    // BestEffort: fall back to base pages.
                }
            }
        }

        // Pre-flight RLIMIT_MEMLOCK (a getrlimit, NOT a cgroup read). Only matters
        // if we intend to lock; gives a clear error before mapping anything.
        let limit = memlock_limit();
        if !opts.lock.is_off() && (limit as u128) < (map_len as u128) && limit != u64::MAX {
            if opts.lock.is_required() {
                return Err(ReserveError::RlimitTooLow { limit, requested: map_len });
            }
            // BestEffort lock that cannot succeed: skip it loudly, proceed unlocked.
            warn_degraded(b"toccata: reservation lock requested (best-effort) but \
                RLIMIT_MEMLOCK too low; proceeding UNLOCKED, never-stall forfeited [bytes=", map_len);
        }

        // One mmap. We pre-fault every page so there are no first-touch faults later
        // (the necessary condition for never tripping a memcg charge on the hot
        // path). With explicit HUGETLB we use MAP_POPULATE in the mmap. With base
        // pages we instead DEFER the populate: map unpopulated, hint THP (below),
        // then let mlock2 fault the range in — so the fault-in materializes 2 MiB
        // transparent huge pages directly rather than 4 KiB pages khugepaged would
        // have to collapse afterward.
        let thp = hp_flags == 0; // no explicit hugetlb ⇒ try transparent huge pages
        let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | hp_flags;
        if opts.populate && !thp {
            flags |= libc::MAP_POPULATE;
        }
        let mut base = unsafe { mmap_region(map_len, flags) };
        if base == libc::MAP_FAILED && hp_flags != 0 && !opts.huge_pages.is_required() {
            // BestEffort huge pages: the HUGETLB mmap failed (no pool / too
            // fragmented). Retry with base pages at the un-rounded length.
            warn_degraded(b"toccata: huge pages requested (best-effort) but mmap \
                failed; falling back to base pages [bytes=", opts.len);
            hp_size = None;
            map_len = opts.len;
            // base-page retry: defer populate to the post-THP-hint mlock as above.
            flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
            base = unsafe { mmap_region(map_len, flags) };
        }
        if base == libc::MAP_FAILED {
            return Err(ReserveError::Mmap(std::io::Error::last_os_error()));
        }
        let base = NonNull::new(base as *mut u8).expect("mmap returned non-null on success");

        // When NOT using explicit HUGETLB, hint the kernel to back this range with
        // transparent huge pages (THP). The arena is large (often GiBs) and touched
        // across many pages on the cross-thread producer/consumer path; with 4 KiB
        // base pages that is heavy dTLB + page-table-walk pressure (measured: the
        // 1.5 KiB prod/cons path's residual gap to jemalloc shows up as
        // dTLB-load-misses + walk-driven LLC misses, not allocator work). 2 MiB THP
        // cuts the page count ~512×. Issued BEFORE the mlock2 fault-in so the populate
        // materializes huge pages directly. Best-effort/advisory: failure (or a THP-
        // disabled kernel) just leaves base pages — never an error, never a stall.
        if thp {
            unsafe {
                libc::madvise(base.as_ptr() as *mut libc::c_void, map_len, libc::MADV_HUGEPAGE);
            }
            // The mmap skipped MAP_POPULATE so this madvise could precede fault-in.
            // If we won't mlock (lock off), populate now via MADV_WILLNEED so the
            // "no first-touch faults later" invariant still holds; with lock on, the
            // mlock2 below faults the whole range in (as huge pages).
            if opts.populate && opts.lock.is_off() {
                unsafe {
                    libc::madvise(base.as_ptr() as *mut libc::c_void, map_len, libc::MADV_WILLNEED);
                }
            }
        }

        // Lock the whole region: Unevictable, never reclaimed, never re-faulted.
        // mlock2(flags=0) faults the range in and fails loudly if it can't — this
        // is the primitive we actually trust (MAP_POPULATE is best-effort).
        let mut locked = false;
        if !opts.lock.is_off() {
            let rc = unsafe { libc::syscall(libc::SYS_mlock2, base.as_ptr(), map_len, 0) };
            if rc == 0 {
                locked = true;
            } else if opts.lock.is_required() {
                let source = std::io::Error::last_os_error();
                // Best-effort cleanup of the mapping before returning the error.
                unsafe { libc::munmap(base.as_ptr() as *mut libc::c_void, map_len) };
                return Err(ReserveError::Mlock { source, limit, requested: map_len });
            } else {
                // BestEffort lock failed: keep the (usable, unlocked) mapping.
                warn_degraded(b"toccata: reservation mlock2 failed (best-effort); \
                    proceeding UNLOCKED, never-stall forfeited [bytes=", map_len);
            }
        }

        // Keep core dumps sane: don't dump the (potentially huge) locked pool.
        if opts.dontdump {
            unsafe {
                libc::madvise(base.as_ptr() as *mut libc::c_void, map_len, libc::MADV_DONTDUMP)
            };
        }

        Ok(Self { base, len: map_len, locked, huge_page_size: hp_size })
    }

    /// Non-Linux dev stub: a plain heap-backed region (no locking, no huge pages;
    /// correctness only, for running the higher layers' tests on macOS). A
    /// `Required` huge-page request errors (mirroring block-lru); `lock` is a
    /// no-op (`is_locked()` is always `false` here — the never-stall guarantee
    /// does not apply to the dev stub).
    #[cfg(not(target_os = "linux"))]
    pub fn reserve_with(opts: ReserveOpts) -> Result<Self, ReserveError> {
        assert!(opts.len > 0);
        if opts.huge_pages.is_required() {
            return Err(ReserveError::InvalidConfig("huge pages are only supported on Linux"));
        }
        let layout = std::alloc::Layout::from_size_align(opts.len, 4096).unwrap();
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        let base = NonNull::new(p).ok_or_else(|| {
            ReserveError::Mmap(std::io::Error::new(std::io::ErrorKind::OutOfMemory, "alloc"))
        })?;
        Ok(Self { base, len: opts.len, locked: false, huge_page_size: None })
    }

    #[inline]
    pub fn base(&self) -> NonNull<u8> {
        self.base
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether `mlock2` actually took. `false` if `lock` was `Off`, a `BestEffort`
    /// lock that failed, or on the non-Linux dev stub. A caller that needs the
    /// never-stall guarantee should `assert!(res.is_locked())`.
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// The huge-page size in bytes actually mapped, or `None` for base pages
    /// (including a `BestEffort` huge-page request that fell back).
    #[inline]
    pub fn huge_page_size(&self) -> Option<usize> {
        self.huge_page_size
    }

    /// The whole reservation as a byte slice view (for carving).
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.base.as_ptr()
    }
}

/// One anonymous mmap of `len` bytes with the given extra `flags`. Returns the
/// raw `mmap` result (`MAP_FAILED` on error) so the caller can branch.
#[cfg(target_os = "linux")]
#[inline]
unsafe fn mmap_region(len: usize, flags: i32) -> *mut libc::c_void {
    libc::mmap(core::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0)
}

/// Resolve a [`HugePages`] request into `(extra mmap flags, page size in bytes)`.
/// `Size(n)` must be a power of two; `Default` reads the system default huge-page
/// size from `/proc/meminfo`.
#[cfg(target_os = "linux")]
fn resolve_huge_pages(h: HugePages) -> Result<(i32, usize), &'static str> {
    let size = match h {
        HugePages::Size(n) => {
            if !n.is_power_of_two() {
                return Err("huge page size must be a power of two");
            }
            n
        }
        HugePages::Default => {
            default_huge_page_size().ok_or("could not determine the default huge page size")?
        }
    };
    // Encode the size in the MAP_HUGE_* bits so we map exactly this page size.
    let shift = size.trailing_zeros() as i32;
    let flags = libc::MAP_HUGETLB | ((shift & ((1 << libc::MAP_HUGE_SHIFT) - 1)) << libc::MAP_HUGE_SHIFT);
    Ok((flags, size))
}

/// The system default huge-page size in bytes, parsed from `/proc/meminfo`'s
/// `Hugepagesize:` line. `None` if huge pages are unconfigured/unreadable. Runs at
/// reservation time only (the `String` allocation goes to the system allocator).
#[cfg(target_os = "linux")]
fn default_huge_page_size() -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("Hugepagesize:") {
            // e.g. "Hugepagesize:    2048 kB"
            let mut it = rest.split_whitespace();
            let val: usize = it.next()?.parse().ok()?;
            let unit = it.next().unwrap_or("kB");
            let mult = match unit {
                "kB" => 1024,
                "MB" => 1024 * 1024,
                _ => 1024,
            };
            let sz = val.checked_mul(mult)?;
            return if sz > 0 { Some(sz) } else { None };
        }
    }
    None
}

/// Emit a no-alloc, raw-`write(2)` degradation notice: `prefix` (a fixed byte
/// string ending mid-sentence) + the decimal byte count + `]\n`. Never `tracing`
/// (it allocates and may lock — fatal on a path that backs the global allocator).
#[cfg(target_os = "linux")]
#[inline]
fn warn_degraded(prefix: &[u8], bytes: usize) {
    let _ = crate::sys::diag::write_stderr(prefix);
    let mut buf = [0u8; 20];
    let _ = crate::sys::diag::write_stderr(crate::sys::diag::usize_to_dec(bytes, &mut buf));
    let _ = crate::sys::diag::write_stderr(b"]\n");
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // Only ever at shutdown: unmapping the region at teardown is legitimate
        // and off any hot path.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::munmap(self.base.as_ptr() as *mut libc::c_void, self.len);
        }
        #[cfg(not(target_os = "linux"))]
        unsafe {
            let layout = std::alloc::Layout::from_size_align(self.len, 4096).unwrap();
            std::alloc::dealloc(self.base.as_ptr(), layout);
        }
    }
}

/// Current `RLIMIT_MEMLOCK` soft limit in bytes; `u64::MAX` means unlimited.
#[cfg(target_os = "linux")]
pub fn memlock_limit() -> u64 {
    let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) };
    if rc != 0 {
        return 0;
    }
    if rl.rlim_cur == libc::RLIM_INFINITY {
        u64::MAX
    } else {
        rl.rlim_cur as u64
    }
}

#[cfg(not(target_os = "linux"))]
pub fn memlock_limit() -> u64 {
    u64::MAX
}
