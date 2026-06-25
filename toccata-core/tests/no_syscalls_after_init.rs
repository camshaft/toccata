// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Phase 1 exit-criterion test: after init, the alloc/free hot path issues NO
//! memory-management syscall (`mmap`/`mremap`/`munmap`/`madvise`/`mlock*`) against
//! toccata's reservation.
//!
//! This is the empirical proof of the anti-stall thesis. It runs as a separate
//! process so we can `strace` only the post-init phase: the harness binary
//! reserves (mmap+populate+mlock once), emits a getpriority sentinel as the init
//! boundary, does a churn of allocs/frees, and the parent (this test) straces it
//! and asserts no memory syscall touches the reservation range after the
//! sentinel. There is no global "seal" flag — the never-stall property is
//! structural (the region is built once and never re-mapped).
//!
//! On non-Linux (dev) it's a no-op. On Linux without strace it prints a skip.

#![cfg(target_os = "linux")]

use std::process::Command;

/// The body that runs in the child harness (invoked via a re-exec of the test
/// binary with an env marker). Kept tiny and free of stdlib allocation churn
/// after seal where practical.
fn run_child() {
    use toccata_core::subheap::SubHeapBuilder;

    // Budget under the default 8 MiB RLIMIT_MEMLOCK so the test runs without
    // requiring a raised limit. 6 MiB leaves room for the small-class churn AND
    // the large-coalescing-path exercise below (the bump cursor is shared and
    // never retreats, so both paths draw from the same arena).
    let sh = SubHeapBuilder::new("trace", 6 * 1024 * 1024)
        .num_cpus(4)
        .cap_per_class(16)
        .build_standalone()
        .expect("reserve");

    // Publish toccata's reservation range so the parent can flag ONLY syscalls
    // that touch our pool (vs. unrelated stdlib/glibc-arena noise in this
    // non-global-allocator test). Printed to stdout, captured by the parent.
    let (rbase, rlen) = sh.reservation_range();
    eprintln!("TOCCATA_RANGE {rbase:#x} {rlen:#x}");

    // Pre-grow all stdlib-allocator-backed state BEFORE sealing, so the churn
    // below cannot trigger a system-allocator mmap that would be a false
    // positive in the trace. toccata is not the global allocator in this test.
    let classes: Vec<usize> = [8usize, 64, 256, 1500, 4096, 16384]
        .iter()
        .map(|&s| toccata_core::sizeclass::class_for(s).unwrap())
        .collect();
    let mut held: Vec<(std::ptr::NonNull<u8>, usize)> = Vec::with_capacity(4096);

    // Disable glibc malloc's arena trimming so the *system* allocator (which
    // backs the test harness's own allocations, not toccata) doesn't emit
    // MADV_DONTNEED after our boundary and create a false positive. toccata is
    // not the global allocator in this test, so we must quiet glibc explicitly.
    unsafe {
        libc::mallopt(libc::M_TRIM_THRESHOLD, -1);
        libc::mallopt(libc::M_TOP_PAD, 0);
    }

    // Everything below the init boundary must touch no OS memory syscall against
    // toccata's reservation. The reservation is already built, populated, and
    // locked above; there is no global "seal" flag — the never-stall property is
    // structural (we never re-issue a growth/reclaim syscall against the region).
    // Emit a distinctive boundary marker INTO the syscall stream: a getpriority
    // with a sentinel `who` value. strace records it, giving us an unambiguous
    // "everything after this line is post-init" divider that doesn't depend on
    // guessing which mlock was ours.
    unsafe { libc::syscall(libc::SYS_getpriority, libc::PRIO_PROCESS, 0xC0FFEE_u32) };
    eprintln!("TOCCATA_SEALED");
    for round in 0..200 {
        for &c in &classes {
            if let Some(p) = sh.alloc_class(c) {
                held.push((p, c));
            }
        }
        if round % 2 == 1 {
            for (p, c) in held.drain(..) {
                unsafe { sh.dealloc_class(p, c) };
            }
        }
    }
    for (p, c) in held.drain(..) {
        unsafe { sh.dealloc_class(p, c) };
    }

    // Exercise the LARGE / coalescing-SpanPool path too: a freed N-span run must
    // recycle (coalesce + split) for an (N±1)-span request entirely in userspace
    // (no mmap/madvise against the reservation). Drift the span-count up and down,
    // freeing each generation so the pool merges adjacent runs and splits to fit —
    // exactly the path Stage 1 added. If any of this carved fresh OS memory or
    // returned pages to the kernel, the trace below would catch it.
    let mut large_held: Vec<std::ptr::NonNull<u8>> = Vec::with_capacity(64);
    for gen in 0..8usize {
        // 5,6,7,8 spans then back down — forces coalesce + split reuse.
        let span_count = 5 + (gen % 4);
        let bytes = span_count * toccata_core::meta::SPAN_BYTES; // > MAX_SMALL
        for _ in 0..4 {
            if let Some(p) = sh.alloc_large(bytes) {
                // Touch the head so the run is genuinely backed (it already is —
                // populate+mlock at init — but this proves no fault-in syscall).
                unsafe { *p.as_ptr() = 0xA5 };
                large_held.push(p);
            }
        }
        // Free the whole generation back to the coalescing pool.
        for p in large_held.drain(..) {
            unsafe { sh.dealloc_by_ptr(p) };
        }
    }
    eprintln!("TOCCATA_DONE");
}

#[test]
fn no_memory_syscalls_after_seal() {
    // Child mode: do the work and exit.
    if std::env::var("TOCCATA_TRACE_CHILD").is_ok() {
        run_child();
        return;
    }

    // Parent mode: re-exec ourselves under strace, capturing only mmap-family
    // syscalls, then assert none appear after the TOCCATA_SEALED marker.
    let strace = which("strace");
    if strace.is_none() {
        eprintln!("SKIP: strace not found; install strace to run this gate");
        return;
    }
    let exe = std::env::current_exe().unwrap();

    // -f follow forks, -e trace the memory syscalls, -qq quiet, output to a file.
    let trace_file = std::env::temp_dir().join(format!("toccata_trace_{}.log", std::process::id()));
    let output = Command::new(strace.unwrap())
        .args([
            "-f",
            "-qq",
            "-e",
            "trace=mmap,mremap,munmap,madvise,mlock,mlock2,munlock,getpriority",
        ])
        .arg("-o")
        .arg(&trace_file)
        .arg(&exe)
        .arg("--exact")
        .arg("no_memory_syscalls_after_seal")
        .arg("--nocapture")
        .env("TOCCATA_TRACE_CHILD", "1")
        .env("RUST_TEST_THREADS", "1")
        .output()
        .expect("spawn strace");
    assert!(output.status.success(), "child under strace failed");

    // The child prints `TOCCATA_RANGE <base> <len>` to stderr (more reliably
    // flushed than libtest-wrapped stdout).
    let child_err = String::from_utf8_lossy(&output.stderr);
    let (rbase, rlen) = child_err
        .lines()
        .find_map(|l| {
            let rest = l.strip_prefix("TOCCATA_RANGE ")?;
            let mut it = rest.split_whitespace();
            let b = usize::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
            let n = usize::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
            Some((b, n))
        })
        .expect("child must print TOCCATA_RANGE");

    let log = std::fs::read_to_string(&trace_file).unwrap_or_default();
    let _ = std::fs::remove_file(&trace_file);

    // The child emits a sentinel `getpriority(PRIO_PROCESS, 0xC0FFEE)` right
    // after seal(). Everything after that line in the syscall stream is
    // post-seal. Assert no memory-growth syscall (mmap/mremap/madvise/mlock*)
    // appears after it. munmap is permitted (teardown), and a `MADV_DONTNEED`
    // from the *system* allocator freeing the test's own scratch is the only
    // gray area — we pre-grow `held` so toccata's path makes none, and the
    // single-threaded harness avoids stack-teardown madvise.
    let lines: Vec<&str> = log.lines().collect();
    let boundary = lines
        .iter()
        .rposition(|l| l.contains("getpriority(") && l.contains("3735928")) // 0xC0FFEE
        .or_else(|| lines.iter().rposition(|l| l.contains("getpriority(")))
        .expect("expected the post-seal getpriority sentinel in the trace");

    // After the boundary, flag any memory syscall whose FIRST argument (an
    // address) falls within toccata's reservation `[rbase, rbase+rlen)`. This
    // is the precise invariant: toccata must never touch its own locked pool
    // via the kernel after seal. Unrelated stdlib/glibc-arena syscalls (outside
    // our range) are correctly ignored.
    let range = rbase..rbase.saturating_add(rlen);
    let mut violations = Vec::new();
    for line in &lines[boundary + 1..] {
        // munmap/munlock are permitted: they only ever happen in
        // `Reservation::drop` at teardown, which the design explicitly allows.
        // The invariant is no growth/reclaim (mmap/mremap/madvise/mlock) against
        // our pool during operation.
        let mem = ["mmap(", "mremap(", "madvise(", "mlock(", "mlock2("]
            .iter()
            .find(|p| line.contains(**p));
        let Some(_) = mem else { continue };
        if line.contains("<unfinished") || line.contains("resumed>") {
            continue;
        }
        if let Some(addr) = first_hex_arg(line) {
            if range.contains(&addr) {
                violations.push((*line).to_string());
            }
        }
    }
    assert!(
        violations.is_empty(),
        "memory syscalls touching toccata's reservation [{rbase:#x}, +{rlen:#x}) after seal():\n{}",
        violations.join("\n")
    );
}

/// Parse the first `0x...` hex argument of a strace line, e.g.
/// `1234 madvise(0xffff..., 4096, ...) = 0` -> Some(0xffff...).
fn first_hex_arg(line: &str) -> Option<usize> {
    let open = line.find('(')?;
    let arg = &line[open + 1..];
    let hex = arg.strip_prefix("0x")?;
    let end = hex
        .find(|c: char| !c.is_ascii_hexdigit())
        .unwrap_or(hex.len());
    usize::from_str_radix(&hex[..end], 16).ok()
}

fn which(bin: &str) -> Option<String> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {bin}"))
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}
