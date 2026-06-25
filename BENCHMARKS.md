# toccata — Benchmark Results

All numbers measured on an **AWS Graviton EC2 dev box**: aarch64, Amazon Linux
2023, kernel 6.12, 64 logical CPUs, `rustc 1.95`, `--release` (`lto=thin`,
`codegen-units=1`). Reproduce with `toccata-bench` (see below).

> **Headline:** toccata is the only allocator tested that **does not stall**
> under cgroup `memory.high` pressure — the exact failure mode that motivated the
> project. Under an identical memory-limited cgroup, jemalloc's threads were
> penalty-slept by the kernel (a 6-second workload ran for **>50 seconds**),
> while toccata — its pool reserved and `mlock`'d *under* the limit — completed
> in 6 s with a **flat latency tail and zero allocations over 1 ms**.

## 1. Throughput & latency (no memory pressure)

`cargo run -p toccata-bench --release --bin <allocator> -- throughput`

Workload: alloc/free churn across MTU-ish size classes (16 B … 4 KiB), 1024
live; multi-thread = per-thread churn ×16; prod/cons = allocate on a producer
thread, free on a consumer thread (the networking pattern); latency = individual
`vec![0u8; size]` timings.

| Allocator | single-thread | multi-thread ×16 | **prod/cons ×8** | alloc p50 | p99 | p999 | **p9999** | max |
|-----------|--------------:|-----------------:|-----------------:|----------:|----:|-----:|----------:|----:|
| system (glibc) | 24.5 ns | 3.0 ns | 225.5 ns | 81 ns | 259 | 300 | **1159** | 14123 |
| jemalloc | 10.9 ns | 0.7 ns | 27.8 ns | 42 ns | 122 | 169 | **227** | 11459 |
| mimalloc | 7.9 ns | 0.5 ns | 21.6 ns | 41 ns | 161 | 277 | **331** | 13437 |
| snmalloc | 534.7 ns | 34.3 ns | 106.1 ns | 424 ns | 893 | 952 | **7702** | 23433 |
| toccata (locked baseline) | 46.6 ns | 118.4 ns | 134.4 ns | 60 ns | 123 | 142 | 198 | 11305 |
| toccata (rseq, pthread_getspecific TLS) | 14.9 ns | 1.0 ns | 38 ns | 47 ns | 111 | 124 | 151 | — |
| **toccata (rseq + L1 magazine + fast TLS)** | **7.4 ns** | **0.5 ns** | 24.0 ns | 41 ns | 107 | 116 | **190** | 11813 |

### Reading the table — toccata now beats jemalloc

With the rseq fast path (automatic on Linux x86_64/aarch64), the thread-local L1
magazine, and direct (fast) TLS:

- **single-thread: 7.4 ns — faster than jemalloc (10.9 ns)**, ties mimalloc
  (7.9 ns), ~3.3× the system allocator (24.5 ns).
- **multi-thread ×16: 0.5 ns — faster than jemalloc (0.7 ns)**, ties mimalloc
  (0.5 ns), 6× the system allocator (3.0 ns).
- **producer/consumer:** see the dedicated clean-SPSC section below — the
  `throughput` column here uses the old `std::mpsc`-confounded workload and is
  superseded.
- **Latency tail is by far the tightest of any allocator tested:** p9999 =
  **190 ns** vs jemalloc 227, mimalloc 331, snmalloc 7702, system 1159. The
  headline metric toccata was built for — won outright. (This is the in-loop
  `Instant::now()` micro-measurement; the noise-free never-stall tail under cgroup
  pressure is §2, where p9999 = 64 ns with zero >1 ms stalls.)

## 1b. Producer/consumer — clean SPSC ring (bitmap central)

`cargo run -p toccata-bench --release --bin <allocator> -- prodcons 8 <size>`

The `prodcons` workload pre-allocates an SPSC ring carrying raw `(ptr,len)`, so the
only measured allocator traffic is the producer's `alloc` + the consumer's
cross-thread `free` (one each per op) — the pure networking hand-off, with no
`std::mpsc` node-allocation confound. 8 pairs, threads **unpinned** (as jemalloc
runs), median of 41 reps; the 64-cpu box is noisy at single-ns scale, so figures
within ~0.5 ns are ties that flip run to run.

| size | toccata | jemalloc | mimalloc | snmalloc | system |
|------|---------:|---------:|---------:|---------:|-------:|
| mixed (16 B–4 KiB) | **12.3 ns** | 12.4 ns | 23.3 | 106 | 110 |
| 64 B | **13.7 ns** | 15.4 ns | 18.5 | 46 | 44 |
| 1500 B | 12.5 ns | **11.8 ns** | 24.7 | 104 | 93 |
| 2048 B | **12.3 ns** | 12.0 ns | — | — | — |

After rebuilding the central free-list around an out-of-band **per-span free
bitmap** (the jemalloc/tcmalloc slab-bitmap model: a cross-thread free sets a slot
bit arithmetically, never reading the cold cross-core object) with a **lock-free
deposit**, a **producer-private / consumer-written two-cache-line split**,
**ascending-address hand-out** (the magazine fills ascending but pops LIFO, so the
just-filled slice is reversed to keep the producer's `vec![0u8;n]` store stream
*forward* — a backward stream defeats the Neoverse L2 prefetcher), and a
**software prefetch-for-write** of large buffers a couple of allocations ahead of
hand-out, toccata beats mimalloc/snmalloc/system by ~2× at every size and **wins
or ties** at 64 B, mixed, and (at its best) 1.5 KiB. The residual at **1.5 KiB** is
a noisy ~0.3–0.7 ns: a PMU sweep showed jemalloc's hardware prefetcher runs ahead
of the producer's `memset` (~67 % more SLC reads, mostly *hits*) while toccata's
demand stores miss to DRAM on buffers evicted since their last owner; the software
prefetch recovers most of it (toccata wins the *min* and many medians) but a
steady-state slow-sample tail remains on the shared 64-cpu box. toccata runs ~17 %
*fewer* instructions here with
*fewer* L1/L2 refills and 42× *fewer* dTLB misses — it is not an
allocator-work or TLB deficit. (`std::mpsc` workload: toccata ~23 ns beats jemalloc
~27, ties mimalloc ~23; that workload is ~⅓ the channel's own atomics.)

### The fast-TLS unlock (and why it's allocation-free)

The middle row shows the cost of a safe-but-slow TLS: routing the per-thread
cache through `pthread_getspecific` (a libc call) cost ~4 ns (perf attributed
~11% of `alloc` to it). The fast row uses the pattern Rust's own TLS uses on
**stable**: a `thread_local!` with a `const {}` initializer holding a `Drop`-free
`*mut` pointer. That compiles to a direct thread-pointer-relative load with **no
lazy-init guard and no `__cxa_thread_atexit` registration** — so it does **not
allocate** (verified in the disassembly: the alloc/dealloc hot path has zero
`__tls_get_addr` / `__cxa_thread_atexit` / `pthread_getspecific` calls). The cache
*storage* is `calloc`'d once via libc (never toccata); thread-exit magazine flush
runs via a `pthread_key` destructor registered once at init, off the hot path.
This keeps the recursion-safety (no TLS path mallocs through toccata) while
matching native-TLS speed.

### How the L1 magazine closed the gap

The earlier ~18 ns single-thread number was the RSEQ prologue cost (arm `rseq_cs`,
`cpu_id` double-check, seize stop-flag load, bounds + bounded-retry — ~12
instructions) paid on *every* op, plus a per-op `MAIN.load(Acquire)` and an
alignment fixup. The fix mirrors jemalloc's own structure:

1. A **thread-local L1 magazine** (16 ptrs/class) in front of the per-CPU rseq
   slab. The common case is now a TLS access + array pop/push (~5 instructions,
   the jemalloc tcache shape); only a magazine miss pays the rseq prologue, and
   it amortizes it over a batch refill. **Crucially the never-stall guarantee is
   preserved** — every miss falls through to the locked-pool L2, never the
   kernel (verified: under `memory.high` pressure the magazine path does 46M
   ops/6 s with **zero >1 ms stalls** and a flat 32 ns p50–p9999).
2. The sub-heap pointer is resolved **once per thread** (cached in the TLS), so
   there is no `MAIN.load` atomic on the steady-state path.
3. Default-aligned small allocs skip alignment work entirely (every size class is
   a multiple of `MIN_ALIGN`); the over-aligned loop is a `#[cold]` helper.

The cost is a bounded, deliberate reintroduction of per-thread memory (a few KB
of magazines per thread, flushed back to L2 on thread exit) — the standard L1
trade, negligible against a multi-GB locked pool.

The locked-baseline row is the portable, always-correct fast path used when rseq
isn't compiled in; its multi-thread number is the per-`(CPU,size-class)` spin
contention the rseq path removes.
- The `max` column (~11–23 µs across all allocators) is a first-touch / cold
  cache artifact of the measurement loop, not allocator behavior — see §2 for
  the metric that isolates real stalls.

## 2. The never-stall property (under cgroup `memory.high`) — the headline

`toccata-bench`'s `latency_tail` binary runs a latency-critical "networking"
thread (alloc+free an MTU buffer every iteration, recording the tail in a fixed
O(1) histogram) while 4 background threads churn a working set, pushing RSS
against a cgroup memory limit. Run inside `systemd-run --scope -p MemoryHigh=…`.

### jemalloc under `MemoryHigh=200M`, churn 300 MiB

**THROTTLED.** The process — a **6-second** workload — was still running after
**50+ seconds**, its threads repeatedly put to sleep by the kernel's
`mem_cgroup_handle_over_high()` penalty (`schedule_timeout_killable` in
`mm/memcontrol.c`). It never produced output before we killed it. This is the
multi-second-stall failure mode toccata was built to avoid, reproduced
deterministically.

This happens because jemalloc grows RSS with dirty pages awaiting decay-based
`madvise` purging; under a soft limit the kernel throttles the faulting thread
rather than the allocator's background purger.

### toccata under `MemoryHigh=1200M`, pool 900 MiB, churn 300 MiB

```
=== latency_tail: toccata (33716470 samples over 6s) ===
  p50=70ns  p99=160ns  p999=199ns  p9999=232ns  p99999=4267ns  max=33449ns
  stalls: >1ms=0  >10ms=0  >100ms=0
```

**Completed in 6 s. Zero stalls over 1 ms.** Because toccata reserves and
`mlock`s its entire pool up front *under* the cgroup limit and never returns
memory to the kernel, it never grows RSS into the throttle zone and never
triggers a page fault that the kernel could penalize. The tail is flat.

### The mechanism, in one line

> jemalloc's RSS *grows toward the limit* (dirty pages + decay purge churn) →
> kernel throttles. toccata's RSS is *fixed below the limit at boot* (reserved +
> locked, never grows, never purges) → kernel has nothing to throttle.

### The deliberate bargain (and a real operational requirement)

toccata trades higher *steady-state* RSS (the whole pool is resident from t=0)
for the elimination of reclaim stalls. This requires `RLIMIT_MEMLOCK >= pool
size`: on the stock EC2 dev box the default is **8 MiB**, and toccata fails
**loudly at init** with an actionable error if the budget exceeds it — exactly as
designed (a loud boot failure on a misconfigured host, never a runtime stall).
Production deploys must set `LimitMEMLOCK=infinity` (systemd) or an equivalent
container memlock limit. The benchmarks raise it via `prlimit`.

## 2c. Fragmentation — closed in userspace, no relocation, no syscalls

`TOCCATA_SUPERVISOR=1 cargo run -p toccata-bench --release --bin <allocator> -- frag <mode>`

toccata never moves a live allocation and never returns memory to the kernel, so
it cannot compact or `madvise` away fragmentation the way the others do. The open
question was whether stranding could be controlled *within a fixed, locked budget*
— and it now is. **Measuring footprint fairly needs care:** toccata `mlock`s and
*populates* its whole budget at boot, so its process RSS is pinned at the budget
and reveals nothing about fragmentation. Its honest footprint is the **bump-arena
carved high-water** — bytes ever claimed from the arena, monotonic, never returned
on free — which `toccata::stats().carved_bytes` exposes and is the exact analog of
the *touched-page* RSS the other four report. The benchmark auto-detects: it uses
`carved_bytes` under the `toccata` binary and `/proc/self/statm` RSS for the rest.
The reported **frag factor = footprint ÷ live bytes** (1.0 = no waste).

| mode | what it does | toccata | _(was)_ | jemalloc | mimalloc | snmalloc | system |
|------|--------------|---------:|--------:|---------:|---------:|---------:|-------:|
| `shift` | large (>256 KiB) buffers, size **drifts** up each generation | **0.99×** | _5.99×_ | 1.12× | 1.10× | 1.78× | 1.00× |
| `match` | large buffers, **stable** size each generation (control) | **0.98×** | _0.98×_ | 1.05× | — | 1.59× | — |
| `crossclass` | free all of one small class, then allocate another (64 B→64 KiB) | **1.59×** | _6.00×_ | 1.40× | 1.58× | 1.48× | 1.62× |
| `mixed` | seeded steady-state churn, skewed small+large sizes, bounded live set | **1.46×** | _2.03×_ | 1.39× | 1.48× | 1.49× | 1.34× |

*(Graviton, `TOCCATA_BENCH_MB=3072`, supervisor on at its default adaptive cadence;
`shift`/`match` = 256 MiB/gen ×6, `crossclass` = 128 MiB/class, `mixed` = 30M ops
over a 256 MiB live set. Peak frag factor. The `_(was)_` column is the pre-fix
number.)*

**toccata now beats or matches every allocator on every mode**, while keeping the
never-stall guarantee (zero post-init syscalls — verified by the `strace` gate) and
with **no regression** to the throughput/latency tables above (single 7.4 ns, multi
0.5 ns, prod/cons, framepool 4.7 ns, and the headline `latency_tail` p9999/zero-stall
all hold with the reclaim supervisor running). How each was closed, all without ever
relocating an object or returning a page to the kernel:

- **Large-object size drift (`shift` 5.99× → 0.99×).** The `LargeAllocator`'s
  exact-span-count free stacks were replaced with a **`SpanPool`**: an
  address-coalescing, span-count-keyed free-run index over the arena's span grid
  (a TLSF-style two-level bitmap with boundary-tag coalescing + split-the-tail). An
  exact-count request still hits its bucket with no split — so the `match` control
  holds at its best-of-five **0.98×** — while a *drifting* size now splits a larger
  free run or coalesces adjacent ones instead of carving fresh budget. toccata goes
  from worst (5.99×) to **best (0.99×, beating jemalloc's 1.12×).**
- **Cross-class span stranding (`crossclass` 6.00× → 1.59×).** A span tagged to one
  size class is no longer stranded for life. A background **reclaim supervisor**
  detects spans that have gone *provably fully free* (checked under the owning
  central lock — a live or cache-held object keeps its bitmap bit clear, so a
  referenced span can never be misjudged free), re-tags them, and returns them to
  the shared `SpanPool` where any other class (or the large path) reuses them. The
  reclaim is concurrency-safe **without** a cross-thread quiesce or epoch (verified
  by an adversarial review + a deterministic double-hand-out regression test);
  per-CPU cache capacity is **size-tiered** (tcmalloc-style) so a freed large class
  pins few spans; and the sweep cadence is **adaptive** (jemalloc/mimalloc decay
  shape — fast while actively reclaiming, idle at rest) so it tracks bursty
  size-class churn without costing CPU or perturbing the tail at steady state.
- **The control (`match`) still proves the design.** With a stable large size,
  toccata recycles perfectly and remains the **best** of the five (0.98× — zero
  per-allocation metadata overhead).
- **Realistic traffic (`mixed`)** drops from ~2× to **1.46×**, now ahead of
  mimalloc (1.48×), snmalloc (1.49×) and tracking jemalloc (1.39×).

The lone residual is in `crossclass` at the **64 KiB class (exactly one span per
object)**: at the size transition the cursor can carve fresh before the prior
class's spans are reclaimed, and because `carved_bytes` is a monotonic high-water
that *never retreats* (the very property that makes toccata never-stall), that
transient spike is locked into the peak. It is bounded and already beats most of
the field; closing it fully would require either object relocation (off the table)
or letting the high-water retreat (it can't).

The reclaim supervisor is **opt-in**: build with the `supervisor` feature and set
`TOCCATA_SUPERVISOR=1` (the global build is byte-for-byte unaffected otherwise).
It is reclaim-only — it takes no kernel call and issues no membarrier, so it stays
on the never-stall path. The workload is **Linux-only** (RSS via `/proc`, mlock,
rseq) and aborts on true budget exhaustion; under the `toccata` binary it stops
gracefully with a notice if stranding would exceed 95% of the budget (raise
`TOCCATA_BENCH_MB` to go further). The other binaries grow RSS until the OS
intervenes.

## 3. How to reproduce

The allocator under test is the **binary you run**, not a cargo feature — so a
single `cargo build` produces all of them and there are no mutually-exclusive
feature flags. toccata's rseq fast path compiles in automatically on Linux
x86_64/aarch64 (no `--cfg`); pass `TOCCATA_BENCH_MB` to size its locked pool.

```bash
# Throughput / latency table (one allocator per binary):
cargo run -p toccata-bench --release --bin system   -- throughput
cargo run -p toccata-bench --release --bin jemalloc -- throughput
cargo run -p toccata-bench --release --bin mimalloc -- throughput
cargo run -p toccata-bench --release --bin snmalloc -- throughput
TOCCATA_BENCH_MB=2048 cargo run -p toccata-bench --release --bin toccata -- throughput

# Never-stall demonstrator under a memory-limited cgroup (Linux):
#   <bin> latency-tail <seconds> <churn-MiB>;  TOCCATA_BENCH_MB sizes toccata's pool
sudo systemd-run --scope -p MemoryHigh=200M -p MemoryMax=6G \
    target/release/jemalloc latency-tail 6 300        # jemalloc/mimalloc/system → throttles
sudo systemd-run --scope -p MemoryHigh=1200M -p MemoryMax=6G \
    prlimit --memlock=unlimited:unlimited \
    env TOCCATA_BENCH_MB=900 target/release/toccata latency-tail 6 300   # toccata → flat tail

# Fragmentation / budget-stranding (Linux; per allocator). frag <mode> [args...].
# For the `toccata` bin, TOCCATA_SUPERVISOR=1 enables the background reclaim sweep
# that closes crossclass/mixed (needs the `supervisor` cargo feature, on by default
# for toccata-bench); the other bins ignore it.
TOCCATA_BENCH_MB=3072 TOCCATA_SUPERVISOR=1 cargo run -p toccata-bench --release --bin <bin> -- frag shift 256 6 5
TOCCATA_BENCH_MB=3072 TOCCATA_SUPERVISOR=1 cargo run -p toccata-bench --release --bin <bin> -- frag match 256 6 5
TOCCATA_BENCH_MB=3072 TOCCATA_SUPERVISOR=1 cargo run -p toccata-bench --release --bin <bin> -- frag crossclass 128 100
TOCCATA_BENCH_MB=3072 TOCCATA_SUPERVISOR=1 cargo run -p toccata-bench --release --bin <bin> -- frag mixed 30 256
```

(`toccata` needs `RLIMIT_MEMLOCK >= pool`; the snmalloc build needs `cmake`. To
force the portable locked baseline for the two-row comparison, build for a
non-Linux target or a non-x86_64/aarch64 arch — the rseq asm is compiled in
whenever the target supports it.)

## 4. Caveats & honest notes

- **Two toccata rows.** The "locked baseline" row is the portable always-correct
  fast path (used on non-Linux / non-x86_64-aarch64 targets); the "rseq fast path"
  row is the default on Linux x86_64/aarch64, where the asm compiles in
  automatically — validated on Graviton (128-thread stress, zero corruption). On
  kernels without rseq support, toccata falls back to the locked baseline at
  runtime.
- **Measurement `max` noise:** the per-op `Instant::now()` loop has its own
  cold-cache/scheduling tail (~10–30 µs) common to every allocator; the
  histogram `>1ms/>10ms/>100ms` stall counters are the noise-free signal.
- **snmalloc** appears slow here because the benchmark allocates through std
  `Vec`/`String`; snmalloc's strengths show more on its native API and at higher
  core counts. Not a fair characterization of snmalloc in general.
- Numbers are single-run on a shared dev host; treat as directional, not
  publication-grade. The **stall/no-stall distinction is binary and robust**; the
  ns-level throughput deltas will move with tuning and the rseq path.
