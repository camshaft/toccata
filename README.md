# toccata

A **real-time, fallible, per-CPU-aware memory allocator** for Rust that **never
pauses or stalls** on a hot path. Built for high-performance networking and
disaggregated-memory systems where a multi-second allocator stall — the kind a
cgroup `memory.high` soft limit inflicts on a process that grows its RSS — drops
packets and breaks the system.

> **Prefer to fail fast over slowing down.** toccata reserves and `mlock`s its
> entire budget up front, never returns memory to the kernel, and never touches
> the kernel for memory after boot. When a budget is exhausted it returns `None`
> (fallible API) or aborts (global allocator) — a clean, instant signal, never a
> stall.

See [`BENCHMARKS.md`](BENCHMARKS.md) for full results, including the never-stall
cgroup reproduction.

## Crates

| Crate | What it provides |
|-------|------------------|
| [`toccata-core`](toccata-core) | the reusable primitive surface: up-front `mmap`+`mlock` reservation, the atomic-free **rseq** per-CPU slab (x86_64 + aarch64), size classes, span metadata, the L1 thread-cache magazine, independently-budgeted `SubHeap`s, the coalescing large-object `SpanPool`, fixed-size `FramePool`s, and optional background reclaim (`supervisor`) / live-usage (`metrics`) features. |
| [`toccata`](toccata) | the application-facing `#[global_allocator]` — `install!` / `configure` wire toccata's never-stall pool in as the process allocator, over the `toccata-core` primitives. |
| [`toccata-bench`](toccata-bench) | comparative benchmarks vs. system / jemalloc / mimalloc / snmalloc. |

The atomic-free rseq fast path is **not** a feature flag: it compiles in
automatically on Linux x86_64/aarch64 and falls back to a portable locked
baseline at runtime on kernels without rseq.

## Status

Implemented and validated on AWS Graviton (aarch64, Amazon Linux 2023, kernel
6.12). With the rseq fast path, an L1 magazine, and direct (fast) TLS, toccata
**beats jemalloc** on single-thread (7.4 ns vs 10.9 ns) and multi-thread churn
(0.5 ns vs 0.7 ns), ties it on the producer/consumer hand-off, and has the
**tightest latency tail of any allocator tested** (p9999 = 190 ns). Under a
memory-limited cgroup it is the only allocator that does not stall: a 6-second
workload completes in 6 s with **zero allocations over 1 ms**, where jemalloc is
penalty-slept by the kernel into a 50+ second runtime. See
[`BENCHMARKS.md`](BENCHMARKS.md).

The test suite (~50 tests, rseq on and off) covers: a `strace` gate proving zero
`mmap`/`madvise`/`mlock` against the pool after seal; a 128-thread occupancy-bitmap
stress of the lockless rseq Pop/Push (2× cores, forcing migration aborts) with
zero duplication/tears; cross-thread `Shared<T>` reclaim; registry configuration
+ metrics snapshot; and the global allocator running real
`Vec`/`Box`/`String`/`HashMap` under 8-thread cross-thread free.

## Quick start

As the process global allocator (abort-on-exhaustion = backpressure):

```rust
// Reserve+lock the pool in a constructor before main, so the hot path needs no
// "configured yet?" check.
fn heap_budget() -> usize {
    std::env::var("TOCCATA_HEAP_MB").ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|mb| mb << 20)
        .unwrap_or(8 << 30) // 8 GiB default
}
toccata::install!(heap_budget);

fn main() { /* all std Box/Vec/String now served by toccata */ }
```

Or wire it up explicitly:

```rust
#[global_allocator]
static GLOBAL: toccata::Toccata = toccata::Toccata::new();

fn main() {
    toccata::configure(8 << 30); // reserve+lock 8 GiB up front
    // ...
}
```

As a fallible sub-heap (return `None` instead of stalling when the budget is hit):

```rust
use toccata_core::{SubHeapBuilder, sizeclass};
let sh = SubHeapBuilder::new("packet", 256 << 20).build_standalone()?;
let class = sizeclass::class_for(1500).unwrap();
if let Some(ptr) = sh.alloc_class(class) {
    // use ptr; None means the packet sub-heap budget is exhausted (drop & retransmit)
}
```

As a typed frame pool (an object-recycler replacement; `Owned`/`Shared` handles):

```rust
use toccata_core::{typed_frame_pool, ReserveOpts};
typed_frame_pool!(pub Packets, MyPacket);          // defines Packets, ::Owned, ::Shared
Packets::configure(1 << 20, ReserveOpts::new(0))?; // n_frames; frame size is implied
let pkt = Packets::owned(MyPacket::default()).unwrap(); // Owned, frees on drop
let shared = pkt.into_shared();                         // Arc-like; last drop returns the frame
```

## Building & testing

toccata targets **Linux** (rseq / membarrier / mlock). It builds on macOS for
development (the Linux-specific paths degrade to portable stubs), but tests that
exercise the reservation and rseq must run on Linux.

```bash
cargo test --workspace                    # unit + integration tests (run on Linux)
cargo run -p toccata-bench --release --bin toccata -- throughput
```

**Deployment note:** toccata `mlock`s its whole pool, so the process needs
`RLIMIT_MEMLOCK >= pool size` (systemd `LimitMEMLOCK=infinity`, or a container
memlock limit). Otherwise it fails **loudly at init** with an actionable error —
never at runtime.

## License

Licensed under the [MIT License](LICENSE).
