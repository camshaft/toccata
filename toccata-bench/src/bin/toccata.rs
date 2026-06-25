// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Benchmark shim: toccata as the `#[global_allocator]`. Reserves + locks the
//! pool up front (size via `TOCCATA_BENCH_MB`, default 2 GiB — needs a raised
//! `RLIMIT_MEMLOCK`). The atomic-free rseq fast path compiles in automatically on
//! Linux x86_64/aarch64; no flags needed.
//!
//! `cargo run -p toccata-bench --release --bin toccata -- <workload> [args...]`

#[global_allocator]
static A: toccata::Toccata = toccata::Toccata::new();

fn main() {
    // `TOCCATA_SHARDS=N` overrides the central-list shard count (default 4*ncpu)
    // for experiments; otherwise the standard `configure`.
    if let Some(n) = std::env::var("TOCCATA_SHARDS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
    {
        let builder = toccata::primitives::SubHeapBuilder::new(
            "global",
            toccata_bench::toccata_budget_bytes(),
        )
        .on_exhaust(toccata::primitives::OnExhaust::Abort)
        .num_shards(n);
        toccata::configure_with(builder);
    } else {
        toccata::configure(toccata_bench::toccata_budget_bytes());
    }
    toccata_bench::run("toccata");
}
