// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Benchmark shim: mimalloc as the `#[global_allocator]`.
//! `cargo run -p toccata-bench --release --bin mimalloc -- <workload> [args...]`

#[global_allocator]
static A: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    toccata_bench::run("mimalloc");
}
