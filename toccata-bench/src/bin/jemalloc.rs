// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Benchmark shim: jemalloc as the `#[global_allocator]`.
//! `cargo run -p toccata-bench --release --bin jemalloc -- <workload> [args...]`

#[global_allocator]
static A: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    toccata_bench::run("jemalloc");
}
