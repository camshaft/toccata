// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Benchmark shim: snmalloc as the `#[global_allocator]`.
//! `cargo run -p toccata-bench --release --bin snmalloc -- <workload> [args...]`

#[global_allocator]
static A: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

fn main() {
    toccata_bench::run("snmalloc");
}
