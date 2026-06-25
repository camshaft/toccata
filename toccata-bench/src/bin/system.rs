// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Benchmark shim: the default system allocator (no `#[global_allocator]`).
//! `cargo run -p toccata-bench --release --bin system -- <workload> [args...]`

fn main() {
    toccata_bench::run("system");
}
