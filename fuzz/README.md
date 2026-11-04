# Fuzzing

Short-lived [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html)
targets for the geometry kernels, in the standard `fuzz/` layout. The crate is
detached from the workspace (its own `[workspace]` in
[`Cargo.toml`](Cargo.toml)), so it never affects the normal build, the MSRV
check or the published package. It needs a nightly toolchain, like every
cargo-fuzz project.

## Targets

| Target        | What it fuzzes |
| ------------- | -------------- |
| `fill`        | `fill_holes` on random soup (ragged boundary) and grids with punched-out cells (clean square holes), with capped options |
| `weld`        | `weld` on random soup at small tolerances (quantized coordinates give coincident vertices) |
| `triangulate` | `triangulate` on arbitrary polygons with up to two holes; asserts every emitted triangle numbers a real point |
| `bvh`         | `Bvh::build` plus `query_aabb` through both buffer constructors (`from_buffers`, `from_buffers_f64`); asserts every hit is a live triangle |
| `boolean`     | `boolean_plan` plus `apply_boolean` on soup pairs (input rejection) and box pairs (split and classify) |

Every target fuzzes the kernels' no-panic contract over small inputs; the
`triangulate` and `bvh` targets additionally assert output invariants. Inputs
are finite and bounded (see
[`common.rs`](fuzz_targets/common.rs)); wider input spaces (non-finite
coordinates, larger meshes) are future work once the current ones run clean.

There are no file-format parsers in Rust to fuzz (STL/OBJ/PLY live in
TypeScript): the buffer constructors are the parsers here, and the `bvh`
target covers both.

## Running locally

```sh
cargo install cargo-fuzz --locked
rustup toolchain install nightly --component rust-src
cd fuzz
cargo +nightly fuzz run fill -- -max_total_time=60
```

Useful flags: `-runs=N` for a fixed number of inputs instead of a time budget,
`-max_len=4096` to cap input size, `-rss_limit_mb=2048` to catch memory blowups.
List targets with `cargo +nightly fuzz list`. A crash writes its repro to
`artifacts/`; replay it with `cargo +nightly fuzz run fill artifacts/fill-<...>`.

## CI

[`.github/workflows/fuzz.yml`](../.github/workflows/fuzz.yml) runs every target
briefly (seconds each) on pull requests that touch `crates/`, `fuzz/` or the
workflow itself, and longer (minutes each) on a nightly schedule, reusing the
pinned wasm nightly so no second toolchain is installed. Only that job installs
nightly; everything else stays on stable.

Crash policy: on failure the job uploads `fuzz/artifacts/` (the libFuzzer repro
inputs) as the `fuzz-artifacts` artifact instead of filing an issue, so one
flaky crash cannot spam the tracker. To triage: download the artifact, replay it
as above, minimize with `-merge=1`, then file an issue with the minimized repro
attached and the target name in the title.

## Adding a target

1. Add a `[[bin]]` entry in [`Cargo.toml`](Cargo.toml) and a
   `fuzz_targets/<name>.rs` file (reuse `common.rs` via
   `#[path = "common.rs"] mod common;`).
2. Keep inputs small and every option that scales work capped; assert output
   invariants where the kernel documents them.
3. Add the target name to the `TARGETS` list in `fuzz.yml`.
