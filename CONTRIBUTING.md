# Contributing to fast-mesh-wasm

Thanks for your interest! Bug reports, docs fixes, new mesh operations and performance work are all welcome.
For anything larger than a small fix, please open an issue or a discussion first so we can agree on the approach.

## Project layout

```
crates/fast-mesh/        Pure Rust algorithms. No WebAssembly code. Most work happens here.
crates/fast-mesh-wasm/   #[wasm_bindgen] wrappers that expose the core to JavaScript.
crates/fast-mesh-store/  Editable mesh storage: slots, channels, transactions, undo/redo history.
crates/fast-mesh-bvh/    BVH over a store mesh: build, incremental update, box query.
crates/fast-mesh-query/  Raycast, closest-point, region and mesh-vs-mesh queries; plane slices, fill and gap.
crates/fast-mesh-scene/  Scene: meshes, placements, update modes, world queries, contacts, fields, plane sections, multi-mesh undo.
crates/fast-mesh-fields/  Per-vertex fields: signed distance (pseudonormal, fast winding), colormaps.
crates/fast-mesh-ops/     Mesh repair: edge maps, components, boundary loops; analysis, weld, orient, hole filling.
packages/fast-mesh-wasm/ The npm package: TypeScript entry points, tests, build config.
examples/vite/           Browser example, also used by CI to check bundler support.
fixtures/smoke/          Script that CI runs after installing the package with npm, pnpm and bun.
scripts/                 Cross-platform build and test scripts (Node.js, no bash needed).
```

## Setup

You need:

- [Rust](https://rustup.rs). The exact toolchain and the `wasm32-unknown-unknown` target are installed automatically from `rust-toolchain.toml`.
- `wasm-bindgen-cli`, the same version as the `wasm-bindgen` crate in the root `Cargo.toml`:
  ```sh
  cargo install wasm-bindgen-cli --version =0.2.129 --locked
  ```
- [Node.js](https://nodejs.org) 22.11 or newer
- [pnpm](https://pnpm.io). The repo pins its version in `packageManager`; `npm install -g pnpm` is enough.
- Optional: [Bun](https://bun.sh), to run the bun smoke test locally.

Then:

```sh
git clone https://github.com/<your-username>/fast-mesh-wasm.git
cd fast-mesh-wasm
pnpm install
pnpm build
pnpm test
```

## Everyday commands

| Command | What it does |
| --- | --- |
| `pnpm build` | Rust → `.wasm` → wasm-bindgen glue and `.d.ts` → wasm-opt → TypeScript bundle in `packages/fast-mesh-wasm/dist` |
| `pnpm test` | `cargo test` for the core, then a test build of the package and Vitest against it |
| `pnpm lint` | `cargo fmt --check` and `cargo clippy -D warnings` |
| `pnpm check:pkg` | publint and Are The Types Wrong on the package |
| `pnpm smoke` | Packs the package and installs it with npm, pnpm and bun in temporary projects |
| `cargo bench -p fast-mesh-bvh` | BVH benchmarks on 1M triangles (`FAST_MESH_BENCH_LARGE=1` adds 10M); results go in `docs/benchmarks.md` |
| `cargo bench -p fast-mesh-query` | Raycast, closest-point and region benchmarks on 1M triangles (`FAST_MESH_BENCH_LARGE=1` adds 10M); results go in `docs/benchmarks.md` |
| `cargo bench -p fast-mesh-scene` | Scene overhead against direct queries on two 1M-triangle meshes, and `find_contacts` over 2, 8 and 32 small meshes; results go in `docs/benchmarks.md` |
| `cargo bench -p fast-mesh-fields` | `SignData::build` (pseudonormal and winding) and `signed_distances` on 1M triangles; the scene benchmark adds the 64-pose field drag; results go in `docs/benchmarks.md` |
| `cargo bench -p fast-mesh-ops` | `analyze` (with and without the self-intersection check), `orient`, `fill_holes` (fair on and off), `weld` and `remove_small_components` on 1M triangles; results go in `docs/benchmarks.md` |
| `pnpm --filter fast-mesh-wasm-example-vite dev` | Runs the browser example |

The package's `test` script (`pnpm --filter fast-mesh-wasm test`) rebuilds the package first as a test build: `scripts/build-wasm.mjs --test` (or `FAST_MESH_TEST_BUILD=1`) enables the `test-panic` cargo feature, which adds a `__test_panic` export for the poisoning test. Run `pnpm build` afterwards before packing or publishing by hand; `prepublishOnly` refuses a test build, and `pnpm release` always rebuilds. The root `pnpm test` runs it too, so it also leaves `dist/` as a test build: run `pnpm build` before `pnpm smoke`, which packs `dist/` as it is.

## Adding or changing an API

1. Implement and unit-test the algorithm in `crates/fast-mesh/src`. Keep it free of WebAssembly types.
2. Expose it in `crates/fast-mesh-wasm/src/lib.rs` with `#[wasm_bindgen(js_name = camelCaseName)]`.
3. Write `///` doc comments with `@param`, `@returns` and `@throws`. They become the JSDoc users see in their editor.
4. Re-export it from both `packages/fast-mesh-wasm/src/web.ts` and `src/node.ts`.
5. Add a test in `packages/fast-mesh-wasm/test/` and update the API table in `README.md`.
6. Add a changeset (see below).

Prefer typed arrays (`&[f32]`, `&[u32]`, `Vec<f32>`) over `JsValue` in signatures, so the generated types stay precise and data is not serialized.

## Supported Node.js versions

The package supports every Node.js LTS line that has not reached end-of-life; currently Node.js 22 and newer. The lowest supported version is kept in sync in four places:

- `engines.node` in `package.json` and `packages/fast-mesh-wasm/package.json`
- the major version of `@types/node`, so code cannot use APIs that the lowest version lacks
- the lowest Node.js version in the CI `install` matrix
- this section

When that version reaches end-of-life, raise all four together in one pull request, with a `minor` changeset (dropping a runtime is a breaking change). Dependabot skips `@types/node` major updates for this reason.

## Commits and pull requests

- Branch from `main` in your fork, for example `feat/weld-vertices` or `fix/degenerate-normals`.
- Keep pull requests focused on one change. They are squash-merged, so the PR title becomes the commit message.
  Please write it in the [Conventional Commits](https://www.conventionalcommits.org) style, e.g. `feat: add weldVertices`.
- CI must pass: Rust checks, the package build and checks, and installs with npm, pnpm and bun on Linux, Windows and macOS.

## Changesets and versioning

This project follows [Semantic Versioning](https://semver.org). Every PR that changes what users get from the npm package needs a changeset:

```sh
pnpm changeset
```

Pick the bump type and write one line for the changelog. While the version is `0.x`:

| Change | Bump |
| --- | --- |
| Breaking API change | `minor` |
| New feature, bug fix, performance | `patch` |

After 1.0 the usual major / minor / patch rules apply. Docs, CI and test-only changes do not need a changeset.

## Releasing

Maintainers only. Releases are automated with [changesets/action](https://github.com/changesets/action):

1. Merged PRs with changesets make the Release workflow open or update a **Version Packages** PR. It bumps the version and writes `CHANGELOG.md`.
2. Merging that PR builds the package and publishes it to npm through [Trusted Publishing](https://docs.npmjs.com/trusted-publishers) with provenance, then creates the git tag and GitHub release.

No npm token is stored in the repository.

## License

By contributing, you agree that your contributions will be dual licensed under the MIT and Apache-2.0 licenses, as described in the [README](README.md#license).
