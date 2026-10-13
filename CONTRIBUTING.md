# Contributing to fast-mesh-wasm

Thanks for your interest! Bug reports, docs fixes, new mesh operations and performance work are all welcome.
For anything larger than a small fix, please open an issue or a discussion first so we can agree on the approach.

## Project layout

```
crates/fast-mesh/        Pure Rust algorithms. No WebAssembly code. Most work happens here.
crates/fast-mesh-wasm/   #[wasm_bindgen] wrappers that expose the core to JavaScript.
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
| `pnpm test` | `cargo test` for the core, then Vitest against the built package |
| `pnpm lint` | `cargo fmt --check` and `cargo clippy -D warnings` |
| `pnpm check:pkg` | publint and Are The Types Wrong on the package |
| `pnpm smoke` | Packs the package and installs it with npm, pnpm and bun in temporary projects |
| `pnpm --filter fast-mesh-wasm-example-vite dev` | Runs the browser example |

Re-run `pnpm build` after changing Rust code; the JS tests import the built output.

## Adding or changing an API

1. Implement and unit-test the algorithm in `crates/fast-mesh/src`. Keep it free of WebAssembly types.
2. Expose it in `crates/fast-mesh-wasm/src/lib.rs` with `#[wasm_bindgen(js_name = camelCaseName)]`.
3. Write `///` doc comments with `@param`, `@returns` and `@throws`. They become the JSDoc users see in their editor.
4. Re-export it from both `packages/fast-mesh-wasm/src/web.ts` and `src/node.ts`.
5. Add a test in `packages/fast-mesh-wasm/test/` and update the API table in `README.md`.
6. Add a changeset (see below).

Prefer typed arrays (`&[f32]`, `&[u32]`, `Vec<f32>`) over `JsValue` in signatures, so the generated types stay precise and data is not serialized.

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
