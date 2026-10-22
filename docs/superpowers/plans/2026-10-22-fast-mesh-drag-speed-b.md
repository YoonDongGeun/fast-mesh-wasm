# Wide-range drag colouring, part B (wasm threads build) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a second, threaded wasm artifact. It is built with atomics, shared memory and rayon through `wasm-bindgen-rayon`. The Worker scene loads it on cross-origin-isolated pages and falls back to the single-threaded artifact everywhere else, with identical results. The docs tell users how to enable COOP/COEP.

**Architecture:**
- `scripts/build-wasm.mjs --threads` builds `fast-mesh-wasm` on a pinned nightly with `-Z build-std` and the `threads` crate feature, which enables `parallel` and `wasm-bindgen-rayon`. The output goes to `src/wasm-threads/`, next to the current `src/wasm/`.
- The Worker entry chooses the artifact at start-up. It uses the threaded one when `crossOriginIsolated` is true and `threads !== 1`, then calls `initThreadPool(n)`. Any failure falls back to the single-threaded artifact and warns once in development builds.
- Core entries and the main-thread `Scene` never load the threaded artifact.

**Tech Stack:** Rust stable 1.99 (MSRV 1.88) plus a pinned nightly for the threaded artifact; `wasm-bindgen =0.2.129`; `wasm-bindgen-rayon`; tsdown; vitest; headless Chrome for browser checks.

**Spec:** `docs/superpowers/specs/2026-10-20-fast-mesh-drag-speed-design.md` (§2 goals 4–5, §3, §6.5–6.6; §8 holds the part A notes and the open points for B).

## Global Constraints

- **Toolchains.** Stable for everything except the threaded artifact. That artifact uses a pinned nightly (`-Z build-std=panic_abort,std`, `RUSTFLAGS=-C target-feature=+atomics,+bulk-memory,+mutable-globals`, plus the link args `wasm-bindgen-rayon` requires).
  - The nightly date lives in one place: a constant in `scripts/build-wasm.mjs`, which CI reads too.
  - If `wasm-bindgen-rayon` needs a different `wasm-bindgen` than `=0.2.129`, bump `wasm-bindgen` and `wasm-bindgen-cli` together, and say so in the PR.
- **Package.**
  - The threaded artifact and its glue (including `wasm-bindgen-rayon`'s worker helpers) live in `dist/wasm-threads/`.
  - Only the Worker entry (`worker-entry.js`) may import it. Core entries (`.`, `./three`, `./store`, `./react`, `./r3f`, `./worker`) and the main-thread `Scene` must not.
  - publint and attw stay green.
- **Selection.**
  - The Worker uses the threaded artifact when `crossOriginIsolated` is true and the `threads` option is not 1. `createWorkerScene({ threads?: number })` defaults to `navigator.hardwareConcurrency` capped at 8; `threads: 1` forces the single-threaded artifact.
  - On any threaded start-up failure (import, instantiate, shared memory or `initThreadPool`), the Worker falls back to the single-threaded artifact and the scene works.
  - Development builds warn once, for the fallback and for a non-isolated page. Development means `import.meta.env?.DEV` or the `debug` option.
- **Identical results.** Threaded and single-threaded give bit-identical values, colours and change records (parallel stage 1 guarantees this natively).
- **Memory.** Shared memory needs a maximum. Set it to 4 GiB (wasm32's limit), so 10M scenes still fit. Record the initial and maximum memory.
- **Browsers.** Current Chrome/Edge, Firefox and Safari. CI runs headless Chromium. Firefox and WebKit are checked once (Playwright, if installable) and recorded.
- **Bundlers.** The goal is no configuration beyond the COOP/COEP headers in Vite and webpack 5, tested against an `npm pack` install. Otherwise document the minimal setting and test it.
- **Gate.** The part A gate, plus:
  - `node scripts/build-wasm.mjs --threads` (or the package build that includes it);
  - both example builds with COOP/COEP headers set;
  - a final `pnpm build`, so the release `dist` contains both artifacts without `__test_panic`.

## Review Focus

1. **Isolated page, nested workers fail.** On an isolated page where nested workers or `initThreadPool` fail, the scene still starts, single-threaded, warns once, and every API works. Owned by Task 2: `threaded_start_failure_falls_back`.
2. **Restart.** `restart()` on a threaded scene starts a new Worker and a new pool. Threads keep working after the restart, and the old pool's workers are terminated. Owned by Task 2: `restart_reinitialises_pool` (browser check in Task 3).
3. **Memory growth to large scenes.** Adding meshes that grow the shared memory far beyond the initial size works and never fails below the 4 GiB maximum. Owned by Task 3: `threaded_memory_grows` (browser, a few hundred MB).
4. **`threads: 1`, and non-isolated pages.** Neither loads the threaded artifact at all: no fetch of `wasm-threads/*`. Owned by Task 2: `single_thread_never_loads_threaded`.
5. **Package boundary.** No core entry's import graph reaches `wasm-threads/`. Owned by Task 1: `package_threads_boundary`.

---

### Task 1: Threaded artifact build

**Files:**
- Modify: `crates/fast-mesh-wasm/Cargo.toml`. Add the feature `threads = ["parallel", "dep:wasm-bindgen-rayon"]`; also add a `parallel` feature forwarding to `fast-mesh-scene/parallel` if none exists.
- Modify: `crates/fast-mesh-wasm/src/lib.rs`. Under `#[cfg(feature = "threads")]`, re-export `pub use wasm_bindgen_rayon::init_thread_pool;` (JS name `initThreadPool`).
- Modify: `scripts/build-wasm.mjs`. Add `--threads`: nightly, build-std, RUSTFLAGS, `wasm-bindgen --target web` into `src/wasm-threads/`, then wasm-opt with `--enable-threads`. Add the `NIGHTLY` constant.
- Modify: `packages/fast-mesh-wasm/package.json` and `tsdown.config.ts`. The `build` script builds both artifacts. Copy `src/wasm-threads` to `dist/wasm-threads`. Add `neverBundle` for the threaded glue.
- Modify: `.github/workflows/ci.yml`. Install the pinned nightly with `rust-src` and `wasm32-unknown-unknown`, and build the threaded artifact.
- Test: `packages/fast-mesh-wasm/test/threads-package.test.ts`.

**Interfaces:**
- **Produces:** `dist/wasm-threads/fast_mesh_wasm.js`, its `.wasm` and the rayon worker helper files. The glue exports `default init` and `initThreadPool(n: number): Promise<void>`, plus the same `WasmScene` API as the single-threaded glue.
- **Produces:** a `NIGHTLY` constant (for example `"nightly-2026-10-01"`; the builder picks a version that builds) that CI also uses.

- [ ] **Step 1: Write the failing tests** in `threads-package.test.ts`:
  - `threaded_artifact_built`: `dist/wasm-threads/fast_mesh_wasm.js` exists and exports `initThreadPool`. The `.wasm` imports a shared memory with a maximum of 65536 pages, checked by parsing the import section, for example with `WebAssembly.Module.imports` on the compiled module under Node.
  - `package_threads_boundary` (Review Focus 5): walk the transitive `dist` import graph of every entry except `worker-entry.js`. None reaches `wasm-threads/`.
  - `same_api_both_artifacts`: the `WasmScene` method names in `dist/wasm/fast_mesh_wasm.d.ts` and `dist/wasm-threads/fast_mesh_wasm.d.ts` are equal.
- [ ] **Step 2: Run** `pnpm --filter fast-mesh-wasm exec vitest run test/threads-package.test.ts`. Expected: FAIL (no artifact).
- [ ] **Step 3: Implement.**
  - Install the nightly with `rustup toolchain install <NIGHTLY> --component rust-src --target wasm32-unknown-unknown`.
  - Find a `wasm-bindgen-rayon` version compatible with `wasm-bindgen =0.2.129`, or bump both, and record the choice in the report.
- [ ] **Step 4: Run** the test, then the full gate. Expected: PASS. Record the artifact sizes (raw and gzipped) in the report.
- [ ] **Step 5: Commit** `build(wasm): threaded artifact on a pinned nightly`.

### Task 2: Worker artifact selection, pool start-up and fallback

**Files:**
- Modify: `packages/fast-mesh-wasm/src/worker/entry.ts`. Choose and load the artifact, call `initThreadPool`, fall back on failure and warn once.
- Modify: `packages/fast-mesh-wasm/src/worker/client.ts` and `protocol.ts`. Add the `threads?: number` option. Pass `threads` and `debug` in the init message. The ready message reports `{ threads: number }`, which is exposed as `WorkerScene.threads` (read-only: 1 when single-threaded).
- Modify: `packages/fast-mesh-wasm/test/harness/node-worker.mjs`, so a test can inject a failing threaded loader.
- Test: `packages/fast-mesh-wasm/test/worker-threads.test.ts`.

**Interfaces:**
- **Produces:** `createWorkerScene({ threads?: number })`, `WorkerScene.threads: number`, and an internal `loadEngine({ threads, isolated, importThreaded, importSingle, warn }): Promise<{ wasm, threads }>` in `entry.ts`. Loaders are injectable for tests.

- [ ] **Step 1: Write the failing tests** in `worker-threads.test.ts`, as unit tests of `loadEngine` with fake loaders, plus Node-harness tests:
  - `isolated_loads_threaded`: with `isolated: true` and `threads: 4`, it calls `importThreaded` and `initThreadPool(4)`, and returns `threads: 4`.
  - `single_thread_never_loads_threaded` (Review Focus 4): `threads: 1`, or `isolated: false`, never calls `importThreaded`.
  - `threaded_start_failure_falls_back` (Review Focus 1): if `importThreaded` rejects, or `initThreadPool` rejects, it returns the single-threaded engine with `threads: 1`, and `warn` is called exactly once (only in debug/dev).
  - `non_isolated_warns_once_in_dev`.
  - `restart_reinitialises_pool` (Review Focus 2, harness level): after `restart()`, `WorkerScene.threads` is reported again by the new ready message.
  - `threads_option_reported`: in the Node harness (not isolated), `scene.threads === 1`.
- [ ] **Step 2: Run** `pnpm --filter fast-mesh-wasm exec vitest run test/worker-threads.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - The default `threads` is `Math.min(navigator.hardwareConcurrency ?? 4, 8)`.
  - `initThreadPool` runs once per Worker.
  - The dev warning text names the README section "Multithreading (COOP/COEP)".
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** `feat(worker): threaded engine on isolated pages with fallback`.

### Task 3: Headers in the examples, browser verification and bundlers

**Files:**
- Modify: `examples/react/vite.config.ts` and `examples/webgpu/vite.config.ts`. Set `server.headers` and `preview.headers` to `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Add `optimizeDeps.exclude: ["fast-mesh-wasm"]` only if it is needed.
- Create: `scripts/browser-check.mjs`. It drives headless Chrome over CDP, or Playwright if available, against `vite preview` of an example. It runs a fixed op sequence (two terrains, wide colouring, drags, an edit, undo, a section) once with `threads: 1` and once threaded. It then compares `debugDump` output (colours, positions, field values) bit for bit, and prints `scene.threads`, timings and pass/fail.
- Create: `scripts/bundler-check.mjs`. It runs `npm pack` on the package, installs the tarball into fresh scratch apps (Vite, webpack 5) under the scratchpad or `os.tmpdir()`, builds them, serves them with the headers, and checks in headless Chrome that `scene.threads > 1`.
- Modify: `.github/workflows/ci.yml`. Run `browser-check` on Chromium. `bundler-check` runs in CI if it takes under about 5 minutes; otherwise it stays a documented manual script.

- [ ] **Step 1: Run** `node scripts/browser-check.mjs`. Expected: `threads > 1` threaded, `threads == 1` single, dumps identical, PASS.
- [ ] **Step 2: Extend the browser check.**
  - `threaded_memory_grows` (Review Focus 3): add meshes until memory passes 512 MB, with no failure.
  - `restart_reinitialises_pool`: poison, restart, and `threads > 1` again.
- [ ] **Step 3: Run** `node scripts/bundler-check.mjs` for Vite and webpack 5. Expected: `threads > 1` in both. If either needs configuration, document the minimal setting and re-run with it.
- [ ] **Step 4: Check Firefox and WebKit once** with Playwright, if it can be installed. Record the results, including part A's open point: messages arriving mid-unit.
- [ ] **Step 5: Run** the full gate. **Commit** `test(worker): browser and bundler checks for the threaded build`.

### Task 4: Benchmarks, docs and changeset

**Files:**
- Modify: `docs/benchmarks.md`. Add the "Threaded Worker" section: the wide drag at 1M and 10M (median Worker block per pose, update interval during a sustained drag, release latency), threaded against single-threaded, in headless Chrome. Measure through `browser-check.mjs --bench`, or a bench page.
- Modify: `README.md`. Add a "Multithreading (COOP/COEP)" section covering:
  - the header setup for Vite (dev and preview), webpack-dev-server and nginx/Express;
  - why COEP affects cross-origin assets (`crossorigin`, CORP);
  - the `threads` option and `WorkerScene.threads`;
  - the fallback and the dev warning;
  - that a coi-serviceworker exists for hosts without header control (mentioned only, not shipped);
  - any bundler setting Task 3 found.
- Modify: the spec. Add §9 "Part B implementation notes": the rulings, the nightly pin, the sizes and the numbers.
- Create: `.changeset/<name>.md`, a minor bump for `fast-mesh-wasm`. It covers the threaded Worker engine on cross-origin-isolated pages, the `threads` option and `WorkerScene.threads`.

- [ ] **Step 1: Run the benchmarks** in headless Chrome on a release build, and record real numbers only. If 10M takes over about 15 minutes per mode, record what was measured.
- [ ] **Step 2: Write** the docs, the spec §9 and the changeset.
- [ ] **Step 3: Run** the full gate. Expected: green.
- [ ] **Step 4: Commit** `bench(worker): threaded wide drag` and `docs(worker): multithreading and COOP/COEP`.
