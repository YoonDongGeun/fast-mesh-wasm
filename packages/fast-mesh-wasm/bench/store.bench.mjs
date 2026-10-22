// Store benchmarks (`fast-mesh-wasm/store` over the synchronous `Scene`), run against the built
// package (a release build, not the test build):
//
//   pnpm build && pnpm --filter fast-mesh-wasm bench:store [-- --quick]
//
// Plain Node with performance.now(); every number is a median (and p95) over the iterations
// shown. The scene holds 1000 small meshes (closed tetrahedra) added through the store, 100 of
// them coloured by distance to a neighbour, and 10 sections over 100 meshes each, so a refresh
// has many mesh and section states to derive. The costs are those of the store's actions and
// events: the scene work under them is tiny (four-vertex meshes).
import { readFileSync } from "node:fs";
import { Scene } from "../dist/node.js";
import { createSceneStore } from "../dist/store.js";

const quick = process.argv.includes("--quick");

if (readFileSync(new URL("../dist/wasm/fast_mesh_wasm.d.ts", import.meta.url), "utf8").includes("__test_panic")) {
  console.error("dist/ is a test build: run `pnpm build` first (benches need the release build)");
  process.exit(1);
}

const now = () => performance.now();

function stats(samples) {
  const s = Float64Array.from(samples).sort();
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { median: q(0.5), p95: q(0.95), n: s.length };
}

const fmt = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));

/** A closed tetrahedron offset by `dx` along x. */
function tetra(dx) {
  return {
    positions: Float64Array.from([dx, 0, 0, dx + 1, 0, 0, dx, 1, 0, dx, 0, 1]),
    indices: Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]),
  };
}

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const MESHES = 1000;
const COLOURED = 100;
const SECTIONS = 10;

/** A store over a fresh scene with `MESHES` meshes; returns the add burst's time (ms) too. */
function build() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const meshes = [];
  const t0 = now();
  for (let k = 0; k < MESHES; k++) {
    const { positions, indices } = tetra(2 * k);
    meshes.push(store.addMesh(positions, indices));
  }
  const burst = now() - t0;
  for (let k = 0; k < COLOURED; k++) {
    store.setDisplay(meshes[k], {
      coloring: { distanceTo: meshes[k + 1], maxDistance: 10, map: "viridis", range: [0, 5] },
    });
  }
  for (let s = 0; s < SECTIONS; s++) {
    store.addSection({
      plane: { origin: [0, 0, 0.5], normal: [0, 0, 1] },
      meshes: meshes.slice(100 * s, 100 * s + 100),
    });
  }
  store.commitFrame();
  return { scene, store, meshes, burst };
}

/** Median and p95 (µs) of `fn` over `n` runs, after `warm` unmeasured ones. */
function time(n, warm, fn) {
  for (let k = 0; k < warm; k++) {
    fn(k);
  }
  const samples = [];
  for (let k = 0; k < n; k++) {
    const t0 = now();
    fn(k);
    samples.push((now() - t0) * 1000);
  }
  return stats(samples);
}

const row = (name, s, note = "") =>
  console.log(`| ${name} | ${fmt(s.median)} | ${fmt(s.p95)} | ${s.n} |${note ? ` ${note}` : ""}`);

const runs = quick ? 3 : 10;
const iters = quick ? 200 : 2000;

// ---- the N-add burst ----

const bursts = [];
for (let r = 0; r < runs; r++) {
  const { scene, store, burst } = build();
  bursts.push(burst);
  store.dispose();
  scene.free();
}
const b = stats(bursts);
console.log(`\nStore over Scene: ${MESHES} meshes (${COLOURED} coloured), ${SECTIONS} sections of 100 meshes\n`);
console.log(`Adding ${MESHES} meshes through the store, one after another: median ${fmt(b.median)} ms, p95 ${fmt(b.p95)} ms (${b.n} runs)`);
console.log(`  = ${fmt((b.median * 1000) / MESHES)} µs per add\n`);

// ---- per event, on the full scene ----

const { scene, store, meshes } = build();
console.log("| per event (µs) | median | p95 | n |");
console.log("| --- | --- | --- | --- |");
row("`select(mesh)` (store only)", time(iters, 100, (k) => store.select(meshes[k % MESHES])));
row(
  "`setDisplay(opacity)` + `commitFrame()`",
  time(iters, 100, (k) => {
    store.setDisplay(meshes[k % MESHES], { opacity: k % 2 === 0 ? 0.5 : 1 });
    store.commitFrame();
  }),
);
row(
  "`setPlacement` of a coloured, sectioned mesh",
  time(iters, 100, (k) => store.setPlacement(meshes[1], { rotation: IDENTITY, translation: [0, 0, (k % 2) * 0.01] })),
);
row("`commitFrame()`, nothing changed", time(iters, 100, () => store.commitFrame()));
// The store may take a scene event on a microtask: reading the snapshot brings it up to date.
row(
  "scene event (`scene.commitFrame()`) + `getSnapshot()`",
  time(iters, 100, () => {
    scene.commitFrame();
    store.getSnapshot();
  }),
);

// ---- a burst of adds on top of the full scene ----

const extra = quick ? 100 : 500;
const t0 = now();
for (let k = 0; k < extra; k++) {
  const { positions, indices } = tetra(2 * (MESHES + k));
  store.addMesh(positions, indices);
}
const t = now() - t0;
console.log(`\nAdding ${extra} more meshes: ${fmt(t)} ms (${fmt((t * 1000) / extra)} µs per add)`);
store.dispose();
scene.free();
