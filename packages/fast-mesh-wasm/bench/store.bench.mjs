// Store benchmarks (`fast-mesh-wasm/store` over the synchronous `Scene`, then over a `WorkerScene`),
// run against the built package (a release build, not the test build):
//
//   pnpm build && pnpm --filter fast-mesh-wasm bench:store [-- --quick]
//
// Add `--json <path>` to also write every case (median, p95, heap) as JSON;
// console output is unchanged. The release workflow compares the head JSON
// against the latest release tag's and posts the delta table on the PR.
// Plain Node with performance.now(); every number is a median (and p95) over the iterations
// shown. The scene holds 1000 small meshes (closed tetrahedra) added through the store, 100 of
// them coloured by distance to a neighbour, and 10 sections over 100 meshes each, so a refresh
// has many mesh and section states to derive. The costs are those of the store's actions and
// events: the scene work under them is tiny (four-vertex meshes). The Worker case (the engine in a
// `worker_threads` Worker, bench/worker-harness.mjs) times the store's work on the status changes
// a placement makes: the refresh that re-derives the states reading the changed keys.
import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { Scene } from "../dist/node.js";
import { createSceneStore } from "../dist/store.js";
import { createReporter } from "./report.mjs";

const quick = process.argv.includes("--quick");
const reporter = createReporter(process.argv.slice(2), { suite: "store" });

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

/** A store over `scene` (a fresh `Scene` by default) with `MESHES` meshes; returns the add burst's time (ms) too. */
function build(scene = new Scene()) {
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

/** Median and p95 (µs) of `fn(k)` over `n` runs, after `warm` unmeasured ones; `k` counts all of them. */
function time(n, warm, fn) {
  for (let k = 0; k < warm; k++) {
    fn(k);
  }
  const samples = [];
  for (let k = warm; k < warm + n; k++) {
    const t0 = now();
    fn(k);
    samples.push((now() - t0) * 1000);
  }
  return stats(samples);
}

const row = (name, s, unit, note = "") => {
  reporter.log(`| ${name} | ${fmt(s.median)} | ${fmt(s.p95)} | ${s.n} |${note ? ` ${note}` : ""}`);
  reporter.record(name, s, unit);
};

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
reporter.log(`\nStore over Scene: ${MESHES} meshes (${COLOURED} coloured), ${SECTIONS} sections of 100 meshes\n`);
reporter.log(`Adding ${MESHES} meshes through the store, one after another: median ${fmt(b.median)} ms, p95 ${fmt(b.p95)} ms (${b.n} runs)`);
reporter.log(`  = ${fmt((b.median * 1000) / MESHES)} µs per add\n`);
reporter.record(`add ${MESHES} meshes (burst)`, b, "ms");

// ---- per event, on the full scene ----

const { scene, store, meshes } = build();
reporter.log("| per event (µs) | median | p95 | n |");
reporter.log("| --- | --- | --- | --- |");
row("`select(mesh)` (store only)", time(iters, 100, (k) => store.select(meshes[k % MESHES])), "µs");
row(
  "`setDisplay(opacity)` + `commitFrame()`",
  time(iters, 100, (k) => {
    // Every call is a change: 0.5 and 0.75 alternate per visit (1 is the default).
    store.setDisplay(meshes[k % MESHES], { opacity: Math.floor(k / MESHES) % 2 === 0 ? 0.5 : 0.75 });
    store.commitFrame();
  }),
  "µs",
);
row(
  "`setPlacement` of a coloured, sectioned mesh",
  time(iters, 100, (k) => store.setPlacement(meshes[1], { rotation: IDENTITY, translation: [0, 0, (k % 2) * 0.01] })),
  "µs",
);
row("`commitFrame()`, nothing changed", time(iters, 100, () => store.commitFrame()), "µs");
// The store may take a scene event on a microtask: reading the snapshot brings it up to date.
row(
  "scene event (`scene.commitFrame()`) + `getSnapshot()`",
  time(iters, 100, () => {
    scene.commitFrame();
    store.getSnapshot();
  }),
  "µs",
);

// ---- a burst of adds on top of the full scene ----

const extra = quick ? 100 : 500;
const t0 = now();
for (let k = 0; k < extra; k++) {
  const { positions, indices } = tetra(2 * (MESHES + k));
  store.addMesh(positions, indices);
}
const t = now() - t0;
reporter.log(`\nAdding ${extra} more meshes: ${fmt(t)} ms (${fmt((t * 1000) / extra)} µs per add)`);
reporter.record(`add ${extra} more meshes`, { median: t, p95: t, n: 1 }, "ms");
store.dispose();
scene.free();

// ---- over a Worker scene: the status changes of a placement ----

/** The bench harness Worker as a `WorkerLike`. */
class HarnessWorker {
  constructor() {
    this.worker = new Worker(new URL("./worker-harness.mjs", import.meta.url));
    this.message = [];
    this.worker.on("message", (data) => {
      if (data?.kind === "benchStats") {
        return;
      }
      for (const cb of this.message) {
        cb({ data });
      }
    });
  }
  postMessage(msg, transfer) {
    this.worker.postMessage(msg, transfer);
  }
  addEventListener(type, cb) {
    if (type === "message") {
      this.message.push(cb);
    }
  }
  terminate() {
    void this.worker.terminate();
  }
}

const { createWorkerScene } = await import("../dist/worker.js");
const ws = await createWorkerScene({ createWorker: () => new HarnessWorker() });
// Each status delivery runs the scene's `onStatus` listeners in order: the first one here, the
// store's, then the last one here, which reads the snapshot. Between the two is the store's work
// on the delivery, whether it refreshes in its own listener or when the snapshot is read.
let t0w = 0;
let armed = false;
const deliveries = [];
ws.onStatus(() => {
  t0w = now();
});
const worker = build(ws);
ws.onStatus(() => {
  worker.store.getSnapshot();
  if (armed) {
    deliveries.push((now() - t0w) * 1000);
  }
});
const idle = async () => {
  for (let k = 0; k < 20000 && worker.store.getSnapshot().status === "computing"; k++) {
    worker.store.commitFrame();
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
};
await idle();
const rounds = quick ? 50 : 300;
const actions = [];
armed = true;
for (let k = 0; k < rounds; k++) {
  const t = now();
  worker.store.setPlacement(worker.meshes[1], { rotation: IDENTITY, translation: [0, 0, (k % 2) * 0.01] });
  actions.push((now() - t) * 1000);
  await idle();
}
armed = false;
reporter.log(`\nWorker scene, ${MESHES} meshes as above: \`setPlacement\` of a coloured mesh in a section, ${rounds} round trips\n`);
reporter.log("| per event (µs) | median | p95 | n |");
reporter.log("| --- | --- | --- | --- |");
row("`setPlacement` (the action: its key goes computing)", stats(actions), "µs");
row("a status delivery (results arrive)", stats(deliveries), "µs");
worker.store.dispose();
ws.terminate();
reporter.save();
