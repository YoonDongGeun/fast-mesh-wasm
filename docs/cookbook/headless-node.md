# Headless Node pipeline

Batch repair and batch distance checks in Node.js — no browser, no
canvas. (This is the documented outcome of the Node worker gap: use the
sync `Scene` in Node, and only reach for `WorkerScene` through a
`worker_threads` adapter when you need the engine off the main thread.)

## Option A: sync `Scene` (recommended)

`fast-mesh-wasm/node` reads the `.wasm` from disk and instantiates it
synchronously on import, so `init()` is a no-op kept for API parity.
Every call below blocks the event loop while it runs — fine for batch
jobs, and the simplest correct pipeline:

```ts
// pipeline.mjs
import { Scene } from "fast-mesh-wasm/node";
import { exportSTL, importGLTF } from "fast-mesh-wasm/io";
import { readFileSync, writeFileSync } from "node:fs";

// No init() needed; the module is ready on import.

const scene = new Scene();
for (const part of importGLTF(readFileSync("assembly.glb"))) {
  // positions are world-baked; part.placement holds the node matrix.
  const mesh = scene.addMesh(part.positions, part.indices);
  const m = scene.measure(mesh);
  if (!m.closed) {
    scene.repair(mesh, [
      { op: "weld", tolerance: 1e-6 },
      { op: "removeDegenerate" },
      { op: "orient" },
      { op: "fillHoles" },
    ]);
  }
  writeFileSync(`out/${part.name}.stl`, exportSTL(scene, mesh));
}
```

On `WorkerScene` state changes are async and tracked per key via
`scene.status(...)`; note `measure` currently exists only on the sync
`Scene` (WorkerScene parity gap — see
`docs/1.0-api-stabilization-checklist.md`).

## Option B: `WorkerScene` in Node via `createWorker`

`fast-mesh-wasm/worker` is browser-first (its default entry spawns a
module `Worker` running `worker-entry.js`), but `createWorkerScene`
accepts any `WorkerLike`. In Node >= 22, wrap a `worker_threads`
`Worker` that serves the engine:

```ts
import { Worker } from "node:worker_threads";
import { createWorkerScene } from "fast-mesh-wasm/worker";

// worker.js runs the engine — see test/harness/node-worker.mjs in this
// repo for a full example. It listens on parentPort and posts results back.
const scene = await createWorkerScene({
  threads: 1, // single-threaded engine: Node has no nested Web Workers.
  createWorker: () => {
    const worker = new Worker(new URL("./worker.js", import.meta.url));
    return {
      postMessage: (msg, transfer) => worker.postMessage(msg, transfer),
      addEventListener: (type, cb) => {
        if (type === "message") worker.on("message", (data) => cb({ data }));
        else worker.on("error", cb);
      },
      terminate: () => void worker.terminate(),
    };
  },
});

scene.commitFrame(); // flush each frame; or await a query (auto-flushes).
```

Which to pick:

- One-shot scripts, CI jobs, small-to-medium meshes → Option A.
- A long-lived Node service where a 1M+ triangle add would stall the
  event loop → Option B (background adds at 65,536 triangles by
  default; tune with `backgroundAddTriangles`).
