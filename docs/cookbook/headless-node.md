# Headless Node pipeline

Batch repair and batch distance checks in Node.js — no browser, no
canvas. (This is the documented outcome of the Node worker gap: use the
sync `Scene` in Node, and only reach for `WorkerScene` through a
`worker_threads` adapter when you need the engine off the main thread.)

## Option A: sync `Scene` (recommended)

In Node, `fast-mesh-wasm` resolves to the Node entry, which reads the
`.wasm` from disk and instantiates it synchronously on import, so
`init()` is a no-op kept for API parity. Every call below blocks the
event loop while it runs — fine for batch jobs, and the simplest
correct pipeline:

```ts
// pipeline.mjs
import { Scene } from "fast-mesh-wasm";
import { exportGLTF, exportSTL, importGLTF } from "fast-mesh-wasm/io";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// No init() needed; the module is ready on import.
const scene = new Scene();

// A unit box: 8 vertices, 12 triangles, outward winding.
const boxPositions = new Float64Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
const boxIndices = new Uint32Array([
  0, 3, 2, 0, 2, 1, // bottom (z = 0)
  4, 5, 6, 4, 6, 7, // top (z = 1)
  0, 1, 5, 0, 5, 4, // front (y = 0)
  2, 3, 7, 2, 7, 6, // back (y = 1)
  0, 4, 7, 0, 7, 3, // left (x = 0)
  1, 2, 6, 1, 6, 5, // right (x = 1)
]);

// In your pipeline the .glb files already exist; here one is written
// first so the snippet runs standalone.
const dir = mkdtempSync(join(tmpdir(), "fast-mesh-"));
{
  const template = scene.addMesh(boxPositions, boxIndices);
  writeFileSync(join(dir, "assembly.glb"), exportGLTF(scene, template));
  scene.removeMesh(template);
}

const parts = importGLTF(readFileSync(join(dir, "assembly.glb")));
for (const part of parts) {
  // positions are world-baked; part.placement holds the node matrix.
  const mesh = scene.addMesh(part.positions, part.indices);
  const m = scene.measure(mesh);
  if (!m.closed) {
    scene.repair(mesh, [
      { op: "weld", tolerance: 1e-6 },
      { op: "removeDegenerate" },
      { op: "orient" },
      { op: "fillHoles", maxEdges: 64 },
    ]);
  }
  const stl = exportSTL(scene, mesh);
  if (stl.length === 0) throw new Error("expected STL bytes");
  writeFileSync(join(dir, `${part.name}.stl`), stl);
}
scene.free();
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
const workerScene = await createWorkerScene({
  createWorker: () => {
    const worker = new Worker(new URL("./worker.js", import.meta.url), {
      workerData: { serve: true },
    });
    return {
      postMessage: (msg: unknown, transfer?: Transferable[]) => worker.postMessage(msg, transfer),
      addEventListener: (type: string, cb: (e: { data: unknown }) => void) => {
        if (type === "message") worker.on("message", (data) => cb({ data }));
        else worker.on("error", cb);
      },
      terminate: () => void worker.terminate(),
    };
  },
});

// Node is always single-threaded (no crossOriginIsolated): threads is ignored.
if (workerScene.threads !== 1) throw new Error("expected the single-threaded engine");

const wmesh = workerScene.addMesh(boxPositions, boxIndices);
workerScene.commitFrame(); // flush each frame; or await a query (auto-flushes).
await workerScene.analyze(wmesh);
const key = `mesh:${workerScene.idOf(wmesh)}`;
if (workerScene.status(key).status !== "ready") throw new Error("expected the mesh to land");
if (workerScene.views(wmesh).positions().length === 0) throw new Error("expected positions");
workerScene.terminate();
```

Which to pick:

- One-shot scripts, CI jobs, small-to-medium meshes → Option A.
- A long-lived Node service where a 1M+ triangle add would stall the
  event loop → Option B (background adds at 65,536 triangles by
  default; tune with `backgroundAddTriangles`).
