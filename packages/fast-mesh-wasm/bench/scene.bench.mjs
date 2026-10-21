// Scene benchmarks, run against the built package (a release build, not the test build):
//
//   pnpm build && pnpm --filter fast-mesh-wasm bench [-- --10m] [-- --quick] [-- --adapter] [-- --worker] [-- --wide]
//
// Plain Node with performance.now(); every number is a median (and p95) over the iterations
// shown. Nothing here uploads to a GPU: a "frame" is what an adapter does on the JS side,
// which is setPlacement, commitFrame() and copying every dirty range from the views into a
// JS mirror (the three.js adapter's job, spec section 11 item 2).
import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { Scene } from "../dist/node.js";
import { bindMesh } from "../dist/three.js";

const args = new Set(process.argv.slice(2));
const quick = args.has("--quick");
const with10m = args.has("--10m");
const onlyAdapter = args.has("--adapter"); // run just the three.js adapter case
const onlyWorker = args.has("--worker"); // run just the Worker scene cases (with --10m: also 10M)
const onlyWide = args.has("--wide"); // with --worker: run just the Worker wide-range drag cases

if (readFileSync(new URL("../dist/wasm/fast_mesh_wasm.d.ts", import.meta.url), "utf8").includes("__test_panic")) {
  console.error("dist/ is a test build: run `pnpm build` first (benches need the release build)");
  process.exit(1);
}

// ---- helpers ----

const now = () => performance.now();

function stats(samples) {
  const s = Float64Array.from(samples).sort();
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { median: q(0.5), p95: q(0.95), min: s[0], max: s[s.length - 1], n: s.length };
}

const fmt = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));

/** A height field of 2 n^2 triangles: (n+1)^2 vertices on a unit grid, world coordinates. */
function terrain(n, { phase = 0, amplitude = 4, dx = 0, dy = 0 } = {}) {
  const w = n + 1;
  const positions = new Float64Array(3 * w * w);
  for (let j = 0; j < w; j++) {
    for (let i = 0; i < w; i++) {
      const o = 3 * (j * w + i);
      positions[o] = i + dx;
      positions[o + 1] = j + dy;
      positions[o + 2] =
        (amplitude / 2) * (2 + Math.sin(i * 0.05 + phase) * Math.cos(j * 0.04 - phase) + 0.3 * Math.sin(i * 0.31 + j * 0.27));
    }
  }
  const indices = new Uint32Array(6 * n * n);
  let t = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * w + i;
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      indices[t++] = a;
      indices[t++] = b;
      indices[t++] = d;
      indices[t++] = a;
      indices[t++] = d;
      indices[t++] = c;
    }
  }
  return { positions, indices, triangles: 2 * n * n, vertices: w * w };
}

/** B flipped 180 degrees about x, then turned `deg` about z around (cx, cy), lifted by `lift`. */
function flippedPose(deg, cx, cy, lift) {
  const h = (deg * Math.PI) / 360;
  const sz = Math.sin(h);
  const cz = Math.cos(h);
  const c2 = cz * cz - sz * sz; // cos(deg)
  const s2 = 2 * sz * cz; // sin(deg)
  return {
    rotation: [cz, sz, 0, 0], // quaternion [x, y, z, w] of Rz * Rx(pi)
    translation: [cx - (c2 * cx - s2 * cy), cy - (s2 * cx + c2 * cy), lift],
  };
}

/** The JS-side upload simulation: copies every dirty range of `changes` into mirrors. */
class Mirror {
  constructor(scene) {
    this.scene = scene;
    this.meshes = new Map();
    this.bytes = 0;
  }

  apply(changes) {
    for (const c of changes) {
      if (c.removed) {
        this.meshes.delete(c.mesh);
        continue;
      }
      const v = this.scene.views(c.mesh);
      let m = this.meshes.get(c.mesh);
      const rebuild = m === undefined || c.capacityChanged;
      if (rebuild) {
        m = {
          positions: new Float32Array(this.scene.vertexSlots(c.mesh) * 3),
          normals: new Float32Array(this.scene.vertexSlots(c.mesh) * 3),
          colors: new Uint8Array(this.scene.vertexSlots(c.mesh) * 4),
          indices: new Uint32Array(this.scene.triangleSlots(c.mesh) * 3),
        };
        this.meshes.set(c.mesh, m);
      }
      this.copy(c.positions, v.positions(), m.positions, 3, rebuild);
      this.copy(c.normals, v.normals(), m.normals, 3, rebuild);
      this.copy(c.colors, v.colors(), m.colors, 4, rebuild);
      this.copy(c.indices, v.indices(), m.indices, 3, rebuild);
    }
  }

  copy(ranges, src, dst, stride, rebuild) {
    if (src === null) {
      return;
    }
    if (ranges === "all" || rebuild) {
      dst.set(src);
      this.bytes += src.byteLength;
      return;
    }
    for (const [start, count] of ranges) {
      dst.set(src.subarray(start * stride, (start + count) * stride), start * stride);
      this.bytes += count * stride * src.BYTES_PER_ELEMENT;
    }
  }
}

function report(name, samples, extra = "") {
  const s = stats(samples);
  console.log(`${name}: median ${fmt(s.median)} ms, p95 ${fmt(s.p95)} ms, min ${fmt(s.min)}, max ${fmt(s.max)} (n=${s.n})${extra}`);
  return s;
}

// ---- 1. addMesh from Float64Array ----

function benchAdd(label, n, runs) {
  const g = terrain(n);
  const scene = new Scene();
  const times = [];
  for (let r = 0; r < runs; r++) {
    const t0 = now();
    const m = scene.addMesh(g.positions, g.indices);
    times.push(now() - t0);
    scene.removeMesh(m);
    scene.takeChanges();
  }
  report(`addMesh ${label} (${g.triangles.toLocaleString("en")} triangles, eager)`, times);
  scene.free();
}

// ---- 2. Large-scale drag: A fixed, B flipped over it, 64 poses, colouring on ----

function benchDrag(label, n, cycles, { maxDistance, range }) {
  const a = terrain(n);
  const b = terrain(n, { phase: 1.3 });
  const scene = new Scene();
  let t0 = now();
  const meshA = scene.addMesh(a.positions, a.indices);
  const meshB = scene.addMesh(b.positions, b.indices);
  console.log(`${label}: added 2 meshes of ${a.triangles.toLocaleString("en")} triangles in ${fmt(now() - t0)} ms`);
  const cx = n / 2;
  const cy = n / 2;
  // Lift B until the surfaces just touch (as the Rust benches do): measure the gap at pose 0.
  let lift = 8;
  scene.setPlacement(meshB, flippedPose(0, cx, cy, lift));
  lift -= scene.meshDistance(meshA, meshB)?.distance ?? 0;
  const pose = (i) => flippedPose(0.1 * i, cx, cy, lift);
  scene.setPlacement(meshB, pose(0));
  scene.takeChanges();
  t0 = now();
  const field = scene.addDistanceField({ name: "gap", on: meshA, to: meshB, maxDistance });
  scene.setColoring(meshA, { field, map: "turbo", range, below: "clamp", above: "base" });
  console.log(`${label}: addDistanceField + setColoring ${fmt(now() - t0)} ms`);
  const mirror = new Mirror(scene);
  t0 = now();
  mirror.apply(scene.commitFrame());
  console.log(`${label}: first commitFrame + full mirror copy ${fmt(now() - t0)} ms`);

  const drag = scene.drag(meshB);
  const total = [];
  const parts = { move: [], commit: [], copy: [] };
  let dirtyBytes = [];
  let dirtyTriangles = [];
  for (let c = 0; c < cycles + 1; c++) {
    for (let i = 0; i < 64; i++) {
      const p = pose(i);
      const f0 = now();
      drag.move(p);
      const f1 = now();
      const changes = scene.commitFrame();
      const f2 = now();
      const before = mirror.bytes;
      mirror.apply(changes);
      const f3 = now();
      if (c > 0) {
        total.push(f3 - f0);
        parts.move.push(f1 - f0);
        parts.commit.push(f2 - f1);
        parts.copy.push(f3 - f2);
        dirtyBytes.push(mirror.bytes - before);
        const colors = changes.find((x) => x.mesh === meshA)?.colors;
        dirtyTriangles.push(colors === "all" ? -1 : (colors ?? []).reduce((s, r) => s + r[1], 0));
      }
    }
  }
  drag.end();
  const s = report(`${label} frame (move + commitFrame + copy)`, total);
  console.log(
    `  parts (median ms): move ${fmt(stats(parts.move).median)}, commitFrame ${fmt(stats(parts.commit).median)}, copy ${fmt(stats(parts.copy).median)}`,
  );
  console.log(
    `  dirty per frame: median ${fmt(stats(dirtyBytes).median / 1e6)} MB copied, median ${stats(dirtyTriangles).median.toLocaleString("en")} colour slots`,
  );
  scene.free();
  return s;
}

// ---- 3. Owner-realistic scene: 6 meshes of ~30k triangles, 30 distance fields, drag one ----

function benchOwner(label, frames, { gap, maxDistance }) {
  const n = 123; // 2 * 123^2 = 30,258 triangles per mesh
  const scene = new Scene();
  const meshes = [];
  for (let k = 0; k < 6; k++) {
    const g = terrain(n, { phase: k, dx: (k % 3) * (n + gap), dy: Math.floor(k / 3) * (n + gap), amplitude: 6 });
    meshes.push(scene.addMesh(g.positions, g.indices));
  }
  const fields = new Map();
  const t0 = now();
  for (let k = 0; k < 6; k++) {
    for (let j = 0; j < 6; j++) {
      if (k !== j) {
        fields.set(`${k}:${j}`, scene.addDistanceField({ name: `to${j}`, on: meshes[k], to: meshes[j], maxDistance }));
      }
    }
  }
  // Each mesh shows the distance to its next neighbour; all 30 fields stay live.
  meshes.forEach((m, k) => {
    scene.setColoring(m, { field: fields.get(`${k}:${(k + 1) % 6}`), map: "viridis", range: [0, 8] });
  });
  const mirror = new Mirror(scene);
  mirror.apply(scene.commitFrame());
  console.log(`${label}: 6 x ${(2 * n * n).toLocaleString("en")} triangles, ${fields.size} fields, set up in ${fmt(now() - t0)} ms`);

  const drag = scene.drag(meshes[0]);
  const total = [];
  const parts = { move: [], commit: [], copy: [] };
  for (let i = 0; i < frames + 20; i++) {
    const a = (i * 2 * Math.PI) / 64;
    const h = (0.05 * Math.sin(a)) / 2;
    const p = { rotation: [0, 0, Math.sin(h), Math.cos(h)], translation: [1.5 * Math.sin(a), 1.5 * Math.cos(a) - 1.5, 0] };
    const f0 = now();
    drag.move(p);
    const f1 = now();
    const changes = scene.commitFrame();
    const f2 = now();
    mirror.apply(changes);
    const f3 = now();
    if (i >= 20) {
      total.push(f3 - f0);
      parts.move.push(f1 - f0);
      parts.commit.push(f2 - f1);
      parts.copy.push(f3 - f2);
    }
  }
  drag.end();
  report(`${label} frame (move + commitFrame + copy)`, total);
  console.log(
    `  parts (median ms): move ${fmt(stats(parts.move).median)}, commitFrame ${fmt(stats(parts.commit).median)}, copy ${fmt(stats(parts.copy).median)}`,
  );
  scene.free();
}

// ---- 4. three.js adapter: the 64-pose drag of two touching 1M meshes, bindMesh on both ----

function benchAdapter(label, n, cycles, { maxDistance, range }) {
  const a = terrain(n);
  const b = terrain(n, { phase: 1.3 });
  const scene = new Scene();
  const meshA = scene.addMesh(a.positions, a.indices);
  const meshB = scene.addMesh(b.positions, b.indices);
  const cx = n / 2;
  const cy = n / 2;
  let lift = 8;
  scene.setPlacement(meshB, flippedPose(0, cx, cy, lift));
  lift -= scene.meshDistance(meshA, meshB)?.distance ?? 0;
  const pose = (i) => flippedPose(0.1 * i, cx, cy, lift);
  scene.setPlacement(meshB, pose(0));
  scene.takeChanges();
  const field = scene.addDistanceField({ name: "gap", on: meshA, to: meshB, maxDistance });
  scene.setColoring(meshA, { field, map: "turbo", range, below: "clamp", above: "base" });
  // Both meshes are bound before the first commitFrame, which builds the geometries (no GPU upload in Node).
  let t0 = now();
  const boundA = bindMesh(scene, meshA);
  const boundB = bindMesh(scene, meshB);
  scene.commitFrame();
  console.log(`${label}: bindMesh x2 + first commitFrame (builds both geometries) ${fmt(now() - t0)} ms`);

  const drag = scene.drag(meshB);
  const total = [];
  const parts = { move: [], commit: [] };
  for (let c = 0; c < cycles + 1; c++) {
    for (let i = 0; i < 64; i++) {
      const p = pose(i);
      const f0 = now();
      drag.move(p);
      const f1 = now();
      scene.commitFrame(); // the adapters' listeners apply the ranges inside this call
      const f2 = now();
      if (c > 0) {
        total.push(f2 - f0);
        parts.move.push(f1 - f0);
        parts.commit.push(f2 - f1);
      }
    }
  }
  drag.end();
  scene.commitFrame();
  report(`${label} frame (move + commitFrame with bindMesh on both)`, total);
  console.log(
    `  parts (median ms): move ${fmt(stats(parts.move).median)}, commitFrame incl. adapter ${fmt(stats(parts.commit).median)} (p95 ${fmt(stats(parts.commit).p95)})`,
  );
  boundA.dispose();
  boundB.dispose();
  scene.free();
}

// ---- 5. Worker scene (fast-mesh-wasm/worker) in the worker_threads harness ----
//
// `WorkerScene` from dist/worker.js drives the engine in a Node Worker
// (bench/worker-harness.mjs: the engine served as the browser Worker script
// serves it, default 4 ms yield budget, plus timing probes). Main-thread
// numbers: the synchronous calls, the time the scene's message handler takes
// to apply a result to the mirrors ("handler"), the adapter-style copy of the
// delivered dirty ranges, and the longest event-loop block seen by a
// setImmediate probe (which also covers the structured-clone of the message).
// Worker numbers come from the harness: each chunk's time, and the longest
// block of the Worker's event loop.

/** The bench harness Worker as a `WorkerLike`, timing the scene's message handler. */
class BenchWorker {
  constructor() {
    this.worker = new Worker(new URL("./worker-harness.mjs", import.meta.url));
    this.message = [];
    this.error = [];
    this.posted = 0; // request batches posted
    this.sentAt = []; // the time each request batch was posted, in order
    this.results = []; // { at, handler, colored } per result, in order
    this.waiters = []; // { resolve, reject } of drained(), woken by each result
    this.statsWaiters = []; // the same for stats(), woken by its reply
    this.failure = undefined; // set once the Worker failed: every wait rejects with it
    this.worker.on("message", (data) => {
      if (data?.kind === "benchStats") {
        this.#wake(this.statsWaiters, data);
        return;
      }
      if (data?.kind !== undefined && data.kind !== "ready") {
        this.#fail(new Error(`the bench Worker sent ${data.kind}: ${JSON.stringify(data.error ?? null)}`));
      } else if (data?.poisoned) {
        this.#fail(new Error(`the engine trapped: ${JSON.stringify(data.status?.[0]?.error ?? null)}`));
      }
      const t0 = now();
      for (const cb of this.message) {
        cb({ data });
      }
      const t1 = now();
      if (data?.kind === undefined) {
        const colored = (data.changes ?? []).some((c) => c.colors === "all" || c.colors.length > 0);
        this.results.push({ at: t1, handler: t1 - t0, colored });
        this.#wake(this.waiters);
      }
    });
    this.worker.on("error", (e) => {
      this.#fail(e);
      for (const cb of this.error) {
        cb(e);
      }
    });
    this.worker.on("exit", (code) => this.#fail(new Error(`the bench Worker exited (code ${code})`)));
  }

  #wake(waiters, data) {
    for (const w of waiters.splice(0)) {
      w.resolve(data);
    }
  }

  #fail(error) {
    this.failure ??= error;
    for (const w of [...this.waiters.splice(0), ...this.statsWaiters.splice(0)]) {
      w.reject(this.failure);
    }
  }

  #wait(waiters) {
    if (this.failure !== undefined) {
      return Promise.reject(this.failure);
    }
    return new Promise((resolve, reject) => waiters.push({ resolve, reject }));
  }

  postMessage(msg, transfer) {
    if (Array.isArray(msg)) {
      this.posted++;
      this.sentAt.push(now());
    }
    this.worker.postMessage(msg, transfer);
  }

  addEventListener(type, cb) {
    (type === "message" ? this.message : this.error).push(cb);
  }

  terminate() {
    void this.worker.terminate();
  }

  /** Resolves once every posted batch has its result applied; returns the results since `from`. */
  async drained(from = 0) {
    while (this.results.length < this.posted) {
      await this.#wait(this.waiters);
    }
    return this.results.slice(from);
  }

  /** The harness's chunk times and longest Worker block since the last call (call only when drained). */
  stats() {
    const reply = this.#wait(this.statsWaiters);
    if (this.failure === undefined) {
      this.worker.postMessage({ benchStats: true });
    }
    return reply;
  }
}

/** The longest main-thread event-loop block, from a setImmediate loop. */
class Probe {
  start() {
    this.max = 0;
    this.on = true;
    this.last = now();
    const tick = () => {
      if (!this.on) {
        return;
      }
      const t = now();
      this.max = Math.max(this.max, t - this.last);
      this.last = t;
      setImmediate(tick);
    };
    setImmediate(tick);
  }

  async stop() {
    // One more turn, so a block at the very end is seen.
    await new Promise((resolve) => setImmediate(resolve));
    this.on = false;
    return this.max;
  }
}

/** `Mirror` for a `SceneLike` without slot counts: sizes come from the views (full slot length). */
class ViewMirror extends Mirror {
  apply(changes) {
    for (const c of changes) {
      if (c.removed) {
        this.meshes.delete(c.mesh);
        continue;
      }
      const v = this.scene.views(c.mesh);
      let m = this.meshes.get(c.mesh);
      const rebuild = m === undefined || c.capacityChanged;
      if (rebuild) {
        const slots = v.positions().length / 3;
        m = {
          positions: new Float32Array(slots * 3),
          normals: new Float32Array(slots * 3),
          colors: new Uint8Array(slots * 4),
          indices: new Uint32Array(v.indices().length),
        };
        this.meshes.set(c.mesh, m);
      }
      this.copy(c.positions, v.positions(), m.positions, 3, rebuild);
      this.copy(c.normals, v.normals(), m.normals, 3, rebuild);
      if (v.colors() !== null && m.colors.length !== v.colors().length) {
        m.colors = new Uint8Array(v.colors().length);
      }
      this.copy(c.colors, v.colors(), m.colors, 4, rebuild);
      this.copy(c.indices, v.indices(), m.indices, 3, rebuild);
    }
  }
}

let createWorkerSceneFn;
async function openWorkerScene() {
  createWorkerSceneFn ??= (await import("../dist/worker.js")).createWorkerScene;
  let bw;
  const scene = await createWorkerSceneFn({
    createWorker: () => {
      bw = new BenchWorker();
      return bw;
    },
  });
  return { scene, bw };
}

/** Sends this frame's requests and waits for their results; returns the round trip and handler times. */
async function roundTrip(scene, bw) {
  const from = bw.results.length;
  const t0 = now();
  scene.commitFrame();
  const results = await bw.drained(from);
  return { rt: now() - t0, handler: results.reduce((s, r) => s + r.handler, 0) };
}

const maxOf = (xs) => xs.reduce((m, x) => Math.max(m, x), 0);
const chunkMs = (st, kind) => st.chunks.filter((c) => c.kind === kind).map((c) => c.ms);

// 5a. addMesh: main-thread blocks while a big mesh goes to the Worker and comes back.
async function benchWorkerAdd(label, n, runs) {
  const g = terrain(n);
  const { scene, bw } = await openWorkerScene();
  const probe = new Probe();
  const parts = { call: [], send: [], handler: [], copy: [], block: [], rt: [], worker: [], workerBlock: [] };
  for (let r = 0; r < runs; r++) {
    const mirror = new ViewMirror(scene);
    probe.start();
    const t0 = now();
    const m = scene.addMesh(g.positions, g.indices);
    const t1 = now();
    const from = bw.results.length;
    scene.commitFrame();
    const t2 = now();
    const results = await bw.drained(from);
    const t3 = now();
    mirror.apply(scene.commitFrame());
    const t4 = now();
    parts.block.push(await probe.stop());
    parts.call.push(t1 - t0);
    parts.send.push(t2 - t1);
    parts.rt.push(t3 - t1);
    parts.handler.push(maxOf(results.map((x) => x.handler)));
    parts.copy.push(t4 - t3);
    const st = await bw.stats();
    parts.worker.push(maxOf(chunkMs(st, "addMesh")));
    parts.workerBlock.push(st.maxGap);
    scene.removeMesh(m);
    await roundTrip(scene, bw);
    scene.commitFrame();
    await bw.stats();
  }
  const med = (k) => fmt(stats(parts[k]).median);
  console.log(`worker addMesh ${label} (${g.triangles.toLocaleString("en")} triangles, ${runs} runs, medians):`);
  console.log(
    `  main: addMesh() call ${med("call")} ms (max ${fmt(maxOf(parts.call))}), commitFrame() send ${med("send")} ms, result handler ${med("handler")} ms (max ${fmt(maxOf(parts.handler))}), full mirror copy on the next commitFrame ${med("copy")} ms`,
  );
  console.log(
    `  main: longest event-loop block ${med("block")} ms (max ${fmt(maxOf(parts.block))}); round trip to applied ${med("rt")} ms`,
  );
  console.log(`  worker: addMesh request ${med("worker")} ms, longest Worker block ${med("workerBlock")} ms`);
  scene.terminate();
}

/** Two touching meshes (B flipped over A), a distance field on A to B colouring A. */
async function workerPair(n, { maxDistance, range }) {
  const a = terrain(n);
  const b = terrain(n, { phase: 1.3 });
  const { scene, bw } = await openWorkerScene();
  const meshA = scene.addMesh(a.positions, a.indices);
  const meshB = scene.addMesh(b.positions, b.indices);
  const cx = n / 2;
  const cy = n / 2;
  let lift = 8;
  scene.setPlacement(meshB, flippedPose(0, cx, cy, lift));
  const gap = scene.meshDistance(meshA, meshB);
  await roundTrip(scene, bw);
  lift -= (await gap)?.distance ?? 0;
  const pose = (i) => flippedPose(0.1 * i, cx, cy, lift);
  scene.setPlacement(meshB, pose(0));
  const field = scene.addDistanceField({ name: "gap", on: meshA, to: meshB, maxDistance });
  const rule = { field, map: "turbo", range, below: "clamp", above: "base" };
  scene.setColoring(meshA, rule);
  await roundTrip(scene, bw);
  return { scene, bw, meshA, meshB, pose, field, rule };
}

// 5b. Colouring toggle: round trip from commitFrame() to the result applied.
async function benchWorkerToggle(label, n, toggles, opts) {
  const { scene, bw, meshA, rule } = await workerPair(n, opts);
  const mirror = new ViewMirror(scene);
  mirror.apply(scene.commitFrame());
  await bw.stats();
  const off = { rt: [], handler: [], copy: [], worker: [], block: [] };
  const on = { rt: [], handler: [], copy: [], worker: [], block: [] };
  for (let k = 0; k < toggles; k++) {
    const s = k % 2 === 0 ? off : on;
    scene.setColoring(meshA, k % 2 === 0 ? null : rule);
    const { rt, handler } = await roundTrip(scene, bw);
    const t0 = now();
    mirror.apply(scene.commitFrame());
    s.copy.push(now() - t0);
    s.rt.push(rt);
    s.handler.push(handler);
    const st = await bw.stats();
    s.worker.push(maxOf(chunkMs(st, "coloring")));
    s.block.push(st.maxGap);
  }
  for (const [name, s] of [
    ["off", off],
    ["on", on],
  ]) {
    report(
      `worker ${label} colouring ${name}, round trip`,
      s.rt,
      `; handler median ${fmt(stats(s.handler).median)} ms, copy ${fmt(stats(s.copy).median)} ms, worker request ${fmt(stats(s.worker).median)} ms, longest Worker block ${fmt(stats(s.block).median)} ms`,
    );
  }
  scene.terminate();
}

// 5c. The 64-pose drag, one move per frame, in lock step with the results.
async function benchWorkerDrag(label, n, cycles, opts) {
  const { scene, bw, meshB, pose } = await workerPair(n, opts);
  const mirror = new ViewMirror(scene);
  mirror.apply(scene.commitFrame());
  const drag = scene.drag(meshB);
  await roundTrip(scene, bw);
  await bw.stats();
  const main = [];
  const parts = { move: [], commit: [], copy: [], handler: [], rt: [], bytes: [], worker: [], workerBlock: [] };
  for (let c = 0; c < cycles + 1; c++) {
    for (let i = 0; i < 64; i++) {
      const from = bw.results.length;
      const before = mirror.bytes;
      const f0 = now();
      drag.move(pose(i));
      const f1 = now();
      const changes = scene.commitFrame(); // sends this move, delivers the previous result
      const f2 = now();
      mirror.apply(changes);
      const f3 = now();
      const results = await bw.drained(from);
      const f4 = now();
      const handler = results.reduce((s, r) => s + r.handler, 0);
      const st = await bw.stats();
      if (c > 0) {
        main.push(f3 - f0 + handler);
        parts.move.push(f1 - f0);
        parts.commit.push(f2 - f1);
        parts.copy.push(f3 - f2);
        parts.handler.push(handler);
        parts.rt.push(f4 - f1);
        parts.bytes.push(mirror.bytes - before);
        parts.worker.push(maxOf(chunkMs(st, "drag")));
        parts.workerBlock.push(st.maxGap);
      }
    }
  }
  drag.end();
  await roundTrip(scene, bw);
  const med = (k) => fmt(stats(parts[k]).median);
  report(`worker ${label} main-thread cost per frame (move + commitFrame + copy + result handler)`, main);
  console.log(
    `  parts (median ms): move ${med("move")}, commitFrame ${med("commit")}, copy ${med("copy")}, handler ${med("handler")} (p95 ${fmt(stats(parts.handler).p95)}); dirty bytes copied ${fmt(stats(parts.bytes).median / 1e3)} KB`,
  );
  console.log(
    `  round trip move -> result applied: median ${med("rt")} ms (p95 ${fmt(stats(parts.rt).p95)}); worker drag request median ${med("worker")} ms, longest Worker block max ${fmt(maxOf(parts.workerBlock))} ms`,
  );
  scene.terminate();
}

// 5f. Wide-range drag (maxDistance 8): the field recompute is stepped and a stale pose is
// dropped at a step boundary when a newer move is queued (drag-speed part A). Two terrains:
// "9a", the pair of 5c, and "native", the terrain and touching poses of the Rust
// `scene wide drag` bench (`crates/fast-mesh-scene/benches/scene.rs`), built here in f64.

/** The Rust bench's terrain: z = sin(0.05 x) cos(0.07 y) 4 on the unit grid. */
function nativeTerrain(n) {
  const w = n + 1;
  const positions = new Float64Array(3 * w * w);
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      const o = 3 * (y * w + x);
      positions[o] = x;
      positions[o + 1] = y;
      positions[o + 2] = Math.sin(x * 0.05) * Math.cos(y * 0.07) * 4;
    }
  }
  const indices = new Uint32Array(6 * n * n);
  let t = 0;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const a = y * w + x;
      indices.set([a, a + 1, a + w + 1, a, a + w + 1, a + w], t);
      t += 6;
    }
  }
  return { positions, indices, triangles: 2 * n * n, vertices: w * w };
}

/** The Rust bench's `contact_lift`: B turned by `deg` just touches A (sampled on a grid). */
function nativeContactLift(side, deg) {
  const r = (deg * Math.PI) / 180;
  const s = Math.sin(r);
  const c = Math.cos(r);
  const mid = side / 2;
  const step = Math.max(1, Math.ceil(side / 500));
  const height = (x, y) => Math.sin(x * 0.05) * Math.cos(y * 0.07) * 4;
  let lift = -Infinity;
  for (let y = 0; y <= side; y += step) {
    for (let x = 0; x <= side; x += step) {
      const dx = x - mid;
      const dy = y - mid;
      const u = c * dx + s * dy + mid;
      const v = side - (-s * dx + c * dy + mid);
      if (u >= 0 && u <= side && v >= 0 && v <= side) {
        lift = Math.max(lift, height(x, y) + height(u, v));
      }
    }
  }
  return lift;
}

/** The Rust bench's `pose`: B flipped about x, moved by (0, side, lift), turned by `deg` about the middle. */
function nativePose(side, deg) {
  const lift = nativeContactLift(side, deg);
  const r = (deg * Math.PI) / 180;
  const s = Math.sin(r);
  const c = Math.cos(r);
  const mid = side / 2;
  const tx = -mid;
  const ty = side - mid;
  return {
    rotation: [Math.cos(r / 2), Math.sin(r / 2), 0, 0], // Rz(deg) * Rx(pi)
    translation: [c * tx - s * ty + mid, s * tx + c * ty + mid, lift],
  };
}

/** The wide pair on either terrain, coloured and settled at pose 0. */
async function wideWorkerPair(kind, n) {
  if (kind === "9a") {
    return workerPair(n, wide);
  }
  const g = nativeTerrain(n);
  const { scene, bw } = await openWorkerScene();
  const meshA = scene.addMesh(g.positions, g.indices);
  const meshB = scene.addMesh(g.positions, g.indices);
  const cache = new Map();
  const pose = (i) => {
    if (!cache.has(i)) {
      cache.set(i, nativePose(n, 0.1 * i));
    }
    return cache.get(i);
  };
  scene.setPlacement(meshB, pose(0));
  const field = scene.addDistanceField({ name: "gap", on: meshA, to: meshB, maxDistance: wide.maxDistance });
  scene.setColoring(meshA, { field, map: "turbo", range: wide.range, below: "clamp", above: "base" });
  await roundTrip(scene, bw);
  return { scene, bw, meshA, meshB, pose };
}

const sleepUntil = (t) => new Promise((resolve) => setTimeout(resolve, Math.max(0, t - now())));

async function benchWorkerWideDrag(label, kind, n, { lockStep, seconds, hz = 60 }) {
  let t0 = now();
  const { scene, bw, meshB, pose } = await wideWorkerPair(kind, n);
  console.log(`worker wide drag ${label}: set up in ${fmt((now() - t0) / 1e3)} s`);
  const mirror = new ViewMirror(scene);
  mirror.apply(scene.commitFrame());
  const drag = scene.drag(meshB);
  await roundTrip(scene, bw);
  await bw.stats();

  // Lock step: one move, then wait for its colours, at poses 16, 32, 48, 0 in turn (as the
  // Rust bench's WIDE_POSES), so every pose recolours most of A.
  const ls = { rt: [], block: [], open: [], steps: [], colored: 0 };
  for (let k = 1; k <= lockStep; k++) {
    const from = bw.results.length;
    const f0 = now();
    drag.move(pose((16 * k) % 64));
    mirror.apply(scene.commitFrame());
    const results = await bw.drained(from);
    ls.rt.push(now() - f0);
    ls.colored += results.filter((r) => r.colored).length;
    const st = await bw.stats();
    const steps = chunkMs(st, "fieldStep");
    ls.block.push(st.maxGap);
    ls.open.push(steps[0] ?? 0);
    ls.steps.push(steps.length);
  }
  const med = (xs) => fmt(stats(xs).median);
  report(
    `worker wide drag ${label}, lock step (one move, wait for its colours): move -> colours applied`,
    ls.rt,
    `; ${ls.colored} of ${lockStep} poses recoloured`,
  );
  console.log(
    `  longest Worker block per pose median ${med(ls.block)} ms (max ${fmt(maxOf(ls.block))}); first field step (unit open) median ${med(ls.open)} ms; ${med(ls.steps)} field steps per pose`,
  );

  // Continuous drag: a move every 1000/hz ms for `seconds`, whatever the Worker is doing,
  // turning B through the 64 poses and back. Results arrive one per batch, in order; the
  // results that recolour A are the poses that completed, the rest were dropped as stale.
  const period = 1000 / hz;
  const frames = Math.round((seconds * 1000) / period);
  const first = bw.results.length;
  const firstSent = bw.sentAt.length;
  const probe = new Probe();
  probe.start();
  const main = [];
  t0 = now();
  for (let k = 0; k < frames; k++) {
    await sleepUntil(t0 + k * period);
    const f0 = now();
    const i = k % 128;
    drag.move(pose(i < 64 ? i : 127 - i));
    mirror.apply(scene.commitFrame());
    main.push(now() - f0);
  }
  const lastSent = bw.sentAt[bw.sentAt.length - 1];
  const results = await bw.drained(first);
  const settled = now();
  mirror.apply(scene.commitFrame());
  const mainBlock = await probe.stop();
  const st = await bw.stats();
  const sent = bw.sentAt.slice(firstSent);
  const lag = []; // move sent -> its colours applied, for every pose that completed
  let backlog = 0; // the most batches posted but not yet answered, seen at a result
  results.forEach((r, k) => {
    backlog = Math.max(backlog, sent.filter((s) => s < r.at).length - k - 1);
    if (r.colored) {
      lag.push(r.at - sent[k]);
    }
  });
  const steps = chunkMs(st, "fieldStep");
  console.log(
    `worker wide drag ${label}, continuous ${hz} Hz drag for ${seconds} s: ${frames} moves, ${lag.length} poses recoloured, ${chunkMs(st, "fieldStepAbandon").length} units abandoned`,
  );
  report(`  move -> colours applied, per recoloured pose`, lag);
  console.log(
    `  last move -> its colours applied ${fmt(results[results.length - 1].at - lastSent)} ms (last result recolours: ${results[results.length - 1].colored}); all drained ${fmt(settled - lastSent)} ms after the last move`,
  );
  console.log(
    `  most batches waiting for a result ${backlog}; longest field step ${fmt(maxOf(steps))} ms, longest Worker block ${fmt(st.maxGap)} ms; main thread per frame median ${med(main)} ms, longest main block ${fmt(mainBlock)} ms`,
  );
  drag.end();
  await roundTrip(scene, bw);
  scene.terminate();
}

// 5d. The longest single requests: one edit with several dependent fields, a section over several meshes.
async function benchWorkerLongRequests(label, n, edits) {
  const { scene, bw, meshA, meshB } = await workerPair(n, { maxDistance: 0.08, range: [-0.08, 0] });
  // A third tile C beside the pair, with no field: the edit baseline, and cut by the section.
  const c = terrain(n, { phase: 2.1, dx: n + 4 });
  const meshC = scene.addMesh(c.positions, c.indices);
  await roundTrip(scene, bw);
  await bw.stats();
  const w = n + 1;
  const centre = Math.floor(w / 2) * w + Math.floor(w / 2); // a vertex in the contact area
  const g = terrain(n);
  async function editRound(name, mesh, src) {
    const block = [];
    const total = [];
    const rt = [];
    for (let k = 0; k < edits; k++) {
      const z = src.positions[3 * centre + 2] + (k % 2 === 0 ? 0.01 : 0);
      const done = scene.edit(mesh, [
        {
          op: "setPositions",
          vertices: [centre],
          positions: [src.positions[3 * centre], src.positions[3 * centre + 1], z],
        },
      ]);
      const r = await roundTrip(scene, bw);
      await done;
      const st = await bw.stats();
      rt.push(r.rt);
      total.push(st.chunks.reduce((s, x) => s + x.ms, 0));
      block.push(st.maxGap);
    }
    report(
      `worker ${label} one-vertex edit, ${name}: longest Worker block`,
      block,
      `; worker time for the batch median ${fmt(stats(total).median)} ms, round trip median ${fmt(stats(rt).median)} ms`,
    );
  }
  await editRound("of C, no field on it", meshC, c);
  await editRound("of A, 1 field (on A to B)", meshA, g);
  scene.addDistanceField({ name: "gapBack", on: meshB, to: meshA, maxDistance: 0.08 });
  await roundTrip(scene, bw);
  console.log(`  adding the B-to-A contact field: longest Worker block ${fmt((await bw.stats()).maxGap)} ms`);
  await editRound("of A, 2 contact fields (A to B, B to A)", meshA, g);
  scene.addDistanceField({ name: "wide", on: meshA, to: meshB, maxDistance: 8 });
  scene.addDistanceField({ name: "wideBack", on: meshB, to: meshA, maxDistance: 8 });
  await roundTrip(scene, bw);
  const st = await bw.stats();
  console.log(
    `  adding the 2 wide fields: field builds ${chunkMs(st, "fieldBuild").map(fmt).join(", ")} ms, longest Worker block ${fmt(st.maxGap)} ms`,
  );
  await editRound("of A, 4 fields (2 contact, 2 wide with maxDistance 8)", meshA, g);

  // A section over the three meshes.
  for (const fill of [false, true]) {
    const plane = (k) => ({ origin: [0, n * (0.3 + 0.01 * k), 0], normal: [0, 1, 0] });
    const section = scene.addSection({ plane: plane(0), meshes: [meshA, meshB, meshC], fill });
    await roundTrip(scene, bw);
    const add = maxOf(chunkMs(await bw.stats(), "section"));
    const worker = [];
    for (let k = 1; k <= 16; k++) {
      scene.setSectionPlane(section, plane(k));
      await roundTrip(scene, bw);
      worker.push(maxOf(chunkMs(await bw.stats(), "section")));
    }
    report(
      `worker ${label} section over 3 meshes (fill ${fill ? "on" : "off"}), plane change: worker section request`,
      worker,
      `; addSection ${fmt(add)} ms`,
    );
    scene.removeSection(section);
    await roundTrip(scene, bw);
    await bw.stats();
  }
  scene.terminate();
}

// 5e. Many small requests in one frame: 1000 setPlacement calls over 10 small meshes.
async function benchWorkerManySmall(label, calls, rounds) {
  const { scene, bw } = await openWorkerScene();
  const meshes = [];
  for (let k = 0; k < 10; k++) {
    const g = terrain(50, { phase: k, dx: k * 60 });
    meshes.push(scene.addMesh(g.positions, g.indices));
  }
  await roundTrip(scene, bw);
  await bw.stats();
  const parts = { queue: [], rt: [], handler: [], worker: [], workerBlock: [], chunks: [] };
  for (let r = 0; r < rounds + 1; r++) {
    const t0 = now();
    for (let k = 0; k < calls; k++) {
      const x = 0.001 * (k + r);
      scene.setPlacement(meshes[k % 10], { rotation: [0, 0, 0, 1], translation: [x, 0, 0] });
    }
    const t1 = now();
    const { rt, handler } = await roundTrip(scene, bw);
    const st = await bw.stats();
    if (r > 0) {
      parts.queue.push(t1 - t0);
      parts.rt.push(rt);
      parts.handler.push(handler);
      const run = chunkMs(st, "setPlacement");
      parts.worker.push(run.reduce((s, x) => s + x, 0));
      parts.chunks.push(run.length);
      parts.workerBlock.push(st.maxGap);
    }
  }
  const med = (k) => fmt(stats(parts[k]).median);
  console.log(`worker ${label}: ${calls} setPlacement calls in one frame over 10 meshes of 5,000 triangles (${rounds} rounds, medians)`);
  console.log(
    `  main: queueing the calls ${med("queue")} ms, round trip commitFrame -> result applied ${med("rt")} ms (p95 ${fmt(stats(parts.rt).p95)}), result handler ${med("handler")} ms`,
  );
  console.log(
    `  worker: ${stats(parts.chunks).median} requests run, ${med("worker")} ms summed over them, longest Worker block ${med("workerBlock")} ms (max ${fmt(maxOf(parts.workerBlock))})`,
  );
  // The same calls on a synchronous Scene, for comparison.
  const sync = new Scene();
  const syncMeshes = [];
  for (let k = 0; k < 10; k++) {
    const g = terrain(50, { phase: k, dx: k * 60 });
    syncMeshes.push(sync.addMesh(g.positions, g.indices));
  }
  const times = [];
  for (let r = 0; r < rounds + 1; r++) {
    const t0 = now();
    for (let k = 0; k < calls; k++) {
      sync.setPlacement(syncMeshes[k % 10], { rotation: [0, 0, 0, 1], translation: [0.001 * (k + r), 0, 0] });
    }
    sync.commitFrame();
    if (r > 0) {
      times.push(now() - t0);
    }
  }
  report(`  sync Scene, the same ${calls} calls + commitFrame`, times);
  sync.free();
  scene.terminate();
}

// ---- run ----

console.log(`node ${process.version}, ${process.platform} ${process.arch}${quick ? " (quick)" : ""}`);
const contact = { maxDistance: 0.08, range: [-0.08, 0] }; // as the Rust drag bench
const wide = { maxDistance: 8, range: [0, 6] }; // every vertex of the dragged pair is in range
if (onlyWorker && onlyWide) {
  await benchWorkerWideDrag("1M, 9a terrain", "9a", 708, { lockStep: quick ? 4 : 8, seconds: quick ? 3 : 10 });
  await benchWorkerWideDrag("1M, native terrain", "native", 708, { lockStep: quick ? 4 : 8, seconds: quick ? 3 : 10 });
  if (with10m) {
    await benchWorkerWideDrag("10M, 9a terrain", "9a", 2236, { lockStep: 4, seconds: 30 });
    await benchWorkerWideDrag("10M, native terrain", "native", 2236, { lockStep: 4, seconds: 30 });
  }
  process.exit(0);
}
if (onlyWorker) {
  await benchWorkerAdd("1M", 708, quick ? 3 : 7);
  await benchWorkerToggle("1M contact", 708, quick ? 10 : 20, contact);
  await benchWorkerToggle("1M wide", 708, quick ? 10 : 20, wide);
  await benchWorkerDrag("1M contact", 708, quick ? 1 : 3, contact);
  if (!quick) {
    await benchWorkerDrag("1M wide", 708, 1, wide); // 64 poses, plus a warm-up cycle
  }
  await benchWorkerWideDrag("1M, 9a terrain", "9a", 708, { lockStep: quick ? 4 : 8, seconds: quick ? 3 : 10 });
  await benchWorkerWideDrag("1M, native terrain", "native", 708, { lockStep: quick ? 4 : 8, seconds: quick ? 3 : 10 });
  await benchWorkerLongRequests("1M", 708, quick ? 4 : 10);
  await benchWorkerManySmall("small requests", 1000, quick ? 5 : 20);
  if (with10m) {
    await benchWorkerAdd("10M", 2236, 1);
    await benchWorkerToggle("10M contact", 2236, 6, contact);
    await benchWorkerDrag("10M contact", 2236, 1, contact);
    await benchWorkerWideDrag("10M, 9a terrain", "9a", 2236, { lockStep: 4, seconds: 30 });
    await benchWorkerWideDrag("10M, native terrain", "native", 2236, { lockStep: 4, seconds: 30 });
  }
  process.exit(0);
}
if (onlyAdapter) {
  benchAdapter("adapter 1M contact", 708, quick ? 1 : 3, contact);
  benchAdapter("adapter 1M wide", 708, quick ? 1 : 3, wide);
  process.exit(0);
}
benchAdd("1M", 708, quick ? 3 : 7);
benchDrag("1M contact", 708, quick ? 1 : 3, contact);
benchDrag("1M wide", 708, quick ? 1 : 3, wide);
benchOwner("owner scene, contact", quick ? 100 : 400, { gap: 0.5, maxDistance: 1 });
benchOwner("owner scene, wide", quick ? 100 : 400, { gap: 2, maxDistance: 10 });
benchAdapter("adapter 1M contact", 708, quick ? 1 : 3, contact);
benchAdapter("adapter 1M wide", 708, quick ? 1 : 3, wide);
if (with10m) {
  benchAdd("10M", 2236, 3);
  benchDrag("10M contact", 2236, 1, contact);
}
