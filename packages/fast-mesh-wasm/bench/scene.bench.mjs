// Scene benchmarks, run against the built package (a release build, not the test build):
//
//   pnpm build && pnpm --filter fast-mesh-wasm bench [-- --10m] [-- --quick] [-- --adapter]
//
// Plain Node with performance.now(); every number is a median (and p95) over the iterations
// shown. Nothing here uploads to a GPU: a "frame" is what an adapter does on the JS side,
// which is setPlacement, commitFrame() and copying every dirty range from the views into a
// JS mirror (the three.js adapter's job, spec section 11 item 2).
import { readFileSync } from "node:fs";
import { Scene } from "../dist/node.js";
import { bindMesh } from "../dist/three.js";

const args = new Set(process.argv.slice(2));
const quick = args.has("--quick");
const with10m = args.has("--10m");
const onlyAdapter = args.has("--adapter"); // run just the three.js adapter case

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

// ---- run ----

console.log(`node ${process.version}, ${process.platform} ${process.arch}${quick ? " (quick)" : ""}`);
const contact = { maxDistance: 0.08, range: [-0.08, 0] }; // as the Rust drag bench
const wide = { maxDistance: 8, range: [0, 6] }; // every vertex of the dragged pair is in range
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
