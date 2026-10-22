// The browser check page (check.html), driven by `scripts/browser-check.mjs`:
// each check is a function of `window.fmCheck` that returns plain data.
//
// - `compare`: the same op sequence (two terrains, wide colouring, drags, an
//   edit, a section, undo) on a single-threaded and a threaded Worker scene,
//   both on the package's own Worker script, compared bit for bit: every
//   change record, every buffer of the mirrors (positions, normals, colours,
//   indices), placements, field values and section results.
// - `memory`: meshes added to a threaded scene until its shared memory is past
//   a size, then a field and a query on them.
// - `restartOpen`, `restartRun`, `close`: a threaded scene whose Worker crashes,
//   then `restart()`: the new Worker starts a new pool and colours as before.
// - `nestedFail`: a threaded scene whose nested Workers cannot load.
// - `bench`: the wide-range drag of the Node bench (`bench/scene.bench.mjs`
//   5f), lock step and a continuous drag, with the Worker's block times.
//
// The last four use the check Worker (check-worker.ts) for its probes.
import type { FieldHandle, MeshChange, MeshHandle, PlacementInput, SectionHandle } from "fast-mesh-wasm";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "fast-mesh-wasm/worker";
import checkWorkerUrl from "./check-worker.ts?worker&url";

const now = () => performance.now();
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// ---- Meshes and poses (as bench/scene.bench.mjs) ----

interface Grid {
  positions: Float64Array;
  indices: Uint32Array;
  triangles: number;
}

/** A height field of 2 n^2 triangles: (n+1)^2 vertices on a unit grid, world coordinates. */
function terrain(n: number, { phase = 0, amplitude = 4, dx = 0, dy = 0 } = {}): Grid {
  const w = n + 1;
  const positions = new Float64Array(3 * w * w);
  for (let j = 0; j < w; j++) {
    for (let i = 0; i < w; i++) {
      const o = 3 * (j * w + i);
      positions[o] = i + dx;
      positions[o + 1] = j + dy;
      positions[o + 2] =
        (amplitude / 2) *
        (2 + Math.sin(i * 0.05 + phase) * Math.cos(j * 0.04 - phase) + 0.3 * Math.sin(i * 0.31 + j * 0.27));
    }
  }
  return { positions, indices: gridIndices(n), triangles: 2 * n * n };
}

/** The Rust bench's terrain: z = sin(0.05 x) cos(0.07 y) 4 on the unit grid. */
function nativeTerrain(n: number): Grid {
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
  return { positions, indices: gridIndices(n), triangles: 2 * n * n };
}

function gridIndices(n: number): Uint32Array {
  const w = n + 1;
  const indices = new Uint32Array(6 * n * n);
  let t = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * w + i;
      indices.set([a, a + 1, a + w + 1, a, a + w + 1, a + w], t);
      t += 6;
    }
  }
  return indices;
}

/** B flipped 180 degrees about x, then turned `deg` about z around (cx, cy), lifted by `lift`. */
function flippedPose(deg: number, cx: number, cy: number, lift: number): PlacementInput {
  const h = (deg * Math.PI) / 360;
  const sz = Math.sin(h);
  const cz = Math.cos(h);
  const c2 = cz * cz - sz * sz;
  const s2 = 2 * sz * cz;
  return {
    rotation: [cz, sz, 0, 0],
    translation: [cx - (c2 * cx - s2 * cy), cy - (s2 * cx + c2 * cy), lift],
  };
}

/** The Rust bench's `contact_lift`: B turned by `deg` just touches A (sampled on a grid). */
function nativeContactLift(side: number, deg: number): number {
  const r = (deg * Math.PI) / 180;
  const s = Math.sin(r);
  const c = Math.cos(r);
  const mid = side / 2;
  const step = Math.max(1, Math.ceil(side / 500));
  const height = (x: number, y: number) => Math.sin(x * 0.05) * Math.cos(y * 0.07) * 4;
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
function nativePose(side: number, deg: number): PlacementInput {
  const lift = nativeContactLift(side, deg);
  const r = (deg * Math.PI) / 180;
  const s = Math.sin(r);
  const c = Math.cos(r);
  const mid = side / 2;
  const tx = -mid;
  const ty = side - mid;
  return {
    rotation: [Math.cos(r / 2), Math.sin(r / 2), 0, 0],
    translation: [c * tx - s * ty + mid, s * tx + c * ty + mid, lift],
  };
}

const WIDE = { maxDistance: 8, range: [0, 6] as const };

// ---- The check Worker ----

interface CheckStats {
  chunks: { kind: string; ms: number }[];
  maxGap: number;
  /** The WebAssembly memory's size in bytes. */
  memory: number;
}

/**
 * A check Worker as a `WorkerLike`: it times every request batch posted and
 * every result received, and answers the check messages.
 */
class CheckWorker implements WorkerLike {
  readonly worker: Worker;
  /** Request batches posted. */
  posted = 0;
  /** When each request batch was posted, in order. */
  sentAt: number[] = [];
  /** Per result, in order: when it was handled, and whether it recoloured a mesh. */
  results: { at: number; colored: boolean }[] = [];
  /** The Worker's `error` events. */
  errors: string[] = [];
  #listeners: Record<string, ((e: unknown) => void)[]> = { message: [], error: [], messageerror: [] };
  #waiters: (() => void)[] = [];
  #stats: ((s: CheckStats) => void)[] = [];

  constructor(flags: readonly string[]) {
    const url = new URL(checkWorkerUrl, location.href);
    for (const flag of flags) {
      url.searchParams.set(flag, "");
    }
    this.worker = new Worker(url, { type: "module" });
    this.worker.addEventListener("message", (e) => {
      const data = e.data;
      if (data?.kind === "checkStats") {
        for (const w of this.#stats.splice(0)) {
          w(data);
        }
        return;
      }
      for (const cb of this.#listeners.message) {
        cb(e);
      }
      if (data?.kind === undefined) {
        const changes: { colors: unknown[] | "all" }[] = data.changes ?? [];
        this.results.push({ at: now(), colored: changes.some((c) => c.colors === "all" || c.colors.length > 0) });
        this.#wake();
      }
    });
    for (const type of ["error", "messageerror"] as const) {
      this.worker.addEventListener(type, (e) => {
        this.errors.push(e instanceof ErrorEvent ? e.message : type);
        this.#wake();
        for (const cb of this.#listeners[type]) {
          cb(e);
        }
      });
    }
  }

  #wake(): void {
    for (const w of this.#waiters.splice(0)) {
      w();
    }
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    if (Array.isArray(msg)) {
      this.posted++;
      this.sentAt.push(now());
    }
    this.worker.postMessage(msg, transfer ?? []);
  }

  addEventListener(type: "message" | "error" | "messageerror", cb: (e: unknown) => void): void {
    this.#listeners[type].push(cb);
  }

  terminate(): void {
    this.worker.terminate();
  }

  /** Resolves once every posted batch has its result; returns the results since `from`. */
  async drained(from = 0): Promise<{ at: number; colored: boolean }[]> {
    while (this.results.length < this.posted) {
      if (this.errors.length > 0) {
        throw new Error(`the check Worker failed: ${this.errors.join("; ")}`);
      }
      await new Promise<void>((resolve) => this.#waiters.push(resolve));
    }
    return this.results.slice(from);
  }

  /** The probes since the last call (call only when drained). */
  stats(): Promise<CheckStats> {
    const reply = new Promise<CheckStats>((resolve) => this.#stats.push(resolve));
    this.worker.postMessage({ checkStats: true });
    return reply;
  }
}

/** A Worker scene on check Workers (`flags`: the check Worker's query switches); `workers` grows on restart. */
async function openCheckScene(
  flags: readonly string[],
  options: { threads?: number; debug?: boolean } = {},
): Promise<{ scene: WorkerScene; workers: CheckWorker[] }> {
  const workers: CheckWorker[] = [];
  const scene = await createWorkerScene({
    ...options,
    createWorker: () => {
      const w = new CheckWorker(flags);
      workers.push(w);
      return w;
    },
  });
  return { scene, workers };
}

/** Sends this frame's requests and waits for their results; returns the round trip in ms. */
async function roundTrip(scene: WorkerScene, worker: CheckWorker): Promise<number> {
  const from = worker.results.length;
  const t0 = now();
  scene.commitFrame();
  await worker.drained(from);
  return now() - t0;
}

// ---- compare: single-threaded against threaded, bit for bit ----

/** A copy of the bytes of `a`. */
function bytes(a: ArrayBufferView): Uint8Array {
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength).slice();
}

const text = (value: unknown) => new TextEncoder().encode(JSON.stringify(value));

/** Change records with handles as ids, as text. */
function record(scene: WorkerScene, changes: MeshChange[]): string {
  return JSON.stringify(
    changes.map((c) => ({
      ...c,
      mesh: scene.idOf(c.mesh),
      fieldsChanged: c.fieldsChanged.map((f) => scene.idOf(f)),
      sectionsChanged: c.sectionsChanged.map((s) => scene.idOf(s)),
    })),
  );
}

interface Run {
  threads: number;
  startMs: number;
  runMs: number;
  /** Change records of every `commitFrame()`, in order. */
  log: string[];
  /** Labelled bytes: mirrors, placements, field values and section results along the way. */
  dump: [string, Uint8Array][];
}

/** The op sequence on a Worker scene with the package's own Worker script. */
async function sequence(threads: number | undefined, n: number): Promise<Run> {
  const t0 = now();
  const scene = await createWorkerScene(threads === undefined ? {} : { threads });
  const startMs = now() - t0;
  const log: string[] = [];
  const dump: [string, Uint8Array][] = [];
  const frame = () => log.push(record(scene, scene.commitFrame()));
  const settle = async <T>(p: Promise<T>): Promise<T> => {
    frame();
    const value = await p;
    frame();
    return value;
  };
  const meshes: [string, MeshHandle][] = [];
  const fields: [string, FieldHandle][] = [];
  const snapshot = async (at: string) => {
    for (const [name, m] of meshes) {
      const v = scene.views(m);
      dump.push([`${at} ${name} positions`, bytes(v.positions())]);
      const normals = v.normals();
      const colors = v.colors();
      dump.push([`${at} ${name} normals`, normals === null ? text(null) : bytes(normals)]);
      dump.push([`${at} ${name} colors`, colors === null ? text(null) : bytes(colors)]);
      dump.push([`${at} ${name} indices`, bytes(v.indices())]);
      dump.push([`${at} ${name} placement`, text(scene.placement(m))]);
      dump.push([`${at} ${name} matrix4`, bytes(scene.matrix4(m))]);
    }
    for (const [name, f] of fields) {
      dump.push([`${at} field ${name}`, bytes(await settle(scene.fetchFieldValues(f)))]);
    }
  };
  const sectionDump = (at: string, section: SectionHandle) => {
    for (const [name, m] of meshes) {
      const r = scene.section(section, m);
      dump.push([
        `${at} section ${name}`,
        text({
          status: r.status,
          length: r.length,
          area: r.area,
          contours: r.contours.map((c) => ({ closed: c.closed, points: Array.from(c.points) })),
          fill: r.fill === undefined ? null : { positions: Array.from(r.fill.positions), indices: Array.from(r.fill.indices) },
        }),
      ]);
    }
  };

  const a = terrain(n);
  const b = terrain(n, { phase: 1.3 });
  const meshA = scene.addMesh(a.positions, a.indices);
  const meshB = scene.addMesh(b.positions, b.indices);
  meshes.push(["A", meshA], ["B", meshB]);
  const cx = n / 2;
  const cy = n / 2;
  scene.setPlacement(meshB, flippedPose(0, cx, cy, 8));
  const gap = await settle(scene.meshDistance(meshA, meshB));
  const lift = 8 - (gap?.distance ?? 0);
  const pose = (i: number) => flippedPose(0.1 * i, cx, cy, lift);
  scene.setPlacement(meshB, pose(0));
  // Wide colouring both ways: every vertex of the pair is in range.
  const wideA = scene.addDistanceField({ name: "wide", on: meshA, to: meshB, maxDistance: WIDE.maxDistance });
  const wideB = scene.addDistanceField({ name: "wide", on: meshB, to: meshA, maxDistance: WIDE.maxDistance });
  fields.push(["wideA", wideA], ["wideB", wideB]);
  scene.setColoring(meshA, { field: wideA, map: "turbo", range: WIDE.range, below: "clamp", above: "base" });
  scene.setColoring(meshB, { field: wideB, map: "viridis", range: WIDE.range });
  await settle(scene.fieldStatus(wideA));
  await snapshot("coloured");

  // Drags, one pose at a time (each one's colours are waited for, so both runs see the same batches).
  const drag = scene.drag(meshB);
  for (let k = 1; k <= 4; k++) {
    drag.move(pose((16 * k) % 64));
    await settle(scene.fieldStatus(wideA));
    await snapshot(`drag ${k}`);
  }
  drag.end();
  await settle(scene.fieldStatus(wideA));

  // An edit: lift a patch of A.
  const w = n + 1;
  const vertices: number[] = [];
  const positions: number[] = [];
  for (let j = Math.floor(n / 4); j < Math.floor(n / 4) + 24; j++) {
    for (let i = Math.floor(n / 4); i < Math.floor(n / 4) + 24; i++) {
      const v = j * w + i;
      vertices.push(v);
      positions.push(a.positions[3 * v], a.positions[3 * v + 1], a.positions[3 * v + 2] + 0.75);
    }
  }
  await settle(scene.edit(meshA, [{ op: "setPositions", vertices, positions }]));
  await snapshot("edit");

  // A section across both, filled.
  const section = scene.addSection({ plane: { origin: [cx, cy, 0], normal: [1, 0, 0] }, meshes: [meshA, meshB], fill: true });
  await settle(scene.fieldStatus(wideA));
  sectionDump("section", section);

  // Undo twice: the edit (or the section, if that is a step), then what came before.
  scene.undo();
  await settle(scene.fieldStatus(wideA));
  sectionDump("undo 1", section);
  await snapshot("undo 1");
  scene.undo();
  await settle(scene.fieldStatus(wideA));
  sectionDump("undo 2", section);
  await snapshot("undo 2");

  const runMs = now() - t0 - startMs;
  const used = scene.threads;
  scene.terminate();
  return { threads: used, startMs, runMs, log, dump };
}

/** The first `max` differences between two runs. */
function differences(x: Run, y: Run, max = 5): string[] {
  const out: string[] = [];
  if (x.log.length !== y.log.length) {
    out.push(`change records: ${x.log.length} frames against ${y.log.length}`);
  }
  for (let i = 0; i < Math.min(x.log.length, y.log.length) && out.length < max; i++) {
    if (x.log[i] !== y.log[i]) {
      out.push(`change records of frame ${i} differ`);
    }
  }
  if (x.dump.length !== y.dump.length) {
    out.push(`dumps: ${x.dump.length} entries against ${y.dump.length}`);
  }
  for (let i = 0; i < Math.min(x.dump.length, y.dump.length) && out.length < max; i++) {
    const [la, ba] = x.dump[i];
    const [lb, bb] = y.dump[i];
    if (la !== lb) {
      out.push(`dump entry ${i}: ${la} against ${lb}`);
      continue;
    }
    if (ba.length !== bb.length) {
      out.push(`${la}: ${ba.length} bytes against ${bb.length}`);
      continue;
    }
    const at = ba.findIndex((v, k) => v !== bb[k]);
    if (at >= 0) {
      out.push(`${la}: first differing byte ${at} of ${ba.length}`);
    }
  }
  return out;
}

async function digest(run: Run): Promise<string> {
  const parts = [...run.log.map((s) => new TextEncoder().encode(s)), ...run.dump.map(([, b]) => b)];
  const all = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    all.set(p, o);
    o += p.length;
  }
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", all));
  return Array.from(hash.slice(0, 12), (v) => v.toString(16).padStart(2, "0")).join("");
}

async function compare({ n = 300, threads }: { n?: number; threads?: number } = {}) {
  const single = await sequence(1, n);
  const threaded = await sequence(threads, n);
  const diffs = differences(single, threaded);
  const coloured = threaded.dump.filter(([l, b]) => l.endsWith("colors") && b.length > 16).length;
  const summary = async (r: Run) => ({
    threads: r.threads,
    startMs: r.startMs,
    runMs: r.runMs,
    frames: r.log.length,
    entries: r.dump.length,
    bytes: r.dump.reduce((s, [, b]) => s + b.length, 0),
    digest: await digest(r),
  });
  return {
    n,
    triangles: 2 * 2 * n * n,
    single: await summary(single),
    threaded: await summary(threaded),
    colouredEntries: coloured,
    identical: diffs.length === 0,
    differences: diffs,
  };
}

// ---- memory: grow a threaded scene's shared memory ----

async function memory({ targetMb = 512, n = 500, threads }: { targetMb?: number; n?: number; threads?: number } = {}) {
  const { scene, workers } = await openCheckScene([], { threads });
  const worker = workers[0];
  const initial = (await worker.stats()).memory;
  const sizes: number[] = [];
  const meshes: MeshHandle[] = [];
  const t0 = now();
  let size = initial;
  while (size < targetMb * 2 ** 20 && meshes.length < 64) {
    const k = meshes.length;
    const g = terrain(n, { phase: k, dx: k * (n + 4) });
    meshes.push(scene.addMesh(g.positions, g.indices));
    await roundTrip(scene, worker);
    size = (await worker.stats()).memory;
    sizes.push(size);
  }
  const addMs = now() - t0;
  // Everything still works on the grown memory: a field across the last two
  // meshes, its colours, and a query.
  const [a, b] = meshes.slice(-2);
  const field = scene.addDistanceField({ name: "near", on: b, to: a, maxDistance: 8 });
  scene.setColoring(b, { field, map: "turbo", range: [0, 6] });
  const status = scene.fieldStatus(field);
  await roundTrip(scene, worker);
  const fieldStatus = await status;
  scene.commitFrame();
  const colors = scene.views(b).colors();
  const hit = scene.raycast([n / 2 + (meshes.length - 1) * (n + 4), n / 2, 100], [0, 0, -1]);
  await roundTrip(scene, worker);
  const ray = await hit;
  const final = (await worker.stats()).memory;
  const out = {
    threads: scene.threads,
    meshes: meshes.length,
    triangles: meshes.length * 2 * n * n,
    initialBytes: initial,
    finalBytes: final,
    sizes,
    addMs,
    fieldCurrent: (fieldStatus as { current: boolean }).current,
    coloured: colors !== null && colors.some((v, i) => i % 4 !== 3 && v !== 200),
    rayHit: ray !== null && scene.idOf(ray.mesh) === scene.idOf(meshes[meshes.length - 1]),
    workerErrors: worker.errors,
  };
  scene.terminate();
  return out;
}

// ---- restart: a crashed threaded Worker, then restart() ----

let restartState:
  | {
      scene: WorkerScene;
      workers: CheckWorker[];
      meshA: MeshHandle;
      meshB: MeshHandle;
      field: FieldHandle;
      pose: (i: number) => PlacementInput;
      colours: Uint8Array;
    }
  | undefined;

/** Opens a threaded scene with a coloured pair; the driver counts its Workers next. */
async function restartOpen({ n = 200, threads }: { n?: number; threads?: number } = {}) {
  const { scene, workers } = await openCheckScene([], { threads });
  const a = terrain(n);
  const b = terrain(n, { phase: 1.3 });
  const meshA = scene.addMesh(a.positions, a.indices);
  const meshB = scene.addMesh(b.positions, b.indices);
  const pose = (i: number) => flippedPose(0.1 * i, n / 2, n / 2, 4);
  scene.setPlacement(meshB, pose(0));
  const field = scene.addDistanceField({ name: "wide", on: meshA, to: meshB, maxDistance: WIDE.maxDistance });
  scene.setColoring(meshA, { field, map: "turbo", range: WIDE.range });
  await roundTrip(scene, workers[0]);
  scene.commitFrame();
  const colours = bytes(scene.views(meshA).colors()!);
  restartState = { scene, workers, meshA, meshB, field, pose, colours };
  return { threads: scene.threads };
}

/** Crashes the Worker, restarts, and checks the new one: threads, a drag's colours, and back. */
async function restartRun() {
  const st = restartState!;
  const { scene, workers } = st;
  const old = workers[0];
  old.worker.postMessage({ checkPoison: true });
  const t0 = now();
  while (old.errors.length === 0 && now() - t0 < 5000) {
    await sleep(10);
  }
  let poisoned = "";
  try {
    const p = scene.fieldStatus(st.field);
    scene.commitFrame();
    await p;
  } catch (e) {
    poisoned = (e as { code?: string }).code ?? String(e);
  }
  const r0 = now();
  await scene.restart();
  const restartMs = now() - r0;
  const next = workers[workers.length - 1];
  // A drag pose recolours A; dragging back gives the first colours again, bit for bit.
  const drag = scene.drag(st.meshB);
  drag.move(st.pose(32));
  await roundTrip(scene, next);
  scene.commitFrame();
  const moved = bytes(scene.views(st.meshA).colors()!);
  drag.move(st.pose(0));
  drag.end();
  await roundTrip(scene, next);
  scene.commitFrame();
  const back = bytes(scene.views(st.meshA).colors()!);
  const same = (x: Uint8Array, y: Uint8Array) => x.length === y.length && x.every((v, i) => v === y[i]);
  return {
    crashErrors: old.errors,
    poisoned,
    restartMs,
    workersMade: workers.length,
    threads: scene.threads,
    recoloured: !same(moved, st.colours),
    backToFirst: same(back, st.colours),
  };
}

function close() {
  restartState?.scene.terminate();
  restartState = undefined;
}

// ---- nestedFail: the pool's nested Workers cannot load ----

async function nestedFail({ threads }: { threads?: number } = {}) {
  const warnings: string[] = [];
  const warn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map(String).join(" "));
    warn(...args);
  };
  const t0 = now();
  try {
    const { scene, workers } = await openCheckScene(["failNested"], { threads, debug: true });
    const readyMs = now() - t0;
    const n = 100;
    const a = terrain(n);
    const b = terrain(n, { phase: 1.3 });
    const meshA = scene.addMesh(a.positions, a.indices);
    const meshB = scene.addMesh(b.positions, b.indices);
    scene.setPlacement(meshB, flippedPose(0, n / 2, n / 2, 4));
    const field = scene.addDistanceField({ name: "wide", on: meshA, to: meshB, maxDistance: 8 });
    scene.setColoring(meshA, { field, map: "turbo", range: [0, 6] });
    const gap = scene.meshDistance(meshA, meshB);
    scene.commitFrame();
    const distance = (await gap)?.distance ?? null;
    scene.commitFrame();
    const out = {
      readyMs,
      threads: scene.threads,
      workersMade: workers.length,
      outerErrors: workers.map((w) => w.errors),
      warnings,
      works: distance !== null && scene.views(meshA).colors() !== null,
    };
    scene.terminate();
    return out;
  } catch (e) {
    return { readyMs: now() - t0, rejected: `${(e as { code?: string }).code ?? ""} ${String(e)}`, warnings };
  } finally {
    console.warn = warn;
  }
}

// ---- bench: the wide-range drag (bench/scene.bench.mjs 5f) ----

function stats(samples: number[]) {
  const s = Float64Array.from(samples).sort();
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { median: q(0.5), p95: q(0.95), min: s[0], max: s[s.length - 1], n: s.length };
}
const maxOf = (xs: number[]) => xs.reduce((m, x) => Math.max(m, x), 0);
const chunkMs = (st: CheckStats, kind: string) => st.chunks.filter((c) => c.kind === kind).map((c) => c.ms);
const sleepUntil = (t: number) => sleep(Math.max(0, t - now()));

async function bench({
  n = 708,
  kind = "9a",
  threads,
  lockStep = 8,
  seconds = 10,
  hz = 60,
}: { n?: number; kind?: "9a" | "native"; threads?: number; lockStep?: number; seconds?: number; hz?: number } = {}) {
  let t0 = now();
  const { scene, workers } = await openCheckScene(["stats"], { threads });
  const worker = workers[0];
  const g = kind === "9a" ? terrain(n) : nativeTerrain(n);
  const meshA = scene.addMesh(g.positions, g.indices);
  // The "native" pair is one terrain twice (addMesh copies its input).
  const meshB = scene.addMesh(kind === "9a" ? terrain(n, { phase: 1.3 }).positions : g.positions, g.indices);
  let pose: (i: number) => PlacementInput;
  if (kind === "9a") {
    const cx = n / 2;
    const cy = n / 2;
    scene.setPlacement(meshB, flippedPose(0, cx, cy, 8));
    const gap = scene.meshDistance(meshA, meshB);
    await roundTrip(scene, worker);
    const lift = 8 - ((await gap)?.distance ?? 0);
    pose = (i) => flippedPose(0.1 * i, cx, cy, lift);
  } else {
    const cache = new Map<number, PlacementInput>();
    pose = (i) => {
      if (!cache.has(i)) {
        cache.set(i, nativePose(n, 0.1 * i));
      }
      return cache.get(i)!;
    };
  }
  scene.setPlacement(meshB, pose(0));
  const field = scene.addDistanceField({ name: "gap", on: meshA, to: meshB, maxDistance: WIDE.maxDistance });
  scene.setColoring(meshA, { field, map: "turbo", range: WIDE.range, below: "clamp", above: "base" });
  await roundTrip(scene, worker);
  const setupMs = now() - t0;
  scene.commitFrame();
  const drag = scene.drag(meshB);
  await roundTrip(scene, worker);
  await worker.stats();

  // Lock step: one move, then wait for its colours, at poses 16, 32, 48, 0 in turn.
  const ls = { rt: [] as number[], block: [] as number[], open: [] as number[], steps: [] as number[], colored: 0 };
  for (let k = 1; k <= lockStep; k++) {
    const from = worker.results.length;
    const f0 = now();
    drag.move(pose((16 * k) % 64));
    scene.commitFrame();
    const results = await worker.drained(from);
    ls.rt.push(now() - f0);
    ls.colored += results.filter((r) => r.colored).length;
    const st = await worker.stats();
    const steps = chunkMs(st, "fieldStep");
    ls.block.push(st.maxGap);
    ls.open.push(steps[0] ?? 0);
    ls.steps.push(steps.length);
  }

  // Continuous drag: a move every 1000/hz ms for `seconds`, turning B through
  // the 64 poses and back, then release.
  const period = 1000 / hz;
  const frames = Math.round((seconds * 1000) / period);
  const first = worker.results.length;
  const firstSent = worker.sentAt.length;
  const main: number[] = [];
  t0 = now();
  for (let k = 0; k < frames; k++) {
    await sleepUntil(t0 + k * period);
    const f0 = now();
    const i = k % 128;
    drag.move(pose(i < 64 ? i : 127 - i));
    scene.commitFrame();
    main.push(now() - f0);
  }
  const lastSent = worker.sentAt[worker.sentAt.length - 1];
  drag.end();
  scene.commitFrame();
  const results = await worker.drained(first);
  const settled = now();
  scene.commitFrame();
  const st = await worker.stats();
  const sent = worker.sentAt.slice(firstSent);
  const lag: number[] = [];
  let backlog = 0;
  results.forEach((r, k) => {
    backlog = Math.max(backlog, sent.filter((s) => s < r.at).length - k - 1);
    if (r.colored) {
      lag.push(r.at - sent[k]);
    }
  });
  const updates = results.filter((r) => r.colored && r.at <= lastSent).map((r) => r.at);
  const tail = results.slice(-2);
  const final = [...tail].reverse().find((r) => r.colored) ?? tail[0];
  const steps = chunkMs(st, "fieldStep");
  const out = {
    n,
    kind,
    triangles: 2 * n * n,
    threads: scene.threads,
    setupMs,
    lockStep: {
      moveToColours: stats(ls.rt),
      recoloured: ls.colored,
      workerBlockPerPose: stats(ls.block),
      unitOpen: stats(ls.open),
      stepsPerPose: stats(ls.steps),
    },
    continuous: {
      hz,
      seconds,
      moves: frames,
      recolouredPoses: lag.length,
      abandonedUnits: chunkMs(st, "fieldStepAbandon").length,
      moveToColours: lag.length > 0 ? stats(lag) : null,
      updateInterval: updates.length > 1 ? stats(updates.slice(1).map((at, k) => at - updates[k])) : null,
      releaseToColours: final.at - lastSent,
      releaseRecoloured: final.colored,
      drainedAfterLastMove: settled - lastSent,
      mostBatchesWaiting: backlog,
      longestFieldStep: maxOf(steps),
      longestWorkerBlock: st.maxGap,
      mainPerFrame: stats(main),
    },
    memoryBytes: st.memory,
  };
  scene.terminate();
  return out;
}

function info() {
  return {
    isolated: crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer === "function",
    cores: navigator.hardwareConcurrency,
    userAgent: navigator.userAgent,
  };
}

const fmCheck = { info, compare, memory, restartOpen, restartRun, close, nestedFail, bench };
(window as unknown as { fmCheck: typeof fmCheck }).fmCheck = fmCheck;
document.querySelector("#out")!.textContent = `ready: ${JSON.stringify(info())}`;
