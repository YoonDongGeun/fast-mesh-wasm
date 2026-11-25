// Geodesic queries (#274): the Worker runs them in steps, so that one can be
// cancelled (a `cancel` request to the engine, an `AbortSignal` to the SDK),
// and the heat method's operators are kept between queries of a mesh. The
// engine part runs in-process with its test-only `onChunk` hook, as in
// worker-chunking.test.ts; the rest over a `worker_threads` Worker.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { generatePrimitive, type GeodesicOptions, type MeshHandle, Scene } from "../src/node.js";
import { runEngine } from "../src/worker/engine.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import type { ClientId, Request, Result } from "../src/worker/protocol.js";

const HEAT: GeodesicOptions = { method: "heatMethod" };

function sphere(subdivisions: number): { positions: Float32Array; indices: Uint32Array } {
  return generatePrimitive({ kind: "icosphere", radius: 1, subdivisions });
}

/** The mean relative error of distances on the unit sphere, from a quarter radian on. */
function meanError(positions: Float32Array, source: number, d: Float32Array): number {
  const at = (v: number) => [positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]];
  const a = at(source);
  let sum = 0;
  let count = 0;
  for (let v = 0; v < d.length; v++) {
    const b = at(v);
    const cross = Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
    const exact = Math.atan2(cross, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
    if (exact > 0.25) {
      sum += Math.abs(d[v] - exact) / exact;
      count++;
    }
  }
  return sum / count;
}

describe("the engine", () => {
  let revision = 0;
  let queries = 0;
  const query = (name: string, args: unknown): Extract<Request, { kind: "query" }> => ({
    kind: "query",
    rev: ++revision,
    qid: ++queries,
    name,
    args,
  });
  const cancel = (qid: number): Request => ({ kind: "cancel", rev: ++revision, qid });

  function add(id: ClientId, mesh: { positions: Float32Array; indices: Uint32Array }): Request {
    return {
      kind: "addMesh",
      rev: ++revision,
      id,
      positions: mesh.positions.slice().buffer as ArrayBuffer,
      f64: false,
      indices: mesh.indices.slice().buffer as ArrayBuffer,
      options: {},
    };
  }

  /** An engine that yields after every small step; `onStep` sees the steps of geodesic queries. */
  function engine() {
    const results: Result[] = [];
    const chunks: string[] = [];
    const waiting: (() => void)[] = [];
    let deliver!: (batch: Request[]) => void;
    const h = {
      results,
      chunks,
      onStep: undefined as ((n: number) => void) | undefined,
      steps: () => chunks.filter((k) => k === "geodesicStep").length,
      send(...batch: Request[]) {
        deliver(batch);
      },
      /** Delivers a batch between two chunks of the engine. */
      later(...batch: Request[]) {
        queueMicrotask(() => deliver(batch));
      },
      async until(n: number) {
        while (results.length < n) {
          await new Promise<void>((resolve) => waiting.push(resolve));
        }
      },
      reply(qid: number) {
        const replies = results.flatMap((r) => r.replies).filter((r) => r.qid === qid);
        expect(replies).toHaveLength(1);
        return replies[0];
      },
    };
    runEngine(
      {
        post: (msg) => {
          results.push(msg);
          for (const f of waiting.splice(0)) {
            f();
          }
        },
        onMessage: (cb) => {
          deliver = cb;
        },
      },
      {
        yieldMs: 0,
        geodesicStepWork: 1,
        onChunk: (chunk) => {
          chunks.push(chunk.kind);
          if (chunk.kind === "geodesicStep") {
            h.onStep?.(h.steps());
          }
        },
      },
    );
    return h;
  }

  const mesh = sphere(4);

  it("runs a query in steps, with the result of one call", async () => {
    using s = new Scene();
    const m = s.addMesh(mesh.positions, mesh.indices);
    const h = engine();
    h.send(add(1, mesh));
    await h.until(1);
    for (const options of [HEAT, { method: "dijkstra" } as GeodesicOptions, { ...HEAT, maxDistance: 1, timeStepScale: 2 }]) {
      const before = h.steps();
      const distances = query("geodesicDistances", { mesh: 1, sources: [3, 70], options });
      const path = query("shortestSurfacePath", { mesh: 1, sources: [3], targetVertex: 500, options: { ...options, maxDistance: undefined } });
      const n = h.results.length;
      h.send(distances);
      h.send(path);
      await h.until(n + 2);
      expect(h.steps()).toBeGreaterThan(before + (options.method === "heatMethod" ? 20 : 2));
      expect(h.reply(distances.qid).value).toEqual(s.geodesicDistances(m, [3, 70], options));
      expect(h.reply(path.qid).value).toEqual(s.shortestSurfacePath(m, 3, 500, { ...options, maxDistance: undefined }));
    }
    // With its distances given, a path takes no steps.
    const before = h.steps();
    const distances = s.geodesicDistances(m, 3, HEAT);
    const path = query("shortestSurfacePath", { mesh: 1, sources: [3], targetVertex: 500, options: { ...HEAT, distances } });
    const n = h.results.length;
    h.send(path);
    await h.until(n + 1);
    expect(h.steps()).toBe(before);
    expect(h.reply(path.qid).value).toEqual(s.shortestSurfacePath(m, 3, 500, { distances }));
  });

  it("ends a running query at its next step when it is cancelled", async () => {
    const h = engine();
    const distances = query("geodesicDistances", { mesh: 1, sources: [0], options: HEAT });
    const measure = query("measure", { mesh: 1 });
    h.onStep = (n) => {
      if (n === 5) {
        h.later(cancel(distances.qid));
        h.later(measure);
      }
    };
    h.send(add(1, mesh));
    h.send(distances);
    await h.until(4);
    expect(h.steps()).toBeLessThanOrEqual(6);
    expect(h.reply(distances.qid)).toMatchObject({ ok: false, error: { code: "CANCELLED" } });
    // What was queued behind it runs, one result per batch.
    expect(h.reply(measure.qid).ok).toBe(true);
    expect(h.results).toHaveLength(4);

    // The next query is as good as any.
    using s = new Scene();
    const m = s.addMesh(mesh.positions, mesh.indices);
    h.onStep = undefined;
    const again = query("geodesicDistances", { mesh: 1, sources: [0], options: HEAT });
    h.send(again);
    await h.until(5);
    expect(h.reply(again.qid).value).toEqual(s.geodesicDistances(m, 0, HEAT));
  });

  it("keeps the operators of a query cancelled after they were built", async () => {
    const h = engine();
    const first = query("geodesicDistances", { mesh: 1, sources: [0], options: HEAT });
    h.send(add(1, mesh));
    h.send(first);
    await h.until(2);
    const whole = h.steps();
    const second = query("geodesicDistances", { mesh: 1, sources: [9], options: HEAT });
    h.send(second);
    await h.until(3);
    const reused = h.steps() - whole;
    expect(reused).toBeLessThan(whole);
    // Released, they are built again; cancelled near the end, they are kept.
    const release = query("releaseGeodesics", {});
    const third = query("geodesicDistances", { mesh: 1, sources: [9], options: HEAT });
    h.send(release);
    const start = h.steps();
    h.onStep = (n) => {
      if (n - start === whole - 2) {
        h.later(cancel(third.qid));
      }
    };
    h.send(third);
    await h.until(6);
    expect(h.reply(release.qid)).toMatchObject({ ok: true, value: null });
    expect(h.reply(third.qid).ok).toBe(false);
    h.onStep = undefined;
    const before = h.steps();
    const fourth = query("geodesicDistances", { mesh: 1, sources: [9], options: HEAT });
    h.send(fourth);
    await h.until(7);
    expect(h.steps() - before).toBe(reused);
    expect(h.reply(fourth.qid).value).toEqual(h.reply(second.qid).value);
  });

  it("drops a query cancelled before it starts, and leaves a finished one alone", async () => {
    const h = engine();
    const first = query("geodesicDistances", { mesh: 1, sources: [0], options: HEAT });
    const waiting = query("measure", { mesh: 1 });
    h.onStep = (n) => {
      if (n === 3) {
        h.later(cancel(waiting.qid));
      }
    };
    h.send(add(1, mesh));
    h.send(first);
    h.send(waiting);
    await h.until(4);
    expect(h.reply(first.qid).ok).toBe(true);
    expect(h.reply(waiting.qid)).toMatchObject({ ok: false, error: { code: "CANCELLED" } });
    h.send(cancel(first.qid));
    await h.until(5);
    expect(h.reply(first.qid).ok).toBe(true);
    // A path is cancelled while its distances are computed.
    const path = query("shortestSurfacePath", { mesh: 1, sources: [0], targetVertex: 60, options: HEAT });
    const from = h.steps();
    h.onStep = (n) => {
      if (n - from === 2) {
        h.later(cancel(path.qid));
      }
    };
    h.send(path);
    await h.until(7);
    expect(h.reply(path.qid).ok).toBe(false);
  });

  it("reports what is wrong with a query, and goes on", async () => {
    const h = engine();
    const bad = [
      query("geodesicDistances", { mesh: 9, sources: [0], options: HEAT }),
      query("geodesicDistances", { mesh: 1, sources: [1e6], options: HEAT }),
      query("geodesicDistances", { mesh: 1, sources: [], options: HEAT }),
      query("shortestSurfacePath", { mesh: 1, sources: [0], targetVertex: 1e6, options: HEAT }),
    ];
    h.send(add(1, mesh));
    h.send(...bad);
    await h.until(2);
    expect(bad.map((q) => h.reply(q.qid).error?.code)).toEqual([
      "UNKNOWN_MESH",
      expect.stringMatching(/^INVALID_/),
      "INVALID_ARGUMENT",
      expect.stringMatching(/^INVALID_/),
    ]);
    const good = query("geodesicDistances", { mesh: 1, sources: [0], options: HEAT });
    h.send(good);
    await h.until(3);
    expect(h.reply(good.qid).ok).toBe(true);
  });
});

class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly sent: Request[] = [];
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true },
    });
    this.worker.on("message", (data) => {
      for (const l of [...this.#message]) l({ data });
    });
    this.worker.on("error", (err) => {
      const ev = { message: err instanceof Error ? err.message : String(err) };
      for (const l of [...this.#error]) l(ev);
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    if (Array.isArray(msg)) {
      this.sent.push(...(msg as Request[]));
    }
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", listener: (e: never) => void): void {
    (type === "message" ? this.#message : this.#error).push(listener as never);
  }

  removeEventListener(type: "message" | "error", listener: (e: never) => void): void {
    const list = type === "message" ? this.#message : this.#error;
    const i = list.indexOf(listener as never);
    if (i >= 0) list.splice(i, 1);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

describe("a Worker scene", () => {
  let scene: WorkerScene | undefined;
  let worker!: NodeWorker;

  afterEach(() => {
    scene?.terminate();
    scene = undefined;
  });

  async function open(subdivisions: number): Promise<{ s: WorkerScene; m: MeshHandle; mesh: ReturnType<typeof sphere> }> {
    const s = await createWorkerScene({ createWorker: () => (worker = new NodeWorker()) });
    scene = s;
    const mesh = sphere(subdivisions);
    const m = s.addMesh(mesh.positions, mesh.indices);
    await s.measure(m);
    return { s, m, mesh };
  }

  const sentQueries = () =>
    worker.sent.filter((r) => r.kind === "query" && (r.name === "geodesicDistances" || r.name === "shortestSurfacePath")).length;
  const sentCancels = () => worker.sent.filter((r) => r.kind === "cancel").length;

  it("gives the distances of a local scene, which are accurate", async () => {
    const { s, m, mesh } = await open(5);
    using local = new Scene();
    const lm = local.addMesh(mesh.positions, mesh.indices);
    for (const source of [0, 3000]) {
      const d = await s.inspect.geodesicDistances(m, source, HEAT);
      expect(d).toEqual(local.inspect.geodesicDistances(lm, source, HEAT));
      expect(meanError(mesh.positions, source, d)).toBeLessThan(0.01);
    }
    await expect(s.inspect.releaseGeodesics()).resolves.toBeUndefined();
    expect(local.inspect.releaseGeodesics()).toBeUndefined();
    expect(await s.inspect.geodesicDistances(m, 0, HEAT)).toEqual(local.inspect.geodesicDistances(lm, 0, HEAT));
  });

  it("rejects with the signal's reason and never sends a query aborted before", async () => {
    const { s, m } = await open(2);
    const before = sentQueries();
    const done = AbortSignal.abort(new Error("no longer wanted"));
    await expect(s.inspect.geodesicDistances(m, 0, { ...HEAT, signal: done })).rejects.toThrow("no longer wanted");
    await expect(s.inspect.shortestSurfacePath(m, 0, 5, { signal: done })).rejects.toThrow("no longer wanted");
    // Aborted before the requests of this turn go out.
    const c = new AbortController();
    const d = s.inspect.geodesicDistances(m, 0, { ...HEAT, signal: c.signal });
    c.abort();
    await expect(d).rejects.toMatchObject({ name: "AbortError" });
    await s.measure(m);
    expect(sentQueries()).toBe(before);
    expect(sentCancels()).toBe(0);
    // Arguments are checked first.
    await expect(s.inspect.geodesicDistances(m, [], { signal: done })).rejects.toMatchObject({ code: "INVALID_ARGUMENT" });
  });

  it("cancels a running query: the next request does not wait for it", async () => {
    const { s, m, mesh } = await open(7); // 327,680 triangles
    const c = new AbortController();
    let settled = false;
    const d = s.inspect.geodesicDistances(m, 0, { ...HEAT, signal: c.signal }).finally(() => {
      settled = true;
    });
    const path = s.inspect.shortestSurfacePath(m, 0, 99, { ...HEAT, signal: c.signal });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(settled).toBe(false);
    const start = performance.now();
    c.abort();
    await expect(d).rejects.toMatchObject({ name: "AbortError" });
    await expect(path).rejects.toMatchObject({ name: "AbortError" });
    const rejected = performance.now() - start;
    await s.inspect.measure(m);
    const free = performance.now() - start;
    expect(sentCancels()).toBe(2);

    const whole = performance.now();
    await s.inspect.geodesicDistances(m, 0, HEAT);
    const took = performance.now() - whole;
    // (Wide margins: the machine may be busy.)
    expect(rejected).toBeLessThan(Math.max(50, took / 2));
    expect(free).toBeLessThan(took);

    // Aborting a settled query does nothing, and its listener is gone.
    const again = await s.inspect.geodesicDistances(m, 7, { ...HEAT, signal: c.signal }).catch((e: unknown) => e);
    expect(again).toMatchObject({ name: "AbortError" });
    const c2 = new AbortController();
    const values = await s.inspect.geodesicDistances(m, 7, { ...HEAT, signal: c2.signal });
    expect(values.length).toBe(mesh.positions.length / 3);
    c2.abort();
    await s.measure(m);
    expect(sentCancels()).toBe(2);
  });

  it("checks the signal", async () => {
    const { s, m } = await open(1);
    for (const signal of [5, null, {}, { aborted: false }]) {
      await expect(s.inspect.geodesicDistances(m, 0, { signal: signal as never })).rejects.toMatchObject({
        code: "INVALID_ARGUMENT",
      });
    }
    using local = new Scene();
    const ball = sphere(1);
    const lm = local.addMesh(ball.positions, ball.indices);
    expect(() => local.inspect.geodesicDistances(lm, 0, { signal: 5 as never })).toThrow("AbortSignal");
    const reason = new Error("stop");
    expect(() => local.inspect.geodesicDistances(lm, 0, { signal: AbortSignal.abort(reason) })).toThrow(reason);
    expect(() => local.inspect.shortestSurfacePath(lm, 0, 3, { signal: AbortSignal.abort(reason) })).toThrow(reason);
    expect(local.inspect.geodesicDistances(lm, 0, { signal: new AbortController().signal }).length).toBe(42);
  });
});
