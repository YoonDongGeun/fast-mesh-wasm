// Issue #48 in the Worker: a field add builds its field in steps (sign data,
// near search and values), yielding between them, and a build for a
// superseded display is abandoned at a step boundary. The engine runs
// in-process (sources, same wasm) with its test-only `onChunk` hook, which
// delivers newer batches between chunks.
import { describe, expect, it } from "vitest";
import { Scene } from "../src/node.js";
import { runEngine } from "../src/worker/engine.js";
import type { ClientId, Request, Result } from "../src/worker/protocol.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** Few vertices a step, so a build spans many steps. */
const STEP_POINTS = 16;

/** A wavy `n × n` terrain grid at height `z`, its corner at `(x, y)`. */
function terrain(x: number, y: number, z: number, n = 24, size = 4) {
  const positions: number[] = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const px = x + (size * i) / (n - 1);
      const py = y + (size * j) / (n - 1);
      positions.push(px, py, z + 0.3 * Math.sin(1.7 * px) * Math.cos(1.3 * py));
    }
  }
  const indices: number[] = [];
  for (let j = 0; j + 1 < n; j++) {
    for (let i = 0; i + 1 < n; i++) {
      const a = j * n + i;
      indices.push(a, a + 1, a + n + 1, a, a + n + 1, a + n);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

const A = terrain(0, 0, 0);
const B = terrain(0.5, 0.3, 0.4);
const C = terrain(10, 0, 0, 8);

let revision = 0;
const rev = () => ++revision;

function addMeshReq(id: ClientId, mesh: { positions: Float64Array; indices: Uint32Array }): Request {
  return {
    kind: "addMesh",
    rev: rev(),
    id,
    positions: mesh.positions.slice().buffer,
    f64: true,
    indices: mesh.indices.slice().buffer,
    options: {},
  };
}

function fieldAdd(field: ClientId, on: ClientId, to: ClientId, maxDistance: number, key?: string): Request {
  const req: Request = { kind: "field", rev: rev(), field, op: "add", desc: { name: `f${field}`, on, to: [to], maxDistance } };
  return key === undefined ? req : { ...req, key };
}

/** The engine in-process with a small step; `onChunk` sees every chunk. */
function engine() {
  const results: Result[] = [];
  const chunks: { kind: string; id?: ClientId }[] = [];
  const waiting: (() => void)[] = [];
  let deliver!: (batch: Request[]) => void;
  const h = {
    results,
    chunks,
    onChunk: undefined as ((chunk: { kind: string; id?: ClientId }) => void) | undefined,
    /** Delivers `batch` on a microtask: between two chunks of the engine. */
    later(batch: Request[]) {
      queueMicrotask(() => deliver(batch));
    },
    send(batch: Request[]) {
      deliver(batch);
    },
    async until(n: number) {
      while (results.length < n) {
        await new Promise<void>((resolve) => waiting.push(resolve));
      }
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
      debug: true,
      stepPoints: STEP_POINTS,
      yieldMs: 0,
      onChunk: (chunk) => {
        chunks.push(chunk);
        h.onChunk?.(chunk);
      },
    },
  );
  return h;
}

/** A synchronous scene's values of a field on `a` to `b` (meshes as the engine has them). */
function referenceValues(maxDistance: number): Float32Array {
  const s = new Scene();
  try {
    const a = s.addMesh(A.positions, A.indices);
    const b = s.addMesh(B.positions, B.indices);
    s.addMesh(C.positions, C.indices);
    const f = s.addDistanceField({ name: "r", on: a, to: b, maxDistance });
    return s.fieldValues(f).slice();
  } finally {
    s.free();
  }
}

async function fetchValues(h: ReturnType<typeof engine>, field: ClientId): Promise<Float32Array> {
  const n = h.results.length + 1;
  h.send([{ kind: "fetchFieldValues", rev: rev(), qid: 5000 + n, field }]);
  await h.until(n);
  const reply = h.results[n - 1].replies[0];
  expect(reply.ok).toBe(true);
  return reply.value as Float32Array;
}

const bytes = (v: Float32Array) => Buffer.from(v.buffer, v.byteOffset, v.byteLength);

describe("field builds in steps", () => {
  it("field_add_builds_in_steps_with_yields", async () => {
    const h = engine();
    h.send([addMeshReq(1, A), addMeshReq(2, B), addMeshReq(3, C)]);
    await h.until(1);
    // A batch arriving after the build's first step is received before the
    // build finishes, and runs after it, in its own result.
    const toggle: Request = { kind: "setPlacement", rev: rev(), id: 3, rotation: IDENTITY, translation: [0, 1, 0] };
    let receivedAt = -1;
    h.onChunk = (chunk) => {
      if (chunk.kind === "fieldBuildStep" && receivedAt < 0) {
        receivedAt = h.chunks.length;
        h.later([toggle]);
      }
    };
    h.send([
      fieldAdd(1, 1, 2, Number.POSITIVE_INFINITY),
      { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [0, 1] } },
    ]);
    await h.until(3);
    const kinds = h.chunks.map((c) => c.kind);
    const built = kinds.indexOf("fieldBuild");
    const steps = kinds.filter((k) => k === "fieldBuildStep").length;
    expect(steps).toBeGreaterThan(10);
    expect(receivedAt).toBeGreaterThan(0);
    expect(receivedAt).toBeLessThan(built);
    // The toggle ran after the build: its result is the third.
    expect(kinds.lastIndexOf("setPlacement")).toBeGreaterThan(built);
    expect(h.results[1].status).toContainEqual({ key: "field:1", rev: expect.any(Number), status: "ready" });
    expect(h.results[1].changes.find((c) => c.id === 1)?.fields).toEqual([1]);
    // The same values as a synchronous build, bit for bit.
    expect(bytes(await fetchValues(h, 1)).equals(bytes(referenceValues(Number.POSITIVE_INFINITY)))).toBe(true);
    // A finite reach (near search and sign data both stepped).
    h.onChunk = undefined;
    h.send([fieldAdd(2, 1, 2, 0.3)]);
    await h.until(h.results.length + 1);
    expect(bytes(await fetchValues(h, 2)).equals(bytes(referenceValues(0.3)))).toBe(true);
  });

  it("superseded_display_build_is_abandoned_mid_build", async () => {
    const h = engine();
    h.send([addMeshReq(1, A), addMeshReq(2, B), addMeshReq(3, C)]);
    await h.until(1);
    const display = "display:1";
    const first: Request = { kind: "coloring", rev: rev(), id: 1, rule: { field: 1, map: "turbo", range: [0, 1] }, key: display };
    const second: Request = { kind: "coloring", rev: rev(), id: 1, rule: null, key: display };
    let step = 0;
    h.onChunk = (chunk) => {
      if (chunk.kind === "fieldBuildStep" && ++step === 3) {
        h.later([second]);
      }
    };
    h.send([fieldAdd(1, 1, 2, Number.POSITIVE_INFINITY, display), first]);
    await h.until(3);
    const kinds = h.chunks.map((c) => c.kind);
    expect(kinds).toContain("fieldBuildAbandon");
    expect(kinds).not.toContain("fieldBuild");
    // Nothing of the abandoned build is reported: no field change, no status.
    const [r1, r2] = h.results.slice(1);
    expect(r1.changes.flatMap((c) => c.fields)).toEqual([]);
    expect(r1.status.filter((st) => st.key === display)).toEqual([]);
    expect(r2.status).toEqual([{ key: display, rev: second.rev, status: "ready" }]);
    // The abandoned field still works: it is built when first needed.
    h.onChunk = undefined;
    expect(bytes(await fetchValues(h, 1)).equals(bytes(referenceValues(Number.POSITIVE_INFINITY)))).toBe(true);
    expect(h.chunks.map((c) => c.kind)).toContain("fieldBuild");
  });

  it("field_add_errors_leave_no_field", async () => {
    const h = engine();
    h.send([addMeshReq(1, A), addMeshReq(2, B)]);
    await h.until(1);
    // A taken name fails at once; the client id is free again.
    h.send([fieldAdd(1, 1, 2, 0.3), { ...fieldAdd(2, 1, 2, 0.3), desc: { name: "f1", on: 1, to: [2], maxDistance: 0.3 } }]);
    await h.until(2);
    expect(h.results[1].status.find((st) => st.key === "field:2")?.status).toBe("error");
    h.send([fieldAdd(2, 1, 2, 0.5)]);
    await h.until(3);
    expect(h.results[2].status.find((st) => st.key === "field:2")?.status).toBe("ready");
  });
});
