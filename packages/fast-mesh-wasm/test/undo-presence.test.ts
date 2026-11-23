// Undoable adds and removals (#269): `Scene` and `WorkerScene` (against the
// Worker engine in the Node `worker_threads` harness) bring a mesh back with
// the same handle, data and display state, their change records say so, a
// Worker's mirrors follow, and the store's snapshot follows the records.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { type FieldHandle, type MeshChange, type MeshHandle, Scene } from "../src/node.js";
import { createSceneStore, meshState, type SceneStore } from "../src/store/index.js";
import { debugDump } from "../src/worker/client.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), { workerData: { serve: true } });

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    this.worker.on(type, (data: unknown) => cb(type === "message" ? { data } : data));
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

const cleanup: (() => void)[] = [];
afterEach(() => {
  for (const f of cleanup.splice(0)) {
    f();
  }
});

type AnyScene = Scene | WorkerScene;

async function makeLocal(): Promise<Scene> {
  const scene = new Scene();
  cleanup.push(() => scene.free());
  return scene;
}

async function makeWorker(options: { backgroundAddTriangles?: number } = {}): Promise<WorkerScene> {
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker(), ...options });
  cleanup.push(() => scene.terminate());
  return scene;
}

/** A closed unit cube offset by `dx` along x, with UVs. */
function cube(dx = 0) {
  const positions: number[] = [];
  const uvs: number[] = [];
  for (let i = 0; i < 8; i++) {
    positions.push((i & 1) + dx, (i >> 1) & 1, (i >> 2) & 1);
    uvs.push((i & 1) / 2, ((i >> 1) & 1) / 4);
  }
  return {
    positions: Float64Array.from(positions),
    indices: Uint32Array.from([
      0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
    ]),
    uvs: Float32Array.from(uvs),
  };
}

/** Sends the frame, waits for every result so far and delivers it. Returns the records delivered. */
async function settle(s: AnyScene): Promise<MeshChange[]> {
  if (s instanceof Scene) {
    return s.commitFrame();
  }
  const barrier = s.raycast([0, 0, 1e6], [0, 0, 1]);
  const before = s.commitFrame();
  await barrier;
  return [...before, ...s.commitFrame()];
}

/** The records about `mesh`, as `[added, removed, restorable]` (`-` for other changes). */
function about(records: MeshChange[], mesh: MeshHandle): string[] {
  return records
    .filter((c) => c.mesh === mesh)
    .map((c) => (c.added ? "added" : c.removed ? (c.restorable === true ? "parked" : "removed") : "-"));
}

/** Everything drawn of a mesh, copied. */
function drawn(s: AnyScene, mesh: MeshHandle) {
  const v = s.views(mesh);
  return {
    positions: [...v.positions()],
    normals: v.normals() === null ? null : [...v.normals()!],
    colors: v.colors() === null ? null : [...v.colors()!],
    uvs: v.uvs() === null ? null : [...v.uvs()!],
    indices: [...v.indices()],
    matrix4: [...s.matrix4(mesh)],
    placement: s.placement(mesh),
    bounds: s.bounds(mesh),
    coloring: s.coloring(mesh),
  };
}

/** The triangles drawn of a mesh, as positions (slots an undone edit left behind are not). */
function triangles(s: AnyScene, mesh: MeshHandle): number[][] {
  const v = s.views(mesh);
  const [p, idx] = [v.positions(), v.indices()];
  const out: number[][] = [];
  for (let t = 0; t < idx.length; t += 3) {
    const tri = [idx[t], idx[t + 1], idx[t + 2]];
    if (tri.some((x) => x !== 0)) {
      out.push(tri.flatMap((x) => [...p.subarray(3 * x, 3 * x + 3)]));
    }
  }
  return out;
}

/** The Worker's mirrors of a mesh are the engine's buffers. */
async function mirrored(s: AnyScene, mesh: MeshHandle): Promise<void> {
  if (s instanceof Scene) {
    return;
  }
  const pending = debugDump(s, mesh);
  s.commitFrame();
  const dump = await pending;
  const v = s.views(mesh);
  expect(v.positions()).toEqual(dump.positions);
  expect(v.normals()).toEqual(dump.normals);
  expect(v.colors()).toEqual(dump.colors);
  expect(v.uvs()).toEqual(dump.uvs);
  expect(v.indices()).toEqual(dump.indices);
  expect([...s.matrix4(mesh)]).toEqual(dump.matrix4);
}

function code(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    return (e as { code?: string }).code ?? String(e);
  }
  return "no error";
}

function canUndo(s: AnyScene): boolean {
  return s instanceof Scene ? s.canUndo() : s.canUndo;
}

for (const [name, make] of [
  ["Scene", makeLocal],
  ["WorkerScene", makeWorker],
] as const) {
  describe(`undoable add and remove: ${name}`, () => {
    it("undo of an add removes the mesh, redo brings it back with its handle and data", async () => {
      const s: AnyScene = await make();
      const c = cube();
      const other = s.addMesh(c.positions, c.indices);
      await settle(s);
      expect(canUndo(s)).toBe(false);
      const mesh = s.addMesh(c.positions, c.indices, { uvs: c.uvs, origin: [0.25, 0, 0], undoable: true });
      expect(about(await settle(s), mesh)).toEqual(["added"]);
      expect(canUndo(s)).toBe(true);
      expect([s.has(mesh), s.isRestorable(mesh)]).toEqual([true, false]);
      const before = drawn(s, mesh);
      expect(before.uvs).toEqual([...c.uvs]);

      for (let k = 0; k < 2; k++) {
        s.undo();
        expect(about(await settle(s), mesh)).toEqual(["parked"]);
        expect([s.has(mesh), s.isRestorable(mesh)]).toEqual([false, true]);
        expect(code(() => s.views(mesh))).toBe("UNKNOWN_MESH");
        expect(code(() => s.setPlacement(mesh, { rotation: IDENTITY, translation: [1, 0, 0] }))).toBe("UNKNOWN_MESH");
        expect(code(() => s.removeMesh(mesh))).toBe("UNKNOWN_MESH");
        expect(canUndo(s)).toBe(false);
        s.redo();
        expect(about(await settle(s), mesh)).toEqual(["added"]);
        expect([s.has(mesh), s.isRestorable(mesh)]).toEqual([true, false]);
        expect(drawn(s, mesh)).toEqual(before);
        await mirrored(s, mesh);
      }
      // It is a mesh like any other again.
      s.setPlacement(mesh, { rotation: IDENTITY, translation: [5, 0, 0] });
      expect(about(await settle(s), mesh)).toEqual(["-"]);
      expect(s.placement(mesh).translation).toEqual([5, 0, 0]);
      expect(s.has(other)).toBe(true);
    });

    it("undo of a removal brings back data, placement, normals, fields and colouring", async () => {
      const s: AnyScene = await make();
      const [a, b] = [cube(), cube(1.5)];
      const mesh = s.addMesh(a.positions, a.indices, { uvs: a.uvs });
      const target = s.addMesh(b.positions, b.indices);
      s.setPlacement(mesh, { rotation: IDENTITY, translation: [0, 0.25, 0] });
      await s.edit(mesh, [{ op: "setPositions", vertices: [7], positions: [1.25, 1.5, 1] }]);
      const field = s.addDistanceField({ name: "d", on: mesh, to: target, maxDistance: 10 });
      s.setColoring(mesh, { field, map: "turbo", range: [0, 2] });
      await settle(s);
      const before = drawn(s, mesh);
      expect(before.colors).not.toBeNull();
      expect(before.coloring?.field).toBe(field);
      const values = async (f: FieldHandle) =>
        s instanceof Scene ? [...s.fieldValues(f)] : [...(await frame(s, s.fetchFieldValues(f)))];
      const distances = await values(field);

      for (let k = 0; k < 2; k++) {
        s.removeMesh(mesh, { undoable: true });
        expect([s.has(mesh), s.isRestorable(mesh)]).toEqual([false, true]);
        expect(about(await settle(s), mesh)).toEqual(["parked"]);
        expect(code(() => s.views(mesh))).toBe("UNKNOWN_MESH");
        s.undo();
        const records = await settle(s);
        expect(about(records, mesh)).toEqual(["added"]);
        expect(s.has(mesh)).toBe(true);
        expect(drawn(s, mesh)).toEqual(before);
        await mirrored(s, mesh);
        expect(await values(field)).toEqual(distances);
        // The history before the removal is still there.
        s.undo();
        await settle(s);
        expect(s.views(mesh).positions()[21]).not.toBe(before.positions[21]);
        s.redo();
        await settle(s);
        expect(drawn(s, mesh)).toEqual(before);
        s.redo();
        expect(about(await settle(s), mesh)).toEqual(["parked"]);
        s.undo();
        await settle(s);
        expect(drawn(s, mesh)).toEqual(before);
      }
    });

    it("adds, edits, moves and removals of one step undo together", async () => {
      const s: AnyScene = await make();
      const c = cube();
      const old = s.addMesh(c.positions, c.indices);
      await settle(s);
      const was = drawn(s, old);
      s.beginStep();
      // The pattern tools use: add empty, fill, move; and remove the input.
      const made = s.addMesh(new Float64Array(0), new Uint32Array(0), { origin: [0, 0, 0], undoable: true });
      const filled = await frame(
        s,
        s.edit(made, [
          { op: "addVertices", positions: c.positions },
          { op: "addTriangles", triangles: Array.from(c.indices, (v) => ({ ref: [0, v] as const })) },
        ]),
      );
      expect(filled.added[0].length).toBe(8);
      s.setPlacement(made, { rotation: IDENTITY, translation: [0, 0, 3] });
      s.removeMesh(old, { undoable: true });
      s.endStep();
      await settle(s);
      const is = drawn(s, made);
      expect(is.indices.length).toBe(36);
      for (let k = 0; k < 2; k++) {
        s.undo();
        await settle(s);
        expect([s.has(old), s.has(made)]).toEqual([true, false]);
        expect(drawn(s, old)).toEqual(was);
        await mirrored(s, old);
        expect(canUndo(s)).toBe(false);
        s.redo();
        await settle(s);
        expect([s.has(old), s.has(made)]).toEqual([false, true]);
        expect(drawn(s, made)).toEqual(is);
        await mirrored(s, made);
      }
      // A cancelled step takes its adds back for good, and its removals.
      s.beginStep();
      const gone = s.addMesh(c.positions, c.indices, { undoable: true });
      s.removeMesh(made, { undoable: true });
      s.cancelStep();
      await settle(s);
      expect([s.has(made), s.has(gone), s.isRestorable(gone)]).toEqual([true, false, false]);
      expect(drawn(s, made)).toEqual(is);
      await mirrored(s, made);
    });

    it("a removal undone within the frame reports the mesh as changed, never as gone", async () => {
      const s: AnyScene = await make();
      const c = cube();
      const mesh = s.addMesh(c.positions, c.indices);
      await settle(s);
      const before = drawn(s, mesh);
      s.removeMesh(mesh, { undoable: true });
      s.undo();
      const records = (await settle(s)).filter((r) => r.mesh === mesh);
      expect(records.map((r) => [r.added, r.removed, r.positions, r.placementChanged])).toEqual([
        [false, false, "all", true],
      ]);
      expect([s.has(mesh), s.isRestorable(mesh)]).toEqual([true, false]);
      expect(drawn(s, mesh)).toEqual(before);
      // And the mesh takes calls again.
      s.setPlacement(mesh, { rotation: IDENTITY, translation: [1, 0, 0] });
      await settle(s);
      expect(s.placement(mesh).translation).toEqual([1, 0, 0]);
    });

    it("an add undone within the frame can be redone", async () => {
      const s: AnyScene = await make();
      const c = cube();
      const mesh = s.addMesh(c.positions, c.indices, { undoable: true });
      s.undo();
      await settle(s);
      expect([s.has(mesh), s.isRestorable(mesh)]).toEqual([false, true]);
      s.redo();
      expect(about(await settle(s), mesh)).toEqual(["added"]);
      expect(s.views(mesh).indices().length).toBe(36);
      await mirrored(s, mesh);
    });

    it("what the history drops is gone for good", async () => {
      const s: AnyScene = await make();
      const c = cube();
      const keep = s.addMesh(c.positions, c.indices);
      const undone = s.addMesh(c.positions, c.indices, { undoable: true });
      s.undo();
      await settle(s);
      expect(s.isRestorable(undone)).toBe(true);
      // A new step drops the redo.
      s.setPlacement(keep, { rotation: IDENTITY, translation: [1, 0, 0] });
      await settle(s);
      expect([s.has(undone), s.isRestorable(undone)]).toEqual([false, false]);
      s.removeMesh(keep, { undoable: true });
      await settle(s);
      expect(s.isRestorable(keep)).toBe(true);
      s.clearHistory();
      await settle(s);
      expect([s.has(keep), s.isRestorable(keep)]).toEqual([false, false]);
      expect(code(() => s.removeMesh(keep))).toBe("UNKNOWN_MESH");
      // A permanent removal is final, whatever the history.
      const last = s.addMesh(c.positions, c.indices, { undoable: true });
      await settle(s);
      s.removeMesh(last);
      expect(about(await settle(s), last)).toEqual(["removed"]);
      expect([s.isRestorable(last), canUndo(s)]).toEqual([false, false]);
    });

    it("hulls, cuts and batch booleans undo as one step", async () => {
      const s: AnyScene = await make();
      const [a, b] = [cube(), cube(0.5)];
      const ma = s.addMesh(a.positions, a.indices);
      const mb = s.addMesh(b.positions, b.indices);
      await settle(s);
      const before = triangles(s, ma);

      const hull = await frame(s, s.convexHull([ma, mb]));
      await settle(s);
      const cut = await frame(s, s.boolean.cut(ma, { origin: [0.5, 0, 0], normal: [1, 0, 0] }, { cap: true }));
      await settle(s);
      const union = await frame(
        s,
        s.boolean.booleanBatch([ma, mb], {
          kind: "op",
          op: "union",
          left: { kind: "leaf", meshIndex: 0 },
          right: { kind: "leaf", meshIndex: 1 },
        }),
      );
      await settle(s);
      const all = () => [hull.mesh, cut.removed, union].map((m) => s.has(m));
      expect(all()).toEqual([true, true, true]);
      s.undo();
      await settle(s);
      expect(all()).toEqual([true, true, false]);
      s.undo();
      await settle(s);
      expect(all()).toEqual([true, false, false]);
      expect(triangles(s, ma)).toEqual(before);
      s.undo();
      await settle(s);
      expect(all()).toEqual([false, false, false]);
      expect(canUndo(s)).toBe(false);
      for (let k = 0; k < 3; k++) {
        s.redo();
      }
      await settle(s);
      expect(all()).toEqual([true, true, true]);
      for (const m of [ma, hull.mesh, cut.removed, union]) {
        await mirrored(s, m);
      }
    });

    it("cuts the links other meshes own, and keeps fields between meshes removed together", async () => {
      const s: AnyScene = await make();
      const [a, b, c] = [cube(), cube(2), cube(4)];
      const [ma, mb, mc] = [a, b, c].map((x) => s.addMesh(x.positions, x.indices)) as [MeshHandle, MeshHandle, MeshHandle];
      const toB = s.addDistanceField({ name: "toB", on: ma, to: mb, maxDistance: 10 });
      const toBoth = s.addDistanceField({ name: "toBoth", on: ma, to: [mb, mc], maxDistance: 10 });
      const bc = s.addDistanceField({ name: "bc", on: mb, to: mc, maxDistance: 10 });
      s.setColoring(ma, { field: toB, map: "turbo", range: [0, 2] });
      s.setColoring(mb, { field: bc, map: "turbo", range: [0, 2] });
      const section = s.addSection({ plane: { origin: [0, 0, 0.5], normal: [0, 0, 1] }, meshes: [ma, mb] });
      await settle(s);
      const shown = drawn(s, mb);
      expect(s.section(section, mb).contours.length).toBe(1);

      s.beginStep();
      s.removeMesh(mb, { undoable: true });
      s.removeMesh(mc, { undoable: true });
      s.endStep();
      await settle(s);
      expect(drawn(s, ma).colors).toBeNull();
      expect(code(() => s.setColoring(ma, { field: toB, map: "turbo", range: [0, 2] }))).toBe("UNKNOWN_FIELD");

      s.undo();
      await settle(s);
      expect(drawn(s, mb)).toEqual(shown);
      await mirrored(s, mb);
      // `mb`'s own field and colouring are back, those of others are not.
      expect(s.coloring(mb)?.field).toBe(bc);
      expect(s.coloring(ma)).toBeNull();
      expect(drawn(s, ma).colors).toBeNull();
      const nearest = s instanceof Scene ? [...s.fieldNearest(bc)] : [...(await frame(s, s.fetchFieldNearest(bc)))];
      expect(nearest.some((n) => n === 0)).toBe(true);
      expect(code(() => s.removeField(toB))).toBe("UNKNOWN_FIELD");
      // `toBoth` lost both its targets, one after the other.
      expect(code(() => s.removeField(toBoth))).toBe("UNKNOWN_FIELD");
      // Removing the field that came back works.
      s.removeField(bc);
      await settle(s);
      expect(drawn(s, mb).colors).toBeNull();
      await mirrored(s, mb);
    });
  });
}

/** Sends the frame a promise call waits for, then awaits it. */
async function frame<T>(s: AnyScene, value: T | Promise<T>): Promise<T> {
  if (!(s instanceof Scene)) {
    s.commitFrame();
  }
  return value;
}

describe("WorkerScene", () => {
  it("an undoable add lands in its step, never in the background", async () => {
    const s = await makeWorker({ backgroundAddTriangles: 0 });
    const c = cube();
    const other = s.addMesh(c.positions, c.indices);
    s.beginStep();
    const mesh = s.addMesh(c.positions, c.indices, { undoable: true });
    s.setPlacement(other, { rotation: IDENTITY, translation: [2, 0, 0] });
    s.endStep();
    await settle(s);
    s.undo();
    await settle(s);
    expect(s.has(mesh)).toBe(false);
    expect(s.placement(other).translation).toEqual([0, 0, 0]);
    expect(s.canUndo).toBe(false);
  });

  it("frees the state of a mesh the history dropped", async () => {
    const s = await makeWorker();
    const c = cube();
    const mesh = s.addMesh(c.positions, c.indices, { undoable: true });
    const id = s.idOf(mesh);
    await settle(s);
    s.undo();
    await settle(s);
    expect(s.idOf(mesh)).toBe(id);
    expect(s.status(`mesh:${id}`).status).toBe("ready");
    s.clearHistory();
    await settle(s);
    expect(code(() => s.idOf(mesh))).toBe("UNKNOWN_MESH");
    expect(s.status(`mesh:${id}`).status).toBe("idle");
    expect(s.has(mesh)).toBe(false);
  });

  it("requests made before the undo's result fail on the mesh it removed", async () => {
    const s = await makeWorker();
    const c = cube();
    const mesh = s.addMesh(c.positions, c.indices, { undoable: true });
    await settle(s);
    s.undo();
    const edit = s.edit(mesh, [{ op: "removeTriangles", triangles: [0] }]);
    s.commitFrame();
    await expect(edit).rejects.toMatchObject({ code: "UNKNOWN_MESH" });
    await settle(s);
    s.redo();
    await settle(s);
    expect(s.views(mesh).indices().length).toBe(36);
    await mirrored(s, mesh);
  });

  it("a removal requested while an undo is in flight decides", async () => {
    const s = await makeWorker();
    const c = cube();
    const mesh = s.addMesh(c.positions, c.indices);
    await settle(s);
    s.removeMesh(mesh, { undoable: true });
    await settle(s);
    s.undo();
    s.commitFrame();
    // Not known to be back yet: the client still refuses.
    expect(code(() => s.removeMesh(mesh))).toBe("UNKNOWN_MESH");
    await settle(s);
    expect(s.has(mesh)).toBe(true);
    // Removed, undone and removed again within one batch.
    s.removeMesh(mesh, { undoable: true });
    s.undo();
    await settle(s);
    s.removeMesh(mesh);
    expect(about(await settle(s), mesh)).toEqual(["removed"]);
    expect([s.has(mesh), s.isRestorable(mesh)]).toEqual([false, false]);
    expect(code(() => s.idOf(mesh))).toBe("UNKNOWN_MESH");
  });

  it("removals of a mesh that an undo in flight took out", async () => {
    const s = await makeWorker();
    const c = cube();
    for (const undoable of [true, false]) {
      const mesh = s.addMesh(c.positions, c.indices, { undoable: true });
      await settle(s);
      s.undo();
      s.removeMesh(mesh, { undoable });
      await settle(s);
      expect([s.has(mesh), s.isRestorable(mesh)]).toEqual([false, undoable]);
      s.redo();
      await settle(s);
      // Removed for good while it was out: it does not stay when it is back.
      expect(s.has(mesh)).toBe(undoable);
      if (undoable) {
        expect(s.views(mesh).indices().length).toBe(36);
        await mirrored(s, mesh);
        s.removeMesh(mesh);
        await settle(s);
      }
      const hit = s.raycast([0.25, 0.25, 5], [0, 0, -1]);
      s.commitFrame();
      expect(await hit).toBeNull();
    }
  });

  it("an undoable removal of a mesh whose add failed is final", async () => {
    const s = await makeWorker();
    const bad = s.addMesh(new Float64Array([0, 0, 0]), new Uint32Array([0, 1, 2]), { undoable: true });
    await settle(s);
    expect(s.status(`mesh:${s.idOf(bad)}`).status).toBe("error");
    s.removeMesh(bad, { undoable: true });
    expect(about(await settle(s), bad)).toEqual(["removed"]);
    expect([s.has(bad), s.isRestorable(bad), s.canUndo]).toEqual([false, false, false]);
    expect(code(() => s.idOf(bad))).toBe("UNKNOWN_MESH");
  });

  it("restart rebuilds from the mirrors; what only the history held is gone", async () => {
    const s = await makeWorker();
    const [a, b] = [cube(), cube(2)];
    const live = s.addMesh(a.positions, a.indices, { undoable: true });
    const back = s.addMesh(b.positions, b.indices, { uvs: b.uvs });
    const undone = s.addMesh(b.positions, b.indices, { undoable: true });
    const removed = s.addMesh(b.positions, b.indices);
    const field = s.addDistanceField({ name: "f", on: removed, to: live, maxDistance: 10 });
    s.setColoring(removed, { field, map: "turbo", range: [0, 2] });
    await settle(s);
    // `back` leaves and returns: its mirrors must be whole again.
    s.removeMesh(back, { undoable: true });
    await settle(s);
    s.undo();
    await settle(s);
    s.undo();
    await settle(s);
    expect([s.has(undone), s.isRestorable(undone)]).toEqual([false, true]);
    s.removeMesh(removed, { undoable: true });
    await settle(s);
    const before = [live, back].map((m) => drawn(s, m));

    const done = s.restart();
    await done;
    await settle(s);
    expect([live, back, undone, removed].map((m) => s.has(m))).toEqual([true, true, false, false]);
    expect([undone, removed].map((m) => s.isRestorable(m))).toEqual([false, false]);
    expect([s.canUndo, s.canRedo]).toEqual([false, false]);
    for (const [k, m] of [live, back].entries()) {
      const now = drawn(s, m);
      expect(now.positions).toEqual(before[k].positions);
      expect(now.indices).toEqual(before[k].indices);
      expect(now.uvs).toEqual(before[k].uvs);
      await mirrored(s, m);
    }
    expect(code(() => s.removeField(field))).toBe("UNKNOWN_FIELD");
    expect(code(() => s.idOf(undone))).toBe("UNKNOWN_MESH");
    // And undoable removals work in the new Worker.
    s.removeMesh(live, { undoable: true });
    await settle(s);
    s.undo();
    await settle(s);
    expect(drawn(s, live).positions).toEqual(before[0].positions);
    await mirrored(s, live);
  });
});

for (const [name, make] of [
  ["local", makeLocal],
  ["worker", makeWorker],
] as const) {
  describe(`SceneStore follows undoable adds and removals: ${name}`, () => {
    async function open(): Promise<{ store: SceneStore; s: AnyScene; sync(): Promise<MeshChange[]> }> {
      const s: AnyScene = await make();
      const store = createSceneStore(s);
      const sync = async (): Promise<MeshChange[]> => {
        const first = store.commitFrame();
        if (s instanceof Scene) {
          return first;
        }
        const barrier = s.raycast([0, 0, 1e6], [0, 0, 1]);
        const second = store.commitFrame();
        await barrier;
        return [...first, ...second, ...store.commitFrame()];
      };
      return { store, s, sync };
    }
    const has = (store: SceneStore, m: MeshHandle) => store.getSnapshot().meshes.has(m);

    it("an undone add leaves the snapshot and the selection, a redone one is back", async () => {
      const { store, sync } = await open();
      const c = cube();
      const mesh = store.addMesh(c.positions, c.indices, { undoable: true });
      store.setDisplay(mesh, { opacity: 0.5 });
      store.select(mesh);
      await sync();
      let calls = 0;
      store.subscribe(() => calls++);
      store.undo();
      await sync();
      expect(has(store, mesh)).toBe(false);
      expect(store.getSnapshot().selection).toBeNull();
      expect(store.isRestorable(mesh)).toBe(true);
      expect(calls).toBeGreaterThan(0);
      expect(() => store.setDisplay(mesh, { opacity: 1 })).toThrow(/not in the store/);
      store.redo();
      await sync();
      expect(has(store, mesh)).toBe(true);
      expect(store.isRestorable(mesh)).toBe(false);
      expect(meshState(store.getSnapshot(), mesh).display.opacity).toBe(0.5);
      expect(meshState(store.getSnapshot(), mesh).status).not.toBe("error");
      expect(store.getSnapshot().status).toBe("idle");
    });

    it("a removed mesh comes back with its colouring, and colours again what was measured to it", async () => {
      const { store, s, sync } = await open();
      const [a, b, c] = [cube(), cube(2), cube(4)];
      const [ma, mb, mc] = [a, b, c].map((x) => store.addMesh(x.positions, x.indices)) as [
        MeshHandle,
        MeshHandle,
        MeshHandle,
      ];
      const coloring = (to: MeshHandle | MeshHandle[]) => ({
        distanceTo: to,
        maxDistance: 10,
        range: [0, 2] as [number, number],
        map: "turbo" as const,
      });
      store.setDisplay(mb, { coloring: coloring(mc), opacity: 0.25 });
      store.setDisplay(ma, { coloring: coloring(mb) });
      store.setDisplay(mc, { coloring: coloring([ma, mb]) });
      const section = store.addSection({ plane: { origin: [0, 0, 0.5], normal: [0, 0, 1] }, meshes: [ma, mb] });
      await sync();
      await sync();
      const colors = () => [ma, mb, mc].map((m) => [...(s.views(m).colors() ?? [])]);
      const before = colors();
      expect(before.every((x) => x.length === 32)).toBe(true);
      const shown = () => [ma, mb, mc].map((m) => meshState(store.getSnapshot(), m).display);
      const displays = shown();

      for (let k = 0; k < 2; k++) {
        store.removeMesh(mb, { undoable: true });
        expect(has(store, mb)).toBe(false);
        await sync();
        await sync();
        expect(store.isRestorable(mb)).toBe(true);
        expect(meshState(store.getSnapshot(), ma).display.coloring).toBeNull();
        expect(meshState(store.getSnapshot(), ma).status).toBe("error");
        expect(meshState(store.getSnapshot(), mc).display.coloring?.distanceTo).toEqual([ma]);
        expect(store.getSnapshot().sections.get(section)!.meshes).toEqual([ma]);

        store.undo();
        await sync();
        await sync();
        await sync();
        expect(shown()).toEqual(displays);
        expect(colors()).toEqual(before);
        expect([ma, mb, mc].map((m) => meshState(store.getSnapshot(), m).status)).toEqual(["ready", "ready", "ready"]);
        expect(store.getSnapshot().status).toBe("idle");
        expect(new Set(store.getSnapshot().sections.get(section)!.meshes)).toEqual(new Set([ma, mb]));
        expect(s.section(section, mb).contours.length).toBe(1);
        expect(store.coloringTargets(mb)).toEqual([mc]);
        await mirrored(s, mb);
      }

      // A colouring changed meanwhile is left alone.
      store.removeMesh(mb, { undoable: true });
      store.setDisplay(ma, { coloring: null });
      store.setDisplay(mc, { coloring: coloring(ma) });
      await sync();
      store.undo();
      await sync();
      await sync();
      expect(meshState(store.getSnapshot(), ma).desired.coloring).toBeNull();
      expect(meshState(store.getSnapshot(), mc).desired.coloring?.distanceTo).toBe(ma);
      expect(meshState(store.getSnapshot(), mb).desired.coloring?.distanceTo).toBe(mc);
      expect(meshState(store.getSnapshot(), mb).display.coloring?.distanceTo).toBe(mc);

      // A target removed for good while the mesh is out.
      store.removeMesh(mb, { undoable: true });
      await sync();
      store.removeMesh(mc);
      await sync();
      store.undo();
      await sync();
      await sync();
      expect(has(store, mb)).toBe(true);
      expect(meshState(store.getSnapshot(), mb).display.coloring).toBeNull();
      expect(s.views(mb).colors()).toBeNull();
      store.setDisplay(mb, { coloring: coloring(ma) });
      await sync();
      await sync();
      expect(meshState(store.getSnapshot(), mb).status).toBe("ready");
      expect(s.views(mb).colors()).not.toBeNull();
    });

    it("meshes removed in one step come back coloured by each other", async () => {
      const { store, s, sync } = await open();
      const [a, b] = [cube(), cube(2)];
      const ma = store.addMesh(a.positions, a.indices);
      const mb = store.addMesh(b.positions, b.indices);
      for (const [m, to] of [
        [ma, mb],
        [mb, ma],
      ]) {
        store.setDisplay(m, { coloring: { distanceTo: to, maxDistance: 10, range: [0, 2], map: "turbo" } });
      }
      await sync();
      await sync();
      const colors = () => [ma, mb].map((m) => [...(s.views(m).colors() ?? [])]);
      const before = colors();
      store.beginStep();
      store.removeMesh(ma, { undoable: true });
      store.removeMesh(mb, { undoable: true });
      store.endStep();
      await sync();
      expect(store.getSnapshot().meshes.size).toBe(0);
      store.undo();
      await sync();
      await sync();
      await sync();
      expect(colors()).toEqual(before);
      expect([ma, mb].map((m) => meshState(store.getSnapshot(), m).status)).toEqual(["ready", "ready"]);
      expect([ma, mb].map((m) => meshState(store.getSnapshot(), m).display.coloring?.distanceTo)).toEqual([mb, ma]);
    });

    it("forgets what the history drops, and adopts meshes that ops add", async () => {
      const { store, s, sync } = await open();
      const c = cube();
      const mesh = store.addMesh(c.positions, c.indices);
      await sync();
      const hull = await (async () => {
        const p = s.convexHull(mesh);
        await sync();
        return (await p).mesh;
      })();
      await sync();
      store.adoptMesh(hull);
      store.undo();
      await sync();
      expect(has(store, hull)).toBe(false);
      store.redo();
      await sync();
      expect(has(store, hull)).toBe(true);
      store.removeMesh(hull, { undoable: true });
      await sync();
      expect(store.isRestorable(hull)).toBe(true);
      s.clearHistory();
      await sync();
      expect(store.isRestorable(hull)).toBe(false);
      store.undo();
      await sync();
      expect(has(store, hull)).toBe(false);
      // A cancelled step.
      store.beginStep();
      const tmp = store.addMesh(c.positions, c.indices, { undoable: true });
      store.removeMesh(mesh, { undoable: true });
      store.cancelStep();
      await sync();
      await sync();
      expect([has(store, tmp), has(store, mesh)]).toEqual([false, true]);
    });
  });
}

/** A seeded generator of numbers in [0, 1). */
function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("random sequences", () => {
  it.each(Array.from({ length: 24 }, (_, k) => k + 1))("Scene, WorkerScene and their stores agree (seed %i)", async (seed) => {
    const rnd = random(seed * 7919);
    const pick = <T>(list: readonly T[]): T => list[Math.floor(rnd() * list.length)];
    const local = await makeLocal();
    const worker = await makeWorker({ backgroundAddTriangles: seed % 2 === 0 ? 0 : 65_536 });
    const scenes = [local, worker] as const;
    const stores = scenes.map((s) => createSceneStore(s)) as [SceneStore, SceneStore];
    /** Each mesh's handles, local and Worker. */
    const meshes: [MeshHandle, MeshHandle][] = [];
    const both = () => meshes.filter(([a, b]) => local.has(a) && worker.has(b));
    const each = (f: (store: SceneStore, scene: AnyScene, k: 0 | 1) => unknown) => {
      f(stores[0], local, 0);
      f(stores[1], worker, 1);
    };
    const sync = async () => {
      const barrier = worker.raycast([0, 0, 1e6], [0, 0, 1]);
      each((store) => store.commitFrame());
      await barrier;
      each((store) => store.commitFrame());
    };
    // Errors of requests on a mesh that an undo in flight removes are fine.
    const ignored = (p: unknown) => void Promise.resolve(p).catch(() => {});
    const plane = { origin: [0, 0, 0.5] as [number, number, number], normal: [0, 0, 1] as [number, number, number] };
    const sections = stores.map((st) => st.addSection({ plane, meshes: [] }));
    let depth = 0;
    const history: string[] = [];
    for (let n = 0; n < 60; n++) {
      const op = pick(["add", "add", "remove", "remove", "move", "edit", "color", "section", "begin", "end", "cancel", "undo", "undo", "undo", "redo", "redo", "sync", "sync"]);
      history.push(op);
      const live = both();
      const mesh = live.length > 0 ? pick(live) : undefined;
      if (op === "add" && live.length < 5) {
        const c = cube(meshes.length * 1.5);
        const undoable = rnd() < 0.7;
        const uvs = rnd() < 0.5 ? c.uvs : undefined;
        meshes.push(stores.map((st) => st.addMesh(c.positions, c.indices, { undoable, uvs })) as [MeshHandle, MeshHandle]);
      } else if (op === "remove" && mesh !== undefined) {
        const undoable = rnd() < 0.8;
        each((store, _, k) => store.removeMesh(mesh[k], { undoable }));
      } else if (op === "move" && mesh !== undefined) {
        const translation: [number, number, number] = [Math.round(rnd() * 8) / 2, 0, 0];
        each((store, _, k) => store.setPlacement(mesh[k], { rotation: IDENTITY, translation }));
      } else if (op === "edit" && mesh !== undefined) {
        const z = 1 + Math.round(rnd() * 8) / 8;
        const v = Math.floor(rnd() * 8);
        each((store, _, k) => ignored(store.edit(mesh[k], [{ op: "setPositions", vertices: [v], positions: [z, 0, 0] }])));
      } else if (op === "color" && mesh !== undefined) {
        const others = live.filter((m) => m !== mesh);
        const on = others.length > 0 && rnd() < 0.8;
        const to = others.slice(0, rnd() < 0.5 ? 1 : 2);
        each((store, _, k) =>
          store.setDisplay(mesh[k], {
            coloring: on ? { distanceTo: to.map((m) => m[k]), maxDistance: 20, range: [0, 4], map: "turbo" } : null,
          }),
        );
      } else if (op === "section") {
        const cut = live.filter(() => rnd() < 0.7);
        each((store, _, k) => store.setSection(sections[k], { meshes: cut.map((m) => m[k]) }));
      } else if (op === "begin" && depth < 2) {
        depth++;
        each((store) => store.beginStep());
      } else if ((op === "end" || op === "cancel") && depth > 0) {
        depth = op === "end" ? depth - 1 : 0;
        each((store) => (op === "end" ? store.endStep() : store.cancelStep()));
      } else if (op === "undo" || op === "redo") {
        depth = 0;
        each((store) => (op === "undo" ? store.undo() : store.redo()));
      } else if (op !== "sync") {
        continue;
      }
      if (rnd() < 0.4) {
        continue;
      }
      await sync();
      await sync();
      await sync();
      const at = `${history.length}: ${history.join(" ")}`;
      expect(worker.canUndo, at).toBe(local.canUndo());
      expect(worker.canRedo, at).toBe(local.canRedo());
      const cuts = stores.map((st, k) =>
        st
          .getSnapshot()
          .sections.get(sections[k])!
          .meshes.map((m) => meshes.findIndex((p) => p[k] === m))
          .sort(),
      );
      expect(cuts[1], at).toEqual(cuts[0]);
      for (const i of cuts[0]) {
        const [x, y] = scenes.map((s, k) => s.section(sections[k], meshes[i][k]).contours.map((c) => [...c.points]));
        expect(y, at).toEqual(x);
      }
      for (const [a, b] of meshes) {
        expect(worker.has(b), at).toBe(local.has(a));
        expect(worker.isRestorable(b), at).toBe(local.isRestorable(a));
        expect(has2(stores[0], a), at).toBe(local.has(a));
        expect(has2(stores[1], b), at).toBe(local.has(a));
        expect(stores[0].isRestorable(a), at).toBe(local.isRestorable(a));
        expect(stores[1].isRestorable(b), at).toBe(local.isRestorable(a));
        if (!local.has(a)) {
          continue;
        }
        const [x, y] = [drawn(local, a), drawn(worker, b)];
        expect({ ...y, coloring: y.coloring !== null }, at).toEqual({ ...x, coloring: x.coloring !== null });
        const states = [meshState(stores[0].getSnapshot(), a), meshState(stores[1].getSnapshot(), b)];
        const targets = states.map((st, k) => {
          const to = st.display.coloring?.distanceTo as readonly MeshHandle[] | MeshHandle | undefined;
          return (to === undefined ? [] : Array.isArray(to) ? to : [to]).map((m) => meshes.findIndex((p) => p[k] === m));
        });
        expect(targets[1], at).toEqual(targets[0]);
        expect(states[1].status, at).toBe(states[0].status);
        await mirrored(worker, b);
      }
    }
  }, 60_000);
});

function has2(store: SceneStore, m: MeshHandle): boolean {
  return store.getSnapshot().meshes.has(m);
}
