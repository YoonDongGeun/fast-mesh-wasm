// The Scene class through the Node entry's sources: handles, placements, views,
// change records, frames, history and drags.
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  computeVertexNormals,
  FastMeshError,
  type MeshChange,
  type MeshHandle,
  Scene,
} from "../src/node.js";
import { decodeChanges } from "../src/changes.js";
import { wasmMemory } from "../src/core.js";

// Two triangles on the XY plane facing +Z, and a third tilted one.
const quad = [0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 1, 1, 1];
const quadTris = [0, 1, 2, 0, 2, 3, 1, 2, 4];
const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
// A quarter turn about +z, row-major.
const QUARTER_Z = [0, -1, 0, 1, 0, 0, 0, 0, 1];

function codeOf(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    return (e as FastMeshError).code;
  }
  throw new Error("expected a throw");
}

/** An n×n vertex grid on the XY plane. */
function grid(n: number): { positions: Float64Array; indices: Uint32Array } {
  const positions = new Float64Array(n * n * 3);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      positions.set([x, y, 0], (y * n + x) * 3);
    }
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let k = 0;
  for (let y = 0; y < n - 1; y++) {
    for (let x = 0; x < n - 1; x++) {
      const a = y * n + x;
      indices.set([a, a + 1, a + n + 1, a, a + n + 1, a + n], k);
      k += 6;
    }
  }
  return { positions, indices };
}

/** Applies a column-major 4×4 matrix to a point. */
function transform(m: Float64Array, p: ArrayLike<number>): number[] {
  return [0, 1, 2].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r]);
}

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, eps: number) {
  expect(actual.length).toBe(expected.length);
  for (let k = 0; k < expected.length; k++) {
    expect(Math.abs(actual[k] - expected[k])).toBeLessThanOrEqual(eps);
  }
}

const scenes: Scene[] = [];
function scene(options?: { debug?: boolean }): Scene {
  const s = new Scene(options);
  scenes.push(s);
  return s;
}
afterEach(() => {
  for (const s of scenes.splice(0)) {
    s.free();
  }
  vi.restoreAllMocks();
});

describe("Scene", () => {
  it("add_mesh_world_positions_and_origin", () => {
    const s = scene();
    // Far from the world origin: f32 world coordinates would lose ~0.008.
    const base = [1e5 + 0.123456789, -2e5 + 0.5, 3e5 + 0.987654321];
    const world = quad.map((v, k) => v + base[k % 3]);
    const mesh = s.addMesh(new Float64Array(world), quadTris);
    expect(s.vertexSlots(mesh)).toBe(5);
    expect(s.triangleSlots(mesh)).toBe(3);
    expect(s.vertexCount(mesh)).toBe(5);
    expect(s.triangleCount(mesh)).toBe(3);

    // The default origin is the bounding-box centre, in f64.
    const origin = [0, 1, 2].map((a) => {
      const vs = world.filter((_, k) => k % 3 === a);
      return (Math.min(...vs) + Math.max(...vs)) / 2;
    });
    const local = s.views(mesh).positions();
    expect(Array.from(local)).toEqual(world.map((v, k) => Math.fround(v - origin[k % 3])));
    const m = s.matrix4(mesh);
    expect(m).toBeInstanceOf(Float64Array);
    expect(m.length).toBe(16);
    for (let v = 0; v < 5; v++) {
      expectClose(transform(m, local.subarray(v * 3, v * 3 + 3)), world.slice(v * 3, v * 3 + 3), 1e-6);
    }

    // An explicit origin is used as given; plain arrays are accepted.
    const at = s.addMesh(world, new Uint16Array(quadTris), { origin: [1e5, -2e5, 3e5] });
    const p = s.views(at).positions();
    expect(p[0]).toBe(Math.fround(world[0] - 1e5));
    expectClose(s.matrix4(at).subarray(12), [1e5, -2e5, 3e5, 1], 0);

    expect(codeOf(() => s.addMesh(quad.slice(0, 4), quadTris))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.addMesh(quad, [0, 1, 9]))).toBe("INVALID_INDEX");
    expect(codeOf(() => s.addMesh(quad, quadTris, { origin: [0, 0] as never }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.addMesh(quad, quadTris, { mode: "toString" as never }))).toBe(
      "INVALID_ARGUMENT",
    );
  });

  it("quaternion_and_matrix_placement", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    expect(s.placement(mesh)).toEqual({ rotation: IDENTITY, translation: [0, 0, 0] });

    s.setPlacement(mesh, { rotation: QUARTER_Z, translation: [10, 20, 30] });
    expect(s.placement(mesh)).toEqual({ rotation: QUARTER_Z, translation: [10, 20, 30] });

    // The same quarter turn as a quaternion, unnormalised: normalised first.
    s.setPlacement(mesh, { rotation: [0, 0, 3, 3], translation: [1, 2, 3] });
    const p = s.placement(mesh);
    expectClose(p.rotation, QUARTER_Z, 1e-15);
    expect(p.translation).toEqual([1, 2, 3]);
    // matrix4 = placement · translate(origin); the origin is (1, 1, 0.5).
    const m = s.matrix4(mesh);
    expectClose(m, [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 3, 3.5, 1], 1e-15);

    // A 30° turn about x as a unit quaternion matches its matrix.
    const h = Math.PI / 12;
    s.setPlacement(mesh, { rotation: [Math.sin(h), 0, 0, Math.cos(h)], translation: [0, 0, 0] });
    const c = Math.cos(2 * h);
    const sn = Math.sin(2 * h);
    expectClose(s.placement(mesh).rotation, [1, 0, 0, 0, c, -sn, 0, sn, c], 1e-15);

    const t: [number, number, number] = [0, 0, 0];
    expect(codeOf(() => s.setPlacement(mesh, { rotation: [0, 0, 0, 0], translation: t }))).toBe(
      "INVALID_ARGUMENT",
    );
    // Not a proper rotation: a scale, and a reflection.
    expect(
      codeOf(() => s.setPlacement(mesh, { rotation: [2, 0, 0, 0, 2, 0, 0, 0, 2], translation: t })),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(() => s.setPlacement(mesh, { rotation: [-1, 0, 0, 0, 1, 0, 0, 0, 1], translation: t })),
    ).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.setPlacement(mesh, { rotation: [1, 0, 0], translation: t }))).toBe(
      "INVALID_ARGUMENT",
    );
    // Failed calls leave the placement alone.
    expectClose(s.placement(mesh).rotation, [1, 0, 0, 0, c, -sn, 0, sn, c], 1e-15);
  });

  it("views_match_engine", () => {
    const s = scene();
    const indices = new Uint32Array(quadTris);
    const mesh = s.addMesh(quad, indices);
    const v = s.views(mesh);
    expect(v.indices()).toBeInstanceOf(Uint32Array);
    expect(Array.from(v.indices())).toEqual(quadTris);
    const positions = v.positions();
    const normals = v.normals();
    expect(normals).toBeInstanceOf(Float32Array);
    expectClose(normals as Float32Array, computeVertexNormals(positions, indices), 1e-6);
    // Colouring is off by default.
    expect(v.colors()).toBeNull();
    // Views read wasm memory in place, without copies.
    expect(positions.buffer).toBe(wasmMemory().buffer);
    // Fresh views per call.
    expect(v.positions()).not.toBe(positions);

    // Normals off: no view.
    const flat = s.addMesh(quad, indices, { normals: false, mode: "manual" });
    expect(s.views(flat).normals()).toBeNull();
    s.setMode(flat, "lazy");
    s.update(flat);
    s.flush();
  });

  it("views_survive_memory_growth", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s = scene({ debug: true });
    const mesh = s.addMesh(quad, quadTris);
    const kept = s.views(mesh).positions();
    const expected = Array.from(kept);
    const before = wasmMemory().buffer;
    const big = grid(300);
    for (let k = 0; k < 50 && wasmMemory().buffer === before; k++) {
      s.addMesh(big.positions, big.indices);
    }
    expect(wasmMemory().buffer).not.toBe(before);
    // The kept view is detached; a fresh one reads the same data.
    expect(kept.byteLength).toBe(0);
    expect(warn).not.toHaveBeenCalled();
    const fresh = s.views(mesh).positions();
    expect(Array.from(fresh)).toEqual(expected);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("detached");
    // Once per scene.
    s.views(mesh).indices();
    expect(warn).toHaveBeenCalledTimes(1);

    // Without debug there is no check.
    const quiet = scene();
    const q = quiet.addMesh(quad, quadTris);
    const keptQuiet = quiet.views(q).positions();
    const now = wasmMemory().buffer;
    for (let k = 0; k < 50 && wasmMemory().buffer === now; k++) {
      quiet.addMesh(big.positions, big.indices);
    }
    expect(keptQuiet.byteLength).toBe(0);
    quiet.views(q).positions();
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("stale_handle_after_slot_reuse", () => {
    const s = scene();
    const a = s.addMesh(quad, quadTris);
    s.takeChanges();
    s.removeMesh(a);
    const b = s.addMesh(quad, quadTris);
    // b reuses a's slot under a new generation: a stays dead.
    expect(b).not.toBe(a);
    expect(codeOf(() => s.vertexCount(a))).toBe("UNKNOWN_MESH");
    expect(codeOf(() => s.removeMesh(a))).toBe("UNKNOWN_MESH");
    expect(codeOf(() => s.views(a))).toBe("UNKNOWN_MESH");
    expect(s.vertexCount(b)).toBe(5);
    // The records carry the same handle objects.
    const changes = s.takeChanges();
    expect(changes.map((c) => [c.mesh, c.removed, c.added])).toEqual([
      [a, true, false],
      [b, false, true],
    ]);
    expect(changes[0].mesh).toBe(a);
    expect(changes[1].mesh).toBe(b);

    // A handle of another scene, and a forged one.
    const other = scene();
    const c = other.addMesh(quad, quadTris);
    expect(codeOf(() => s.vertexCount(c))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.vertexCount({} as MeshHandle))).toBe("INVALID_ARGUMENT");
  });

  it("changes_ranges_after_placement_and_drag", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    const [added] = s.takeChanges();
    expect(added).toEqual({
      mesh,
      added: true,
      removed: false,
      placementChanged: false,
      capacityChanged: false,
      positions: "all",
      normals: "all",
      colors: "all",
      indices: "all",
      fieldsChanged: [],
      sectionsChanged: [],
    } satisfies MeshChange);
    expect(s.takeChanges()).toEqual([]);

    const placed: MeshChange = {
      mesh,
      added: false,
      removed: false,
      placementChanged: true,
      capacityChanged: false,
      positions: [],
      normals: [],
      colors: [],
      indices: [],
      fieldsChanged: [],
      sectionsChanged: [],
    };
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [1, 0, 0] });
    expect(s.takeChanges()).toEqual([placed]);

    const drag = s.drag(mesh);
    for (let k = 1; k <= 5; k++) {
      drag.move({ rotation: QUARTER_Z, translation: [k, 0, 0] });
    }
    drag.end();
    expect(s.takeChanges()).toEqual([placed]);

    s.removeMesh(mesh);
    const [removed] = s.takeChanges();
    expect(removed.mesh).toBe(mesh);
    expect(removed.removed).toBe(true);
    expect(removed.positions).toEqual([]);
  });

  it("commit_frame_fans_out_once", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    const got: MeshChange[][] = [];
    s.onChanges((c) => got.push(c));
    const off = s.onChanges((c) => got.push(c));
    const frame = s.commitFrame();
    expect(frame).toHaveLength(1);
    expect(frame[0].mesh).toBe(mesh);
    expect(got).toHaveLength(2);
    expect(got[0]).toBe(frame);
    expect(got[1]).toBe(frame);
    // Drained: the next frame is empty.
    expect(s.commitFrame()).toEqual([]);
    expect(s.takeChanges()).toEqual([]);

    // Unsubscribed listeners get nothing; a throwing one does not starve the rest.
    off();
    got.length = 0;
    s.onChanges(() => {
      throw new Error("listener failed");
    });
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [0, 1, 0] });
    expect(() => s.commitFrame()).toThrow("listener failed");
    expect(got).toHaveLength(1);
    expect(got[0][0].placementChanged).toBe(true);
  });

  it("drag_is_one_undo_step", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    expect(s.canUndo()).toBe(false);
    const drag = s.drag(mesh);
    expect(codeOf(() => s.drag(mesh))).toBe("INVALID_ARGUMENT");
    for (let k = 1; k <= 30; k++) {
      drag.move({ rotation: IDENTITY, translation: [k, 0, 0] });
    }
    drag.end();
    drag.end(); // no-op
    expect(codeOf(() => drag.move({ rotation: IDENTITY, translation: [0, 0, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(s.placement(mesh).translation).toEqual([30, 0, 0]);
    expect(s.undo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    expect(s.canUndo()).toBe(false);
    expect(s.redo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([30, 0, 0]);

    // Cancel puts the placement back and records nothing.
    const c = s.drag(mesh);
    c.move({ rotation: QUARTER_Z, translation: [5, 5, 5] });
    c.cancel();
    expect(s.placement(mesh)).toEqual({ rotation: IDENTITY, translation: [30, 0, 0] });
    expect(s.undo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    expect(s.canUndo()).toBe(false);

    // Inside the caller's step, cancel keeps that step open.
    s.beginStep();
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [1, 0, 0] });
    const n = s.drag(mesh);
    n.move({ rotation: IDENTITY, translation: [2, 0, 0] });
    n.cancel();
    expect(s.placement(mesh).translation).toEqual([1, 0, 0]);
    s.endStep();
    expect(codeOf(() => s.endStep())).toBe("STEP_NOT_OPEN");
    expect(s.undo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);

    // Plain steps.
    s.beginStep();
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [4, 0, 0] });
    s.cancelStep();
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [4, 0, 0] });
    s.clearHistory();
    expect(s.canUndo() || s.canRedo()).toBe(false);
  });

  it("undo_mid_drag", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [1, 0, 0] });
    const drag = s.drag(mesh);
    drag.move({ rotation: IDENTITY, translation: [2, 0, 0] });
    drag.move({ rotation: IDENTITY, translation: [3, 0, 0] });
    // Undo ends the drag first, then undoes it as a whole.
    expect(s.undo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([1, 0, 0]);
    expect(codeOf(() => drag.move({ rotation: IDENTITY, translation: [9, 0, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    drag.end(); // already over: no STEP_NOT_OPEN
    drag.cancel();
    expect(s.placement(mesh).translation).toEqual([1, 0, 0]);
    // Redo brings the drag back; a new drag can start.
    expect(s.redo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([3, 0, 0]);
    const next = s.drag(mesh);
    next.move({ rotation: IDENTITY, translation: [4, 0, 0] });
    next.end();
    expect(s.undo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([3, 0, 0]);
  });

  it("step_calls_during_drag_abandon_it", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);

    // cancelStep takes the drag's step away: the drag is over.
    const d = s.drag(mesh);
    d.move({ rotation: IDENTITY, translation: [1, 0, 0] });
    s.cancelStep();
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    expect(() => s.undo()).not.toThrow();
    expect(() => d.move({ rotation: IDENTITY, translation: [2, 0, 0] })).toThrow("the drag is over");
    d.end();
    d.cancel();
    expect(s.canUndo()).toBe(false);

    // The caller's endStep closing an outermost drag's step: recorded, and over.
    const e = s.drag(mesh);
    e.move({ rotation: IDENTITY, translation: [3, 0, 0] });
    s.endStep();
    expect(() => e.end()).not.toThrow();
    expect(() => e.move({ rotation: IDENTITY, translation: [4, 0, 0] })).toThrow("the drag is over");
    expect(s.undo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);

    // Nested: closing the drag's level leaves the caller's step to them.
    s.beginStep();
    const n = s.drag(mesh);
    n.move({ rotation: IDENTITY, translation: [5, 0, 0] });
    s.endStep();
    n.end();
    s.endStep();
    expect(codeOf(() => s.endStep())).toBe("STEP_NOT_OPEN");

    // A step opened inside the drag closes with it.
    const m = s.drag(mesh);
    s.beginStep();
    m.move({ rotation: IDENTITY, translation: [6, 0, 0] });
    m.end();
    expect(codeOf(() => s.endStep())).toBe("STEP_NOT_OPEN");
    expect(s.undo()).toBe(true);
    expect(s.placement(mesh).translation).toEqual([5, 0, 0]);
  });

  it("empty_views_do_not_warn", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s = scene({ debug: true });
    const mesh = s.addMesh(quad, []);
    expect(s.views(mesh).indices().length).toBe(0);
    const before = wasmMemory().buffer;
    const big = grid(300);
    for (let k = 0; k < 50 && wasmMemory().buffer === before; k++) {
      s.addMesh(big.positions, big.indices);
    }
    expect(wasmMemory().buffer).not.toBe(before);
    s.views(mesh).positions();
    expect(warn).not.toHaveBeenCalled();
  });

  it("decoder_rejects_malformed_messages", () => {
    expect(decodeChanges(new Uint32Array([0]))).toEqual([]);
    expect(() => decodeChanges(new Uint32Array([0, 7]))).toThrow("trailing");
    expect(() => decodeChanges(new Uint32Array([1, 0, 0]))).toThrow("truncated");
    const [c] = decodeChanges(new Uint32Array([1, 3, 1, 4 | 32, 1, 2, 3, 0, 0, 1, 0, 1, 2, 5, 6, 1, 9]));
    expect(c).toMatchObject({
      index: 3,
      generation: 1,
      placementChanged: true,
      positions: [[2, 3]],
      normals: "all",
      colors: [],
      indices: [[0, 1]],
      fields: [5, 6],
      sections: [9],
    });
  });

  it("free_then_use_throws", () => {
    const s = new Scene();
    const mesh = s.addMesh(quad, quadTris);
    const views = s.views(mesh);
    s.free();
    s.free(); // no-op
    expect(codeOf(() => s.addMesh(quad, quadTris))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.vertexCount(mesh))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.takeChanges())).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => views.positions())).toBe("INVALID_ARGUMENT");
    expect(() => s.undo()).toThrow("scene is freed");
  });

  it("using_disposes", () => {
    let outer: Scene | undefined;
    {
      using s = new Scene();
      outer = s;
      s.addMesh(quad, quadTris);
    }
    expect(codeOf(() => outer?.canUndo())).toBe("INVALID_ARGUMENT");
  });
});
