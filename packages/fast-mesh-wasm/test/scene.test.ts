// The Scene class through the Node entry's sources: handles, placements, views,
// change records, frames, history and drags.
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type ColorRule,
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

/**
 * A lumpy closed surface without symmetries: a UV sphere pushed out by an
 * analytic radius and squashed per axis, optionally moved by `shift`.
 */
function lump(stacks: number, slices: number) {
  const point = (x: number, y: number, z: number) => {
    const r = 1 + 0.25 * Math.sin(3 * x + 1) * Math.cos(2 * y) + 0.15 * Math.sin(4 * z);
    return [1.3 * r * x, 0.9 * r * y, 0.6 * r * z];
  };
  const positions = [...point(0, 0, 1)];
  for (let i = 1; i < stacks; i++) {
    const phi = (Math.PI * i) / stacks;
    for (let j = 0; j < slices; j++) {
      const a = (2 * Math.PI * j) / slices;
      positions.push(...point(Math.sin(phi) * Math.cos(a), Math.sin(phi) * Math.sin(a), Math.cos(phi)));
    }
  }
  positions.push(...point(0, 0, -1));
  const south = 1 + (stacks - 1) * slices;
  const ring = (i: number, j: number) => 1 + (i - 1) * slices + (j % slices);
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    indices.push(0, ring(1, j), ring(1, j + 1), south, ring(stacks - 1, j + 1), ring(stacks - 1, j));
    for (let i = 1; i < stacks - 1; i++) {
      indices.push(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j), ring(i + 1, j + 1), ring(i, j + 1));
    }
  }
  return { positions, indices };
}

/** A plain UV sphere of radius 1. */
function lumpless(stacks: number, slices: number) {
  const positions = [0, 0, 1];
  for (let i = 1; i < stacks; i++) {
    const phi = (Math.PI * i) / stacks;
    for (let j = 0; j < slices; j++) {
      const a = (2 * Math.PI * j) / slices;
      positions.push(Math.sin(phi) * Math.cos(a), Math.sin(phi) * Math.sin(a), Math.cos(phi));
    }
  }
  positions.push(0, 0, -1);
  return { positions, indices: lump(stacks, slices).indices };
}

/** A small rotation about z, row-major. */
function smallTurn(angle: number): number[] {
  const [s, c] = [Math.sin(angle), Math.cos(angle)];
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}

function expectNear(got: readonly number[], want: readonly number[], tol: number) {
  expect(got.length).toBe(want.length);
  got.forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThan(tol));
}

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
    expect(s.placement(mesh)).toEqual({ rotation: IDENTITY, translation: [0, 0, 0], scale: [1, 1, 1] });

    s.setPlacement(mesh, { rotation: QUARTER_Z, translation: [10, 20, 30] });
    expect(s.placement(mesh)).toEqual({
      rotation: QUARTER_Z,
      translation: [10, 20, 30],
      scale: [1, 1, 1],
    });

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

  it("scaled_placement", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    // The default scale is rigid.
    expect(s.placement(mesh).scale).toEqual([1, 1, 1]);

    // Uniform and non-uniform scales round-trip; the matrix columns scale.
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [10, 20, 30], scale: [2, 2, 2] });
    expect(s.placement(mesh)).toEqual({
      rotation: IDENTITY,
      translation: [10, 20, 30],
      scale: [2, 2, 2],
    });
    // matrix4 = placement · translate(origin); the origin is (1, 1, 0.5),
    // so the last column is R·(S·o) + t = (2, 2, 1) + (10, 20, 30).
    expectClose(
      s.matrix4(mesh),
      [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 12, 22, 31, 1],
      1e-15,
    );
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [0, 0, 0], scale: [2, 1, 0.5] });
    expect(s.placement(mesh).scale).toEqual([2, 1, 0.5]);
    expectClose(
      s.matrix4(mesh),
      [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 2, 1, 0.25, 1],
      1e-15,
    );

    // Omitting the scale leaves the mesh rigid.
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [0, 0, 0] });
    expect(s.placement(mesh).scale).toEqual([1, 1, 1]);

    // Bad scales are rejected and leave the placement alone.
    const t: [number, number, number] = [0, 0, 0];
    expect(
      codeOf(() => s.setPlacement(mesh, { rotation: IDENTITY, translation: t, scale: [0, 1, 1] })),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(() => s.setPlacement(mesh, { rotation: IDENTITY, translation: t, scale: [1, -1, 1] })),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(() => s.setPlacement(mesh, { rotation: IDENTITY, translation: t, scale: [1, 1] as never })),
    ).toBe("INVALID_ARGUMENT");
    expect(s.placement(mesh).scale).toEqual([1, 1, 1]);
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
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    const kept = s.views(mesh).positions();
    const expected = Array.from(kept);
    const before = wasmMemory().buffer;
    const big = grid(300);
    for (let k = 0; k < 50 && wasmMemory().buffer === before; k++) {
      s.addMesh(big.positions, big.indices);
    }
    expect(wasmMemory().buffer).not.toBe(before);
    // The kept view is detached; a fresh one reads the same data. Without
    // debug there is no check, and views are plain typed arrays.
    expect(kept.byteLength).toBe(0);
    const fresh = s.views(mesh).positions();
    expect(fresh).toBeInstanceOf(Float32Array);
    expect(ArrayBuffer.isView(fresh)).toBe(true);
    expect(Array.from(fresh)).toEqual(expected);
    expect(warn).not.toHaveBeenCalled();
  });

  it("debug_warns_on_a_view_kept_across_a_reallocation", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s = scene({ debug: true });
    // Headroom first, so the growth below moves the buffer without growing
    // the memory: the kept view stays attached and reads freed memory.
    const big = grid(200);
    s.removeMesh(s.addMesh(big.positions, big.indices, { normals: false }));
    const mesh = s.addMesh([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2]);
    const kept = s.views(mesh).positions();
    expect(kept.length).toBe(9);
    expect(warn).not.toHaveBeenCalled();
    s.edit(mesh, [{ op: "addVertices", positions: new Float64Array(3 * 64) }]);
    void kept[0];
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("re-acquire");
    // Once per scene.
    void kept.length;
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("field_interaction_keeps_values_then_settles_to_the_exact_field", () => {
    const s = scene();
    const mesh = grid(40);
    const a = s.addMesh(mesh.positions, mesh.indices);
    const b = s.addMesh(mesh.positions, mesh.indices);
    s.setPlacement(b, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0.5, 0.5, 0.2] });
    const field = s.addDistanceField({ name: "d", on: a, to: b, maxDistance: Infinity });
    s.setColoring(a, { field, map: "turbo", range: [0, 2] });
    const before = Array.from(s.fieldValues(field));
    s.setFieldInteraction(64);
    s.setPlacement(b, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0.5, 0.5, 0.25] });
    const during = Array.from(s.fieldValues(field));
    // Most vertices keep their value; none lost it.
    const same = during.filter((x, k) => x === before[k]).length;
    expect(same).toBeGreaterThan(during.length / 2);
    expect(during.every((x) => Number.isFinite(x))).toBe(true);
    s.setFieldInteraction(null);
    const settled = Array.from(s.fieldValues(field));
    const exact = s.addDistanceField({ name: "exact", on: a, to: b, maxDistance: Infinity });
    expect(settled).toEqual(Array.from(s.fieldValues(exact)));
  });

  it("field_interaction_rejects_a_bad_budget", () => {
    const s = scene();
    for (const points of [0, -1, 1.5, Number.NaN, 2 ** 32]) {
      expect(codeOf(() => s.setFieldInteraction(points))).toBe("INVALID_ARGUMENT");
    }
  });

  it("debug_warns_on_a_kept_values_view", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s = scene({ debug: true });
    const a = s.addMesh(quad, quadTris);
    const b = s.addMesh([0, 0, 1, 1, 0, 1, 0, 1, 1], [0, 1, 2]);
    const field = s.addDistanceField({ name: "d", on: a, to: b, maxDistance: 2 });
    expect(s.views(a).values!()).toBeNull();
    s.setColoring(a, { field, map: "turbo", range: [0, 2] });
    const kept = s.views(a).values!()!;
    expect(kept.length).toBe(s.vertexSlots(a));
    expect(warn).not.toHaveBeenCalled();
    s.setPlacement(b, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0.5] });
    void kept[0];
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain("re-acquire");
  });

  it("coloring_returns_a_copy_of_the_rule", () => {
    const s = scene();
    const a = s.addMesh(quad, quadTris);
    const b = s.addMesh([0, 0, 1, 1, 0, 1, 0, 1, 1], [0, 1, 2]);
    const field = s.addDistanceField({ name: "d", on: a, to: b, maxDistance: 2 });
    const color: [number, number, number, number] = [255, 0, 0, 255];
    const rule: ColorRule & { range: [number, number] } = {
      field,
      map: {
        stops: [
          [0, color],
          [1, [0, 0, 255, 255]],
        ],
      },
      range: [0, 2],
    };
    s.setColoring(a, rule);
    const shown = s.coloring(a)!;
    expect(shown).not.toBe(rule);
    // Later changes to the caller's object, its stops included, do not leak in.
    color[0] = 1;
    rule.range[1] = 9;
    expect(shown.range).toEqual([0, 2]);
    expect((shown.map as { stops: readonly (readonly [number, readonly number[]])[] }).stops[0][1]).toEqual([255, 0, 0, 255]);
    expect(shown.field).toBe(field);
    // Frozen, deep (the field handle aside), and the same object on every read.
    const stops = (shown.map as { stops: readonly (readonly [number, readonly number[]])[] }).stops;
    for (const part of [shown, shown.range, shown.map, stops, stops[0], stops[0][1]]) {
      expect(Object.isFrozen(part)).toBe(true);
    }
    expect(s.coloring(a)).toBe(shown);
    s.removeField(field);
    expect(s.coloring(a)).toBeNull();
  });

  it("debug_warns_on_a_view_detached_by_another_scene", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s = scene({ debug: true });
    const mesh = s.addMesh(quad, quadTris);
    const kept = s.views(mesh).indices();
    const other = scene();
    const before = wasmMemory().buffer;
    const big = grid(300);
    for (let k = 0; k < 50 && wasmMemory().buffer === before; k++) {
      other.addMesh(big.positions, big.indices);
    }
    expect(warn).not.toHaveBeenCalled();
    expect(kept.length).toBe(0);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("debug_views_used_correctly_do_not_warn", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const s = scene({ debug: true });
    const mesh = s.addMesh(quad, quadTris);
    // Read from a fresh view and keep nothing (the garbage collector has not
    // run yet), then grow the memory and read again.
    const frame = () => s.views(mesh).positions()[0];
    const first = frame();
    const before = wasmMemory().buffer;
    const big = grid(300);
    for (let k = 0; k < 50 && wasmMemory().buffer === before; k++) {
      s.addMesh(big.positions, big.indices);
    }
    expect(wasmMemory().buffer).not.toBe(before);
    expect(frame()).toBe(first);
    // Views of one mesh survive each other and the read-only calls.
    const views = s.views(mesh);
    const positions = views.positions();
    const indices = views.indices();
    s.vertexSlots(mesh);
    s.matrix4(mesh);
    s.canUndo();
    expect(Array.from(indices)).toEqual(quadTris);
    expect(positions.subarray(0, 3)).toBeInstanceOf(Float32Array);
    expect(Array.from(positions.slice(3, 6))).toEqual(Array.from(s.views(mesh).positions().slice(3, 6)));
    expect(warn).not.toHaveBeenCalled();
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

  it("cut_by_plane_splits_and_caps", () => {
    const s = scene();
    // A closed unit cube, outward wound.
    const box = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1];
    const boxTris = [
      0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6,
      1, 6, 5, 3, 0, 4, 3, 4, 7,
    ];
    const mesh = s.addMesh(box, boxTris);
    s.takeChanges();
    const r = s.cutByPlane(mesh, { origin: [0.5, 0, 0], normal: [1, 0, 0] });
    expect(r.kept).toBe(mesh);
    expect(r.removed).not.toBe(mesh);
    // Both parts closed: the caps are there.
    expect(s.analyze(mesh).loops).toEqual([]);
    expect(s.analyze(r.removed).loops).toEqual([]);
    expect(r.keptTriangles).toBe(s.triangleCount(mesh));
    expect(r.removedTriangles).toBe(s.triangleCount(r.removed));
    expect(r.capTriangles).toBeGreaterThan(0);
    // The kept side is the + one: bounds cover the live triangles in
    // local coordinates, and the identity placement puts the origin at
    // matrix4[12..14] (dead slots keep stale positions, so the raw
    // buffer is not checked directly).
    const keptBounds = s.bounds(mesh)!;
    const keptOriginX = s.matrix4(mesh)[12];
    expect(keptBounds.min[0] + keptOriginX).toBeGreaterThanOrEqual(0.5 - 1e-6);
    // One undo step restores the cut mesh; the added part stays.
    expect(s.undo()).toBe(true);
    expect(s.triangleCount(mesh)).toBe(12);
    expect(s.triangleCount(r.removed)).toBe(r.removedTriangles);
    // A plane past the cube misses it.
    expect(codeOf(() => s.cutByPlane(mesh, { origin: [5, 0, 0], normal: [1, 0, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(s.triangleCount(mesh)).toBe(12);
  });

  it("register_recovers_a_known_placement_and_undoes_as_one_step", () => {
    const s = scene();
    const { positions, indices } = lump(32, 64);
    const fixed = s.addMesh(positions, indices);
    const moving = s.addMesh(positions, indices);
    const start = { rotation: smallTurn(0.08), translation: [0.05, -0.04, 0.03] as const, scale: [1, 1, 1] as const };
    s.setPlacement(moving, { ...start, translation: [...start.translation], scale: [...start.scale] });
    const r = s.register(moving, fixed);
    expect(r.status).toBe("converged");
    expect(r.converged).toBe(true);
    expect(r.applied).toBe(true);
    expect(r.rmse).toBeLessThan(1e-5);
    expect(r.iterations).toBeGreaterThan(0);
    expect(r.iterations).toBeLessThanOrEqual(50);
    expect(r.samples).toBeGreaterThan(0);
    expect(r.inliers).toBe(r.samples);
    expect(r.conditioning).toBeGreaterThan(1e-6);
    expect(r.notApplied).toBeNull();
    expect(r.undetermined).toEqual([]);
    expectNear(r.placement.rotation as number[], IDENTITY, 1e-5);
    expectNear(r.placement.translation, [0, 0, 0], 1e-5);
    expect(r.placement.scale).toEqual([1, 1, 1]);
    expect(s.placement(moving)).toEqual(r.placement);
    // One undo step back to the start, and redo returns the result.
    s.undo();
    expectNear(s.placement(moving).rotation as number[], start.rotation, 1e-12);
    expectNear(s.placement(moving).translation, start.translation, 1e-12);
    s.redo();
    expect(s.placement(moving)).toEqual(r.placement);
  });

  it("register_apply_false_leaves_the_scene_alone", () => {
    const s = scene();
    const { positions, indices } = lump(16, 32);
    const fixed = s.addMesh(positions, indices);
    const moving = s.addMesh(positions, indices);
    const start = { rotation: smallTurn(0.05), translation: [0.04, 0, 0] as [number, number, number] };
    s.setPlacement(moving, start);
    const before = s.placement(moving);
    const r = s.register(moving, fixed, { apply: false, maxIterations: 40, tolerance: 1e-7, maxSamples: 500 });
    expect(r.converged).toBe(true);
    expect(r.applied).toBe(false);
    expect(r.samples).toBe(500);
    expectNear(r.placement.translation, [0, 0, 0], 1e-4);
    expect(s.placement(moving)).toEqual(before);
  });

  it("register_reports_a_plane_sliding_in_its_plane_as_not_converged", () => {
    const s = scene();
    const sheet = (n: number) => {
      const positions: number[] = [];
      const indices: number[] = [];
      for (let y = 0; y <= n; y++) {
        for (let x = 0; x <= n; x++) {
          positions.push(x, y, 0);
        }
      }
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const a = y * (n + 1) + x;
          indices.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
        }
      }
      return { positions, indices };
    };
    const big = sheet(20);
    const small = sheet(6);
    const fixed = s.addMesh(big.positions, big.indices);
    const moving = s.addMesh(small.positions, small.indices);
    s.setPlacement(moving, { rotation: IDENTITY, translation: [4, 3, 0.5] });
    const before = s.placement(moving);
    const r = s.register(moving, fixed);
    expect(r.status).toBe("underconstrained");
    expect(r.converged).toBe(false);
    expect(r.applied).toBe(false);
    expect(s.placement(moving)).toEqual(before);
    // Every number finite: z is fixed and the slide left alone.
    expectNear(r.placement.translation, [4, 3, 0], 1e-5);
    expect(Number.isFinite(r.rmse)).toBe(true);
    expect(r.conditioning).toBeLessThan(1e-6);
    expect(r.undetermined.length).toBe(3);
  });

  it("register_keeps_the_moving_scale_and_measures_in_world_units", () => {
    const s = scene();
    const { positions, indices } = lump(24, 48);
    const fixed = s.addMesh(positions, indices);
    // The same shape stored at half size, placed with a scale of 2.
    const moving = s.addMesh(positions.map((v) => v / 2), indices);
    s.setPlacement(moving, { rotation: smallTurn(0.04), translation: [0.03, 0.02, -0.03], scale: [2, 2, 2] });
    const r = s.register(moving, fixed);
    expect(r.converged).toBe(true);
    expect(r.placement.scale).toEqual([2, 2, 2]);
    expectNear(r.placement.translation, [0, 0, 0], 1e-4);
    // maxIterations 0 only measures: a lift of 0.5 is 0.5 off a flat patch.
    const plane = s.addMesh([-9, -9, 0, 9, -9, 0, 9, 9, 0, -9, 9, 0], [0, 1, 2, 0, 2, 3]);
    const patch = s.addMesh([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 1, 2, 0, 2, 3]);
    s.setPlacement(patch, { rotation: IDENTITY, translation: [0, 0, 0.5] });
    const m = s.register(patch, plane, { maxIterations: 0 });
    expect(m.status).toBe("maxIterations");
    expect(m.iterations).toBe(0);
    expect(m.rmse).toBeCloseTo(0.5, 6);
  });

  it("register_with_bad_options_leaves_a_drag_running", () => {
    const s = scene();
    const { positions, indices } = lump(8, 16);
    const a = s.addMesh(positions, indices);
    const b = s.addMesh(positions, indices);
    const d = s.drag(a);
    d.move({ rotation: IDENTITY, translation: [0.5, 0, 0] });
    expect(codeOf(() => s.register(a, b, { overlap: 2 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register(a, a))).toBe("INVALID_ARGUMENT");
    // Still the same drag: a second one is refused until it ends.
    expect(codeOf(() => s.drag(a))).toBe("INVALID_ARGUMENT");
    d.end();
  });

  it("register_does_not_apply_a_wrong_minimum_or_an_undetermined_pose", () => {
    const s = scene();
    const { positions, indices } = lump(32, 64);
    const fixed = s.addMesh(positions, indices);
    const moving = s.addMesh(positions, indices);
    s.setPlacement(moving, { rotation: smallTurn(2.8), translation: [0, 0, 0] });
    const before = s.placement(moving);
    const r = s.register(moving, fixed, { tolerance: 1e-3 });
    expect(r.converged).toBe(true);
    expect(r.applied).toBe(false);
    expect(r.notApplied).toBe("rmseAboveLimit");
    expect(r.rmse).toBeGreaterThan(r.maxRmse);
    expect(s.placement(moving)).toEqual(before);
    expect(s.register(moving, fixed, { tolerance: 1e-3, maxRmse: Infinity }).applied).toBe(true);
    // A sphere: its spin is free, so nothing is applied and the motions are named.
    const ball = lumpless(32, 64);
    const f = s.addMesh(ball.positions, ball.indices);
    const m = s.addMesh(ball.positions, ball.indices);
    s.setPlacement(m, { rotation: smallTurn(0.06), translation: [0, 0, 0] });
    const u = s.register(m, f);
    expect(u.status).toBe("underconstrained");
    expect(u.converged).toBe(false);
    expect(u.applied).toBe(false);
    expect(u.notApplied).toBe("notConverged");
    expect(u.undetermined.length).toBe(3);
    expect(u.undetermined[0].rotation.length).toBe(3);
    expect(u.undetermined[0].about.length).toBe(3);
  });

  it("register_rejects_bad_input", () => {
    const s = scene();
    const { positions, indices } = lump(8, 16);
    const a = s.addMesh(positions, indices);
    const b = s.addMesh(positions, indices);
    expect(codeOf(() => s.register(a, a))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register(a, b, { overlap: 0 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register(a, b, { overlap: 1.5 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register(a, b, { tolerance: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register(a, b, { maxIterations: -1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register(a, b, { maxIterations: Number.NaN }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register(a, b, { maxSamples: 0 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register(a, b, { maxDistance: 0 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.register({} as MeshHandle, b))).toBe("INVALID_ARGUMENT");
    s.removeMesh(b);
    expect(codeOf(() => s.register(a, b))).toBe("UNKNOWN_MESH");
  });

  it("measure_reports_area_and_volume", () => {
    const s = scene();
    // A closed unit cube, outward wound.
    const box = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1];
    const boxTris = [
      0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6,
      1, 6, 5, 3, 0, 4, 3, 4, 7,
    ];
    const mesh = s.addMesh(box, boxTris);
    const got = s.measure(mesh);
    expect(got.area).toBeCloseTo(6, 9);
    expect(got.volume).toBeCloseTo(1, 9);
    expect(got.closed).toBe(true);
    expect(got.volumeReason).toBeNull();
    expect(got.boundaryLoops).toBe(0);
    expect(got.nonManifoldEdges).toBe(0);
    expect(got.orientationConflicts).toBe(0);
    // A uniform scale of 2: area x4, volume x8.
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [0, 0, 0], scale: [2, 2, 2] });
    const scaled = s.measure(mesh);
    expect(scaled.area).toBeCloseTo(24, 9);
    expect(scaled.volume).toBeCloseTo(8, 9);
    expect(scaled.closed).toBe(true);
    // An open quad: area without volume, with the reason.
    const open = s.addMesh(quad, quadTris);
    const m = s.measure(open);
    expect(m.area).toBeCloseTo(4 + Math.SQRT2, 9);
    expect(m.volume).toBeNull();
    expect(m.volumeReason).toBe("open");
    expect(m.closed).toBe(false);
    expect(m.boundaryLoops).toBeGreaterThan(0);
    expect(codeOf(() => s.measure({} as MeshHandle))).toBe("INVALID_ARGUMENT");
  });

  it("thickness_reads_wall_depth_and_colors", () => {
    const s = scene();
    // A closed unit cube, outward wound.
    const box = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1];
    const boxTris = [
      0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6,
      1, 6, 5, 3, 0, 4, 3, 4, 7,
    ];
    const mesh = s.addMesh(box, boxTris);
    const thick = s.addThicknessField({ name: "t", on: mesh, maxDistance: Number.POSITIVE_INFINITY });
    const values = Array.from(s.fieldValues(thick));
    expect(values).toHaveLength(8);
    for (const v of values) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(1 - 1e-3);
      expect(v).toBeLessThanOrEqual(2 * Math.sqrt(3) + 1e-3);
    }
    // Colours like any field: Infinity would follow `above`, NaN `nan`.
    s.setColoring(mesh, { field: thick, map: "turbo", range: [0, 5] });
    expect(s.coloring(mesh)?.field).toBe(thick);
    expect(s.fieldTargets(thick)).toEqual([]);
    expect(s.fieldTargetStatus(thick)).toEqual([]);
    // A uniform scale of 2 doubles every thickness.
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [0, 0, 0], scale: [2, 2, 2] });
    const scaled = Array.from(s.fieldValues(thick));
    for (let k = 0; k < values.length; k++) {
      expect(scaled[k]).toBeCloseTo(2 * values[k], 4);
    }
    // An open sheet reads NaN everywhere (open boundary → null thickness).
    const open = s.addMesh(quad.slice(0, 12), [0, 1, 2, 0, 2, 3]);
    const thin = s.addThicknessField({ name: "t", on: open, maxDistance: Number.POSITIVE_INFINITY });
    expect(Array.from(s.fieldValues(thin))).toEqual([NaN, NaN, NaN, NaN]);
    s.setColoring(open, { field: thin, map: "turbo", range: [0, 5] });
    const colors = s.views(open).colors!()!;
    expect(colors.length).toBe(4 * 4);
    for (let k = 0; k < colors.length; k += 4) {
      expect(Array.from(colors.slice(k, k + 4))).toEqual([128, 128, 128, 255]);
    }
    // maxDistance caps: a bound below every thickness reads Infinity.
    s.removeField(thick);
    const capped = s.addThicknessField({ name: "c", on: mesh, maxDistance: 0.1 });
    expect(Array.from(s.fieldValues(capped)).every((v) => v === Number.POSITIVE_INFINITY)).toBe(true);
    expect(codeOf(() => s.addThicknessField({ name: "c", on: mesh, maxDistance: 1 }))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.addThicknessField({ name: "n", on: mesh, maxDistance: -1 }))).toBe("INVALID_ARGUMENT");
  });

  it("curvature_reads_signed_mean_curvature_and_colors", () => {
    const s = scene();
    // A closed unit cube, outward wound: sharp convex corners read positive.
    const box = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1];
    const boxTris = [
      0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6,
      1, 6, 5, 3, 0, 4, 3, 4, 7,
    ];
    const mesh = s.addMesh(box, boxTris);
    const curv = s.addCurvatureField({ name: "k", on: mesh });
    const values = Array.from(s.fieldValues(curv));
    expect(values).toHaveLength(8);
    for (const v of values) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThan(0);
    }
    // Colours like any field: NaN would take the rule's `nan` colour.
    s.setColoring(mesh, { field: curv, map: "turbo", range: [-1, 2] });
    expect(s.coloring(mesh)?.field).toBe(curv);
    expect(s.fieldTargets(curv)).toEqual([]);
    expect(s.fieldTargetStatus(curv)).toEqual([]);
    // A uniform scale of 2 halves every curvature (1/world-unit).
    s.setPlacement(mesh, { rotation: IDENTITY, translation: [0, 0, 0], scale: [2, 2, 2] });
    const scaled = Array.from(s.fieldValues(curv));
    for (let k = 0; k < values.length; k++) {
      expect(scaled[k]).toBeCloseTo(values[k] / 2, 4);
    }
    // A flat open quad reads zero everywhere (curvature is defined on
    // open sheets, unlike thickness).
    const open = s.addMesh(quad.slice(0, 12), [0, 1, 2, 0, 2, 3]);
    const flat = s.addCurvatureField({ name: "k", on: open });
    for (const v of Array.from(s.fieldValues(flat))) {
      expect(Math.abs(v)).toBeLessThan(1e-6);
    }
    // A saddle reads the sign of its dominant bend: negative at the middle.
    const grid: number[] = [];
    const n = 8;
    for (let y = 0; y <= n; y++) {
      for (let x = 0; x <= n; x++) {
        const fx = (x / n) * 2 - 1;
        const fy = (y / n) * 2 - 1;
        grid.push(fx, fy, (2 * fx * fx - fy * fy) / 4);
      }
    }
    const gridTris: number[] = [];
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const a = y * (n + 1) + x;
        gridTris.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
      }
    }
    const saddle = s.addMesh(grid, gridTris);
    const bend = s.addCurvatureField({ name: "k", on: saddle });
    const bent = Array.from(s.fieldValues(bend));
    expect(bent.every((v) => Number.isFinite(v))).toBe(true);
    expect(bent[(4 * (n + 1) + 4) as number]).toBeLessThan(-0.1);
    expect(codeOf(() => s.addCurvatureField({ name: "k", on: saddle }))).toBe("INVALID_ARGUMENT");
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
    expect(drag.active).toBe(true);
    for (let k = 1; k <= 30; k++) {
      drag.move({ rotation: IDENTITY, translation: [k, 0, 0] });
    }
    drag.end();
    drag.end(); // no-op
    expect(drag.active).toBe(false);
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
    expect(c.active).toBe(false);
    expect(s.placement(mesh)).toEqual({ rotation: IDENTITY, translation: [30, 0, 0], scale: [1, 1, 1] });
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
    expect(drag.active).toBe(false);
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

  it("remove_mesh_mid_drag_ends_it", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    const other = s.addMesh(quad, quadTris);
    s.clearHistory();
    const drag = s.drag(mesh);
    drag.move({ rotation: IDENTITY, translation: [2, 0, 0] });
    // Removing another mesh leaves the drag alone.
    s.removeMesh(other);
    expect(drag.active).toBe(true);
    s.removeMesh(mesh);
    expect(drag.active).toBe(false);
    // The mesh's history, the drag's step included, goes with it.
    expect(s.canUndo()).toBe(false);
    expect(codeOf(() => drag.move({ rotation: IDENTITY, translation: [3, 0, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    drag.end();
    expect(codeOf(() => s.endStep())).toBe("STEP_NOT_OPEN");
    // No step is left open, and a new drag can start.
    const third = s.addMesh(quad, quadTris);
    s.drag(third).end();
  });

  it("step_calls_during_drag_abandon_it", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);

    // cancelStep takes the drag's step away: the drag is over.
    const d = s.drag(mesh);
    d.move({ rotation: IDENTITY, translation: [1, 0, 0] });
    s.cancelStep();
    expect(d.active).toBe(false);
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

  it("mode_getter_follows_set_mode", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    expect(s.mode(mesh)).toBe("eager");
    s.setMode(mesh, "lazy");
    expect(s.mode(mesh)).toBe("lazy");
    s.setMode(mesh, "manual");
    expect(s.mode(mesh)).toBe("manual");
    s.removeMesh(mesh);
    expect(codeOf(() => s.mode(mesh))).toBe("UNKNOWN_MESH");
  });

  it("bounds_cover_live_triangles_only", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    // Local coordinates (origin = box centre (1, 1, 0.5)).
    const b = s.bounds(mesh)!;
    expect(b.min).toEqual([-1, -1, -0.5]);
    expect(b.max).toEqual([1, 1, 0.5]);
    expect(b.center).toEqual([0, 0, 0]);
    expect(b.size).toEqual([2, 2, 1]);
    // Removing every triangle leaves no bounds; the slots still hold positions.
    s.edit(mesh, [{ op: "removeTriangles", triangles: [0, 1, 2] }]);
    expect(s.bounds(mesh)).toBeNull();
    s.removeMesh(mesh);
    expect(codeOf(() => s.bounds(mesh))).toBe("UNKNOWN_MESH");
  });

  it("coalesced_drag_applies_once_per_frame", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    s.commitFrame();
    const drag = s.drag(mesh, { coalesce: true });
    // Pointer-rate moves: nothing reaches the engine until the frame.
    drag.move({ rotation: IDENTITY, translation: [1, 0, 0] });
    drag.move({ rotation: IDENTITY, translation: [2, 0, 0] });
    drag.move({ rotation: IDENTITY, translation: [3, 0, 0] });
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    const changes = s.commitFrame();
    // One frame, one placement: the latest pose.
    expect(s.placement(mesh).translation).toEqual([3, 0, 0]);
    expect(changes.filter((c) => c.mesh === mesh && c.placementChanged).length).toBe(1);
    // end() applies a still-pending move, still one undo step in all.
    drag.move({ rotation: IDENTITY, translation: [4, 0, 0] });
    drag.end();
    expect(s.placement(mesh).translation).toEqual([4, 0, 0]);
    expect(s.canUndo()).toBe(true);
    s.undo();
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    expect(s.canUndo()).toBe(false);
    // A bad pose still throws at move(), not at the frame.
    const d2 = s.drag(mesh, { coalesce: true });
    expect(codeOf(() => d2.move({ rotation: [1, 0, 0], translation: [0, 0, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(d2.active).toBe(true);
    d2.cancel();
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    // Cancelling drops a pending move and records nothing.
    const d3 = s.drag(mesh, { coalesce: true });
    d3.move({ rotation: IDENTITY, translation: [9, 0, 0] });
    d3.cancel();
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    expect(s.canUndo()).toBe(false);
  });

  it("coalesced_drag_manual_end_step_keeps_pending_pose", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    s.commitFrame();
    // A coalesced move with no frame yet: the caller's endStep lands it in
    // the closing step, like a non-coalesced move, instead of dropping it.
    const drag = s.drag(mesh, { coalesce: true });
    drag.move({ rotation: IDENTITY, translation: [3, 0, 0] });
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    s.endStep();
    expect(drag.active).toBe(false);
    expect(s.placement(mesh).translation).toEqual([3, 0, 0]);
    expect(s.canUndo()).toBe(true);
    s.undo();
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    expect(s.canUndo()).toBe(false);
    // Closing a step opened inside the drag keeps the pose pending instead.
    const inner = s.drag(mesh, { coalesce: true });
    inner.move({ rotation: IDENTITY, translation: [5, 0, 0] });
    s.beginStep();
    s.endStep();
    expect(inner.active).toBe(true);
    expect(s.placement(mesh).translation).toEqual([0, 0, 0]);
    s.commitFrame();
    expect(s.placement(mesh).translation).toEqual([5, 0, 0]);
    inner.end();
    expect(s.canUndo()).toBe(true);
  });

  it("coalesced_drag_move_validates_pose_values", () => {
    const s = scene();
    const mesh = s.addMesh(quad, quadTris);
    s.commitFrame();
    const drag = s.drag(mesh, { coalesce: true });
    // Non-finite values throw at move(), not at the frame...
    expect(codeOf(() => drag.move({ rotation: IDENTITY, translation: [NaN, 0, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(
      codeOf(() => drag.move({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, Infinity], translation: [0, 0, 0] })),
    ).toBe("INVALID_ARGUMENT");
    // ...as does a zero quaternion...
    expect(codeOf(() => drag.move({ rotation: [0, 0, 0, 0], translation: [0, 0, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(drag.active).toBe(true);
    // ...while a non-rotation matrix still throws when applied.
    drag.move({ rotation: [2, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] });
    expect(codeOf(() => s.commitFrame())).toBe("INVALID_ARGUMENT");
    drag.cancel();
    expect(s.canUndo()).toBe(false);
  });
});
