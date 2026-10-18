// The three.js adapter's bindMesh in Node, with three and no renderer:
// attributes, update ranges, matrices, colours, opacity and disposal.
import { readdirSync, readFileSync } from "node:fs";
import * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";
import { type MeshHandle, Scene, type Vec3 } from "../src/node.js";
import { bindMesh, type BoundMesh } from "../src/three/index.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const QUARTER_Z = [0, -1, 0, 1, 0, 0, 0, 0, 1];

/** An n×n vertex grid on the XY plane, offset by `at`. */
function grid(n: number, at: Vec3 = [0, 0, 0]): { positions: Float64Array; indices: Uint32Array } {
  const positions = new Float64Array(n * n * 3);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      positions.set([x + at[0], y + at[1], at[2]], (y * n + x) * 3);
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

// Counts each scene's registered onChanges listeners, to catch leaks.
const liveListeners = new WeakMap<Scene, Set<object>>();
const originalOnChanges = Scene.prototype.onChanges;
Scene.prototype.onChanges = function (this: Scene, listener) {
  const set = liveListeners.get(this) ?? new Set();
  liveListeners.set(this, set);
  const token = {};
  set.add(token);
  const off = originalOnChanges.call(this, listener);
  return () => {
    set.delete(token);
    off();
  };
};

let scenes: Scene[] = [];
afterEach(() => {
  for (const s of scenes) {
    s.free();
  }
  scenes = [];
  vi.restoreAllMocks();
});

function scene(debug = false): Scene {
  const s = new Scene({ debug });
  scenes.push(s);
  return s;
}

/** A scene with two grids, one bound, and the first frame committed. */
function setup(debug = false) {
  const s = scene(debug);
  const g = grid(6);
  const m = s.addMesh(g.positions, g.indices);
  const other = grid(6, [0, 0, 1.5]);
  const n = s.addMesh(other.positions, other.indices);
  const bound = bindMesh(s, m);
  s.commitFrame();
  clearRanges(bound);
  return { s, m, n, bound };
}

function attr(b: BoundMesh, name: string): THREE.BufferAttribute {
  return b.object.geometry.getAttribute(name) as THREE.BufferAttribute;
}

function index(b: BoundMesh): THREE.BufferAttribute {
  return b.object.geometry.getIndex()!;
}

/** What a renderer does after uploading. */
function clearRanges(b: BoundMesh) {
  const g = b.object.geometry;
  for (const a of [...Object.values(g.attributes), g.getIndex()]) {
    (a as THREE.BufferAttribute | null)?.clearUpdateRanges();
  }
}

/** Every attribute equals the scene's views. */
function expectMirrorsMatch(s: Scene, m: MeshHandle, b: BoundMesh) {
  const v = s.views(m);
  expect(Array.from(attr(b, "position").array)).toEqual(Array.from(v.positions()));
  const normals = v.normals();
  if (normals === null) {
    expect(attr(b, "normal")).toBeUndefined();
  } else {
    expect(Array.from(attr(b, "normal").array)).toEqual(Array.from(normals));
  }
  const colors = v.colors();
  if (colors === null) {
    expect(attr(b, "color")).toBeUndefined();
  } else {
    expect(Array.from(attr(b, "color").array)).toEqual(Array.from(colors));
  }
  expect(Array.from(index(b).array)).toEqual(Array.from(v.indices()));
}

function expectMatrix(s: Scene, m: MeshHandle, b: BoundMesh) {
  expect(b.object.matrix.elements).toEqual(Array.from(s.matrix4(m)));
}

describe("bindMesh", () => {
  it("bind_creates_attributes_matching_views", () => {
    const s = scene();
    const g = grid(4);
    const m = s.addMesh(g.positions, g.indices);
    s.setPlacement(m, { rotation: QUARTER_Z, translation: [1, 2, 3] });
    const b = bindMesh(s, m, { color: 0xff0000 });
    expect(b.mesh).toBe(m);
    expect(b.object).toBeInstanceOf(THREE.Mesh);
    expect(attr(b, "position")).toBeInstanceOf(THREE.BufferAttribute);
    expect(attr(b, "position").array).toBeInstanceOf(Float32Array);
    expect(attr(b, "position").itemSize).toBe(3);
    expect(attr(b, "normal").array).toBeInstanceOf(Float32Array);
    expect(attr(b, "color")).toBeUndefined();
    expect(index(b).array).toBeInstanceOf(Uint32Array);
    expect(index(b).count).toBe(3 * s.triangleSlots(m));
    expectMirrorsMatch(s, m, b);
    // Mirrors are JS-owned, not views into wasm memory.
    expect(attr(b, "position").array.buffer).not.toBe(s.views(m).positions().buffer);
    expect(b.object.matrixAutoUpdate).toBe(false);
    expectMatrix(s, m, b);
    const material = b.object.material as THREE.MeshStandardMaterial;
    expect(material).toBeInstanceOf(THREE.MeshStandardMaterial);
    expect(material.vertexColors).toBe(false);
    expect(material.color.getHex()).toBe(0xff0000);
    // Bounds of the live geometry, in local coordinates (origin = box centre).
    const sphere = b.object.geometry.boundingSphere!;
    expect(sphere.center.toArray()).toEqual([0, 0, 0]);
    expect(sphere.radius).toBeCloseTo(Math.hypot(1.5, 1.5), 5);

    // A mesh without normals has no normal attribute.
    const n = s.addMesh(g.positions, g.indices, { normals: false });
    const bn = bindMesh(s, n);
    expect(attr(bn, "normal")).toBeUndefined();
    expectMirrorsMatch(s, n, bn);
  });

  it("ranges_only_update_dirty_parts", () => {
    const { s, m, bound } = setup();
    const position = attr(bound, "position");
    const normal = attr(bound, "normal");
    const version = position.version;
    // Vertices 3 and 5 are within the merge gap; 30 is not.
    s.edit(m, [
      { op: "setPositions", vertices: [3, 5, 30], positions: [3, 0, 0.5, 5, 0, 0.5, 0, 5, 0.5] },
    ]);
    const changes = s.commitFrame();
    const change = changes.find((c) => c.mesh === m)!;
    expect(change.positions).not.toBe("all");
    const expected = (change.positions as [number, number][]).map(([start, count]) => ({
      start: start * 3,
      count: count * 3,
    }));
    expect(expected.length).toBeGreaterThan(0);
    expect(position.updateRanges).toEqual(expected);
    // Every range covers an edited vertex and only the gap around it.
    const covered = expected.reduce((n, r) => n + r.count / 3, 0);
    expect(covered).toBeLessThan(s.vertexSlots(m));
    for (const v of [3, 5, 30]) {
      expect(expected.some((r) => 3 * v >= r.start && 3 * v < r.start + r.count)).toBe(true);
    }
    expect(position.version).toBeGreaterThan(version);
    // The same attribute objects: nothing was recreated.
    expect(attr(bound, "position")).toBe(position);
    expect(normal.updateRanges.length).toBeGreaterThan(0);
    expect(index(bound).updateRanges).toEqual([]);
    expectMirrorsMatch(s, m, bound);
    // The bounds follow the edit.
    expect(bound.object.geometry.boundingBox!.max.z).toBeCloseTo(0.5, 5);
  });

  it("placement_updates_matrix_only", () => {
    const { s, m, bound } = setup();
    const position = attr(bound, "position");
    const version = position.version;
    const radius = bound.object.geometry.boundingSphere!.radius;
    // The bounds walk expands a box by every live vertex.
    const boundsWalk = vi.spyOn(THREE.Box3.prototype, "expandByPoint");
    const drag = s.drag(m);
    drag.move({ rotation: QUARTER_Z, translation: [4, 5, 6] });
    s.commitFrame();
    // A render's matrix update clears the flag; the next frame sets it again.
    bound.object.updateMatrixWorld();
    expect(bound.object.matrixWorldNeedsUpdate).toBe(false);
    drag.move({ rotation: IDENTITY, translation: [7, 8, 9] });
    s.commitFrame();
    drag.end();
    expectMatrix(s, m, bound);
    expect(bound.object.matrixWorldNeedsUpdate).toBe(true);
    expect(position.version).toBe(version);
    expect(position.updateRanges).toEqual([]);
    expect(attr(bound, "normal").updateRanges).toEqual([]);
    expect(index(bound).updateRanges).toEqual([]);
    expect(boundsWalk).not.toHaveBeenCalled();
    expect(bound.object.geometry.boundingSphere!.radius).toBe(radius);
    // The world matrix follows on the next render's matrix update.
    bound.object.updateMatrixWorld();
    expect(bound.object.matrixWorld.elements).toEqual(Array.from(s.matrix4(m)));
    // Undo moves it back through the change stream.
    s.undo();
    s.commitFrame();
    expectMatrix(s, m, bound);
  });

  it("capacity_growth_recreates_attributes", () => {
    const { s, m, bound } = setup();
    const geometry = bound.object.geometry;
    const oldPosition = attr(bound, "position");
    const oldIndex = index(bound);
    const oldLength = oldPosition.array.length;
    const disposed = vi.fn();
    geometry.addEventListener("dispose", disposed);
    // A far vertex and a triangle using it.
    s.edit(m, [
      { op: "addVertices", positions: [20, 0, 0] },
      { op: "addTriangles", triangles: [0, 1, { ref: [0, 0] }] },
    ]);
    const change = s.commitFrame().find((c) => c.mesh === m)!;
    expect(change.capacityChanged).toBe(true);
    expect(attr(bound, "position")).not.toBe(oldPosition);
    expect(index(bound)).not.toBe(oldIndex);
    expect(attr(bound, "position").array.length).toBe(3 * s.vertexSlots(m));
    expect(attr(bound, "position").array.length).toBeGreaterThan(oldLength);
    expect(attr(bound, "normal").array.length).toBe(3 * s.vertexSlots(m));
    expect(index(bound).count).toBe(3 * s.triangleSlots(m));
    expect(geometry.drawRange.count).toBe(Number.POSITIVE_INFINITY);
    expect(disposed).toHaveBeenCalledTimes(1);
    expectMirrorsMatch(s, m, bound);
    // The new triangle widened the bounds.
    expect(geometry.boundingBox!.max.x).toBeGreaterThan(10);

    // Undo leaves the slots (dead vertex with a stale far position) but drops
    // the triangle: the bounds use live triangles only.
    s.undo();
    s.commitFrame();
    expectMirrorsMatch(s, m, bound);
    expect(attr(bound, "position").array.length).toBe(3 * s.vertexSlots(m));
    expect(geometry.boundingBox!.max.x).toBeCloseTo(2.5, 5);
    expect(geometry.boundingSphere!.radius).toBeCloseTo(Math.hypot(2.5, 2.5), 5);

    // Compaction shrinks the slots: recreated again at the smaller length.
    s.compact(m);
    s.commitFrame();
    expect(attr(bound, "position").array.length).toBe(3 * s.vertexSlots(m));
    expect(index(bound).count).toBe(3 * s.triangleSlots(m));
    expectMirrorsMatch(s, m, bound);
  });

  it("coloring_toggle_adds_and_removes_attribute", () => {
    const { s, m, n, bound } = setup();
    bound.setOpacity(0.4);
    const material = bound.object.material as THREE.MeshStandardMaterial;
    const baseHex = material.color.getHex();
    const field = s.addDistanceField({ name: "gap", on: m, to: n, maxDistance: 3 });
    s.commitFrame();
    const canUndo = s.canUndo();
    const canRedo = s.canRedo();
    const position = attr(bound, "position");
    for (let round = 0; round < 3; round++) {
      s.setColoring(m, { field, map: "turbo", range: [0, 3] });
      s.commitFrame();
      const color = attr(bound, "color");
      expect(color).toBeInstanceOf(THREE.BufferAttribute);
      expect(color.array).toBeInstanceOf(Uint8Array);
      expect(color.itemSize).toBe(4);
      expect(color.normalized).toBe(true);
      expect(material.vertexColors).toBe(true);
      expect(material.color.getHex()).toBe(0xffffff);
      expectMirrorsMatch(s, m, bound);

      s.setColoring(m, null);
      s.commitFrame();
      expect(attr(bound, "color")).toBeUndefined();
      expect(material.vertexColors).toBe(false);
      expect(material.color.getHex()).toBe(baseHex);
      expect(material.opacity).toBe(0.4);
      expect(material.transparent).toBe(true);
      expect([s.canUndo(), s.canRedo()]).toEqual([canUndo, canRedo]);
    }
    // Positions keep their attribute through the toggles.
    expect(attr(bound, "position")).toBe(position);

    // While coloured, a drag of the other mesh updates colour ranges only.
    s.setColoring(m, { field, map: "turbo", range: [0, 3] });
    s.commitFrame();
    const color = attr(bound, "color");
    color.clearUpdateRanges();
    s.setPlacement(n, { rotation: IDENTITY, translation: [0, 0, -0.5] });
    s.commitFrame();
    expect(attr(bound, "color")).toBe(color);
    expectMirrorsMatch(s, m, bound);
  });

  it("opacity_sets_transparent_and_depthwrite", () => {
    const { s, bound } = setup();
    const material = bound.object.material as THREE.MeshStandardMaterial;
    expect(material.transparent).toBe(false);
    expect(material.depthWrite).toBe(true);
    const canUndo = s.canUndo();
    const version = material.version;
    bound.setOpacity(0.5);
    expect(material.opacity).toBe(0.5);
    expect(material.transparent).toBe(true);
    expect(material.depthWrite).toBe(false);
    expect(material.version).toBeGreaterThan(version);
    bound.setOpacity(1);
    expect(material.opacity).toBe(1);
    expect(material.transparent).toBe(false);
    expect(material.depthWrite).toBe(true);
    expect(s.canUndo()).toBe(canUndo);
    expect(() => bound.setOpacity(1.5)).toThrow(/opacity/);
    expect(() => bound.setOpacity(Number.NaN)).toThrow(/opacity/);

    // Options: initial opacity, and a caller's material is used as given.
    const phong = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
    const g = grid(3);
    const other = s.addMesh(g.positions, g.indices);
    const b2 = bindMesh(s, other, { material: phong, opacity: 0.25 });
    expect(b2.object.material).toBe(phong);
    expect(phong.color.getHex()).toBe(0x00ff00);
    expect([phong.opacity, phong.transparent, phong.depthWrite]).toEqual([0.25, true, false]);
  });

  it("dispose_and_remove_any_order", () => {
    const s = scene();
    const g = grid(4);
    const listeners = () => liveListeners.get(s)?.size ?? 0;
    const base = listeners();

    // Dispose, then remove.
    const a = s.addMesh(g.positions, g.indices);
    const ba = bindMesh(s, a);
    expect(listeners()).toBe(base + 1);
    const geometryDisposed = vi.fn();
    const materialDisposed = vi.fn();
    ba.object.geometry.addEventListener("dispose", geometryDisposed);
    (ba.object.material as THREE.Material).addEventListener("dispose", materialDisposed);
    s.commitFrame();
    ba.dispose();
    expect(listeners()).toBe(base);
    expect(geometryDisposed).toHaveBeenCalledTimes(1);
    expect(materialDisposed).toHaveBeenCalledTimes(1);
    s.removeMesh(a);
    expect(() => s.commitFrame()).not.toThrow();
    ba.dispose();
    expect(geometryDisposed).toHaveBeenCalledTimes(1);

    // Remove, then dispose: the binding detaches on the removal record.
    const b = s.addMesh(g.positions, g.indices);
    const phong = new THREE.MeshPhongMaterial();
    const bb = bindMesh(s, b, { material: phong });
    s.commitFrame();
    const matrix = bb.object.matrix.clone();
    const position = attr(bb, "position");
    s.removeMesh(b);
    expect(() => s.commitFrame()).not.toThrow();
    expect(listeners()).toBe(base);
    // Left in place, as it was.
    expect(bb.object.matrix.equals(matrix)).toBe(true);
    expect(attr(bb, "position")).toBe(position);
    const disposed = vi.fn();
    const phongDisposed = vi.fn();
    bb.object.geometry.addEventListener("dispose", disposed);
    phong.addEventListener("dispose", phongDisposed);
    bb.dispose();
    expect(disposed).toHaveBeenCalledTimes(1);
    // A caller's material is the caller's to dispose.
    expect(phongDisposed).not.toHaveBeenCalled();
    expect(() => s.commitFrame()).not.toThrow();

    // Remove and dispose before any frame.
    const c = s.addMesh(g.positions, g.indices);
    const bc = bindMesh(s, c);
    s.removeMesh(c);
    bc.dispose();
    expect(listeners()).toBe(base);
    expect(() => s.commitFrame()).not.toThrow();
  });

  it("works_with_debug_scene", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { s, m, n, bound } = setup(true);
    expectMirrorsMatch(s, m, bound);
    s.edit(m, [{ op: "setPositions", vertices: [2], positions: [2, 0, 0.25] }]);
    s.commitFrame();
    expectMirrorsMatch(s, m, bound);
    s.edit(m, [
      { op: "addVertices", positions: [9, 0, 0] },
      { op: "addTriangles", triangles: [0, 1, { ref: [0, 0] }] },
    ]);
    s.commitFrame();
    expectMirrorsMatch(s, m, bound);
    const field = s.addDistanceField({ name: "gap", on: m, to: n, maxDistance: 3 });
    s.setColoring(m, { field, map: "viridis", range: [0, 3] });
    s.commitFrame();
    expectMirrorsMatch(s, m, bound);
    s.setPlacement(m, { rotation: QUARTER_Z, translation: [1, 1, 1] });
    s.commitFrame();
    expectMatrix(s, m, bound);
    expect(warn).not.toHaveBeenCalled();
  });
});

describe("packaging", () => {
  it("core_entry_does_not_import_three", () => {
    const dist = new URL("../dist/", import.meta.url);
    const imports = /from\s*["']three["']|import\s*["']three["']/;
    // The core entries and every chunk but the three entry's.
    const core = readdirSync(dist).filter((f) => f.endsWith(".js") && f !== "three.js");
    expect(core).toEqual(expect.arrayContaining(["web.js", "node.js"]));
    for (const file of core) {
      expect(readFileSync(new URL(file, dist), "utf8")).not.toMatch(imports);
    }
    // The three entry imports `three` and the shared core, and does not
    // inline the wasm glue.
    const three = readFileSync(new URL("three.js", dist), "utf8");
    expect(three).toMatch(imports);
    expect(three).not.toMatch(/__wbindgen|__wbg_/);
  });
});
