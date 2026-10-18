// Queries, distance fields with colouring, and sections through the Node
// entry's sources.
import { afterEach, describe, expect, it } from "vitest";
import { FastMeshError, type MeshChange, type MeshHandle, Scene, type Vec3 } from "../src/node.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function codeOf(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    return (e as FastMeshError).code;
  }
  throw new Error("expected a throw");
}

/**
 * A closed UV sphere with an even number of stacks, so its equator vertices
 * lie exactly on `z = center.z`, and a vertex at longitude 0 (`+x`) and π (`-x`).
 */
function sphere(
  center: Vec3,
  radius: number,
  stacks = 16,
  slices = 32,
): { positions: Float64Array; indices: Uint32Array } {
  const positions: number[] = [center[0], center[1], center[2] + radius];
  for (let i = 1; i < stacks; i++) {
    const phi = (Math.PI * i) / stacks;
    for (let j = 0; j < slices; j++) {
      const theta = (2 * Math.PI * j) / slices;
      positions.push(
        center[0] + radius * Math.sin(phi) * Math.cos(theta),
        center[1] + radius * Math.sin(phi) * Math.sin(theta),
        center[2] + radius * Math.cos(phi),
      );
    }
  }
  positions.push(center[0], center[1], center[2] - radius);
  const south = positions.length / 3 - 1;
  const ring = (i: number, j: number) => 1 + (i - 1) * slices + (j % slices);
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    indices.push(0, ring(1, j), ring(1, j + 1));
    indices.push(south, ring(stacks - 1, j + 1), ring(stacks - 1, j));
  }
  for (let i = 1; i < stacks - 1; i++) {
    for (let j = 0; j < slices; j++) {
      const a = ring(i, j);
      const b = ring(i, j + 1);
      const c = ring(i + 1, j + 1);
      const d = ring(i + 1, j);
      indices.push(a, d, c, a, c, b);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function addSphere(s: Scene, center: Vec3, radius = 1, stacks?: number, slices?: number) {
  const { positions, indices } = sphere(center, radius, stacks, slices);
  return s.addMesh(positions, indices);
}

function changeOf(changes: MeshChange[], mesh: MeshHandle): MeshChange | undefined {
  return changes.find((c) => c.mesh === mesh);
}

function rangeTotal(r: MeshChange["colors"]): number {
  return r === "all" ? Number.POSITIVE_INFINITY : r.reduce((n, [, count]) => n + count, 0);
}

const scenes: Scene[] = [];
function scene(): Scene {
  const s = new Scene();
  scenes.push(s);
  return s;
}
afterEach(() => {
  for (const s of scenes.splice(0)) {
    s.free();
  }
});

describe("queries", () => {
  it("raycast_picks_nearest_mesh", () => {
    const s = scene();
    const a = addSphere(s, [0, 0, 0]);
    const b = addSphere(s, [3, 0, 0]);
    // From the left: A is in front of B.
    const hit = s.raycast([-5, 0, 0.01], [1, 0, 0]);
    expect(hit).not.toBeNull();
    expect(hit!.mesh).toBe(a);
    expect(hit!.distance).toBeGreaterThan(3.9);
    // Faceted: the hit lies slightly inside the unit sphere.
    expect(hit!.distance).toBeLessThan(4.01);
    expect(hit!.point[0]).toBeCloseTo(-5 + hit!.distance, 6);
    expect(hit!.barycentric.reduce((x, y) => x + y)).toBeCloseTo(1, 5);
    expect(Number.isInteger(hit!.triangle)).toBe(true);
    // From the right: B; a placement moves it in the world.
    expect(s.raycast([10, 0, 0.01], [-1, 0, 0])!.mesh).toBe(b);
    s.setPlacement(b, { rotation: IDENTITY, translation: [0, 10, 0] });
    expect(s.raycast([10, 0, 0.01], [-1, 0, 0])!.mesh).toBe(a);
    // Bounds and misses.
    expect(s.raycast([-5, 0, 0.01], [1, 0, 0], { maxDistance: 3 })).toBeNull();
    expect(s.raycast([-5, 5, 0], [1, 0, 0])).toBeNull();
    expect(codeOf(() => s.raycast([0, 0, 0], [0, 0, 0]))).toBe("INVALID_ARGUMENT");
  });

  it("rotated_placement_off_centre_origin", () => {
    const s = scene();
    // A unit sphere around (3, 0, 0) stored about origin (2, 1, 0), not its centre.
    const { positions, indices } = sphere([3, 0, 0], 1);
    const m = s.addMesh(positions, indices, { origin: [2, 1, 0] });
    // A quarter turn about z then up 1: the centre goes to (0, 3, 1).
    s.setPlacement(m, { rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1], translation: [0, 0, 1] });
    const hit = s.raycast([0.01, 10, 1], [0, -1, 0])!;
    expect(hit.mesh).toBe(m);
    expect(hit.distance).toBeGreaterThan(5.99);
    expect(hit.distance).toBeLessThan(6.01);
    expect(hit.point[1]).toBeCloseTo(10 - hit.distance, 9);
    // Nothing is left where the unrotated sphere would be.
    expect(s.raycast([3, -10, 1], [0, 1, 0])).toBeNull();
    const c = s.closestPoint([0, 3, 3])!;
    expect(c.distance).toBeCloseTo(1, 2);
    expect(c.point[2]).toBeCloseTo(2, 2);
  });

  it("closest_point_distance_intersects_contacts", () => {
    const s = scene();
    const a = addSphere(s, [0, 0, 0]);
    const b = addSphere(s, [3, 0, 0]);
    const c = s.closestPoint([1.4, 0, 0]);
    expect(c!.mesh).toBe(a);
    expect(c!.distance).toBeCloseTo(0.4, 3);
    expect(c!.point[0]).toBeCloseTo(1, 3);
    expect(s.closestPoint([1.4, 0, 0], { mesh: b })!.mesh).toBe(b);
    expect(s.closestPoint([1.4, 0, 0], { maxDistance: 0.1 })).toBeNull();

    const d = s.meshDistance(a, b);
    expect(d!.distance).toBeCloseTo(1, 3);
    expect(d!.a[0]).toBeCloseTo(1, 3);
    expect(d!.b[0]).toBeCloseTo(2, 3);
    expect(s.meshDistance(a, b, { maxDistance: 0.5 })).toBeNull();
    expect(s.intersects(a, b)).toBe(false);

    expect(s.findContacts({ margin: 0.5 })).toEqual([]);
    const [contact] = s.findContacts({ margin: 1.5 });
    expect(contact.a).toBe(a);
    expect(contact.b).toBe(b);
    expect(contact.intersecting).toBe(false);
    expect(contact.distance).toBeCloseTo(1, 3);
    expect(contact.trianglesA.length).toBeGreaterThan(0);
    expect(contact.trianglesB.length).toBeGreaterThan(0);

    s.setPlacement(b, { rotation: IDENTITY, translation: [-1.5, 0, 0] });
    expect(s.intersects(a, b)).toBe(true);
    const [touching] = s.findContacts({ pairs: [[b, a]] });
    expect(touching.intersecting).toBe(true);
    expect(touching.distance).toBe(0);
    expect(codeOf(() => s.findContacts({ pairs: [[a, a]] }))).toBe("INVALID_ARGUMENT");
  });
});

describe("fields and colouring", () => {
  it("distance_coloring_toggle", () => {
    const s = scene();
    const a = addSphere(s, [0, 0, 0]);
    const b = addSphere(s, [2.5, 0, 0]);
    const field = s.addDistanceField({ name: "gap", on: a, to: b, maxDistance: 2 });
    expect(s.fieldStatus(field)).toEqual({
      current: true,
      sign: {
        mode: "pseudonormal",
        closed: true,
        consistentOrientation: true,
        nonManifoldVertices: 0,
        degenerateTriangles: 0,
      },
    });
    const values = s.fieldValues(field);
    expect(values.length).toBe(s.vertexSlots(a));
    // The vertex at +x is 0.5 away from B.
    expect(Math.min(...values)).toBeCloseTo(0.5, 3);
    expect(values.some((v) => v === Number.POSITIVE_INFINITY)).toBe(true);
    s.takeChanges();
    // A placement then an undo leaves a redo step, which history changes would drop.
    s.setPlacement(b, { rotation: IDENTITY, translation: [0.01, 0, 0] });
    s.undo();
    s.takeChanges();
    const canUndo = s.canUndo();
    expect(s.canRedo()).toBe(true);

    expect(s.views(a).colors()).toBeNull();
    s.setColoring(a, { field, map: "turbo", range: [0, 2] });
    const colors = s.views(a).colors()!;
    expect(colors).not.toBeNull();
    expect(colors.length).toBe(4 * s.vertexSlots(a));
    const distinct = new Set<number>();
    for (let k = 0; k < colors.length; k += 4) {
      distinct.add((colors[k] << 16) | (colors[k + 1] << 8) | colors[k + 2]);
    }
    expect(distinct.size).toBeGreaterThan(2);
    const on = changeOf(s.takeChanges(), a)!;
    expect(on.colors === "all" || on.colors.length > 0).toBe(true);
    expect([s.canUndo(), s.canRedo()]).toEqual([canUndo, true]);

    // Off: no colours, history untouched, the change is reported.
    s.setColoring(a, null);
    expect(s.views(a).colors()).toBeNull();
    expect(changeOf(s.takeChanges(), a)!.colors).toBe("all");
    expect(s.views(a).colors()).toBeNull();
    expect([s.canUndo(), s.canRedo()]).toEqual([canUndo, true]);

    // Rules: custom stops, centre, out-of-range choices; invalid ones throw.
    s.setColoring(a, {
      field,
      map: { stops: [[0, [255, 0, 0, 255]], [1, [0, 0, 255, 255]]] },
      range: [0, 2],
      center: 1,
      below: "base",
      above: "clamp",
      base: [1, 2, 3, 4],
      nan: [5, 6, 7, 8],
    });
    expect(s.views(a).colors()).not.toBeNull();
    expect(codeOf(() => s.setColoring(a, { field, map: "viridis", range: [2, 0] }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.setColoring(b, { field, map: "viridis", range: [0, 1] }))).toBe(
      "INVALID_ARGUMENT",
    );
    for (const bad of [
      { center: Number.NaN },
      { center: 3 },
      { map: "jet" as "turbo" },
      { above: "wrap" as "base" },
      { base: [1, 2, 3] as unknown as [number, number, number, number] },
      { nan: [1, 2, 3] as unknown as [number, number, number, number] },
      { base: [1, 2, 3, 4, 5] as unknown as [number, number, number, number], nan: [1, 2, 3] as unknown as [number, number, number, number] },
      { nan: [0, 0, 0, 256] as const },
      { map: { stops: [[0, [0, 0, 0, 255]]] as const } },
      // Malformed stops: not TypeErrors.
      { map: { stops: [0, 1] as unknown as [] } },
      { map: { stops: [[0, [0, 0, 0, 255]], null] as unknown as [] } },
      { map: { stops: [[0, [0, 0, 0, 255]], [1]] as unknown as [] } },
      { map: { stops: [[0, [0, 0, 0, 255]], [1, 7]] as unknown as [] } },
    ]) {
      expect(codeOf(() => s.setColoring(a, { field, map: "turbo", range: [0, 2], ...bad }))).toBe(
        "INVALID_ARGUMENT",
      );
    }
  });

  it("drag_updates_colors_incrementally", () => {
    const s = scene();
    const a = addSphere(s, [0, 0, 0], 1, 32, 64);
    const b = addSphere(s, [2.2, 0, 0]);
    const field = s.addDistanceField({ name: "gap", on: a, to: b, maxDistance: 0.6 });
    s.setColoring(a, { field, map: "coolwarm", range: [0, 0.6] });
    s.takeChanges();
    const before = Uint8Array.from(s.views(a).colors()!);

    const drag = s.drag(b);
    drag.move({ rotation: IDENTITY, translation: [-0.05, 0, 0] });
    const changes = s.takeChanges();
    drag.end();
    const change = changeOf(changes, a)!;
    expect(change).toBeDefined();
    expect(change.fieldsChanged).toEqual([field]);
    expect(change.colors).not.toBe("all");
    const total = rangeTotal(change.colors);
    expect(total).toBeGreaterThan(0);
    expect(total).toBeLessThan(s.vertexSlots(a));
    const after = s.views(a).colors()!;
    expect(after).not.toEqual(before);
    // Only the reported ranges changed.
    const inRange = new Uint8Array(s.vertexSlots(a));
    for (const [start, count] of change.colors as [number, number][]) {
      inRange.fill(1, start, start + count);
    }
    for (let v = 0; v < inRange.length; v++) {
      if (!inRange[v]) {
        expect(after.subarray(4 * v, 4 * v + 4)).toEqual(before.subarray(4 * v, 4 * v + 4));
      }
    }
    // The drag is one undo step and colours follow it back.
    expect(s.undo()).toBe(true);
    expect(Uint8Array.from(s.views(a).colors()!)).toEqual(before);
  });
});

describe("sections", () => {
  it("section_contours_and_gap", () => {
    const s = scene();
    const a = addSphere(s, [0, 0, 0]);
    const b = addSphere(s, [3, 0, 0]);
    const plane = { origin: [0, 0, 0] as Vec3, normal: [0, 0, 1] as Vec3 };
    const sec = s.addSection({ plane, meshes: [a, b], fill: true });
    s.updateSections();
    const ra = s.section(sec, a);
    expect(ra.status).toBe("current");
    expect(ra.contours.length).toBe(1);
    const [loop] = ra.contours;
    expect(loop.closed).toBe(true);
    expect(loop.points).toBeInstanceOf(Float32Array);
    // The equator vertices lie on the plane: each comes once per crossed edge.
    expect(loop.points.length % 3).toBe(0);
    expect(loop.points.length).toBeGreaterThanOrEqual(3 * 32);
    for (let k = 0; k < loop.points.length; k += 3) {
      expect(Math.abs(loop.points[k + 2])).toBeLessThan(1e-6);
      expect(Math.hypot(loop.points[k], loop.points[k + 1])).toBeCloseTo(1, 5);
    }
    // A 32-gon inscribed in the unit circle.
    expect(ra.length).toBeCloseTo(64 * Math.sin(Math.PI / 32), 4);
    expect(Math.abs(ra.area)).toBeCloseTo(16 * Math.sin(Math.PI / 16), 4);
    const fill = ra.fill!;
    expect(fill.positions.length % 3).toBe(0);
    expect(fill.indices.length % 3).toBe(0);
    expect(fill.indices.length).toBeGreaterThanOrEqual(3 * 30);
    expect(Math.max(...fill.indices)).toBeLessThan(fill.positions.length / 3);
    for (let k = 2; k < ra.fill!.positions.length; k += 3) {
      expect(Math.abs(ra.fill!.positions[k])).toBeLessThan(1e-6);
    }
    // World coordinates: B's contour is around x = 3.
    const rb = s.section(sec, b);
    expect(rb.contours[0].points[0]).toBeGreaterThan(1.9);

    const gap = s.sectionGap(sec, a, b)!;
    expect(Math.abs(gap.distance - 1)).toBeLessThan(1e-3);
    expect(gap.a[0]).toBeCloseTo(1, 3);
    expect(gap.b[0]).toBeCloseTo(2, 3);

    // Moving the plane above the spheres: no contours, no gap. The plane
    // reads back as set, the normal normalised.
    expect(s.sectionPlane(sec)).toEqual(plane);
    s.setSectionPlane(sec, { origin: [0, 0, 5], normal: [0, 0, 2] });
    expect(s.sectionPlane(sec)).toEqual({ origin: [0, 0, 5], normal: [0, 0, 1] });
    s.updateSections();
    expect(s.section(sec, a).contours).toEqual([]);
    expect(s.sectionGap(sec, a, b)).toBeNull();
    s.setSectionPlane(sec, plane);
    s.setSectionFill(sec, false);
    s.updateSections();
    expect(s.section(sec, a).fill).toBeUndefined();
    s.setSectionMeshes(sec, [a]);
    expect(codeOf(() => s.section(sec, b))).toBe("INVALID_ARGUMENT");
    // A Manual mesh's entry stays stale until updated.
    s.setSectionMeshes(sec, [a, b]);
    s.setMode(b, "manual");
    s.setPlacement(b, { rotation: IDENTITY, translation: [0.5, 0, 0] });
    expect(s.section(sec, b).status).toBe("stale");
  });

  it("error_codes_unknown_field_section", () => {
    const s = scene();
    const a = addSphere(s, [0, 0, 0]);
    const b = addSphere(s, [3, 0, 0]);
    const field = s.addDistanceField({ name: "d", on: a, to: b, maxDistance: 1, sign: "winding" });
    expect(s.fieldStatus(field).sign.mode).toBe("winding");
    const sec = s.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [a] });
    // Duplicate names, same meshes, bad modes.
    expect(codeOf(() => s.addDistanceField({ name: "d", on: a, to: b, maxDistance: 1 }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(codeOf(() => s.addDistanceField({ name: "e", on: a, to: a, maxDistance: 1 }))).toBe(
      "INVALID_ARGUMENT",
    );
    expect(
      codeOf(() =>
        s.addDistanceField({
          name: "e",
          on: a,
          to: b,
          maxDistance: 1,
          sign: "nope" as "auto",
        }),
      ),
    ).toBe("INVALID_ARGUMENT");
    expect(
      codeOf(() => s.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, 0] }, meshes: [a] })),
    ).toBe("INVALID_ARGUMENT");

    s.removeField(field);
    expect(codeOf(() => s.fieldValues(field))).toBe("UNKNOWN_FIELD");
    expect(codeOf(() => s.fieldStatus(field))).toBe("UNKNOWN_FIELD");
    expect(codeOf(() => s.removeField(field))).toBe("UNKNOWN_FIELD");
    expect(codeOf(() => s.setColoring(a, { field, map: "turbo", range: [0, 1] }))).toBe(
      "UNKNOWN_FIELD",
    );
    s.removeSection(sec);
    expect(codeOf(() => s.section(sec, a))).toBe("UNKNOWN_SECTION");
    expect(codeOf(() => s.setSectionFill(sec, true))).toBe("UNKNOWN_SECTION");
    expect(codeOf(() => s.sectionPlane(sec))).toBe("UNKNOWN_SECTION");
    expect(codeOf(() => s.removeSection(sec))).toBe("UNKNOWN_SECTION");
    // The same handles come back in change records.
    const changes = s.takeChanges();
    expect(changeOf(changes, a)!.fieldsChanged).toContain(field);
    expect(changeOf(changes, a)!.sectionsChanged).toContain(sec);

    // Handles of another scene, and things that are not handles.
    const other = scene();
    const c = addSphere(other, [0, 0, 0]);
    const d = addSphere(other, [3, 0, 0]);
    const otherField = other.addDistanceField({ name: "d", on: c, to: d, maxDistance: 1 });
    expect(codeOf(() => s.fieldValues(otherField))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.meshDistance(a, c))).toBe("INVALID_ARGUMENT");
    expect(codeOf(() => s.fieldValues(a as never))).toBe("INVALID_ARGUMENT");
    // Removing a mesh drops its fields (both of c's are measured to d): they
    // are reported in the remaining mesh's record, with the handles held.
    const f2 = other.addDistanceField({ name: "e", on: c, to: d, maxDistance: 1 });
    other.takeChanges();
    other.removeMesh(d);
    expect(codeOf(() => other.fieldValues(f2))).toBe("UNKNOWN_FIELD");
    const dropped = other.takeChanges();
    const fieldsChanged = changeOf(dropped, c)!.fieldsChanged;
    expect(fieldsChanged.length).toBe(2);
    expect(fieldsChanged).toContain(otherField);
    expect(fieldsChanged).toContain(f2);
    expect(changeOf(dropped, d)!.removed).toBe(true);
  });
});
