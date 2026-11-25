// The placement helpers (#256): move to origin, lay flat, align to axis,
// 3-2-1 and apply transform. The math is checked exactly (toEqual, no
// tolerance) on an axis-aligned box placed with a quarter turn, and each
// helper runs through the real engine as one undo step.
import { type MeshHandle, Scene, type Vec3 } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { afterEach, describe, expect, it } from "vitest";
import { AppState } from "../src/app/state/app-state";
import { applyTransform, placeMesh, readPlacedMesh, worldTriangleNormal } from "../src/features/transform/ux/placement-helpers";
import {
  alignToAxis,
  axisVector,
  bakedPositions,
  datumAlign,
  isIdentityPlacement,
  layFlat,
  type Mat3,
  mulMat3Vec,
  moveToOrigin,
  type PlacedMesh,
  type PlacementLike,
  placementMatrix,
  rotationBetween,
  samePlacement,
  type SignedAxis,
  triangleNormal,
  worldBounds,
} from "../src/features/transform/ux/placement-math";
import {
  addPlacementHit,
  PICKS_NEEDED,
  type PlacementPick,
  pickMarkerGroup,
  pickPrompt,
  placementForPick,
} from "../src/features/transform/ux/placement-picks";
import { landPlacementHit } from "../src/features/transform/ux/use-placement-picks";

const cleanups: (() => void)[] = [];
afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

/** `-0` as `0`, for exact comparisons. */
const n0 = (v: readonly number[]): number[] => v.map((x) => x + 0);

// A box of half extents (1, 2, 3) about the origin point (10, 20, 30): render
// positions are relative to the origin. Its triangles face outward:
// -Z (0, 1), +Z (2, 3), -Y (4, 5), +Y (6, 7), -X (8, 9), +X (10, 11).
const HALF: Vec3 = [1, 2, 3];
const ORIGIN: Vec3 = [10, 20, 30];

function boxLocal(): { positions: Float32Array; indices: Uint32Array } {
  const [hx, hy, hz] = HALF;
  const positions = Float32Array.from([
    -hx, -hy, -hz,
    hx, -hy, -hz,
    hx, hy, -hz,
    -hx, hy, -hz,
    -hx, -hy, hz,
    hx, -hy, hz,
    hx, hy, hz,
    -hx, hy, hz,
  ]);
  const indices = Uint32Array.from([
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6,
    0, 4, 7, 0, 7, 3,
    1, 2, 6, 1, 6, 5,
  ]);
  return { positions, indices };
}

/** A quarter turn about +Z: (x, y, z) -> (-y, x, z). */
const QUARTER_Z: Mat3 = [0, -1, 0, 1, 0, 0, 0, 0, 1];

/** The box placed by a quarter turn about Z and a move: world (x, y, z) = (5 - Y, X - 7, Z + 2) for its points (X, Y, Z). */
const P0: PlacementLike = { rotation: QUARTER_Z, translation: [5, -7, 2], scale: [1, 1, 1] };

function placed(placement: PlacementLike = P0): PlacedMesh {
  const { positions, indices } = boxLocal();
  return { positions, indices, matrix: placementMatrix(placement, ORIGIN), placement };
}

/** The world box of the mesh under `placement`. */
function boundsUnder(placement: PlacementLike): { min: number[]; max: number[] } {
  const b = worldBounds(placed(placement))!;
  return { min: n0(b.min), max: n0(b.max) };
}

/** The world position of the box's render vertex `slot` under `placement`. */
function vertexUnder(placement: PlacementLike, slot: number): number[] {
  const { positions } = boxLocal();
  const m = placementMatrix(placement, ORIGIN);
  const [x, y, z] = [positions[3 * slot], positions[3 * slot + 1], positions[3 * slot + 2]];
  return n0([
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ]);
}

const AXES: readonly SignedAxis[] = ["+x", "-x", "+y", "-y", "+z", "-z"];

describe("placement math (#256)", () => {
  it("starts from the expected world box", () => {
    expect(boundsUnder(P0)).toEqual({ min: [-17, 2, 29], max: [-13, 4, 35] });
  });

  it("turns every signed axis onto every other exactly, with a proper rotation", () => {
    for (const from of AXES) {
      for (const to of AXES) {
        const r = rotationBetween(axisVector(from), axisVector(to));
        expect(n0(mulMat3Vec(r, axisVector(from)))).toEqual(n0(axisVector(to)));
        expect(r.every((v) => v === 0 || v === 1 || v === -1)).toBe(true);
        // Orthonormal rows, determinant +1: R Rᵀ = I and row0 × row1 = row2.
        const rows = [r.slice(0, 3), r.slice(3, 6), r.slice(6, 9)];
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            expect(rows[i][0] * rows[j][0] + rows[i][1] * rows[j][1] + rows[i][2] * rows[j][2]).toBe(i === j ? 1 : 0);
          }
        }
        const c = [
          rows[0][1] * rows[1][2] - rows[0][2] * rows[1][1],
          rows[0][2] * rows[1][0] - rows[0][0] * rows[1][2],
          rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0],
        ];
        expect(n0(c)).toEqual(n0(rows[2]));
      }
    }
  });

  it("turns a general direction onto an axis (and back from nearly opposite) as a rotation", () => {
    const cases: [Vec3, Vec3][] = [
      [[1, 2, 3], [0, 0, 1]],
      [[-0.3, 0.1, -5], [0, 0, 1]],
      [[1e-9, 0, -1], [0, 0, 1]],
    ];
    for (const [from, to] of cases) {
      const r = rotationBetween(from, to);
      const len = Math.hypot(...from);
      const got = mulMat3Vec(r, [from[0] / len, from[1] / len, from[2] / len]);
      for (let k = 0; k < 3; k++) {
        expect(got[k]).toBeCloseTo(to[k], 12);
      }
    }
    expect(() => rotationBetween([0, 0, 0], [1, 0, 0])).toThrow();
  });

  it("moves the box centre, or its bottom centre (Z or Y up), to the origin exactly", () => {
    const center = moveToOrigin(placed(), "center");
    expect(n0(center.translation as Vec3)).toEqual([20, -10, -30]);
    expect(center.rotation).toEqual(QUARTER_Z);
    expect(boundsUnder(center as PlacementLike)).toEqual({ min: [-2, -1, -3], max: [2, 1, 3] });

    const bottomZ = moveToOrigin(placed(), "bottom", "z");
    expect(boundsUnder(bottomZ as PlacementLike)).toEqual({ min: [-2, -1, 0], max: [2, 1, 6] });

    const bottomY = moveToOrigin(placed(), "bottom", "y");
    expect(boundsUnder(bottomY as PlacementLike)).toEqual({ min: [-2, 0, -3], max: [2, 2, 3] });
  });

  it("lays the box flat on a side face: the face points down and rests on Z = 0", () => {
    // The box's -Y face (vertices 0, 1, 4, 5) faces world +X after the quarter turn.
    const normal = triangleNormal(
      vertexUnder(P0, 0) as Vec3,
      vertexUnder(P0, 1) as Vec3,
      vertexUnder(P0, 5) as Vec3,
    );
    expect(normal).toEqual([1, 0, 0]);
    const next = layFlat(placed(), normal!, "z") as PlacementLike;
    expect(n0(next.rotation)).toEqual([0, 0, 1, 1, 0, 0, 0, 1, 0]);
    expect(n0(next.translation)).toEqual([-45, -7, -18]);
    expect(next.scale).toEqual([1, 1, 1]);
    expect(boundsUnder(next)).toEqual({ min: [-18, 2, 0], max: [-12, 4, 4] });
    for (const slot of [0, 1, 4, 5]) {
      expect(vertexUnder(next, slot)[2]).toBe(0);
    }
    // The turned face's normal is straight down.
    expect(triangleNormal(vertexUnder(next, 0) as Vec3, vertexUnder(next, 1) as Vec3, vertexUnder(next, 5) as Vec3)).toEqual([0, 0, -1]);
  });

  it("lays flat on the face already down (a drop) and on the top face (a half turn)", () => {
    const down = layFlat(placed(), [0, 0, -1], "z") as PlacementLike;
    expect(down.rotation).toEqual(QUARTER_Z);
    expect(boundsUnder(down)).toEqual({ min: [-17, 2, 0], max: [-13, 4, 6] });

    const top = layFlat(placed(), [0, 0, 1], "z") as PlacementLike;
    expect(boundsUnder(top)).toEqual({ min: [-17, 2, 0], max: [-13, 4, 6] });
    // The top face's vertices (4..7) are now on the floor.
    for (const slot of [4, 5, 6, 7]) {
      expect(vertexUnder(top, slot)[2]).toBe(0);
    }
  });

  it("lays flat with Y up: the face points along -Y and rests on Y = 0", () => {
    const next = layFlat(placed(), [1, 0, 0], "y") as PlacementLike;
    expect(boundsUnder(next).min[1]).toBe(0);
    for (const slot of [0, 1, 4, 5]) {
      expect(vertexUnder(next, slot)[1]).toBe(0);
    }
  });

  it("aligns a two-point direction or a face normal to each signed axis exactly, about the box centre", () => {
    const a: Vec3 = [-17, 2, 29];
    const b: Vec3 = [-17, 4, 29];
    for (const target of AXES) {
      const next = alignToAxis(placed(), [b[0] - a[0], b[1] - a[1], b[2] - a[2]], target) as PlacementLike;
      // Vertex 3 is at a, vertex 2 at b: the box's X edge, world +Y after the quarter turn.
      expect(vertexUnder(P0, 3)).toEqual(a);
      expect(vertexUnder(P0, 2)).toEqual(b);
      const p = vertexUnder(next, 3);
      const q = vertexUnder(next, 2);
      const v = axisVector(target);
      expect(n0([q[0] - p[0], q[1] - p[1], q[2] - p[2]])).toEqual(n0([2 * v[0], 2 * v[1], 2 * v[2]]));
      const box = boundsUnder(next);
      expect(n0([(box.min[0] + box.max[0]) / 2, (box.min[1] + box.max[1]) / 2, (box.min[2] + box.max[2]) / 2])).toEqual([-15, 3, 32]);
    }
    const face = alignToAxis(placed(), [1, 0, 0], "+z") as PlacementLike;
    expect(triangleNormal(vertexUnder(face, 0) as Vec3, vertexUnder(face, 1) as Vec3, vertexUnder(face, 5) as Vec3)).toEqual([0, 0, 1]);
  });

  it("3-2-1 puts the datum planes on the world planes and the datum origin at the origin (Z up)", () => {
    // Primary plane: the world +X face (x = -13); secondary: along +Z on the
    // y = 2 face; origin: a point on the z = 29 face.
    const picks = {
      plane: [[-13, 2, 29], [-13, 4, 29], [-13, 2, 35]],
      line: [[-15, 2, 30], [-15, 2, 34]],
      origin: [-14, 3, 29],
    } as const;
    const next = datumAlign(placed(), picks as Parameters<typeof datumAlign>[1], "z") as PlacementLike;
    // World (x, y, z) -> (z - 29, y - 2, -x - 13): render points (X, Y, Z) land at (Z - 27, X - 9, Y - 18).
    expect(n0(next.rotation)).toEqual([0, 0, 1, 1, 0, 0, 0, 1, 0]);
    expect(n0(next.translation)).toEqual([-27, -9, -18]);
    expect(boundsUnder(next)).toEqual({ min: [0, 0, 0], max: [6, 2, 4] });
    // The primary plane's face (world x = -13, the box's -Y face) is on Z = 0.
    for (const slot of [0, 1, 4, 5]) {
      expect(vertexUnder(next, slot)[2]).toBe(0);
    }
    // The picked points land on their planes: map each old world point back
    // to a render point (the old placement is rigid: transpose its turn) and
    // through the new placement.
    const toNew = (w: readonly number[]) => {
      const old = placementMatrix(P0, ORIGIN);
      const d = [w[0] - old[12], w[1] - old[13], w[2] - old[14]];
      const local = [0, 4, 8].map((c) => old[c] * d[0] + old[c + 1] * d[1] + old[c + 2] * d[2]);
      const n = placementMatrix(next, ORIGIN);
      return n0([
        n[0] * local[0] + n[4] * local[1] + n[8] * local[2] + n[12],
        n[1] * local[0] + n[5] * local[1] + n[9] * local[2] + n[13],
        n[2] * local[0] + n[6] * local[1] + n[10] * local[2] + n[14],
      ]);
    };
    for (const p of picks.plane) expect(toNew(p)[2]).toBe(0);
    for (const p of picks.line) expect(toNew(p)[1]).toBe(0);
    expect(toNew(picks.origin)[0]).toBe(0);
    // The line runs along +X.
    const [l0, l1] = picks.line.map(toNew);
    expect(n0([l1[0] - l0[0], l1[1] - l0[1], l1[2] - l0[2]])).toEqual([4, 0, 0]);
  });

  it("3-2-1 with Y up puts the primary plane on Y = 0, the mesh above it", () => {
    const picks = {
      plane: [[-17, 2, 29], [-13, 2, 29], [-17, 2, 35]],
      line: [[-17, 3, 29], [-13, 3, 29]],
      origin: [-17, 2, 30],
    } as const;
    const next = datumAlign(placed(), picks as Parameters<typeof datumAlign>[1], "y") as PlacementLike;
    // Primary plane y = 2 (the box's -X face, vertices 0, 3, 4, 7) onto Y = 0, the box above.
    expect(boundsUnder(next)).toEqual({ min: [0, 0, 0], max: [4, 2, 6] });
    for (const slot of [0, 3, 4, 7]) {
      expect(vertexUnder(next, slot)[1]).toBe(0);
    }
  });

  it("3-2-1 rejects collinear plane points and a line along the plane normal", () => {
    const line = [[-15, 2, 30], [-15, 2, 34]] as const;
    expect(() =>
      datumAlign(placed(), { plane: [[0, 0, 0], [1, 0, 0], [2, 0, 0]], line, origin: [0, 0, 0] }),
    ).toThrow(/one line/);
    expect(() =>
      datumAlign(placed(), {
        plane: [[-13, 2, 29], [-13, 4, 29], [-13, 2, 35]],
        line: [[-14, 2, 30], [-16, 2, 30]],
        origin: [0, 0, 0],
      }),
    ).toThrow(/normal/);
  });

  it("keeps a placement's scale, and bakes rotation, scale and move into world positions", () => {
    const scaled: PlacementLike = { rotation: QUARTER_Z, translation: [5, -7, 2], scale: [2, 1, 1] };
    const moved = moveToOrigin(placed(scaled), "center") as PlacementLike;
    expect(moved.scale).toEqual([2, 1, 1]);
    expect(boundsUnder(moved)).toEqual({ min: [-2, -2, -3], max: [2, 2, 3] });

    const { vertices, positions } = bakedPositions(placed(scaled));
    expect(Array.from(vertices)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    for (let i = 0; i < 8; i++) {
      expect(n0(Array.from(positions.subarray(3 * i, 3 * i + 3)))).toEqual(vertexUnder(scaled, i));
    }
    // Vertex 0: (X, Y, Z) = (9, 18, 27) -> (5 - 18, 2 * 9 - 7, 27 + 2).
    expect(n0(Array.from(positions.subarray(0, 3)))).toEqual([-13, 11, 29]);
    expect(isIdentityPlacement({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] })).toBe(true);
    expect(isIdentityPlacement(scaled)).toBe(false);
  });

  it("compares placements entry for entry", () => {
    expect(samePlacement(P0, { ...P0, scale: undefined })).toBe(true);
    expect(samePlacement(P0, { ...P0, translation: [5, -7, 3] })).toBe(false);
  });
});

describe("placement picks (#256)", () => {
  const pick = (kind: PlacementPick["kind"], target: SignedAxis = "+z"): PlacementPick => ({
    kind,
    mesh: 1 as unknown as MeshHandle,
    points: [],
    normal: null,
    up: "z",
    target,
  });

  it("counts clicks, prompts for each and colours 3-2-1's groups", () => {
    let p = pick("datum321");
    const prompts: string[] = [];
    for (let i = 0; i < PICKS_NEEDED.datum321; i++) {
      prompts.push(pickPrompt(p));
      const r = addPlacementHit(p, { point: [i, 0, 0], normal: null });
      p = r.pick;
      expect(r.complete).toBe(i === 5);
    }
    expect(prompts[0]).toContain("Primary plane: click point 1 of 3");
    expect(prompts[3]).toContain("Secondary direction: click point 1 of 2");
    expect(prompts[5]).toContain("Origin");
    expect([0, 1, 2, 3, 4, 5].map((i) => pickMarkerGroup("datum321", i))).toEqual([0, 0, 0, 1, 1, 2]);
    expect(pickMarkerGroup("alignPoints", 1)).toBe(0);
  });

  it("ignores a degenerate face on a face pick and turns completed picks into placements", () => {
    const lay = pick("layFlat");
    expect(addPlacementHit(lay, { point: [0, 0, 0], normal: null })).toEqual({ pick: lay, complete: false });
    const done = addPlacementHit(lay, { point: [-13, 3, 32], normal: [1, 0, 0] });
    expect(done.complete).toBe(true);
    expect(placementForPick(placed(), done.pick)).toEqual(layFlat(placed(), [1, 0, 0], "z"));

    let two = pick("alignPoints", "+x");
    two = addPlacementHit(two, { point: [-17, 2, 29], normal: [0, 0, -1] }).pick;
    const both = addPlacementHit(two, { point: [-17, 4, 29], normal: [0, 0, -1] });
    expect(both.complete).toBe(true);
    expect(placementForPick(placed(), both.pick)).toEqual(alignToAxis(placed(), [0, 2, 0], "+x"));
  });
});

// ---- Through the real engine ----

function engineBox(placement: PlacementLike = P0): { scene: Scene; store: SceneStore; mesh: MeshHandle } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  const { positions, indices } = boxLocal();
  const world = Float64Array.from(positions, (v, i) => v + ORIGIN[i % 3]);
  const mesh = store.addMesh(world, indices, { origin: ORIGIN });
  store.setPlacement(mesh, { rotation: [...placement.rotation] as Mat3, translation: placement.translation, scale: placement.scale });
  scene.clearHistory();
  return { scene, store, mesh };
}

function engineBounds(scene: Scene, mesh: MeshHandle): { min: number[]; max: number[] } {
  const b = worldBounds(readPlacedMesh(scene, mesh))!;
  return { min: n0(b.min), max: n0(b.max) };
}

describe("placement helpers through the engine (#256)", () => {
  it("each placement helper is one undo step with exact world bounds", () => {
    const cases: [string, (m: PlacedMesh) => ReturnType<typeof moveToOrigin>, { min: number[]; max: number[] }][] = [
      ["centre", (m) => moveToOrigin(m, "center"), { min: [-2, -1, -3], max: [2, 1, 3] }],
      ["bottom", (m) => moveToOrigin(m, "bottom", "z"), { min: [-2, -1, 0], max: [2, 1, 6] }],
      ["lay flat", (m) => layFlat(m, [1, 0, 0], "z"), { min: [-18, 2, 0], max: [-12, 4, 4] }],
      ["align", (m) => alignToAxis(m, [0, 1, 0], "+x"), { min: [-16, 1, 29], max: [-14, 5, 35] }],
      [
        "3-2-1",
        (m) =>
          datumAlign(m, {
            plane: [[-13, 2, 29], [-13, 4, 29], [-13, 2, 35]],
            line: [[-15, 2, 30], [-15, 2, 34]],
            origin: [-14, 3, 29],
          }),
        { min: [0, 0, 0], max: [6, 2, 4] },
      ],
    ];
    for (const [name, compute, bounds] of cases) {
      const { scene, store, mesh } = engineBox();
      const before = scene.placement(mesh);
      expect(scene.canUndo(), name).toBe(false);
      const next = placeMesh(store, mesh, compute);
      expect(next, name).not.toBeNull();
      expect(engineBounds(scene, mesh), name).toEqual(bounds);
      store.undo();
      expect(scene.placement(mesh), name).toEqual(before);
      expect(scene.canUndo(), name).toBe(false);
      store.redo();
      expect(engineBounds(scene, mesh), name).toEqual(bounds);
    }
  });

  it("records nothing when the mesh is already in place", () => {
    const { scene, store, mesh } = engineBox();
    expect(placeMesh(store, mesh, (m) => moveToOrigin(m, "center"))).not.toBeNull();
    expect(placeMesh(store, mesh, (m) => moveToOrigin(m, "center"))).toBeNull();
    store.undo();
    expect(scene.canUndo()).toBe(false);
  });

  it("apply transform bakes rotation, scale and move into the vertices as one undo step", async () => {
    const scaled: PlacementLike = { rotation: QUARTER_Z, translation: [5, -7, 2], scale: [2, 1, 1] };
    const { scene, store, mesh } = engineBox(scaled);
    const before = engineBounds(scene, mesh);
    const positionsBefore = Array.from(scene.views(mesh).positions());
    const placementBefore = scene.placement(mesh);
    expect(await applyTransform(store, mesh)).toBe(true);
    const after = scene.placement(mesh);
    expect(after.rotation).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(after.translation).toEqual([0, 0, 0]);
    expect(after.scale).toEqual([1, 1, 1]);
    expect(engineBounds(scene, mesh)).toEqual(before);
    // Every vertex keeps its world position.
    const baked = readPlacedMesh(scene, mesh);
    for (let i = 0; i < 8; i++) {
      const m = baked.matrix;
      const p = baked.positions;
      const w = [
        m[0] * p[3 * i] + m[4] * p[3 * i + 1] + m[8] * p[3 * i + 2] + m[12],
        m[1] * p[3 * i] + m[5] * p[3 * i + 1] + m[9] * p[3 * i + 2] + m[13],
        m[2] * p[3 * i] + m[6] * p[3 * i + 1] + m[10] * p[3 * i + 2] + m[14],
      ];
      expect(n0(w)).toEqual(vertexUnder(scaled, i));
    }
    expect(await applyTransform(store, mesh)).toBe(false);

    store.undo();
    expect(scene.placement(mesh)).toEqual(placementBefore);
    expect(Array.from(scene.views(mesh).positions())).toEqual(positionsBefore);
    expect(scene.canUndo()).toBe(false);
    store.redo();
    expect(isIdentityPlacement(scene.placement(mesh))).toBe(true);
    expect(engineBounds(scene, mesh)).toEqual(before);
  });

  it("lands surface picks from engine hits: lay flat on a clicked face, 3-2-1 from six clicks", () => {
    const { scene, store, mesh } = engineBox();
    const app = new AppState();
    // The box's -Y face is triangles 4 and 5; it faces world +X.
    expect(worldTriangleNormal(scene, mesh, 4)).toEqual([1, 0, 0]);
    app.setPlacementTool({ pick: { kind: "layFlat", mesh, points: [], normal: null, up: "z", target: "+z" } });
    landPlacementHit(app, store, { mesh, triangle: 4, point: [-13, 3, 32] });
    expect(app.getSnapshot().placementTool.pick).toBeNull();
    expect(app.getSnapshot().placementTool.report).toEqual({ text: "Laid flat on Z = 0", error: false });
    expect(engineBounds(scene, mesh)).toEqual({ min: [-18, 2, 0], max: [-12, 4, 4] });
    store.undo();
    expect(scene.canUndo()).toBe(false);

    app.setPlacementTool({ pick: { kind: "datum321", mesh, points: [], normal: null, up: "z", target: "+z" } });
    const clicks: Vec3[] = [[-13, 2, 29], [-13, 4, 29], [-13, 2, 35], [-15, 2, 30], [-15, 2, 34], [-14, 3, 29]];
    clicks.forEach((point, i) => {
      landPlacementHit(app, store, { mesh, triangle: 0, point });
      expect(app.getSnapshot().placementTool.pick?.points.length ?? 6).toBe(i + 1);
    });
    expect(app.getSnapshot().placementTool.report?.text).toBe("3-2-1 alignment applied");
    expect(engineBounds(scene, mesh)).toEqual({ min: [0, 0, 0], max: [6, 2, 4] });
    store.undo();
    expect(scene.canUndo()).toBe(false);

    // A degenerate pick (two equal points) reports an error and changes nothing.
    app.setPlacementTool({ pick: { kind: "alignPoints", mesh, points: [], normal: null, up: "z", target: "+x" } });
    landPlacementHit(app, store, { mesh, triangle: 0, point: [1, 1, 1] });
    landPlacementHit(app, store, { mesh, triangle: 0, point: [1, 1, 1] });
    expect(app.getSnapshot().placementTool.report?.error).toBe(true);
    expect(app.getSnapshot().placementTool.pick).toBeNull();
    expect(scene.canUndo()).toBe(false);

    // A hit on another mesh, or with nothing armed, does nothing.
    landPlacementHit(app, store, { mesh, triangle: 0, point: [0, 0, 0] });
    expect(scene.canUndo()).toBe(false);
  });
});
