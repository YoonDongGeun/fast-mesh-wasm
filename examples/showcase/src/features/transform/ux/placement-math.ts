// The placement helpers' math (#256): move to origin, lay flat, align to an
// axis and the 3-2-1 datum alignment each give a new rigid placement for the
// mesh, and apply transform gives the world positions its placement puts the
// vertices at. Every helper works on world points, so it turns or moves the
// mesh as it is placed now; the new placement keeps the scale. Rotations are
// built so that axis-aligned inputs give exact 0 and ±1 entries, so an
// axis-aligned mesh lands exactly.
import type { PlacementInput, Vec3 } from "fast-mesh-wasm";
import { liveVertexIds } from "./transform-math";

/** A row-major 3x3 matrix. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];

/** The axis that points up: the floor is the plane where that coordinate is 0. */
export type UpAxis = "z" | "y";

/** A world axis with a sign, the target of an alignment. */
export type SignedAxis = "+x" | "-x" | "+y" | "-y" | "+z" | "-z";

/** Which point of the bounding box moves to the origin. */
export type OriginAnchor = "center" | "bottom";

/** A placement as the scene reports it: rotation row-major, scale optional (`[1, 1, 1]`). */
export interface PlacementLike {
  rotation: readonly number[];
  translation: Readonly<Vec3>;
  scale?: Readonly<Vec3>;
}

/**
 * A mesh as the helpers read it: its render positions and triangles, the
 * column-major world matrix of those positions (`scene.matrix4`: the
 * placement times the translation to the mesh origin) and its placement.
 */
export interface PlacedMesh {
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
  matrix: ArrayLike<number>;
  placement: PlacementLike;
}

/** A world axis-aligned box. */
export interface WorldBounds {
  min: Vec3;
  max: Vec3;
}

/** The 3-2-1 picks: three points on the primary plane, two along the secondary direction, one for the origin. */
export interface DatumPicks {
  plane: readonly [Readonly<Vec3>, Readonly<Vec3>, Readonly<Vec3>];
  line: readonly [Readonly<Vec3>, Readonly<Vec3>];
  origin: Readonly<Vec3>;
}

export const IDENTITY_MAT3: Readonly<Mat3> = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** Below this length a direction counts as zero. */
const TINY = 1e-12;

/** `-0` as `0`, so placements and readouts never show a signed zero. */
const z0 = (x: number): number => x + 0;

export function dot(a: Readonly<Vec3>, b: Readonly<Vec3>): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function sub(a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** `a` scaled to unit length, or `null` when it is (nearly) zero or not finite. */
export function unit(a: Readonly<Vec3>): Vec3 | null {
  const n = Math.hypot(a[0], a[1], a[2]);
  if (!(n > TINY) || !Number.isFinite(n)) {
    return null;
  }
  return [a[0] / n, a[1] / n, a[2] / n];
}

export function mulMat3(a: Readonly<Mat3>, b: Readonly<Mat3>): Mat3 {
  const out = new Array<number>(9) as Mat3;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    }
  }
  return out;
}

export function mulMat3Vec(m: Readonly<Mat3>, v: Readonly<Vec3>): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

/** The rows made orthonormal (Gram-Schmidt, third row from the cross product), as the engine does. */
export function orthonormalize(m: Readonly<Mat3>): Mat3 {
  const r0 = unit([m[0], m[1], m[2]]) ?? [1, 0, 0];
  const m1: Vec3 = [m[3], m[4], m[5]];
  const d = dot(m1, r0);
  const r1 = unit([m1[0] - d * r0[0], m1[1] - d * r0[1], m1[2] - d * r0[2]]) ?? [0, 1, 0];
  const r2 = cross(r0, r1);
  return [r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2]].map(z0) as Mat3;
}

/** The unit vector of a signed axis. */
export function axisVector(axis: SignedAxis): Vec3 {
  const s = axis[0] === "-" ? -1 : 1;
  const k = axis[1] === "x" ? 0 : axis[1] === "y" ? 1 : 2;
  const v: Vec3 = [0, 0, 0];
  v[k] = s;
  return v;
}

/** The index of the up axis (1 for Y, 2 for Z). */
export function upIndex(up: UpAxis): 1 | 2 {
  return up === "y" ? 1 : 2;
}

/** A unit vector perpendicular to the unit vector `a`: `a` crossed with the world axis it is least along. */
function perpendicular(a: Readonly<Vec3>): Vec3 {
  const ax = Math.abs(a[0]);
  const ay = Math.abs(a[1]);
  const az = Math.abs(a[2]);
  const e: Vec3 = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1];
  return unit(cross(a, e)) ?? [0, 1, 0];
}

/**
 * The smallest rotation turning direction `from` onto direction `to`
 * (Rodrigues: `c I + s K + (1 - c) u uᵀ` about the unit axis `u` of
 * `from × to`, with `c` and `s` the angle's cosine and sine). Opposite
 * directions turn half a turn about an axis perpendicular to `from`.
 * Axis-aligned directions give exact 0 and ±1 entries.
 *
 * @throws when either direction is zero.
 */
export function rotationBetween(from: Readonly<Vec3>, to: Readonly<Vec3>): Mat3 {
  const a = unit(from);
  const b = unit(to);
  if (a === null || b === null) {
    throw new Error("The direction has no length");
  }
  let c = dot(a, b);
  const v = cross(a, b);
  let s = Math.hypot(v[0], v[1], v[2]);
  let u: Vec3;
  if (s <= TINY) {
    if (c > 0) {
      return [...IDENTITY_MAT3] as Mat3;
    }
    u = perpendicular(a);
    s = 0;
    c = -1;
  } else {
    u = [v[0] / s, v[1] / s, v[2] / s];
  }
  const t = 1 - c;
  return [
    c + t * u[0] * u[0],
    -s * u[2] + t * u[0] * u[1],
    s * u[1] + t * u[0] * u[2],
    s * u[2] + t * u[1] * u[0],
    c + t * u[1] * u[1],
    -s * u[0] + t * u[1] * u[2],
    -s * u[1] + t * u[2] * u[0],
    s * u[0] + t * u[2] * u[1],
    c + t * u[2] * u[2],
  ].map(z0) as Mat3;
}

/**
 * The column-major world matrix of a mesh's render positions, as
 * `scene.matrix4` gives it: `[R S | R S origin + t]`.
 */
export function placementMatrix(placement: PlacementLike, origin: Readonly<Vec3>): Float64Array {
  const r = placement.rotation;
  const t = placement.translation;
  const s = placement.scale ?? [1, 1, 1];
  const m = new Float64Array(16);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      m[4 * j + i] = r[3 * i + j] * s[j];
    }
    m[12 + i] = m[i] * origin[0] + m[4 + i] * origin[1] + m[8 + i] * origin[2] + t[i];
  }
  m[15] = 1;
  return m;
}

/** The render position `(x, y, z)` in the world through the column-major `matrix`. */
export function worldPoint(matrix: ArrayLike<number>, x: number, y: number, z: number): Vec3 {
  const m = matrix;
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

/**
 * Calls `visit` with each live vertex (one a live triangle uses) once, in
 * ascending slot order, with its world position. Dead triangles are all
 * zeros in the render indices.
 */
export function forEachWorldVertex(
  mesh: PlacedMesh,
  visit: (slot: number, x: number, y: number, z: number) => void,
  slots: ArrayLike<number> = liveVertexIds(Math.floor(mesh.positions.length / 3), mesh.indices),
): void {
  const { positions, matrix: m } = mesh;
  const [m0, m1, m2, m4, m5, m6, m8, m9, m10, m12, m13, m14] = [
    m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10], m[12], m[13], m[14],
  ];
  for (let i = 0; i < slots.length; i++) {
    const v = slots[i];
    const x = positions[3 * v];
    const y = positions[3 * v + 1];
    const z = positions[3 * v + 2];
    visit(v, m0 * x + m4 * y + m8 * z + m12, m1 * x + m5 * y + m9 * z + m13, m2 * x + m6 * y + m10 * z + m14);
  }
}

/** The world box around the live vertices, or `null` without any. */
export function worldBounds(mesh: PlacedMesh): WorldBounds | null {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  forEachWorldVertex(mesh, (_slot, x, y, z) => {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return;
    }
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
  });
  return min[0] <= max[0] ? { min, max } : null;
}

/** The box's centre. */
export function boundsCenter(b: WorldBounds): Vec3 {
  return [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2];
}

/**
 * The placement that applies the world motion `w ↦ dR w + dT` after
 * `placement`: rotation `dR R` (kept orthonormal), translation `dR t + dT`,
 * the scale unchanged.
 */
export function moveWorld(placement: PlacementLike, dR: Readonly<Mat3>, dT: Readonly<Vec3>): PlacementInput {
  const r0 = placement.rotation as unknown as Mat3;
  // A pure move keeps the rotation bit for bit.
  const turns = dR.some((v, i) => v !== IDENTITY_MAT3[i]);
  const t = turns ? mulMat3Vec(dR, placement.translation) : [...placement.translation];
  const s = placement.scale ?? [1, 1, 1];
  return {
    rotation: turns ? orthonormalize(mulMat3(dR, r0)) : (r0.map(z0) as Mat3),
    translation: [z0(t[0] + dT[0]), z0(t[1] + dT[1]), z0(t[2] + dT[2])],
    scale: [s[0], s[1], s[2]],
  };
}

/** The translation that makes the rotation `dR` turn about `pivot`: `pivot − dR pivot`. */
function aboutPivot(dR: Readonly<Mat3>, pivot: Readonly<Vec3>): Vec3 {
  const q = mulMat3Vec(dR, pivot);
  return [pivot[0] - q[0], pivot[1] - q[1], pivot[2] - q[2]];
}

function requireBounds(mesh: PlacedMesh): WorldBounds {
  const b = worldBounds(mesh);
  if (b === null) {
    throw new Error("The mesh has no triangles");
  }
  return b;
}

/**
 * Moves the mesh so its world bounding box centre (`"center"`), or the
 * centre of its bottom face along `up` (`"bottom"`), is at the origin.
 */
export function moveToOrigin(mesh: PlacedMesh, anchor: OriginAnchor, up: UpAxis = "z"): PlacementInput {
  const b = requireBounds(mesh);
  const at = boundsCenter(b);
  if (anchor === "bottom") {
    const k = upIndex(up);
    at[k] = b.min[k];
  }
  return moveWorld(mesh.placement, IDENTITY_MAT3, [-at[0], -at[1], -at[2]]);
}

/**
 * Lays the mesh flat on a face: turns it about its world bounding box centre
 * so the face's outward world `normal` points down (against `up`), then
 * moves it along `up` until its lowest point rests on the floor.
 */
export function layFlat(mesh: PlacedMesh, normal: Readonly<Vec3>, up: UpAxis = "z"): PlacementInput {
  const b = requireBounds(mesh);
  const k = upIndex(up);
  const down: Vec3 = [0, 0, 0];
  down[k] = -1;
  const dR = rotationBetween(normal, down);
  const dT = aboutPivot(dR, boundsCenter(b));
  // The lowest point once turned.
  let low = Infinity;
  forEachWorldVertex(mesh, (_slot, x, y, z) => {
    const h = dR[3 * k] * x + dR[3 * k + 1] * y + dR[3 * k + 2] * z + dT[k];
    if (h < low) low = h;
  });
  dT[k] -= low;
  return moveWorld(mesh.placement, dR, dT);
}

/** Turns the mesh about its world bounding box centre so the world `direction` points along `target`. */
export function alignToAxis(mesh: PlacedMesh, direction: Readonly<Vec3>, target: SignedAxis): PlacementInput {
  const b = requireBounds(mesh);
  const dR = rotationBetween(direction, axisVector(target));
  return moveWorld(mesh.placement, dR, aboutPivot(dR, boundsCenter(b)));
}

/**
 * The 3-2-1 datum alignment. The three plane points give the primary plane:
 * it becomes the floor (`up` = 0) with the mesh's bounding box centre on the
 * up side. The two line points, projected onto that plane, give the
 * secondary direction: it becomes +X, and the plane through the line
 * perpendicular to the floor becomes the third axis' zero (Y = 0 with Z up,
 * Z = 0 with Y up). The origin point sets X = 0. The datum origin, where
 * the three planes meet, lands at the world origin; an origin point on the
 * secondary line and the primary plane lands there itself.
 *
 * @throws when the plane points are collinear or the line is perpendicular to the plane.
 */
export function datumAlign(mesh: PlacedMesh, picks: DatumPicks, up: UpAxis = "z"): PlacementInput {
  const b = requireBounds(mesh);
  const [p1, p2, p3] = picks.plane;
  let n = unit(cross(sub(p2, p1), sub(p3, p1)));
  if (n === null) {
    throw new Error("The three plane points are on one line");
  }
  // The mesh stands on the plane: its box centre on the up side.
  if (dot(n, sub(boundsCenter(b), p1)) < 0) {
    n = [-n[0], -n[1], -n[2]];
  }
  const [q1, q2] = picks.line;
  const d = sub(q2, q1);
  const along = dot(d, n);
  const x = unit([d[0] - along * n[0], d[1] - along * n[1], d[2] - along * n[2]]);
  if (x === null) {
    throw new Error("The two line points are along the plane's normal");
  }
  // Rows: the datum axes, each onto its world axis; right-handed either way.
  let rows: [Vec3, Vec3, Vec3];
  let anchors: [Readonly<Vec3>, Readonly<Vec3>, Readonly<Vec3>];
  if (up === "z") {
    rows = [x, cross(n, x), n];
    anchors = [picks.origin, q1, p1];
  } else {
    rows = [x, n, cross(x, n)];
    anchors = [picks.origin, p1, q1];
  }
  const dR = [...rows[0], ...rows[1], ...rows[2]].map(z0) as Mat3;
  const dT: Vec3 = [-dot(rows[0], anchors[0]), -dot(rows[1], anchors[1]), -dot(rows[2], anchors[2])];
  return moveWorld(mesh.placement, dR, dT);
}

/** The outward unit normal of the world triangle `(a, b, c)` (counter-clockwise), or `null` when it is degenerate. */
export function triangleNormal(a: Readonly<Vec3>, b: Readonly<Vec3>, c: Readonly<Vec3>): Vec3 | null {
  const n = unit(cross(sub(b, a), sub(c, a)));
  return n === null ? null : (n.map(z0) as Vec3);
}

/** Whether two placements are equal entry for entry (a missing scale is `[1, 1, 1]`). */
export function samePlacement(a: PlacementLike, b: PlacementLike): boolean {
  const sa = a.scale ?? [1, 1, 1];
  const sb = b.scale ?? [1, 1, 1];
  return (
    a.rotation.length === b.rotation.length &&
    a.rotation.every((v, i) => v === b.rotation[i]) &&
    a.translation.every((v, i) => v === b.translation[i]) &&
    sa.every((v, i) => v === sb[i])
  );
}

/** Whether `placement` is the identity: no rotation, translation or scale. */
export function isIdentityPlacement(placement: PlacementLike): boolean {
  const s = placement.scale ?? [1, 1, 1];
  return (
    placement.rotation.every((v, i) => v === IDENTITY_MAT3[i]) &&
    placement.translation.every((v) => v === 0) &&
    s.every((v) => v === 1)
  );
}

/**
 * Apply transform: the live vertices and the world positions the placement
 * puts them at, for an edit's `setPositions` under the identity placement.
 */
export function bakedPositions(mesh: PlacedMesh): { vertices: Uint32Array; positions: Float64Array } {
  const vertices = liveVertexIds(Math.floor(mesh.positions.length / 3), mesh.indices);
  const positions = new Float64Array(3 * vertices.length);
  let i = 0;
  forEachWorldVertex(
    mesh,
    (_slot, x, y, z) => {
      positions[3 * i] = x;
      positions[3 * i + 1] = y;
      positions[3 * i + 2] = z;
      i++;
    },
    vertices,
  );
  return { vertices, positions };
}
