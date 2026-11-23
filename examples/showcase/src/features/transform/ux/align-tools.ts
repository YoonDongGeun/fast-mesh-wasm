// The align panel's landmark, coarse-to-fine and symmetry tools (#253), as
// plain data and math. Landmarks are picked in pairs, a point on the moving
// mesh and then its match on the fixed one, and kept in each mesh's local
// render frame, so their markers stay on the meshes when they move. The
// symmetry plane is kept in its mesh's local frame too, and drawn as a quad
// over the mesh's bounds. The armed pick and the plane live in the app state
// (`alignTool`), shared by the panel and the viewport layer.
import type { CoarseAlignMethod, IcpObjective, LandmarkPair, MeshHandle, RegistrationReport, Vec3 } from "fast-mesh-wasm";
import { cross, dot, sub, unit, worldPoint } from "./placement-math";

/** Which mesh of a pair a landmark is on. */
export type LandmarkSide = "moving" | "fixed";

/** The landmark pairs picked so far, each point in its mesh's local render frame. */
export interface LandmarkSet {
  moving: MeshHandle;
  fixed: MeshHandle;
  movingPoints: readonly Vec3[];
  fixedPoints: readonly Vec3[];
  /** Whether canvas clicks add points (Esc stops it and keeps the points). */
  picking: boolean;
}

/** A detected symmetry plane, kept in its mesh's local render frame. */
export interface SymmetryPlaneShown {
  mesh: MeshHandle;
  /** A point on the plane (local). */
  point: Vec3;
  /** The plane's unit normal (local). */
  normal: Vec3;
  /** The RMS distance from the reflected surface to the mesh (mm) when it was detected. */
  errorRms: number;
  /** 1 for an exact mirror image, toward 0 for none. */
  score: number;
}

/** The align tools' app state: the landmark pairs and the symmetry plane on show. */
export interface AlignToolState {
  landmarks: LandmarkSet | null;
  symmetry: SymmetryPlaneShown | null;
}

export const DEFAULT_ALIGN_TOOL: AlignToolState = { landmarks: null, symmetry: null };

/** The fewest pairs a landmark alignment takes. */
export const MIN_LANDMARK_PAIRS = 3;

/** An empty set of pairs between `moving` and `fixed`, picking. */
export function startLandmarks(moving: MeshHandle, fixed: MeshHandle): LandmarkSet {
  return { moving, fixed, movingPoints: [], fixedPoints: [], picking: true };
}

/** The mesh the next click lands on: the moving mesh starts each pair, the fixed one completes it. */
export function nextLandmarkSide(set: LandmarkSet): LandmarkSide {
  return set.movingPoints.length > set.fixedPoints.length ? "fixed" : "moving";
}

/** How many pairs have both points. */
export function completePairs(set: LandmarkSet): number {
  return Math.min(set.movingPoints.length, set.fixedPoints.length);
}

/** `set` with the local point `local` added on the side the next click lands on. */
export function addLandmark(set: LandmarkSet, local: Readonly<Vec3>): LandmarkSet {
  const p: Vec3 = [local[0], local[1], local[2]];
  return nextLandmarkSide(set) === "moving"
    ? { ...set, movingPoints: [...set.movingPoints, p] }
    : { ...set, fixedPoints: [...set.fixedPoints, p] };
}

/** `set` without its latest point (the same set when it has none). */
export function removeLastLandmark(set: LandmarkSet): LandmarkSet {
  if (set.movingPoints.length > set.fixedPoints.length) {
    return { ...set, movingPoints: set.movingPoints.slice(0, -1) };
  }
  if (set.fixedPoints.length > 0) {
    return { ...set, fixedPoints: set.fixedPoints.slice(0, -1) };
  }
  return set;
}

/** What the next click should land on. */
export function landmarkPrompt(set: LandmarkSet, movingName: string, fixedName: string): string {
  const pair = set.fixedPoints.length + 1;
  return nextLandmarkSide(set) === "moving"
    ? `Pair ${pair}: click a point on the moving mesh (${movingName}).`
    : `Pair ${pair}: click the matching point on the fixed mesh (${fixedName}).`;
}

/**
 * Whether `points` lie on one line, or nearly: none is farther from the line
 * through the two ends than `tolerance` times its length.
 */
export function pointsOnALine(points: readonly Readonly<Vec3>[], tolerance = 0.01): boolean {
  if (points.length < 3) {
    return true;
  }
  const farthest = (from: Readonly<Vec3>): Readonly<Vec3> => {
    let best = from;
    let far = -1;
    for (const p of points) {
      const d = Math.hypot(...sub(p, from));
      if (d > far) {
        far = d;
        best = p;
      }
    }
    return best;
  };
  const a = farthest(points[0]);
  const b = farthest(a);
  const along = unit(sub(b, a));
  if (along === null) {
    return true; // one spot, picked again and again
  }
  const length = Math.hypot(...sub(b, a));
  return points.every((p) => Math.hypot(...cross(sub(p, a), along)) <= tolerance * length);
}

/**
 * Why the complete pairs cannot fix a pose, or `null`: points on a line leave
 * the turn about it open, and the fit's RMSE is 0 whichever turn it takes.
 */
export function landmarksProblem(set: LandmarkSet): string | null {
  const n = completePairs(set);
  if (n < MIN_LANDMARK_PAIRS) {
    return null;
  }
  for (const [side, points] of [
    ["moving", set.movingPoints],
    ["fixed", set.fixedPoints],
  ] as const) {
    if (pointsOnALine(points.slice(0, n))) {
      return `The points on the ${side} mesh lie on a line, so the turn about it is open. Add a pair off the line.`;
    }
  }
  return null;
}

/** The complete pairs as world points, through each mesh's column-major world matrix. */
export function landmarkPairs(
  set: LandmarkSet,
  movingMatrix: ArrayLike<number>,
  fixedMatrix: ArrayLike<number>,
): LandmarkPair[] {
  const n = completePairs(set);
  const out: LandmarkPair[] = [];
  for (let i = 0; i < n; i++) {
    const m = set.movingPoints[i];
    const f = set.fixedPoints[i];
    out.push({
      source: worldPoint(movingMatrix, m[0], m[1], m[2]),
      target: worldPoint(fixedMatrix, f[0], f[1], f[2]),
    });
  }
  return out;
}

/** Each complete pair's world distance at the given world matrices. */
export function landmarkResiduals(
  set: LandmarkSet,
  movingMatrix: ArrayLike<number>,
  fixedMatrix: ArrayLike<number>,
): number[] {
  return landmarkPairs(set, movingMatrix, fixedMatrix).map(({ source, target }) =>
    Math.hypot(source[0] - target[0], source[1] - target[1], source[2] - target[2]),
  );
}

/** The root mean square of `values` (0 for none). */
export function rms(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  let s = 0;
  for (const v of values) {
    s += v * v;
  }
  return Math.sqrt(s / values.length);
}

// ---- Coarse + ICP ----

export const COARSE_METHODS: readonly (readonly [Exclude<CoarseAlignMethod, "none">, string])[] = [
  ["orientedBoundingBox", "Oriented bounding box"],
  ["pcaCentroid", "Principal axes"],
];

export const ICP_OBJECTIVES: readonly (readonly [IcpObjective, string])[] = [
  ["pointToPlane", "Point to plane"],
  ["pointToPoint", "Point to point"],
  ["symmetricNormal", "Symmetric"],
];

/** One stage of a coarse + ICP run, for the report. */
export interface AlignStage {
  label: string;
  iterations: number;
  /** The trimmed RMS distance after the stage (mm; `Infinity` without correspondences). */
  rmse: number;
  /** Why the ICP stopped (`null` for the coarse stage). */
  status: RegistrationReport["status"] | null;
}

/** The report rows of a coarse stage (`maxIterations: 0`) and the ICP stage after it. */
export function multistageStages(
  coarseMethod: CoarseAlignMethod,
  coarse: Pick<RegistrationReport, "finalRmse">,
  fine: Pick<RegistrationReport, "finalRmse" | "iterations" | "status">,
): AlignStage[] {
  const label = COARSE_METHODS.find(([m]) => m === coarseMethod)?.[1] ?? "Coarse";
  return [
    { label: `Coarse (${label.toLowerCase()})`, iterations: 0, rmse: coarse.finalRmse, status: null },
    { label: "ICP", iterations: fine.iterations, rmse: fine.finalRmse, status: fine.status },
  ];
}

/** What to check after a coarse + ICP run, or `null` when the ICP converged. */
export function multistageWarning(
  coarse: Pick<RegistrationReport, "finalRmse">,
  fine: Pick<RegistrationReport, "finalRmse" | "status">,
): string | null {
  if (!Number.isFinite(coarse.finalRmse) && !Number.isFinite(fine.finalRmse)) {
    return "No point of the moving mesh is within reach of the fixed one, so nothing moved.";
  }
  switch (fine.status) {
    case "converged":
      return null;
    case "maxIterations":
      return "The ICP reached the iteration limit before it settled.";
    case "underconstrained":
      return "The shapes do not determine the pose: the mesh may slide or spin with the same fit.";
    case "tooFewCorrespondences":
      return "Too few points are within reach: the pose is arbitrary, whatever its RMSE.";
    case "diverged":
      return "The ICP diverged.";
  }
}

// ---- Symmetry plane ----

/** The inverse of the column-major affine `matrix` (its 3x3 part must be invertible). */
export function invertAffine(matrix: ArrayLike<number>): Float64Array {
  const m = matrix;
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!(Math.abs(det) > 1e-300) || !Number.isFinite(det)) {
    throw new Error("The mesh's placement cannot be inverted");
  }
  // Row-major inverse of [[a b c] [d e f] [g h i]].
  const r = [
    A / det,
    (c * h - b * i) / det,
    (b * f - c * e) / det,
    B / det,
    (a * i - c * g) / det,
    (c * d - a * f) / det,
    C / det,
    (b * g - a * h) / det,
    (a * e - b * d) / det,
  ];
  const t = [m[12], m[13], m[14]];
  const out = new Float64Array(16);
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      out[4 * col + row] = r[3 * row + col];
    }
    out[12 + row] = -(r[3 * row] * t[0] + r[3 * row + 1] * t[1] + r[3 * row + 2] * t[2]);
  }
  out[15] = 1;
  return out;
}

/** A normal through the column-major `matrix` (by its inverse transpose), made unit; `null` when it vanishes. */
function transformNormal(inverse: ArrayLike<number>, n: Readonly<Vec3>): Vec3 | null {
  const q = inverse;
  // (M⁻¹)ᵀ n: column k of M⁻¹ dotted with n.
  return unit([
    q[0] * n[0] + q[1] * n[1] + q[2] * n[2],
    q[4] * n[0] + q[5] * n[1] + q[6] * n[2],
    q[8] * n[0] + q[9] * n[1] + q[10] * n[2],
  ]);
}

/** The plane (`point`, `normal`) carried by the column-major affine `matrix`. */
export function transformPlane(
  matrix: ArrayLike<number>,
  point: Readonly<Vec3>,
  normal: Readonly<Vec3>,
): { point: Vec3; normal: Vec3 } {
  const n = transformNormal(invertAffine(matrix), normal);
  if (n === null) {
    throw new Error("The plane has no normal");
  }
  return { point: worldPoint(matrix, point[0], point[1], point[2]), normal: n };
}

/** A world plane in the local frame of the mesh whose world matrix is `matrix`. */
export function planeToLocal(
  matrix: ArrayLike<number>,
  point: Readonly<Vec3>,
  normal: Readonly<Vec3>,
): { point: Vec3; normal: Vec3 } {
  return transformPlane(invertAffine(matrix), point, normal);
}

/** A local plane in the world, through the mesh's world `matrix`. */
export function planeToWorld(
  matrix: ArrayLike<number>,
  point: Readonly<Vec3>,
  normal: Readonly<Vec3>,
): { point: Vec3; normal: Vec3 } {
  return transformPlane(matrix, point, normal);
}

/** A quad on a plane: its centre, two unit in-plane axes and its half sizes along them. */
export interface PlaneQuad {
  center: Vec3;
  u: Vec3;
  v: Vec3;
  halfU: number;
  halfV: number;
}

/** Two unit axes spanning the plane with unit normal `n` (`u × v = n`). */
export function planeAxes(n: Readonly<Vec3>): [Vec3, Vec3] {
  const ax = Math.abs(n[0]);
  const ay = Math.abs(n[1]);
  const az = Math.abs(n[2]);
  const e: Vec3 = ax <= ay && ax <= az ? [1, 0, 0] : ay <= az ? [0, 1, 0] : [0, 0, 1];
  const u = unit(cross(e, n)) ?? [1, 0, 0];
  return [u, cross(n, u)];
}

/**
 * The quad on the plane (`point`, unit `normal`) that covers the box
 * `min`..`max` seen along the normal, grown by `margin` (a fraction of
 * each half size): centred on the box centre's foot on the plane.
 */
export function planeQuad(
  point: Readonly<Vec3>,
  normal: Readonly<Vec3>,
  min: Readonly<Vec3>,
  max: Readonly<Vec3>,
  margin = 0.08,
): PlaneQuad {
  const [u, v] = planeAxes(normal);
  const mid: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const off = dot(sub(mid, point), normal);
  const center: Vec3 = [mid[0] - off * normal[0], mid[1] - off * normal[1], mid[2] - off * normal[2]];
  let halfU = 0;
  let halfV = 0;
  for (let k = 0; k < 8; k++) {
    const corner: Vec3 = [k & 1 ? max[0] : min[0], k & 2 ? max[1] : min[1], k & 4 ? max[2] : min[2]];
    const d = sub(corner, center);
    halfU = Math.max(halfU, Math.abs(dot(d, u)));
    halfV = Math.max(halfV, Math.abs(dot(d, v)));
  }
  return { center, u, v, halfU: halfU * (1 + margin), halfV: halfV * (1 + margin) };
}

/** The quad's four corners, in order around it. */
export function quadCorners(q: PlaneQuad): [Vec3, Vec3, Vec3, Vec3] {
  const at = (su: number, sv: number): Vec3 => [
    q.center[0] + su * q.halfU * q.u[0] + sv * q.halfV * q.v[0],
    q.center[1] + su * q.halfU * q.u[1] + sv * q.halfV * q.v[1],
    q.center[2] + su * q.halfU * q.u[2] + sv * q.halfV * q.v[2],
  ];
  return [at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)];
}

/** The plane's normal for the report, its signed zeros and rounding noise cleared. */
export function formatNormal(n: Readonly<Vec3>): string {
  const f = (x: number) => {
    const r = Math.round(x * 1000) / 1000;
    return (r === 0 ? 0 : r).toFixed(3);
  };
  return `(${f(n[0])}, ${f(n[1])}, ${f(n[2])})`;
}

/** The align commands the command palette offers, by the panel button they press. */
export const ALIGN_COMMANDS: readonly { action: string; label: string; keywords: string; meshes: 1 | 2 }[] = [
  {
    action: "landmarks-pick",
    label: "Align: Pick Landmark Pairs",
    keywords: "align landmark pairs points pick correspond register",
    meshes: 2,
  },
  {
    action: "multistage-run",
    label: "Align: Coarse + ICP",
    keywords: "align register coarse icp multistage bounding box principal axes",
    meshes: 2,
  },
  {
    action: "symmetry-detect",
    label: "Align: Detect Symmetry Plane",
    keywords: "align symmetry plane mirror reflect detect",
    meshes: 1,
  },
];
