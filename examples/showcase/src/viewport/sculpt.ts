// The sculpt stroke: what one dab applies, and one press-and-hold stroke
// over the meshes (the viewport's ring and pointer loop drive it).
import type {
  FastMeshErrorCode,
  GrabMethod,
  InflateMethod,
  MeshHandle,
  RepairOp,
  SmoothMethod,
  SurfaceFilter,
  Symmetry,
  TwistMethod,
} from "fast-mesh-wasm";
import type {
  SculptFilterKind,
  SculptFilterSettings,
  SculptMethods,
  SculptSymmetrySettings,
} from "../panels/settings";
import type { MaskField } from "./mask-protect";

export type SculptOpName = "inflate" | "smooth" | "relax" | "grab" | "flatten" | "crease" | "clay" | "twist";

/** The toolbox's brush picker (the viewport paints its strokes). */
export type SculptTool =
  | "inflate"
  | "deflate"
  | "smooth"
  | "grab"
  | "flatten"
  | "scrape"
  | "fill"
  | "crease"
  | "pinch"
  | "clay"
  | "clayStrips"
  | "twist";

/** A sculpt brush armed on a mesh (app-state `sculpt`). */
export interface SculptBrush {
  mesh: MeshHandle;
  tool?: SculptTool;
  op: SculptOpName;
  radius: number;
  strength: number;
  falloff: "linear" | "smooth";
  cycles: number;
  /** The grab drag vector in local coordinates (the grab tool only). */
  direction: [number, number, number];
  methods?: SculptMethods;
  grabPoisson?: number;
  filter?: SculptFilterSettings;
  symmetry?: SculptSymmetrySettings;
}

const DEFAULT_SCULPT_METHODS: SculptMethods = {
  inflate: "vertexNormal",
  smooth: "laplacian",
  grab: "sphere",
  twist: "vortex",
};

export interface MethodOption<M extends string> {
  value: M;
  label: string;
  description: string;
}

export const INFLATE_METHOD_OPTIONS: readonly MethodOption<InflateMethod>[] = [
  { value: "vertexNormal", label: "Vertex normal", description: "Displace each vertex along its live smooth normal" },
  { value: "averageNormal", label: "Average normal", description: "Push uniformly along the brush's falloff-weighted normal" },
  { value: "strokeNormal", label: "Stroke normal", description: "Follow frozen pre-stroke normals to prevent folding" },
];

export const SMOOTH_METHOD_OPTIONS: readonly MethodOption<SmoothMethod>[] = [
  { value: "laplacian", label: "Laplacian", description: "Fast uniform umbrella relaxation that melts bumps quickly" },
  { value: "taubin", label: "Taubin", description: "Two-step λ/μ low-pass filter that preserves volume" },
  { value: "hcLaplacian", label: "HC-Laplacian", description: "Humphrey-class pushback toward original form" },
  { value: "cotangent", label: "Cotangent", description: "Mean-curvature cotangent weights independent of triangulation" },
];

export const GRAB_METHOD_OPTIONS: readonly MethodOption<GrabMethod>[] = [
  { value: "sphere", label: "Sphere", description: "Euclidean brush sphere with fold-safe adaptive reach" },
  { value: "geodesic", label: "Geodesic", description: "Surface-connected geodesic distance that respects gaps" },
  { value: "kelvinlet", label: "Kelvinlet", description: "Regularized elastic Green's function deformation" },
];

export const TWIST_METHOD_OPTIONS: readonly MethodOption<TwistMethod>[] = [
  { value: "vortex", label: "Vortex", description: "Rigid falloff-weighted Rodrigues rotation around brush normal" },
  { value: "kelvinlet", label: "Kelvinlet", description: "Elastic rotational Kelvinlet swirl with smooth shear" },
];

export const FILTER_KIND_OPTIONS: readonly MethodOption<SculptFilterKind>[] = [
  { value: "none", label: "None", description: "Affect all surface features inside the brush" },
  { value: "peaksOnly", label: "Peaks only", description: "Mask brush to convex ridges and outward bumps" },
  { value: "valleysOnly", label: "Valleys only", description: "Mask brush to concave grooves and cavities" },
  { value: "normalAngle", label: "Normal angle", description: "Restrict brush to faces within a normal cone" },
];

/**
 * The repair op a tool paints (deflate is inflate with a flipped strength;
 * scrape/fill are flatten peak/valley modes; pinch is zero-normal crease).
 */
export function sculptOpForTool(
  tool: SculptTool,
): "inflate" | "smooth" | "grab" | "flatten" | "crease" | "clay" | "twist" {
  switch (tool) {
    case "inflate":
    case "deflate":
      return "inflate";
    case "smooth":
      return "smooth";
    case "grab":
      return "grab";
    case "flatten":
    case "scrape":
    case "fill":
      return "flatten";
    case "crease":
    case "pinch":
      return "crease";
    case "clay":
    case "clayStrips":
      return "clay";
    case "twist":
      return "twist";
  }
}

/**
 * Each tool's starting strength (grab at 1 follows the pointer exactly).
 * Inflate's is the height of one dab over the brush radius (see `dabOperation`),
 * so a stroke looks the same at any model size.
 */
export const DEFAULT_STRENGTHS: Record<SculptTool, number> = {
  inflate: 0.1,
  deflate: 0.1,
  smooth: 0.5,
  grab: 1,
  flatten: 0.5,
  scrape: 0.5,
  fill: 0.5,
  crease: 0.15,
  pinch: 0.5,
  clay: 0.5,
  clayStrips: 0.5,
  twist: 0.5,
};

/** Laplacian passes per smooth dab (each melts the full weight toward the centroid). */
export const RELAX_PASSES = 2;

export interface SculptBrushExtras {
  methods?: SculptMethods;
  grabPoisson?: number;
  filter?: SculptFilterSettings;
  symmetry?: SculptSymmetrySettings;
}

/**
 * Arms (or steers) the sculpt brush for `tool` with a dab-sized strength.
 * The toolbox calls it live, so the radius/strength sliders stay adjustable
 * mid-stroke (the stroke reads its brush through a getter).
 */
export function sculptBrush(
  tool: SculptTool,
  mesh: MeshHandle,
  radius: number,
  strength: number,
  falloff: "linear" | "smooth",
  extras?: SculptBrushExtras,
): SculptBrush {
  return {
    mesh,
    tool,
    op: sculptOpForTool(tool),
    radius,
    strength: tool === "deflate" ? -Math.abs(strength) : Math.abs(strength),
    falloff,
    cycles: RELAX_PASSES,
    direction: [0, 0, 0],
    ...(extras?.methods !== undefined ? { methods: { ...extras.methods } } : {}),
    ...(extras?.grabPoisson !== undefined ? { grabPoisson: extras.grabPoisson } : {}),
    ...(extras?.filter !== undefined ? { filter: { ...extras.filter } } : {}),
    ...(extras?.symmetry !== undefined ? { symmetry: { ...extras.symmetry } } : {}),
  };
}

/**
 * Past this many brush radii between two pointer samples the stroke jumps
 * instead of laying dabs along the way: that is the pointer leaving one part
 * of the surface for another, not a fast move (see `SculptStroke.dab`).
 */
export const TELEPORT_RADII = 8;

/**
 * Past this many triangles the optional "even out" remesh stands down: it
 * runs over the whole mesh (measured ~2.6 s floor on 300 k, minutes for wide
 * regions), which would stall the worker and the dabs behind it. Dabs cost by
 * the brush region, and the stretch splits by the stretched edges: neither has
 * such a limit.
 */
export const REMESH_TRI_LIMIT = 100_000;

/** Whether `triangles` stands the optional stroke-end remesh down. */
export function remeshTooBig(triangles: number): boolean {
  return triangles > REMESH_TRI_LIMIT;
}

/**
 * Stretch splits while sculpting (#129, #175): the per-frame budget.
 *
 * At most one live pass per stroke region per `LIVE_REMESH_MIN_MS`, and only
 * once `LIVE_REMESH_MIN_DABS` dabs landed since the last pass, never two in
 * flight. A pass splits only the edges the stroke stretched past their limit
 * (`splitStretched`): what the stroke did not stretch keeps its vertices and
 * triangles, and the vertex ids a mask stores stay valid. Each stroke asks for
 * rest sizes before its first dab (`captureRestSizes`, `onlyIfMissing`), so
 * repeated strokes measure against the same shape instead of ratcheting, until
 * an edit outdates it and the engine drops the sizes.
 */
export const LIVE_REMESH_MIN_DABS = 4;
/** Minimum milliseconds between two live remesh passes of one stroke. */
export const LIVE_REMESH_MIN_MS = 150;
/** Most interior splits one pass makes; a later pass takes the rest, longest first. */
export const LIVE_MAX_SPLITS = 50_000;
/** Per-pass split cap above `REMESH_TRI_LIMIT`, avoiding 50k-split cascades on 1M meshes. */
export const LIVE_HUGE_MAX_SPLITS = 2_000;

/** Max splits per live pass for `triangles` (`LIVE_MAX_SPLITS` up to 100k, capped above). */
export function liveMaxSplits(triangles: number | null): number {
  return triangles !== null && remeshTooBig(triangles) ? LIVE_HUGE_MAX_SPLITS : LIVE_MAX_SPLITS;
}

/**
 * Whether a live remesh pass may fire: enough new dabs since the last pass,
 * enough time since it, and none still in flight.
 */
export function liveRemeshDue(dabsSinceLive: number, msSinceLive: number, running: boolean): boolean {
  return !running && dabsSinceLive >= LIVE_REMESH_MIN_DABS && msSinceLive >= LIVE_REMESH_MIN_MS;
}

/**
 * Undo semantics for sculpt remeshes (the #129 open question): SAME step.
 *
 * Both the live passes and the stroke-end evening-out land inside the
 * stroke's open undo step on the sync scene, so one Ctrl+Z takes the whole
 * gesture (dabs and density fixes) back. A separate remesh step would fight
 * the stroke: undoing the stroke would leave its splits behind, and undoing
 * the remesh first would strand the dabs on a density they never painted.
 * The live pass must therefore never open or close a step itself; it just
 * repairs while the stroke's step is open. (On the Worker there is no step
 * nesting — one step per batch of dabs — so the passes land as their own steps
 * there; that divergence is inherent to the Worker scene, not a choice.)
 */

/**
 * Dab spacing: 5% of the radius (never zero, so dabs terminate). Close enough
 * that one dab is too small to see on its own, so a stroke grows instead of stepping.
 */
export function dabSpacing(radius: number): number {
  return Math.max(radius * 0.05, 1e-6);
}

/** What one spaced dab applies of the brush's strength: the same bump per distance, in finer steps. */
export const DAB_SHARE = 1 / 3;

/** Whether `at` is at least `spacing` from the last dab: a new dab. */
export function farEnough(
  last: readonly [number, number, number],
  at: readonly [number, number, number],
  spacing: number,
): boolean {
  const dx = at[0] - last[0];
  const dy = at[1] - last[1];
  const dz = at[2] - last[2];
  return dx * dx + dy * dy + dz * dz >= spacing * spacing;
}

/** Converts persisted filter settings into an engine `SurfaceFilter` (`undefined` when `"none"`). */
export function toSurfaceFilter(filter?: SculptFilterSettings): SurfaceFilter | undefined {
  if (filter === undefined || filter.kind === "none") {
    return undefined;
  }
  if (filter.kind === "peaksOnly" || filter.kind === "valleysOnly") {
    return {
      kind: filter.kind,
      threshold: filter.threshold,
      contrast: filter.contrast,
    };
  }
  return {
    kind: "normalAngle",
    maxAngleDeg: filter.maxAngleDeg,
  };
}

/** Converts persisted symmetry settings into an engine `Symmetry` (`undefined` when all off). */
export function toSymmetry(symmetry?: SculptSymmetrySettings): Symmetry | undefined {
  if (symmetry === undefined || (!symmetry.x && !symmetry.y && !symmetry.z)) {
    return undefined;
  }
  return { x: symmetry.x, y: symmetry.y, z: symmetry.z };
}

/** Active mirror-reflected centers across local `x = 0`, `y = 0`, `z = 0` planes. */
export function reflectCenters(
  center: readonly [number, number, number],
  symmetry?: Symmetry,
): [number, number, number][] {
  const [cx, cy, cz] = center;
  const sx = symmetry?.x === true;
  const sy = symmetry?.y === true;
  const sz = symmetry?.z === true;
  const out: [number, number, number][] = [];
  for (let bits = 0; bits < 8; bits++) {
    if (((bits & 1) !== 0 && !sx) || ((bits & 2) !== 0 && !sy) || ((bits & 4) !== 0 && !sz)) {
      continue;
    }
    const rx = (bits & 1) !== 0 ? (cx === 0 ? 0 : -cx) : cx;
    const ry = (bits & 2) !== 0 ? (cy === 0 ? 0 : -cy) : cy;
    const rz = (bits & 4) !== 0 ? (cz === 0 ? 0 : -cz) : cz;
    if (!out.some((p) => p[0] === rx && p[1] === ry && p[2] === rz)) {
      out.push([rx, ry, rz]);
    }
  }
  return out;
}

/**
 * Maps an armed `SculptBrush` (including algorithm, filter, symmetry, and optional
 * stroke tangent) to the exact `RepairOp` / `SculptOp` payload for one dab.
 */
export function dabOperation(
  brush: SculptBrush,
  center: readonly [number, number, number],
  share = 1,
  tangent?: readonly [number, number, number],
): RepairOp {
  const c: [number, number, number] = [center[0], center[1], center[2]];
  if (brush.tool === undefined) {
    const shared = {
      center: c,
      radius: brush.radius,
      falloff: brush.falloff,
      strength: (brush.op === "inflate" ? brush.strength * brush.radius : brush.strength) * share,
    };
    if (brush.op === "relax") {
      return { op: "relax", ...shared, cycles: brush.cycles };
    }
    if (brush.op === "grab") {
      return { op: "grab", ...shared, direction: [...brush.direction] as [number, number, number] };
    }
    return { op: "inflate", ...shared };
  }

  const methods = brush.methods ?? DEFAULT_SCULPT_METHODS;
  const filter = toSurfaceFilter(brush.filter);
  const symmetry = toSymmetry(brush.symmetry);
  const mods = {
    ...(filter !== undefined ? { filter } : {}),
    ...(symmetry !== undefined ? { symmetry } : {}),
  };

  switch (brush.tool) {
    case "inflate":
    case "deflate":
      return {
        op: "inflate",
        center: c,
        radius: brush.radius,
        falloff: brush.falloff,
        strength: brush.strength * brush.radius * share,
        method: methods.inflate,
        ...mods,
      };
    case "smooth":
      return {
        op: "smooth",
        center: c,
        radius: brush.radius,
        falloff: brush.falloff,
        strength: Math.abs(brush.strength) * share,
        cycles: brush.cycles,
        method: methods.smooth,
        ...mods,
      };
    case "grab":
      return {
        op: "grab",
        center: c,
        radius: brush.radius,
        falloff: brush.falloff,
        strength: brush.strength * share,
        direction: [brush.direction[0], brush.direction[1], brush.direction[2]],
        method: methods.grab,
        ...(methods.grab === "kelvinlet" ? { poisson: brush.grabPoisson ?? 0.4 } : {}),
        fromStroke: true,
        adaptiveReach: true,
        ...mods,
      };
    case "flatten":
    case "scrape":
    case "fill":
      return {
        op: "flatten",
        center: c,
        radius: brush.radius,
        falloff: brush.falloff,
        strength: Math.abs(brush.strength) * share,
        mode: brush.tool === "scrape" ? "peaks" : brush.tool === "fill" ? "valleys" : "both",
        ...mods,
      };
    case "crease":
      return {
        op: "crease",
        center: c,
        radius: brush.radius,
        falloff: brush.falloff,
        strength: -Math.abs(brush.strength) * brush.radius * share,
        pinch: Math.min(1, 0.5 * share),
        ...mods,
      };
    case "pinch":
      return {
        op: "crease",
        center: c,
        radius: brush.radius,
        falloff: brush.falloff,
        strength: 0,
        pinch: Math.min(1, Math.abs(brush.strength) * share),
        ...mods,
      };
    case "clay":
    case "clayStrips":
      return {
        op: "clay",
        center: c,
        radius: brush.radius,
        falloff: brush.falloff,
        strength: Math.abs(brush.strength) * share,
        profile: brush.tool === "clayStrips" ? "strips" : "round",
        ...(brush.tool === "clayStrips" && tangent !== undefined
          ? { tangent: [tangent[0], tangent[1], tangent[2]] as [number, number, number] }
          : {}),
        ...mods,
      };
    case "twist":
      return {
        op: "twist",
        center: c,
        radius: brush.radius,
        falloff: brush.falloff,
        strength: brush.strength * share,
        method: methods.twist,
        ...mods,
      };
  }
}

/**
 * The repair op one dab applies at `center` (local coordinates): the
 * displacement step of a painted stroke.
 */
export function dabOp(
  brush: SculptBrush,
  center: readonly [number, number, number],
  share = 1,
  tangent?: readonly [number, number, number],
): RepairOp {
  return dabOperation(brush, center, share, tangent);
}

/** The unit face normal of the triangle `(a, b, c)` over flat positions. */
export function faceNormal(positions: ArrayLike<number>, a: number, b: number, c: number): [number, number, number] {
  const ux = positions[3 * b] - positions[3 * a];
  const uy = positions[3 * b + 1] - positions[3 * a + 1];
  const uz = positions[3 * b + 2] - positions[3 * a + 2];
  const vx = positions[3 * c] - positions[3 * a];
  const vy = positions[3 * c + 1] - positions[3 * a + 1];
  const vz = positions[3 * c + 2] - positions[3 * a + 2];
  const n: [number, number, number] = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
  const len = Math.hypot(n[0], n[1], n[2]);
  return len === 0 ? [0, 0, 1] : [n[0] / len, n[1] / len, n[2] / len];
}

/** World units per CSS pixel at `distance` from a perspective camera. */
export function worldPerPixel(distance: number, fovDegrees: number, canvasCssHeight: number): number {
  return (2 * distance * Math.tan((fovDegrees * Math.PI) / 360)) / canvasCssHeight;
}

/** Rotates `v` by the unit quaternion `q` ([x, y, z, w]). */
function rotateByQuat(
  v: readonly [number, number, number],
  q: readonly [number, number, number, number],
): [number, number, number] {
  const [qx, qy, qz, qw] = q;
  // t = 2 * cross(q.xyz, v); v + qw * t + cross(q.xyz, t).
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  return [
    v[0] + qw * tx + (qy * tz - qz * ty),
    v[1] + qw * ty + (qz * tx - qx * tz),
    v[2] + qw * tz + (qx * ty - qy * tx),
  ];
}

/**
 * A grab dab's drag vector in the mesh's local frame: the screen delta
 * (`pixelDx`, `pixelDy`, y down) at `wpp` world units per pixel, along the
 * camera's `camRight`/`camUp`, carried by `meshQuatConj` (the conjugate of
 * the mesh's world quaternion).
 */
export function grabLocalDelta(
  pixelDx: number,
  pixelDy: number,
  wpp: number,
  camRight: readonly [number, number, number],
  camUp: readonly [number, number, number],
  meshQuatConj: readonly [number, number, number, number],
): [number, number, number] {
  const sx = pixelDx * wpp;
  const sy = -pixelDy * wpp;
  const world: [number, number, number] = [
    camRight[0] * sx + camUp[0] * sy,
    camRight[1] * sx + camUp[1] * sy,
    camRight[2] * sx + camUp[2] * sy,
  ];
  return rotateByQuat(world, meshQuatConj);
}

/** The falloff weight at normalized distance `t`, as the engine's brushes compute it. */
export function falloffWeight(falloff: "linear" | "smooth", t: number): number {
  if (!(t < 1)) {
    return 0;
  }
  const s = 1 - t;
  return falloff === "linear" ? s : s * s * (3 - 2 * s);
}

/**
 * The most a grab may stretch the surface along its drag: its displacement grows
 * by at most this much per unit of distance. Under 1, points moving along the drag
 * cannot pass each other, so the surface cannot fold over itself.
 */
export const GRAB_MAX_STRETCH = 0.8;

/** The steepest slope of each falloff over the normalized distance. */
const FALLOFF_SLOPE = { linear: 1, smooth: 1.5 } as const;

/**
 * O(1) upper bound on a fold-safe adaptive grab's influence radius (matching
 * `grab_widest_reach` in `fast-mesh-ops`), for the cursor ring and live-split
 * bounds without scanning mesh vertices on the main thread (#197).
 */
export function widestGrabReach(
  radius: number,
  falloff: "linear" | "smooth",
  method: GrabMethod,
  strength: number,
  drag: readonly [number, number, number],
): number {
  const slope = method === "kelvinlet" ? 1.75 : FALLOFF_SLOPE[falloff];
  const len = Math.abs(strength) * Math.hypot(drag[0], drag[1], drag[2]);
  return Math.max(radius, (slope * len) / GRAB_MAX_STRETCH);
}

/**
 * What a grab stroke holds on to: where the vertices rested at the press. Each
 * move is written from these and the total drag, so the shape depends on where
 * the pointer is and not on how many events it took to get there. A vertex moves
 * by `strength * falloff(distance / influence) * drag`.
 *
 * A surface folds where it lies ahead of the press along the drag, and the front
 * of the lump runs over it. So the influence is the brush radius while nothing
 * lies ahead, as in pulling straight off the surface, which may go as far as it
 * likes. With surface ahead, as in dragging along it, the influence widens with
 * the drag just enough to keep the stretch under `GRAB_MAX_STRETCH`, and takes more
 * of the surface along. The vertex at the press follows the pointer either way.
 */
export class GrabRegion {
  /** The vertex slots the last move wrote. */
  ids: Uint32Array;
  /** Where each slot rests: three numbers per slot. */
  #rest: Float32Array;
  /** Which slots a live triangle uses. */
  #live: Uint8Array;
  /** What the mask leaves of each slot's move. */
  #scale: Float32Array;
  /** The vertices a pass added, in the order it made them: `new, a, b`. Each moves as the mean of its edge. */
  #splits: number[] = [];
  /** Which slots are in `#splits`. */
  #isSplit: Uint8Array;
  /** Which slots a move has written, so that a shrinking influence puts them back. */
  #touched: Uint8Array;
  /** The vertices the influence could reach, without the masked ones. */
  #near: Uint32Array;
  /** How far `#near` goes. */
  #reach = 0;
  /** The influence of the last move. */
  #influence: number;
  /** The widest influence so far. */
  #peak: number;
  readonly #center: readonly [number, number, number];
  readonly #radius: number;
  readonly #falloff: "linear" | "smooth";
  readonly #strength: number;
  readonly #origin: readonly [number, number, number];
  readonly #hold: () => MaskField | null;
  /** The drag of the last `at`: what the mesh shows, once it landed. */
  #drag: readonly [number, number, number] = [0, 0, 0];

  constructor(
    positions: ArrayLike<number>,
    indices: ArrayLike<number>,
    center: readonly [number, number, number],
    radius: number,
    falloff: "linear" | "smooth",
    strength: number,
    origin: readonly [number, number, number] = [0, 0, 0],
    hold: MaskField | null | (() => MaskField | null) = null,
  ) {
    this.#origin = [origin[0], origin[1], origin[2]];
    this.#hold = typeof hold === "function" ? hold : () => hold;
    this.#center = [center[0], center[1], center[2]];
    this.#radius = radius;
    this.#influence = radius;
    this.#peak = radius;
    this.#falloff = falloff;
    this.#strength = strength;
    this.ids = new Uint32Array();
    this.#near = new Uint32Array();
    const slots = positions.length / 3;
    this.#rest = Float32Array.from(positions as ArrayLike<number>);
    this.#live = liveSlots(indices, slots);
    this.#touched = new Uint8Array(slots);
    this.#isSplit = new Uint8Array(slots);
    this.#scale = new Float32Array(slots);
    const field = this.#hold();
    for (let v = 0; v < slots; v++) {
      this.#scale[v] = this.#held(v, field);
    }
    this.#take(radius);
    this.at([0, 0, 0]);
  }

  /** The longest reach a drag this long could need: the influence that holds whatever lies ahead. */
  #widest(drag: readonly [number, number, number]): number {
    const length = Math.abs(this.#strength) * Math.hypot(drag[0], drag[1], drag[2]);
    return (FALLOFF_SLOPE[this.#falloff] * length) / GRAB_MAX_STRETCH;
  }

  /**
   * The radius a `drag` reaches: the brush's, or wider by how much surface lies
   * ahead of the press along the drag. A vertex counts by how straight ahead it
   * lies, and less toward the edge of the widest reach, so that the influence
   * grows smoothly as the drag brings more of the surface in.
   */
  influence(drag: readonly [number, number, number]): number {
    const widest = this.#widest(drag);
    if (widest <= this.#radius) {
      return this.#radius;
    }
    if (widest > this.#reach) {
      // Well past it, so that a long drag scans the mesh only a few times.
      this.#take(widest * 1.25);
    }
    const length = Math.hypot(drag[0], drag[1], drag[2]);
    const sign = Math.sign(this.#strength);
    const [ux, uy, uz] = [(sign * drag[0]) / length, (sign * drag[1]) / length, (sign * drag[2]) / length];
    const [cx, cy, cz] = this.#center;
    const rest = this.#rest;
    let ahead = 0;
    for (let k = 0; k < this.#near.length; k++) {
      const v = this.#near[k];
      const dx = rest[3 * v] - cx;
      const dy = rest[3 * v + 1] - cy;
      const dz = rest[3 * v + 2] - cz;
      const along = dx * ux + dy * uy + dz * uz;
      if (along <= 0) {
        continue;
      }
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (distance < widest) {
        ahead = Math.max(ahead, (along / distance) * Math.min(1, 2 * (1 - distance / widest)));
      }
    }
    return Math.max(this.#radius, widest * ahead);
  }

  /** How far from the press the grab has reached so far. */
  get reach(): number {
    return this.#peak;
  }

  /** What the mask leaves of a vertex's move: 0 on a masked one. A slot newer than the field is free. */
  #held(id: number, field: MaskField | null): number {
    return field === null || id >= field.scale.length ? 1 : field.scale[id];
  }

  /** The vertices out to `reach`, without the masked ones: no write ever reaches those. */
  #take(reach: number): void {
    this.#reach = reach;
    const [cx, cy, cz] = this.#center;
    const rest = this.#rest;
    const reach2 = reach * reach;
    const near: number[] = [];
    for (let v = 0; v < this.#live.length; v++) {
      if (this.#live[v] === 0 || this.#scale[v] === 0) {
        continue;
      }
      const dx = rest[3 * v] - cx;
      const dy = rest[3 * v + 1] - cy;
      const dz = rest[3 * v + 2] - cz;
      if (dx * dx + dy * dy + dz * dz <= reach2) {
        near.push(v);
      }
    }
    this.#near = Uint32Array.from(near);
  }

  /** The share of the drag a vertex resting at `x, y, z` takes by its falloff alone. */
  #weight(x: number, y: number, z: number): number {
    const c = this.#center;
    return this.#strength * falloffWeight(this.#falloff, Math.hypot(x - c[0], y - c[1], z - c[2]) / this.#influence);
  }

  /** The share of the drag each split vertex takes: the mean of its edge, in the order they were made. */
  #splitWeights(): Map<number, number> {
    const weights = new Map<number, number>();
    const rest = this.#rest;
    const of = (v: number) =>
      weights.get(v) ?? this.#scale[v] * this.#weight(rest[3 * v], rest[3 * v + 1], rest[3 * v + 2]);
    for (let k = 0; k + 2 < this.#splits.length; k += 3) {
      const v = this.#splits[k];
      weights.set(v, (this.#scale[v] * (of(this.#splits[k + 1]) + of(this.#splits[k + 2]))) / 2);
    }
    return weights;
  }

  /**
   * Takes the region again over `positions`, with every move sent so far
   * landed: for a pass that added vertices or moved some mid-stroke. A vertex that
   * is still where this region put it keeps its rest. A vertex from `splits`
   * (`new, a, b` triples, as `splitStretched` reports them) moves as the mean of
   * its edge from here on: its position alone cannot say which part of the surface
   * it belongs to. Any other gets the weight of the rest position the drag would
   * have carried here (see `#weightAt`). Each then rests exactly where it lands on
   * it now, so that nothing jumps.
   */
  rebase(positions: ArrayLike<number>, indices: ArrayLike<number>, splits?: ArrayLike<number>): void {
    const drag = this.#drag;
    const slots = positions.length / 3;
    const oldSlots = this.#live.length;
    const oldRest = this.#rest;
    const oldLive = this.#live;
    const oldScale = this.#scale;
    const live = liveSlots(indices, slots);
    const rest = new Float32Array(positions.length);
    const scale = new Float32Array(slots);
    const touched = new Uint8Array(slots);
    touched.set(this.#touched.subarray(0, Math.min(oldSlots, slots)));
    const isSplit = new Uint8Array(slots);
    isSplit.set(this.#isSplit.subarray(0, Math.min(oldSlots, slots)));
    for (let k = 0; splits !== undefined && k + 2 < splits.length; k += 3) {
      this.#splits.push(splits[k], splits[k + 1], splits[k + 2]);
      isSplit[splits[k]] = 1;
    }
    const field = this.#hold();
    const tolerance = 1e-5 * this.#radius;
    const put = (v: number, x: number, y: number, z: number, share: number) => {
      rest[3 * v] = x;
      rest[3 * v + 1] = y;
      rest[3 * v + 2] = z;
      scale[v] = share;
    };
    for (let v = 0; v < slots; v++) {
      const [px, py, pz] = [positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]];
      const held = this.#held(v, field);
      if (live[v] === 0 || held === 0 || isSplit[v] === 1) {
        put(v, px, py, pz, held);
        continue;
      }
      if (v < oldSlots && oldLive[v] === 1) {
        const [rx, ry, rz] = [oldRest[3 * v], oldRest[3 * v + 1], oldRest[3 * v + 2]];
        const w = oldScale[v] * this.#weight(rx, ry, rz);
        if (Math.hypot(rx + w * drag[0] - px, ry + w * drag[1] - py, rz + w * drag[2] - pz) <= tolerance) {
          put(v, rx, ry, rz, held);
          continue;
        }
      }
      const w = held * this.#weightAt(px, py, pz, drag);
      put(v, px - w * drag[0], py - w * drag[1], pz - w * drag[2], held);
      touched[v] = 1;
    }
    this.#rest = rest;
    this.#live = live;
    this.#scale = scale;
    this.#touched = touched;
    this.#isSplit = isSplit;
    for (const [v, w] of this.#splitWeights()) {
      rest[3 * v] = positions[3 * v] - w * drag[0];
      rest[3 * v + 1] = positions[3 * v + 1] - w * drag[1];
      rest[3 * v + 2] = positions[3 * v + 2] - w * drag[2];
      touched[v] = w === 0 ? touched[v] : 1;
    }
    this.#take(Math.max(this.#reach, this.#influence));
    this.at(drag);
  }

  /**
   * The weight `w` of a vertex at `x, y, z` under `drag`: one with
   * `w = weight(p - w * drag)`. Fixed-point steps diverge once the drag is
   * longer than the falloff, which leaves the vertices on a long lump's sides
   * behind, so this solves for the share `u = w / strength` directly.
   *
   * The falloff is largest at `top`, where the line through `p` along the drag
   * passes closest to the center, and falls beyond it. So above `top`, the
   * excess `falloff - u` only falls and has at most one root, found by
   * bisection. That largest root is the one on the surface: a vertex that was
   * dragged here rested where the brush weighed it most, and the smaller
   * roots, down to 0 for a point now outside the ball, are off the surface.
   * With no root above `top`, the vertex lies past the lump's rim, and the
   * largest root below it is found by scanning.
   */
  #weightAt(x: number, y: number, z: number, drag: readonly [number, number, number]): number {
    const s = this.#strength;
    const [ax, ay, az] = [s * drag[0], s * drag[1], s * drag[2]];
    const c = this.#center;
    const length2 = ax * ax + ay * ay + az * az;
    if (length2 === 0) {
      return this.#weight(x, y, z);
    }
    const at = (u: number): number => this.#weight(x - u * ax, y - u * ay, z - u * az);
    const excess = (u: number): number => at(u) / s - u;
    const closest = ((x - c[0]) * ax + (y - c[1]) * ay + (z - c[2]) * az) / length2;
    const top = Math.min(1, Math.max(0, closest));
    let low = top;
    let high = 1;
    if (excess(top) < 0) {
      const STEPS = 16;
      high = top;
      low = 0;
      for (let k = STEPS - 1; k >= 0; k--) {
        const u = (top * k) / STEPS;
        if (excess(u) >= 0) {
          low = u;
          break;
        }
        high = u;
      }
    }
    for (let k = 0; k < 30; k++) {
      const mid = (low + high) / 2;
      if (excess(mid) >= 0) {
        low = mid;
      } else {
        high = mid;
      }
    }
    return at(low);
  }

  /**
   * The edit that puts the grabbed vertices where they belong at the total `drag`,
   * as `setPositions` takes it: the region works in the mesh's stored positions,
   * which are relative to its `origin`. A longer drag can take in more vertices.
   */
  at(drag: readonly [number, number, number]): { vertices: Uint32Array; positions: Float64Array } {
    this.#drag = [drag[0], drag[1], drag[2]];
    this.#influence = this.influence(drag);
    this.#peak = Math.max(this.#peak, this.#influence);
    if (this.#influence > this.#reach) {
      this.#take(this.#influence * 1.25);
    }
    const near = this.#near;
    const rest = this.#rest;
    const splitWeights = this.#splitWeights();
    const room = near.length + splitWeights.size;
    const ids = new Uint32Array(room);
    const out = new Float64Array(3 * room);
    let n = 0;
    const write = (v: number, w: number) => {
      if (w === 0 && this.#touched[v] === 0) {
        return;
      }
      this.#touched[v] = 1;
      ids[n] = v;
      out[3 * n] = rest[3 * v] + w * drag[0] + this.#origin[0];
      out[3 * n + 1] = rest[3 * v + 1] + w * drag[1] + this.#origin[1];
      out[3 * n + 2] = rest[3 * v + 2] + w * drag[2] + this.#origin[2];
      n++;
    };
    for (let k = 0; k < near.length; k++) {
      const v = near[k];
      if (this.#isSplit[v] === 0) {
        write(v, this.#scale[v] * this.#weight(rest[3 * v], rest[3 * v + 1], rest[3 * v + 2]));
      }
    }
    for (const [v, w] of splitWeights) {
      if (this.#scale[v] !== 0) {
        write(v, w);
      }
    }
    this.ids = ids.subarray(0, n);
    return { vertices: this.ids, positions: out.subarray(0, 3 * n) };
  }
}

/** Which slots a live triangle uses (a removed triangle reads `0, 0, 0`). */
function liveSlots(indices: ArrayLike<number>, slots: number): Uint8Array {
  const used = new Uint8Array(slots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const [a, b, c] = [indices[t], indices[t + 1], indices[t + 2]];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    used[a] = 1;
    used[b] = 1;
    used[c] = 1;
  }
  return used;
}

/**
 * The mean live-edge length over flat positions and a triangle index: each
 * undirected edge once, zero-length (dead-slot) and non-finite edges
 * skipped. `null` when no edge survives. A sculpt session anchors its
 * remeshes to this (taken when the brush arms), so repeated strokes converge
 * back to the armed density instead of ratcheting ever finer.
 */
export function meanEdgeLength(positions: ArrayLike<number>, indices: ArrayLike<number>): number | null {
  const seen = new Set<number>();
  let sum = 0;
  let count = 0;
  const edge = (a: number, b: number) => {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const key = lo * 4294967296 + hi;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    const dx = positions[3 * a] - positions[3 * b];
    const dy = positions[3 * a + 1] - positions[3 * b + 1];
    const dz = positions[3 * a + 2] - positions[3 * b + 2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(len > 0) || !Number.isFinite(len)) {
      return;
    }
    sum += len;
    count++;
  };
  for (let t = 0; t + 2 < indices.length; t += 3) {
    edge(indices[t], indices[t + 1]);
    edge(indices[t + 1], indices[t + 2]);
    edge(indices[t + 2], indices[t]);
  }
  return count === 0 ? null : sum / count;
}

/** Interpolates a raycast hit over flat local positions by its barycentric weights. */
export function interpolateHit(
  positions: ArrayLike<number>,
  corners: ArrayLike<number>,
  barycentric: readonly [number, number, number],
): [number, number, number] {
  const [u, v, w] = barycentric;
  const at = (s: number): readonly [number, number, number] => [
    positions[3 * s],
    positions[3 * s + 1],
    positions[3 * s + 2],
  ];
  const [a, b, c] = [at(corners[0]), at(corners[1]), at(corners[2])];
  return [u * a[0] + v * b[0] + w * c[0], u * a[1] + v * b[1] + w * c[1], u * a[2] + v * b[2] + w * c[2]];
}

/** The scene's step nesting, where it has one (the sync scene; the Worker applies one step per dab). */
export interface StrokeSteps {
  beginStep?: () => void;
  endStep?: () => void;
  cancelStep?: () => void;
}

/**
 * One press-and-hold stroke: dabs through `apply`, coalescing to one undo
 * step where the scene nests steps. Lazy: no step opens until the first dab
 * lands, and a stroke whose dabs all failed cancels its step instead.
 *
 * The dabs lie every spacing along the pointer's path, whatever its event rate
 * and however long a dab takes: those that come due while one is in flight
 * wait, and go together when it lands, so a slow scene lags but leaves no gaps.
 */
export class SculptStroke {
  readonly #brush: () => SculptBrush;
  readonly #steps: StrokeSteps;
  readonly #apply: (ops: readonly RepairOp[]) => Promise<unknown>;
  readonly #blocked: (at: readonly [number, number, number], radius: number) => boolean;
  #last: [number, number, number] | null = null;
  #lastTangent: [number, number, number] | undefined = undefined;
  #min: [number, number, number] | null = null;
  #max: [number, number, number] | null = null;
  #queue: RepairOp[] = [];
  #latest: (() => Promise<unknown>) | null = null;
  #inFlight = 0;
  /** A pass is out: nothing else goes until it settles. */
  #passing = false;
  /** What came due behind a pass, in order; a `true` entry is a pass itself. */
  #held: { run: () => Promise<unknown>; pass: boolean }[] = [];
  #begun = false;
  #landed = false;
  #ended = false;
  #closed = false;
  #dabs = 0;
  #close!: () => void;
  /** Resolves once the stroke is ended and its step closed. */
  readonly done: Promise<void>;

  /**
   * `brush` is read live, per dab: pass a getter and the brush inputs stay
   * adjustable mid-stroke (a plain brush is read as-is). `apply` gets the dabs
   * that came due, in order, one or more at a time. A dab `blocked` says no
   * to (the mask) is left out, and the path goes on past it.
   */
  constructor(
    brush: SculptBrush | (() => SculptBrush),
    steps: StrokeSteps,
    apply: (ops: readonly RepairOp[]) => Promise<unknown>,
    blocked: (at: readonly [number, number, number], radius: number) => boolean = () => false,
  ) {
    this.#brush = typeof brush === "function" ? brush : () => brush;
    this.#steps = steps;
    this.#apply = apply;
    this.#blocked = blocked;
    this.done = new Promise<void>((resolve) => {
      this.#close = resolve;
    });
  }

  /** Dabs attempted so far. */
  get dabs(): number {
    return this.#dabs;
  }

  /**
   * Moves the stroke to `center` (local coordinates), laying a dab every
   * spacing from the last one: `skipped` when it is under a spacing away,
   * `ended` once the stroke closed. With `force` (the hold-still buildup,
   * which goes by time and not by distance) one dab lands right there with
   * `share` of the strength, or `busy` while one is in flight. Past `TELEPORT_RADII` radii it starts over
   * at `center`, as after `lift`.
   */
  dab(center: readonly [number, number, number], force = false, share = 1): "stamped" | "skipped" | "busy" | "ended" {
    if (this.#ended) {
      return "ended";
    }
    const brush = this.#brush();
    const at: [number, number, number] = [center[0], center[1], center[2]];
    if (force || this.#last === null) {
      if (force && this.#inFlight > 0) {
        return "busy";
      }
      this.#stamp(brush, at, force ? share : DAB_SHARE, force ? this.#lastTangent : undefined);
      this.#sendIfIdle();
      return "stamped";
    }
    const spacing = dabSpacing(brush.radius);
    const dx = at[0] - this.#last[0];
    const dy = at[1] - this.#last[1];
    const dz = at[2] - this.#last[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < spacing) {
      return "skipped";
    }
    if (dist > TELEPORT_RADII * brush.radius) {
      this.#lastTangent = undefined;
      this.#stamp(brush, at, DAB_SHARE);
      this.#sendIfIdle();
      return "stamped";
    }
    const [x, y, z] = this.#last;
    const tangent: [number, number, number] = [dx / dist, dy / dist, dz / dist];
    this.#lastTangent = tangent;
    const steps = Math.floor(dist / spacing);
    for (let k = 1; k <= steps; k++) {
      const u = (k * spacing) / dist;
      this.#stamp(brush, [x + dx * u, y + dy * u, z + dz * u], DAB_SHARE, tangent);
    }
    this.#sendIfIdle();
    return "stamped";
  }

  /** The pointer left the surface: the next `dab` starts over where it lands. */
  lift(): void {
    this.#last = null;
    this.#lastTangent = undefined;
  }

  /**
   * Writes a new displacement of a stroke that holds its own state (a grab):
   * `edit` runs once nothing is in flight, and a newer `move` replaces one that
   * has not started. `at` is where the brush is now, for `bounds`.
   */
  move(edit: () => Promise<unknown>, at: readonly [number, number, number]): "stamped" | "ended" {
    if (this.#ended) {
      return "ended";
    }
    this.#open();
    this.#dabs++;
    this.#cover(at);
    this.#latest = edit;
    this.#sendIfIdle();
    return "stamped";
  }

  #sendIfIdle(): void {
    if (this.#inFlight === 0) {
      this.flush();
    }
  }

  #open(): void {
    if (!this.#begun) {
      this.#begun = true;
      this.#steps.beginStep?.();
    }
  }

  #cover(at: readonly [number, number, number]): void {
    if (this.#min === null || this.#max === null) {
      this.#min = [at[0], at[1], at[2]];
      this.#max = [at[0], at[1], at[2]];
    } else {
      for (let k = 0; k < 3; k++) {
        this.#min[k] = Math.min(this.#min[k], at[k]);
        this.#max[k] = Math.max(this.#max[k], at[k]);
      }
    }
  }

  #stamp(
    brush: SculptBrush,
    at: [number, number, number],
    share: number,
    tangent?: readonly [number, number, number],
  ): void {
    this.#last = at;
    if (this.#blocked(at, brush.radius)) {
      return;
    }
    this.#open();
    this.#dabs++;
    this.#cover(at);
    this.#queue.push(dabOperation(brush, at, share, tangent));
  }

  /**
   * Runs `work` in order with the stroke's own edits, after what is due and
   * before what comes due next (a remesh): the stroke does not close, and sends
   * nothing more, until it settles. A failure is the caller's to swallow.
   */
  pass(work: () => Promise<unknown>): Promise<void> {
    return new Promise<void>((resolve) => {
      const run = () => {
        const running = work();
        void running.then(
          () => resolve(),
          () => resolve(),
        );
        return running;
      };
      if (this.#passing) {
        this.#hold();
        this.#held.push({ run, pass: true });
        return;
      }
      this.flush();
      this.#startPass(run);
    });
  }

  #startPass(run: () => Promise<unknown>): void {
    this.#passing = true;
    this.#send(run, true);
  }

  /** Moves what is due behind the pass that is out. */
  #hold(): void {
    if (this.#queue.length > 0) {
      const ops = this.#queue;
      this.#queue = [];
      this.#held.push({ run: () => this.#apply(ops), pass: false });
    }
    if (this.#latest !== null) {
      this.#held.push({ run: this.#latest, pass: false });
      this.#latest = null;
    }
  }

  /** After a pass: what waited behind it, up to the next pass. */
  #release(): void {
    while (!this.#passing && this.#held.length > 0) {
      const next = this.#held.shift();
      if (next?.pass === true) {
        this.#startPass(next.run);
      } else if (next !== undefined) {
        this.#send(next.run);
      }
    }
  }

  /**
   * Sends what is due now, even with a dab still in flight: for a caller about
   * to send something else that has to land after it (a remesh). On its own
   * the stroke sends the next batch when the last one lands. A pass that is out
   * holds it: what is due goes when the pass settles.
   */
  flush(): void {
    if (this.#passing) {
      return;
    }
    if (this.#queue.length > 0) {
      const ops = this.#queue;
      this.#queue = [];
      this.#send(() => this.#apply(ops));
    }
    if (this.#latest !== null) {
      const edit = this.#latest;
      this.#latest = null;
      this.#send(edit);
    }
  }

  #send(run: () => Promise<unknown>, pass = false): void {
    this.#inFlight++;
    let settled: Promise<void>;
    try {
      settled = Promise.resolve(run()).then(
        () => {
          if (!pass || this.#begun) {
            this.#landed = true;
          }
        },
        () => {
          // A dab rejected (the mesh was removed meanwhile): the stroke
          // keeps its step, and cancels it when nothing landed.
        },
      );
    } catch {
      // The applier threw outright: a failed dab, like a rejection.
      settled = Promise.resolve();
    }
    void settled.finally(() => {
      this.#inFlight--;
      if (pass) {
        this.#passing = false;
        this.#release();
      }
      if (this.#inFlight === 0) {
        this.flush();
      }
      if (this.#inFlight === 0 && this.#ended && !this.#passing && this.#held.length === 0) {
        this.#finish();
      }
    });
  }

  /**
   * The stamped dabs' bounding sphere, grown by twice the brush radius: the
   * region a stroke-end remesh should even out. The extra margin covers
   * hold-buildup growth past the dab centers (a spike outgrows its ball).
   * `null` before the first dab.
   */
  bounds(): { center: [number, number, number]; radius: number } | null {
    if (this.#min === null || this.#max === null) {
      return null;
    }
    const center: [number, number, number] = [
      (this.#min[0] + this.#max[0]) / 2,
      (this.#min[1] + this.#max[1]) / 2,
      (this.#min[2] + this.#max[2]) / 2,
    ];
    const dx = this.#max[0] - this.#min[0];
    const dy = this.#max[1] - this.#min[1];
    const dz = this.#max[2] - this.#min[2];
    const grown = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2 + 2 * this.#brush().radius;
    return { center, radius: grown };
  }

  /** Closes the stroke: its step ends (or cancels, when nothing landed), once what it sent settles. */
  end(): void {
    if (this.#ended) {
      return;
    }
    this.#ended = true;
    this.flush();
    if (this.#inFlight === 0) {
      this.#finish();
    }
  }

  #finish(): void {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
    if (this.#begun) {
      if (this.#landed) {
        this.#steps.endStep?.();
      } else {
        this.#steps.cancelStep?.();
      }
    }
    this.#close();
  }
}

// ---- Sculpt mask (#130) ----
//
// A painted protected region the deform brushes leave in place: the mask
// stores the painted vertex slots, and every dab carries the ones it can
// reach to the engine, which holds them byte-identical while the free
// vertices move (see `viewport/mask-protect.ts`).
//
// The storage contract answers #130's question against #106's selection
// contract. A selection snapshot pins the mesh's content stamp, so the
// engine throws `STALE_SELECTION` after any edit — including the stroke's
// own position-only dabs, which would kill a mask mid-stroke. The showcase
// mask therefore pins the vertex ids instead of the stamp: ids are stable
// across position-only edits (inflate/relax/grab move vertices without
// touching slots), so the mask survives the strokes it protects. It fails
// loudly — `checkMask` throws `STALE_SELECTION`, the same code — as soon as
// the ids stop naming live vertices: a masked slot out of range, or no
// longer used by a live triangle (a collapse, compact, weld or isolate took
// it). Surviving cases re-verify cheaply: splits add slots but keep the old
// vertices alive, so a split stroke's mask carries over (the new vertices
// land unmasked, unless they split a masked edge: those join the mask, see
// `inherited`); undo/redo restores the buffers, and the mask with them
// when its ids are live again. The documented hole: a merge or compact can
// remap ids while keeping them live, silently re-pointing the mask —
// reselect after structural repairs, as with #106 selections.

/**
 * A painted protected region on one mesh: the masked vertex slots,
 * ascending, and the vertex slot count when painted (a topology edit that
 * grows or compacts the slots re-verifies against this).
 */
export interface SculptMask {
  /** The mesh the slots belong to. */
  mesh: MeshHandle;
  /** The masked vertex slots, ascending. */
  vertices: Uint32Array;
  /** `positions().length / 3` when painted; see the contract above. */
  slots: number;
  /**
   * The slots in `vertices` no one painted: splits between masked vertices,
   * held like them. An undo takes them with the split and a redo brings them
   * back, so one dying says nothing about the mask: it is skipped, not stale.
   */
  inherited?: ReadonlySet<number>;
}

/**
 * Whether the mask still names live vertices of `mesh` over its current
 * buffers: every masked slot in range and used by a live triangle. Removed
 * slots read `0, 0, 0`, so triples of three zeros are skipped — mirroring the
 * engine's `buildTopology`, where a genuine all-zero triangle is degenerate
 * (zero area) and losing it is harmless. One pass over the triangles, then one
 * over the mask: painting calls this per dab. The stroke start and
 * paint/invert paths call `checkMask` to fail loudly instead.
 */
export function maskUsable(
  mask: Pick<SculptMask, "vertices" | "slots" | "inherited">,
  indices: ArrayLike<number>,
  slots: number,
): boolean {
  // A slots mismatch (splits grew the mesh, a compact shrank it) never
  // fails on its own: the ids may still name the same vertices (splits only
  // add), so liveness below decides. An empty mask protects nothing.
  if (mask.vertices.length === 0) {
    return true;
  }
  const used = new Uint8Array(slots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a < slots) {
      used[a] = 1;
    }
    if (b < slots) {
      used[b] = 1;
    }
    if (c < slots) {
      used[c] = 1;
    }
  }
  for (let k = 0; k < mask.vertices.length; k++) {
    const id = mask.vertices[k];
    if ((!Number.isInteger(id) || id < 0 || id >= slots || used[id] === 0) && mask.inherited?.has(id) !== true) {
      return false;
    }
  }
  return true;
}

/**
 * What {@link checkMask} throws: the showcase twin of the engine's
 * `FastMeshError`, carrying the same machine-readable `code`
 * (`"INVALID_ARGUMENT"` for the wrong mesh, `"STALE_SELECTION"` once a
 * topology edit takes a masked vertex). A local class — not the engine's —
 * so this pure module stays importable without the wasm build (like every
 * other headless-tested showcase module); catch sites treat both by `code`.
 */
export class MaskError extends Error {
  /** What went wrong; the same codes the engine reports stale selections with. */
  readonly code: FastMeshErrorCode;

  constructor(code: FastMeshErrorCode, message: string) {
    super(message);
    this.name = "MaskError";
    this.code = code;
  }
}

/**
 * Fails loudly when `mask` is not for `mesh` or no longer usable over its
 * current buffers: `INVALID_ARGUMENT` for the wrong mesh (like the
 * engine's `checkSelection`), `STALE_SELECTION` once a topology edit took a
 * masked vertex (repaint it). Callers clear the stored mask on the throw.
 */
export function checkMask(
  mask: SculptMask,
  mesh: MeshHandle,
  indices: ArrayLike<number>,
  slots: number,
): void {
  if (mask === null || typeof mask !== "object" || mask.mesh !== mesh) {
    throw new MaskError("INVALID_ARGUMENT", "the mask is not for this mesh");
  }
  if (!maskUsable(mask, indices, slots)) {
    throw new MaskError("STALE_SELECTION", "the mask predates the mesh's latest topology edit; repaint it");
  }
}

/**
 * The sphere-brush paint: every live vertex within `radius` of `center`
 * (squared distances, `<=`, like the engine's `brushPick`), ascending. Dead
 * slots and out-of-range ids never paint.
 */
export function paintMaskIds(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  slots: number,
  center: readonly [number, number, number],
  radius: number,
): Uint32Array {
  const used = new Uint8Array(slots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a >= 0 && a < slots) {
      used[a] = 1;
    }
    if (b >= 0 && b < slots) {
      used[b] = 1;
    }
    if (c >= 0 && c < slots) {
      used[c] = 1;
    }
  }
  const r2 = radius * radius;
  const out: number[] = [];
  for (let v = 0; v < slots; v++) {
    if (used[v] === 0) {
      continue;
    }
    const dx = positions[v * 3] - center[0];
    const dy = positions[v * 3 + 1] - center[1];
    const dz = positions[v * 3 + 2] - center[2];
    if (dx * dx + dy * dy + dz * dz <= r2) {
      out.push(v);
    }
  }
  return Uint32Array.from(out);
}

/** The sorted union of two ascending id lists (paint accumulates through this). */
export function unionMaskIds(a: ArrayLike<number>, b: ArrayLike<number>): Uint32Array {
  const out: number[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const x = a[i];
    const y = b[j];
    if (x === y) {
      out.push(x);
      i++;
      j++;
    } else if (x < y) {
      out.push(x);
      i++;
    } else {
      out.push(y);
      j++;
    }
  }
  while (i < a.length) {
    out.push(a[i++]);
  }
  while (j < b.length) {
    out.push(b[j++]);
  }
  return Uint32Array.from(out);
}

/**
 * Inverts a mask over the mesh's current buffers: every live vertex outside
 * it, ascending. The engine's `selectInvert` needs a live selection; this
 * takes the same universe without one, so invert works freshly painted.
 */
export function invertMaskIds(
  masked: ArrayLike<number>,
  indices: ArrayLike<number>,
  slots: number,
): Uint32Array {
  const used = new Uint8Array(slots);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t];
    const b = indices[t + 1];
    const c = indices[t + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a >= 0 && a < slots) {
      used[a] = 1;
    }
    if (b >= 0 && b < slots) {
      used[b] = 1;
    }
    if (c >= 0 && c < slots) {
      used[c] = 1;
    }
  }
  const inSet = new Set<number>();
  for (let k = 0; k < masked.length; k++) {
    inSet.add(masked[k]);
  }
  const out: number[] = [];
  for (let v = 0; v < slots; v++) {
    if (used[v] === 1 && !inSet.has(v)) {
      out.push(v);
    }
  }
  return Uint32Array.from(out);
}
