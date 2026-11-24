// The free-form solvers' inputs and outputs (#193): the scene's deform domain,
// the ARAP and Kelvinlet options built from `wave6.deform`, the Kelvinlet
// loads from the panel's inputs or a viewport pull, the vertex picks the panel
// seeds anchors, handles, pins and centers with, and the one-line report.
import type {
  ArapOptions,
  FreeformProtection,
  FreeformReport,
  KelvinletLoad,
  KelvinletOptions,
  XpbdPinInput,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { Wave6DeformState } from "../../../app/state/wave6-state";
import type { Vec3Tuple } from "./ffd-lattice";

type DeformDomain = NonNullable<SceneStore["scene"]["deform"]>;

/** Retrieves `store.scene.deform`, throwing if the scene was built without the deform domain. */
export function requireDeformDomain(store: SceneStore): DeformDomain {
  const deform = store.scene.deform;
  if (!deform) {
    throw new Error("Deform domain is unavailable on this scene.");
  }
  return deform;
}

/** Formats a concise status summary line for `wave6.deform.lastReport`. */
export function formatFreeformReport(label: string, report: FreeformReport): string {
  const base = `${label}: moved ${report.verticesMoved}/${report.verticesEvaluated} verts, max disp ${report.maxDisplacement.toFixed(4)}`;
  return report.selfCollisionContactsResolved > 0
    ? `${base}, ${report.selfCollisionContactsResolved} self-collisions resolved`
    : base;
}

/** A thrown value's message, for the panel's error line and a failed drag's report. */
export function formatError(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** ARAP options from the deform state's anchors, handles and weights. */
export function arapOptions(
  deform: Wave6DeformState,
  iterations: number,
  protect: FreeformProtection | undefined,
): ArapOptions {
  return {
    anchors: deform.arapAnchors.map((vertex) => ({ vertex })),
    handles: deform.arapHandles,
    weights: deform.arapWeights,
    iterations,
    protect,
  };
}

/** Kelvinlet options around `center` from the deform state's radius, Poisson ratio, scale and symmetry. */
export function kelvinletOptions(
  deform: Wave6DeformState,
  center: Vec3Tuple,
  load: KelvinletLoad,
  protect: FreeformProtection | undefined,
): KelvinletOptions {
  return {
    center,
    load,
    epsilon: deform.kelvinletEpsilon,
    nu: deform.kelvinletNu,
    scale: deform.kelvinletScale,
    symmetry: {
      x: deform.kelvinletSymX,
      y: deform.kelvinletSymY,
      z: deform.kelvinletSymZ,
    },
    protect,
  };
}

/**
 * The panel's Kelvinlet load of `kind`: `force` is the grab force, the twist
 * torque, or the pinch dipole's diagonal; `magnitude` is the scale's.
 */
export function buildKelvinletLoad(
  kind: Wave6DeformState["kelvinletKind"],
  force: Vec3Tuple,
  magnitude: number,
): KelvinletLoad {
  switch (kind) {
    case "grab":
      return { kind: "grab", force };
    case "twist":
      return { kind: "twist", torque: force };
    case "scale":
      return { kind: "scale", magnitude };
    case "pinch":
      return {
        kind: "pinch",
        forceDipole: [
          [force[0], 0, 0],
          [0, force[1], 0],
          [0, 0, force[2]],
        ],
      };
  }
}

/**
 * The Kelvinlet load of `kind` for a viewport pull by `delta`: the twist
 * doubles it, and the scale signs its length by the pull's X + Y.
 */
export function buildKelvinletLoadFromDrag(kind: Wave6DeformState["kelvinletKind"], delta: Vec3Tuple): KelvinletLoad {
  switch (kind) {
    case "grab":
      return { kind: "grab", force: delta };
    case "twist":
      return { kind: "twist", torque: [delta[0] * 2, delta[1] * 2, delta[2] * 2] };
    case "scale": {
      const mag = Math.hypot(delta[0], delta[1], delta[2]) * (delta[1] + delta[0] >= 0 ? 1 : -1);
      return { kind: "scale", magnitude: mag };
    }
    case "pinch":
      return {
        kind: "pinch",
        forceDipole: [
          [delta[0], 0, 0],
          [0, delta[1], 0],
          [0, 0, delta[2]],
        ],
      };
  }
}

/** XPBD pins holding each in-range anchor vertex at its current position. */
export function anchorPins(positions: ArrayLike<number>, anchors: readonly number[]): XpbdPinInput[] {
  const pins: XpbdPinInput[] = [];
  for (const v of anchors) {
    const b = v * 3;
    if (b >= 0 && b + 2 < positions.length) {
      pins.push({ vertex: v, target: [positions[b], positions[b + 1], positions[b + 2]] });
    }
  }
  return pins;
}

/**
 * The base band to pin: every vertex in the lowest 20% of the Y extent, and
 * the (last) highest vertex, a handle that stays free. Null without a finite
 * Y extent.
 */
export function bottomBandAnchors(positions: ArrayLike<number>): { pinned: number[]; topVertex: number } | null {
  let minY = Infinity;
  let maxY = -Infinity;
  let topVertex = 0;
  for (let v = 0; v * 3 + 1 < positions.length; v++) {
    const y = positions[v * 3 + 1];
    if (y < minY) minY = y;
    if (y >= maxY) {
      maxY = y;
      topVertex = v;
    }
  }
  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) return null;
  const cutoff = minY + (maxY - minY) * 0.2;
  const pinned: number[] = [];
  for (let v = 0; v * 3 + 1 < positions.length; v++) {
    if (positions[v * 3 + 1] <= cutoff) {
      pinned.push(v);
    }
  }
  return { pinned, topVertex };
}

/** An ARAP handle's target: `vertex`'s position moved by `delta`. Throws when `vertex` is out of range. */
export function handleTargetFromDelta(positions: ArrayLike<number>, vertex: number, delta: Vec3Tuple): Vec3Tuple {
  const b = vertex * 3;
  if (b < 0 || b + 2 >= positions.length) {
    throw new Error(`Vertex ${vertex} is out of range.`);
  }
  return [positions[b] + delta[0], positions[b + 1] + delta[1], positions[b + 2] + delta[2]];
}

/** The center of the positions' bounding box, rounded to 4 decimals for the inputs. Null when empty. */
export function boundsCenter(positions: ArrayLike<number>): Vec3Tuple | null {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX)) return null;
  return [
    Number(((minX + maxX) * 0.5).toFixed(4)),
    Number(((minY + maxY) * 0.5).toFixed(4)),
    Number(((minZ + maxZ) * 0.5).toFixed(4)),
  ];
}
