// Where the deform gizmos go and how big they are (#193): the FFD cage's
// control points, wireframe and sphere size, and the ARAP / XPBD anchor and
// handle markers read from the live vertex positions, all in mesh-local space.
import type { FfdLatticeData } from "fast-mesh-wasm";
import type { Wave6DeformState } from "../../../app/state/wave6-state";
import { buildLatticeEdgeSegments, computeLatticeControlPoints, type Vec3Tuple } from "./ffd-lattice";

/** The FFD cage: control points, wireframe segments, and the control point sphere radius. */
export interface LatticeCage {
  cpPoints: Vec3Tuple[];
  edgePositions: Float32Array;
  sphereRadius: number;
}

/** The anchor and handle markers, and their sphere radius. */
export interface DeformMarkers {
  anchorPoints: Vec3Tuple[];
  handlePairs: Array<{ from: Vec3Tuple; to: Vec3Tuple }>;
  markerRadius: number;
}

/** The cage for `lattice`, with spheres sized to its frame; an empty cage for `null`. */
export function latticeCage(lattice: FfdLatticeData | null): LatticeCage {
  if (!lattice) {
    return { cpPoints: [] as Vec3Tuple[], edgePositions: new Float32Array(0), sphereRadius: 0.03 };
  }
  const pts = computeLatticeControlPoints(lattice);
  const edges = buildLatticeEdgeSegments(lattice);
  const maxHalf = Math.max(
    lattice.frame.halfExtents[0],
    lattice.frame.halfExtents[1],
    lattice.frame.halfExtents[2],
    0.25,
  );
  return {
    cpPoints: pts,
    edgePositions: edges,
    sphereRadius: Math.max(0.015, maxHalf * 0.045),
  };
}

/**
 * The markers at the in-range anchors and handles over `positions`, sized to
 * the mesh's X span; none for `null` (no mesh, or its views went away).
 */
export function deformMarkers(
  positions: ArrayLike<number> | null,
  anchors: Wave6DeformState["arapAnchors"],
  handles: Wave6DeformState["arapHandles"],
): DeformMarkers {
  if (positions === null) {
    return { anchorPoints: [], handlePairs: [], markerRadius: 0.03 };
  }
  const anchorPoints: Vec3Tuple[] = [];
  for (const v of anchors) {
    const b = v * 3;
    if (b >= 0 && b + 2 < positions.length) {
      anchorPoints.push([positions[b], positions[b + 1], positions[b + 2]]);
    }
  }
  const handlePairs: Array<{ from: Vec3Tuple; to: Vec3Tuple }> = [];
  for (const h of handles) {
    const b = h.vertex * 3;
    if (b >= 0 && b + 2 < positions.length) {
      handlePairs.push({
        from: [positions[b], positions[b + 1], positions[b + 2]],
        to: [h.target[0], h.target[1], h.target[2]],
      });
    }
  }
  let minX = Infinity;
  let maxX = -Infinity;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    if (positions[i] < minX) minX = positions[i];
    if (positions[i] > maxX) maxX = positions[i];
  }
  const span = Number.isFinite(minX) && Number.isFinite(maxX) ? Math.max(0.2, maxX - minX) : 1;
  return { anchorPoints, handlePairs, markerRadius: Math.max(0.015, span * 0.025) };
}
