/**
 * Extended 3D visual defect overlays for the Mesh Healer (#235). Complements
 * `repair-overlays.ts` with self-intersecting triangles, inverted shells /
 * orientation conflicts and sliver faces; which triangles and edges those
 * are comes from `ux/healer-defects.ts`.
 */
import type { Analysis } from "fast-mesh-wasm";
import * as THREE from "three";
import type { HealerOverlayKind, RepairOverlayKind } from "../../../app/state/app-state";
import {
  detectInvertedClosedComponents,
  detectOrientationConflictEdges,
  detectSelfIntersectingTriangles,
  detectSliverTriangles,
} from "../ux/healer-defects";
import { buildRepairOverlayGroup } from "./repair-overlays";

export const HEALER_OVERLAY_COLORS: Record<HealerOverlayKind, number> = {
  selfIntersections: 0xf97316, // vivid orange
  inverted: 0xf43f5e, // crimson rose
  slivers: 0xeab308, // warm gold
};

function buildTriangleOverlayMesh(
  positions: Float32Array,
  indices: Uint32Array,
  tris: Uint32Array,
  color: number,
  name: string,
): THREE.Mesh | null {
  if (tris.length === 0) return null;
  const triPos = new Float32Array(tris.length * 9);
  for (let i = 0; i < tris.length; i++) {
    const base = tris[i]! * 3;
    const dst = i * 9;
    for (let k = 0; k < 3; k++) {
      const src = (indices[base + k] ?? 0) * 3;
      triPos[dst + k * 3] = positions[src] ?? 0;
      triPos[dst + k * 3 + 1] = positions[src + 1] ?? 0;
      triPos[dst + k * 3 + 2] = positions[src + 2] ?? 0;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(triPos, 3));
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = name;
  mesh.renderOrder = 2;
  mesh.frustumCulled = false;
  return mesh;
}

function buildEdgeSegments(
  positions: Float32Array,
  edgePairs: Uint32Array,
  color: number,
  name: string,
): THREE.LineSegments | null {
  if (edgePairs.length < 2) return null;
  const edgeCount = Math.floor(edgePairs.length / 2);
  const segPos = new Float32Array(edgeCount * 6);
  for (let e = 0; e < edgeCount; e++) {
    const a = edgePairs[e * 2]! * 3;
    const b = edgePairs[e * 2 + 1]! * 3;
    const dst = e * 6;
    segPos[dst] = positions[a] ?? 0;
    segPos[dst + 1] = positions[a + 1] ?? 0;
    segPos[dst + 2] = positions[a + 2] ?? 0;
    segPos[dst + 3] = positions[b] ?? 0;
    segPos[dst + 4] = positions[b + 1] ?? 0;
    segPos[dst + 5] = positions[b + 2] ?? 0;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(segPos, 3));
  const mat = new THREE.LineBasicMaterial({
    color,
    depthTest: false,
    transparent: true,
    opacity: 0.95,
  });
  const lines = new THREE.LineSegments(geo, mat);
  lines.name = name;
  lines.renderOrder = 3;
  lines.frustumCulled = false;
  return lines;
}

/**
 * Combines standard `repair-overlays.ts` primitives with Mesh Healer overlays
 * (self-intersections, inverted winding / orientation conflicts, sliver faces).
 */
export function buildHealerOverlayGroup(
  analysis: Analysis,
  positions: Float32Array,
  indices: Uint32Array,
  repairKinds: Record<RepairOverlayKind, boolean>,
  healerKinds: Record<HealerOverlayKind, boolean>,
): THREE.Group | null {
  const base = buildRepairOverlayGroup(analysis, positions, indices, repairKinds);
  const extra: THREE.Object3D[] = [];

  if (healerKinds.slivers) {
    const tris = detectSliverTriangles(positions, indices);
    const mesh = buildTriangleOverlayMesh(positions, indices, tris, HEALER_OVERLAY_COLORS.slivers, "overlay-slivers");
    if (mesh !== null) extra.push(mesh);
  }
  if (healerKinds.inverted) {
    const conflicts = detectOrientationConflictEdges(indices);
    const lines = buildEdgeSegments(positions, conflicts, HEALER_OVERLAY_COLORS.inverted, "overlay-inverted-edges");
    if (lines !== null) extra.push(lines);
    const invTris = detectInvertedClosedComponents(analysis, positions, indices);
    const mesh = buildTriangleOverlayMesh(
      positions,
      indices,
      invTris,
      HEALER_OVERLAY_COLORS.inverted,
      "overlay-inverted-shells",
    );
    if (mesh !== null) extra.push(mesh);
  }
  if (healerKinds.selfIntersections && analysis.selfIntersects !== false) {
    const tris = detectSelfIntersectingTriangles(positions, indices);
    const mesh = buildTriangleOverlayMesh(
      positions,
      indices,
      tris,
      HEALER_OVERLAY_COLORS.selfIntersections,
      "overlay-selfIntersections",
    );
    if (mesh !== null) extra.push(mesh);
  }

  if (extra.length === 0) return base;
  const root = base ?? new THREE.Group();
  root.name = "repair-overlays";
  for (const child of extra) {
    root.add(child);
  }
  return root;
}

export function disposeHealerOverlayGroup(group: THREE.Group): void {
  for (const child of group.children) {
    if (child instanceof THREE.LineSegments || child instanceof THREE.Mesh) {
      child.geometry.dispose();
      const mat = child.material;
      if (Array.isArray(mat)) {
        for (const m of mat) m.dispose();
      } else {
        mat.dispose();
      }
    }
  }
}
