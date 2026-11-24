// Repair-issue overlays: hole boundaries, non-manifold edges, degenerate
// triangles and small components drawn in 3D from `analyze` (read-only).
// They hang under the bound mesh's object (the wire overlay's path, not a
// viewport fork), so gizmo moves carry them along. The store's coloring path
// is distance-to-another-mesh only, so issue highlights draw as three.js
// lines and fills over whatever coloring the collision panel shows (depth
// test off, drawn last). The segments and triangles each kind draws come
// from `ux/repair-overlay-data.ts`.
import type { Analysis } from "fast-mesh-wasm";
import * as THREE from "three";
import type { RepairOverlayKind } from "../../../app/state/app-state";
import {
  holeSegments,
  nonManifoldSegments,
  smallComponentTris,
  triangleEdgeSegments,
  triangleSoup,
} from "../ux/repair-overlay-data";

/** The overlay colour of each issue kind (lines and the component fill). */
export const REPAIR_OVERLAY_COLORS: Record<RepairOverlayKind, number> = {
  holes: 0xffd166,
  nonManifold: 0xff5964,
  degenerate: 0x9e7bff,
  components: 0x59d6ff,
};

const TAG = "fastMeshRepairOverlay";

const lineMaterials: Record<RepairOverlayKind, THREE.LineBasicMaterial> = {
  holes: new THREE.LineBasicMaterial({ color: REPAIR_OVERLAY_COLORS.holes, depthTest: false, transparent: true }),
  nonManifold: new THREE.LineBasicMaterial({
    color: REPAIR_OVERLAY_COLORS.nonManifold,
    depthTest: false,
    transparent: true,
  }),
  degenerate: new THREE.LineBasicMaterial({
    color: REPAIR_OVERLAY_COLORS.degenerate,
    depthTest: false,
    transparent: true,
  }),
  components: new THREE.LineBasicMaterial({
    color: REPAIR_OVERLAY_COLORS.components,
    depthTest: false,
    transparent: true,
  }),
};

const fillMaterial = new THREE.MeshBasicMaterial({
  color: REPAIR_OVERLAY_COLORS.components,
  transparent: true,
  opacity: 0.22,
  side: THREE.DoubleSide,
  depthTest: false,
  depthWrite: false,
});

function lineSegments(data: Float32Array, kind: RepairOverlayKind): THREE.LineSegments | null {
  if (data.length === 0) {
    return null;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(data, 3));
  const lines = new THREE.LineSegments(geometry, lineMaterials[kind]);
  // The binding's object never updates its matrix (`matrixAutoUpdate` is
  // off): the overlay inherits the mesh's transform as-is, like the wire overlay.
  lines.matrixAutoUpdate = false;
  lines.renderOrder = 999;
  lines.userData[TAG] = true;
  return lines;
}

/**
 * The overlay group for the enabled `kinds` (`null` when nothing is enabled
 * or nothing is found): hole and non-manifold boundary lines, degenerate
 * triangle edges, and the small components as a translucent fill. Dispose
 * with {@link disposeRepairOverlayGroup} (geometries only; materials are shared).
 */
export function buildRepairOverlayGroup(
  analysis: Analysis,
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  kinds: Readonly<Record<RepairOverlayKind, boolean>>,
): THREE.Group | null {
  let built: THREE.Object3D | null = null;
  const add = (child: THREE.Object3D | null) => {
    if (child !== null) {
      (built ??= new THREE.Group()).add(child);
    }
  };
  if (kinds.holes) {
    add(lineSegments(holeSegments(analysis.loops, positions), "holes"));
  }
  if (kinds.nonManifold) {
    add(lineSegments(nonManifoldSegments(analysis.nonManifoldEdgeList, positions), "nonManifold"));
  }
  if (kinds.degenerate) {
    const tris = new Uint32Array(analysis.degenerateList.length + analysis.duplicateList.length);
    tris.set(analysis.degenerateList, 0);
    tris.set(analysis.duplicateList, analysis.degenerateList.length);
    add(lineSegments(triangleEdgeSegments(tris, indices, positions), "degenerate"));
  }
  if (kinds.components) {
    const soup = triangleSoup(smallComponentTris(analysis.components), indices, positions);
    if (soup.length > 0) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(soup, 3));
      const fill = new THREE.Mesh(geometry, fillMaterial);
      fill.matrixAutoUpdate = false;
      fill.renderOrder = 998;
      fill.userData[TAG] = true;
      add(fill);
    }
  }
  if (built === null) {
    return null;
  }
  const group = new THREE.Group();
  group.add(built);
  group.matrixAutoUpdate = false;
  group.userData[TAG] = true;
  return group;
}

/** Whether `object` currently shows repair-issue overlays. */
export function hasRepairOverlays(object: THREE.Object3D): boolean {
  return object.children.some((c) => c.userData[TAG] === true);
}

/** Disposes an overlay group's geometries (its materials are shared). */
export function disposeRepairOverlayGroup(group: THREE.Group): void {
  group.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.userData[TAG] === true && mesh.geometry !== undefined) {
      (mesh.geometry as THREE.BufferGeometry).dispose();
    }
  });
}

/** Removes every repair-issue overlay from `object`, disposing its geometries. */
export function clearRepairOverlays(object: THREE.Object3D): void {
  for (const child of [...object.children]) {
    if (child.userData[TAG] === true) {
      object.remove(child);
      disposeRepairOverlayGroup(child as THREE.Group);
    }
  }
}
