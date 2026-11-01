// Repair-issue overlays: hole boundaries, non-manifold edges, degenerate
// triangles and small components drawn in 3D from `analyze` (read-only).
// They hang under the bound mesh's object (the wire overlay's path, not a
// viewport fork), so gizmo moves carry them along. The store's coloring path
// is distance-to-another-mesh only, so issue highlights draw as three.js
// lines and fills over whatever coloring the collision panel shows (depth
// test off, drawn last).
import type { Analysis, BoundaryLoop, MeshComponent } from "fast-mesh-wasm";
import * as THREE from "three";
import type { RepairOverlayKind } from "../scene/app-state";

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

/** Copies vertex `v`'s position into `out`; `false` when `v` is out of range. */
function at(positions: ArrayLike<number>, v: number, out: number[]): boolean {
  const o = 3 * v;
  if (!Number.isInteger(v) || v < 0 || o + 2 >= positions.length) {
    return false;
  }
  out[0] = positions[o];
  out[1] = positions[o + 1];
  out[2] = positions[o + 2];
  return true;
}

/** Pushes the segment `a`-`b` unless it is degenerate (a point). */
function pushSegment(segs: number[], a: number[], b: number[]): void {
  if (a[0] !== b[0] || a[1] !== b[1] || a[2] !== b[2]) {
    segs.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
}

/**
 * Segment endpoints `[x0, y0, z0, x1, y1, z1, ...]` of every loop's
 * boundary polyline (closed loops wrap around; open chains do not). Loops
 * with fewer than two known vertices draw nothing. Out-of-range vertices
 * (an analysis older than the positions) break the polyline there.
 */
export function holeSegments(loops: readonly BoundaryLoop[], positions: ArrayLike<number>): Float32Array {
  const segs: number[] = [];
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  for (const loop of loops) {
    const n = loop.vertices.length;
    if (n < 2) {
      continue;
    }
    const edges = loop.closed ? n : n - 1;
    for (let k = 0; k < edges; k++) {
      if (!at(positions, loop.vertices[k], a) || !at(positions, loop.vertices[(k + 1) % n], b)) {
        continue;
      }
      pushSegment(segs, a, b);
    }
  }
  return Float32Array.from(segs);
}

/**
 * Segment endpoints of the non-manifold edges (`[min, max]` vertex pairs).
 * Out-of-range ends (an analysis older than the positions) are skipped.
 */
export function nonManifoldSegments(
  edges: readonly (readonly [min: number, max: number])[],
  positions: ArrayLike<number>,
): Float32Array {
  const segs: number[] = [];
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  for (const [u, v] of edges) {
    if (!at(positions, u, a) || !at(positions, v, b)) {
      continue;
    }
    pushSegment(segs, a, b);
  }
  return Float32Array.from(segs);
}

/**
 * Segment endpoints of every listed triangle's three edges. `tris` holds
 * triangle slots into `indices` (as `analyze`'s triangle lists do); slots
 * out of range, and corners out of range, are skipped. Shared edges repeat:
 * overdraw is harmless (one draw call, depth test off).
 */
export function triangleEdgeSegments(
  tris: ArrayLike<number>,
  indices: ArrayLike<number>,
  positions: ArrayLike<number>,
): Float32Array {
  const segs: number[] = [];
  const corners = [0, 0, 0];
  const p = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const slots = indices.length / 3;
  for (let k = 0; k < tris.length; k++) {
    const t = tris[k];
    if (!Number.isInteger(t) || t < 0 || t >= slots) {
      continue;
    }
    let known = true;
    for (let c = 0; c < 3; c++) {
      corners[c] = indices[3 * t + c];
      if (!at(positions, corners[c], p[c])) {
        known = false;
        break;
      }
    }
    if (!known) {
      continue;
    }
    pushSegment(segs, p[0], p[1]);
    pushSegment(segs, p[1], p[2]);
    pushSegment(segs, p[2], p[0]);
  }
  return Float32Array.from(segs);
}

/**
 * The triangle slots of every component but the largest (by triangle
 * count): the "small components" a removal would drop. Ties keep the first
 * largest; a single component (or none) highlights nothing.
 */
export function smallComponentTris(components: readonly MeshComponent[]): Uint32Array {
  if (components.length < 2) {
    return new Uint32Array(0);
  }
  let largest = 0;
  for (let k = 1; k < components.length; k++) {
    if (components[k].triangles.length > components[largest].triangles.length) {
      largest = k;
    }
  }
  let n = 0;
  for (let k = 0; k < components.length; k++) {
    if (k !== largest) {
      n += components[k].triangles.length;
    }
  }
  const out = new Uint32Array(n);
  let offset = 0;
  for (let k = 0; k < components.length; k++) {
    if (k !== largest) {
      out.set(components[k].triangles, offset);
      offset += components[k].triangles.length;
    }
  }
  return out;
}

/**
 * A non-indexed triangle soup of `tris` (slots into `indices`): the small
 * components' fill. Slots or corners out of range are skipped, so a soup
 * may hold fewer triangles than listed.
 */
export function triangleSoup(
  tris: ArrayLike<number>,
  indices: ArrayLike<number>,
  positions: ArrayLike<number>,
): Float32Array {
  const soup: number[] = [];
  const corner = [0, 0, 0];
  const slots = indices.length / 3;
  for (let k = 0; k < tris.length; k++) {
    const t = tris[k];
    if (!Number.isInteger(t) || t < 0 || t >= slots) {
      continue;
    }
    let known = true;
    for (let c = 0; c < 3; c++) {
      if (!at(positions, indices[3 * t + c], corner)) {
        known = false;
        break;
      }
      soup.push(corner[0], corner[1], corner[2]);
    }
    if (!known) {
      // Drop the partial triangle (0, 3 or 6 numbers just pushed).
      soup.length = Math.floor(soup.length / 9) * 9;
    }
  }
  return Float32Array.from(soup);
}

/** How many primitives each kind would draw (for the test hook). */
export interface RepairOverlayCounts {
  holes: number;
  nonManifold: number;
  degenerate: number;
  components: number;
}

/** Counts the overlay primitives without building three.js objects. */
export function countRepairOverlays(analysis: Analysis): RepairOverlayCounts {
  return {
    holes: analysis.loops.reduce(
      (n, l) => n + (l.vertices.length >= 2 ? (l.closed ? l.vertices.length : l.vertices.length - 1) : 0),
      0,
    ),
    nonManifold: analysis.nonManifoldEdgeList.length,
    degenerate: analysis.degenerateList.length + analysis.duplicateList.length,
    components: smallComponentTris(analysis.components).length,
  };
}

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
