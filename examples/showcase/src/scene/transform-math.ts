// Pure transform & mesh-local scaling math for TransformPanel and Scale gizmo mode (#195).
// Scene placements are rigid [rotation, translation]; scaling updates local vertex positions
// via `store.edit` (`setPositions`) so every scale operation integrates cleanly with undo/redo.
import type { EditResult, MeshHandle, Placement, PlacementInput } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import * as THREE from "three";
import type { Vec3 } from "../load/mesh-data";

export interface TransformValues {
  position: [number, number, number];
  rotationDeg: [number, number, number];
}

/** Converts a row-major 3x3 rotation matrix to Euler XYZ angles in degrees. */
export function mat3ToEulerDeg(rotation: readonly number[]): [number, number, number] {
  const m = new THREE.Matrix4().set(
    rotation[0],
    rotation[1],
    rotation[2],
    0,
    rotation[3],
    rotation[4],
    rotation[5],
    0,
    rotation[6],
    rotation[7],
    rotation[8],
    0,
    0,
    0,
    0,
    1,
  );
  const euler = new THREE.Euler().setFromRotationMatrix(m, "XYZ");
  return [
    THREE.MathUtils.radToDeg(euler.x),
    THREE.MathUtils.radToDeg(euler.y),
    THREE.MathUtils.radToDeg(euler.z),
  ];
}

/** Converts Euler XYZ angles in degrees to a row-major 3x3 proper rotation matrix. */
export function eulerDegToMat3(deg: readonly [number, number, number]): [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
] {
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(deg[0]),
    THREE.MathUtils.degToRad(deg[1]),
    THREE.MathUtils.degToRad(deg[2]),
    "XYZ",
  );
  const m = new THREE.Matrix4().makeRotationFromEuler(euler);
  const e = m.elements; // column-major
  return [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]];
}

/** Reads position and Euler XYZ degrees from a Scene Placement. */
export function placementToValues(placement: Placement): TransformValues {
  return {
    position: [placement.translation[0], placement.translation[1], placement.translation[2]],
    rotationDeg: mat3ToEulerDeg(placement.rotation),
  };
}

/** Builds a rigid PlacementInput from world translation and Euler XYZ degrees. */
export function valuesToPlacement(values: TransformValues): PlacementInput {
  return {
    translation: [values.position[0], values.position[1], values.position[2]],
    rotation: eulerDegToMat3(values.rotationDeg),
  };
}

/** Computes axis-aligned bounds across live triangles (falling back to all vertex slots if none live). */
export function computeLocalExtents(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
): { min: Vec3; max: Vec3; extents: Vec3 } {
  let x0 = Infinity;
  let y0 = Infinity;
  let z0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  let z1 = -Infinity;
  const expand = (v: number) => {
    const x = positions[3 * v];
    const y = positions[3 * v + 1];
    const z = positions[3 * v + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return;
    }
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
    if (z < z0) z0 = z;
    if (z > z1) z1 = z;
  };
  let anyLive = false;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    const a = indices[k];
    const b = indices[k + 1];
    const c = indices[k + 2];
    if (a !== 0 || b !== 0 || c !== 0) {
      anyLive = true;
      expand(a);
      expand(b);
      expand(c);
    }
  }
  if (!anyLive) {
    const n = Math.floor(positions.length / 3);
    for (let v = 0; v < n; v++) {
      expand(v);
    }
  }
  if (!(x0 <= x1)) {
    return { min: [0, 0, 0], max: [0, 0, 0], extents: [0, 0, 0] };
  }
  return {
    min: [x0, y0, z0],
    max: [x1, y1, z1],
    extents: [x1 - x0, y1 - y0, z1 - z0],
  };
}

/** Returns live vertex ids present in at least one live triangle (or all vertices if soup has none flagged dead). */
export function liveVertexIds(vertexCount: number, indices: ArrayLike<number>): Uint32Array {
  const seen = new Uint8Array(vertexCount);
  let count = 0;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    const a = indices[k];
    const b = indices[k + 1];
    const c = indices[k + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue;
    }
    if (a < vertexCount && seen[a] === 0) {
      seen[a] = 1;
      count++;
    }
    if (b < vertexCount && seen[b] === 0) {
      seen[b] = 1;
      count++;
    }
    if (c < vertexCount && seen[c] === 0) {
      seen[c] = 1;
      count++;
    }
  }
  const out = new Uint32Array(count);
  let p = 0;
  for (let i = 0; i < vertexCount; i++) {
    if (seen[i] === 1) {
      out[p++] = i;
    }
  }
  return out;
}

/**
 * Computes world-space `setPositions` payloads scaling local vertex positions by `factor`
 * around `(0, 0, 0)` in mesh-local space, then transforming back through `matrix4`.
 */
export function buildScaledWorldPositions(
  localPositions: ArrayLike<number>,
  indices: ArrayLike<number>,
  matrix4: ArrayLike<number>,
  factor: readonly [number, number, number],
): { vertices: Uint32Array; positions: Float64Array } {
  const vertexCount = Math.floor(localPositions.length / 3);
  const vertices = liveVertexIds(vertexCount, indices);
  const out = new Float64Array(vertices.length * 3);
  const m = new THREE.Matrix4().fromArray(matrix4);
  const v = new THREE.Vector3();
  const [sx, sy, sz] = factor;
  for (let i = 0; i < vertices.length; i++) {
    const slot = vertices[i];
    v.set(
      localPositions[3 * slot] * sx,
      localPositions[3 * slot + 1] * sy,
      localPositions[3 * slot + 2] * sz,
    ).applyMatrix4(m);
    out[3 * i] = v.x;
    out[3 * i + 1] = v.y;
    out[3 * i + 2] = v.z;
  }
  return { vertices, positions: out };
}

/**
 * Scales a mesh cleanly in mesh-local space as a single undoable edit step.
 */
export async function applyMeshScale(
  store: SceneStore,
  mesh: MeshHandle,
  factor: readonly [number, number, number],
): Promise<EditResult | null> {
  const [sx, sy, sz] = factor;
  if (
    !Number.isFinite(sx) ||
    !Number.isFinite(sy) ||
    !Number.isFinite(sz) ||
    sx <= 0 ||
    sy <= 0 ||
    sz <= 0
  ) {
    throw new Error("Scale factors must be positive finite numbers");
  }
  if (Math.abs(sx - 1) < 1e-9 && Math.abs(sy - 1) < 1e-9 && Math.abs(sz - 1) < 1e-9) {
    return null;
  }
  const views = store.scene.views(mesh);
  const positions = views.positions();
  const indices = views.indices();
  const matrix = store.scene.matrix4(mesh);
  const { vertices, positions: nextWorld } = buildScaledWorldPositions(positions, indices, matrix, [sx, sy, sz]);
  if (vertices.length === 0) {
    return null;
  }
  return store.edit(mesh, [{ op: "setPositions", vertices, positions: nextWorld }]);
}
