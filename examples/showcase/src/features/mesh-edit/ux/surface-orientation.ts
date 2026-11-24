// Surface frames for the print-prep tools: the world-space point and
// outward unit normal of a raycast hit, and the rotation that stands a drain
// hole or stamp on that normal (local +Z along it, optionally twisted).
import type { MeshHandle, SceneLike, Vec3 } from "fast-mesh-wasm";
import * as THREE from "three";

export interface SurfaceOrientation {
  axis: Vec3;
  quaternion: [number, number, number, number];
  matrix4: number[];
}

/**
 * Computes an orientation aligning local +Z with `normal`, optionally twisted around `normal`.
 */
export function orientationFromSurfaceNormal(normal: Readonly<Vec3>, twistRad = 0): SurfaceOrientation {
  const n = new THREE.Vector3(normal[0], normal[1], normal[2]);
  if (n.lengthSq() <= 1e-24 || !Number.isFinite(n.x) || !Number.isFinite(n.y) || !Number.isFinite(n.z)) {
    n.set(0, 0, 1);
  } else {
    n.normalize();
  }
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
  if (Number.isFinite(twistRad) && twistRad !== 0) {
    const twist = new THREE.Quaternion().setFromAxisAngle(n, twistRad);
    q.premultiply(twist);
  }
  q.normalize();
  const m = new THREE.Matrix4().makeRotationFromQuaternion(q);
  return {
    axis: [n.x, n.y, n.z],
    quaternion: [q.x, q.y, q.z, q.w],
    matrix4: Array.from(m.elements),
  };
}

/**
 * Computes world-space hit point and outward unit normal from a scene raycast hit.
 */
export function computeHitPointAndWorldNormal(
  scene: Pick<SceneLike, "views" | "matrix4">,
  hit: { mesh: MeshHandle; triangle: number; point: Readonly<Vec3> },
): { point: Vec3; normal: Vec3 } | null {
  const views = scene.views(hit.mesh);
  const indices = views.indices();
  const positions = views.positions();
  const t = hit.triangle;
  if (t < 0 || t * 3 + 2 >= indices.length) {
    return null;
  }
  const i0 = indices[t * 3] * 3;
  const i1 = indices[t * 3 + 1] * 3;
  const i2 = indices[t * 3 + 2] * 3;
  const ax = positions[i0];
  const ay = positions[i0 + 1];
  const az = positions[i0 + 2];
  const e1x = positions[i1] - ax;
  const e1y = positions[i1 + 1] - ay;
  const e1z = positions[i1 + 2] - az;
  const e2x = positions[i2] - ax;
  const e2y = positions[i2 + 1] - ay;
  const e2z = positions[i2 + 2] - az;
  const localNormal = new THREE.Vector3(
    e1y * e2z - e1z * e2y,
    e1z * e2x - e1x * e2z,
    e1x * e2y - e1y * e2x,
  );
  if (localNormal.lengthSq() <= 1e-24) {
    return null;
  }
  const modelMatrix = new THREE.Matrix4().fromArray(Array.from(scene.matrix4(hit.mesh)));
  const normalMatrix = new THREE.Matrix3().getNormalMatrix(modelMatrix);
  const worldNormal = localNormal.applyMatrix3(normalMatrix);
  if (worldNormal.lengthSq() <= 1e-24) {
    return null;
  }
  worldNormal.normalize();
  return {
    point: [hit.point[0], hit.point[1], hit.point[2]],
    normal: [worldNormal.x, worldNormal.y, worldNormal.z],
  };
}
