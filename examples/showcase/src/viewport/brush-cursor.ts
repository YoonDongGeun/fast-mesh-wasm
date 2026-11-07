// The brush on the surface: a translucent disc lying on it at the brush radius,
// with a small circle that shows which way the brush will move the surface.
import * as THREE from "three";

/** Where the direction marker sits along the normal, as a share of the radius. */
export const NORMAL_MARKER = 0.6;

const Z = new THREE.Vector3(0, 0, 1);
const MARKER = "direction-marker";
const LINK = "direction-link";

/** A hidden cursor in `color`, one unit in radius: `placeBrushCursor` lays it on the surface. */
export function makeBrushCursor(color: number): THREE.Group {
  const onTop = { color, transparent: true, depthTest: false, depthWrite: false };
  const group = new THREE.Group();
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(1, 64),
    new THREE.MeshBasicMaterial({ ...onTop, opacity: 0.22, side: THREE.DoubleSide }),
  );
  const rim = new THREE.Mesh(
    new THREE.RingGeometry(0.97, 1, 64),
    new THREE.MeshBasicMaterial({ ...onTop, opacity: 0.85, side: THREE.DoubleSide }),
  );
  const center = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 8), new THREE.MeshBasicMaterial({ ...onTop }));
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({ ...onTop }));
  marker.name = MARKER;
  marker.visible = false;
  const link = new THREE.Line(
    new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3)),
    new THREE.LineBasicMaterial({ ...onTop, opacity: 0.85 }),
  );
  link.name = LINK;
  link.visible = false;
  // Everything frustum-culled with the disc alone would vanish with the marker still on screen.
  for (const part of [disc, rim, center, marker, link]) {
    part.frustumCulled = false;
    part.renderOrder = 10;
    group.add(part);
  }
  group.visible = false;
  return group;
}

/** Lays the cursor on the surface at `point`, facing along the unit `normal`, in its parent's frame. */
export function placeBrushCursor(
  cursor: THREE.Group,
  point: readonly [number, number, number],
  normal: readonly [number, number, number],
  radius: number,
): void {
  cursor.visible = true;
  cursor.position.set(point[0], point[1], point[2]);
  cursor.quaternion.setFromUnitVectors(Z, new THREE.Vector3(normal[0], normal[1], normal[2]));
  cursor.scale.setScalar(Math.max(radius, 1e-6));
}

/**
 * Shows the brush's way with a small circle at `offset` from the cursor's center,
 * in its parent's frame, or hides it with `null` or a zero offset. Call it after
 * `placeBrushCursor`, whose pose it reads.
 */
export function setBrushDirection(cursor: THREE.Group, offset: readonly [number, number, number] | null): void {
  const marker = cursor.getObjectByName(MARKER);
  const link = cursor.getObjectByName(LINK) as THREE.Line | undefined;
  if (marker === undefined || link === undefined) {
    return;
  }
  const shown = offset !== null && (offset[0] !== 0 || offset[1] !== 0 || offset[2] !== 0);
  marker.visible = shown;
  link.visible = shown;
  if (!shown) {
    return;
  }
  const at = new THREE.Vector3(offset[0], offset[1], offset[2])
    .applyQuaternion(cursor.quaternion.clone().invert())
    .divideScalar(cursor.scale.x);
  marker.position.copy(at);
  const line = link.geometry.getAttribute("position") as THREE.BufferAttribute;
  line.setXYZ(1, at.x, at.y, at.z);
  line.needsUpdate = true;
}

/** Eases the last normal toward `next`, so the disc does not snap at every triangle of a coarse mesh. */
export function easeNormal(
  last: readonly [number, number, number] | null,
  next: readonly [number, number, number],
  share = 0.35,
): [number, number, number] {
  if (last === null) {
    return [next[0], next[1], next[2]];
  }
  const x = last[0] + (next[0] - last[0]) * share;
  const y = last[1] + (next[1] - last[1]) * share;
  const z = last[2] + (next[2] - last[2]) * share;
  const length = Math.hypot(x, y, z);
  return length === 0 ? [next[0], next[1], next[2]] : [x / length, y / length, z / length];
}
