// The canvas's raycasts, all through the engine's BVH (`scene.raycast`, async
// on the Worker), never three's per-triangle `Mesh.raycast`: the ray through
// a pointer, a raycast that looks past hidden meshes, and what a hit means in
// its mesh's local positions. The hover and the brushes throttle theirs to
// the periods here, one raycast at a time.
import type { MeshHandle, RayHit, Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import * as THREE from "three";

/** The hover raycast runs at most this often while the pointer moves. */
export const HOVER_PERIOD_MS = 50;
/** The sculpt raycast runs at most this often while a stroke paints. */
export const STROKE_PERIOD_MS = 16;

/** Aims `caster` from `camera` through the page point (`x`, `y`, CSS pixels) on `canvas`, and returns its ray. */
export function rayThroughPointer(
  caster: THREE.Raycaster,
  canvas: HTMLCanvasElement,
  camera: THREE.Camera,
  x: number,
  y: number,
): THREE.Ray {
  const r = canvas.getBoundingClientRect();
  caster.setFromCamera(new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2), camera);
  return caster.ray;
}

/** Raycasts skipping invisible (hidden) meshes so hidden meshes never occlude sculpt/mask hits (#229). */
export async function raycastVisibleSurface(
  scene: Scene | WorkerScene,
  origin: readonly [number, number, number],
  direction: readonly [number, number, number],
  isVisible: (mesh: MeshHandle) => boolean,
): Promise<Awaited<ReturnType<Scene["raycast"]>>> {
  let minDistance = 0;
  for (let attempt = 0; attempt < 6; attempt++) {
    const hit = await scene.raycast(origin, direction, minDistance > 0 ? { minDistance } : undefined);
    if (hit === null || isVisible(hit.mesh)) {
      return hit;
    }
    minDistance = hit.distance + 1e-4;
  }
  return null;
}

/** `raycastVisibleSurface` from `camera` through the page point (`x`, `y`) on `canvas`. */
export function castVisibleSurface(
  scene: Scene | WorkerScene,
  caster: THREE.Raycaster,
  canvas: HTMLCanvasElement,
  camera: THREE.Camera,
  x: number,
  y: number,
  isVisible: (mesh: MeshHandle) => boolean,
): Promise<Awaited<ReturnType<Scene["raycast"]>>> {
  const { origin, direction } = rayThroughPointer(caster, canvas, camera, x, y);
  return raycastVisibleSurface(
    scene,
    [origin.x, origin.y, origin.z],
    [direction.x, direction.y, direction.z],
    isVisible,
  );
}

/** The vertex slot nearest to a hit: the hit triangle's corner with the largest barycentric weight. */
export function nearestCorner(
  scene: Pick<Scene | WorkerScene, "views">,
  hit: Pick<RayHit, "mesh" | "triangle" | "barycentric">,
): number {
  const corners = scene.views(hit.mesh).indices().subarray(3 * hit.triangle, 3 * hit.triangle + 3);
  const [u, v, w] = hit.barycentric;
  const k = u >= v && u >= w ? 0 : v >= w ? 1 : 2;
  return corners[k];
}

/** A hit interpolated in its mesh's local positions, the frame a brush center is in. */
export function localHitPoint(
  scene: Pick<Scene | WorkerScene, "views">,
  hit: Pick<RayHit, "mesh" | "triangle" | "barycentric">,
): readonly [number, number, number] {
  const corners = scene.views(hit.mesh).indices().subarray(3 * hit.triangle, 3 * hit.triangle + 3);
  const positions = scene.views(hit.mesh).positions();
  const [u, v, w] = hit.barycentric;
  const at = (s: number): readonly [number, number, number] => [
    positions[3 * s],
    positions[3 * s + 1],
    positions[3 * s + 2],
  ];
  const [a, b, c] = [at(corners[0]), at(corners[1]), at(corners[2])];
  return [
    u * a[0] + v * b[0] + w * c[0],
    u * a[1] + v * b[1] + w * c[1],
    u * a[2] + v * b[2] + w * c[2],
  ];
}
