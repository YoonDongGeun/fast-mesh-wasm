// Vertex picking for the ruler and the query (#110): the engine raycast
// snapped to the hit triangle's nearest corner and mapped into the world
// through the mesh's placement (scale-aware, #103).
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { localVertex, snapSlot, worldFromMatrix } from "./inspect-tools";

/** The engine hit's snapped vertex in world coordinates. */
export interface VertexPick {
  mesh: MeshHandle;
  /** The vertex slot, for the coloring value. */
  slot: number;
  /** The vertex in world coordinates. */
  point: [number, number, number];
}

/**
 * Raycasts the engine (its BVH, never three's per-triangle raycast) and
 * snaps to the hit triangle's nearest vertex, mapped into the world
 * through the mesh's placement.
 */
export async function pickVertex(
  scene: Scene | WorkerScene,
  origin: readonly [number, number, number],
  direction: readonly [number, number, number],
): Promise<VertexPick | null> {
  const hit = await scene.raycast(origin, direction);
  if (hit === null) {
    return null;
  }
  try {
    const views = scene.views(hit.mesh);
    const k = snapSlot(hit.barycentric);
    const slot = views.indices()[3 * hit.triangle + k];
    const point = worldFromMatrix(scene.matrix4(hit.mesh), localVertex(views.positions(), slot));
    return { mesh: hit.mesh, slot, point };
  } catch {
    return null; // removed meanwhile
  }
}
