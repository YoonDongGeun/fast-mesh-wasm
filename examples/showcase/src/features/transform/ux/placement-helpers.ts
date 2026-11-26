// The placement helpers' store calls (#256). Move to origin, lay flat, align
// to axis and 3-2-1 each set the mesh's placement once (one undo step).
// Apply transform bakes the placement into the vertices: it sets the
// identity placement and then writes the world positions the old one put
// the vertices at (an edit's positions are world positions), both inside one
// history step, so one undo puts the placement and the vertices back. Where
// the mesh is too far from its origin for that, a translation stays.
import type { MeshHandle, PlacementInput, SceneLike, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { meshOrigin } from "../../../shared/ux/mesh-center";
import {
  bakedPositions,
  isIdentityPlacement,
  isTranslation,
  keptTranslation,
  type PlacedMesh,
  samePlacement,
  triangleNormal,
  worldPoint,
} from "./placement-math";

/** The scene calls the helpers read a mesh through. */
export type PlacementScene = Pick<SceneLike, "views" | "placement" | "matrix4">;

/** The mesh as the helpers read it: render positions, triangles, world matrix and placement. */
export function readPlacedMesh(scene: PlacementScene, mesh: MeshHandle): PlacedMesh {
  const views = scene.views(mesh);
  return {
    positions: views.positions(),
    indices: views.indices(),
    matrix: scene.matrix4(mesh),
    placement: scene.placement(mesh),
  };
}

/**
 * Sets the placement `compute` gives the mesh, one undo step. Returns it,
 * or `null` when it equals the current one (nothing is recorded then).
 */
export function placeMesh(
  store: SceneStore,
  mesh: MeshHandle,
  compute: (placed: PlacedMesh) => PlacementInput,
): PlacementInput | null {
  const placed = readPlacedMesh(store.scene, mesh);
  const next = compute(placed);
  if (samePlacement(placed.placement, next as Parameters<typeof samePlacement>[1])) {
    return null;
  }
  store.setPlacement(mesh, next);
  return next;
}

/**
 * Apply transform: the placement becomes the identity and the vertices keep
 * their world positions, as one undo step. Resolves `false` when the
 * placement already is the identity.
 *
 * A mesh further than 1024 times its size from its origin, which is for good,
 * keeps a translation: without it the `f32` positions would lose the shape
 * (0.008 at 1e5). Resolves `false` when it already has no other placement.
 */
export async function applyTransform(store: SceneStore, mesh: MeshHandle): Promise<boolean> {
  const placed = readPlacedMesh(store.scene, mesh);
  if (isIdentityPlacement(placed.placement)) {
    return false;
  }
  const { vertices, positions } = bakedPositions(placed);
  const translation = keptTranslation(positions, meshOrigin(store.scene, mesh));
  if (translation.some((c) => c !== 0) && isTranslation(placed.placement)) {
    return false;
  }
  store.beginStep();
  try {
    store.setPlacement(mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation });
    if (vertices.length > 0) {
      // Under the identity placement these world positions are the new local ones.
      await store.edit(mesh, [{ op: "setPositions", vertices, positions }]);
    }
    store.endStep();
  } catch (err) {
    store.cancelStep();
    throw err;
  }
  return true;
}

/** The outward world normal of the mesh's triangle `triangle`, or `null` when it is degenerate or gone. */
export function worldTriangleNormal(scene: PlacementScene, mesh: MeshHandle, triangle: number): Vec3 | null {
  const placed = readPlacedMesh(scene, mesh);
  const { positions, indices, matrix } = placed;
  if (triangle < 0 || 3 * triangle + 2 >= indices.length) {
    return null;
  }
  const corner = (k: number): Vec3 => {
    const v = indices[3 * triangle + k];
    return worldPoint(matrix, positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
  };
  return triangleNormal(corner(0), corner(1), corner(2));
}
