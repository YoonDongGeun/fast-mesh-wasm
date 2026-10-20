// The app's meshes: a scene mesh, its name and the material `<FastMesh>`
// draws it with. Shared by the canvas (App.tsx) and the panels (Panel.tsx).
import type { MeshHandle } from "fast-mesh-wasm";
import type { DistanceColoring, SceneStore } from "fast-mesh-wasm/store";
import * as THREE from "three";
import { bounds, type MeshData, type Vec3 } from "./meshes";

/** The section plane's normal: it cuts across z, keeping the +z side. */
export const SECTION_NORMAL: Vec3 = [0, 0, 1];

/** Distances beyond this are not computed (and show the base colour). */
export const MAX_DISTANCE = 1.5;

export interface Item {
  /** A React key. */
  id: number;
  mesh: MeshHandle;
  name: string;
  /** Owned by the app (one per mesh, as `bindMesh` requires); disposed when the mesh is replaced. */
  material: THREE.MeshStandardMaterial;
  /** Where the mesh was added and its size: a dropped STL replacing it is fitted there. */
  center: Vec3;
  radius: number;
}

let nextId = 0;

/** Adds `data` to the store and returns its item. */
export function makeItem(store: SceneStore, name: string, data: MeshData, color: THREE.ColorRepresentation): Item {
  const mesh = store.addMesh(data.positions, data.indices);
  const { center, radius } = bounds(data.positions);
  // Double-sided: the open plate, and the inside of a mesh cut by the section, stay visible.
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.6, side: THREE.DoubleSide });
  return { id: nextId++, mesh, name, material, center, radius };
}

/** The colouring the toggle requests: turbo map over [0, MAX_DISTANCE] to `target`. */
export function coloringTo(target: MeshHandle): DistanceColoring {
  return { distanceTo: target, maxDistance: MAX_DISTANCE, map: "turbo", range: [0, MAX_DISTANCE] };
}
