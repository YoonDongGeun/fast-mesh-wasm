// The lattice preview in the viewport (#265): the walls of a coarse lattice,
// drawn solid where the mesh they are in stands, and that mesh see-through
// while they are up, so that walls inside a shell show. It takes no clicks.
import { useFrame } from "@react-three/fiber";
import type { LatticePreview, MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { meshState, type SceneStore } from "fast-mesh-wasm/store";
import type { BoundMesh, MeshBinding } from "fast-mesh-wasm/three";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { AppState } from "../../../app/state/app-state";
import { useLatticePreview } from "../ux/lattice-preview-state";

/** The opacity of the mesh whose lattice is previewed. */
export const SEE_THROUGH_OPACITY = 0.18;

/** The preview's scene object for `walls`, and how to free it. */
export function buildLatticePreview(walls: LatticePreview): { root: THREE.Mesh; dispose(): void } {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(walls.positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(walls.indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  const material = new THREE.MeshStandardMaterial({ color: "#7cc4ff", roughness: 0.6, side: THREE.DoubleSide });
  const root = new THREE.Mesh(geometry, material);
  root.name = "lattice-preview";
  root.matrixAutoUpdate = false;
  // Never under the pointer: clicks go to the meshes.
  root.raycast = () => {};
  return {
    root,
    dispose: () => {
      geometry.dispose();
      material.dispose();
    },
  };
}

/** Puts `root` where `mesh` stands now; hides it when the mesh is gone or hidden. */
export function placeLatticePreview(root: THREE.Object3D, store: SceneStore, mesh: MeshHandle): void {
  const snapshot = store.getSnapshot();
  root.visible = snapshot.meshes.has(mesh) && meshState(snapshot, mesh).desired.visible;
  if (root.visible) {
    root.matrix.fromArray(store.scene.matrix4(mesh));
    root.matrixWorldNeedsUpdate = true;
  }
}

export function LatticePreviewLayer(props: { app: AppState; store: SceneStore }) {
  const { app, store } = props;
  const shown = useLatticePreview(app);
  const built = useMemo(
    () => (shown === null || shown.walls.indices.length === 0 ? null : buildLatticePreview(shown.walls)),
    [shown],
  );
  useEffect(() => () => built?.dispose(), [built]);
  // With the mesh through gizmo drags.
  useFrame(() => {
    if (built !== null && shown !== null) {
      placeLatticePreview(built.root, store, shown.mesh);
    }
  });
  return built === null ? null : <primitive object={built.root} dispose={null} />;
}

/** Draws a mesh see-through while its lattice is previewed, and as it was after. */
export function LatticeSeeThrough(props: { app: AppState; mesh: MeshHandle; bound: MeshBinding | null }): null {
  const { app, mesh, bound } = props;
  const store = useScene();
  const shown = useLatticePreview(app);
  const on = shown !== null && shown.mesh === mesh && shown.walls.indices.length > 0;
  useEffect(() => {
    if (!on || bound === null) {
      return;
    }
    // The whole binding: what `<FastMesh onBind>` delivers.
    const whole = bound as BoundMesh;
    whole.setOpacity(SEE_THROUGH_OPACITY);
    return () => {
      const snapshot = store.getSnapshot();
      if (snapshot.meshes.has(mesh)) {
        whole.setOpacity(meshState(snapshot, mesh).desired.opacity);
      }
    };
  }, [on, bound, store, mesh]);
  return null;
}
