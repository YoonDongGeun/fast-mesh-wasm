// The Create dialog's live preview (#259): the generated shape drawn
// translucent at the dialog's centre and rotation, with its wireframe on
// smaller shapes. It takes no clicks. The geometry is rebuilt when a new
// shape arrives; centre and rotation only move it.
import { useThree } from "@react-three/fiber";
import type { PrimitiveMesh } from "fast-mesh-wasm";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import type { AppState } from "../../../app/state/app-state";
import { useCreateDialog } from "../ux/create-dialog-state";

/** Shapes up to this many triangles also draw their wireframe. */
export const PREVIEW_WIRE_TRIANGLES = 200_000;

/** The preview's scene objects for `mesh`, and how to free them. */
export function buildCreatePreview(mesh: PrimitiveMesh): { root: THREE.Group; dispose(): void } {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  const fill = new THREE.MeshStandardMaterial({
    color: "#7cc4ff",
    roughness: 0.6,
    transparent: true,
    opacity: 0.45,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const root = new THREE.Group();
  root.name = "create-preview";
  root.add(new THREE.Mesh(geometry, fill));
  let wire: THREE.Material | null = null;
  if (mesh.indices.length / 3 <= PREVIEW_WIRE_TRIANGLES) {
    wire = new THREE.MeshBasicMaterial({
      color: "#d6ecff",
      wireframe: true,
      transparent: true,
      opacity: 0.3,
      depthWrite: false,
    });
    root.add(new THREE.Mesh(geometry, wire));
  }
  root.traverse((o) => {
    o.renderOrder = 10;
    // Never under the pointer: clicks go to the meshes behind it.
    o.raycast = () => {};
  });
  return {
    root,
    dispose: () => {
      geometry.dispose();
      fill.dispose();
      wire?.dispose();
    },
  };
}

export function CreatePreviewLayer(props: { app: AppState }) {
  const preview = useCreateDialog(props.app, (s) => s.preview);
  const invalidate = useThree((s) => s.invalidate);
  const mesh = preview?.mesh ?? null;
  const built = useMemo(() => (mesh === null ? null : buildCreatePreview(mesh)), [mesh]);
  useEffect(() => () => built?.dispose(), [built]);
  useLayoutEffect(() => {
    if (built !== null && preview !== null) {
      const [x, y, z] = preview.centre;
      const [rx, ry, rz] = preview.rotationDeg.map((d) => THREE.MathUtils.degToRad(d));
      built.root.position.set(x, y, z);
      built.root.rotation.set(rx, ry, rz, "XYZ");
    }
    invalidate();
  }, [built, preview, invalidate]);
  return built === null ? null : <primitive object={built.root} />;
}
