// The wireframe overlay: a wireframe child sharing the bound mesh's
// geometry, so the edges draw over the shaded solid instead of replacing
// it (a hollow look). Idempotent; the geometry belongs to the binding and
// is never disposed here, and the overlay material is shared.
import * as THREE from "three";

const TAG = "fastMeshWireOverlay";

const overlayMaterial = new THREE.MeshBasicMaterial({
  wireframe: true,
  color: 0x101216,
  transparent: true,
  opacity: 0.55,
  depthTest: true,
  depthWrite: false,
});

function overlayOf(mesh: THREE.Mesh): THREE.Mesh | undefined {
  return mesh.children.find((c): c is THREE.Mesh => c.userData[TAG] === true);
}

/** Whether `mesh` currently shows the wireframe overlay. */
export function hasWireOverlay(mesh: THREE.Mesh): boolean {
  return overlayOf(mesh) !== undefined;
}

/**
 * Shows (`on`) or removes the wireframe overlay over `mesh`'s solid. While
 * shown, the solid's material pushes back a polygon offset so the lines sit
 * cleanly on top; off restores its defaults (the binding never touches
 * `polygonOffset`, so this is additive).
 */
export function setWireOverlay(mesh: THREE.Mesh, on: boolean): void {
  const solid = mesh.material as THREE.Material;
  const overlay = overlayOf(mesh);
  if (on) {
    if (overlay === undefined) {
      const child = new THREE.Mesh(mesh.geometry, overlayMaterial);
      child.userData[TAG] = true;
      // The binding's object never updates its matrix (`matrixAutoUpdate`
      // is off): the child inherits the mesh's transform as-is.
      child.matrixAutoUpdate = false;
      mesh.add(child);
    }
    solid.polygonOffset = true;
    solid.polygonOffsetFactor = 1;
    solid.polygonOffsetUnits = 1;
  } else {
    if (overlay !== undefined) {
      mesh.remove(overlay);
    }
    solid.polygonOffset = false;
    solid.polygonOffsetFactor = 0;
    solid.polygonOffsetUnits = 0;
  }
}
