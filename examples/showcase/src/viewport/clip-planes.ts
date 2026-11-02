// The viewport's composed clipping: the slice cutaway, the inspect section
// plane and the inspect clip box all clip the same mesh materials, so one
// composer owns every material's `clippingPlanes` and each feature only
// sets its own part. Without it the last writer would wipe the others
// (the slice overlay assigns the array wholesale).
import * as THREE from "three";

interface ClipParts {
  slice: THREE.Plane[] | null;
  section: THREE.Plane | null;
  box: THREE.Plane[] | null;
}

const parts = new WeakMap<THREE.Material, ClipParts>();

function compose(material: THREE.Material): void {
  const p = parts.get(material);
  const planes: THREE.Plane[] = [
    ...(p?.slice ?? []),
    ...(p?.section !== null && p?.section !== undefined ? [p.section] : []),
    ...(p?.box ?? []),
  ];
  material.clippingPlanes = planes.length === 0 ? null : planes;
  material.needsUpdate = true;
}

function edit(material: THREE.Material, patch: Partial<ClipParts>): void {
  const prev = parts.get(material);
  const next: ClipParts = { slice: null, section: null, box: null, ...prev, ...patch };
  if (next.slice === null && next.section === null && next.box === null) {
    parts.delete(material);
  } else {
    parts.set(material, next);
  }
  compose(material);
}

/** The slice cutaway's planes (`null` hides the cutaway), keeping the section and box parts. */
export function setSliceClip(material: THREE.Material, planes: THREE.Plane[] | null): void {
  edit(material, { slice: planes });
}

/** The inspect section's plane (`null` removes it), keeping the slice and box parts. */
export function setSectionClip(material: THREE.Material, plane: THREE.Plane | null): void {
  edit(material, { section: plane });
}

/** The inspect clip box's faces (`null` removes them), keeping the slice and section parts. */
export function setBoxClip(material: THREE.Material, planes: THREE.Plane[] | null): void {
  edit(material, { box: planes });
}

/** Forgets every part (a removed mesh's material): back to no clipping. */
export function clearClip(material: THREE.Material): void {
  parts.delete(material);
  material.clippingPlanes = null;
}
