// The viewport's composed clipping: the slice cutaway, the inspect
// section and the clip box share each material's `clippingPlanes`, so each
// feature only sets its own part and never wipes the others.
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { clearClip, setBoxClip, setSectionClip, setSliceClip } from "../src/viewport/clip-planes";

function material(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial();
}

describe("clip-planes", () => {
  it("starts_unclipped", () => {
    const m = material();
    expect(m.clippingPlanes).toBe(null);
  });

  it("merges_slice_section_and_box", () => {
    const m = material();
    const slice = [new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)];
    const section = new THREE.Plane(new THREE.Vector3(1, 0, 0), -1);
    const box = [new THREE.Plane(new THREE.Vector3(0, 1, 0), 2)];
    setSliceClip(m, slice);
    setSectionClip(m, section);
    setBoxClip(m, box);
    expect(m.clippingPlanes).toEqual([...slice, section, ...box]);
  });

  it("removing_one_part_keeps_the_others", () => {
    const m = material();
    const slice = [new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)];
    const section = new THREE.Plane(new THREE.Vector3(1, 0, 0), -1);
    setSliceClip(m, slice);
    setSectionClip(m, section);
    setSectionClip(m, null);
    expect(m.clippingPlanes).toEqual(slice);
    setSliceClip(m, null);
    expect(m.clippingPlanes).toBe(null);
  });

  it("clear_forgets_everything", () => {
    const m = material();
    setSliceClip(m, [new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)]);
    setSectionClip(m, new THREE.Plane(new THREE.Vector3(1, 0, 0), -1));
    clearClip(m);
    expect(m.clippingPlanes).toBe(null);
  });

  it("materials_do_not_share_parts", () => {
    const a = material();
    const b = material();
    setSliceClip(a, [new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)]);
    expect(b.clippingPlanes).toBe(null);
  });
});
