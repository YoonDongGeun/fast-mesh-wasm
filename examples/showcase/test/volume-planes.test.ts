// The 3D slice planes: the quad frames map texture space to the right
// world axes (u follows the first plane axis, v the second), so no plane
// renders mirrored or swapped.
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { planeSpec } from "../src/panels/volume-three";

/** The world spot of the top-left texel (u 0, v 1) through the frame. */
function topLeft(axis: "axial" | "coronal" | "sagittal", size: [number, number, number]): [number, number, number] {
  const spec = planeSpec(axis, size);
  const geometry = new THREE.PlaneGeometry(spec.width, spec.height);
  const uv = geometry.getAttribute("uv");
  expect([uv.getX(0), uv.getY(0)]).toEqual([0, 1]);
  if (spec.spinZ) {
    geometry.rotateZ(Math.PI / 2);
  }
  const pos = geometry.getAttribute("position");
  const v = new THREE.Vector3(pos.getX(0), pos.getY(0), pos.getZ(0));
  v.applyEuler(new THREE.Euler(spec.rotation[0], spec.rotation[1], spec.rotation[2]));
  return [v.x, v.y, v.z];
}

describe("planeSpec", () => {
  const size: [number, number, number] = [10, 20, 30];

  it("lays_axial_u_on_x_and_v_on_y", () => {
    const [x, y, z] = topLeft("axial", size);
    expect(x).toBeCloseTo(-5, 9);
    expect(y).toBeCloseTo(10, 9);
    expect(z).toBeCloseTo(0, 9);
  });

  it("lays_coronal_u_on_x_and_v_on_z", () => {
    const [x, y, z] = topLeft("coronal", size);
    expect(x).toBeCloseTo(-5, 9);
    expect(y).toBeCloseTo(0, 9);
    expect(z).toBeCloseTo(15, 9);
  });

  it("lays_sagittal_u_on_y_and_v_on_z", () => {
    const [x, y, z] = topLeft("sagittal", size);
    expect(x).toBeCloseTo(0, 9);
    expect(y).toBeCloseTo(-10, 9);
    expect(z).toBeCloseTo(15, 9);
  });
});
