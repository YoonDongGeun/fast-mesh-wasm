// The wireframe overlay: the solid mesh keeps its shaded material while a
// wireframe child draws the edges on top (idempotent, sharing the geometry).
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { setWireOverlay } from "../src/viewport/wire";

function mesh(): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array([0, 1, 2]), 1));
  return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial());
}

describe("setWireOverlay", () => {
  it("lays_wire_over_the_solid", () => {
    const m = mesh();
    setWireOverlay(m, true);
    expect(m.children.length).toBe(1);
    const overlay = m.children[0] as THREE.Mesh;
    expect(overlay.geometry).toBe(m.geometry);
    expect((overlay.material as THREE.MeshBasicMaterial).wireframe).toBe(true);
    // The solid stays shaded: its own material never flips to wireframe.
    expect((m.material as THREE.MeshStandardMaterial).wireframe).toBe(false);
  });

  it("is_idempotent", () => {
    const m = mesh();
    setWireOverlay(m, true);
    setWireOverlay(m, true);
    expect(m.children.length).toBe(1);
    setWireOverlay(m, false);
    setWireOverlay(m, false);
    expect(m.children.length).toBe(0);
  });

  it("off_restores_the_solid", () => {
    const m = mesh();
    const solid = m.material as THREE.MeshStandardMaterial;
    setWireOverlay(m, true);
    expect(solid.polygonOffset).toBe(true);
    setWireOverlay(m, false);
    expect(m.children.length).toBe(0);
    expect(solid.polygonOffset).toBe(false);
    expect(solid.polygonOffsetFactor).toBe(0);
    expect(solid.polygonOffsetUnits).toBe(0);
  });
});
