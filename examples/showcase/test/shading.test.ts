import { describe, expect, it } from "vitest";
import * as THREE from "three";
import type { MeshBinding } from "fast-mesh-wasm/three";
import {
  buildStudioMatcapRgba,
  clearMeshDiagnosticShading,
  computeCreasedGeometry,
  DIAGNOSTIC_SHADING_OPTIONS,
  installDiagnosticShader,
  syncMeshDiagnosticShading,
} from "../src/features/view/ui/shading";
import { hasWireOverlay } from "../src/features/view/ui/wire";

function twoTriangleHinge(angleDeg: number) {
  const rad = THREE.MathUtils.degToRad(angleDeg);
  // Tri 0 lies in XY plane (normal +Z); Tri 1 hinges along edge x-axis (0,0,0)->(1,0,0).
  const y2 = -Math.cos(rad);
  const z2 = Math.sin(rad);
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
    0, y2, z2,
  ]);
  const indices = new Uint32Array([
    0, 1, 2,
    1, 0, 3,
  ]);
  return { positions, indices };
}

describe("Diagnostic Shading Modes (#232)", () => {
  it("exposes_all_6_diagnostic_shading_modes_with_procedural_studio_matcap", () => {
    expect(DIAGNOSTIC_SHADING_OPTIONS.map((o) => o.mode)).toEqual([
      "smooth",
      "flat",
      "autoCrease",
      "wireframeShaded",
      "matcap",
      "backfaceWarning",
    ]);
    const rgba = buildStudioMatcapRgba(64);
    expect(rgba.length).toBe(64 * 64 * 4);
    // Center texel has bright lit values and opaque alpha.
    const centerIdx = (32 * 64 + 32) * 4;
    expect(rgba[centerIdx]).toBeGreaterThan(100);
    expect(rgba[centerIdx + 3]).toBe(255);
  });

  it("computes_smooth_normals_below_threshold_and_splits_above_dihedral_threshold", () => {
    const { positions, indices } = twoTriangleHinge(60); // 60 deg dihedral angle between faces
    // High threshold (90 deg) -> smooth averaged normal at shared vertices 0 and 1.
    const smooth = computeCreasedGeometry(positions, indices, 90);
    expect(smooth.normals[0]).toBeCloseTo(smooth.normals[9], 5);
    expect(smooth.normals[1]).toBeCloseTo(smooth.normals[10], 5);
    expect(smooth.normals[2]).toBeCloseTo(smooth.normals[11], 5);

    // Low threshold (30 deg < 60 deg) -> sharp crease splits vertex normals cleanly.
    const creased = computeCreasedGeometry(positions, indices, 30);
    const n0 = new THREE.Vector3(creased.normals[0], creased.normals[1], creased.normals[2]);
    const n1 = new THREE.Vector3(creased.normals[9], creased.normals[10], creased.normals[11]);
    expect(n0.angleTo(n1)).toBeGreaterThan(THREE.MathUtils.degToRad(45));
  });

  it("switches_material_flags_uniforms_wireframe_and_autoc_overlays_cleanly", () => {
    const { positions, indices } = twoTriangleHinge(75);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setIndex(new THREE.BufferAttribute(indices, 1));
    const material = new THREE.MeshStandardMaterial({ color: 0xcccccc, side: THREE.DoubleSide });
    const meshObj = new THREE.Mesh(geom, material);
    const bound: MeshBinding = { mesh: {} as never, object: meshObj };

    installDiagnosticShader(material);
    const u = material.userData.__diagUniforms as { fmDiagMode: { value: number } };

    let state = syncMeshDiagnosticShading({ bound, material, mode: "flat", globalWireframe: false, creaseAngleDeg: 35, prev: null });
    expect(material.flatShading).toBe(true);
    expect(u.fmDiagMode.value).toBe(0);

    state = syncMeshDiagnosticShading({ bound, material, mode: "matcap", globalWireframe: false, creaseAngleDeg: 35, prev: state });
    expect(material.flatShading).toBe(false);
    expect(u.fmDiagMode.value).toBe(1);

    state = syncMeshDiagnosticShading({ bound, material, mode: "backfaceWarning", globalWireframe: false, creaseAngleDeg: 35, prev: state });
    expect(u.fmDiagMode.value).toBe(2);

    state = syncMeshDiagnosticShading({ bound, material, mode: "wireframeShaded", globalWireframe: false, creaseAngleDeg: 35, prev: state });
    expect(hasWireOverlay(meshObj)).toBe(true);

    state = syncMeshDiagnosticShading({ bound, material, mode: "autoCrease", globalWireframe: false, creaseAngleDeg: 30, prev: state });
    expect(meshObj.getObjectByName("fastMeshAutoCreaseOverlay")).toBeDefined();
    expect(hasWireOverlay(meshObj)).toBe(false);

    clearMeshDiagnosticShading(bound, material);
    expect(meshObj.getObjectByName("fastMeshAutoCreaseOverlay")).toBeUndefined();
    expect(material.flatShading).toBe(false);
    expect(u.fmDiagMode.value).toBe(0);

    geom.dispose();
    material.dispose();
  });
});
