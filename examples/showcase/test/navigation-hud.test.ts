import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  buildViewCubeMatrix4,
  computeMetricScaleBar,
  projectBasisTriad,
  VIEW_CUBE_CORNERS,
  VIEW_CUBE_EDGES,
  VIEW_CUBE_FACES,
} from "../src/viewport/navigation-hud";

describe("ViewCube, XYZ triad & Live Metric Scale Bar (#232)", () => {
  it("provides_6_faces_12_edges_and_8_isometric_corners_with_unit_normals", () => {
    expect(VIEW_CUBE_FACES).toHaveLength(6);
    expect(VIEW_CUBE_EDGES).toHaveLength(12);
    expect(VIEW_CUBE_CORNERS).toHaveLength(8);
    for (const target of [...VIEW_CUBE_FACES, ...VIEW_CUBE_EDGES, ...VIEW_CUBE_CORNERS]) {
      const len = Math.hypot(target.dir[0], target.dir[1], target.dir[2]);
      expect(len).toBeCloseTo(1, 5);
      const upLen = Math.hypot(target.up[0], target.up[1], target.up[2]);
      expect(upLen).toBeCloseTo(1, 5);
    }
  });

  it("syncs_css_view_cube_matrix_and_xyz_triad_accurately_with_camera_orientation", () => {
    // Camera looking straight down from +Y onto origin (top preset: up = -Z).
    const cam = new THREE.PerspectiveCamera();
    cam.position.set(0, 10, 0);
    cam.up.set(0, 0, -1);
    cam.lookAt(0, 0, 0);

    const triad = projectBasisTriad(cam.quaternion, 20);
    const xAxis = triad.find((a) => a.axis === "X")!;
    const yAxis = triad.find((a) => a.axis === "Y")!;
    const zAxis = triad.find((a) => a.axis === "Z")!;
    expect(xAxis.x).toBeCloseTo(20, 4);
    expect(yAxis.z).toBeCloseTo(1, 4); // +Y points out toward viewer from top view
    expect(zAxis.y).toBeGreaterThan(15); // +Z points down on screen when up is -Z

    // Top face in CSS cube starts at (0, -1, 0) in local CSS space; applying matrix must move it toward viewer (+Z).
    const e = buildViewCubeMatrix4(cam.quaternion);
    const mat = new THREE.Matrix4().fromArray(e);
    const topVec = new THREE.Vector3(0, -1, 0).applyMatrix4(mat);
    expect(topVec.z).toBeCloseTo(1, 4);
  });

  it("computes_clean_1_2_5_scale_bars_for_perspective_and_orthographic_in_mm_and_in", () => {
    const persp = new THREE.PerspectiveCamera(45, 1.5, 0.1, 1000);
    const barMm = computeMetricScaleBar({
      camera: persp,
      viewportHeightPx: 600,
      targetDistanceWorld: 100,
      units: "mm",
    });
    expect(barMm.label).toMatch(/^\d+(\.\d+)? (mm|m)$/);
    expect(barMm.widthPx).toBeGreaterThanOrEqual(24);
    expect(barMm.widthPx).toBeLessThanOrEqual(180);

    const barIn = computeMetricScaleBar({
      camera: persp,
      viewportHeightPx: 600,
      targetDistanceWorld: 100,
      units: "in",
    });
    expect(barIn.label).toMatch(/^\d+(\.\d+)? in$/);
    expect(Math.abs(barIn.mmLength - barIn.valueInUnits * 25.4)).toBeLessThan(1e-6);

    const ortho = new THREE.OrthographicCamera(-50, 50, 50, -50, 0.1, 500);
    ortho.zoom = 2;
    const orthoBar = computeMetricScaleBar({
      camera: ortho,
      viewportHeightPx: 500,
      targetDistanceWorld: 100,
      units: "mm",
    });
    expect(orthoBar.label).toBe("10 mm");
  });
});
