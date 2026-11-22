// Integration & UI tests for Issue #234 using real WASM `Scene` + `SceneStore` + `AppState`:
// - InspectPanel preserved `data-action="inspect-*"` controls & `SurfaceFieldControls`
// - `fitPrimitive`, `alignToFittedPrimitive`, `sliceContours`, `sliceContourStack`, `sliceContoursExport` (SVG) & DXF export
// - Point cloud parse (`.xyz`, `.pts`, `.ply`, `.pcd`) & mesh/synthetic sampling -> SOR/ROR/Voxel/Normals -> `reconstructPointCloud` -> `app.adoptSceneMesh`
// - Overlay geometry builders (`buildPointCloudGeometry`, `buildContourSegmentsGeometry`, `buildPrimitiveGuideSegments`)
import {
  type ContourSlice,
  type FittedPrimitiveReport,
  Scene,
  init,
  packPointCloud,
  parsePointCloudBytes,
  parsePointCloudText,
  pointCloudEstimateNormals,
  pointCloudRor,
  pointCloudSor,
  pointCloudVoxelDownsample,
  sliceContoursExport,
} from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as THREE from "three";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { InspectPanel } from "../src/features/inspect/ui/InspectPanel";
import { exportContourSlicesDxf } from "../src/features/inspect/ux/use-contour-slices";
import {
  PointsPanel,
  computePointCloudBounds,
  createSyntheticSphereCloud,
  parsePointCloudFileContent,
  sampleMeshToPointCloud,
} from "../src/features/points/ui/PointsPanel";
import { AppState } from "../src/app/state/app-state";
import {
  MetrologyOverlaysLayer,
  buildContourSegmentsGeometry,
  buildPointCloudGeometry,
  buildPrimitiveGuideSegments,
} from "../src/features/points/ui/MetrologyOverlaysLayer";

const cleanups: (() => void)[] = [];

beforeAll(async () => {
  await init();
});

afterEach(() => {
  vi.restoreAllMocks();
  for (const fn of cleanups.splice(0).reverse()) {
    fn();
  }
});

/** Builds a watertight UV sphere mesh with verified outward CCW winding. */
function buildSphereMesh(radius = 2, latSeg = 16, lonSeg = 24): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  const verts: number[] = [];
  const tris: number[] = [];
  verts.push(0, radius, 0);
  for (let i = 1; i < latSeg; i++) {
    const phi = (i / latSeg) * Math.PI;
    const y = Math.cos(phi) * radius;
    const r = Math.sin(phi) * radius;
    for (let j = 0; j < lonSeg; j++) {
      const theta = (j / lonSeg) * Math.PI * 2;
      verts.push(Math.cos(theta) * r, y, Math.sin(theta) * r);
    }
  }
  const bottomIdx = 1 + (latSeg - 1) * lonSeg;
  verts.push(0, -radius, 0);

  const pushTri = (a: number, b: number, c: number) => {
    const ax = verts[3 * a], ay = verts[3 * a + 1], az = verts[3 * a + 2];
    const bx = verts[3 * b], by = verts[3 * b + 1], bz = verts[3 * b + 2];
    const cx = verts[3 * c], cy = verts[3 * c + 1], cz = verts[3 * c + 2];
    const nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
    const ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
    const nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const dot = nx * (ax + bx + cx) + ny * (ay + by + cy) + nz * (az + bz + cz);
    if (dot >= 0) {
      tris.push(a, b, c);
    } else {
      tris.push(a, c, b);
    }
  };

  for (let j = 0; j < lonSeg; j++) {
    const nextJ = (j + 1) % lonSeg;
    pushTri(0, 1 + j, 1 + nextJ);
  }
  for (let i = 0; i < latSeg - 2; i++) {
    const row0 = 1 + i * lonSeg;
    const row1 = 1 + (i + 1) * lonSeg;
    for (let j = 0; j < lonSeg; j++) {
      const nextJ = (j + 1) % lonSeg;
      pushTri(row0 + j, row1 + j, row1 + nextJ);
      pushTri(row0 + j, row1 + nextJ, row0 + nextJ);
    }
  }
  const lastRow = 1 + (latSeg - 2) * lonSeg;
  for (let j = 0; j < lonSeg; j++) {
    const nextJ = (j + 1) % lonSeg;
    pushTri(bottomIdx, lastRow + nextJ, lastRow + j);
  }
  return {
    positions: Float64Array.from(verts),
    indices: Uint32Array.from(tris),
  };
}

/** Builds a watertight cylinder tilted away from +Y for primitive fitting & datum alignment. */
function buildTiltedCylinderMesh(radius = 1.5, height = 6, segs = 24): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 5);
  const verts: number[] = [];
  const tris: number[] = [];
  const pushVert = (x: number, y: number, z: number) => {
    const v = new THREE.Vector3(x, y, z).applyQuaternion(q);
    verts.push(v.x, v.y, v.z);
  };
  const half = height / 2;
  pushVert(0, 0, -half);
  pushVert(0, 0, half);
  for (let i = 0; i < segs; i++) {
    const t = (i / segs) * Math.PI * 2;
    pushVert(Math.cos(t) * radius, Math.sin(t) * radius, -half);
  }
  for (let i = 0; i < segs; i++) {
    const t = (i / segs) * Math.PI * 2;
    pushVert(Math.cos(t) * radius, Math.sin(t) * radius, half);
  }
  const botStart = 2;
  const topStart = 2 + segs;
  for (let i = 0; i < segs; i++) {
    const ni = (i + 1) % segs;
    tris.push(0, botStart + ni, botStart + i);
    tris.push(1, topStart + i, topStart + ni);
    tris.push(botStart + i, botStart + ni, topStart + ni);
    tris.push(botStart + i, topStart + ni, topStart + i);
  }
  return {
    positions: Float64Array.from(verts),
    indices: Uint32Array.from(tris),
  };
}

function createHarness(): { scene: Scene; store: SceneStore; app: AppState } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

describe("#234 InspectPanel preservation, primitive fitting, datum alignment & contours", () => {
  it("renders all preserved data-action='inspect-*' controls, SurfaceFieldControls, and #234 sections", () => {
    const { scene, store, app } = createHarness();
    const sphere = buildSphereMesh(2);
    const mesh = store.addMesh(sphere.positions, sphere.indices);
    app.adoptSceneMesh(store, mesh, "Sphere");
    store.select(mesh);
    store.commitFrame();

    const htmlInitial = renderToStaticMarkup(
      createElement(SceneProvider, {
        store,
        children: createElement(InspectPanel, { app, scene }),
      }),
    );

    expect(htmlInitial).toContain('data-action="inspect-ruler"');
    expect(htmlInitial).toContain('data-action="inspect-query"');
    expect(htmlInitial).toContain('data-action="inspect-section"');
    expect(htmlInitial).toContain('data-action="inspect-clip"');
    expect(htmlInitial).toContain('class="op surface-field-panel"');
    expect(htmlInitial).toContain('data-inspect-field-mode="none"');
    expect(htmlInitial).toContain("data-inspect-primitive-fit");
    expect(htmlInitial).toContain('data-action="inspect-fit-primitive"');
    expect(htmlInitial).toContain('data-action="inspect-align-datum"');
    expect(htmlInitial).toContain("data-inspect-contours");
    expect(htmlInitial).toContain('data-action="inspect-slice-contours"');

    // Preserved tool state transitions continue to work cleanly
    app.setInspectTool("ruler");
    expect(app.getSnapshot().inspectTool).toBe("ruler");
    const htmlRuler = renderToStaticMarkup(
      createElement(SceneProvider, {
        store,
        children: createElement(InspectPanel, { app, scene }),
      }),
    );
    expect(htmlRuler).toContain('data-action="inspect-ruler" aria-pressed="true"');

    // Sphere primitive fitting achieves tight RMS fit on watertight mesh
    const sphereFit = scene.inspect.fitPrimitive(mesh, { kind: "sphere" });
    expect(sphereFit.primitive.kind).toBe("sphere");
    expect(sphereFit.rmsError).toBeLessThan(0.05);
  });

  it("executes fitPrimitive, alignToFittedPrimitive + commitFrame, sliceContours, SVG/DXF exports, and renders reports", () => {
    const { scene, store, app } = createHarness();
    const cyl = buildTiltedCylinderMesh(1.5, 6, 24);
    const mesh = store.addMesh(cyl.positions, cyl.indices);
    app.adoptSceneMesh(store, mesh, "TiltedCyl");
    store.select(mesh);
    store.commitFrame();

    const commitSpy = vi.spyOn(store, "commitFrame");

    // 1. Primitive fit (verify store.scene is wired cleanly; capped cylinder includes cap triangles in full-mesh RMS)
    expect(store.scene).toBe(scene);
    const report = scene.inspect.fitPrimitive(mesh, { kind: "cylinder" });
    expect(report.primitive.kind).toBe("cylinder");
    expect(report.inlierCount).toBeGreaterThan(0);
    expect(report.rmsError).toBeLessThan(0.25);

    app.updateWave6((prev) => ({
      ...prev,
      metrology: {
        ...prev.metrology,
        fittedPrimitive: report,
        showPrimitiveOverlay: true,
      },
    }));

    // 2. Datum alignment + commitFrame
    const placement = scene.inspect.alignToFittedPrimitive(mesh, report.primitive, [0, 1, 0], [0, 0, 0]);
    store.commitFrame();
    expect(commitSpy).toHaveBeenCalledTimes(1);
    expect(placement.rotation).toHaveLength(9);

    const refit = scene.inspect.fitPrimitive(mesh, { kind: "cylinder" });
    expect(refit.primitive.kind).toBe("cylinder");
    if (refit.primitive.kind === "cylinder") {
      expect(Math.abs(refit.primitive.axisDir[1])).toBeGreaterThan(0.95);
    }

    // 3. Slice single & stack contours + SVG/DXF export
    const single = scene.inspect.sliceContours(mesh, [0, 0, 0], [0, 1, 0]);
    expect(single.loops.length).toBeGreaterThan(0);
    expect(single.totalPerimeter).toBeGreaterThan(0);

    const stack = scene.inspect.sliceContourStack(mesh, [0, -1, 0], [0, 1, 0], 1, 3);
    expect(stack).toHaveLength(3);
    expect(stack.some((s) => s.loops.length > 0)).toBe(true);

    app.updateWave6((prev) => ({
      ...prev,
      metrology: {
        ...prev.metrology,
        contourSlices: stack,
        showContoursOverlay: true,
      },
    }));

    const svg = sliceContoursExport(stack, "svg");
    expect(svg).toContain("<svg");
    expect(svg.length).toBeGreaterThan(20);

    const dxf = exportContourSlicesDxf(stack);
    expect(dxf).toContain("SECTION");
    expect(dxf).toContain("ENTITIES");
    expect(dxf).toContain("LWPOLYLINE");
    expect(dxf).toContain("EOF");

    // 4. InspectPanel reflects fitted primitive & contour metrics & export buttons
    const htmlPopulated = renderToStaticMarkup(
      createElement(SceneProvider, {
        store,
        children: createElement(InspectPanel, { app, scene }),
      }),
    );
    expect(htmlPopulated).toContain("data-inspect-fit-report");
    expect(htmlPopulated).toContain("cylinder");
    expect(htmlPopulated).toContain("data-inspect-contours-summary");
    expect(htmlPopulated).toContain('data-action="inspect-export-svg"');
    expect(htmlPopulated).toContain('data-action="inspect-export-dxf"');
  });
});

describe("#234 PointsPanel parsing, sampling, filtering & surface reconstruction", () => {
  it("parses XYZ, PTS (with header & intensity), ASCII PLY, and PCD point clouds", async () => {
    const xyz = parsePointCloudText("0 0 0\n1 0 0\n0 1 0\n0 0 1\n", "xyz");
    expect(xyz.count).toBe(4);

    const pts = await parsePointCloudFileContent(
      "scan.pts",
      new TextEncoder().encode("3\n0 0 0 120\n1 0 0 200\n0 1 0 250\n"),
    );
    expect(pts.count).toBe(3);

    const ply = parsePointCloudBytes(
      new TextEncoder().encode(
        [
          "ply",
          "format ascii 1.0",
          "element vertex 3",
          "property float x",
          "property float y",
          "property float z",
          "end_header",
          "0 0 0",
          "1 0 0",
          "0 1 0",
        ].join("\n"),
      ),
    );
    expect(ply.count).toBe(3);

    const pcd = await parsePointCloudFileContent(
      "scan.pcd",
      new TextEncoder().encode(
        [
          "# .PCD v0.7",
          "VERSION 0.7",
          "FIELDS x y z",
          "SIZE 4 4 4",
          "TYPE F F F",
          "COUNT 1 1 1",
          "WIDTH 2",
          "HEIGHT 1",
          "POINTS 2",
          "DATA ascii",
          "0 0 0",
          "1 2 3",
        ].join("\n"),
      ),
    );
    expect(pcd.count).toBe(2);
  });

  it("runs point cloud parse -> SOR/ROR/Voxel/Normals -> reconstructPointCloud -> adoptSceneMesh -> visible in SceneStore", async () => {
    const { scene, store, app } = createHarness();

    // Build sphere XYZ text + 1 gross outlier at (50, 50, 50)
    const baseSphere = createSyntheticSphereCloud(200, 1);
    const xyzLines: string[] = [];
    for (let i = 0; i < baseSphere.count; i++) {
      const off = 3 + i * 6;
      xyzLines.push(`${baseSphere.data[off]} ${baseSphere.data[off + 1]} ${baseSphere.data[off + 2]}`);
    }
    xyzLines.push("50 50 50");
    const parsed = await parsePointCloudFileContent("sphere.xyz", new TextEncoder().encode(xyzLines.join("\n")));
    expect(parsed.count).toBe(201);

    // SOR removes outlier
    const sorRes = pointCloudSor(parsed, 12, 2.0);
    expect(sorRes.report.removedCount).toBe(1);
    expect(sorRes.cloud.count).toBe(200);

    // ROR preserves dense sphere cluster
    const rorRes = pointCloudRor(sorRes.cloud, 0.5, 4);
    expect(rorRes.cloud.count).toBe(200);

    // Voxel downsample reduces point count cleanly
    const downsampled = pointCloudVoxelDownsample(rorRes.cloud, 0.15);
    expect(downsampled.count).toBeGreaterThan(20);
    expect(downsampled.count).toBeLessThanOrEqual(200);

    // Estimate normals
    const withNormals = pointCloudEstimateNormals(downsampled, 12);
    expect(withNormals.hasNormals).toBe(true);

    const bounds = computePointCloudBounds(withNormals);
    expect(bounds).not.toBeNull();
    expect(bounds!.min[0]).toBeLessThan(0);
    expect(bounds!.max[0]).toBeGreaterThan(0);

    app.updateWave6((prev) => ({
      ...prev,
      metrology: {
        ...prev.metrology,
        pointCloud: withNormals,
        showPointCloudOverlay: true,
        pointCloudReport: `Ready (${withNormals.count} points)`,
      },
    }));

    // Render PointsPanel HTML before & after reconstruction
    const htmlWithCloud = renderToStaticMarkup(createElement(PointsPanel, { app, store }));
    expect(htmlWithCloud).toContain("data-points-panel");
    expect(htmlWithCloud).toContain('data-action="points-sample-mesh"');
    expect(htmlWithCloud).toContain('data-action="points-sample-synthetic"');
    expect(htmlWithCloud).toContain('data-action="points-filter-sor"');
    expect(htmlWithCloud).toContain('data-action="points-filter-ror"');
    expect(htmlWithCloud).toContain('data-action="points-voxel-downsample"');
    expect(htmlWithCloud).toContain('data-action="points-estimate-normals"');
    expect(htmlWithCloud).toContain('data-action="points-reconstruct"');
    expect(htmlWithCloud).toContain('data-action="points-clear"');
    expect(htmlWithCloud).toContain("data-points-stats");
    expect(htmlWithCloud).toContain(`<strong>${withNormals.count}</strong>`);
    expect(htmlWithCloud).toContain("Normals: Yes");

    // Reconstruct via Implicit MLS / Poisson and adopt into SceneStore + AppState
    const mlsHandle = scene.inspect.reconstructPointCloud(withNormals, {
      method: "implicitMlsMarchingCubes",
      voxelSize: 0.25,
      kNeighbors: 12,
      normals: true,
    });
    const adoptedMls = app.adoptSceneMesh(store, mlsHandle, "Reconstructed Surface");
    store.select(adoptedMls.mesh);
    store.commitFrame();

    expect(adoptedMls.triangles).toBeGreaterThan(0);
    expect(store.getSnapshot().meshes.has(adoptedMls.mesh)).toBe(true);
    expect(store.getSnapshot().selection).toBe(adoptedMls.mesh);
    expect(app.getSnapshot().items.some((i) => i.name === "Reconstructed Surface")).toBe(true);

    // Sample Active Mesh directly from reconstructed mesh & reconstruct with Ball Pivoting (BPA)
    const resampled = sampleMeshToPointCloud(scene, adoptedMls.mesh);
    expect(resampled.count).toBeGreaterThan(0);
    expect(resampled.hasNormals).toBe(true);

    const bpaHandle = scene.inspect.reconstructPointCloud(createSyntheticSphereCloud(256, 1), {
      method: "ballPivoting",
      ballRadius: 0.35,
      kNeighbors: 12,
      normals: true,
    });
    const adoptedBpa = app.adoptSceneMesh(store, bpaHandle, "BPA Surface");
    store.commitFrame();
    expect(adoptedBpa.triangles).toBeGreaterThan(0);
    expect(store.getSnapshot().meshes.has(adoptedBpa.mesh)).toBe(true);
  });
});

describe("#234 MetrologyOverlaysLayer pure geometry builders & R3F mount markup", () => {
  it("builds Three.js BufferGeometry for packed point clouds with optional normals and 0-255 colors", () => {
    expect(buildPointCloudGeometry(null).getAttribute("position").count).toBe(0);

    const cloud = packPointCloud(
      Float32Array.from([0, 0, 0, 1, 2, 3]),
      Float32Array.from([0, 0, 1, 0, 1, 0]),
      Float32Array.from([255, 128, 0, 0, 255, 64]),
    );
    const geo = buildPointCloudGeometry(cloud);
    expect(geo.getAttribute("position").count).toBe(2);
    expect(geo.getAttribute("normal").count).toBe(2);
    const colorAttr = geo.getAttribute("color");
    expect(colorAttr.count).toBe(2);
    expect(colorAttr.getX(0)).toBeCloseTo(1, 5);
    expect(colorAttr.getX(1)).toBeCloseTo(0, 5);
    expect(colorAttr.getY(1)).toBeCloseTo(1, 5);
    geo.dispose();
  });

  it("builds exact LineSegments vertex counts for closed & open ContourSlice loops", () => {
    expect(buildContourSegmentsGeometry(null).getAttribute("position").count).toBe(0);

    const slices: ContourSlice[] = [
      {
        planeOrigin: [0, 0, 0],
        planeNormal: [0, 0, 1],
        uAxis: [1, 0, 0],
        vAxis: [0, 1, 0],
        totalPerimeter: 10,
        totalArea: 4,
        loops: [
          {
            closed: true,
            perimeter: 4,
            enclosedArea: 1,
            centroid2d: [0.5, 0.5],
            centroid3d: [0.5, 0.5, 0],
            minCaliperDiameter: 1,
            maxCaliperDiameter: 1.4,
            points2d: Float64Array.from([0, 0, 1, 0, 1, 1, 0, 1]),
            points3d: Float32Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
          },
          {
            closed: false,
            perimeter: 2,
            enclosedArea: 0,
            centroid2d: [1, 0],
            centroid3d: [1, 0, 0],
            minCaliperDiameter: 0,
            maxCaliperDiameter: 2,
            points2d: Float64Array.from([0, 0, 1, 0, 2, 0]),
            points3d: Float32Array.from([0, 0, 0, 1, 0, 0, 2, 0, 0]),
          },
        ],
      },
    ];

    const geo = buildContourSegmentsGeometry(slices);
    const pos = geo.getAttribute("position");
    // 4-point closed loop = 4 segments (8 verts); 3-point open loop = 2 segments (4 verts) -> 12 verts total.
    expect(pos.count).toBe(12);
    for (let i = 0; i < pos.count; i++) {
      expect(Number.isFinite(pos.getX(i))).toBe(true);
      expect(Number.isFinite(pos.getY(i))).toBe(true);
      expect(Number.isFinite(pos.getZ(i))).toBe(true);
    }
    geo.dispose();
  });

  it("builds valid datum guide segments for plane, sphere, cylinder, and cone primitives", () => {
    expect(buildPrimitiveGuideSegments(null).getAttribute("position").count).toBe(0);

    const makeReport = (primitive: FittedPrimitiveReport["primitive"]): FittedPrimitiveReport => ({
      primitive,
      rmsError: 0.01,
      maxError: 0.02,
      inlierCount: 100,
      inlierRatio: 1,
      residuals: new Float32Array(0),
    });

    const cases: FittedPrimitiveReport[] = [
      makeReport({ kind: "plane", origin: [0, 0, 0], normal: [0, 0, 1] }),
      makeReport({ kind: "sphere", center: [1, 2, 3], radius: 2 }),
      makeReport({ kind: "cylinder", axisOrigin: [0, 0, 0], axisDir: [0, 1, 0], radius: 1, height: 4 }),
      makeReport({ kind: "cone", apex: [0, 2, 0], axisDir: [0, -1, 0], halfAngleRad: 0.35, height: 3 }),
    ];

    for (const report of cases) {
      const geo = buildPrimitiveGuideSegments(report);
      const pos = geo.getAttribute("position");
      expect(pos.count).toBeGreaterThan(0);
      expect(pos.count % 2).toBe(0);
      for (let i = 0; i < pos.count; i++) {
        expect(Number.isFinite(pos.getX(i))).toBe(true);
        expect(Number.isFinite(pos.getY(i))).toBe(true);
        expect(Number.isFinite(pos.getZ(i))).toBe(true);
      }
      geo.dispose();
    }

    // Verify MetrologyOverlaysLayer mounts cleanly as React element
    const { store, app } = createHarness();
    const overlayMarkup = renderToStaticMarkup(createElement(MetrologyOverlaysLayer, { app, store }));
    expect(overlayMarkup).toContain('name="metrology-overlays"');
  });
});
