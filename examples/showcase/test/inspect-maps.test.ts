// The inspect maps' ux logic (#251) against the real engine: each map's
// readouts are the SDK query's own numbers (on a local scene and through a
// promise-returning one, as a WorkerScene answers), the requests, styles,
// legends and readout lines, the band-aware overlay colour, and which scene
// changes refresh or clear a shown map.
import { init, type MeshChange, type MeshHandle, Scene, type SceneLike } from "fast-mesh-wasm";
import * as THREE from "three";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { evaluateOverlayColor, evaluateSurfaceColormap } from "../src/engine/workbench-ops";
import {
  buildInspectMapRequest,
  DEFAULT_INSPECT_MAP_SETTINGS,
  type InspectMapReport,
  type InspectMapRequest,
  type InspectMapSettings,
  inspectMapChangeAction,
  inspectMapLegend,
  inspectMapOverlay,
  inspectMapReadoutLines,
  NO_SUPPORT_COLOR,
  PAST_THRESHOLD_COLOR,
  resolveBuildDirection,
  resolveReference,
  runInspectMap,
  styleInspectMap,
  TOLERANCE_COLOR,
  WORLD_UP,
} from "../src/features/inspect/ux/inspect-maps";

beforeAll(async () => {
  await init();
});

const scenes: Scene[] = [];
afterEach(() => {
  for (const s of scenes.splice(0)) {
    s.free();
  }
});

/** An axis-aligned cube of half-width `h` around `c`, outward-wound: 8 vertices, 12 right-isosceles triangles. */
function cube(h: number, c: [number, number, number] = [0, 0, 0]): { positions: Float64Array; indices: Uint32Array } {
  const p: number[] = [];
  for (const z of [-h, h]) {
    for (const y of [-h, h]) {
      for (const x of [-h, h]) {
        p.push(c[0] + x, c[1] + y, c[2] + z);
      }
    }
  }
  // Vertex i = x + 2y + 4z (bits).
  const quads = [
    [0, 2, 3, 1], // -z
    [4, 5, 7, 6], // +z
    [0, 1, 5, 4], // -y
    [2, 6, 7, 3], // +y
    [0, 4, 6, 2], // -x
    [1, 3, 7, 5], // +x
  ];
  const t: number[] = [];
  for (const [a, b, cc, d] of quads) {
    t.push(a, b, cc, a, cc, d);
  }
  return { positions: Float64Array.from(p), indices: Uint32Array.from(t) };
}

function setup() {
  const scene = new Scene();
  scenes.push(scene);
  const inner = cube(1);
  const outer = cube(1.25);
  const a = scene.addMesh(inner.positions, inner.indices, { origin: [0, 0, 0] });
  const b = scene.addMesh(outer.positions, outer.indices, { origin: [0, 0, 0] });
  scene.commitFrame();
  return { scene, a, b };
}

/** The scene with every inspect query answering through a promise, as a WorkerScene does. */
function asyncScene(scene: Scene): SceneLike {
  const inspect = scene.inspect;
  const later = <T>(v: T) => new Promise<T>((resolve) => setTimeout(() => resolve(v), 0));
  return {
    inspect: {
      ...inspect,
      meshDeviation: (s: MeshHandle, t: MeshHandle, o?: { signed?: boolean }) => later(inspect.meshDeviation(s, t, o)),
      overhangs: (m: MeshHandle, o?: Parameters<typeof inspect.overhangs>[1]) => later(inspect.overhangs(m, o)),
      triangleQuality: (m: MeshHandle, o?: Parameters<typeof inspect.triangleQuality>[1]) =>
        later(inspect.triangleQuality(m, o)),
    },
  } as unknown as SceneLike;
}

/** The report as its kind, failing the test on another kind. */
function as<K extends InspectMapReport["kind"]>(report: InspectMapReport, kind: K): Extract<InspectMapReport, { kind: K }> {
  expect(report.kind).toBe(kind);
  return report as Extract<InspectMapReport, { kind: K }>;
}

const settings = (patch: Partial<InspectMapSettings> = {}): InspectMapSettings => ({
  ...DEFAULT_INSPECT_MAP_SETTINGS,
  ...patch,
});

describe("runInspectMap readouts are the SDK's", () => {
  it("deviation_unsigned_and_signed", async () => {
    const { scene, a, b } = setup();
    for (const signed of [false, true]) {
      const request: InspectMapRequest = { kind: "deviation", mesh: a, reference: b, signed };
      const report = as(await runInspectMap(scene, request), "deviation");
      const sdk = scene.inspect.meshDeviation(a, b, { signed });
      expect(report.readout).toEqual({
        min: sdk.min,
        max: sdk.max,
        mean: sdk.mean,
        rms: sdk.rms,
        p95: sdk.p95,
        samples: sdk.sampleCount,
      });
      expect(Array.from(report.values)).toEqual(Array.from(sdk.vertexDeviations));
      // Each inner corner is 0.25 inside the outer cube's nearest faces.
      expect(report.readout.samples).toBe(8);
      expect(report.readout.min).toBeCloseTo(signed ? -0.25 : 0.25, 5);
      expect(report.readout.max).toBeCloseTo(signed ? -0.25 : 0.25, 5);
      expect(report.readout.rms).toBeCloseTo(0.25, 5);
    }
  });

  it("overhang_area_and_faces", async () => {
    const { scene, a } = setup();
    const request: InspectMapRequest = { kind: "overhang", mesh: a, direction: [0, 0, 1], criticalAngleDeg: 45 };
    const report = as(await runInspectMap(scene, request), "overhang");
    const sdk = scene.inspect.overhangs(a, { buildDirection: [0, 0, 1], criticalAngleDeg: 45 });
    expect(report.readout).toEqual({
      area: sdk.totalSupportArea,
      faces: sdk.overhangFaceCount,
      maxAngleDeg: sdk.maxOverhangAngleDeg,
    });
    expect(Array.from(report.values)).toEqual(Array.from(sdk.vertexAnglesDeg));
    // The cube's floor: two triangles, 2 x 2.
    expect(report.readout.faces).toBe(2);
    expect(report.readout.area).toBeCloseTo(4, 6);
    expect(report.readout.maxAngleDeg).toBeCloseTo(90, 4);
  });

  it("quality_aspect_ratio_and_min_angle", async () => {
    const { scene, a } = setup();
    for (const [metric, threshold] of [
      ["aspectRatio", 1.2],
      ["minAngleDeg", 50],
      ["minAngleDeg", 40],
    ] as const) {
      const report = as(await runInspectMap(scene, { kind: "quality", mesh: a, metric, threshold }), "quality");
      const sdk = scene.inspect.triangleQuality(a, { metric, sliverThreshold: threshold });
      expect(report.readout).toEqual({
        past: sdk.sliverTriangleCount,
        total: sdk.liveTriangleCount,
        min: sdk.min,
        max: sdk.max,
        mean: sdk.mean,
      });
      expect(Array.from(report.values)).toEqual(Array.from(sdk.vertexQuality));
      expect(report.readout.total).toBe(12);
    }
    // Right isosceles triangles: 45° smallest angle, all past 50°, none past 40°.
    const past = async (threshold: number) =>
      as(await runInspectMap(scene, { kind: "quality", mesh: a, metric: "minAngleDeg", threshold }), "quality").readout.past;
    expect(await past(50)).toBe(12);
    expect(await past(40)).toBe(0);
  });

  it("a_promise_answering_scene_gives_the_same_reports", async () => {
    const { scene, a, b } = setup();
    const worker = asyncScene(scene);
    const requests: InspectMapRequest[] = [
      { kind: "deviation", mesh: a, reference: b, signed: true },
      { kind: "overhang", mesh: a, direction: [0, 1, 0], criticalAngleDeg: 30 },
      { kind: "quality", mesh: a, metric: "aspectRatio", threshold: 10 },
    ];
    for (const request of requests) {
      const local = await runInspectMap(scene, request);
      const remote = await runInspectMap(worker, request);
      expect(remote.readout).toEqual(local.readout);
      expect(Array.from(remote.values)).toEqual(Array.from(local.values));
    }
  });

  it("running_a_query_records_no_change_that_would_refresh_the_map", async () => {
    const { scene, a, b } = setup();
    const requests: InspectMapRequest[] = [
      { kind: "deviation", mesh: a, reference: b, signed: true },
      { kind: "overhang", mesh: a, direction: [0, 0, 1], criticalAngleDeg: 45 },
      { kind: "quality", mesh: a, metric: "minAngleDeg", threshold: 10 },
    ];
    for (const request of requests) {
      await runInspectMap(scene, request);
      expect(inspectMapChangeAction(scene.commitFrame(), request)).toBeNull();
    }
  });

  it("a_scene_without_inspect_queries_fails_plainly", async () => {
    await expect(
      runInspectMap({} as SceneLike, { kind: "quality", mesh: {} as MeshHandle, metric: "aspectRatio", threshold: 10 }),
    ).rejects.toThrow(/inspect/);
  });
});

describe("requests", () => {
  const m1 = {} as MeshHandle;
  const m2 = {} as MeshHandle;
  const m3 = {} as MeshHandle;

  it("deviation_needs_another_mesh", () => {
    expect(buildInspectMapRequest(settings({ kind: "deviation" }), m1, null, WORLD_UP)).toBeNull();
    expect(buildInspectMapRequest(settings({ kind: "deviation" }), m1, m1, WORLD_UP)).toBeNull();
    expect(buildInspectMapRequest(settings({ kind: "deviation", signed: true }), m1, m2, WORLD_UP)).toEqual({
      kind: "deviation",
      mesh: m1,
      reference: m2,
      signed: true,
    });
  });

  it("overhang_clamps_the_angle_and_copies_the_direction", () => {
    const dir: [number, number, number] = [0, 1, 0];
    const r = buildInspectMapRequest(settings({ kind: "overhang", criticalAngleDeg: 120 }), m1, null, dir);
    expect(r).toEqual({ kind: "overhang", mesh: m1, direction: [0, 1, 0], criticalAngleDeg: 90 });
    dir[1] = 5;
    expect(r !== null && r.kind === "overhang" ? r.direction : null).toEqual([0, 1, 0]);
  });

  it("quality_takes_metric_and_threshold", () => {
    expect(buildInspectMapRequest(settings({ kind: "quality", metric: "minAngleDeg", threshold: 20 }), m1, null, WORLD_UP)).toEqual({
      kind: "quality",
      mesh: m1,
      metric: "minAngleDeg",
      threshold: 20,
    });
  });

  it("the_reference_falls_back_to_the_first_other_mesh", () => {
    expect(resolveReference(m3, m1, [m1, m2, m3])).toBe(m3);
    expect(resolveReference(m1, m1, [m1, m2, m3])).toBe(m2);
    expect(resolveReference({} as MeshHandle, m1, [m1, m2])).toBe(m2);
    expect(resolveReference(null, m1, [m1])).toBeNull();
    expect(resolveReference(null, null, [m1, m2])).toBe(m1);
  });

  it("the_build_direction_is_world_z_or_the_camera_up", () => {
    expect(resolveBuildDirection("worldZ", null)).toEqual([0, 0, 1]);
    expect(resolveBuildDirection("viewUp", null)).toEqual([0, 0, 1]);
    const camera = new THREE.PerspectiveCamera();
    camera.position.set(5, 0, 0);
    camera.up.set(0, 1, 0);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const up = resolveBuildDirection("viewUp", camera.matrixWorld.elements);
    expect(up[0]).toBeCloseTo(0, 9);
    expect(up[1]).toBeCloseTo(1, 9);
    expect(up[2]).toBeCloseTo(0, 9);
    // Rolled 90° about the view axis: up turns with it.
    camera.rotateZ(Math.PI / 2);
    camera.updateMatrixWorld();
    const rolled = resolveBuildDirection("viewUp", camera.matrixWorld.elements);
    expect(Math.hypot(...rolled)).toBeCloseTo(1, 9);
    expect(Math.abs(rolled[2])).toBeCloseTo(1, 9);
    expect(resolveBuildDirection("viewUp", new Float32Array(16))).toEqual([0, 0, 1]);
  });
});

function fakeReport(patch: Partial<InspectMapReport> & Pick<InspectMapReport, "kind" | "request" | "readout">): InspectMapReport {
  return { values: new Float32Array(0), ...patch } as InspectMapReport;
}

describe("styles, legends and readouts", () => {
  const mesh = {} as MeshHandle;
  const reference = {} as MeshHandle;
  const dev = (signed: boolean, min: number, max: number) =>
    fakeReport({
      kind: "deviation",
      request: { kind: "deviation", mesh, reference, signed },
      readout: { min, max, mean: (min + max) / 2, rms: 0.3, p95: 0.4, samples: 10 },
    });

  it("deviation_ranges_and_tolerance_band", () => {
    expect(styleInspectMap(dev(false, 0.1, 0.5), settings())).toEqual({ map: "coolwarm", min: 0, max: 0.5, band: null });
    expect(styleInspectMap(dev(true, -0.2, 0.5), settings())).toMatchObject({ min: -0.5, max: 0.5 });
    expect(styleInspectMap(dev(true, 0, 0), settings())).toMatchObject({ min: -1, max: 1 });
    expect(styleInspectMap(dev(false, 0, 0), settings())).toMatchObject({ min: 0, max: 1 });
    const banded = styleInspectMap(dev(true, -0.2, 0.5), settings({ toleranceOn: true, tolerance: 0.05 }));
    expect(banded.band).toEqual({ lo: -0.05, hi: 0.05, color: TOLERANCE_COLOR });
    expect(styleInspectMap(dev(true, -0.2, 0.5), settings({ toleranceOn: false, tolerance: 0.05 })).band).toBeNull();
    expect(styleInspectMap(dev(true, -0.2, 0.5), settings({ toleranceOn: true, tolerance: 0 })).band).toBeNull();
  });

  it("overhang_and_quality_bands", () => {
    const over = fakeReport({
      kind: "overhang",
      request: { kind: "overhang", mesh, direction: [0, 0, 1], criticalAngleDeg: 45 },
      readout: { area: 4, faces: 2, maxAngleDeg: 90 },
    });
    expect(styleInspectMap(over, settings())).toEqual({
      map: "turbo",
      min: 45,
      max: 90,
      band: { lo: -Infinity, hi: 45, color: NO_SUPPORT_COLOR },
    });
    const aspect = fakeReport({
      kind: "quality",
      request: { kind: "quality", mesh, metric: "aspectRatio", threshold: 5 },
      readout: { past: 3, total: 12, min: 1, max: 9, mean: 2 },
    });
    expect(styleInspectMap(aspect, settings())).toEqual({
      map: "viridis",
      min: 1,
      max: 5,
      band: { lo: 5, hi: Infinity, color: PAST_THRESHOLD_COLOR },
    });
    const angle = fakeReport({
      kind: "quality",
      request: { kind: "quality", mesh, metric: "minAngleDeg", threshold: 20 },
      readout: { past: 1, total: 12, min: 10, max: 60, mean: 40 },
    });
    expect(styleInspectMap(angle, settings())).toEqual({
      map: "viridis",
      min: 20,
      max: 60,
      band: { lo: -Infinity, hi: 20, color: PAST_THRESHOLD_COLOR },
    });
  });

  it("the_overlay_carries_values_range_and_band", () => {
    const report = { ...dev(false, 0, 1), values: Float32Array.from([0, 0.5, 1]) } as InspectMapReport;
    const style = styleInspectMap(report, settings({ toleranceOn: true, tolerance: 0.2 }));
    const overlay = inspectMapOverlay(report, style);
    expect(overlay.mesh).toBe(mesh);
    expect(overlay.values).toBe(report.values);
    expect([overlay.min, overlay.max]).toEqual([0, 1]);
    expect(evaluateOverlayColor(overlay, 0.1)).toEqual([...TOLERANCE_COLOR]);
    expect(evaluateOverlayColor(overlay, 0.2)).toEqual([...TOLERANCE_COLOR]);
    expect(evaluateOverlayColor(overlay, 0.7)).toEqual(evaluateSurfaceColormap("coolwarm", 0.7, 0, 1));
    expect(evaluateOverlayColor(overlay, Number.NaN)).toEqual([0.5, 0.5, 0.5]);
    // No band: the colormap alone, as the surface field draws it.
    expect(evaluateOverlayColor({ map: "turbo", min: 0, max: 1 }, 0.3)).toEqual(evaluateSurfaceColormap("turbo", 0.3, 0, 1));
    // Infinite values (a degenerate triangle's aspect ratio) fall in an open-ended band.
    expect(evaluateOverlayColor({ map: "viridis", min: 1, max: 5, band: { lo: 5, hi: Infinity, color: PAST_THRESHOLD_COLOR } }, Infinity)).toEqual([
      ...PAST_THRESHOLD_COLOR,
    ]);
  });

  it("deviation_text_follows_the_unit", () => {
    const report = dev(true, -25.4, 50.8);
    const style = styleInspectMap(report, settings({ toleranceOn: true, tolerance: 2.54 }));
    expect(inspectMapLegend(report, style, "mm")).toEqual({
      map: "coolwarm",
      lo: "-50.8 mm",
      hi: "50.8 mm",
      band: { color: "rgb(56, 184, 82)", label: "within ±2.54 mm" },
    });
    expect(inspectMapLegend(report, style, "in")).toMatchObject({ lo: "-2 in", hi: "2 in", band: { label: "within ±0.1 in" } });
    const lines = inspectMapReadoutLines({ ...report, readout: { min: -25.4, max: 50.8, mean: 12.7, rms: 25.4, p95: 25.4, samples: 8 } } as InspectMapReport, "in", "Ref");
    expect(lines).toEqual(["max 2 in · mean 0.5 in · RMS 1 in", "min -1 in · p95 1 in · 8 vertices", "signed, to Ref"]);
  });

  it("overhang_area_follows_the_unit", () => {
    const report = fakeReport({
      kind: "overhang",
      request: { kind: "overhang", mesh, direction: [0, 1, 0], criticalAngleDeg: 45 },
      readout: { area: 645.16, faces: 2, maxAngleDeg: 90 },
    });
    expect(inspectMapReadoutLines(report, "mm")[0]).toBe("overhang area 645 mm² · 2 faces");
    expect(inspectMapReadoutLines(report, "in")).toEqual([
      "overhang area 1 in² · 2 faces",
      "steepest 90° · critical 45° · build (0, 1, 0)",
    ]);
    expect(inspectMapLegend(report, styleInspectMap(report, settings()), "in")).toMatchObject({
      lo: "45°",
      hi: "90°",
      band: { label: "up to 45°: no support" },
    });
  });

  it("quality_text_names_the_side_of_the_threshold", () => {
    const aspect = fakeReport({
      kind: "quality",
      request: { kind: "quality", mesh, metric: "aspectRatio", threshold: 5 },
      readout: { past: 3, total: 12, min: 1, max: 9.5, mean: 2.25 },
    });
    expect(inspectMapReadoutLines(aspect, "mm")).toEqual(["3 of 12 triangles above 5", "min 1 · mean 2.25 · max 9.5"]);
    expect(inspectMapLegend(aspect, styleInspectMap(aspect, settings()), "mm").band?.label).toBe("above 5");
    const angle = fakeReport({
      kind: "quality",
      request: { kind: "quality", mesh, metric: "minAngleDeg", threshold: 20 },
      readout: { past: 1, total: 12, min: 10, max: 60, mean: 40 },
    });
    expect(inspectMapReadoutLines(angle, "in")).toEqual(["1 of 12 triangles below 20°", "min 10° · mean 40° · max 60°"]);
  });
});

describe("inspectMapChangeAction", () => {
  const mesh = { id: "a" } as unknown as MeshHandle;
  const reference = { id: "b" } as unknown as MeshHandle;
  const other = { id: "c" } as unknown as MeshHandle;
  const change = (m: MeshHandle, patch: Partial<MeshChange> = {}): MeshChange => ({
    mesh: m,
    added: false,
    removed: false,
    placementChanged: false,
    capacityChanged: false,
    positions: [],
    normals: [],
    colors: [],
    indices: [],
    fieldsChanged: [],
    sectionsChanged: [],
    ...patch,
  });
  const quality: InspectMapRequest = { kind: "quality", mesh, metric: "aspectRatio", threshold: 10 };
  const deviation: InspectMapRequest = { kind: "deviation", mesh, reference, signed: false };

  it("geometry_and_placement_refresh", () => {
    expect(inspectMapChangeAction([change(mesh, { positions: [[0, 3]] })], quality)).toBe("refresh");
    expect(inspectMapChangeAction([change(mesh, { indices: "all" })], quality)).toBe("refresh");
    expect(inspectMapChangeAction([change(mesh, { placementChanged: true })], quality)).toBe("refresh");
    expect(inspectMapChangeAction([change(mesh, { capacityChanged: true })], quality)).toBe("refresh");
  });

  it("colours_and_other_meshes_do_not", () => {
    expect(inspectMapChangeAction([], quality)).toBeNull();
    expect(inspectMapChangeAction([change(mesh, { colors: "all", normals: "all" })], quality)).toBeNull();
    expect(inspectMapChangeAction([change(other, { positions: "all" })], quality)).toBeNull();
    expect(inspectMapChangeAction([change(reference, { positions: "all" })], quality)).toBeNull();
  });

  it("the_deviation_reference_counts_too", () => {
    expect(inspectMapChangeAction([change(reference, { placementChanged: true })], deviation)).toBe("refresh");
    expect(inspectMapChangeAction([change(reference, { removed: true })], deviation)).toBe("clear");
  });

  it("removal_clears_first", () => {
    expect(inspectMapChangeAction([change(mesh, { positions: "all" }), change(mesh, { removed: true })], quality)).toBe("clear");
  });
});
