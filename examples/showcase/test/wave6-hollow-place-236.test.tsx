import { type MeshHandle, Scene, type Vec3 } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { SceneStore } from "fast-mesh-wasm/store";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import { ModelingPanel } from "../src/panels/ModelingPanel";
import { AppState } from "../src/scene/app-state";
import { selectMesh } from "../src/scene/selection";
import {
  computeHitPointAndWorldNormal,
  createBuiltinStampMeshGeometry,
  executeDrainHoleAtHit,
  executePlaceStampAtHit,
  findCenterSurfaceHit,
  getPlaceAlignToNormal,
  orientationFromSurfaceNormal,
  runHollowMeshOp,
  setPlaceAlignToNormal,
} from "../src/viewport/HollowPlaceToolsLayer";

function boxMesh(cx = 0, cy = 0, cz = 0, hx = 10, hy = 10, hz = 10) {
  const x0 = cx - hx;
  const x1 = cx + hx;
  const y0 = cy - hy;
  const y1 = cy + hy;
  const z0 = cz - hz;
  const z1 = cz + hz;
  const positions = Float64Array.from([
    x0, y0, z0,
    x1, y0, z0,
    x1, y1, z0,
    x0, y1, z0,
    x0, y0, z1,
    x1, y0, z1,
    x1, y1, z1,
    x0, y1, z1,
  ]);
  const indices = Uint32Array.from([
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6,
    1, 2, 6, 1, 6, 5,
    3, 0, 4, 3, 4, 7,
  ]);
  return { positions, indices };
}

function createHarness() {
  const scene = new Scene();
  const store = new SceneStore(scene);
  const app = new AppState();
  const targetData = boxMesh(0, 0, 0, 10, 10, 10);
  const targetItem = app.addMesh(store, {
    name: "Target Box",
    positions: targetData.positions,
    indices: targetData.indices,
    center: [0, 0, 0],
    radius: 17.4,
    volumeId: null,
    preset: null,
  });
  const stampData = boxMesh(0, 0, 1, 1.5, 1.5, 1.5);
  const stampItem = app.addMesh(store, {
    name: "Stamp Cube",
    positions: stampData.positions,
    indices: stampData.indices,
    center: [0, 0, 1],
    radius: 2.6,
    volumeId: null,
    preset: null,
  });
  selectMesh(app, (m) => store.select(m), null, targetItem.mesh, false);
  return { scene, store, app, targetItem, stampItem };
}

const cleanups: Array<() => void> = [];

afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

describe("Issue #236 Hollowing, Drain Holes & Place Mesh UI + Tools", () => {
  it("preserves all data-modeling-* selectors and renders Wave 6 Print Prep controls & state bindings", () => {
    const { store, app, stampItem, targetItem } = createHarness();
    cleanups.push(() => store.dispose());

    const renderHtml = () =>
      renderToStaticMarkup(
        <SceneProvider store={store}>
          <ModelingPanel app={app} />
        </SceneProvider>,
      );

    const html = renderHtml();

    const preservedAttrs = [
      "data-modeling-boolean-target",
      "data-modeling-boolean-source",
      "data-modeling-boolean-op",
      "data-modeling-keep-originals",
      "data-modeling-boolean-run",
      "data-modeling-cut-axis",
      "data-modeling-cut-offset",
      "data-modeling-cut-cap",
      "data-modeling-cut-side",
      "data-modeling-cut-run",
      "data-modeling-offset-algorithm",
      "data-modeling-offset-distance",
      "data-modeling-offset-run",
      "data-modeling-shell-thickness",
      "data-modeling-shell-run",
      "data-modeling-lock-boundaries",
      "data-modeling-simplify-ratio",
      "data-modeling-simplify-mode",
      "data-modeling-simplify-run",
      "data-modeling-subdivide-levels",
      "data-modeling-subdivide-run",
    ];
    for (const attr of preservedAttrs) {
      expect(html, `Expected attribute ${attr} in ModelingPanel markup`).toContain(attr);
    }

    const printPrepAttrs = [
      "data-modeling-hollow-thickness",
      "data-modeling-hollow-backend",
      "data-modeling-hollow-voxel-size",
      "data-modeling-hollow-max-voxels",
      "data-modeling-hollow-inner-shell",
      "data-modeling-hollow-run",
      "data-modeling-drain-radius",
      "data-modeling-drain-depth",
      "data-modeling-drain-through",
      "data-modeling-drain-plug",
      "data-modeling-drain-tool",
      "data-modeling-drain-center",
      "data-modeling-place-source",
      "data-modeling-place-stamp-mesh",
      "data-modeling-place-mode",
      "data-modeling-place-up-axis",
      "data-modeling-place-scale",
      "data-modeling-place-embed-depth",
      "data-modeling-place-spin",
      "data-modeling-place-align-normal",
      "data-modeling-place-conform",
      "data-modeling-place-tool",
    ];
    for (const attr of printPrepAttrs) {
      expect(html, `Expected Wave 6 Print Prep attribute ${attr}`).toContain(attr);
    }

    // Verify upAxis select options strictly omit "posX" and keep only negZ | posZ | posY | negY
    const upAxisMatch = html.match(/<select[^>]*data-modeling-place-up-axis[^>]*>([\s\S]*?)<\/select>/);
    expect(upAxisMatch).not.toBeNull();
    const upAxisOptions = Array.from(upAxisMatch![1].matchAll(/value="([^"]+)"/g)).map((m) => m[1]);
    expect(upAxisOptions).toEqual(["posZ", "negZ", "posY", "negY"]);
    expect(upAxisOptions).not.toContain("posX");

    // Verify stamp mesh selector includes non-target item and omits target item
    const stampSelectMatch = html.match(/<select[^>]*data-modeling-place-stamp-mesh[^>]*>([\s\S]*?)<\/select>/);
    expect(stampSelectMatch).not.toBeNull();
    expect(stampSelectMatch![1]).toContain(`value="${stampItem.id}"`);
    expect(stampSelectMatch![1]).not.toContain(`value="${targetItem.id}"`);

    // Verify activeTool & Wave 6 Print Prep state reflection in rendered HTML
    const prev = app.getSnapshot().wave6.printPrep;
    app.updateWave6({
      printPrep: {
        ...prev,
        activeTool: "drainHole",
        drainParams: { ...prev.drainParams, generatePlug: true, depth: 0 },
        placeParams: { ...prev.placeParams, stampSource: "sceneMesh", stampMeshId: stampItem.id },
      },
    });
    const armedDrainHtml = renderHtml();
    expect(armedDrainHtml).toMatch(/aria-pressed="true"[^>]*data-modeling-drain-tool|data-modeling-drain-tool[^>]*aria-pressed="true"/);
    expect(armedDrainHtml).toMatch(/disabled[^>]*data-modeling-drain-depth|data-modeling-drain-depth[^>]*disabled/);
    expect(armedDrainHtml).toMatch(/data-modeling-drain-plug[^>]*checked|checked[^>]*data-modeling-drain-plug/);

    app.updateWave6({
      printPrep: {
        ...app.getSnapshot().wave6.printPrep,
        activeTool: "placeStamp",
      },
    });
    const armedPlaceHtml = renderHtml();
    expect(armedPlaceHtml).toMatch(/aria-pressed="true"[^>]*data-modeling-place-tool|data-modeling-place-tool[^>]*aria-pressed="true"/);

    setPlaceAlignToNormal(app, false);
    expect(getPlaceAlignToNormal(app)).toBe(false);
    setPlaceAlignToNormal(app, true);
    expect(getPlaceAlignToNormal(app)).toBe(true);
  });

  it("computes orientation from surface normal, world hit normal under scale, and valid builtin stamps", () => {
    const orientX = orientationFromSurfaceNormal([1, 0, 0]);
    expect(orientX.axis).toEqual([1, 0, 0]);
    expect(orientX.matrix4[8]).toBeCloseTo(1, 5);
    expect(orientX.matrix4[9]).toBeCloseTo(0, 5);
    expect(orientX.matrix4[10]).toBeCloseTo(0, 5);

    const orientTwisted = orientationFromSurfaceNormal([0, 0, 1], Math.PI * 0.5);
    expect(orientTwisted.matrix4[0]).toBeCloseTo(0, 5);
    expect(orientTwisted.matrix4[1]).toBeCloseTo(1, 5);

    for (const kind of ["peg", "hexBoss", "dome", "cube"] as const) {
      const geom = createBuiltinStampMeshGeometry(kind);
      expect(geom.positions.length).toBeGreaterThan(0);
      expect(geom.indices.length).toBeGreaterThan(0);
      expect(geom.indices.length % 3).toBe(0);
    }

    const scene = new Scene();
    const box = boxMesh(0, 0, 0, 1, 1, 1);
    const mesh = scene.addMesh(box.positions, box.indices);
    scene.setPlacement(mesh, {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      translation: [0, 0, 0],
      scale: [2, 1, 0.5],
    });
    const hit = scene.raycast([0, 0, 5], [0, 0, -1]);
    expect(hit).not.toBeNull();
    const resolved = computeHitPointAndWorldNormal(scene, hit!);
    expect(resolved).not.toBeNull();
    expect(resolved!.normal[0]).toBeCloseTo(0, 4);
    expect(resolved!.normal[1]).toBeCloseTo(0, 4);
    expect(resolved!.normal[2]).toBeCloseTo(1, 4);
  });

  it("runHollowMeshOp hollows a closed box and updates SceneStore triangle stats", async () => {
    const { store, targetItem } = createHarness();
    cleanups.push(() => store.dispose());

    const result = await runHollowMeshOp({
      store,
      mesh: targetItem.mesh,
      options: {
        wallThickness: 2.0,
        backend: "bvhSdf",
        maxVoxels: 262144,
      },
    });

    expect(result.beforeTriangles).toBe(12);
    expect(result.report.cavitiesCreated).toBe(1);
    expect(result.report.cavityTriangles).toBeGreaterThan(0);
    expect(result.afterTriangles).toBe(result.beforeTriangles + result.report.cavityTriangles);
    expect(store.scene.views(targetItem.mesh).indices().length / 3).toBe(result.afterTriangles);
  });

  it("executeDrainHoleAtHit punches drain hole at center hit and adopts plug mesh as visible in SceneStore", async () => {
    const { store, app, targetItem } = createHarness();
    cleanups.push(() => store.dispose());

    await runHollowMeshOp({
      store,
      mesh: targetItem.mesh,
      options: {
        wallThickness: 2.0,
        backend: "bvhSdf",
      },
    });

    const centerHit = await findCenterSurfaceHit(store.scene, targetItem.mesh);
    expect(centerHit).not.toBeNull();
    expect(centerHit!.point[2]).toBeCloseTo(10, 3);
    expect(centerHit!.normal[2]).toBeCloseTo(1, 3);

    const drainRes = await executeDrainHoleAtHit({
      store,
      app,
      target: targetItem.mesh,
      targetName: targetItem.name,
      hitPoint: centerHit!.point,
      hitNormal: centerHit!.normal,
      drainParams: {
        diameter: 4.0,
        depth: 0,
        outerChamfer: 0.4,
        taperDeg: 5,
        generatePlug: true,
      },
      through: true,
    });

    expect(drainRes.report.holesPunched).toBe(1);
    expect(drainRes.plugs.length).toBe(1);
    expect(drainRes.afterTriangles).toBeGreaterThan(0);

    const plugHandle: MeshHandle = drainRes.plugs[0];
    const plugItem = app.getSnapshot().items.find((i) => i.mesh === plugHandle);
    expect(plugItem).toBeDefined();
    expect(plugItem!.name).toContain("plug");

    const plugMeshState = store.getSnapshot().meshes.get(plugHandle);
    expect(plugMeshState).toBeDefined();
    expect(plugMeshState!.desired.visible).toBe(true);
    expect(store.scene.views(plugHandle).indices().length).toBeGreaterThan(0);
  });

  it("executePlaceStampAtHit aligns stamps onto target surface across all 4 upAxis options", async () => {
    const upAxes = ["negZ", "posZ", "posY", "negY"] as const;

    for (const upAxis of upAxes) {
      const { store, app, targetItem, stampItem } = createHarness();
      const basePlaceParams = app.getSnapshot().wave6.printPrep.placeParams;

      const hitPoint: Vec3 = [0, 0, 10];
      const hitNormal: Vec3 = [0, 0, 1];

      const unionRes = await executePlaceStampAtHit({
        store,
        app,
        target: targetItem.mesh,
        targetName: targetItem.name,
        hitPoint,
        hitNormal,
        placeParams: {
          ...basePlaceParams,
          stampSource: "sceneMesh",
          stampMeshId: stampItem.id,
          mode: "embedUnion",
          upAxis,
          embedDepth: 0.5,
          uniformScale: 1.0,
          twistDeg: 15,
          conformToSurface: false,
        },
      });

      expect(unionRes.report.transform.translation[0]).toBeCloseTo(0, 1);
      expect(unionRes.report.transform.translation[1]).toBeCloseTo(0, 1);
      expect(Number.isFinite(unionRes.report.transform.translation[2])).toBe(true);
      expect(unionRes.afterTriangles).toBeGreaterThan(unionRes.beforeTriangles);

      const sepRes = await executePlaceStampAtHit({
        store,
        app,
        target: targetItem.mesh,
        targetName: targetItem.name,
        hitPoint: [10, 0, 0],
        hitNormal: [1, 0, 0],
        placeParams: {
          ...basePlaceParams,
          stampSource: "peg",
          stampMeshId: null,
          mode: "keepSeparate",
          upAxis,
          embedDepth: 0.25,
          uniformScale: 0.8,
          twistDeg: 0,
          conformToSurface: false,
        },
      });

      expect(sepRes.added).not.toBeNull();
      const addedHandle = sepRes.added!;
      const addedState = store.getSnapshot().meshes.get(addedHandle);
      expect(addedState).toBeDefined();
      expect(addedState!.desired.visible).toBe(true);
      expect(app.getSnapshot().items.some((item) => item.mesh === addedHandle)).toBe(true);

      store.dispose();
    }
  });
});
