import { Scene } from "fast-mesh-wasm";
import { createSceneStore, meshState } from "fast-mesh-wasm/store";
import { afterEach, describe, expect, it } from "vitest";
import { AppState } from "../src/scene/app-state";
import { DEFAULT_WAVE6_STATE } from "../src/scene/wave6-state";

function makeBox() {
  const positions = Float64Array.from([
    -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
    -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
  ]);
  const indices = Uint32Array.from([
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6,
    0, 4, 7, 0, 7, 3,
    1, 2, 6, 1, 6, 5,
  ]);
  return { positions, indices };
}

describe("Showcase Wave 6 AppState & adoptSceneMesh", () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const fn of cleanups.splice(0).reverse()) fn();
  });

  it("initializes Wave 6 defaults, updates/disarms tools, adopts scene-created meshes, and cleans up removed items", () => {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => store.dispose());

    expect(app.getSnapshot().wave6).toEqual(DEFAULT_WAVE6_STATE);

    const box = makeBox();
    const item = app.addMesh(store, {
      name: "box",
      positions: box.positions,
      indices: box.indices,
      center: [0, 0, 0],
      radius: Math.sqrt(3),
      volumeId: null,
      preset: null,
    });
    store.commitFrame();

    const extracted = scene.repair.extractSelectedFaces(item.mesh, [0, 1]);
    store.commitFrame();
    const adopted = app.adoptSceneMesh(store, extracted, "box-extracted");
    expect(meshState(store.getSnapshot(), adopted.mesh).desired.visible).toBe(true);
    expect(adopted.triangles).toBe(2);
    expect(app.adoptSceneMesh(store, extracted, "dup")).toBe(adopted);
    expect(app.getSnapshot().items.map((i) => i.id)).toEqual([item.id, adopted.id]);

    app.updateWave6((prev) => ({
      ...prev,
      select: {
        ...prev.select,
        activeTool: "lasso",
        selectedByMesh: { ...prev.select.selectedByMesh, [adopted.id]: new Uint32Array([0, 1]) },
      },
      deform: { ...prev.deform, activeTool: "ffdCage" },
      printPrep: {
        ...prev.printPrep,
        activeTool: "placeStamp",
        placeParams: { ...prev.printPrep.placeParams, stampMeshId: adopted.id },
      },
    }));

    app.disarmWave6Tools();
    const afterDisarm = app.getSnapshot().wave6;
    expect(afterDisarm.select.activeTool).toBeNull();
    expect(afterDisarm.deform.activeTool).toBeNull();
    expect(afterDisarm.printPrep.activeTool).toBeNull();
    expect(afterDisarm.select.selectedByMesh[adopted.id]).toEqual(new Uint32Array([0, 1]));

    app.removeMesh(store, adopted);
    const afterRemove = app.getSnapshot().wave6;
    expect(adopted.id in afterRemove.select.selectedByMesh).toBe(false);
    expect(afterRemove.printPrep.placeParams.stampMeshId).toBeNull();
  });
});
