import { Scene } from "fast-mesh-wasm";
import { SceneStore } from "fast-mesh-wasm/store";
import { describe, expect, it, vi } from "vitest";
import { buildCommandPaletteItems, fuzzyMatchCommand } from "../src/shell/ui/CommandPalette";
import { AppState } from "../src/app/state/app-state";

const TET_POS = new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
const TET_IDX = new Uint32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);

function addTet(app: AppState, store: SceneStore, name: string) {
  return app.addMesh(store, {
    name,
    positions: TET_POS,
    indices: TET_IDX,
    center: [0.25, 0.25, 0.25],
    radius: 1,
    volumeId: null,
    preset: null,
    importScale: { units: "mm", factor: 1 },
  });
}

describe("CommandPalette & hierarchical scene tree state (#235)", () => {
  it("ranks fuzzy matches sensibly and rejects unmatched queries", () => {
    const exact = fuzzyMatchCommand("sculpt", "Sculpt & Deform");
    const subseq = fuzzyMatchCommand("sct", "Sculpt & Deform");
    const miss = fuzzyMatchCommand("zzzz", "Sculpt & Deform");
    expect(exact).not.toBeNull();
    expect(subseq).not.toBeNull();
    expect(exact!).toBeGreaterThan(subseq!);
    expect(miss).toBeNull();
  });

  it("indexes tabs, shading modes, camera presets, tools, and scene items cleanly", () => {
    const scene = new Scene();
    const store = new SceneStore(scene);
    const app = new AppState();
    const a = addTet(app, store, "bunny.stl");
    const b = addTet(app, store, "jaw.stl");
    const selectSpy = vi.fn();

    const cmds = buildCommandPaletteItems({
      app,
      items: app.getSnapshot().items,
      selection: a.mesh,
      canUndo: true,
      canRedo: false,
      ortho: false,
      units: "mm",
      onSelectItem: selectSpy,
      onSetMode: vi.fn(),
      onResetView: vi.fn(),
      onUndo: vi.fn(),
      onRedo: vi.fn(),
      onOpenHelp: vi.fn(),
    });

    const repairTab = cmds.find((c) => c.id === "tab:repair");
    expect(repairTab).toBeDefined();
    repairTab!.run();
    expect(app.getSnapshot().activeTab).toBe("repair");

    const backface = cmds.find((c) => c.id === "shading:backfaceWarning");
    expect(backface).toBeDefined();
    backface!.run();
    expect(app.getSnapshot().shadingMode).toBe("backfaceWarning");

    const camTop = cmds.find((c) => c.id === "camera:top");
    expect(camTop).toBeDefined();
    camTop!.run();
    expect(app.getSnapshot().cameraPresetRequest?.preset).toBe("top");

    const lockJaw = cmds.find((c) => c.id === `lock:${b.id}`);
    expect(lockJaw).toBeDefined();
    lockJaw!.run();
    expect(app.getSnapshot().items.find((i) => i.id === b.id)?.locked).toBe(true);

    const selectBunny = cmds.find((c) => c.id === `object:${a.id}`);
    selectBunny!.run();
    expect(selectSpy).toHaveBeenCalledWith(a.mesh);

    store.dispose();
  });

  it("updates lock, per-mesh shading override, and focusDefect in AppState", () => {
    const scene = new Scene();
    const store = new SceneStore(scene);
    const app = new AppState();
    const item = addTet(app, store, "part.stl");

    app.setMeshLocked(item.mesh, true);
    expect(app.getSnapshot().items[0]?.locked).toBe(true);

    app.setMeshShadingOverride(item.mesh, "matcap");
    expect(app.getSnapshot().items[0]?.shadingOverride).toBe("matcap");

    app.focusDefect(item.mesh, [1, 2, 3], 0.4);
    expect(app.getSnapshot().healerFocusRequest).toMatchObject({
      mesh: item.mesh,
      center: [1, 2, 3],
      radius: 0.4,
    });

    store.dispose();
  });
});
