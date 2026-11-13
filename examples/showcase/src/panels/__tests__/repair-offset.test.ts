// @vitest-environment jsdom
// Offset, Shell, and Voxel remesh report lines (`describeReport`) and
// `RepairPanel` controls (`[data-offset-*]`, `[data-shell-*]`).
import type { Analysis, MeshHandle, OffsetReport, RepairOp, RepairReport, Scene, ShellReport } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppState } from "../../scene/app-state";
import { describeReport } from "../repair-summary";
import { RepairPanel } from "../RepairPanel";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function offsetReport(patch: Partial<OffsetReport> = {}): OffsetReport {
  return {
    moved: 0,
    maxDisplacement: 0,
    degenerateRemoved: 0,
    verticesRemoved: 0,
    verticesAdded: 0,
    trianglesAdded: 0,
    trianglesRemoved: 0,
    voxelSize: 0,
    uvsDropped: false,
    ...patch,
  };
}

function emptyAnalysis(): Analysis {
  return {
    loops: [],
    coincidentGroups: 0,
    coincident: [],
    degenerate: 0,
    degenerateList: new Uint32Array(),
    duplicate: 0,
    duplicateList: new Uint32Array(),
    nonManifoldEdges: 0,
    nonManifoldEdgeList: [],
    nonManifoldVertices: new Uint32Array(),
    pinchVertices: new Uint32Array(),
    components: [{ triangles: new Uint32Array(12), area: 6, closed: true }],
    orientationConflicts: 0,
    selfIntersects: null,
  };
}

const cleanups: (() => Promise<void> | void)[] = [];

afterEach(async () => {
  vi.restoreAllMocks();
  for (const fn of cleanups.splice(0).reverse()) {
    await fn();
  }
});

async function mountRepairPanel(options?: {
  units?: "mm" | "in";
  onRepair?: (mesh: MeshHandle, op: RepairOp) => RepairReport;
}): Promise<{
  container: HTMLDivElement;
  app: AppState;
  store: SceneStore;
  mesh: MeshHandle;
  repairSpy: ReturnType<typeof vi.spyOn>;
}> {
  let nextId = 0;
  const fakeScene = {
    addMesh: () => ({ __mesh: ++nextId }) as unknown as MeshHandle,
    removeMesh: () => {},
    onChanges: () => () => {},
    canUndo: () => false,
    canRedo: () => false,
    analyze: vi.fn(() => emptyAnalysis()),
    repair: vi.fn((mesh: MeshHandle, op: RepairOp): RepairReport => {
      if (options?.onRepair !== undefined) {
        return options.onRepair(mesh, op);
      }
      if (op.op === "offset") {
        if (op.algorithm !== undefined && op.algorithm !== "normal") {
          return offsetReport({
            verticesRemoved: 4,
            trianglesRemoved: 4,
            verticesAdded: 62,
            trianglesAdded: 120,
            voxelSize: op.voxelSize ?? 0.25,
          });
        }
        return offsetReport({ moved: 4, maxDisplacement: Math.abs(op.distance) });
      }
      if (op.op === "shell") {
        const shell: ShellReport = {
          verticesAdded: 4,
          trianglesAdded: 4,
          loopsClosed: 0,
          degenerateRemoved: 0,
          verticesRemoved: 0,
        };
        return shell;
      }
      throw new Error(`unexpected op: ${op.op}`);
    }),
  };

  const scene = fakeScene as unknown as Scene;
  const store = createSceneStore(scene);
  const app = new AppState();
  if (options?.units !== undefined) {
    app.setUnits(options.units);
  }
  const item = app.addMesh(store, {
    name: "Tetra",
    positions: Float64Array.from([0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]),
    indices: Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]),
    center: [2.5, 2.5, 2.5],
    radius: 10,
    volumeId: null,
    preset: null,
  });
  store.select(item.mesh);

  const repairSpy = vi.spyOn(store, "repair");

  const container = document.createElement("div");
  document.body.appendChild(container);
  let root: Root;
  await act(async () => {
    root = createRoot(container);
    root.render(createElement(SceneProvider, { store }, createElement(RepairPanel, { app, scene })));
  });

  cleanups.push(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    app.removeMesh(store, item);
    store.dispose();
  });

  return { container, app, store, mesh: item.mesh, repairSpy };
}

async function setInputValue(input: HTMLInputElement, value: string): Promise<void> {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  await act(async () => {
    setter?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function setSelectValue(select: HTMLSelectElement, value: string): Promise<void> {
  const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
  await act(async () => {
    setter?.call(select, value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function clickButton(button: HTMLButtonElement): Promise<void> {
  await act(async () => {
    button.click();
  });
}

describe("describeReport for offset", () => {
  it("formats normal displacement reports with and without display units", () => {
    expect(describeReport("offset", offsetReport({ moved: 1234, maxDisplacement: 0.5 }))).toBe(
      "Offset: moved 1,234 vertices (max 0.5)",
    );
    expect(describeReport("offset", offsetReport({ moved: 1, maxDisplacement: 2 }), "mm")).toBe(
      "Offset: moved 1 vertex (max 2 mm)",
    );
    expect(describeReport("offset", offsetReport({ moved: 8, maxDisplacement: 25.4 }), "in")).toBe(
      "Offset: moved 8 vertices (max 1 in)",
    );
  });

  it("formats volumetric rebuild reports in bare numbers, mm, and inches", () => {
    const rebuilt = offsetReport({
      verticesRemoved: 8,
      trianglesRemoved: 12,
      verticesAdded: 62,
      trianglesAdded: 120,
      voxelSize: 2.54,
    });
    expect(describeReport("offset", rebuilt)).toBe("Offset: rebuilt 120 triangles, 62 vertices (voxel 2.54)");
    expect(describeReport("offset", rebuilt, "mm")).toBe(
      "Offset: rebuilt 120 triangles, 62 vertices (voxel 2.54 mm)",
    );
    expect(describeReport("offset", rebuilt, "in")).toBe("Offset: rebuilt 120 triangles, 62 vertices (voxel 0.1 in)");

    const singular = offsetReport({
      verticesRemoved: 4,
      trianglesRemoved: 4,
      verticesAdded: 1,
      trianglesAdded: 1,
      voxelSize: 0.5,
    });
    expect(describeReport("offset", singular, "mm")).toBe("Offset: rebuilt 1 triangle, 1 vertex (voxel 0.5 mm)");
  });

  it("formats volumetric full erosion when all triangles are removed", () => {
    const cleared = offsetReport({
      verticesRemoved: 8,
      trianglesRemoved: 12,
      verticesAdded: 0,
      trianglesAdded: 0,
      voxelSize: 0.25,
    });
    expect(describeReport("offset", cleared)).toBe("Offset: cleared mesh");
    expect(describeReport("offset", cleared, "in")).toBe("Offset: cleared mesh");
  });
});

describe("RepairPanel Offset & Shell controls", () => {
  it("runs normal offset by default, hides voxel size, and disables normal offset at distance 0", async () => {
    const { container, mesh, repairSpy } = await mountRepairPanel();

    const algorithm = container.querySelector<HTMLSelectElement>("[data-offset-algorithm]");
    const distance = container.querySelector<HTMLInputElement>("[data-offset-distance]");
    const runBtn = container.querySelector<HTMLButtonElement>("[data-offset-run]");

    expect(algorithm).not.toBeNull();
    expect(distance).not.toBeNull();
    expect(runBtn).not.toBeNull();
    expect(algorithm?.value).toBe("normal");
    expect(container.querySelector("[data-offset-voxel-size]")).toBeNull();
    expect(runBtn?.getAttribute("data-repair-op")).toBe("offset");
    expect(runBtn?.textContent).toBe("Offset");
    expect(runBtn?.disabled).toBe(false);

    await setInputValue(distance!, "-0.75");
    await clickButton(runBtn!);

    expect(repairSpy).toHaveBeenCalledWith(mesh, {
      op: "offset",
      distance: -0.75,
      algorithm: "normal",
    });
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain(
      "Offset: moved 4 vertices (max 0.75 mm)",
    );

    await setInputValue(distance!, "0");
    expect(runBtn?.disabled).toBe(true);
    expect(runBtn?.textContent).toBe("Offset");
  });

  it("switches to fhEdt and bvhSdf, toggles Voxel remesh at distance 0, and forwards voxelSize", async () => {
    const { container, mesh, repairSpy } = await mountRepairPanel();

    const algorithm = container.querySelector<HTMLSelectElement>("[data-offset-algorithm]")!;
    const distance = container.querySelector<HTMLInputElement>("[data-offset-distance]")!;
    const runBtn = container.querySelector<HTMLButtonElement>("[data-offset-run]")!;

    await setSelectValue(algorithm, "fhEdt");
    const voxelInput = container.querySelector<HTMLInputElement>("[data-offset-voxel-size]");
    expect(voxelInput).not.toBeNull();
    expect(voxelInput?.value).toBe("");
    expect(runBtn.textContent).toBe("Offset");

    await setInputValue(distance, "0");
    expect(runBtn.disabled).toBe(false);
    expect(runBtn.textContent).toBe("Voxel remesh");

    await clickButton(runBtn);
    expect(repairSpy).toHaveBeenLastCalledWith(mesh, {
      op: "offset",
      distance: 0,
      algorithm: "fhEdt",
    });
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain(
      "Offset: rebuilt 120 triangles, 62 vertices (voxel 0.25 mm)",
    );

    await setSelectValue(algorithm, "bvhSdf");
    await setInputValue(distance, "1.25");
    expect(runBtn.textContent).toBe("Offset");

    // Non-positive explicit voxel size disables the action button.
    await setInputValue(voxelInput!, "0");
    expect(runBtn.disabled).toBe(true);

    await setInputValue(voxelInput!, "0.4");
    expect(runBtn.disabled).toBe(false);
    await clickButton(runBtn);
    expect(repairSpy).toHaveBeenLastCalledWith(mesh, {
      op: "offset",
      distance: 1.25,
      algorithm: "bvhSdf",
      voxelSize: 0.4,
    });

    // Clearing voxel size reverts to automatic (omits voxelSize).
    await setInputValue(voxelInput!, "");
    expect(runBtn.disabled).toBe(false);
    await clickButton(runBtn);
    expect(repairSpy).toHaveBeenLastCalledWith(mesh, {
      op: "offset",
      distance: 1.25,
      algorithm: "bvhSdf",
    });
  });

  it("displays cleared mesh summary when volumetric inward offset erodes all triangles", async () => {
    const { container, repairSpy } = await mountRepairPanel({
      onRepair: () =>
        offsetReport({
          verticesRemoved: 4,
          trianglesRemoved: 4,
          verticesAdded: 0,
          trianglesAdded: 0,
          voxelSize: 0.5,
        }),
    });

    const algorithm = container.querySelector<HTMLSelectElement>("[data-offset-algorithm]")!;
    const distance = container.querySelector<HTMLInputElement>("[data-offset-distance]")!;
    const runBtn = container.querySelector<HTMLButtonElement>("[data-offset-run]")!;

    await setSelectValue(algorithm, "bvhSdf");
    await setInputValue(distance, "-10");
    await clickButton(runBtn);

    expect(repairSpy).toHaveBeenCalledOnce();
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain("Offset: cleared mesh");
  });

  it("converts Offset distance, voxelSize, and Shell thickness from inches to mm", async () => {
    const { container, mesh, repairSpy } = await mountRepairPanel({ units: "in" });

    const algorithm = container.querySelector<HTMLSelectElement>("[data-offset-algorithm]")!;
    const distance = container.querySelector<HTMLInputElement>("[data-offset-distance]")!;
    const offsetRun = container.querySelector<HTMLButtonElement>("[data-offset-run]")!;
    const shellThickness = container.querySelector<HTMLInputElement>("[data-shell-thickness]");
    const shellRun = container.querySelector<HTMLButtonElement>("[data-shell-run]");

    expect(shellThickness).not.toBeNull();
    expect(shellRun).not.toBeNull();
    expect(shellRun?.getAttribute("data-repair-op")).toBe("shell");

    await setSelectValue(algorithm, "fhEdt");
    const voxelInput = container.querySelector<HTMLInputElement>("[data-offset-voxel-size]")!;
    await setInputValue(distance, "0.1");
    await setInputValue(voxelInput, "0.05");
    await clickButton(offsetRun);

    expect(repairSpy).toHaveBeenLastCalledWith(mesh, {
      op: "offset",
      distance: 2.54,
      algorithm: "fhEdt",
      voxelSize: 1.27,
    });
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain("(voxel 0.05 in)");

    await setInputValue(shellThickness!, "0");
    expect(shellRun?.disabled).toBe(true);

    await setInputValue(shellThickness!, "0.1");
    expect(shellRun?.disabled).toBe(false);
    await clickButton(shellRun!);

    expect(repairSpy).toHaveBeenLastCalledWith(mesh, {
      op: "shell",
      thickness: 2.54,
    });
    expect(container.querySelector("[data-repair-report]")?.textContent).toContain(
      "Shell: added 4 vertices and 4 triangles",
    );
  });
});
