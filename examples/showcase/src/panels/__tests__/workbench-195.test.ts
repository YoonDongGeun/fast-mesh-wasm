// @vitest-environment jsdom
// React + JSDOM component and workflow tests for Issue #195:
// ModelingPanel, AlignPanel, InspectPanel surface fields, TransformPanel,
// MeshList Duplicate & Mirror, WorkbenchDomains, Help, and Camera state.
import type {
  Analysis,
  CutReport,
  EditOp,
  EditResult,
  FieldHandle,
  MeshHandle,
  OffsetReport,
  Placement,
  PlacementInput,
  RegisterOptions,
  RegisterResult,
  RepairOp,
  RepairReport,
  Scene,
  SceneLike,
  SectionPlane,
  ShellReport,
  SimplifyReport,
  SubdivideReport,
} from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppState, type CameraPreset, type Item } from "../../scene/app-state";
import { HelpOverlay } from "../Help";
import { AlignPanel } from "../AlignPanel";
import { InspectPanel } from "../InspectPanel";
import { MeshList } from "../MeshList";
import { ModelingPanel } from "../ModelingPanel";
import { TransformPanel } from "../TransformPanel";
import { domainMatchesPanel, WorkbenchDomains } from "../WorkbenchDomains";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

interface FakeMeshRecord {
  positions: Float64Array;
  indices: Uint32Array;
  uvs: Float32Array | null;
  origin: [number, number, number];
  placement: Placement;
  coloring: Float32Array | null;
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

function makeBox(hx: number, hy: number, hz: number): { positions: Float64Array; indices: Uint32Array } {
  const positions = Float64Array.from([
    -hx, -hy, -hz,
    hx, -hy, -hz,
    hx, hy, -hz,
    -hx, hy, -hz,
    -hx, -hy, hz,
    hx, -hy, hz,
    hx, hy, hz,
    -hx, hy, hz,
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

function createFakeSyncScene() {
  let nextMeshId = 1;
  let nextFieldId = 1;
  const meshes = new Map<number, FakeMeshRecord>();
  const fields = new Map<number, Float32Array>();
  const listeners = new Set<
    (
      changes: {
        mesh: MeshHandle;
        positions: "all";
        normals: "all";
        colors: number[];
        indices: "all";
        fieldsChanged: FieldHandle[];
        sectionsChanged: never[];
      }[],
    ) => void
  >();
  const meshMap = new Map<MeshHandle, number>();

  const makeHandle = (id: number): MeshHandle => ({ brand: "mesh", id } as unknown as MeshHandle);
  const idOf = (h: MeshHandle): number => meshMap.get(h) ?? (h as unknown as { id: number }).id;

  const emitChange = (mesh: MeshHandle) => {
    for (const cb of listeners) {
      cb([
        {
          mesh,
          positions: "all",
          normals: "all",
          colors: [],
          indices: "all",
          fieldsChanged: [],
          sectionsChanged: [],
        },
      ]);
    }
  };

  const booleanCutSpy = vi.fn((mesh: MeshHandle, plane: SectionPlane, options?: { cap?: boolean }): CutReport => {
    const id = idOf(mesh);
    const rec = meshes.get(id)!;
    const removed = makeHandle(nextMeshId++);
    meshes.set(idOf(removed), {
      positions: new Float64Array(rec.positions),
      indices: new Uint32Array(rec.indices),
      uvs: null,
      origin: [0, 0, 0],
      placement: { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] },
      coloring: null,
    });
    meshMap.set(removed, idOf(removed));
    void plane;
    void options;
    emitChange(mesh);
    return {
      kept: mesh,
      removed,
      keptTriangles: 6,
      removedTriangles: 6,
      capTriangles: options?.cap === false ? 0 : 2,
    };
  });

  const booleanCall = vi.fn((target: MeshHandle, source: MeshHandle, op: string): MeshHandle => {
    void source;
    void op;
    emitChange(target);
    return target;
  });

  const registerSpy = vi.fn(
    (moving: MeshHandle, fixed: MeshHandle, opts?: RegisterOptions): RegisterResult => {
      void fixed;
      void opts;
      const resultPlacement: Placement = {
        rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        translation: [0.01, -0.02, 0.03],
      };
      void moving;
      return {
        status: "converged",
        converged: true,
        rmse: 0.0123,
        iterations: 7,
        correspondences: 64,
        placement: resultPlacement,
        applied: false,
        notApplied: "apply-false",
      } as unknown as RegisterResult;
    },
  );

  const undoStack: { mesh: MeshHandle; placement: Placement }[] = [];
  const redoStack: { mesh: MeshHandle; placement: Placement }[] = [];

  const scene = {
    addMesh(positions: Float64Array, indices: Uint32Array, opts?: { origin?: [number, number, number]; uvs?: Float32Array }) {
      const id = nextMeshId++;
      const handle = makeHandle(id);
      meshMap.set(handle, id);
      const o = opts?.origin ?? [0, 0, 0];
      meshes.set(id, {
        positions: new Float64Array(positions),
        indices: new Uint32Array(indices),
        uvs: opts?.uvs ? new Float32Array(opts.uvs) : null,
        origin: [...o] as [number, number, number],
        placement: { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] },
        coloring: null,
      });
      return handle;
    },
    removeMesh: vi.fn((mesh: MeshHandle) => {
      meshes.delete(idOf(mesh));
    }),
    origin(mesh: MeshHandle) {
      return [...(meshes.get(idOf(mesh))?.origin ?? [0, 0, 0])] as [number, number, number];
    },
    placement(mesh: MeshHandle): Placement {
      const p = meshes.get(idOf(mesh))!.placement;
      return { rotation: [...p.rotation], translation: [...p.translation] };
    },
    setPlacement(mesh: MeshHandle, p: PlacementInput) {
      const rec = meshes.get(idOf(mesh))!;
      undoStack.push({
        mesh,
        placement: { rotation: [...rec.placement.rotation], translation: [...rec.placement.translation] },
      });
      redoStack.length = 0;
      rec.placement = {
        rotation: [...p.rotation],
        translation: [p.translation[0], p.translation[1], p.translation[2]],
      };
      emitChange(mesh);
    },
    matrix4(mesh: MeshHandle): Float64Array {
      const p = meshes.get(idOf(mesh))!.placement;
      const r = p.rotation;
      const t = p.translation;
      return Float64Array.from([
        r[0], r[3], r[6], 0,
        r[1], r[4], r[7], 0,
        r[2], r[5], r[8], 0,
        t[0], t[1], t[2], 1,
      ]);
    },
    views(mesh: MeshHandle) {
      const rec = meshes.get(idOf(mesh))!;
      return {
        positions: () => rec.positions,
        indices: () => rec.indices,
        colors: () => (rec.coloring ? new Uint8Array(Math.floor(rec.positions.length / 3) * 4).fill(200) : null),
        coloringValues: () => rec.coloring,
        uvs: () => rec.uvs,
        normals: () => null,
      };
    },
    bounds(mesh: MeshHandle) {
      const p = meshes.get(idOf(mesh))!.positions;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let i = 0; i < p.length; i += 3) {
        if (p[i] < minX) minX = p[i];
        if (p[i] > maxX) maxX = p[i];
        if (p[i + 1] < minY) minY = p[i + 1];
        if (p[i + 1] > maxY) maxY = p[i + 1];
        if (p[i + 2] < minZ) minZ = p[i + 2];
        if (p[i + 2] > maxZ) maxZ = p[i + 2];
      }
      return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
    },
    edit(mesh: MeshHandle, ops: EditOp[]): EditResult {
      const rec = meshes.get(idOf(mesh))!;
      for (const op of ops) {
        if (op.op === "setPositions") {
          const r = rec.placement.rotation;
          const tx = rec.placement.translation[0];
          const ty = rec.placement.translation[1];
          const tz = rec.placement.translation[2];
          for (let i = 0; i < op.vertices.length; i++) {
            const v = op.vertices[i];
            const dx = op.positions[3 * i] - tx;
            const dy = op.positions[3 * i + 1] - ty;
            const dz = op.positions[3 * i + 2] - tz;
            rec.positions[3 * v] = r[0] * dx + r[3] * dy + r[6] * dz;
            rec.positions[3 * v + 1] = r[1] * dx + r[4] * dy + r[7] * dz;
            rec.positions[3 * v + 2] = r[2] * dx + r[5] * dy + r[8] * dz;
          }
        } else if (op.op === "removeTriangles") {
          for (const t of op.triangles) {
            rec.indices[3 * t] = 0;
            rec.indices[3 * t + 1] = 0;
            rec.indices[3 * t + 2] = 0;
          }
        } else if (op.op === "addTriangles") {
          const next = new Uint32Array(op.triangles.length);
          for (let i = 0; i < op.triangles.length; i++) {
            next[i] = Number(op.triangles[i]);
          }
          rec.indices = next;
        }
      }
      emitChange(mesh);
      return { added: [] };
    },
    repair(mesh: MeshHandle, op: RepairOp): RepairReport {
      const rec = meshes.get(idOf(mesh))!;
      emitChange(mesh);
      if (op.op === "offset") {
        return {
          moved: Math.floor(rec.positions.length / 3),
          maxDisplacement: Math.abs(op.distance),
          degenerateRemoved: 0,
          verticesRemoved: 0,
          verticesAdded: 0,
          trianglesAdded: 0,
          trianglesRemoved: 0,
          voxelSize: 0,
          uvsDropped: false,
        } as OffsetReport;
      }
      if (op.op === "shell") {
        return { verticesAdded: 8, trianglesAdded: 12, loopsClosed: 0, degenerateRemoved: 0 } as ShellReport;
      }
      if (op.op === "simplify") {
        return { collapsed: 10, trianglesRemoved: 20, verticesRemoved: 10 } as SimplifyReport;
      }
      if (op.op === "subdivide") {
        return { verticesAdded: 18, trianglesAdded: 36 } as SubdivideReport;
      }
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
      } as OffsetReport;
    },
    canUndo: () => undoStack.length > 0,
    canRedo: () => redoStack.length > 0,
    undo() {
      const entry = undoStack.pop();
      if (!entry) return false;
      const rec = meshes.get(idOf(entry.mesh));
      if (!rec) return false;
      redoStack.push({
        mesh: entry.mesh,
        placement: { rotation: [...rec.placement.rotation], translation: [...rec.placement.translation] },
      });
      rec.placement = entry.placement;
      emitChange(entry.mesh);
      return true;
    },
    redo() {
      const entry = redoStack.pop();
      if (!entry) return false;
      const rec = meshes.get(idOf(entry.mesh));
      if (!rec) return false;
      undoStack.push({
        mesh: entry.mesh,
        placement: { rotation: [...rec.placement.rotation], translation: [...rec.placement.translation] },
      });
      rec.placement = entry.placement;
      emitChange(entry.mesh);
      return true;
    },
    analyze: () => emptyAnalysis(),
    analyzeAsync: async () => emptyAnalysis(),
    dispose: () => {},
    free: () => {},
    onChanges(
      cb: (
        changes: {
          mesh: MeshHandle;
          positions: "all";
          normals: "all";
          colors: number[];
          indices: "all";
          fieldsChanged: FieldHandle[];
          sectionsChanged: never[];
        }[],
      ) => void,
    ) {
      listeners.add(cb);
      return () => {
        listeners.delete(cb);
      };
    },
    boolean: Object.assign(booleanCall, { cut: booleanCutSpy }),
    align: { register: registerSpy },
    inspect: {
      analyze: () => emptyAnalysis(),
      measure: () => ({ area: 24, volume: 8, bounds: { min: [-1, -1, -1], max: [1, 1, 1] }, closed: true }),
      addDistanceField: ({ on }: { on: MeshHandle }) => {
        const fid = nextFieldId++ as unknown as FieldHandle;
        const n = Math.floor(meshes.get(idOf(on))!.positions.length / 3);
        fields.set(fid as unknown as number, Float32Array.from({ length: n }, (_, i) => i * 0.5));
        return fid;
      },
      addThicknessField: ({ on }: { on: MeshHandle }) => {
        const fid = nextFieldId++ as unknown as FieldHandle;
        const n = Math.floor(meshes.get(idOf(on))!.positions.length / 3);
        fields.set(fid as unknown as number, Float32Array.from({ length: n }, (_, i) => 2 + i * 0.1));
        return fid;
      },
      addCurvatureField: ({ on }: { on: MeshHandle }) => {
        const fid = nextFieldId++ as unknown as FieldHandle;
        const n = Math.floor(meshes.get(idOf(on))!.positions.length / 3);
        fields.set(fid as unknown as number, Float32Array.from({ length: n }, (_, i) => (i - 3) * 0.25));
        return fid;
      },
      fieldValues: (f: FieldHandle) => fields.get(f as unknown as number) ?? new Float32Array(),
      removeField: vi.fn((f: FieldHandle) => {
        fields.delete(f as unknown as number);
      }),
      setColoring: vi.fn((mesh: MeshHandle, cfg: { field: FieldHandle } | null) => {
        const rec = meshes.get(idOf(mesh));
        if (!rec) return;
        rec.coloring = cfg ? (fields.get(cfg.field as unknown as number) ?? null) : null;
      }),
    },
  };

  return {
    scene: scene as unknown as Scene & SceneLike,
    spies: { booleanCall, booleanCutSpy, registerSpy, removeMeshSpy: scene.removeMesh },
  };
}

const cleanups: (() => void | Promise<void>)[] = [];

afterEach(async () => {
  while (cleanups.length > 0) {
    await cleanups.pop()?.();
  }
});

async function mountComponent(store: SceneStore, element: ReactNode): Promise<HTMLDivElement> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  let root!: Root;
  await act(async () => {
    root = createRoot(container);
    root.render(createElement(SceneProvider, { store }, element));
  });
  cleanups.push(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
  });
  return container;
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

async function setCheckboxValue(input: HTMLInputElement, checked: boolean): Promise<void> {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "checked")?.set;
  await act(async () => {
    setter?.call(input, checked);
    input.dispatchEvent(new Event("click", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function clickElement(el: HTMLElement): Promise<void> {
  await act(async () => {
    el.click();
  });
}

function addTestItem(store: SceneStore, app: AppState, name: string, hx = 5): Item {
  const { positions, indices } = makeBox(hx, hx, hx);
  const uvs = Float32Array.from({ length: 16 }, (_, i) => (i % 2 === 0 ? 0.25 : 0.75));
  return app.addMesh(store, {
    name,
    positions,
    indices,
    uvs,
    center: [0, 0, 0],
    radius: Math.hypot(hx, hx, hx),
    volumeId: null,
    preset: null,
  });
}

describe("Issue #195 Panel UI & Component Tests (JSDOM)", () => {
  it("ModelingPanel drives Boolean CSG (+ keepOriginals), Plane Cut, Offset (auto/bvhSdf), Shell, Simplify (+ Lock boundaries), and Subdivide", async () => {
    const { scene, spies } = createFakeSyncScene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => store.dispose());

    const a = addTestItem(store, app, "TargetA");
    const b = addTestItem(store, app, "SourceB");
    store.select(a.mesh);

    const container = await mountComponent(store, createElement(ModelingPanel, { app }));

    // 1. Boolean CSG with keepOriginals = true -> adds 3rd item
    expect(container.querySelector<HTMLInputElement>("[data-modeling-keep-originals]")!.checked).toBe(true);
    await clickElement(container.querySelector<HTMLButtonElement>('[data-modeling-boolean-op="union"]')!);
    await clickElement(container.querySelector<HTMLButtonElement>("[data-modeling-boolean-run]")!);
    await act(async () => Promise.resolve());
    expect(spies.booleanCall).toHaveBeenCalledTimes(1);
    expect(app.getSnapshot().items.length).toBe(3);

    // keepOriginals = false -> removes source item
    await setCheckboxValue(container.querySelector<HTMLInputElement>("[data-modeling-keep-originals]")!, false);
    await setSelectValue(container.querySelector<HTMLSelectElement>("[data-modeling-boolean-target]")!, "0");
    await setSelectValue(container.querySelector<HTMLSelectElement>("[data-modeling-boolean-source]")!, "1");
    await clickElement(container.querySelector<HTMLButtonElement>('[data-modeling-boolean-op="subtract"]')!);
    await clickElement(container.querySelector<HTMLButtonElement>("[data-modeling-boolean-run]")!);
    await act(async () => Promise.resolve());
    expect(app.getSnapshot().items.some((i) => i.mesh === b.mesh)).toBe(false);

    // 2. Plane cut: X axis, keep "-", Cap holes unchecked
    await clickElement(container.querySelector<HTMLButtonElement>('[data-modeling-cut-axis="x"]')!);
    await setInputValue(container.querySelector<HTMLInputElement>("[data-modeling-cut-offset]")!, "2");
    await setCheckboxValue(container.querySelector<HTMLInputElement>("[data-modeling-cut-cap]")!, false);
    await clickElement(container.querySelector<HTMLButtonElement>('[data-modeling-cut-side="-"]')!);
    await clickElement(container.querySelector<HTMLButtonElement>("[data-modeling-cut-run]")!);
    await act(async () => Promise.resolve());
    expect(spies.booleanCutSpy).toHaveBeenLastCalledWith(
      a.mesh,
      { origin: [2, 0, 0], normal: [-1, 0, 0] },
      { cap: false },
    );
    expect(spies.removeMeshSpy).toHaveBeenCalled();

    // 3. Offset & Shell
    const repairSpy = vi.spyOn(store, "repair");
    await setSelectValue(container.querySelector<HTMLSelectElement>("[data-modeling-offset-algorithm]")!, "auto");
    await setInputValue(container.querySelector<HTMLInputElement>("[data-modeling-offset-distance]")!, "1.5");
    await clickElement(container.querySelector<HTMLButtonElement>("[data-modeling-offset-run]")!);
    await act(async () => Promise.resolve());
    expect(repairSpy).toHaveBeenLastCalledWith(a.mesh, { op: "offset", distance: 1.5, algorithm: "normal" });

    await setInputValue(container.querySelector<HTMLInputElement>("[data-modeling-shell-thickness]")!, "2");
    await clickElement(container.querySelector<HTMLButtonElement>("[data-modeling-shell-run]")!);
    await act(async () => Promise.resolve());
    expect(repairSpy).toHaveBeenLastCalledWith(a.mesh, { op: "shell", thickness: 2 });

    // 4. Resampling: Simplify w/ Lock boundaries + Subdivide Loop
    expect(container.querySelector<HTMLInputElement>("[data-modeling-lock-boundaries]")!.checked).toBe(true);
    await clickElement(container.querySelector<HTMLButtonElement>("[data-modeling-simplify-mode]")!);
    await setInputValue(container.querySelector<HTMLInputElement>("[data-modeling-simplify-target]")!, "24");
    await clickElement(container.querySelector<HTMLButtonElement>("[data-modeling-simplify-run]")!);
    await act(async () => Promise.resolve());
    expect(repairSpy).toHaveBeenLastCalledWith(a.mesh, {
      op: "simplify",
      targetTriangles: 24,
      preserveBoundary: true,
    });

    await setSelectValue(container.querySelector<HTMLSelectElement>("[data-modeling-subdivide-levels]")!, "2");
    await clickElement(container.querySelector<HTMLButtonElement>("[data-modeling-subdivide-run]")!);
    await act(async () => Promise.resolve());
    expect(repairSpy).toHaveBeenLastCalledWith(a.mesh, { op: "subdivide", method: "loop", levels: 2 });
  });

  it("AlignPanel runs ICP, renders RMSE/status/iterations, and reverts in one undo step", async () => {
    const { scene, spies } = createFakeSyncScene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => store.dispose());

    const m0 = addTestItem(store, app, "Fixed");
    const m1 = addTestItem(store, app, "Moving");
    store.setPlacement(m1.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [5, 0, 0] });
    store.select(m1.mesh);

    const container = await mountComponent(store, createElement(AlignPanel, { app }));
    await setInputValue(container.querySelector<HTMLInputElement>("[data-align-iterations]")!, "35");
    await setInputValue(container.querySelector<HTMLInputElement>("[data-align-overlap]")!, "0.8");
    await clickElement(container.querySelector<HTMLButtonElement>("[data-align-run]")!);
    await act(async () => Promise.resolve());

    expect(spies.registerSpy).toHaveBeenCalledWith(m1.mesh, m0.mesh, {
      maxIterations: 35,
      overlap: 0.8,
      apply: false,
    });
    expect(container.querySelector("[data-align-status]")?.textContent).toBe("converged");
    expect(container.querySelector("[data-align-rmse]")?.textContent).toBe("0.0123 mm");
    expect(container.querySelector("[data-align-iters]")?.textContent).toBe("7");
    expect(scene.placement(m1.mesh).translation).toEqual([0.01, -0.02, 0.03]);

    await store.undo();
    expect(scene.placement(m1.mesh).translation).toEqual([5, 0, 0]);
  });

  it("InspectPanel switches surface field modes (distance, thickness, meanCurvature, gaussianCurvature, none) and colormap/range", async () => {
    const { scene } = createFakeSyncScene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => store.dispose());

    const m0 = addTestItem(store, app, "ShellA");
    addTestItem(store, app, "ShellB");
    store.select(m0.mesh);

    const container = await mountComponent(store, createElement(InspectPanel, { app, scene }));

    for (const mode of ["distance", "thickness", "meanCurvature", "gaussianCurvature"] as const) {
      await clickElement(container.querySelector<HTMLButtonElement>(`[data-inspect-field-mode="${mode}"]`)!);
      await act(async () => Promise.resolve());
      expect(app.getSnapshot().surfaceField.mode).toBe(mode);
      expect(container.querySelector("[data-inspect-field-stats]")?.textContent).toContain("min ");
    }

    await setSelectValue(container.querySelector<HTMLSelectElement>("[data-inspect-field-map]")!, "coolwarm");
    await setInputValue(container.querySelector<HTMLInputElement>("[data-inspect-field-min]")!, "-2");
    await setInputValue(container.querySelector<HTMLInputElement>("[data-inspect-field-max]")!, "8");
    expect(app.getSnapshot().surfaceField.map).toBe("coolwarm");
    expect(app.getSnapshot().surfaceField.min).toBe(-2);
    expect(app.getSnapshot().surfaceField.max).toBe(8);

    await clickElement(container.querySelector<HTMLButtonElement>('[data-inspect-field-mode="none"]')!);
    expect(app.getSnapshot().surfaceField.mode).toBe("none");
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();
  });

  it("TransformPanel updates unit-aware Position, Euler Rotation, uniform/non-uniform Scale, and Scale mode", async () => {
    const { scene } = createFakeSyncScene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => store.dispose());

    const box = addTestItem(store, app, "TBox", 5);
    store.select(box.mesh);

    let mode: "translate" | "rotate" | "scale" = "translate";
    const container = await mountComponent(
      store,
      createElement(TransformPanel, { app, mode, onMode: (m) => (mode = m) }),
    );

    await setInputValue(container.querySelector<HTMLInputElement>('[data-transform-pos="x"]')!, "25.4");
    expect(scene.placement(box.mesh).translation[0]).toBeCloseTo(25.4, 4);

    await act(async () => {
      app.setUnits("in");
    });
    expect(Number(container.querySelector<HTMLInputElement>('[data-transform-pos="x"]')!.value)).toBeCloseTo(1, 3);

    await setInputValue(container.querySelector<HTMLInputElement>('[data-transform-rot="y"]')!, "45");
    expect(scene.placement(box.mesh).rotation[0]).toBeCloseTo(Math.SQRT1_2, 4);

    // Uncheck Uniform and apply non-uniform scale
    await setCheckboxValue(container.querySelector<HTMLInputElement>("[data-transform-scale-uniform]")!, false);
    await setInputValue(container.querySelector<HTMLInputElement>('[data-transform-scale="z"]')!, "2");
    await clickElement(container.querySelector<HTMLButtonElement>("[data-transform-scale-apply]")!);
    await act(async () => Promise.resolve());
    const b = scene.bounds(box.mesh)!;
    expect(b.max[2] - b.min[2]).toBeCloseTo(20, 4);

    await clickElement(container.querySelector<HTMLButtonElement>("[data-transform-mode-scale]")!);
    expect(mode).toBe("scale");
  });

  it("MeshList Duplicate clones geometry/UVs/placement and Mirror flips axis with reversed triangle winding", async () => {
    const { scene } = createFakeSyncScene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => store.dispose());

    const item = addTestItem(store, app, "Widget");
    store.setPlacement(item.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [4, 5, 6] });
    store.select(item.mesh);

    const container = await mountComponent(
      store,
      createElement(MeshList, {
        app,
        onLoad: () => {},
        onCancelLoad: () => {},
        onLoadPhantom: () => {},
        onLoadSample: () => {},
        onLoadDemo: () => {},
        onFocus: () => {},
      }),
    );

    await clickElement(container.querySelector<HTMLButtonElement>('[data-action="mesh-duplicate"]')!);
    const items = app.getSnapshot().items;
    expect(items.length).toBe(2);
    expect(items[1].name).toBe("Widget copy");
    expect(scene.placement(items[1].mesh).translation).toEqual([4, 5, 6]);
    expect(Array.from(scene.views(items[1].mesh).uvs()!)).toEqual(Array.from(scene.views(item.mesh).uvs()!));

    const firstTriBefore = Array.from(scene.views(items[1].mesh).indices().slice(0, 3));
    await setSelectValue(container.querySelector<HTMLSelectElement>('[data-action="mirror-axis"]')!, "y");
    await clickElement(container.querySelector<HTMLButtonElement>('[data-action="mesh-mirror"]')!);
    await act(async () => Promise.resolve());
    const firstTriAfter = Array.from(scene.views(items[1].mesh).indices().slice(0, 3));
    expect(firstTriAfter).toEqual([firstTriBefore[0], firstTriBefore[2], firstTriBefore[1]]);
  });

  it("updates Camera projection/presets state, WorkbenchDomains mapping, and Help shortcut table", async () => {
    const app = new AppState();
    expect(app.getSnapshot().cameraProjection).toBe("perspective");
    app.toggleCameraProjection();
    expect(app.getSnapshot().cameraProjection).toBe("orthographic");
    for (const preset of ["front", "back", "left", "right", "top", "bottom"] as CameraPreset[]) {
      app.requestCameraPreset(preset);
      expect(app.getSnapshot().cameraPresetRequest?.preset).toBe(preset);
    }

    expect(domainMatchesPanel("boolean", "modeling")).toBe(true);
    expect(domainMatchesPanel("remesh", "modeling")).toBe(true);
    expect(domainMatchesPanel("align", "align")).toBe(true);

    let active = "mesh" as Parameters<typeof WorkbenchDomains>[0]["activeDomain"];
    const { scene } = createFakeSyncScene();
    const store = createSceneStore(scene);
    cleanups.push(() => store.dispose());
    const domContainer = await mountComponent(
      store,
      createElement(WorkbenchDomains, { activeDomain: active, onSelectDomain: (d) => (active = d) }),
    );
    const booleanBtn = Array.from(domContainer.querySelectorAll<HTMLButtonElement>("button")).find((btn) =>
      /boolean/i.test(btn.textContent ?? ""),
    )!;
    await clickElement(booleanBtn);
    expect(active).toBe("boolean");

    const helpContainer = await mountComponent(
      store,
      createElement(HelpOverlay, { onClose: () => {}, onStartTour: () => {} }),
    );
    expect(helpContainer.textContent).toContain("Scale gizmo");
  });
});
