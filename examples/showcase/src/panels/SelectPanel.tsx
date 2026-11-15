// Face Selection workbench panel (#233): interactive viewport tool selector,
// combine modes, region/brush parameters, selection modifiers (`grow`, `shrink`,
// `invert`, `fillSelectionHoles`, `smoothSelectionBoundary`, `selectSmallFaceIslands`,
// mask conversion), and selection-driven mesh operations (`extract`, `split`, `delete`).
import type { AdjacencyKind } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState, useSyncExternalStore } from "react";
import { type AppState, useApp } from "../scene/app-state";
import type { Wave6SelectState } from "../scene/wave6-state";
import {
  combineFaceSelection,
  countLiveTriangles,
  invertFaceSelection,
  resolveActiveSelectItem,
  sanitizeSelectedTriangles,
  writeMeshFaceSelection,
} from "../viewport/FaceSelectionLayer";

export interface SelectPanelProps {
  app: AppState;
  store: SceneStore;
}

type SelectToolId = NonNullable<Wave6SelectState["activeTool"]>;

interface OptionChip<T extends string> {
  readonly id: T;
  readonly label: string;
}

const TOOL_OPTIONS: readonly OptionChip<SelectToolId>[] = [
  { id: "box", label: "Box" },
  { id: "lasso", label: "Lasso" },
  { id: "brush", label: "Brush" },
  { id: "crease", label: "Crease" },
  { id: "shell", label: "Shell" },
];

const COMBINE_OPTIONS: readonly OptionChip<Wave6SelectState["combineMode"]>[] = [
  { id: "replace", label: "Replace" },
  { id: "add", label: "Add" },
  { id: "subtract", label: "Subtract" },
  { id: "intersect", label: "Intersect" },
];

const ADJACENCY_OPTIONS: readonly OptionChip<AdjacencyKind>[] = [
  { id: "edge", label: "Edge" },
  { id: "vertex", label: "Vertex" },
];

const EMPTY_U32 = new Uint32Array(0);

interface ReportState {
  text: string;
  error: boolean;
}

/** Control panel for Wave 6 interactive face selection and face-set mesh editing. */
export function SelectPanel({ app, store }: SelectPanelProps) {
  const select = useApp(app, (s) => s.wave6.select);
  const items = useApp(app, (s) => s.items);
  const sculptMask = useApp(app, (s) => s.sculptMask);
  const storeSnap = useSyncExternalStore(store.subscribe, () => store.getSnapshot());
  const [revision, setRevision] = useState(0);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<ReportState | null>(null);

  useEffect(() => {
    const unsub = store.scene.onChanges((changes) => {
      if (changes.length > 0) {
        setRevision((r) => r + 1);
      }
    });
    return unsub;
  }, [store]);

  // Trigger live triangle / index sync after selection edits or scene changes.
  void revision;
  void storeSnap;

  const activeItem = resolveActiveSelectItem(app, store);
  const rawIndices = activeItem ? store.scene.views(activeItem.mesh).indices() : null;
  const liveTriCount = activeItem
    ? rawIndices
      ? countLiveTriangles(rawIndices)
      : activeItem.triangles
    : 0;
  const rawSelected = activeItem
    ? (select.selectedByMesh[activeItem.id] ?? EMPTY_U32)
    : EMPTY_U32;
  const selectedTris = activeItem
    ? rawIndices
      ? sanitizeSelectedTriangles(rawIndices, rawSelected)
      : rawSelected
    : EMPTY_U32;
  const selectedCount = selectedTris.length;
  const hasSelection = selectedCount > 0;
  const hasRepair = Boolean(store.scene.repair);

  const patchSelect = (patch: Partial<Wave6SelectState>) => {
    app.updateWave6((prev) => ({
      ...prev,
      select: {
        ...prev.select,
        ...patch,
      },
    }));
  };

  const runAction = async (label: string, fn: () => Promise<string> | string) => {
    if (busy) {
      return;
    }
    setBusy(true);
    try {
      const text = await fn();
      setReport({ text, error: false });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setReport({ text: `${label} failed: ${message}`, error: true });
    } finally {
      setBusy(false);
    }
  };

  const requireRepair = () => {
    const repair = store.scene.repair;
    if (!repair) {
      throw new Error("Repair domain is unavailable on the current scene.");
    }
    return repair;
  };

  const onClear = () => {
    if (!activeItem) return;
    writeMeshFaceSelection(app, activeItem.id, EMPTY_U32);
    setReport({ text: "Cleared selection.", error: false });
  };

  const onInvert = () => {
    if (!activeItem) return;
    const indices = store.scene.views(activeItem.mesh).indices();
    if (!indices) {
      setReport({ text: "Mesh indices unavailable.", error: true });
      return;
    }
    const inverted = invertFaceSelection(indices, selectedTris);
    writeMeshFaceSelection(app, activeItem.id, inverted);
    setReport({ text: `Inverted selection (${inverted.length} faces).`, error: false });
  };

  const onGrow = () => {
    if (!activeItem || !hasSelection) return;
    void runAction("Grow", async () => {
      const repair = requireRepair();
      const next = await Promise.resolve(
        repair.growFaceSelection(activeItem.mesh, selectedTris, 1, select.adjacency),
      );
      writeMeshFaceSelection(app, activeItem.id, next);
      return `Grew selection to ${next.length} faces.`;
    });
  };

  const onShrink = () => {
    if (!activeItem || !hasSelection) return;
    void runAction("Shrink", async () => {
      const repair = requireRepair();
      const next = await Promise.resolve(
        repair.shrinkFaceSelection(activeItem.mesh, selectedTris, 1, select.adjacency),
      );
      writeMeshFaceSelection(app, activeItem.id, next);
      return `Shrank selection to ${next.length} faces.`;
    });
  };

  const onFillSelectionHoles = () => {
    if (!activeItem || !hasSelection) return;
    void runAction("Fill selection holes", async () => {
      const repair = requireRepair();
      const maxHoleTris = Math.max(1, Math.round(select.islandMaxFaces));
      const next = await Promise.resolve(
        repair.fillSelectionHoles(activeItem.mesh, selectedTris, maxHoleTris),
      );
      writeMeshFaceSelection(app, activeItem.id, next);
      return `Filled selection holes (${next.length} faces).`;
    });
  };

  const onSmoothBoundary = () => {
    if (!activeItem || !hasSelection) return;
    void runAction("Smooth boundary", async () => {
      const repair = requireRepair();
      const next = await Promise.resolve(
        repair.smoothSelectionBoundary(activeItem.mesh, selectedTris, 1),
      );
      writeMeshFaceSelection(app, activeItem.id, next);
      return `Smoothed boundary (${next.length} faces).`;
    });
  };

  const onSelectIslands = () => {
    if (!activeItem) return;
    void runAction("Small islands", async () => {
      const repair = requireRepair();
      const maxFaces = Math.max(1, Math.round(select.islandMaxFaces));
      const found = await Promise.resolve(repair.selectSmallFaceIslands(activeItem.mesh, maxFaces));
      const combined = combineFaceSelection(selectedTris, found, select.combineMode);
      writeMeshFaceSelection(app, activeItem.id, combined);
      return `Matched ${found.length} island faces (${combined.length} total).`;
    });
  };

  const onToMask = () => {
    if (!activeItem || !hasSelection) return;
    void runAction("Selection to mask", async () => {
      const repair = requireRepair();
      const weights = await Promise.resolve(
        repair.faceSelectionToVertexMask(activeItem.mesh, selectedTris),
      );
      const maskedIds: number[] = [];
      for (let v = 0; v < weights.length; v++) {
        if (weights[v] >= 0.5) {
          maskedIds.push(v);
        }
      }
      app.setSculptMask({
        mesh: activeItem.mesh,
        vertices: Uint32Array.from(maskedIds),
        slots: weights.length,
      });
      return `Converted ${selectedCount} faces to ${maskedIds.length} masked vertices.`;
    });
  };

  const onFromMask = () => {
    if (!activeItem || !sculptMask || sculptMask.mesh !== activeItem.mesh) return;
    void runAction("Mask to selection", async () => {
      const repair = requireRepair();
      const positions = store.scene.views(activeItem.mesh).positions();
      const vertCount = positions ? Math.floor(positions.length / 3) : sculptMask.slots;
      const weights = new Float32Array(vertCount);
      for (let i = 0; i < sculptMask.vertices.length; i++) {
        const v = sculptMask.vertices[i];
        if (v < vertCount) {
          weights[v] = 1;
        }
      }
      const faces = await Promise.resolve(
        repair.vertexMaskToFaceSelection(activeItem.mesh, weights, 0.5, true),
      );
      const combined = combineFaceSelection(selectedTris, faces, select.combineMode);
      writeMeshFaceSelection(app, activeItem.id, combined);
      return `Loaded ${faces.length} faces from sculpt mask.`;
    });
  };

  const onExtract = () => {
    if (!activeItem || !hasSelection) return;
    void runAction("Extract faces", async () => {
      const repair = requireRepair();
      const extracted = await Promise.resolve(
        repair.extractSelectedFaces(activeItem.mesh, selectedTris),
      );
      const adopted = app.adoptSceneMesh(store, extracted, `${activeItem.name} (Extract)`);
      store.commitFrame();
      setRevision((r) => r + 1);
      return `Extracted ${selectedCount} faces as "${adopted.name}".`;
    });
  };

  const onSplit = () => {
    if (!activeItem || !hasSelection) return;
    void runAction("Split faces", async () => {
      const repair = requireRepair();
      const result = await Promise.resolve(
        repair.splitSelectedFaces(activeItem.mesh, selectedTris),
      );
      writeMeshFaceSelection(app, activeItem.id, EMPTY_U32);
      const adopted = app.adoptSceneMesh(store, result.detached, `${activeItem.name} (Split)`);
      store.commitFrame();
      setRevision((r) => r + 1);
      return `Split ${result.removedTriangles} faces into "${adopted.name}".`;
    });
  };

  const onDelete = () => {
    if (!activeItem || !hasSelection) return;
    void runAction("Delete faces", async () => {
      const repair = requireRepair();
      const result = await Promise.resolve(
        repair.deleteSelectedFaces(activeItem.mesh, selectedTris),
      );
      writeMeshFaceSelection(app, activeItem.id, EMPTY_U32);
      store.commitFrame();
      setRevision((r) => r + 1);
      return `Deleted ${result.removedTriangles} faces (${result.removedVertices} vertices).`;
    });
  };

  const onFillMeshHoles = () => {
    if (!activeItem) return;
    void runAction("Fill holes", async () => {
      const repair = requireRepair();
      const maxEdges = Math.max(1024, Math.round(select.islandMaxFaces));
      const result = await Promise.resolve(repair.fillHoles(activeItem.mesh, { maxEdges }));
      store.commitFrame();
      setRevision((r) => r + 1);
      const skippedCount = result.skipped.length + result.failed.length;
      return skippedCount > 0
        ? `Filled ${result.filled.length} mesh hole(s) (${skippedCount} skipped/failed).`
        : `Filled ${result.filled.length} mesh hole(s).`;
    });
  };

  const canConvertFromMask = Boolean(
    activeItem && sculptMask && sculptMask.mesh === activeItem.mesh && sculptMask.vertices.length > 0,
  );

  return (
    <section className="panel select-panel" aria-label="Face Selection" data-panel="select">
      <header className="panel-head">
        <h2>Select</h2>
      </header>

      <div
        className="readout"
        data-select-readout
        data-select-count={selectedCount}
        data-select-total={liveTriCount}
      >
        {activeItem
          ? `${selectedCount} / ${liveTriCount} triangles — ${activeItem.name}`
          : items.length === 0
            ? "No mesh loaded"
            : "No active mesh"}
      </div>

      <div className="op">
        <div className="op-row">
          <span>Selection Tool</span>
          <button
            type="button"
            disabled={busy || select.activeTool === null}
            data-select-tool="off"
            data-select-disarm="true"
            aria-pressed={select.activeTool === null}
            onClick={() => patchSelect({ activeTool: null })}
          >
            Off
          </button>
        </div>
        <div className="modes" role="group" aria-label="Selection tool">
          {TOOL_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              disabled={busy}
              data-select-tool={opt.id}
              aria-pressed={select.activeTool === opt.id}
              onClick={() => patchSelect({ activeTool: opt.id })}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="op">
        <span>Combine Mode</span>
        <div className="modes" role="group" aria-label="Combine mode">
          {COMBINE_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              disabled={busy}
              data-select-combine={opt.id}
              aria-pressed={select.combineMode === opt.id}
              onClick={() => patchSelect({ combineMode: opt.id })}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="op">
        <label className="check">
          <input
            type="checkbox"
            checked={select.visibleOnly}
            disabled={busy}
            data-select-visible-only
            onChange={(e) => patchSelect({ visibleOnly: e.currentTarget.checked })}
          />
          <span>Visible faces only</span>
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={select.backfaceCull}
            disabled={busy}
            data-select-backface-cull
            onChange={(e) => patchSelect({ backfaceCull: e.currentTarget.checked })}
          />
          <span>Ignore backfaces</span>
        </label>

        <div className="slider">
          <label htmlFor="select-brush-radius">
            <span>Brush radius</span>
            <span>{select.brushRadius.toFixed(2)}</span>
          </label>
          <input
            id="select-brush-radius"
            type="range"
            min={0.01}
            max={5}
            step={0.01}
            value={select.brushRadius}
            disabled={busy}
            data-select-brush-radius
            onChange={(e) => patchSelect({ brushRadius: Number(e.currentTarget.value) })}
          />
        </div>

        <div className="slider">
          <label htmlFor="select-crease-angle">
            <span>Crease angle</span>
            <span>{Math.round(select.creaseAngleDeg)} deg</span>
          </label>
          <input
            id="select-crease-angle"
            type="range"
            min={1}
            max={180}
            step={1}
            value={select.creaseAngleDeg}
            disabled={busy}
            data-select-crease-angle
            onChange={(e) => patchSelect({ creaseAngleDeg: Number(e.currentTarget.value) })}
          />
        </div>

        <div className="slider">
          <label htmlFor="select-island-max-faces">
            <span>Max island / hole tris</span>
            <span>{Math.round(select.islandMaxFaces)}</span>
          </label>
          <input
            id="select-island-max-faces"
            type="range"
            min={1}
            max={500}
            step={1}
            value={select.islandMaxFaces}
            disabled={busy}
            data-select-island-max-faces
            onChange={(e) => patchSelect({ islandMaxFaces: Number(e.currentTarget.value) })}
          />
        </div>

        <div className="op-row">
          <span>Adjacency</span>
          <div className="modes" role="group" aria-label="Grow/shrink adjacency">
            {ADJACENCY_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                disabled={busy}
                data-select-adjacency={opt.id}
                aria-pressed={select.adjacency === opt.id}
                onClick={() => patchSelect({ adjacency: opt.id })}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {!activeItem ? (
        <p className="muted">Load or select a mesh to edit its face selection.</p>
      ) : (
        <>
          <div className="op">
            <span>Modify Selection</span>
            <div className="op-row">
              <button
                type="button"
                disabled={busy || !hasSelection}
                data-select-action="clear"
                onClick={onClear}
              >
                Clear
              </button>
              <button
                type="button"
                disabled={busy || liveTriCount === 0}
                data-select-action="invert"
                onClick={onInvert}
              >
                Invert
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="grow"
                onClick={onGrow}
              >
                Grow
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="shrink"
                onClick={onShrink}
              >
                Shrink
              </button>
            </div>
            <div className="op-row">
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="fill-selection-holes"
                onClick={onFillSelectionHoles}
              >
                Fill Holes
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="smooth-boundary"
                onClick={onSmoothBoundary}
              >
                Smooth
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || liveTriCount === 0}
                data-select-action="select-islands"
                onClick={onSelectIslands}
              >
                Small Islands
              </button>
            </div>
            <div className="op-row">
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="to-mask"
                onClick={onToMask}
              >
                To Mask
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !canConvertFromMask}
                data-select-action="from-mask"
                onClick={onFromMask}
              >
                From Mask
              </button>
            </div>
          </div>

          <div className="op">
            <span>Mesh Actions</span>
            <div className="op-row">
              <button
                type="button"
                className="primary"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="extract"
                onClick={onExtract}
              >
                Extract
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="split"
                onClick={onSplit}
              >
                Split
              </button>
              <button
                type="button"
                className="danger"
                disabled={busy || !hasRepair || !hasSelection}
                data-select-action="delete"
                onClick={onDelete}
              >
                Delete
              </button>
              <button
                type="button"
                disabled={busy || !hasRepair || liveTriCount === 0}
                data-select-action="fill-holes"
                onClick={onFillMeshHoles}
              >
                Fill Mesh Holes
              </button>
            </div>
          </div>
        </>
      )}

      {report && (
        <p
          className={report.error ? "report error" : "report"}
          data-select-report
          data-select-error={report.error ? "true" : "false"}
        >
          {report.text}
        </p>
      )}
    </section>
  );
}
