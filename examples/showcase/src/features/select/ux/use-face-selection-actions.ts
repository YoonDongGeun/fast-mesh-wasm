// The select panel's behaviour (#233): the tool, combine and adjacency
// options it offers, the active mesh's live and selected triangle counts,
// and the selection modifiers (grow, shrink, invert, holes, boundary,
// islands, mask conversion) and face-set mesh actions (extract, split,
// delete, fill holes) through `store.scene.repair`. One action runs at a
// time; each reports a line of text or its error.
import type { AdjacencyKind } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState, useSyncExternalStore } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import type { Wave6SelectState } from "../../../app/state/wave6-state";
import { resolveActiveSelectItem, writeMeshFaceSelection } from "./face-selection-ops";
import {
  combineFaceSelection,
  countLiveTriangles,
  EMPTY_U32,
  invertFaceSelection,
  sanitizeSelectedTriangles,
} from "./face-sets";
import { useSceneRevision } from "./use-scene-revision";

type SelectToolId = NonNullable<Wave6SelectState["activeTool"]>;

interface OptionChip<T extends string> {
  readonly id: T;
  readonly label: string;
}

export const TOOL_OPTIONS: readonly OptionChip<SelectToolId>[] = [
  { id: "box", label: "Box" },
  { id: "lasso", label: "Lasso" },
  { id: "brush", label: "Brush" },
  { id: "crease", label: "Crease" },
  { id: "shell", label: "Shell" },
];

export const COMBINE_OPTIONS: readonly OptionChip<Wave6SelectState["combineMode"]>[] = [
  { id: "replace", label: "Replace" },
  { id: "add", label: "Add" },
  { id: "subtract", label: "Subtract" },
  { id: "intersect", label: "Intersect" },
];

export const ADJACENCY_OPTIONS: readonly OptionChip<AdjacencyKind>[] = [
  { id: "edge", label: "Edge" },
  { id: "vertex", label: "Vertex" },
];

/** The last action's report line, flagged when it failed. */
export interface ReportState {
  text: string;
  error: boolean;
}

/**
 * The select panel's state and handlers. The active mesh is the primary
 * selection, else the first visible mesh; its selection is sanitized
 * against the live index buffer, re-read on every scene change.
 */
export function useFaceSelectionActions(app: AppState, store: SceneStore) {
  const select = useApp(app, (s) => s.wave6.select);
  const items = useApp(app, (s) => s.items);
  const sculptMask = useApp(app, (s) => s.sculptMask);
  const storeSnap = useSyncExternalStore(store.subscribe, () => store.getSnapshot());
  const [revision, setRevision] = useSceneRevision(store);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<ReportState | null>(null);

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

  return {
    select,
    items,
    busy,
    report,
    activeItem,
    liveTriCount,
    selectedCount,
    hasSelection,
    hasRepair,
    canConvertFromMask,
    patchSelect,
    onClear,
    onInvert,
    onGrow,
    onShrink,
    onFillSelectionHoles,
    onSmoothBoundary,
    onSelectIslands,
    onToMask,
    onFromMask,
    onExtract,
    onSplit,
    onDelete,
    onFillMeshHoles,
  };
}
