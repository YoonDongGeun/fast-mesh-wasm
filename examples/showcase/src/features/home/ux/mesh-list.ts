// The mesh list's behaviour: the large-file thresholds from the URL and the
// heavy meshes past them, each load's phase label, the group folder of each
// mesh, a row's live triangle count and its name click, the selected mesh's
// duplicate and mirror actions, and the import scale as typed.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState } from "react";
import { BIG_FILE_DEFAULTS, estimateMeshBytes, parseBigFileParams } from "../../../io/big-file";
import { type AppState, type Item, type Job, useApp } from "../../../app/state/app-state";
import { selectMesh } from "../../select/ux/selection";
import { duplicateMesh, type MirrorAxis, mirrorMesh } from "../../../engine/workbench-ops";
import { type ImportScale, importScaleFromFactor, importScaleOf, isImportUnit, parseImportScale } from "../../../shared/ux/units";

/** A load's phase as the list shows it. */
export const LOAD_PHASE: Record<Job["phase"], string> = {
  reading: "reading…",
  parsing: "parsing…",
  decimating: "decimating…",
  adding: "adding…",
  done: "loaded",
  error: "failed",
};

/** Reads the large-file thresholds from the URL (tolerant; see big-file.ts). */
export function bigFileParams(): { warnTriangles: number; decimateTriangles: number; decimateTarget: number } {
  try {
    const p = parseBigFileParams(location.search);
    return { warnTriangles: p.warnTriangles, decimateTriangles: p.decimateTriangles, decimateTarget: p.decimateTarget };
  } catch {
    return {
      warnTriangles: BIG_FILE_DEFAULTS.warnTriangles,
      decimateTriangles: BIG_FILE_DEFAULTS.decimateTriangles,
      decimateTarget: BIG_FILE_DEFAULTS.decimateTarget,
    };
  }
}

/** The meshes of at least `warnTriangles`, in list order, and their estimated memory in bytes. */
export function heavyMeshes<T extends { triangles: number }>(items: readonly T[], warnTriangles: number): { heavy: T[]; bytes: number } {
  const heavy = items.filter((i) => i.triangles >= warnTriangles);
  return { heavy, bytes: heavy.reduce((n, i) => n + estimateMeshBytes(i.triangles), 0) };
}

/** Each grouped mesh's folder: the group it is a member of (the last listed, when several). */
export function meshGroupIndex<M>(
  groups: readonly { id: number; name: string; members: readonly M[] }[],
): Map<M, { id: number; name: string }> {
  const map = new Map<M, { id: number; name: string }>();
  for (const g of groups) {
    for (const m of g.members) {
      map.set(m, { id: g.id, name: g.name });
    }
  }
  return map;
}

/** "12,288" or "1.05 M". */
export function formatCount(n: number): string {
  return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(2)} M` : n.toLocaleString("en");
}

/** The triangles of an index buffer, a removed one (`0, 0, 0`) not counted. */
export function countTriangles(idx: ArrayLike<number>): number {
  let n = 0;
  for (let t = 0; t < idx.length; t += 3) {
    if (idx[t] !== 0 || idx[t + 1] !== 0 || idx[t + 2] !== 0) {
      n++;
    }
  }
  return n;
}

/**
 * The mesh's live triangle count: as loaded, then recounted from the scene's
 * index view whenever its triangles change (repairs, their undo).
 */
export function useTriangleCount(store: SceneStore, item: Item): number {
  const [count, setCount] = useState(item.triangles);
  useEffect(
    () =>
      store.scene.onChanges((changes) => {
        for (const c of changes) {
          if (c.mesh !== item.mesh || c.removed || c.added) {
            continue;
          }
          if (c.indices === "all" || c.indices.length > 0) {
            setCount(countTriangles(store.scene.views(item.mesh).indices()));
          }
        }
      }),
    [store, item.mesh],
  );
  return count;
}

/**
 * A click on a row's name selects its mesh. Plain clicks reset (a lone
 * selection toggles off); shift-clicks toggle.
 */
export function clickRow(
  app: AppState,
  select: (mesh: MeshHandle | null) => void,
  selection: MeshHandle | null,
  mesh: MeshHandle,
  multi: readonly MeshHandle[],
  shift: boolean,
): void {
  const target = shift || selection !== mesh || multi.length > 1 ? mesh : null;
  selectMesh(app, select, selection, target, shift && target !== null);
}

/**
 * The selected mesh's actions: duplicate it, or mirror it across the chosen
 * axis (`mirrorBusy` until the mirror lands).
 */
export function useMeshActions(
  store: SceneStore,
  app: AppState,
  selected: Item | null,
): {
  mirrorAxis: MirrorAxis;
  setMirrorAxis: (axis: MirrorAxis) => void;
  mirrorBusy: boolean;
  duplicate: () => void;
  mirror: () => void;
} {
  const [mirrorAxis, setMirrorAxis] = useState<MirrorAxis>("x");
  const [mirrorBusy, setMirrorBusy] = useState(false);
  return {
    mirrorAxis,
    setMirrorAxis,
    mirrorBusy,
    duplicate: () => {
      if (selected) {
        duplicateMesh(store, app, selected);
      }
    },
    mirror: () => {
      if (!selected) return;
      setMirrorBusy(true);
      void mirrorMesh(store, selected.mesh, mirrorAxis).finally(() => setMirrorBusy(false));
    },
  };
}

/**
 * The import scale control: a unit picked, or a factor as typed (`draft`
 * while the field has focus). An unusable factor is `invalid` and not
 * applied.
 */
export function useImportScaleDraft(app: AppState): {
  scale: ImportScale;
  draft: string | null;
  invalid: boolean;
  chooseUnits: (units: string) => void;
  typeFactor: (text: string) => void;
  endEdit: () => void;
} {
  const scale = useApp(app, (s) => s.importScale);
  // The factor as typed: an unusable one shows as invalid and is not applied.
  const [draft, setDraft] = useState<string | null>(null);
  return {
    scale,
    draft,
    invalid: draft !== null && parseImportScale(draft) === null,
    chooseUnits: (units) => {
      if (isImportUnit(units)) {
        setDraft(null);
        app.setImportScale(importScaleOf(units));
      }
    },
    typeFactor: (text) => {
      setDraft(text);
      const factor = parseImportScale(text);
      if (factor !== null) {
        app.setImportScale(importScaleFromFactor(factor));
      }
    },
    endEdit: () => setDraft(null),
  };
}
