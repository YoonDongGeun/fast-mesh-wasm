// The selected mesh's live analysis: analysed on selection and again
// whenever its triangles change (a repair, its undo, redo), coalesced to one
// analysis per burst of changes. The summary feeds the panel, the raw
// analysis the 3D overlays (through the app state), and the test hook reads
// both through `inspect.repair`. Analysis is read-only: the app calls it on
// its own scene.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useCallback, useEffect, useRef, useState } from "react";
import type { AppState } from "../../../app/state/app-state";
import { inspect } from "../../../app/state/readouts";
import { type AnalysisSummary, summarize } from "./repair-summary";

/** The most entries the analysis lists (counts stay exact). */
export const MAX_LISTED = 64;

export function useRepairAnalysis(args: {
  app: AppState;
  scene: Scene | WorkerScene;
  store: SceneStore;
  mesh: MeshHandle | null;
  /** The weld tolerance the analysis groups coincident vertices by. */
  weldTolerance: number;
  /** Called on each triangle change of `mesh`, before its re-analysis is scheduled (stable identity). */
  onTriangleChange: () => void;
}) {
  const { app, scene, store, mesh, weldTolerance, onTriangleChange } = args;
  const [analysis, setAnalysis] = useState<{ mesh: MeshHandle; summary: AnalysisSummary } | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [checkSelfIntersections, setCheckSelfIntersections] = useState(false);
  const [selfIntersects, setSelfIntersects] = useState<boolean | null>(null);
  const seq = useRef(0);
  const weldRef = useRef(weldTolerance);
  weldRef.current = weldTolerance;
  const checkSiRef = useRef(checkSelfIntersections);
  checkSiRef.current = checkSelfIntersections;

  const analyze = useCallback(
    async (target: MeshHandle) => {
      const n = ++seq.current;
      setAnalyzing(true);
      try {
        const a = await scene.analyze(target, {
          weldTolerance: weldRef.current,
          maxListed: MAX_LISTED,
          selfIntersection: checkSiRef.current,
        });
        if (n === seq.current) {
          const summary = summarize(a);
          setAnalysis({ mesh: target, summary });
          setSelfIntersects(a.selfIntersects);
          // The 3D overlay shares the raw analysis (loop vertices, edge and
          // triangle lists the summary drops); a new object refreshes it.
          app.setRepairAnalysis({ mesh: target, analysis: a });
          const name = app.getSnapshot().items.find((i) => i.mesh === target)?.name ?? "";
          inspect.repair.analysis = { mesh: name, summary };
          inspect.repair.analyses++;
        }
      } catch {
        // Removed meanwhile: the selection follows.
      } finally {
        if (n === seq.current) {
          setAnalyzing(false);
        }
      }
    },
    [scene, app],
  );

  // Analyse on selection, and again whenever its triangles change (a
  // repair, undo, redo), coalesced to one analysis per burst of changes.
  useEffect(() => {
    if (mesh === null) {
      seq.current++;
      setAnalysis(null);
      setSelfIntersects(null);
      setAnalyzing(false);
      inspect.repair.analysis = null;
      app.setRepairAnalysis(null);
      return;
    }
    void analyze(mesh);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const off = store.scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === mesh && !c.removed && (c.indices === "all" || c.indices.length > 0))) {
        onTriangleChange();
        clearTimeout(timer);
        timer = setTimeout(() => void analyze(mesh), 50);
      }
    });
    return () => {
      off();
      clearTimeout(timer);
    };
  }, [store, mesh, analyze, onTriangleChange]);

  /** Turns self-intersection detection in the live analysis on or off, and re-analyses. */
  const toggleSelfIntersections = (on: boolean) => {
    setCheckSelfIntersections(on);
    checkSiRef.current = on;
    if (mesh !== null) {
      void analyze(mesh);
    }
  };

  return {
    analysis,
    analyzing,
    selfIntersects,
    setSelfIntersects,
    checkSelfIntersections,
    toggleSelfIntersections,
    analyze,
  };
}

export type LiveRepairAnalysis = ReturnType<typeof useRepairAnalysis>;
