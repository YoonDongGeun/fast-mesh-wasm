// The Multi-Clone section's behaviour (#266): the array's layout (pattern,
// copies, columns or axis, spacing), whether it can run on the selected
// mesh, and the run, one undo step, with its report.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { type CloneLayout, cloneMesh, cloneProblem, DEFAULT_CLONE_LAYOUT } from "./multi-clone";

export function useMultiClone(app: AppState, store: SceneStore, selectedItem: Item | null) {
  const units = useApp(app, (s) => s.units);
  const up = useApp(app, (s) => s.buildPlate.up);
  const [layout, setLayout] = useState<CloneLayout>(DEFAULT_CLONE_LAYOUT);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<{ text: string; error: boolean } | null>(null);

  const problem = selectedItem === null ? null : cloneProblem(layout, selectedItem.triangles);
  const ready = !busy && selectedItem !== null && problem === null;

  /** Makes the copies of the selected mesh (one undo step) and selects them with it. */
  const run = async () => {
    if (!ready || selectedItem === null) return;
    setBusy(true);
    setReport(null);
    try {
      const copies = await cloneMesh({ store, app, item: selectedItem, layout, up });
      const made = `${copies.length} ${copies.length === 1 ? "copy" : "copies"}`;
      setReport({ text: `Added ${made} of ${selectedItem.name}, selected with it`, error: false });
    } catch (e) {
      setReport({ text: e instanceof Error ? e.message : String(e), error: true });
    } finally {
      setBusy(false);
    }
  };

  return {
    units,
    up,
    layout,
    set: (patch: Partial<CloneLayout>) => setLayout((l) => ({ ...l, ...patch })),
    problem,
    ready,
    busy,
    report,
    run,
  };
}
