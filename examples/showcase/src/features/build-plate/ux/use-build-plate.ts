// The build plate panel's behaviour (#266): the plate's size, place and
// options (kept in the app state, so they persist and the viewport draws
// them), which meshes Arrange works on, and Arrange itself, one undo step,
// with its report.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useStoreSelector } from "fast-mesh-wasm/react";
import { useState } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { arrangeMeshes, arrangeTargets, describeArrange, meshVisible } from "./arrange";
import { type BuildPlate, plateProblem } from "./plate";

/** What the panel says about the last run: a line, and one per reason a mesh was left where it was. */
export interface PlateReport {
  text: string;
  left: string[];
  error: boolean;
}

export function useBuildPlate(app: AppState, store: SceneStore) {
  const plate = useApp(app, (s) => s.buildPlate);
  const units = useApp(app, (s) => s.units);
  const items = useApp(app, (s) => s.items);
  const multi = useApp(app, (s) => s.multi);
  const selection = useStoreSelector((s) => s.selection);
  // Which meshes show, as a string so that only a change of it renders again.
  useStoreSelector((s) => items.map((i) => (s.meshes.get(i.mesh)?.desired.visible ? "1" : "0")).join(""));
  const [scope, setScope] = useState<"selected" | "all">("selected");
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<PlateReport | null>(null);

  const count = (of: "selected" | "all") =>
    arrangeTargets(of, items, multi, selection, (m) => meshVisible(store, m)).length;
  const selected = count("selected");
  const all = count("all");
  const problem = plateProblem(plate);
  const ready = !busy && problem === null && (scope === "selected" ? selected : all) > 0;

  const set = (patch: Partial<BuildPlate>) => app.setBuildPlate(patch);

  /** Arranges the scope's meshes (one undo step), and shows the plate. */
  const arrange = async () => {
    if (!ready) return;
    setBusy(true);
    setReport(null);
    app.setBuildPlate({ visible: true });
    try {
      const targets = arrangeTargets(scope, items, multi, selection, (m) => meshVisible(store, m));
      const done = await arrangeMeshes({ store, app, items: targets, plate: app.getSnapshot().buildPlate });
      setReport({ ...describeArrange(done), error: done.placed === 0 });
    } catch (e) {
      setReport({ text: e instanceof Error ? e.message : String(e), left: [], error: true });
    } finally {
      setBusy(false);
    }
  };

  return { plate, units, set, scope, setScope, selected, all, problem, ready, busy, report, arrange };
}
