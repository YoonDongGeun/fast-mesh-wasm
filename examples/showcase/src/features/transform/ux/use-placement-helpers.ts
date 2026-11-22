// The placement helpers' panel behaviour (#256): one-click moves (centre or
// bottom centre to the origin, apply transform) and the surface picks (lay
// flat, align to axis, 3-2-1), which arm a pick on the selected mesh and
// disarm the other pointer tools. A pick stands down when the selection
// leaves its mesh, the mesh locks or goes, or another tool arms.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { formatLength } from "../../../shared/ux/units";
import { applyTransform, placeMesh } from "./placement-helpers";
import { moveToOrigin, type OriginAnchor, type SignedAxis, type UpAxis } from "./placement-math";
import type { PlacementPickKind } from "./placement-picks";

/** The panel's state and actions for `selectedItem` (nothing runs without one, or while it is locked). */
export function usePlacementHelpers(
  app: AppState,
  store: SceneStore,
  selectedItem: Item | null,
  /** Called once Apply transform has baked the placement. */
  onBaked?: () => void,
) {
  const tool = useApp(app, (s) => s.placementTool);
  const units = useApp(app, (s) => s.units);
  const otherToolArmed = useApp(
    app,
    (s) =>
      s.sculpt !== null ||
      s.maskPaint ||
      s.inspectTool !== null ||
      s.wave6.select.activeTool !== null ||
      s.wave6.deform.activeTool !== null ||
      s.wave6.printPrep.activeTool !== null,
  );
  const [up, setUp] = useState<UpAxis>("z");
  const [target, setTarget] = useState<SignedAxis>("+z");
  const [busy, setBusy] = useState(false);

  const mesh: MeshHandle | null = selectedItem === null || selectedItem.locked ? null : selectedItem.mesh;
  const pick = tool.pick;

  useEffect(() => {
    if (pick !== null && (pick.mesh !== mesh || otherToolArmed)) {
      app.setPlacementTool({ pick: null });
    }
  }, [app, pick, mesh, otherToolArmed]);

  // A report is about the mesh it ran on.
  const reportedOn = useRef(mesh);
  useEffect(() => {
    if (reportedOn.current !== mesh) {
      reportedOn.current = mesh;
      if (app.getSnapshot().placementTool.report !== null) {
        app.setPlacementTool({ report: null });
      }
    }
  }, [app, mesh]);

  const report = (text: string, error = false) => app.setPlacementTool({ report: { text, error } });

  /** Moves the mesh's box centre or bottom centre to the origin (one undo step). */
  const toOrigin = (anchor: OriginAnchor) => {
    if (mesh === null) return;
    app.setPlacementTool({ pick: null });
    try {
      const before = store.scene.placement(mesh).translation;
      const next = placeMesh(store, mesh, (placed) => moveToOrigin(placed, anchor, up));
      if (next === null) {
        report("Already at the origin");
        return;
      }
      const t = next.translation;
      const moved = Math.hypot(t[0] - before[0], t[1] - before[1], t[2] - before[2]);
      report(`${anchor === "center" ? "Centre" : "Bottom centre"} moved to the origin (${formatLength(moved, units)})`);
    } catch (e) {
      report(e instanceof Error ? e.message : String(e), true);
    }
  };

  /** Arms the surface pick `kind` on the selected mesh, disarming the other pointer tools. */
  const arm = (kind: PlacementPickKind) => {
    if (mesh === null) return;
    const s = app.getSnapshot();
    if (s.sculpt !== null) app.setSculpt(null);
    if (s.maskPaint) app.setMaskPaint(false);
    if (s.inspectTool !== null) app.setInspectTool(null);
    app.disarmWave6Tools();
    app.setPlacementTool({ pick: { kind, mesh, points: [], normal: null, up, target }, report: null });
  };

  const cancel = () => app.setPlacementTool({ pick: null });

  /** Bakes the placement into the vertices (one undo step). */
  const bake = async () => {
    if (mesh === null || busy) return;
    app.setPlacementTool({ pick: null });
    setBusy(true);
    try {
      const done = await applyTransform(store, mesh);
      if (done) {
        onBaked?.();
        report("Transform applied: the placement is now the identity");
      } else {
        report("The placement already is the identity");
      }
    } catch (e) {
      report(e instanceof Error ? e.message : String(e), true);
    } finally {
      setBusy(false);
    }
  };

  return { mesh, pick, report: tool.report, up, setUp, target, setTarget, busy, toOrigin, arm, cancel, bake };
}

/** What `usePlacementHelpers` gives the panel. */
export type PlacementHelpersState = ReturnType<typeof usePlacementHelpers>;
