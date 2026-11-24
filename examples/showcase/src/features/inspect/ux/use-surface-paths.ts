// The geodesic and path picks' panel behaviour (#252): arming a pick on the
// selected mesh (which disarms the other pointer tools), the method, undo
// and clear, Close loop, and Trim and Split along a closed loop. The picks
// belong to their mesh: moving the selection to another mesh clears them,
// and so does any change of the mesh's shape or placement (an edit, a trim,
// an undo or redo, a move) or its removal. Another pointer tool arming
// stands the pick down and keeps the picks.
import type { GeodesicMethod, MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { describeKept, keepPath } from "../../polylines/ux/polyline-actions";
import {
  cancelSurfaceLeg,
  closeSurfaceLoop,
  settleChanges,
  splitAlongPath,
  type TrimKeep,
  trimAlongPath,
} from "./surface-path-actions";
import {
  canClose,
  clearedPicks,
  surfacePicksChanged,
  type SurfacePickMode,
  undoPathPoint,
} from "./surface-paths";

export function useSurfacePaths(app: AppState, items: readonly Item[], selection: MeshHandle | null) {
  const store = useScene();
  const tool = useApp(app, (s) => s.surfacePath);
  const otherToolArmed = useApp(
    app,
    (s) =>
      s.sculpt !== null ||
      s.maskPaint ||
      s.inspectTool !== null ||
      s.wave6.select.activeTool !== null ||
      s.wave6.deform.activeTool !== null ||
      s.wave6.printPrep.activeTool !== null ||
      s.placementTool.pick !== null ||
      (s.brushPick !== null && s.brushPick.point === null),
  );
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  /** A cut is running: its changes are its own (a failed one takes them back), so they clear no picks. */
  const cutting = useRef(false);

  const item = items.find((i) => i.mesh === selection) ?? null;
  const target = item?.mesh ?? null;
  /** The picks shown in the panel: only the selected mesh's. */
  const onTarget = tool.mesh !== null && tool.mesh === target;

  // The picks belong to their mesh: a selection elsewhere (or none) clears them.
  useEffect(() => {
    if (tool.mesh !== null && tool.mesh !== target) {
      app.setSurfacePath(clearedPicks(app.getSnapshot().surfacePath));
    }
  }, [app, tool.mesh, target]);

  // Another pointer tool arming stands the pick down.
  useEffect(() => {
    if (tool.armed !== null && otherToolArmed) {
      app.setSurfacePath({ armed: null });
    }
  }, [app, tool.armed, otherToolArmed]);

  // A change of the mesh's shape or placement, or its removal, clears the picks.
  useEffect(() => {
    const mesh = tool.mesh;
    if (mesh === null) {
      return;
    }
    return store.scene.onChanges((changes) => {
      const s = app.getSnapshot().surfacePath;
      if (cutting.current || s.mesh !== mesh || !surfacePicksChanged(changes, mesh)) {
        return;
      }
      const removed = changes.some((c) => c.mesh === mesh && c.removed);
      if (removed) {
        app.setSurfacePath(clearedPicks(s));
      } else if (s.sources.length > 0 || s.path !== null || s.error !== null) {
        app.setSurfacePath({ sources: [], path: null, error: null, isoLines: null });
      }
    });
  }, [app, store, tool.mesh]);

  /** Arms `mode` on the selected mesh (pressing the armed one disarms it), disarming the other pointer tools. */
  const arm = (mode: SurfacePickMode) => {
    if (target === null) {
      return;
    }
    const s = app.getSnapshot();
    if (s.surfacePath.armed === mode && s.surfacePath.mesh === target) {
      app.setSurfacePath({ armed: null });
      return;
    }
    if (s.sculpt !== null) app.setSculpt(null);
    if (s.maskPaint) app.setMaskPaint(false);
    if (s.inspectTool !== null) app.setInspectTool(null);
    if (s.placementTool.pick !== null) app.setPlacementTool({ pick: null });
    if (s.brushPick !== null && s.brushPick.point === null) app.setBrushPick(null);
    app.disarmWave6Tools();
    const base = s.surfacePath.mesh === target ? s.surfacePath : clearedPicks(s.surfacePath);
    app.setSurfacePath({ ...base, mesh: target, armed: mode, error: null });
    setStatus(null);
  };

  const setMethod = (method: GeodesicMethod) => app.setSurfacePath({ method });

  const undoSource = () => {
    const s = app.getSnapshot().surfacePath;
    if (s.sources.length > 0) {
      app.setSurfacePath({ sources: s.sources.slice(0, -1), error: null });
    }
  };

  const clearSources = () => app.setSurfacePath({ sources: [], error: null });

  /** Removes the latest path step: a closed loop reopens, else the last point goes. */
  const undoPoint = () => {
    const s = app.getSnapshot().surfacePath;
    if (s.path !== null) {
      cancelSurfaceLeg(app);
      app.setSurfacePath({ path: undoPathPoint(s.path), error: null });
    }
  };

  const clearPath = () => {
    cancelSurfaceLeg(app);
    app.setSurfacePath({ path: null, error: null });
    setStatus(null);
  };

  /** Traces the leg from the last point back to the first (queued behind any tap still landing). */
  const close = () => {
    void closeSurfaceLoop(app, store.scene);
  };

  /**
   * Runs a cut on the selected mesh's closed loop. A cut that lands used the
   * picks up: they clear. One that fails left the mesh as it was: they stay,
   * with the reason.
   */
  const cut = async (work: () => Promise<string>) => {
    if (cutting.current || item === null) {
      return;
    }
    const mesh = item.mesh;
    cutting.current = true;
    setBusy(true);
    setStatus(null);
    app.setSurfacePath({ armed: null, error: null });
    try {
      const done = await work();
      const s = app.getSnapshot().surfacePath;
      if (s.mesh === mesh) {
        app.setSurfacePath({ sources: [], path: null, error: null, isoLines: null });
      }
      setStatus(done);
    } catch (e) {
      const text = e instanceof Error ? e.message : String(e);
      await settleChanges(store, mesh); // the changes the failed cut took back go by before the picks react again
      app.setSurfacePath({ error: text });
    } finally {
      cutting.current = false;
      setBusy(false);
    }
  };

  /** Deletes one side of the loop (one undo step). */
  const trim = (keep: TrimKeep) => {
    if (item === null) {
      return Promise.resolve();
    }
    const s = app.getSnapshot().surfacePath;
    const path = s.mesh === item.mesh ? s.path : null;
    return cut(async () => {
      const r = await trimAlongPath(store, item.mesh, path, keep);
      return `Trimmed along the loop: kept the ${r.keep} (${r.triangles} triangles).`;
    });
  };

  /** Splits the mesh along the loop into two meshes (one undo step). */
  const split = () => {
    if (item === null) {
      return Promise.resolve();
    }
    const s = app.getSnapshot().surfacePath;
    const path = s.mesh === item.mesh ? s.path : null;
    return cut(async () => {
      const r = await splitAlongPath(store, app, item, path);
      return `Split along the loop: "${item.name}" keeps the outside (${r.outsideTriangles} triangles) and "${r.inside.name}" holds the inside (${r.insideTriangles}).`;
    });
  };

  /** Keeps the path as a polyline object (#268): one undo step. The path stays. */
  const keep = () => {
    const s = app.getSnapshot().surfacePath;
    if (item === null || s.mesh !== item.mesh) {
      return;
    }
    try {
      setStatus(describeKept(keepPath(app, store, s.path, item.name)));
      app.setSurfacePath({ error: null });
    } catch (e) {
      app.setSurfacePath({ error: e instanceof Error ? e.message : String(e) });
    }
  };

  const path = onTarget ? tool.path : null;
  const sources = onTarget ? tool.sources : [];
  const locked = item?.locked ?? false;
  const closed = path?.closed ?? false;

  return {
    target,
    item,
    armed: onTarget ? tool.armed : null,
    method: tool.method,
    sources,
    path,
    tracing: tool.tracing > 0,
    /** Cancels the leg being traced. */
    cancelTracing: () => cancelSurfaceLeg(app),
    error: onTarget || tool.mesh === null ? tool.error : null,
    busy,
    status,
    locked,
    canClose: canClose(path),
    canCut: closed && !busy && !locked && tool.tracing === 0,
    arm,
    setMethod,
    undoSource,
    clearSources,
    undoPoint,
    clearPath,
    close,
    trim,
    split,
    keep,
  };
}
