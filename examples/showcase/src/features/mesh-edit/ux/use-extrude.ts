// The Extrude section's behaviour (#262). Extrude Faces moves the active
// mesh's face selection (made with the Select tools, kept per mesh) out and
// walls its boundary; the selected faces keep their slots, so they stay
// selected where they went. Extrude Base runs the active mesh's boundary to a
// flat, capped base. Each is one undo step through `engine/workbench-ops.ts`,
// with a one-line report; the faces a result has crossing can be selected.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { runExtrudeBase, runExtrudeFaces } from "../../../engine/workbench-ops";
import type { LengthUnit } from "../../../shared/ux/units";
import { writeMeshFaceSelection } from "../../select/ux/face-selection-ops";
import { EMPTY_U32, sanitizeSelectedTriangles } from "../../select/ux/face-sets";
import { useSceneRevision } from "../../select/ux/use-scene-revision";
import type { SignedAxis } from "../../transform/ux/placement-math";
import {
  type BasePlaneMode,
  DEFAULT_BASE_DIRECTION,
  describeExtrudeBase,
  describeExtrudeFaces,
  extrudeBaseOptions,
  extrudeFacesBlocked,
  extrudeFacesOptions,
  type FaceDirection,
} from "./extrude";
import type { ModelingReport } from "./use-modeling-ops";

/** The active item's selected faces that are live triangles now. */
function liveSelection(app: AppState, store: SceneStore, item: Item): Uint32Array {
  const selected = app.getSnapshot().wave6.select.selectedByMesh[item.id] ?? EMPTY_U32;
  return selected.length === 0
    ? EMPTY_U32
    : sanitizeSelectedTriangles(store.scene.views(item.mesh).indices(), selected);
}

export function useExtrude(args: {
  app: AppState;
  store: SceneStore;
  /** The selected mesh, else the first one. */
  activeItem: Item | null;
  units: LengthUnit;
}) {
  const { app, store, activeItem, units } = args;
  // Faces
  const [distance, setDistance] = useState(1);
  const [faceDirection, setFaceDirection] = useState<FaceDirection>("region");
  // Base
  const [baseDirection, setBaseDirection] = useState<SignedAxis>(DEFAULT_BASE_DIRECTION);
  const [plane, setPlane] = useState<BasePlaneMode>("below");
  const [offset, setOffset] = useState(2);
  const [at, setAt] = useState(0);
  const [allLoops, setAllLoops] = useState(false);
  const [refine, setRefine] = useState(true);

  const [busy, setBusy] = useState<string | null>(null);
  const [report, setReport] = useState<ModelingReport | null>(null);
  /** The crossing faces of the last result, and the item they are on. */
  const [crossing, setCrossing] = useState<{ item: Item; faces: Uint32Array } | null>(null);

  // The live count follows the selection and every edit, undo and redo.
  useApp(app, (s) => (activeItem === null ? undefined : s.wave6.select.selectedByMesh[activeItem.id]));
  useSceneRevision(store);
  const selectedCount = activeItem === null ? 0 : liveSelection(app, store, activeItem).length;

  const run = async (
    label: string,
    item: Item,
    op: () => Promise<{ text: string; crossing: Uint32Array | null }>,
  ) => {
    if (busy !== null) {
      return;
    }
    setBusy(label);
    setReport(null);
    setCrossing(null);
    try {
      const r = await op();
      setReport({ label, text: r.text });
      if (r.crossing !== null && r.crossing.length > 0) {
        setCrossing({ item, faces: r.crossing });
      }
    } catch (e) {
      setReport({ label, text: e instanceof Error ? e.message : String(e), error: true });
    } finally {
      setBusy(null);
    }
  };

  const runFaces = async () => {
    if (activeItem === null) {
      return;
    }
    const item = activeItem;
    await run("Extrude faces", item, async () => {
      const triangles = liveSelection(app, store, item);
      const options = extrudeFacesOptions({ distance, direction: faceDirection });
      const r = await runExtrudeFaces({ store, mesh: item.mesh, triangles, options });
      return { text: describeExtrudeFaces(r, distance, units), crossing: r.selfIntersections };
    });
  };

  const runBase = async () => {
    if (activeItem === null) {
      return;
    }
    const item = activeItem;
    await run("Extrude base", item, async () => {
      const options = extrudeBaseOptions({ direction: baseDirection, plane, offset, at, allLoops, refine });
      const r = await runExtrudeBase({ store, mesh: item.mesh, options });
      return { text: describeExtrudeBase(r, baseDirection, units), crossing: r.selfIntersections };
    });
  };

  /** Replaces the face selection by the crossing faces of the last result. */
  const selectCrossing = () => {
    if (crossing !== null) {
      writeMeshFaceSelection(app, crossing.item.id, crossing.faces);
      setCrossing(null);
    }
  };

  return {
    distance,
    setDistance,
    faceDirection,
    setFaceDirection,
    selectedCount,
    facesBlocked: activeItem === null || extrudeFacesBlocked(selectedCount, distance),
    baseDirection,
    setBaseDirection,
    plane,
    setPlane,
    offset,
    setOffset,
    at,
    setAt,
    allLoops,
    setAllLoops,
    refine,
    setRefine,
    busy,
    report,
    crossingCount: crossing?.faces.length ?? 0,
    selectCrossing,
    runFaces,
    runBase,
  };
}
