// The transform panel's fields (#195): position and Euler rotation read
// from the selected mesh's placement, and its scale read against the local
// extents it first had, all kept in sync with scene edits and undo/redo.
// Position and rotation edits set the placement; a scale applies the ratio
// to the current scale as one undoable mesh edit.
import type { MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useRef, useState } from "react";
import type { Item } from "../../../app/state/app-state";
import {
  applyMeshScale,
  computeLocalExtents,
  placementToValues,
  valuesToPlacement,
} from "./transform-math";

/** The transform fields' state and edits for `selectedItem` (all zero/one without a selection). */
export function useTransformFields(store: SceneStore, selectedItem: Item | null) {
  const [pos, setPos] = useState<[number, number, number]>([0, 0, 0]);
  const [rotDeg, setRotDeg] = useState<[number, number, number]>([0, 0, 0]);
  const [scaleVal, setScaleVal] = useState<[number, number, number]>([1, 1, 1]);
  const [uniformScale, setUniformScale] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Baseline extents per mesh handle so displayed scale stays consistent across undo/redo.
  const baselineExtents = useRef(new Map<MeshHandle, [number, number, number]>());

  useEffect(() => {
    if (selectedItem === null) {
      setPos([0, 0, 0]);
      setRotDeg([0, 0, 0]);
      setScaleVal([1, 1, 1]);
      return;
    }
    const mesh = selectedItem.mesh;
    const syncFromScene = () => {
      try {
        const p = store.scene.placement(mesh);
        const vals = placementToValues(p);
        setPos(vals.position);
        setRotDeg([
          Number(vals.rotationDeg[0].toFixed(3)),
          Number(vals.rotationDeg[1].toFixed(3)),
          Number(vals.rotationDeg[2].toFixed(3)),
        ]);

        const v = store.scene.views(mesh);
        const { extents } = computeLocalExtents(v.positions(), v.indices());
        let base = baselineExtents.current.get(mesh);
        if (base === undefined) {
          base = [extents[0] || 1, extents[1] || 1, extents[2] || 1];
          baselineExtents.current.set(mesh, base);
        }
        setScaleVal([
          Number((base[0] > 0 ? extents[0] / base[0] : 1).toFixed(4)),
          Number((base[1] > 0 ? extents[1] / base[1] : 1).toFixed(4)),
          Number((base[2] > 0 ? extents[2] / base[2] : 1).toFixed(4)),
        ]);
      } catch {
        // Mesh may have been removed mid-frame
      }
    };
    syncFromScene();
    const unsubChanges = store.scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === mesh)) {
        syncFromScene();
      }
    });
    const unsubStore = store.subscribe(syncFromScene);
    return () => {
      unsubChanges();
      unsubStore();
    };
  }, [selectedItem, store]);

  const commitPlacement = (nextPos: [number, number, number], nextRotDeg: [number, number, number]) => {
    if (selectedItem === null) return;
    setError(null);
    store.setPlacement(selectedItem.mesh, valuesToPlacement({ position: nextPos, rotationDeg: nextRotDeg }));
    setPos(nextPos);
    setRotDeg(nextRotDeg);
  };

  const commitScale = async (targetScale: [number, number, number]) => {
    if (selectedItem === null) return;
    if (targetScale[0] <= 0 || targetScale[1] <= 0 || targetScale[2] <= 0) {
      setError("Scale factors must be positive");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const mesh = selectedItem.mesh;
      const v = store.scene.views(mesh);
      const { extents } = computeLocalExtents(v.positions(), v.indices());
      const base = baselineExtents.current.get(mesh) ?? [extents[0] || 1, extents[1] || 1, extents[2] || 1];
      baselineExtents.current.set(mesh, base);

      const cur: [number, number, number] = [
        base[0] > 0 ? extents[0] / base[0] : 1,
        base[1] > 0 ? extents[1] / base[1] : 1,
        base[2] > 0 ? extents[2] / base[2] : 1,
      ];
      const delta: [number, number, number] = [
        cur[0] > 0 ? targetScale[0] / cur[0] : targetScale[0],
        cur[1] > 0 ? targetScale[1] / cur[1] : targetScale[1],
        cur[2] > 0 ? targetScale[2] / cur[2] : targetScale[2],
      ];
      await applyMeshScale(store, mesh, delta);
      setScaleVal(targetScale);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  /** Moves the mesh to `mm` along one axis. */
  const setPosAxis = (idx: number, mm: number) => {
    const next: [number, number, number] = [...pos];
    next[idx] = mm;
    commitPlacement(next, rotDeg);
  };

  /** Sets one Euler angle (degrees) from a typed number, ignoring non-finite input. */
  const setRotAxis = (idx: number, val: number) => {
    if (!Number.isFinite(val)) return;
    const next: [number, number, number] = [...rotDeg];
    next[idx] = val;
    commitPlacement(pos, next);
  };

  /** Takes the current extents as 1× (Apply transform baked the placement's turn and scale into them). */
  const rebaseScale = () => {
    if (selectedItem === null) return;
    try {
      const v = store.scene.views(selectedItem.mesh);
      const { extents } = computeLocalExtents(v.positions(), v.indices());
      baselineExtents.current.set(selectedItem.mesh, [extents[0] || 1, extents[1] || 1, extents[2] || 1]);
      setScaleVal([1, 1, 1]);
    } catch {
      // Removed meanwhile
    }
  };

  /** Edits the pending scale (all three axes while uniform), ignoring non-finite input; Apply commits it. */
  const editScale = (idx: number, val: number) => {
    if (!Number.isFinite(val)) return;
    if (uniformScale) {
      setScaleVal([val, val, val]);
    } else {
      const next: [number, number, number] = [...scaleVal];
      next[idx] = val;
      setScaleVal(next);
    }
  };

  return {
    pos,
    rotDeg,
    scaleVal,
    uniformScale,
    setUniformScale,
    busy,
    error,
    setPosAxis,
    setRotAxis,
    editScale,
    commitScale,
    rebaseScale,
  };
}
