// What the mask paint keeps while no paint is armed: the live-mask cache it
// shares with the sculpt stroke (it follows every edit from the start), and
// when the paint disarms.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useMemo, useRef } from "react";
import type { AppState, Item } from "../../app/state/app-state";
import { MaskLiveTracker } from "../../features/sculpt/ux/mask-protect";

/**
 * The live-mask cache shared between the mask paint and the sculpt stroke, so
 * switching from mask paint into sculpting preserves O(1) verticesInSphere
 * trust on dab 1. Attached while mounted.
 */
export function useMaskTracker(scene: Scene | WorkerScene, store: SceneStore): MaskLiveTracker {
  const maskTracker = useMemo(() => new MaskLiveTracker(scene, store), [scene, store]);
  useEffect(() => {
    maskTracker.attach();
    return () => maskTracker.dispose();
  }, [maskTracker]);
  return maskTracker;
}

/**
 * Disarms mask paint if the mesh it was armed on was removed or no visible
 * unlocked meshes remain (#229). The armed mesh is the selection at arming,
 * then follows the selection while it is not empty.
 */
export function useMaskPaintDisarm(params: {
  app: AppState;
  maskPaint: boolean;
  selection: MeshHandle | null;
  items: readonly Item[];
  hasEligibleMesh: boolean;
}): void {
  const { app, maskPaint, selection, items, hasEligibleMesh } = params;
  const maskArmedOnRef = useRef<MeshHandle | null>(null);
  const wasMaskPaintRef = useRef(false);
  useEffect(() => {
    if (!maskPaint) {
      wasMaskPaintRef.current = false;
      maskArmedOnRef.current = null;
      return;
    }
    if (!wasMaskPaintRef.current) {
      wasMaskPaintRef.current = true;
      maskArmedOnRef.current = selection;
    } else if (selection !== null) {
      maskArmedOnRef.current = selection;
    }
    const armedMesh = maskArmedOnRef.current;
    if (!hasEligibleMesh || (armedMesh !== null && !items.some((i) => i.mesh === armedMesh))) {
      app.setMaskPaint(false);
    }
  }, [app, maskPaint, selection, items, hasEligibleMesh]);
}
