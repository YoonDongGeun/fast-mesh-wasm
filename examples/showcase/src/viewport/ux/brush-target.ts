// Which mesh a brush (the sculpt stroke or the mask paint) lands on (#229):
// any visible, unlocked mesh under the pointer, or only the active one while
// "Pin to active mesh" is on; a stroke stays on the mesh it pressed. Alt+Click
// pins or unpins the hovered mesh, and a hover retarget carries the brush's
// radius over in proportion.
import type { MeshHandle } from "fast-mesh-wasm";
import { meshState, type SceneSnapshot } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../app/state/app-state";
import { selectMesh } from "../../features/select/ux/selection";

/** Scales a brush radius proportionally when hover-retargeting from `fromItem` to `toItem` (#229). */
export function scaleBrushRadiusForTarget(
  currentRadius: number,
  fromItem: Pick<Item, "radius"> | undefined,
  toItem: Pick<Item, "radius"> | undefined,
): number {
  if (fromItem === undefined || toItem === undefined || !(fromItem.radius > 0) || !(toItem.radius > 0)) {
    return currentRadius;
  }
  const scaled = (currentRadius / fromItem.radius) * toItem.radius;
  return Math.max(0.01, Math.min(toItem.radius, scaled));
}

/**
 * Evaluates whether `hitMesh` may be sculpted/painted given the active stroke lock,
 * `"Pin to active mesh"` mode, `selection`, and per-item lock/visibility (#229).
 */
export function shouldAllowStrokeHit(params: {
  hitMesh: MeshHandle;
  strokeTargetMesh: MeshHandle | null;
  pinToActive: boolean;
  selection: MeshHandle | null;
  fallbackMesh: MeshHandle | null;
  items: readonly Item[];
  storeSnapshot: SceneSnapshot;
}): boolean {
  const { hitMesh, strokeTargetMesh, pinToActive, selection, fallbackMesh, items, storeSnapshot } = params;
  if (strokeTargetMesh !== null) {
    return hitMesh === strokeTargetMesh;
  }
  const item = items.find((i) => i.mesh === hitMesh);
  if (item === undefined || item.locked || !meshState(storeSnapshot, hitMesh).desired.visible) {
    return false;
  }
  if (!pinToActive) {
    return true;
  }
  const pinnedMesh = selection ?? fallbackMesh;
  return pinnedMesh !== null && hitMesh === pinnedMesh;
}

/** Handles Alt+Click quick pin/unpin on a hovered eligible mesh (#229). */
export function applyAltPinClick(params: {
  app: AppState;
  select: (mesh: MeshHandle | null) => void;
  selection: MeshHandle | null;
  hitMesh: MeshHandle;
  pinToActive: boolean;
}): boolean {
  const { app, select, selection, hitMesh, pinToActive } = params;
  if (pinToActive && selection === hitMesh) {
    app.setSculptPinToActive(false);
    return false;
  }
  selectMesh(app, select, selection, hitMesh, false);
  app.setSculptPinToActive(true);
  return true;
}
