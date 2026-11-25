// The sculpt brush's viewport side, shown only while a brush or the mask paint
// is armed (#285: part of the sculpt chunk): the brush cursors, which the
// stroke and paint drivers place, and the mask's tint on its mesh.
import { useFrame } from "@react-three/fiber";
import type { Scene } from "fast-mesh-wasm";
import type { MeshBinding } from "fast-mesh-wasm/three";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import type * as THREE from "three";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { inspect } from "../../../app/state/readouts";
import { useMaskPaint } from "../../../viewport/ux/use-mask-paint";
import { useSculptStroke } from "../../../viewport/ux/use-sculpt-stroke";
import type { MaskLiveTracker } from "../ux/mask-protect";
import type { SculptBrush } from "../ux/sculpt";
import { makeBrushCursor } from "./brush-cursor";
import { MaskOverlay } from "./mask-overlay";

/**
 * The sculpt mask's tint on its mesh (see `viewport/mask-overlay.ts`), while
 * a sculpt tool or the mask paint is armed and it is not hidden. A child of
 * the bound mesh, so it hides with it; the frame loop follows the mask and
 * the mesh's triangles (a remesh, an undo), at most one build per frame.
 */
export function SculptMaskOverlay(props: { app: AppState; item: Item; bound: MeshBinding | null }) {
  const { app, item, bound } = props;
  const masked = useApp(app, (s) => (s.sculptMask?.mesh === item.mesh ? s.sculptMask.vertices : null));
  const armed = useApp(app, (s) => s.sculpt !== null || s.maskPaint);
  const shown = useApp(app, (s) => s.sculptMaskShown);
  const on = armed && shown && masked !== null;
  const object = bound?.object ?? null;
  const overlayRef = useRef<MaskOverlay | null>(null);
  useLayoutEffect(() => {
    if (!on || object === null) {
      return;
    }
    const created = new MaskOverlay(object);
    overlayRef.current = created;
    return () => {
      overlayRef.current = null;
      created.dispose();
      inspect.sculpt.maskOverlay.triangles = 0;
    };
  }, [on, object]);
  useFrame(() => {
    const overlay = overlayRef.current;
    if (overlay === null || masked === null) {
      return;
    }
    const start = performance.now();
    if (overlay.sync(masked)) {
      const stats = inspect.sculpt.maskOverlay;
      stats.lastMs = performance.now() - start;
      stats.maxMs = Math.max(stats.maxMs, stats.lastMs);
      stats.builds++;
      stats.triangles = overlay.triangles;
    }
  });
  return null;
}

/**
 * The sculpt brush on its mesh: the brush cursor (amber), which the stroke
 * driver (`useSculptStroke`) lays on the surface under the pointer while it
 * paints the stroke's dabs.
 */
export function SculptLayer(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  sculpt: SculptBrush;
  remesh: boolean;
  liveRemesh: boolean;
  maskTracker: MaskLiveTracker;
}) {
  const { app, scene, sculpt, remesh, liveRemesh, maskTracker } = props;
  const ball = useMemo(() => makeBrushCursor(0xffd166), []);
  useEffect(() => {
    return () => {
      ball.removeFromParent();
      for (const child of ball.children) {
        const part = child as THREE.Mesh | THREE.Line;
        part.geometry.dispose();
        (part.material as THREE.Material).dispose();
      }
    };
  }, [ball]);
  useSculptStroke({ app, scene, sculpt, remesh, liveRemesh, maskTracker, ball });
  return null;
}

/**
 * The mask paint on the selected mesh: the paint ball (blue, against sculpt's
 * amber), which `useMaskPaint` lays on the surface under the pointer while it
 * paints the mask.
 */
export function MaskLayer(props: { app: AppState; scene: Scene | WorkerScene; maskTracker: MaskLiveTracker }) {
  const { app, scene, maskTracker } = props;
  const ball = useMemo(() => makeBrushCursor(0x6ea8d9), []);
  useEffect(() => {
    return () => {
      ball.removeFromParent();
      for (const child of ball.children) {
        const part = child as THREE.Mesh | THREE.Line;
        part.geometry.dispose();
        (part.material as THREE.Material).dispose();
      }
    };
  }, [ball]);
  useMaskPaint({ app, scene, maskTracker, ball });
  return null;
}
