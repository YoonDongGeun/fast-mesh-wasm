// What the viewport shares outside React: the camera, the canvas and the mesh
// bindings, for the test hook (scripts/showcase-check.mjs projects a mesh
// to the screen to click it and drag its gizmo).
import type { MeshHandle } from "fast-mesh-wasm";
import type { MeshBinding } from "fast-mesh-wasm/three";
import type * as THREE from "three";
import type { TransformControls } from "three/addons/controls/TransformControls.js";

export const view: {
  camera: THREE.Camera | null;
  canvas: HTMLCanvasElement | null;
  bindings: ReadonlyMap<MeshHandle, MeshBinding>;
  /** The gizmo on the selected mesh, if any. */
  gizmo: TransformControls | null;
  /** True while the gizmo drags. */
  dragging: boolean;
  /** Called with `dragging` whenever it changes. */
  dragListeners: Set<(dragging: boolean) => void>;
} = { camera: null, canvas: null, bindings: new Map(), gizmo: null, dragging: false, dragListeners: new Set() };

/** Sets `view.dragging` and tells the listeners when it changes. */
export function setDragging(dragging: boolean): void {
  if (view.dragging !== dragging) {
    view.dragging = dragging;
    for (const l of view.dragListeners) {
      l(dragging);
    }
  }
}
