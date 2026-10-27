// What the viewport shares outside React: the camera, the canvas and the mesh
// bindings, for the test hook (scripts/showcase-check.mjs projects a mesh
// to the screen to click it and drag its gizmo), and the vertex under the
// pointer, for the Group mode's hover tip.
import type { MeshHandle } from "fast-mesh-wasm";
import type { MeshBinding } from "fast-mesh-wasm/three";
import type * as THREE from "three";
import type { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { TransformControls } from "three/addons/controls/TransformControls.js";
import type { SceneSliceQuads } from "./SceneSlices";

/** The mesh under the pointer: the vertex slot nearest to the hit, and the pointer on the page (CSS pixels). */
export interface Hover {
  mesh: MeshHandle;
  slot: number;
  x: number;
  y: number;
}

export const view: {
  camera: THREE.Camera | null;
  canvas: HTMLCanvasElement | null;
  bindings: ReadonlyMap<MeshHandle, MeshBinding>;
  /** The gizmo on the selected mesh, if any. */
  gizmo: TransformControls | null;
  /** The orbit camera controls (a sculpt stroke disables them while it paints). */
  orbit: OrbitControls | null;
  /** True while the gizmo drags. */
  dragging: boolean;
  /** Called with `dragging` whenever it changes. */
  dragListeners: Set<(dragging: boolean) => void>;
  /** The vertex under the pointer while hovering is on (Group mode), else `null`. */
  hover: Hover | null;
  /** Called whenever `hover` changes. */
  hoverListeners: Set<() => void>;
  /** The scene's slice quads, for the test hook to project their handles. */
  sliceQuads: SceneSliceQuads | null;
} = {
  camera: null,
  canvas: null,
  bindings: new Map(),
  gizmo: null,
  orbit: null,
  dragging: false,
  dragListeners: new Set(),
  hover: null,
  hoverListeners: new Set(),
  sliceQuads: null,
};

/** Sets `view.dragging` and tells the listeners when it changes. */
export function setDragging(dragging: boolean): void {
  if (view.dragging !== dragging) {
    view.dragging = dragging;
    for (const l of view.dragListeners) {
      l(dragging);
    }
  }
}

/** Sets `view.hover` and tells the listeners when it changes. */
export function setHover(hover: Hover | null): void {
  const h = view.hover;
  const same =
    h === hover ||
    (h !== null && hover !== null && h.mesh === hover.mesh && h.slot === hover.slot && h.x === hover.x && h.y === hover.y);
  if (same) {
    return;
  }
  view.hover = hover;
  for (const l of view.hoverListeners) {
    l();
  }
}
