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
} = { camera: null, canvas: null, bindings: new Map(), gizmo: null, dragging: false };
