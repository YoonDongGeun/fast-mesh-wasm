// The part of a scene the renderer adapters (`fast-mesh-wasm/three`, ...) depend
// on. `Scene` implements it; so can any other host that mirrors a scene
// (for example one running in a Worker). Handles stay opaque: adapters only
// pass them back to these methods.
import type { Placement } from "./api.js";
import type { MeshViews } from "./views.js";
import type {
  DragSession,
  MeshChange,
  MeshHandle,
  SectionHandle,
  SectionPlane,
  SectionResult,
} from "./scene.js";

/**
 * What the adapters need from a scene: render buffers, placements, section
 * results, change notification and drags. All methods are synchronous.
 */
export interface SceneLike {
  /**
   * Views of the mesh's render buffers. Slot counts follow from them:
   * `positions().length / 3` vertex slots and `indices().length / 3`
   * triangle slots.
   */
  views(mesh: MeshHandle): MeshViews;
  /** The column-major world matrix of the mesh's local render coordinates. */
  matrix4(mesh: MeshHandle): Float64Array;
  placement(mesh: MeshHandle): Placement;
  section(s: SectionHandle, mesh: MeshHandle): SectionResult;
  sectionPlane(s: SectionHandle): SectionPlane;
  /** Registers a change listener; returns the function that removes it. */
  onChanges(listener: (c: MeshChange[]) => void): () => void;
  /** Flushes pending changes to the listeners and returns them. */
  commitFrame(): MeshChange[];
  drag(mesh: MeshHandle): DragSession;
}
