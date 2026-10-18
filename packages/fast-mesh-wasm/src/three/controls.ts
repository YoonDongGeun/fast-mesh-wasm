// `attachTransformControls`: maps a TransformControls gizmo dragging a bound
// mesh's object to `scene.drag(mesh)`, one undo step per drag, with at most
// one `move` per animation frame.
import type * as THREE from "three";
import type { Quat, Vec3 } from "../api.js";
import type { DragSession, Scene } from "../scene.js";
import type { BoundMesh } from "./mesh.js";

/**
 * What {@link attachTransformControls} needs of a TransformControls: its
 * `dragging-changed` (`{ value: boolean }`) and `objectChange` events, the
 * object it is attached to and whether it is dragging. Three's
 * `TransformControls` (`three/addons/controls/TransformControls.js`) fits.
 */
export interface TransformControlsLike extends THREE.EventDispatcher<any> {
  object?: THREE.Object3D;
  dragging: boolean;
}

/** Options of {@link attachTransformControls}. */
export interface AttachTransformControlsOptions {
  /**
   * Schedules a callback for the next animation frame. Defaults to
   * `requestAnimationFrame` (or a 16 ms timeout where there is none). Tests
   * can collect the callbacks and run them by hand.
   */
  schedule?: (callback: () => void) => void;
}

/** A pose the object was moved to, waiting for the next frame. */
interface Pose {
  rotation: Quat;
  position: Vec3;
}

const defaultSchedule = (callback: () => void) => {
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() => callback());
  } else {
    setTimeout(callback, 16);
  }
};

/** `R · v` for the rotation of a unit quaternion `[x, y, z, w]`, in f64. */
function rotate(q: Quat, v: Readonly<Vec3>): Vec3 {
  const [x, y, z, w] = q;
  // v + 2w (u × v) + 2 u × (u × v), with u = (x, y, z).
  const cx = y * v[2] - z * v[1];
  const cy = z * v[0] - x * v[2];
  const cz = x * v[1] - y * v[0];
  return [
    v[0] + 2 * (w * cx + y * cz - z * cy),
    v[1] + 2 * (w * cy + z * cx - x * cz),
    v[2] + 2 * (w * cz + x * cy - y * cx),
  ];
}

/**
 * Lets a TransformControls move a bound mesh: while the gizmo drags
 * `bound.object`, its pose is sent to `scene.drag(bound.mesh)`, and the drag
 * is ended (one undo step) when the gizmo lets go. Returns a function that
 * detaches it (ending a drag in progress first).
 *
 * Pose and placement: `bound.object.matrix` is the scene mesh's
 * `matrix4 = placement · translate(origin)`, so with placement `(R, t)` the
 * object's quaternion is `R` and its position is `R · origin + t`. A pose is
 * therefore sent as `{ rotation: quaternion, translation: position − R · origin }`,
 * all in f64. The origin (fixed for the mesh) comes from the placement and
 * matrix at the start of each drag: `origin = Rᵀ (m − t)`, with `m` the
 * matrix's translation column.
 *
 * Placements are rigid: TransformControls' scale mode has no effect (the
 * object's scale is kept at 1). The object must be a direct child of an
 * untransformed parent (its matrix is a world matrix), as for `bindMesh`.
 *
 * The object's `position`, `quaternion` and `scale` are kept in step with its
 * matrix (TransformControls reads them when a drag starts), so call this
 * after `bindMesh` and before the gizmo grabs the object. While the gizmo
 * drags, the object's matrix follows the gizmo, not the change stream (the
 * scene lags by up to a frame); once the drag ends, the final placement is
 * applied. When the drag is ended by the scene instead (`undo()` while
 * dragging, for example), the object follows the scene again and the rest
 * of the gesture is ignored.
 */
export function attachTransformControls(
  controls: TransformControlsLike,
  scene: Scene,
  bound: BoundMesh,
  options: AttachTransformControlsOptions = {},
): () => void {
  const schedule = options.schedule ?? defaultSchedule;
  const { object, mesh } = bound;
  let drag: DragSession | undefined;
  /** True from a drag's start until the gizmo lets go, even if the scene ended the drag. */
  let grabbed = false;
  /** The mesh origin during a drag: `Rᵀ (m − t)`. */
  let origin: Vec3 = [0, 0, 0];
  let pending: Pose | undefined;
  let scheduled = false;
  let attached = true;

  /** Puts the object where the scene has it and its pose in step. */
  const syncFromScene = () => {
    object.matrix.fromArray(scene.matrix4(mesh));
    object.matrix.decompose(object.position, object.quaternion, object.scale);
    object.matrixWorldNeedsUpdate = true;
  };

  /** The drag, if it is still ours and live. */
  const live = (): DragSession | undefined => {
    if (drag !== undefined && !drag.active) {
      // The scene ended it (undo, redo, a step call): stop following the gizmo.
      drag = undefined;
      pending = undefined;
    }
    return drag;
  };

  const flush = () => {
    const d = live();
    const pose = pending;
    pending = undefined;
    if (d === undefined || pose === undefined) {
      return;
    }
    const r = rotate(pose.rotation, origin);
    d.move({
      rotation: pose.rotation,
      translation: [pose.position[0] - r[0], pose.position[1] - r[1], pose.position[2] - r[2]],
    });
  };

  const onFrame = () => {
    scheduled = false;
    if (attached) {
      flush();
    }
  };

  const begin = () => {
    if (grabbed || controls.object !== object) {
      return;
    }
    const p = scene.placement(mesh);
    const m = scene.matrix4(mesh);
    const r = p.rotation;
    const d: Vec3 = [m[12] - p.translation[0], m[13] - p.translation[1], m[14] - p.translation[2]];
    // Rᵀ d, with R row-major.
    origin = [
      r[0] * d[0] + r[3] * d[1] + r[6] * d[2],
      r[1] * d[0] + r[4] * d[1] + r[7] * d[2],
      r[2] * d[0] + r[5] * d[1] + r[8] * d[2],
    ];
    drag = scene.drag(mesh);
    grabbed = true;
  };

  const finish = () => {
    if (!grabbed) {
      return;
    }
    grabbed = false;
    flush();
    const d = live();
    drag = undefined;
    d?.end();
    syncFromScene();
  };

  const onDraggingChanged = (event: { value?: unknown }) => {
    const dragging = typeof event.value === "boolean" ? event.value : controls.dragging;
    if (dragging) {
      begin();
    } else {
      finish();
    }
  };

  const onObjectChange = () => {
    if (live() === undefined || controls.object !== object) {
      return;
    }
    // Rigid placements only: undo scale mode.
    object.scale.set(1, 1, 1);
    // The object shows the gizmo's pose right away (matrixAutoUpdate is off).
    object.updateMatrix();
    const q = object.quaternion;
    const p = object.position;
    pending = { rotation: [q.x, q.y, q.z, q.w], position: [p.x, p.y, p.z] };
    if (!scheduled) {
      scheduled = true;
      schedule(onFrame);
    }
  };

  // Listeners run in registration order, so this one runs after bindMesh's,
  // which has just written the matrix from the scene.
  const unsubscribe = scene.onChanges((changes) => {
    for (const c of changes) {
      if (c.mesh !== mesh || c.removed || !c.placementChanged) {
        continue;
      }
      if (live() !== undefined) {
        // Mid-drag the gizmo leads: put back its pose.
        object.updateMatrix();
      } else {
        object.matrix.decompose(object.position, object.quaternion, object.scale);
      }
    }
  });

  controls.addEventListener("dragging-changed", onDraggingChanged);
  controls.addEventListener("objectChange", onObjectChange);
  syncFromScene();

  return () => {
    if (!attached) {
      return;
    }
    finish();
    attached = false;
    pending = undefined;
    controls.removeEventListener("dragging-changed", onDraggingChanged);
    controls.removeEventListener("objectChange", onObjectChange);
    unsubscribe();
  };
}
