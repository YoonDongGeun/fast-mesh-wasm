// The scene overlay: the linked MPR slices as quads cutting through the
// selected mesh in the main scene, with the mesh clipped away above each
// slice (a cutaway), like a medical viewer. The quads ride the mesh's
// placement, so gizmo moves carry them along.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Loader } from "../load/loader";
import type { VolumePayload } from "../load/protocol";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { inspect } from "../panels/inspect";
import { dataRange } from "../panels/volume-lut";
import { displayValues } from "../panels/volume-data";
import { WINDOW_PRESETS } from "../panels/volume-presets";
import {
  AXIS_COLOR,
  planeSpec,
  slicePlaneImages,
  type SliceAxis,
  type SlicePlane,
  type SlicePlanes,
} from "../panels/volume-slice";
import { clampSlice, dragPlaneNormal, planeHit, sliceSteps } from "./slice-drag";
import { view } from "./view";

const AXES: SliceAxis[] = ["axial", "coronal", "sagittal"];

type SceneVolume = {
  origin: readonly [number, number, number];
  spacing: readonly [number, number, number];
  columns: number;
  rows: number;
  slices: number;
  /** The mesh origin: quads sit in mesh-local coordinates. */
  center: readonly [number, number, number];
};

/** One quad: its mesh, its slice texture, and its corner drag handles. */
type QuadEntry = {
  mesh: THREE.Mesh;
  texture: THREE.DataTexture;
  handles: THREE.Mesh[];
  handleGeometry: THREE.BufferGeometry;
  handleMaterial: THREE.Material;
};

/** The three quads plus their cutaway planes, parented to one matrix. */
export class SceneSliceQuads {
  readonly group = new THREE.Group();
  /** The cutaway planes in world coordinates: assign to the clipped material. */
  readonly clip: [THREE.Plane, THREE.Plane, THREE.Plane] = [
    new THREE.Plane(new THREE.Vector3(0, 0, -1), 0),
    new THREE.Plane(new THREE.Vector3(0, -1, 0), 0),
    new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0),
  ];
  private local: [THREE.Plane, THREE.Plane, THREE.Plane] = [
    new THREE.Plane(new THREE.Vector3(0, 0, -1), 0),
    new THREE.Plane(new THREE.Vector3(0, -1, 0), 0),
    new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0),
  ];
  private entries: (QuadEntry | undefined)[] = [];
  private matrix = new THREE.Matrix4();

  constructor() {
    this.group.matrixAutoUpdate = false;
    this.group.visible = false;
  }

  /** The mesh's placement: the quads and the cutaway ride it. */
  setMatrix(m: ArrayLike<number>): void {
    this.matrix.fromArray(m);
    this.group.matrix.copy(this.matrix);
    this.group.matrixWorldNeedsUpdate = true;
    for (let i = 0; i < 3; i++) {
      this.clip[i].copy(this.local[i]).applyMatrix4(this.matrix);
    }
  }

  /** The slices, or null to hide them. */
  setSlices(images: SlicePlanes | null, volume: SceneVolume): void {
    if (images === null) {
      this.group.visible = false;
      return;
    }
    const [ox, oy, oz] = volume.origin;
    const [sx, sy, sz] = volume.spacing;
    const [cx, cy, cz] = volume.center;
    const ex = volume.columns * sx;
    const ey = volume.rows * sy;
    const ez = volume.slices * sz;
    const list = [images.axial, images.coronal, images.sagittal];
    // Quad centres in mesh-local coordinates (the cut rides the same spot).
    const at: Array<[number, number, number]> = [
      [ox + ex / 2 - cx, oy + ey / 2 - cy, oz + (images.axial.index + 0.5) * sz - cz],
      [ox + ex / 2 - cx, oy + (images.coronal.index + 0.5) * sy - cy, oz + ez / 2 - cz],
      [ox + (images.sagittal.index + 0.5) * sx - cx, oy + ey / 2 - cy, oz + ez / 2 - cz],
    ];
    list.forEach((plane, i) => {
      let entry = this.entries[i];
      if (
        entry !== undefined &&
        (entry.texture.image.width !== plane.width || entry.texture.image.height !== plane.height)
      ) {
        // Another volume's slices: rebuild the quad at its size.
        this.group.remove(entry.mesh);
        disposeEntry(entry);
        entry = undefined;
        this.entries[i] = undefined;
      }
      if (entry === undefined) {
        entry = this.buildQuad(AXES[i], plane, [ex, ey, ez]);
        this.group.add(entry.mesh);
        this.entries[i] = entry;
      } else {
        entry.texture.image.data = plane.data;
        entry.texture.needsUpdate = true;
      }
      entry.mesh.position.set(at[i][0], at[i][1], at[i][2]);
    });
    this.local[0].constant = at[0][2];
    this.local[1].constant = at[1][1];
    this.local[2].constant = at[2][0];
    this.group.visible = true;
  }

  /** The quad's mesh, once built. */
  quadMesh(axis: SliceAxis): THREE.Mesh | null {
    return this.entries[AXES.indexOf(axis)]?.mesh ?? null;
  }

  /**
   * Whether travel along `normal` (world) moves up the stack: the sign of
   * the normal against the axis in world. The placement is rigid, so the
   * matrix rotation carries the local axis.
   */
  axisSign(axis: SliceAxis, normal: THREE.Vector3): 1 | -1 {
    const local = axis === "axial" ? [0, 0, 1] : axis === "coronal" ? [0, 1, 0] : [1, 0, 0];
    const e = this.matrix.elements;
    const along =
      normal.x * (e[0] * local[0] + e[4] * local[1] + e[8] * local[2]) +
      normal.y * (e[1] * local[0] + e[5] * local[1] + e[9] * local[2]) +
      normal.z * (e[2] * local[0] + e[6] * local[1] + e[10] * local[2]);
    return along < 0 ? -1 : 1;
  }

  /** The corner handles, for the drag raycast (each carries its axis). */
  get handles(): { object: THREE.Object3D; axis: SliceAxis }[] {
    const out: { object: THREE.Object3D; axis: SliceAxis }[] = [];
    this.entries.forEach((entry, i) => {
      if (entry !== undefined) {
        for (const handle of entry.handles) {
          out.push({ object: handle, axis: AXES[i] });
        }
      }
    });
    return out;
  }

  /** The `corner`th handle's world position, for the test hook. */
  handleWorld(axis: SliceAxis, corner: number): [number, number, number] | null {
    const handle = this.entries[AXES.indexOf(axis)]?.handles[corner];
    if (handle === undefined) {
      return null;
    }
    const p = handle.getWorldPosition(new THREE.Vector3());
    return [p.x, p.y, p.z];
  }

  private buildQuad(axis: SliceAxis, plane: SlicePlane, size: [number, number, number]): QuadEntry {
    const spec = planeSpec(axis, size);
    const geometry = new THREE.PlaneGeometry(spec.width, spec.height);
    if (spec.spinZ) {
      geometry.rotateZ(Math.PI / 2);
    }
    const texture = new THREE.DataTexture(plane.data, plane.width, plane.height);
    texture.flipY = true;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide, toneMapped: false }),
    );
    mesh.rotation.set(spec.rotation[0], spec.rotation[1], spec.rotation[2]);
    const border = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: AXIS_COLOR[axis], toneMapped: false }),
    );
    mesh.add(border);
    // The drag handles: a ball on each corner (from the baked bounds, so
    // the spin lands them right), in the axis color.
    geometry.computeBoundingBox();
    const bb = geometry.boundingBox ?? new THREE.Box3();
    const radius = 0.02 * Math.hypot(bb.max.x - bb.min.x, bb.max.y - bb.min.y);
    const handleGeometry = new THREE.SphereGeometry(1, 12, 8);
    const handleMaterial = new THREE.MeshBasicMaterial({ color: AXIS_COLOR[axis], toneMapped: false });
    const handles: THREE.Mesh[] = [];
    for (const [hx, hy] of [
      [bb.min.x, bb.min.y],
      [bb.max.x, bb.min.y],
      [bb.min.x, bb.max.y],
      [bb.max.x, bb.max.y],
    ]) {
      const handle = new THREE.Mesh(handleGeometry, handleMaterial);
      handle.position.set(hx, hy, 0);
      handle.scale.setScalar(Math.max(radius, 1e-6));
      handle.userData.sliceAxis = axis;
      mesh.add(handle);
      handles.push(handle);
    }
    mesh.frustumCulled = false;
    return { mesh, texture, handles, handleGeometry, handleMaterial };
  }

  dispose(): void {
    for (const entry of this.entries) {
      if (entry !== undefined) {
        disposeEntry(entry);
      }
    }
    this.entries = [];
  }
}

/** Frees a quad: its mesh, border, handles and texture. */
function disposeEntry(entry: QuadEntry): void {
  entry.mesh.geometry.dispose();
  (entry.mesh.material as THREE.Material).dispose();
  for (const child of entry.mesh.children) {
    const mesh = child as THREE.Mesh | THREE.LineSegments;
    if (mesh !== undefined && (mesh as THREE.Mesh).geometry !== entry.handleGeometry) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
  }
  entry.handleGeometry.dispose();
  entry.handleMaterial.dispose();
  entry.texture.dispose();
}

export function SceneSlices(props: { app: AppState; loader: Loader; scene: Scene | WorkerScene }) {
  const { app, loader, scene } = props;
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const mpr = useApp(app, (s) => s.volumeMpr);
  const gl = useThree((s) => s.gl);
  const quads = useMemo(() => new SceneSliceQuads(), []);
  const [payload, setPayload] = useState<VolumePayload | null>(null);
  const item: Item | null = items.find((i) => i.mesh === selection) ?? null;
  const mesh: MeshHandle | null = item?.mesh ?? null;
  const active =
    item !== null && item.volumeId !== null && mpr !== null && mpr.volumeId === item.volumeId && mpr.inScene;
  const volumeId = active && item !== null ? item.volumeId : null;

  useEffect(() => {
    gl.localClippingEnabled = true;
  }, [gl]);

  useEffect(() => {
    setPayload(null);
    if (volumeId === null) {
      return;
    }
    let alive = true;
    loader
      .fetchVolume(volumeId)
      .then((got: VolumePayload) => {
        if (alive) {
          setPayload(got);
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [loader, volumeId]);

  const display = useMemo(
    () => (payload === null ? null : displayValues(payload.values, payload.pseudoHu)),
    [payload],
  );
  const range = useMemo(() => (display === null ? null : dataRange(display)), [display]);

  useEffect(() => {
    if (!active || payload === null || display === null || range === null || mpr === null || item === null) {
      quads.setSlices(null, {
        origin: [0, 0, 0],
        spacing: [1, 1, 1],
        columns: 0,
        rows: 0,
        slices: 0,
        center: [0, 0, 0],
      });
      if (item !== null) {
        item.material.clippingPlanes = null;
      }
      if (inspect.sceneSlices.id !== null) {
        inspect.sceneSlices = { id: null, shown: 0 };
      }
      return;
    }
    const preset = WINDOW_PRESETS[mpr.window];
    const colorWindow = preset.window === "auto" ? range[1] - range[0] : preset.window;
    const colorLevel = preset.level === "auto" ? (range[0] + range[1]) / 2 : preset.level;
    quads.setSlices(
      slicePlaneImages(
        display,
        { columns: payload.columns, rows: payload.rows, slices: payload.slices },
        mpr.slices,
        colorWindow,
        colorLevel,
      ),
      {
        origin: payload.origin,
        spacing: payload.spacing,
        columns: payload.columns,
        rows: payload.rows,
        slices: payload.slices,
        center: item.center,
      },
    );
    item.material.clippingPlanes = quads.clip;
    inspect.sceneSlices = { id: mpr.volumeId, shown: 3 };
    return () => {
      // The selection may move on: never leave a cutaway behind.
      item.material.clippingPlanes = null;
    };
  }, [active, payload, display, range, mpr, item, quads]);

  useEffect(
    () => () => {
      quads.dispose();
      if (inspect.sceneSlices.id !== null) {
        inspect.sceneSlices = { id: null, shown: 0 };
      }
    },
    [quads],
  );

  useFrame(() => {
    if (mesh === null) {
      return;
    }
    quads.setMatrix(scene.matrix4(mesh));
  });

  useEffect(() => {
    view.sliceQuads = quads;
    return () => {
      if (view.sliceQuads === quads) {
        view.sliceQuads = null;
      }
    };
  }, [quads]);

  return (
    <>
      <primitive object={quads.group} />
      <SliceDrag app={app} quads={quads} payload={payload} active={active} />
    </>
  );
}

type SliceDragState = {
  axis: SliceAxis;
  volumeId: string;
  index0: number;
  grab: [number, number, number];
  planePoint: [number, number, number];
  planeNormal: [number, number, number];
  /** The quad normal towards higher slices (world). */
  along: [number, number, number];
  spacing: number;
  count: number;
};

/**
 * Drags the slice quads by their corner balls: a press on one grabs the
 * stack, and pulling it moves the slice (the linked MPR state, so every
 * slice view follows). The orbit stands down for the gesture, like for a
 * sculpt stroke; the gizmo's own drags and the sculpt brush keep priority.
 */
function SliceDrag(props: { app: AppState; quads: SceneSliceQuads; payload: VolumePayload | null; active: boolean }) {
  const { app, quads } = props;
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const payloadRef = useRef(props.payload);
  payloadRef.current = props.payload;
  const activeRef = useRef(props.active);
  activeRef.current = props.active;

  useEffect(() => {
    const canvas = gl.domElement;
    const caster = new THREE.Raycaster();
    let drag: SliceDragState | null = null;
    let pointerId = -1;

    const rayOf = (x: number, y: number): THREE.Ray => {
      const r = canvas.getBoundingClientRect();
      caster.setFromCamera(
        new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2),
        camera,
      );
      return caster.ray;
    };

    const endDrag = (): void => {
      drag = null;
      pointerId = -1;
      canvas.style.cursor = "";
      if (view.orbit !== null) {
        view.orbit.enabled = true;
      }
    };

    const onDown = (e: PointerEvent): void => {
      if (e.button !== 0 || drag !== null || view.dragging) {
        return;
      }
      const snap = app.getSnapshot();
      if (snap.sculpt !== null || snap.brushPick !== null) {
        return;
      }
      const payload = payloadRef.current;
      const mpr = snap.volumeMpr;
      if (!activeRef.current || payload === null || mpr === null || !quads.group.visible) {
        return;
      }
      const ray = rayOf(e.clientX, e.clientY);
      const hits = caster.intersectObjects(
        quads.handles.map((h) => h.object),
        false,
      );
      if (hits.length === 0) {
        return;
      }
      const axis = (hits[0].object.userData.sliceAxis ?? null) as SliceAxis | null;
      const quad = axis === null ? null : quads.quadMesh(axis);
      if (axis === null || quad === null) {
        return;
      }
      const normal = quad.getWorldDirection(new THREE.Vector3());
      const seen = camera.getWorldDirection(new THREE.Vector3());
      const plane = dragPlaneNormal([normal.x, normal.y, normal.z], [seen.x, seen.y, seen.z]);
      if (plane === null) {
        return;
      }
      const center = hits[0].object.getWorldPosition(new THREE.Vector3());
      const grab = planeHit(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        [center.x, center.y, center.z],
        plane,
      );
      if (grab === null) {
        return;
      }
      const sign = quads.axisSign(axis, normal);
      const spacing = axis === "axial" ? payload.spacing[2] : axis === "coronal" ? payload.spacing[1] : payload.spacing[0];
      const count = axis === "axial" ? payload.slices : axis === "coronal" ? payload.rows : payload.columns;
      drag = {
        axis,
        volumeId: mpr.volumeId,
        index0: mpr.slices[axis],
        grab,
        planePoint: [center.x, center.y, center.z],
        planeNormal: plane,
        along: [sign * normal.x, sign * normal.y, sign * normal.z],
        spacing,
        count,
      };
      pointerId = e.pointerId;
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        // Already released: the drag still tracks until the button lifts.
      }
      if (view.orbit !== null) {
        view.orbit.enabled = false;
      }
      canvas.style.cursor = "grabbing";
    };

    const onMove = (e: PointerEvent): void => {
      if (drag === null) {
        if (e.buttons !== 0 || view.dragging) {
          return;
        }
        const snap = app.getSnapshot();
        if (
          snap.sculpt !== null ||
          snap.brushPick !== null ||
          !activeRef.current ||
          !quads.group.visible
        ) {
          return;
        }
        rayOf(e.clientX, e.clientY);
        const hits = caster.intersectObjects(
          quads.handles.map((h) => h.object),
          false,
        );
        canvas.style.cursor = hits.length === 0 ? "" : "grab";
        return;
      }
      if (e.pointerId !== pointerId) {
        return;
      }
      const ray = rayOf(e.clientX, e.clientY);
      const now = planeHit(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        drag.planePoint,
        drag.planeNormal,
      );
      if (now === null) {
        return;
      }
      const index = clampSlice(drag.index0 + sliceSteps(drag.grab, now, drag.along, drag.spacing), drag.count);
      const cur = app.getSnapshot().volumeMpr;
      if (cur !== null && cur.volumeId === drag.volumeId && cur.slices[drag.axis] !== index) {
        app.setVolumeMpr({ ...cur, slices: { ...cur.slices, [drag.axis]: index } });
      }
    };

    const onUp = (e: PointerEvent): void => {
      if (e.pointerId === pointerId) {
        endDrag();
      }
    };

    const onLeave = (): void => {
      if (drag === null) {
        canvas.style.cursor = "";
      }
    };

    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      endDrag();
    };
  }, [app, quads, camera, gl]);

  return null;
}
