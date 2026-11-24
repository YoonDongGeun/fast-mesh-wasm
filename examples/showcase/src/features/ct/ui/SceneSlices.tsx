// The scene overlay: the linked MPR slices as quads cutting through the
// selected mesh in the main scene, with the mesh clipped away below each
// slice (a cutaway opening towards the boot camera), like a medical
// viewer. The quads ride the mesh's placement, so gizmo moves carry them
// along.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { Loader } from "../../../io/loader";
import type { VolumePayload } from "../../../io/protocol";
import type { AppState } from "../../../app/state/app-state";
import { inspect } from "../../../app/state/readouts";
import { presetWindowLevel, WINDOW_PRESETS } from "../ux/volume-presets";
import {
  AXIS_COLOR,
  planeSpec,
  SLICE_AXES,
  slicePlaneImages,
  type SliceAxis,
  type SlicePlane,
  type SlicePlanes,
} from "../ux/volume-slice";
import { setSliceClip } from "../../../viewport/ux/clip-planes";
import { view } from "../../../viewport/ux/view";
import { useSceneVolume } from "../ux/use-scene-volume";
import { useSliceDrag } from "../ux/use-slice-drag";

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
    new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),
    new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    new THREE.Plane(new THREE.Vector3(1, 0, 0), 0),
  ];
  private local: [THREE.Plane, THREE.Plane, THREE.Plane] = [
    new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),
    new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    new THREE.Plane(new THREE.Vector3(1, 0, 0), 0),
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
        entry = this.buildQuad(SLICE_AXES[i], plane, [ex, ey, ez]);
        this.group.add(entry.mesh);
        this.entries[i] = entry;
      } else {
        entry.texture.image.data = plane.data;
        entry.texture.needsUpdate = true;
      }
      entry.mesh.position.set(at[i][0], at[i][1], at[i][2]);
    });
    this.local[0].constant = -at[0][2];
    this.local[1].constant = -at[1][1];
    this.local[2].constant = -at[2][0];
    this.group.visible = true;
  }

  /** The quad's mesh, once built. */
  quadMesh(axis: SliceAxis): THREE.Mesh | null {
    return this.entries[SLICE_AXES.indexOf(axis)]?.mesh ?? null;
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
          out.push({ object: handle, axis: SLICE_AXES[i] });
        }
      }
    });
    return out;
  }

  /** The `corner`th handle's world position, for the test hook. */
  handleWorld(axis: SliceAxis, corner: number): [number, number, number] | null {
    const handle = this.entries[SLICE_AXES.indexOf(axis)]?.handles[corner];
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
  const gl = useThree((s) => s.gl);
  const quads = useMemo(() => new SceneSliceQuads(), []);

  useEffect(() => {
    gl.localClippingEnabled = true;
  }, [gl]);

  const { item, active, mpr, payload, display, range } = useSceneVolume(app, loader);
  const mesh: MeshHandle | null = item?.mesh ?? null;

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
        setSliceClip(item.material, null);
      }
      if (inspect.sceneSlices.id !== null) {
        inspect.sceneSlices = { id: null, shown: 0 };
      }
      return;
    }
    const { window: colorWindow, level: colorLevel } = presetWindowLevel(WINDOW_PRESETS[mpr.window], range);
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
    setSliceClip(item.material, quads.clip);
    inspect.sceneSlices = { id: mpr.volumeId, shown: 3 };
    return () => {
      // The selection may move on: never leave a cutaway behind.
      setSliceClip(item.material, null);
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

function SliceDrag(props: { app: AppState; quads: SceneSliceQuads; payload: VolumePayload | null; active: boolean }) {
  useSliceDrag(props.app, props.quads, props.payload, props.active);
  return null;
}
