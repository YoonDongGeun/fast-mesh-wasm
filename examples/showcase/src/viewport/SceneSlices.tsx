// The scene overlay: the linked MPR slices as quads cutting through the
// selected mesh in the main scene, with the mesh clipped away above each
// slice (a cutaway), like a medical viewer. The quads ride the mesh's
// placement, so gizmo moves carry them along.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useState } from "react";
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
  type SlicePlanes,
} from "../panels/volume-slice";

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

/** The three quads plus their cutaway planes, parented to one matrix. */
class SceneSliceQuads {
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
  private entries: { mesh: THREE.Mesh; texture: THREE.DataTexture }[] = [];
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
      if (entry === undefined) {
        const spec = planeSpec(AXES[i], [ex, ey, ez]);
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
          new THREE.LineBasicMaterial({ color: AXIS_COLOR[AXES[i]], toneMapped: false }),
        );
        mesh.add(border);
        mesh.frustumCulled = false;
        this.group.add(mesh);
        entry = { mesh, texture };
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

  dispose(): void {
    for (const entry of this.entries) {
      entry.mesh.geometry.dispose();
      (entry.mesh.material as THREE.Material).dispose();
      for (const child of entry.mesh.children) {
        const line = child as THREE.LineSegments;
        line.geometry.dispose();
        (line.material as THREE.Material).dispose();
      }
      entry.texture.dispose();
    }
    this.entries = [];
  }
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

  return <primitive object={quads.group} />;
}
