// The scene's volume rendering: the selected volume raymarched with the
// shared 3D preset, cut away below the linked slices (the kept side
// faces the boot camera), and blended over the scene, while its surface
// mesh hides. Marched on demand into a half-res target (a full march
// every frame would stall software GL); the composite quad blends the
// cached target over the scene, like the panel's box blends over its
// planes — including over anything overlapping the box.
import type { Scene } from "fast-mesh-wasm";
import { useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import type { Loader } from "../load/loader";
import type { VolumePayload } from "../load/protocol";
import { inspect } from "../panels/inspect";
import { displayValues } from "../panels/volume-data";
import { dataRange, normalizeVolume, presetLut } from "../panels/volume-lut";
import { VOLUME_PRESETS } from "../panels/volume-presets";
import { createRaymarchMaterial, type RaymarchMaterial, type VolumeGrid } from "../panels/volume-three";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { cutFractions, matchVolume, volumeBoxLayout } from "./volume-box";

/** The march target's scale: half res marches 4x faster, upscaled on composite. */
const RT_SCALE = 0.5;

const COMPOSITE_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.99999, 1.0);
}
`;

const COMPOSITE_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D uMap;
void main() {
  gl_FragColor = texture2D(uMap, vUv);
}
`;

const CLEAR = new THREE.Color(0, 0, 0);

/** The raymarched volume box in the main scene, composited from a cached march. */
export class SceneVolumeBox {
  /** The composite quad: parent this to the main scene. */
  readonly object: THREE.Mesh;
  private rtScene = new THREE.Scene();
  private group = new THREE.Group();
  private composite: THREE.ShaderMaterial;
  private box: THREE.Mesh | null = null;
  private raymarch: RaymarchMaterial | null = null;
  private target: THREE.WebGLRenderTarget | null = null;
  private volumeId: string | null = null;
  private dirty = true;
  private camPos = new THREE.Vector3(Infinity, Infinity, Infinity);
  private camQuat = new THREE.Quaternion();
  private camProj = new THREE.Matrix4();
  private boxMatrix = new THREE.Matrix4();
  private clearColor = new THREE.Color();
  private scratch = new THREE.Vector3();

  constructor() {
    this.group.matrixAutoUpdate = false;
    this.rtScene.add(this.group);
    this.composite = new THREE.ShaderMaterial({
      vertexShader: COMPOSITE_VERT,
      fragmentShader: COMPOSITE_FRAG,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      uniforms: { uMap: { value: null } },
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    this.object = new THREE.Mesh(geometry, this.composite);
    this.object.frustumCulled = false;
    this.object.raycast = () => {};
    // First among transparents: helpers blend over the volume, not under it.
    this.object.renderOrder = -1;
    this.object.visible = false;
  }

  /** The mesh's placement: the box rides it like the slice quads. */
  setMatrix(m: ArrayLike<number>): void {
    this.group.matrix.fromArray(m);
    this.group.matrixWorldNeedsUpdate = true;
  }

  /** Shows the volume: builds the box on a new series, steers LUT and cuts otherwise. */
  showVolume(
    volumeId: string,
    data: Uint8Array,
    grid: VolumeGrid,
    lut: Uint8Array,
    layout: { center: [number, number, number]; size: [number, number, number] },
    cuts: [number, number, number],
  ): void {
    if (this.volumeId !== volumeId) {
      this.clearBox();
      this.raymarch = createRaymarchMaterial(data, grid, lut);
      this.box = new THREE.Mesh(
        new THREE.BoxGeometry(layout.size[0], layout.size[1], layout.size[2]),
        this.raymarch.material,
      );
      this.box.position.set(layout.center[0], layout.center[1], layout.center[2]);
      this.box.frustumCulled = false;
      this.group.add(this.box);
      this.volumeId = volumeId;
    } else if (this.raymarch !== null) {
      this.raymarch.setLut(lut);
    }
    if (this.raymarch !== null) {
      (this.raymarch.material.uniforms.uCut.value as THREE.Vector3).set(cuts[0], cuts[1], cuts[2]);
    }
    this.object.visible = true;
    this.dirty = true;
  }

  hide(): void {
    this.object.visible = false;
  }

  /**
   * Marches when the camera, the box, or the volume state moved; the
   * composite blends the cached target every frame. Call every frame.
   */
  update(gl: THREE.WebGLRenderer, camera: THREE.Camera): void {
    if (!this.object.visible || this.box === null || this.raymarch === null) {
      return;
    }
    const w = Math.max(1, Math.floor(gl.domElement.width * RT_SCALE));
    const h = Math.max(1, Math.floor(gl.domElement.height * RT_SCALE));
    if (this.target === null || this.target.width !== w || this.target.height !== h) {
      this.target?.dispose();
      this.target = new THREE.WebGLRenderTarget(w, h, { depthBuffer: false, stencilBuffer: false });
      this.composite.uniforms.uMap.value = this.target.texture;
      this.dirty = true;
    }
    const moved =
      this.dirty ||
      this.camPos.distanceToSquared(camera.position) > 1e-8 ||
      this.camQuat.angleTo(camera.quaternion) > 1e-7 ||
      !this.camProj.equals(camera.projectionMatrix) ||
      !this.boxMatrix.equals(this.group.matrix);
    if (!moved) {
      return;
    }
    this.camPos.copy(camera.position);
    this.camQuat.copy(camera.quaternion);
    this.camProj.copy(camera.projectionMatrix);
    this.boxMatrix.copy(this.group.matrix);
    this.dirty = false;
    this.group.updateMatrixWorld(true);
    this.scratch.copy(camera.position);
    this.box.worldToLocal(this.scratch);
    const size = this.raymarch.size;
    (this.raymarch.material.uniforms.uCamObj.value as THREE.Vector3).set(
      this.scratch.x / size.x + 0.5,
      this.scratch.y / size.y + 0.5,
      this.scratch.z / size.z + 0.5,
    );
    gl.getClearColor(this.clearColor);
    const clearAlpha = gl.getClearAlpha();
    gl.setRenderTarget(this.target);
    gl.setClearColor(CLEAR, 0);
    gl.clear();
    gl.render(this.rtScene, camera);
    gl.setRenderTarget(null);
    gl.setClearColor(this.clearColor, clearAlpha);
  }

  private clearBox(): void {
    if (this.box !== null) {
      this.group.remove(this.box);
      this.box.geometry.dispose();
      this.box = null;
    }
    this.raymarch?.dispose();
    this.raymarch = null;
    this.volumeId = null;
  }

  dispose(): void {
    this.clearBox();
    this.target?.dispose();
    this.target = null;
    this.object.geometry.dispose();
    this.composite.dispose();
  }
}

export function SceneVolume(props: { app: AppState; loader: Loader; scene: Scene | WorkerScene }) {
  const { app, loader, scene } = props;
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const mpr = useApp(app, (s) => s.volumeMpr);
  const box = useMemo(() => new SceneVolumeBox(), []);
  const [payload, setPayload] = useState<VolumePayload | null>(null);
  const found = matchVolume(items, selection, mpr);
  const liveItem: Item | null = found?.item ?? null;
  const active = found !== null && found.mpr.inScene;
  const volumeId = active && found !== null ? found.volumeId : null;

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
  const prepared = useMemo(
    () => (display === null || range === null ? null : normalizeVolume(display, range[0], range[1])),
    [display, range],
  );

  useEffect(() => {
    if (
      !active ||
      payload === null ||
      prepared === null ||
      range === null ||
      mpr === null ||
      liveItem === null ||
      volumeId === null
    ) {
      box.hide();
      if (inspect.sceneVolume.id !== null) {
        inspect.sceneVolume = { id: null, preset: "" };
      }
      return;
    }
    const preset = VOLUME_PRESETS[mpr.preset];
    box.showVolume(
      volumeId,
      prepared,
      { columns: payload.columns, rows: payload.rows, slices: payload.slices, spacing: payload.spacing },
      presetLut(preset, range[0], range[1]),
      volumeBoxLayout(payload, liveItem.center),
      cutFractions(mpr.slices, payload),
    );
    inspect.sceneVolume = { id: volumeId, preset: preset.name };
  }, [active, payload, prepared, range, mpr, liveItem, volumeId, box]);

  useEffect(
    () => () => {
      box.dispose();
      if (inspect.sceneVolume.id !== null) {
        inspect.sceneVolume = { id: null, preset: "" };
      }
    },
    [box],
  );

  useFrame(({ gl, camera }) => {
    if (liveItem !== null) {
      box.setMatrix(scene.matrix4(liveItem.mesh));
    }
    box.update(gl, camera);
  });

  return <primitive object={box.object} />;
}
