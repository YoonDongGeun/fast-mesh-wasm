// Our volume renderer: a three.js raycaster through the 8-bit volume,
// shaded by the preset LUT. Composite and MIP modes, trackball/zoom via
// TrackballControls, rendered on demand. No vtk.js.
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import * as THREE from "three";
import { AXIS_COLOR, planeSpec, type SlicePlane, type SlicePlanes } from "./volume-slice";

const STEPS = 256;

const VERT = /* glsl */ `
varying vec3 vObj;
uniform vec3 uSize;
void main() {
  vObj = position / uSize + 0.5;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;
varying vec3 vObj;
uniform sampler3D uVolume;
uniform sampler2D uLut;
uniform vec3 uCamObj;
uniform vec3 uTexel;
uniform float uMip;
uniform float uAlphaScale;
// The cutaway: the slice fractions (see cutFractions); past 1.0 a cut stays open.
uniform vec3 uCut;

vec2 hitBox(vec3 ro, vec3 rd) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (vec3(0.0) - ro) * inv;
  vec3 t1 = (vec3(1.0) - ro) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

float sampleAt(vec3 p) {
  return texture(uVolume, p).r;
}

// Central differences, scaled back to a unit-per-voxel gradient so the
// light behaves the same on any grid size.
vec3 gradientAt(vec3 p) {
  vec3 g;
  g.x = sampleAt(p + vec3(uTexel.x, 0.0, 0.0)) - sampleAt(p - vec3(uTexel.x, 0.0, 0.0));
  g.y = sampleAt(p + vec3(0.0, uTexel.y, 0.0)) - sampleAt(p - vec3(0.0, uTexel.y, 0.0));
  g.z = sampleAt(p + vec3(0.0, 0.0, uTexel.z)) - sampleAt(p - vec3(0.0, 0.0, uTexel.z));
  return g / uTexel;
}

// Keeps the samples below the cut on one axis (the mesh clip's twin).
void clipAxis(float ro, float rd, float cut, inout float t0, inout float t1) {
  if (abs(rd) > 1e-9) {
    float tc = (cut - ro) / rd;
    if (rd > 0.0) {
      t1 = min(t1, tc);
    } else {
      t0 = max(t0, tc);
    }
  } else if (ro > cut) {
    t1 = t0;
  }
}

void main() {
  vec3 rd = normalize(vObj - uCamObj);
  vec2 hit = hitBox(uCamObj, rd);
  if (hit.y <= max(hit.x, 0.0)) {
    discard;
  }
  float t0 = max(hit.x, 0.0);
  float t1 = hit.y;
  clipAxis(uCamObj.x, rd.x, uCut.x, t0, t1);
  clipAxis(uCamObj.y, rd.y, uCut.y, t0, t1);
  clipAxis(uCamObj.z, rd.z, uCut.z, t0, t1);
  if (t1 <= t0) {
    discard;
  }
  float dt = (t1 - t0) / float(${STEPS});
  vec3 lightDir = normalize(vec3(-0.45, 0.8, 0.65));
  if (uMip > 0.5) {
    float best = 0.0;
    for (int i = 0; i < ${STEPS}; i++) {
      float s = texture(uVolume, uCamObj + rd * (t0 + (float(i) + 0.5) * dt)).r;
      best = max(best, s);
    }
    vec4 c = texture(uLut, vec2(best, 0.5));
    gl_FragColor = vec4(c.rgb, 1.0);
  } else {
    vec3 acc = vec3(0.0);
    float alpha = 0.0;
    for (int i = 0; i < ${STEPS}; i++) {
      vec3 p = uCamObj + rd * (t0 + (float(i) + 0.5) * dt);
      float s = sampleAt(p);
      vec4 c = texture(uLut, vec2(s, 0.5));
      float a = 1.0 - pow(1.0 - c.a, uAlphaScale);
      if (a > 0.003) {
        vec3 n = -normalize(gradientAt(p) + vec3(1e-6));
        float shade = 0.35 + 0.75 * max(dot(n, lightDir), 0.0);
        acc += (1.0 - alpha) * c.rgb * (a * shade);
        alpha += (1.0 - alpha) * a;
        if (alpha > 0.99) {
          break;
        }
      }
    }
    gl_FragColor = vec4(acc, alpha);
  }
}
`;

export type VolumeGrid = {
  columns: number;
  rows: number;
  slices: number;
  spacing: [number, number, number];
};

/** The raymarch box's material with its textures, shared by the panel view and the main scene's box. */
export interface RaymarchMaterial {
  material: THREE.ShaderMaterial;
  volumeTexture: THREE.Data3DTexture;
  lutTexture: THREE.DataTexture;
  size: THREE.Vector3;
  setLut(lut: Uint8Array): void;
  dispose(): void;
}

export function createRaymarchMaterial(data: Uint8Array, grid: VolumeGrid, lut: Uint8Array): RaymarchMaterial {
  const size = new THREE.Vector3(
    grid.columns * grid.spacing[0],
    grid.rows * grid.spacing[1],
    grid.slices * grid.spacing[2],
  );
  const volumeTexture = new THREE.Data3DTexture(data, grid.columns, grid.rows, grid.slices);
  volumeTexture.format = THREE.RedFormat;
  volumeTexture.type = THREE.UnsignedByteType;
  volumeTexture.minFilter = THREE.LinearFilter;
  volumeTexture.magFilter = THREE.LinearFilter;
  volumeTexture.wrapS = THREE.ClampToEdgeWrapping;
  volumeTexture.wrapT = THREE.ClampToEdgeWrapping;
  volumeTexture.wrapR = THREE.ClampToEdgeWrapping;
  volumeTexture.needsUpdate = true;

  const lutTexture = new THREE.DataTexture(lut, lut.length / 4, 1, THREE.RGBAFormat);
  lutTexture.minFilter = THREE.LinearFilter;
  lutTexture.magFilter = THREE.LinearFilter;
  lutTexture.wrapS = THREE.ClampToEdgeWrapping;
  lutTexture.wrapT = THREE.ClampToEdgeWrapping;
  lutTexture.needsUpdate = true;

  // The presets assume ~0.8 mm samples; correct the opacity for ours.
  const stepLen = size.length() / STEPS / 0.8;
  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    side: THREE.BackSide,
    transparent: true,
    // The slice planes are opaque and write depth; the box blends over
    // them so they show through wherever the volume is transparent.
    depthTest: false,
    depthWrite: false,
    uniforms: {
      uVolume: { value: volumeTexture },
      uLut: { value: lutTexture },
      uSize: { value: size },
      uCamObj: { value: new THREE.Vector3() },
      uTexel: { value: new THREE.Vector3(1 / grid.columns, 1 / grid.rows, 1 / grid.slices) },
      uMip: { value: 0 },
      uAlphaScale: { value: stepLen },
      uCut: { value: new THREE.Vector3(2, 2, 2) },
    },
  });
  return {
    material,
    volumeTexture,
    lutTexture,
    size,
    setLut(next: Uint8Array): void {
      lutTexture.image.data = next;
      lutTexture.needsUpdate = true;
    },
    dispose(): void {
      material.dispose();
      volumeTexture.dispose();
      lutTexture.dispose();
    },
  };
}

export class VolumeView3D {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: TrackballControls;
  private mesh: THREE.Mesh;
  private raymarch: RaymarchMaterial;
  private material: THREE.ShaderMaterial;
  private size: THREE.Vector3;
  private planes: { mesh: THREE.Mesh; texture: THREE.DataTexture; place: (index: number, count: number) => void }[] = [];
  private onChange: () => void;

  constructor(canvas: HTMLCanvasElement, data: Uint8Array, grid: VolumeGrid, lut: Uint8Array) {
    this.raymarch = createRaymarchMaterial(data, grid, lut);
    this.material = this.raymarch.material;
    this.size = this.raymarch.size;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // The shader outputs display-ready colors (the LUT is authored in
    // sRGB): no output transform, or the clear color and the raycast
    // disagree about the color space.
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.setClearColor(new THREE.Color(0.05, 0.06, 0.08), 1);
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 10000);
    const extent = Math.max(this.size.x, this.size.y, this.size.z);
    const dist = 1.9 * extent;
    this.camera.position.set(0, 0, dist);
    this.camera.near = extent * 0.01;
    this.camera.far = extent * 20;
    this.camera.updateProjectionMatrix();
    this.controls = new TrackballControls(this.camera, canvas);
    this.controls.target.set(0, 0, 0);
    this.controls.minDistance = extent * 0.5;
    this.controls.maxDistance = extent * 8;
    this.controls.staticMoving = true;
    this.controls.update();

    this.mesh = new THREE.Mesh(new THREE.BoxGeometry(this.size.x, this.size.y, this.size.z), this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
    this.onChange = () => this.render();
    this.controls.addEventListener("change", this.onChange);
  }

  setLut(lut: Uint8Array): void {
    this.raymarch.setLut(lut);
    this.render();
  }

  setMip(mip: boolean): void {
    this.material.uniforms.uMip.value = mip ? 1 : 0;
    this.render();
  }

  /**
   * The slice planes (or null to hide them): windowed grayscale quads at
   * the slice positions, like a medical viewer's 3D overlay. Built once;
   * later calls move them and refresh their pixels.
   */
  setPlanes(planes: SlicePlanes | null): void {
    if (planes === null) {
      for (const p of this.planes) {
        p.mesh.visible = false;
      }
      this.render();
      return;
    }
    const list = [planes.axial, planes.coronal, planes.sagittal];
    list.forEach((plane, i) => {
      let entry = this.planes[i];
      if (entry === undefined) {
        entry = this.buildPlane(i, plane);
        this.planes[i] = entry;
      } else {
        entry.texture.image.data = plane.data;
        entry.texture.needsUpdate = true;
      }
      entry.mesh.visible = true;
      entry.place(plane.index, plane.count);
    });
    this.render();
  }

  private buildPlane(i: number, plane: SlicePlane): { mesh: THREE.Mesh; texture: THREE.DataTexture; place: (index: number, count: number) => void } {
    const texture = new THREE.DataTexture(plane.data, plane.width, plane.height);
    texture.flipY = true;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;
    const at = (index: number, count: number, extent: number): number => ((index + 0.5) / count - 0.5) * extent;
    const axis = (["axial", "coronal", "sagittal"] as const)[i];
    const spec = planeSpec(axis, [this.size.x, this.size.y, this.size.z]);
    const geometry = new THREE.PlaneGeometry(spec.width, spec.height);
    if (spec.spinZ) {
      geometry.rotateZ(Math.PI / 2);
    }
    const place =
      i === 0
        ? (m: THREE.Mesh, index: number, count: number) => m.position.set(0, 0, at(index, count, this.size.z))
        : i === 1
          ? (m: THREE.Mesh, index: number, count: number) => m.position.set(0, at(index, count, this.size.y), 0)
          : (m: THREE.Mesh, index: number, count: number) => m.position.set(at(index, count, this.size.x), 0, 0);
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }));
    mesh.rotation.set(spec.rotation[0], spec.rotation[1], spec.rotation[2]);
    // The border in the axis color: the plane itself is mostly black air.
    const border = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry),
      new THREE.LineBasicMaterial({ color: [AXIS_COLOR.axial, AXIS_COLOR.coronal, AXIS_COLOR.sagittal][i] }),
    );
    mesh.add(border);
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    return { mesh, texture, place: (index: number, count: number) => place(mesh, index, count) };
  }

  setSize(width: number, height: number): void {
    this.renderer.setSize(Math.max(1, Math.floor(width)), Math.max(1, Math.floor(height)), false);
    this.camera.aspect = Math.max(1, width) / Math.max(1, height);
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render(): void {
    const cam = this.camera.position.clone().divide(this.size).addScalar(0.5);
    (this.material.uniforms.uCamObj.value as THREE.Vector3).copy(cam);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.controls.removeEventListener("change", this.onChange);
    this.controls.dispose();
    this.mesh.geometry.dispose();
    this.raymarch.dispose();
    for (const p of this.planes) {
      p.mesh.geometry.dispose();
      (p.mesh.material as THREE.Material).dispose();
      for (const child of p.mesh.children) {
        const line = child as THREE.LineSegments;
        line.geometry.dispose();
        (line.material as THREE.Material).dispose();
      }
      p.texture.dispose();
    }
    this.planes = [];
    this.renderer.dispose();
  }
}
