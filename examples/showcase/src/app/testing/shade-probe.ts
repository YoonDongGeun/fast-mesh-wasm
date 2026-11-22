// Test-only (window.__showcase.shadeProbe): runs bindMesh's fragment shading
// GLSL on the GPU so scripts/showcase-check.mjs can hold it to the library's
// JavaScript reference. Each value is drawn as a quad with that value at all
// four corners, through a fragment-shaded binding of an unlit material, into
// a float render target, and read back.
import type { ColorRule, MeshChange, MeshHandle, SceneLike } from "fast-mesh-wasm";
import { bindMesh } from "fast-mesh-wasm/three";
import * as THREE from "three";

/** Pixels per quad side; the centre pixel is read. */
const CELL = 4;

export interface ShadeProbe {
  /** Linear RGBA read back for each value. */
  rgba: number[][];
  /** The shading attribute of each value (its first corner), as bound. */
  attribute: number[][];
}

/** The fragment colour of each of `values` under `rule` at `softness`, from the GPU. */
export function shadeProbe(rule: ColorRule, values: number[], softness: number): ShadeProbe {
  const n = values.length;
  const columns = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / columns);
  const width = columns * CELL;
  const height = rows * CELL;
  // One quad per value: 4 vertex slots, 2 triangles.
  const positions = new Float32Array(n * 12);
  const indices = new Uint32Array(n * 6);
  const vertexValues = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const x = (i % columns) * CELL;
    const y = Math.floor(i / columns) * CELL;
    positions.set([x, y, 0, x + CELL, y, 0, x + CELL, y + CELL, 0, x, y + CELL, 0], 12 * i);
    indices.set([4 * i, 4 * i + 1, 4 * i + 2, 4 * i, 4 * i + 2, 4 * i + 3], 6 * i);
    vertexValues.fill(values[i], 4 * i, 4 * i + 4);
  }
  const colors = new Uint8Array(n * 16);
  const mesh = {} as MeshHandle;
  const identity = new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const host: SceneLike = {
    views: () => ({
      positions: () => positions,
      normals: () => null,
      colors: () => colors,
      values: () => vertexValues,
      indices: () => indices,
    }),
    matrix4: () => identity,
    placement: () => ({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] }),
    section: () => {
      throw new Error("no sections");
    },
    sectionPlane: () => {
      throw new Error("no sections");
    },
    onChanges: () => () => {},
    commitFrame: (): MeshChange[] => [],
    drag: () => {
      throw new Error("no drags");
    },
    coloring: () => rule,
  };
  // Unlit, and alpha kept: the output is the shaded colour itself.
  const material = new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.NoBlending });
  const bound = bindMesh(host, mesh, { material, shading: "fragment", edgeSoftness: softness });
  const renderer = new THREE.WebGLRenderer({ canvas: document.createElement("canvas") });
  const target = new THREE.WebGLRenderTarget(width, height, { type: THREE.FloatType });
  try {
    if (bound.shading !== "fragment") {
      throw new Error("the probe binding is not fragment-shaded");
    }
    const scene = new THREE.Scene();
    scene.add(bound.object);
    const camera = new THREE.OrthographicCamera(0, width, height, 0, -1, 1);
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    const pixels = new Float32Array(width * height * 4);
    renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
    const attr = bound.object.geometry.getAttribute("fmValue").array as Float32Array;
    const rgba: number[][] = [];
    const attribute: number[][] = [];
    for (let i = 0; i < n; i++) {
      const x = (i % columns) * CELL + CELL / 2;
      const y = Math.floor(i / columns) * CELL + CELL / 2;
      const at = (y * width + x) * 4;
      rgba.push(Array.from(pixels.subarray(at, at + 4)));
      attribute.push(Array.from(attr.subarray(16 * i, 16 * i + 4)));
    }
    return { rgba, attribute };
  } finally {
    bound.dispose();
    material.dispose();
    target.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
  }
}
