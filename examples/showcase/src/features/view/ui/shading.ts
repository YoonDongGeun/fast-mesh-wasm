// Viewport diagnostic shading modes (#232):
// Smooth, Flat, Auto-Crease (dihedral threshold), Wireframe-on-Shaded,
// Studio MatCap, and Backface Warning (pink/crimson back-facing fragments).
import type { MeshBinding } from "fast-mesh-wasm/three";
import * as THREE from "three";
import type { DiagnosticShadingMode } from "../../../app/state/app-state";
import { setWireOverlay } from "./wire";

export const DIAGNOSTIC_SHADING_OPTIONS: readonly {
  mode: DiagnosticShadingMode;
  label: string;
  shortLabel: string;
  title: string;
}[] = [
  { mode: "smooth", label: "Smooth", shortLabel: "Smooth", title: "Standard smooth PBR shading" },
  { mode: "flat", label: "Flat", shortLabel: "Flat", title: "Per-triangle flat faceted normals" },
  {
    mode: "autoCrease",
    label: "Auto-Crease",
    shortLabel: "Crease",
    title: "Creased normals & feature edges split above the dihedral threshold",
  },
  {
    mode: "wireframeShaded",
    label: "Wire-on-Shaded",
    shortLabel: "Wire",
    title: "Shaded surface overlaid with triangle wireframe",
  },
  { mode: "matcap", label: "MatCap", shortLabel: "MatCap", title: "Sculptor ceramic MatCap shading" },
  {
    mode: "backfaceWarning",
    label: "Backface Warning",
    shortLabel: "Backface",
    title: "Highlight back-facing / inverted faces in warning pink",
  },
];

const AUTO_CREASE_GROUP = "fastMeshAutoCreaseOverlay";
const MAX_CREASE_OVERLAY_TRIS = 60_000;

interface DiagUniforms {
  fmDiagMode: { value: number };
  fmMatcapTex: { value: THREE.DataTexture };
}

/** Generates a procedural 2D spherical MatCap RGBA8 image (sRGB bytes). */
export function buildStudioMatcapRgba(size = 128): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  const invHalf = 2 / (size - 1);
  // Key, fill, and subtle cool rim directional lights in view space.
  const kLen = Math.hypot(-0.42, 0.56, 0.71);
  const kx = -0.42 / kLen;
  const ky = 0.56 / kLen;
  const kz = 0.71 / kLen;
  const rLen = Math.hypot(0.6, -0.3, 0.45);
  const rx = 0.6 / rLen;
  const ry = -0.3 / rLen;
  const rz = 0.45 / rLen;

  for (let y = 0; y < size; y++) {
    const ny = y * invHalf - 1;
    for (let x = 0; x < size; x++) {
      const nx = x * invHalf - 1;
      const r2 = nx * nx + ny * ny;
      const nz = r2 < 1 ? Math.sqrt(1 - r2) : 0;
      const len = Math.hypot(nx, ny, nz) || 1;
      const ux = nx / len;
      const uy = ny / len;
      const uz = nz / len;

      const diffKey = Math.max(0, ux * kx + uy * ky + uz * kz);
      const diffRim = Math.max(0, ux * rx + uy * ry + uz * rz);
      const hx = kx;
      const hy = ky;
      const hz = kz + 1;
      const hLen = Math.hypot(hx, hy, hz) || 1;
      const ndh = Math.max(0, (ux * hx + uy * hy + uz * hz) / hLen);
      const spec = Math.pow(ndh, 36) * 0.48;
      const fresnel = Math.pow(1 - Math.max(0, uz), 2.4) * 0.22;

      const r = Math.min(255, Math.round((0.18 + 0.64 * diffKey + 0.14 * diffRim + spec + fresnel * 0.75) * 255));
      const g = Math.min(255, Math.round((0.17 + 0.61 * diffKey + 0.17 * diffRim + spec + fresnel * 0.85) * 255));
      const b = Math.min(255, Math.round((0.16 + 0.57 * diffKey + 0.24 * diffRim + spec * 0.95 + fresnel) * 255));

      const i = (y * size + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return data;
}

let cachedMatcap: THREE.DataTexture | null = null;

/** Returns the shared procedural ceramic studio MatCap texture. */
export function getStudioMatcapTexture(): THREE.DataTexture {
  if (cachedMatcap === null) {
    const tex = new THREE.DataTexture(buildStudioMatcapRgba(128), 128, 128, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    cachedMatcap = tex;
  }
  return cachedMatcap;
}

/**
 * Installs the idempotent diagnostic shader hook (MatCap & Backface Warning) onto
 * a `MeshStandardMaterial` BEFORE `<FastMesh>` wraps it with `patchMaterial`.
 */
export function installDiagnosticShader(material: THREE.MeshStandardMaterial): DiagUniforms {
  const existing = material.userData.__diagUniforms as DiagUniforms | undefined;
  if (existing !== undefined) {
    return existing;
  }
  const uniforms: DiagUniforms = {
    fmDiagMode: { value: 0 },
    fmMatcapTex: { value: getStudioMatcapTexture() },
  };
  material.userData.__diagUniforms = uniforms;
  const prevCompile = material.onBeforeCompile;
  const prevCacheKey = material.customProgramCacheKey;

  material.onBeforeCompile = function (shader, renderer) {
    prevCompile.call(this, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    if (!shader.fragmentShader.includes("uniform int fmDiagMode;")) {
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <common>",
        "#include <common>\nuniform int fmDiagMode;\nuniform sampler2D fmMatcapTex;",
      );
    }
    const snippet = [
      "if (fmDiagMode == 1) {",
      "  vec3 mcN = normalize(normal);",
      "  vec2 mcUv = mcN.xy * 0.495 + 0.5;",
      "  vec3 mcCol = texture2D(fmMatcapTex, mcUv).rgb;",
      "  outgoingLight = mcCol * mix(vec3(1.0), diffuseColor.rgb, 0.35) + totalEmissiveRadiance * 0.35;",
      "} else if (fmDiagMode == 2 && !gl_FrontFacing) {",
      "  outgoingLight = vec3(0.95, 0.12, 0.45) * (0.45 + 0.55 * abs(normalize(normal).z)) + totalEmissiveRadiance * 0.2;",
      "}",
    ].join("\n");
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <opaque_fragment>",
      `${snippet}\n#include <opaque_fragment>`,
    );
  };

  material.customProgramCacheKey = function () {
    return `${prevCacheKey.call(this)}|diag-v1`;
  };
  return uniforms;
}

/**
 * Builds non-indexed geometry whose per-corner normals blend adjacent faces within
 * `creaseAngleDeg` while keeping sharp discontinuities across harder dihedral edges.
 */
export function computeCreasedGeometry(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  creaseAngleDeg: number,
): { positions: Float32Array; normals: Float32Array } {
  const triCount = Math.floor(indices.length / 3);
  const outPos = new Float32Array(triCount * 9);
  const outNorm = new Float32Array(triCount * 9);
  if (triCount === 0) {
    return { positions: outPos, normals: outNorm };
  }

  const faceNormals = new Float32Array(triCount * 3);
  const vCount = Math.floor(positions.length / 3);
  const vertTris: number[][] = Array.from({ length: vCount }, () => []);

  for (let t = 0; t < triCount; t++) {
    const a = indices[3 * t];
    const b = indices[3 * t + 1];
    const c = indices[3 * t + 2];
    const ax = positions[3 * a];
    const ay = positions[3 * a + 1];
    const az = positions[3 * a + 2];
    const ux = positions[3 * b] - ax;
    const uy = positions[3 * b + 1] - ay;
    const uz = positions[3 * b + 2] - az;
    const vx = positions[3 * c] - ax;
    const vy = positions[3 * c + 1] - ay;
    const vz = positions[3 * c + 2] - az;
    const cx = uy * vz - uz * vy;
    const cy = uz * vx - ux * vz;
    const cz = ux * vy - uy * vx;
    const len = Math.hypot(cx, cy, cz);
    if (len > 1e-12) {
      faceNormals[3 * t] = cx / len;
      faceNormals[3 * t + 1] = cy / len;
      faceNormals[3 * t + 2] = cz / len;
    } else {
      faceNormals[3 * t + 2] = 1;
    }
    if (a < vCount) vertTris[a].push(t);
    if (b < vCount) vertTris[b].push(t);
    if (c < vCount) vertTris[c].push(t);
  }

  const cosThreshold = Math.cos(THREE.MathUtils.degToRad(Math.max(0, Math.min(180, creaseAngleDeg))));
  for (let t = 0; t < triCount; t++) {
    const fnx = faceNormals[3 * t];
    const fny = faceNormals[3 * t + 1];
    const fnz = faceNormals[3 * t + 2];
    for (let c = 0; c < 3; c++) {
      const vi = indices[3 * t + c];
      const base = (3 * t + c) * 3;
      outPos[base] = positions[3 * vi];
      outPos[base + 1] = positions[3 * vi + 1];
      outPos[base + 2] = positions[3 * vi + 2];

      let sx = 0;
      let sy = 0;
      let sz = 0;
      const adj = vi < vCount ? vertTris[vi] : [];
      for (let k = 0; k < adj.length; k++) {
        const ot = adj[k];
        const onx = faceNormals[3 * ot];
        const ony = faceNormals[3 * ot + 1];
        const onz = faceNormals[3 * ot + 2];
        const dot = fnx * onx + fny * ony + fnz * onz;
        if (dot >= cosThreshold - 1e-6) {
          sx += onx;
          sy += ony;
          sz += onz;
        }
      }
      const nLen = Math.hypot(sx, sy, sz);
      if (nLen > 1e-12) {
        outNorm[base] = sx / nLen;
        outNorm[base + 1] = sy / nLen;
        outNorm[base + 2] = sz / nLen;
      } else {
        outNorm[base] = fnx;
        outNorm[base + 1] = fny;
        outNorm[base + 2] = fnz;
      }
    }
  }
  return { positions: outPos, normals: outNorm };
}

function removeAutoCreaseGroup(object: THREE.Mesh): void {
  const existing = object.getObjectByName(AUTO_CREASE_GROUP);
  if (existing === undefined) {
    return;
  }
  object.remove(existing);
  existing.traverse((child) => {
    const meshOrLine = child as THREE.Mesh | THREE.LineSegments;
    meshOrLine.geometry?.dispose?.();
    const mat = meshOrLine.material;
    if (Array.isArray(mat)) {
      for (const m of mat) m.dispose();
    } else {
      mat?.dispose?.();
    }
  });
}

function buildAutoCreaseOverlay(
  srcGeom: THREE.BufferGeometry,
  baseColor: THREE.Color,
  creaseAngleDeg: number,
): THREE.Group | null {
  const posAttr = srcGeom.getAttribute("position") as THREE.BufferAttribute | undefined;
  const idxAttr = srcGeom.getIndex();
  if (posAttr === undefined || idxAttr === null || idxAttr.count < 3) {
    return null;
  }
  const group = new THREE.Group();
  group.name = AUTO_CREASE_GROUP;
  group.matrixAutoUpdate = false;

  const triCount = Math.floor(idxAttr.count / 3);
  if (triCount <= MAX_CREASE_OVERLAY_TRIS) {
    const creased = computeCreasedGeometry(
      posAttr.array as Float32Array,
      idxAttr.array as Uint32Array | Uint16Array,
      creaseAngleDeg,
    );
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(creased.positions, 3));
    geom.setAttribute("normal", new THREE.BufferAttribute(creased.normals, 3));
    const mat = new THREE.MeshStandardMaterial({
      color: baseColor.clone(),
      roughness: 0.55,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const surface = new THREE.Mesh(geom, mat);
    surface.matrixAutoUpdate = false;
    surface.renderOrder = 1;
    group.add(surface);
  }

  const edgesGeom = new THREE.EdgesGeometry(srcGeom, creaseAngleDeg);
  const edgesMat = new THREE.LineBasicMaterial({
    color: 0x161a20,
    transparent: true,
    opacity: 0.68,
    depthTest: true,
  });
  const lines = new THREE.LineSegments(edgesGeom, edgesMat);
  lines.matrixAutoUpdate = false;
  lines.renderOrder = 2;
  group.add(lines);
  return group;
}

export interface AppliedShadingState {
  mode: DiagnosticShadingMode;
  creaseAngleDeg: number;
  triCount: number;
}

/**
 * Applies `mode` (plus `globalWireframe` and `creaseAngleDeg`) onto a bound `<FastMesh>`,
 * returning an updated state token to track incremental auto-crease rebuilds cleanly.
 */
export function syncMeshDiagnosticShading(params: {
  bound: MeshBinding;
  material: THREE.MeshStandardMaterial;
  mode: DiagnosticShadingMode;
  globalWireframe: boolean;
  creaseAngleDeg: number;
  prev: AppliedShadingState | null;
}): AppliedShadingState {
  const { bound, material, mode, globalWireframe, creaseAngleDeg, prev } = params;
  const uniforms = installDiagnosticShader(material);
  uniforms.fmDiagMode.value = mode === "matcap" ? 1 : mode === "backfaceWarning" ? 2 : 0;

  const wantFlat = mode === "flat";
  if (material.flatShading !== wantFlat) {
    material.flatShading = wantFlat;
    material.needsUpdate = true;
  }

  setWireOverlay(bound.object, globalWireframe || mode === "wireframeShaded");

  const idxCount = bound.object.geometry.getIndex()?.count ?? 0;
  const triCount = Math.floor(idxCount / 3);
  if (mode === "autoCrease") {
    const needsRebuild =
      prev === null ||
      prev.mode !== "autoCrease" ||
      prev.creaseAngleDeg !== creaseAngleDeg ||
      prev.triCount !== triCount ||
      bound.object.getObjectByName(AUTO_CREASE_GROUP) === undefined;
    if (needsRebuild) {
      removeAutoCreaseGroup(bound.object);
      const overlay = buildAutoCreaseOverlay(bound.object.geometry, material.color, creaseAngleDeg);
      if (overlay !== null) {
        bound.object.add(overlay);
      }
    }
  } else if (bound.object.getObjectByName(AUTO_CREASE_GROUP) !== undefined) {
    removeAutoCreaseGroup(bound.object);
  }

  return { mode, creaseAngleDeg, triCount };
}

/** Removes any auto-crease overlay children cleanly on unmount. */
export function clearMeshDiagnosticShading(bound: MeshBinding, material: THREE.MeshStandardMaterial): void {
  removeAutoCreaseGroup(bound.object);
  setWireOverlay(bound.object, false);
  if (material.flatShading) {
    material.flatShading = false;
    material.needsUpdate = true;
  }
  const uniforms = material.userData.__diagUniforms as DiagUniforms | undefined;
  if (uniforms !== undefined) {
    uniforms.fmDiagMode.value = 0;
  }
}
