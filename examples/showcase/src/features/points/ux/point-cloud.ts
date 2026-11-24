// Point cloud sources and stats (#234): bounds, a synthetic Fibonacci
// sphere, `.xyz`/`.ply`/`.pts`/`.pcd` parsing through the SDK parsers, and
// sampling a scene mesh's world-space surface with outward normals.
import type { MeshHandle, PackedPointCloud, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { ensureWave6Wasm, packPointCloudLocal, unpackPointCloudLocal } from "../../../engine/wave6-wasm-lazy";

/** Computes axis-aligned bounds of a packed point cloud, or `null` when empty. */
export function computePointCloudBounds(cloud: PackedPointCloud): { min: Vec3; max: Vec3 } | null {
  if (cloud.count === 0) {
    return null;
  }
  const { positions } = unpackPointCloudLocal(cloud);
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
  };
}

/** Generates an evenly distributed Fibonacci sphere point cloud with unit outward normals. */
export function createSyntheticSphereCloud(count = 256, radius = 1): PackedPointCloud {
  const n = Math.max(12, Math.round(count));
  const r = Math.max(1e-3, radius);
  const positions = new Float32Array(n * 3);
  const normals = new Float32Array(n * 3);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - ((i + 0.5) / n) * 2;
    const radial = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = goldenAngle * i;
    const nx = Math.cos(theta) * radial;
    const nz = Math.sin(theta) * radial;
    positions[3 * i] = nx * r;
    positions[3 * i + 1] = y * r;
    positions[3 * i + 2] = nz * r;
    normals[3 * i] = nx;
    normals[3 * i + 1] = y;
    normals[3 * i + 2] = nz;
  }
  return packPointCloudLocal(positions, normals);
}

/** Normalizes `.pts` text into `x y z` / `x y z r g b` rows before calling `parsePointCloudText`. */
export function normalizePtsText(raw: string): string {
  const lines = raw.split(/\r?\n/);
  const rows: string[] = [];
  let sawFirstDataOrHeader = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("//")) {
      continue;
    }
    const cols = trimmed.split(/\s+/);
    if (!sawFirstDataOrHeader) {
      sawFirstDataOrHeader = true;
      if (cols.length === 1 && /^\d+$/.test(cols[0])) {
        continue;
      }
    }
    if (cols.length === 4) {
      rows.push(`${cols[0]} ${cols[1]} ${cols[2]}`);
    } else if (cols.length === 7) {
      rows.push(`${cols[0]} ${cols[1]} ${cols[2]} ${cols[4]} ${cols[5]} ${cols[6]}`);
    } else {
      rows.push(cols.join(" "));
    }
  }
  return rows.join("\n");
}

/** Parses `.xyz`, `.ply`, `.pts`, or `.pcd` file bytes using SDK parsers. */
export async function parsePointCloudFileContent(fileName: string, bytes: Uint8Array): Promise<PackedPointCloud> {
  const { parsePointCloudBytes, parsePointCloudText } = await ensureWave6Wasm();
  const lower = fileName.toLowerCase();
  if (lower.endsWith(".ply")) {
    return parsePointCloudBytes(bytes);
  }
  if (lower.endsWith(".pcd")) {
    const head = new TextDecoder("utf-8", { fatal: false }).decode(bytes.subarray(0, Math.min(bytes.length, 1024)));
    if (/\bDATA\s+ascii\b/i.test(head)) {
      return parsePointCloudText(new TextDecoder().decode(bytes), "pcd");
    }
    return parsePointCloudBytes(bytes);
  }
  const text = new TextDecoder().decode(bytes);
  if (lower.endsWith(".pts")) {
    return parsePointCloudText(normalizePtsText(text), "xyz");
  }
  return parsePointCloudText(text, "xyz");
}

function transformPoint(m: ArrayLike<number>, x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

/**
 * Samples world-space vertices (and triangle surface subdivisions for coarse
 * meshes) with outward normals from an active scene mesh into a `PackedPointCloud`.
 */
export function sampleMeshToPointCloud(
  scene: SceneStore["scene"],
  mesh: MeshHandle,
  minSamples = 256,
): PackedPointCloud {
  const maybeCustom = (
    scene.inspect as { meshToPointCloud?: (m: MeshHandle) => PackedPointCloud } | undefined
  )?.meshToPointCloud;
  if (typeof maybeCustom === "function") {
    return maybeCustom(mesh);
  }

  const views = scene.views(mesh);
  const localPos = views.positions();
  const idx = views.indices();
  const vColors = views.colors();
  const m = scene.matrix4(mesh);

  const vertCount = Math.floor(localPos.length / 3);
  const worldPos = new Float32Array(vertCount * 3);
  for (let i = 0; i < vertCount; i++) {
    const [wx, wy, wz] = transformPoint(m, localPos[3 * i], localPos[3 * i + 1], localPos[3 * i + 2]);
    worldPos[3 * i] = wx;
    worldPos[3 * i + 1] = wy;
    worldPos[3 * i + 2] = wz;
  }

  const vertNorm = new Float32Array(vertCount * 3);
  const triCount = Math.floor(idx.length / 3);
  for (let t = 0; t < triCount; t++) {
    const a = idx[3 * t];
    const b = idx[3 * t + 1];
    const c = idx[3 * t + 2];
    const ax = worldPos[3 * a];
    const ay = worldPos[3 * a + 1];
    const az = worldPos[3 * a + 2];
    const abx = worldPos[3 * b] - ax;
    const aby = worldPos[3 * b + 1] - ay;
    const abz = worldPos[3 * b + 2] - az;
    const acx = worldPos[3 * c] - ax;
    const acy = worldPos[3 * c + 1] - ay;
    const acz = worldPos[3 * c + 2] - az;
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    for (const v of [a, b, c]) {
      vertNorm[3 * v] += nx;
      vertNorm[3 * v + 1] += ny;
      vertNorm[3 * v + 2] += nz;
    }
  }
  for (let i = 0; i < vertCount; i++) {
    const len = Math.hypot(vertNorm[3 * i], vertNorm[3 * i + 1], vertNorm[3 * i + 2]);
    if (len > 1e-12) {
      vertNorm[3 * i] /= len;
      vertNorm[3 * i + 1] /= len;
      vertNorm[3 * i + 2] /= len;
    } else {
      vertNorm[3 * i + 2] = 1;
    }
  }

  const extraSteps = triCount > 0 && vertCount < minSamples
    ? Math.max(1, Math.min(8, Math.ceil(Math.sqrt((minSamples - vertCount) / Math.max(1, triCount)))))
    : 0;

  const outPos: number[] = [];
  const outNorm: number[] = [];
  const outCol: number[] | undefined = vColors !== null && vColors.length >= vertCount * 4 ? [] : undefined;

  for (let i = 0; i < vertCount; i++) {
    outPos.push(worldPos[3 * i], worldPos[3 * i + 1], worldPos[3 * i + 2]);
    outNorm.push(vertNorm[3 * i], vertNorm[3 * i + 1], vertNorm[3 * i + 2]);
    if (outCol !== undefined && vColors !== null) {
      outCol.push(vColors[4 * i], vColors[4 * i + 1], vColors[4 * i + 2]);
    }
  }

  if (extraSteps > 0) {
    for (let t = 0; t < triCount; t++) {
      const a = idx[3 * t];
      const b = idx[3 * t + 1];
      const c = idx[3 * t + 2];
      const ax = worldPos[3 * a];
      const ay = worldPos[3 * a + 1];
      const az = worldPos[3 * a + 2];
      const bx = worldPos[3 * b];
      const by = worldPos[3 * b + 1];
      const bz = worldPos[3 * b + 2];
      const cx = worldPos[3 * c];
      const cy = worldPos[3 * c + 1];
      const cz = worldPos[3 * c + 2];
      let fnx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
      let fny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
      let fnz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      const flen = Math.hypot(fnx, fny, fnz) || 1;
      fnx /= flen;
      fny /= flen;
      fnz /= flen;
      for (let iu = 1; iu <= extraSteps; iu++) {
        for (let iv = 1; iv <= extraSteps + 1 - iu; iv++) {
          const w0 = iu / (extraSteps + 2);
          const w1 = iv / (extraSteps + 2);
          const w2 = 1 - w0 - w1;
          outPos.push(w0 * ax + w1 * bx + w2 * cx, w0 * ay + w1 * by + w2 * cy, w0 * az + w1 * bz + w2 * cz);
          outNorm.push(fnx, fny, fnz);
          if (outCol !== undefined && vColors !== null) {
            outCol.push(
              w0 * vColors[4 * a] + w1 * vColors[4 * b] + w2 * vColors[4 * c],
              w0 * vColors[4 * a + 1] + w1 * vColors[4 * b + 1] + w2 * vColors[4 * c + 1],
              w0 * vColors[4 * a + 2] + w1 * vColors[4 * b + 2] + w2 * vColors[4 * c + 2],
            );
          }
        }
      }
    }
  }

  return packPointCloudLocal(
    Float32Array.from(outPos),
    Float32Array.from(outNorm),
    outCol === undefined ? undefined : Float32Array.from(outCol),
  );
}
