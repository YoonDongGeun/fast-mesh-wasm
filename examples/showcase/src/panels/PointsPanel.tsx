// Point cloud import, sampling, filtering, normal estimation & surface
// reconstruction panel (#234).
import {
  type MeshHandle,
  type PackedPointCloud,
  type SurfaceReconstructMethod,
  type Vec3,
  packPointCloud,
  parsePointCloudBytes,
  parsePointCloudText,
  pointCloudEstimateNormals,
  pointCloudRor,
  pointCloudSor,
  pointCloudVoxelDownsample,
  unpackPointCloud,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState, useSyncExternalStore } from "react";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { type Wave6MetrologyState } from "../scene/wave6-state";
import { LengthInput } from "./LengthInput";
import { formatPoint } from "./units";

export interface PointsPanelProps {
  app: AppState;
  store: SceneStore;
}

export type ReconstructUiMethod = "poisson" | "bpa";

const INITIAL_METROLOGY: Wave6MetrologyState = {
  fittedPrimitive: null,
  showPrimitiveOverlay: true,
  contourSlices: null,
  showContoursOverlay: true,
  pointCloud: null,
  showPointCloudOverlay: true,
  pointCloudReport: null,
};

const RECONSTRUCT_METHOD_MAP: Record<ReconstructUiMethod, SurfaceReconstructMethod> = {
  poisson: "implicitMlsMarchingCubes",
  bpa: "ballPivoting",
};

/** Computes axis-aligned bounds of a packed point cloud, or `null` when empty. */
export function computePointCloudBounds(cloud: PackedPointCloud): { min: Vec3; max: Vec3 } | null {
  if (cloud.count === 0) {
    return null;
  }
  const { positions } = unpackPointCloud(cloud);
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
  return packPointCloud(positions, normals);
}

/** Normalizes `.pts` text into `x y z` / `x y z r g b` rows before calling `parsePointCloudText`. */
function normalizePtsText(raw: string): string {
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
export function parsePointCloudFileContent(fileName: string, bytes: Uint8Array): PackedPointCloud {
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

  return packPointCloud(
    Float32Array.from(outPos),
    Float32Array.from(outNorm),
    outCol === undefined ? undefined : Float32Array.from(outCol),
  );
}

/** Point Cloud import, filtering, normal estimation & surface reconstruction panel. */
export function PointsPanel({ app, store }: PointsPanelProps) {
  const items = useApp(app, (s) => s.items);
  const unit = useApp(app, (s) => s.units);
  const metrology = useApp(app, (s) => s.wave6?.metrology ?? INITIAL_METROLOGY);
  const selection = useSyncExternalStore(
    store.subscribe,
    () => store.getSnapshot().selection,
    () => null,
  );

  const activeItem: Item | null = items.find((i) => i.mesh === selection) ?? items[0] ?? null;
  const cloud = metrology.pointCloud;
  const bounds = cloud === null ? null : computePointCloudBounds(cloud);

  const [sorK, setSorK] = useState(12);
  const [sorStdRatio, setSorStdRatio] = useState(2.0);
  const [rorRadius, setRorRadius] = useState(0.5);
  const [rorMinNeighbors, setRorMinNeighbors] = useState(4);
  const [voxelSize, setVoxelSize] = useState(0.15);
  const [normalsK, setNormalsK] = useState(12);
  const [reconstructMethod, setReconstructMethod] = useState<ReconstructUiMethod>("poisson");
  const [reconstructVoxelSize, setReconstructVoxelSize] = useState(0.25);
  const [reconstructBallRadius, setReconstructBallRadius] = useState(0.35);
  const [reconstructK, setReconstructK] = useState(12);
  const [reconstructName, setReconstructName] = useState("Reconstructed Surface");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canReconstruct = typeof store?.scene?.inspect?.reconstructPointCloud === "function";

  const commitCloud = (nextCloud: PackedPointCloud | null, report: string | null) => {
    app.updateWave6((prev) => ({
      ...prev,
      metrology: {
        ...prev.metrology,
        pointCloud: nextCloud,
        showPointCloudOverlay: nextCloud !== null ? true : prev.metrology.showPointCloudOverlay,
        pointCloudReport: report,
      },
    }));
  };

  const onFileSelected = async (file: File | undefined) => {
    if (file === undefined) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const parsed = parsePointCloudFileContent(file.name, bytes);
      commitCloud(parsed, `Loaded ${parsed.count.toLocaleString()} points from ${file.name}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onSampleActiveMesh = async () => {
    if (activeItem === null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const maybeAsyncCustom = (
        store.scene.inspect as { meshToPointCloud?: (m: MeshHandle) => PackedPointCloud | Promise<PackedPointCloud> } | undefined
      )?.meshToPointCloud;
      const sampled =
        typeof maybeAsyncCustom === "function"
          ? await Promise.resolve(maybeAsyncCustom(activeItem.mesh))
          : sampleMeshToPointCloud(store.scene, activeItem.mesh);
      commitCloud(sampled, `Sampled ${sampled.count.toLocaleString()} points from ${activeItem.name}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onSampleSynthetic = () => {
    setError(null);
    const synthetic = createSyntheticSphereCloud(256, 1);
    commitCloud(synthetic, `Generated synthetic sphere cloud (${synthetic.count} points)`);
  };

  const onFilterSor = () => {
    if (cloud === null || cloud.count === 0) {
      return;
    }
    setError(null);
    try {
      const k = Math.max(1, Math.min(cloud.count - 1, Math.round(sorK)));
      const res = pointCloudSor(cloud, k, Math.max(0.01, sorStdRatio));
      commitCloud(
        res.cloud,
        `SOR kept ${res.report.keptCount} / ${cloud.count} (removed ${res.report.removedCount})`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onFilterRor = () => {
    if (cloud === null || cloud.count === 0) {
      return;
    }
    setError(null);
    try {
      const res = pointCloudRor(cloud, Math.max(1e-4, rorRadius), Math.max(0, Math.round(rorMinNeighbors)));
      commitCloud(
        res.cloud,
        `ROR kept ${res.report.keptCount} / ${cloud.count} (removed ${res.report.removedCount})`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onVoxelDownsample = () => {
    if (cloud === null || cloud.count === 0) {
      return;
    }
    setError(null);
    try {
      const next = pointCloudVoxelDownsample(cloud, Math.max(1e-4, voxelSize));
      commitCloud(next, `Voxel downsampled ${cloud.count} -> ${next.count} points`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onEstimateNormals = () => {
    if (cloud === null || cloud.count < 3) {
      return;
    }
    setError(null);
    try {
      const k = Math.max(3, Math.min(cloud.count - 1, Math.round(normalsK)));
      const next = pointCloudEstimateNormals(cloud, k);
      commitCloud(next, `Estimated normals (k=${k}) on ${next.count} points`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onReconstruct = async () => {
    if (cloud === null || cloud.count === 0 || !store.scene.inspect?.reconstructPointCloud) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const k = Math.max(3, Math.min(Math.max(3, cloud.count - 1), Math.round(reconstructK)));
      const workingCloud = cloud.hasNormals ? cloud : pointCloudEstimateNormals(cloud, k);
      const sdkMethod = RECONSTRUCT_METHOD_MAP[reconstructMethod];
      const handle = await Promise.resolve(
        store.scene.inspect.reconstructPointCloud(workingCloud, {
          method: sdkMethod,
          voxelSize: Math.max(1e-4, reconstructVoxelSize),
          ballRadius: Math.max(1e-4, reconstructBallRadius),
          kNeighbors: k,
          normals: true,
        }),
      );
      const label = reconstructName.trim() || `Reconstructed (${reconstructMethod})`;
      const item = app.adoptSceneMesh(store, handle, label);
      store.select(item.mesh);
      store.commitFrame();
      commitCloud(
        workingCloud,
        `Reconstructed ${item.name} (${item.triangles.toLocaleString()} tris via ${reconstructMethod})`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel" aria-label="Points" data-points-panel>
      <div className="op">
        <div className="op-head">
          <strong>Point Cloud Source</strong>
          {cloud !== null ? (
            <button
              type="button"
              data-action="points-clear"
              disabled={busy}
              onClick={() => commitCloud(null, null)}
            >
              Clear
            </button>
          ) : null}
        </div>

        <label className="field">
          import (.xyz, .ply, .pts, .pcd)
          <input
            type="file"
            accept=".xyz,.ply,.pts,.pcd"
            aria-label="Load point cloud file"
            data-points-file-input
            disabled={busy}
            onChange={(e) => {
              const file = e.currentTarget.files?.[0];
              e.currentTarget.value = "";
              void onFileSelected(file);
            }}
          />
        </label>

        <div className="op-row">
          <button
            type="button"
            data-action="points-sample-mesh"
            disabled={busy || activeItem === null}
            onClick={() => void onSampleActiveMesh()}
          >
            Sample Active Mesh
          </button>
          <button
            type="button"
            data-action="points-sample-synthetic"
            disabled={busy}
            onClick={onSampleSynthetic}
          >
            Sample Synthetic Cloud
          </button>
        </div>

        <label className="check">
          <input
            type="checkbox"
            data-points-overlay
            checked={metrology.showPointCloudOverlay}
            onChange={(e) => {
              const checked = e.currentTarget.checked;
              app.updateWave6((prev) => ({
                ...prev,
                metrology: { ...prev.metrology, showPointCloudOverlay: checked },
              }));
            }}
          />
          Show point cloud overlay
        </label>

        {cloud !== null ? (
          <div className="readout" data-points-stats>
            <p>
              Points: <strong>{cloud.count.toLocaleString()}</strong>
            </p>
            <p>
              Normals: {cloud.hasNormals ? "Yes" : "No"} · Colors: {cloud.hasColors ? "Yes" : "No"}
            </p>
            {bounds !== null ? (
              <>
                <p>Min: {formatPoint(bounds.min, unit)}</p>
                <p>Max: {formatPoint(bounds.max, unit)}</p>
              </>
            ) : null}
          </div>
        ) : (
          <p className="muted small">Load a cloud file or sample a mesh / synthetic sphere.</p>
        )}

        {metrology.pointCloudReport !== null ? (
          <p className="muted small" data-points-report>
            {metrology.pointCloudReport}
          </p>
        ) : null}
      </div>

      <div className="op">
        <div className="op-head">
          <strong>Filter &amp; Normals</strong>
        </div>

        <div className="op-row">
          <label className="field">
            SOR k
            <input
              type="number"
              aria-label="SOR k neighbors"
              data-points-sor-k
              min={1}
              step={1}
              value={sorK}
              onChange={(e) => setSorK(Math.max(1, Math.round(Number(e.currentTarget.value) || 1)))}
            />
          </label>
          <label className="field">
            std ratio
            <input
              type="number"
              aria-label="SOR std ratio"
              data-points-sor-ratio
              min={0.1}
              step={0.1}
              value={sorStdRatio}
              onChange={(e) => setSorStdRatio(Math.max(0.01, Number(e.currentTarget.value) || 1))}
            />
          </label>
          <button
            type="button"
            data-action="points-filter-sor"
            disabled={busy || cloud === null || cloud.count < 2}
            onClick={onFilterSor}
          >
            SOR
          </button>
        </div>

        <div className="op-row">
          <label className="field">
            ROR r ({unit})
            <LengthInput
              aria-label="ROR radius"
              data-points-ror-radius
              min={0.001}
              mm={rorRadius}
              unit={unit}
              onMm={(mm) => setRorRadius(Math.max(0.001, mm))}
            />
          </label>
          <label className="field">
            min nbrs
            <input
              type="number"
              aria-label="ROR min neighbors"
              data-points-ror-min
              min={0}
              step={1}
              value={rorMinNeighbors}
              onChange={(e) => setRorMinNeighbors(Math.max(0, Math.round(Number(e.currentTarget.value) || 0)))}
            />
          </label>
          <button
            type="button"
            data-action="points-filter-ror"
            disabled={busy || cloud === null || cloud.count === 0}
            onClick={onFilterRor}
          >
            ROR
          </button>
        </div>

        <div className="op-row">
          <label className="field">
            voxel ({unit})
            <LengthInput
              aria-label="Voxel size"
              data-points-voxel-size
              min={0.001}
              mm={voxelSize}
              unit={unit}
              onMm={(mm) => setVoxelSize(Math.max(0.001, mm))}
            />
          </label>
          <button
            type="button"
            data-action="points-voxel-downsample"
            disabled={busy || cloud === null || cloud.count === 0}
            onClick={onVoxelDownsample}
          >
            Voxel Downsample
          </button>
        </div>

        <div className="op-row">
          <label className="field">
            normal k
            <input
              type="number"
              aria-label="Normals k neighbors"
              data-points-normals-k
              min={3}
              step={1}
              value={normalsK}
              onChange={(e) => setNormalsK(Math.max(3, Math.round(Number(e.currentTarget.value) || 3)))}
            />
          </label>
          <button
            type="button"
            data-action="points-estimate-normals"
            disabled={busy || cloud === null || cloud.count < 3}
            onClick={onEstimateNormals}
          >
            Estimate Normals
          </button>
        </div>
      </div>

      <div className="op">
        <div className="op-head">
          <strong>Reconstruct Surface</strong>
        </div>

        <div className="op-row">
          <label className="field">
            method
            <select
              aria-label="Reconstruction method"
              data-points-reconstruct-method
              value={reconstructMethod}
              disabled={busy}
              onChange={(e) => setReconstructMethod(e.currentTarget.value as ReconstructUiMethod)}
            >
              <option value="poisson">Poisson / Implicit MLS</option>
              <option value="bpa">Ball Pivoting (BPA)</option>
            </select>
          </label>
          <label className="field">
            name
            <input
              type="text"
              aria-label="Reconstructed mesh name"
              data-points-reconstruct-name
              value={reconstructName}
              onChange={(e) => setReconstructName(e.currentTarget.value)}
            />
          </label>
        </div>

        <div className="op-row">
          {reconstructMethod === "poisson" ? (
            <label className="field">
              voxel ({unit})
              <LengthInput
                aria-label="Reconstruction voxel size"
                data-points-reconstruct-voxel
                min={0.001}
                mm={reconstructVoxelSize}
                unit={unit}
                onMm={(mm) => setReconstructVoxelSize(Math.max(0.001, mm))}
              />
            </label>
          ) : (
            <label className="field">
              ball radius ({unit})
              <LengthInput
                aria-label="BPA ball radius"
                data-points-reconstruct-radius
                min={0.001}
                mm={reconstructBallRadius}
                unit={unit}
                onMm={(mm) => setReconstructBallRadius(Math.max(0.001, mm))}
              />
            </label>
          )}
          <label className="field">
            k
            <input
              type="number"
              aria-label="Reconstruction k neighbors"
              data-points-reconstruct-k
              min={3}
              step={1}
              value={reconstructK}
              onChange={(e) => setReconstructK(Math.max(3, Math.round(Number(e.currentTarget.value) || 3)))}
            />
          </label>
          <button
            type="button"
            data-action="points-reconstruct"
            disabled={busy || cloud === null || cloud.count < 4 || !canReconstruct}
            onClick={() => void onReconstruct()}
          >
            {busy ? "Working..." : "Reconstruct"}
          </button>
        </div>
      </div>

      {error !== null ? (
        <p className="report error small" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}
