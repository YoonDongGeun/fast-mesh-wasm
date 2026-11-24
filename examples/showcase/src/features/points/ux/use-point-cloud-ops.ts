// The points panel's op sequences (#234): load a cloud from a file, the
// active mesh or a synthetic sphere; filter it (statistical and radius
// outlier removal, voxel downsampling); estimate normals; and reconstruct
// a surface mesh that joins the scene. Every result replaces the cloud in
// the app's metrology state with a one-line report.
import type { MeshHandle, PackedPointCloud, SurfaceReconstructMethod } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState } from "react";
import type { AppState, Item } from "../../../app/state/app-state";
import { ensureWave6Wasm } from "../../../engine/wave6-wasm-lazy";
import { createSyntheticSphereCloud, parsePointCloudFileContent, sampleMeshToPointCloud } from "./point-cloud";

export type ReconstructUiMethod = "poisson" | "bpa";

const RECONSTRUCT_METHOD_MAP: Record<ReconstructUiMethod, SurfaceReconstructMethod> = {
  poisson: "implicitMlsMarchingCubes",
  bpa: "ballPivoting",
};

/**
 * The points panel's parameters, busy and error state, and op handlers.
 * The parameter setters clamp what was typed (counts round to integers,
 * lengths stay positive) the way the inputs always have.
 */
export function usePointCloudOps(
  app: AppState,
  store: SceneStore,
  activeItem: Item | null,
  cloud: PackedPointCloud | null,
) {
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
      const parsed = await parsePointCloudFileContent(file.name, bytes);
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

  const onFilterSor = async () => {
    if (cloud === null || cloud.count === 0) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { pointCloudSor } = await ensureWave6Wasm();
      const k = Math.max(1, Math.min(cloud.count - 1, Math.round(sorK)));
      const res = pointCloudSor(cloud, k, Math.max(0.01, sorStdRatio));
      commitCloud(
        res.cloud,
        `SOR kept ${res.report.keptCount} / ${cloud.count} (removed ${res.report.removedCount})`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onFilterRor = async () => {
    if (cloud === null || cloud.count === 0) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { pointCloudRor } = await ensureWave6Wasm();
      const res = pointCloudRor(cloud, Math.max(1e-4, rorRadius), Math.max(0, Math.round(rorMinNeighbors)));
      commitCloud(
        res.cloud,
        `ROR kept ${res.report.keptCount} / ${cloud.count} (removed ${res.report.removedCount})`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onVoxelDownsample = async () => {
    if (cloud === null || cloud.count === 0) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { pointCloudVoxelDownsample } = await ensureWave6Wasm();
      const next = pointCloudVoxelDownsample(cloud, Math.max(1e-4, voxelSize));
      commitCloud(next, `Voxel downsampled ${cloud.count} -> ${next.count} points`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onEstimateNormals = async () => {
    if (cloud === null || cloud.count < 3) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { pointCloudEstimateNormals } = await ensureWave6Wasm();
      const k = Math.max(3, Math.min(cloud.count - 1, Math.round(normalsK)));
      const next = pointCloudEstimateNormals(cloud, k);
      commitCloud(next, `Estimated normals (k=${k}) on ${next.count} points`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
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
      const workingCloud = cloud.hasNormals ? cloud : (await ensureWave6Wasm()).pointCloudEstimateNormals(cloud, k);
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

  /** Drops the cloud and its report. */
  const onClear = () => commitCloud(null, null);

  const setShowOverlay = (checked: boolean) => {
    app.updateWave6((prev) => ({
      ...prev,
      metrology: { ...prev.metrology, showPointCloudOverlay: checked },
    }));
  };

  return {
    sorK,
    setSorK: (n: number) => setSorK(Math.max(1, Math.round(n || 1))),
    sorStdRatio,
    setSorStdRatio: (n: number) => setSorStdRatio(Math.max(0.01, n || 1)),
    rorRadius,
    setRorRadius: (mm: number) => setRorRadius(Math.max(0.001, mm)),
    rorMinNeighbors,
    setRorMinNeighbors: (n: number) => setRorMinNeighbors(Math.max(0, Math.round(n || 0))),
    voxelSize,
    setVoxelSize: (mm: number) => setVoxelSize(Math.max(0.001, mm)),
    normalsK,
    setNormalsK: (n: number) => setNormalsK(Math.max(3, Math.round(n || 3))),
    reconstructMethod,
    setReconstructMethod,
    reconstructVoxelSize,
    setReconstructVoxelSize: (mm: number) => setReconstructVoxelSize(Math.max(0.001, mm)),
    reconstructBallRadius,
    setReconstructBallRadius: (mm: number) => setReconstructBallRadius(Math.max(0.001, mm)),
    reconstructK,
    setReconstructK: (n: number) => setReconstructK(Math.max(3, Math.round(n || 3))),
    reconstructName,
    setReconstructName,
    busy,
    error,
    canReconstruct,
    onFileSelected,
    onSampleActiveMesh,
    onSampleSynthetic,
    onFilterSor,
    onFilterRor,
    onVoxelDownsample,
    onEstimateNormals,
    onReconstruct,
    onClear,
    setShowOverlay,
  };
}
