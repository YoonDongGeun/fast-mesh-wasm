// Point cloud import, sampling, filtering, normal estimation & surface
// reconstruction panel (#234).
import type { SceneStore } from "fast-mesh-wasm/store";
import { useSyncExternalStore } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { type Wave6MetrologyState } from "../../../app/state/wave6-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { formatPoint } from "../../../shared/ux/units";
import { computePointCloudBounds } from "../ux/point-cloud";
import { type ReconstructUiMethod, usePointCloudOps } from "../ux/use-point-cloud-ops";

export interface PointsPanelProps {
  app: AppState;
  store: SceneStore;
}

const INITIAL_METROLOGY: Wave6MetrologyState = {
  fittedPrimitive: null,
  showPrimitiveOverlay: true,
  contourSlices: null,
  showContoursOverlay: true,
  pointCloud: null,
  showPointCloudOverlay: true,
  pointCloudReport: null,
};

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

  const ops = usePointCloudOps(app, store, activeItem, cloud);
  const {
    sorK,
    sorStdRatio,
    rorRadius,
    rorMinNeighbors,
    voxelSize,
    normalsK,
    reconstructMethod,
    reconstructVoxelSize,
    reconstructBallRadius,
    reconstructK,
    reconstructName,
    busy,
    error,
    canReconstruct,
  } = ops;

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
              onClick={ops.onClear}
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
              void ops.onFileSelected(file);
            }}
          />
        </label>

        <div className="op-row">
          <button
            type="button"
            data-action="points-sample-mesh"
            disabled={busy || activeItem === null}
            onClick={() => void ops.onSampleActiveMesh()}
          >
            Sample Active Mesh
          </button>
          <button
            type="button"
            data-action="points-sample-synthetic"
            disabled={busy}
            onClick={ops.onSampleSynthetic}
          >
            Sample Synthetic Cloud
          </button>
        </div>

        <label className="check">
          <input
            type="checkbox"
            data-points-overlay
            checked={metrology.showPointCloudOverlay}
            onChange={(e) => ops.setShowOverlay(e.currentTarget.checked)}
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
              onChange={(e) => ops.setSorK(Number(e.currentTarget.value))}
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
              onChange={(e) => ops.setSorStdRatio(Number(e.currentTarget.value))}
            />
          </label>
          <button
            type="button"
            data-action="points-filter-sor"
            disabled={busy || cloud === null || cloud.count < 2}
            onClick={() => void ops.onFilterSor()}
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
              onMm={(mm) => ops.setRorRadius(mm)}
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
              onChange={(e) => ops.setRorMinNeighbors(Number(e.currentTarget.value))}
            />
          </label>
          <button
            type="button"
            data-action="points-filter-ror"
            disabled={busy || cloud === null || cloud.count === 0}
            onClick={() => void ops.onFilterRor()}
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
              onMm={(mm) => ops.setVoxelSize(mm)}
            />
          </label>
          <button
            type="button"
            data-action="points-voxel-downsample"
            disabled={busy || cloud === null || cloud.count === 0}
            onClick={() => void ops.onVoxelDownsample()}
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
              onChange={(e) => ops.setNormalsK(Number(e.currentTarget.value))}
            />
          </label>
          <button
            type="button"
            data-action="points-estimate-normals"
            disabled={busy || cloud === null || cloud.count < 3}
            onClick={() => void ops.onEstimateNormals()}
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
              onChange={(e) => ops.setReconstructMethod(e.currentTarget.value as ReconstructUiMethod)}
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
              onChange={(e) => ops.setReconstructName(e.currentTarget.value)}
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
                onMm={(mm) => ops.setReconstructVoxelSize(mm)}
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
                onMm={(mm) => ops.setReconstructBallRadius(mm)}
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
              onChange={(e) => ops.setReconstructK(Number(e.currentTarget.value))}
            />
          </label>
          <button
            type="button"
            data-action="points-reconstruct"
            disabled={busy || cloud === null || cloud.count < 4 || !canReconstruct}
            onClick={() => void ops.onReconstruct()}
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
