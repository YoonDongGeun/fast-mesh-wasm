// Modeling & Resampling panel (#195): Boolean CSG (with Keep originals), destructive Plane Cut,
// Offset & Shell (Solidify), Quadric Simplify, and Loop Subdivide; then the Print Prep sections
// (#236): Hollow Mesh, Drain Hole and Surface Stamp. What they do lives in
// `ux/use-modeling-ops.ts` and `ux/use-print-prep-ops.ts`.
import type { HollowBackend, PlaceUpAxis } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { type AppState, useApp } from "../../../app/state/app-state";
import type { OffsetModeAlgorithm } from "../../../engine/workbench-ops";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { formatLength, lengthInputValue } from "../../../shared/ux/units";
import {
  BOOLEAN_OPERATIONS,
  isVoxelOffset,
  offsetBlocked,
  triangleCountInput,
  useModelingOps,
} from "../ux/use-modeling-ops";
import { spinInput, stampScaleInput, taperInput, usePrintPrepOps } from "../ux/use-print-prep-ops";
import { findCenterSurfaceHit } from "./HollowPlaceToolsLayer";

export function ModelingPanel(props: { app: AppState }) {
  const { app } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const units = useApp(app, (s) => s.units);
  const prep = usePrintPrepOps({ app, store, findCenterHit: findCenterSurfaceHit });

  const selectedItem = selection === null ? null : (items.find((i) => i.mesh === selection) ?? null);
  const activeItem = selectedItem ?? items[0] ?? null;
  const ops = useModelingOps({ app, store, items, activeItem, units });
  const disabled = ops.busy !== null || activeItem === null;

  return (
    <section className="panel modeling-panel" aria-label="Modeling & Resampling" data-panel="modeling">
      <header className="panel-head">
        <h2>Modeling &amp; Resampling</h2>
      </header>

      {items.length === 0 ? (
        <p className="muted">Load or duplicate a mesh to enable CSG, plane cut, offset, shell, and resampling.</p>
      ) : (
        <>
          {/* ---- 1. Boolean CSG ---- */}
          <div className="op">
            <h3>Boolean CSG</h3>
            <div className="picker">
              <span>A</span>
              <select
                aria-label="Boolean target mesh"
                data-modeling-boolean-target
                value={ops.resolvedTarget === null ? "" : String(items.findIndex((i) => i.mesh === ops.resolvedTarget))}
                disabled={ops.busy !== null}
                onChange={(e) => {
                  const idx = Number(e.currentTarget.value);
                  ops.setBoolTarget(items[idx]?.mesh ?? null);
                }}
              >
                {items.map((it, idx) => (
                  <option key={it.id} value={String(idx)}>
                    {it.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="picker">
              <span>B</span>
              <select
                aria-label="Boolean source mesh"
                data-modeling-boolean-source
                value={ops.resolvedSource === null ? "" : String(items.findIndex((i) => i.mesh === ops.resolvedSource))}
                disabled={ops.busy !== null}
                onChange={(e) => {
                  const idx = Number(e.currentTarget.value);
                  ops.setBoolSource(items[idx]?.mesh ?? null);
                }}
              >
                {items.map((it, idx) => (
                  <option key={it.id} value={String(idx)}>
                    {it.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="modes" role="group" aria-label="Boolean operation">
              {BOOLEAN_OPERATIONS.map(([op, label]) => (
                <button
                  key={op}
                  type="button"
                  data-modeling-boolean-op={op}
                  aria-pressed={ops.boolOp === op}
                  disabled={ops.busy !== null}
                  onClick={() => ops.setBoolOp(op)}
                >
                  {label}
                </button>
              ))}
            </div>
            <label className="check">
              <input
                type="checkbox"
                data-modeling-keep-originals
                checked={ops.keepOriginals}
                disabled={ops.busy !== null}
                onChange={(e) => ops.setKeepOriginals(e.currentTarget.checked)}
              />
              Keep originals
            </label>
            <button
              type="button"
              data-modeling-boolean-run
              disabled={ops.busy !== null || ops.targetItem === null || ops.sourceItem === null || items.length < 2}
              onClick={() => ops.runBoolean()}
            >
              Run Boolean
            </button>
          </div>

          {/* ---- 2. Destructive Plane Cut ---- */}
          <div className="op">
            <h3>Plane Cut (Destructive)</h3>
            <p className="muted">Splits geometry destructively on {activeItem?.name ?? "selection"}.</p>
            <div className="modes" role="group" aria-label="Cut plane axis">
              {(["x", "y", "z", "custom"] as const).map((ax) => (
                <button
                  key={ax}
                  type="button"
                  data-modeling-cut-axis={ax}
                  aria-pressed={ops.cutAxis === ax}
                  disabled={disabled}
                  onClick={() => ops.setCutAxis(ax)}
                >
                  {ax === "custom" ? "Custom" : ax.toUpperCase()}
                </button>
              ))}
            </div>
            {ops.cutAxis === "custom" ? (
              <div className="op-row">
                {(["nx", "ny", "nz"] as const).map((label, idx) => (
                  <label key={label} className="field">
                    {label}
                    <input
                      type="number"
                      step="any"
                      aria-label={`Custom cut normal ${label}`}
                      value={ops.cutCustomNormal[idx]}
                      onChange={(e) => ops.setCutNormalComponent(idx, e.currentTarget.value)}
                    />
                  </label>
                ))}
              </div>
            ) : null}
            <label className="slider">
              <div className="slider-head">
                <span>Offset</span>
                <output>{formatLength(ops.cutOffset, units)}</output>
              </div>
              <input
                type="range"
                aria-label="Cut plane offset slider"
                min={ops.cutBounds[0]}
                max={ops.cutBounds[1]}
                step={(ops.cutBounds[1] - ops.cutBounds[0]) / 100 || 0.1}
                value={Math.max(ops.cutBounds[0], Math.min(ops.cutBounds[1], ops.cutOffset))}
                disabled={disabled}
                onChange={(e) => ops.setCutOffset(Number(e.currentTarget.value))}
              />
            </label>
            <div className="op-row">
              <label className="field">
                offset ({units})
                <LengthInput
                  aria-label="Cut plane offset"
                  data-modeling-cut-offset
                  mm={ops.cutOffset}
                  unit={units}
                  onMm={ops.setCutOffset}
                />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  data-modeling-cut-cap
                  checked={ops.cutCap}
                  disabled={disabled}
                  onChange={(e) => ops.setCutCap(e.currentTarget.checked)}
                />
                Cap holes
              </label>
            </div>
            <div className="op-row">
              <span className="muted">Keep side:</span>
              <div className="modes" role="group" aria-label="Keep side">
                {(["+", "-"] as const).map((side) => (
                  <button
                    key={side}
                    type="button"
                    data-modeling-cut-side={side}
                    aria-pressed={ops.cutKeepSide === side}
                    disabled={disabled}
                    onClick={() => ops.setCutKeepSide(side)}
                  >
                    {side} side
                  </button>
                ))}
              </div>
              <button
                type="button"
                data-modeling-cut-run
                disabled={disabled}
                onClick={() => ops.runCut()}
              >
                Cut Mesh
              </button>
            </div>
          </div>

          {/* ---- 3. Offset & Shell ---- */}
          <div className="op">
            <h3>Offset &amp; Shell</h3>
            <div className="op-row">
              <label className="field">
                algorithm
                <select
                  aria-label="Offset algorithm"
                  data-modeling-offset-algorithm
                  value={ops.offsetAlgorithm}
                  disabled={disabled}
                  onChange={(e) => ops.setOffsetAlgorithm(e.currentTarget.value as OffsetModeAlgorithm)}
                >
                  <option value="auto">Auto</option>
                  <option value="normal">Normal</option>
                  <option value="bvhSdf">BVH SDF</option>
                  <option value="fhEdt">FH EDT</option>
                </select>
              </label>
              <label className="field">
                distance ({units})
                <LengthInput
                  aria-label="Offset distance"
                  data-modeling-offset-distance
                  mm={ops.offsetDistance}
                  unit={units}
                  onMm={ops.setOffsetDistance}
                />
              </label>
              {isVoxelOffset(ops.offsetAlgorithm) ? (
                <label className="field">
                  voxel ({units})
                  <input
                    type="number"
                    min={0}
                    step="any"
                    aria-label="Offset voxel size"
                    data-modeling-offset-voxel-size
                    placeholder="auto"
                    value={ops.offsetVoxelSize === "" ? "" : lengthInputValue(Number(ops.offsetVoxelSize), units)}
                    onChange={(e) => ops.setOffsetVoxelInput(e.currentTarget.value)}
                  />
                </label>
              ) : null}
              <button
                type="button"
                data-modeling-offset-run
                disabled={disabled || offsetBlocked(ops.offsetAlgorithm, ops.offsetDistance, ops.offsetVoxelSize)}
                onClick={() => ops.runOffset()}
              >
                {isVoxelOffset(ops.offsetAlgorithm) && ops.offsetDistance === 0 ? "Voxel remesh" : "Offset"}
              </button>
            </div>
            <div className="op-row">
              <label className="field">
                thickness ({units})
                <LengthInput
                  aria-label="Shell wall thickness"
                  data-modeling-shell-thickness
                  min={0}
                  mm={ops.shellThickness}
                  unit={units}
                  onMm={(mm) => ops.setShellThickness(Math.max(0, mm))}
                />
              </label>
              <button
                type="button"
                data-modeling-shell-run
                disabled={disabled || !(ops.shellThickness > 0)}
                onClick={() => ops.runShell()}
              >
                Shell
              </button>
            </div>
          </div>

          {/* ---- 4. Resampling ---- */}
          <div className="op">
            <h3>Resampling</h3>
            <label className="check">
              <input
                type="checkbox"
                data-modeling-lock-boundaries
                checked={ops.lockBoundaries}
                disabled={disabled}
                onChange={(e) => ops.setLockBoundaries(e.currentTarget.checked)}
              />
              Lock boundaries
            </label>
            <div className="op-row">
              <label className="field">
                {ops.simplifyUseRatio ? "ratio (%)" : "target tris"}
                {ops.simplifyUseRatio ? (
                  <input
                    type="number"
                    min={1}
                    max={100}
                    step={5}
                    aria-label="Simplify percentage"
                    data-modeling-simplify-ratio
                    value={ops.simplifyRatio}
                    disabled={disabled}
                    onChange={(e) => ops.setSimplifyPercent(e.currentTarget.value)}
                  />
                ) : (
                  <input
                    type="number"
                    min={1}
                    step={1}
                    aria-label="Simplify target triangles"
                    data-modeling-simplify-target
                    value={ops.simplifyTarget}
                    disabled={disabled}
                    onChange={(e) => ops.setSimplifyTarget(triangleCountInput(e.currentTarget.value))}
                  />
                )}
              </label>
              <button
                type="button"
                data-modeling-simplify-mode
                disabled={disabled}
                onClick={() => ops.setSimplifyUseRatio((v) => !v)}
                title="Switch between triangle percentage ratio and exact count"
              >
                {ops.simplifyUseRatio ? "Count" : "Ratio"}
              </button>
              <button
                type="button"
                data-modeling-simplify-run
                disabled={disabled || ops.liveTris <= 0}
                onClick={() => ops.runSimplify()}
              >
                Simplify ({ops.simplifyCount})
              </button>
            </div>
            <div className="op-row">
              <label className="field">
                levels
                <select
                  aria-label="Subdivide levels"
                  data-modeling-subdivide-levels
                  value={String(ops.subdivideLevels)}
                  disabled={disabled}
                  onChange={(e) => ops.setSubdivideLevels(Number(e.currentTarget.value))}
                >
                  <option value="1">1 (×4 tris)</option>
                  <option value="2">2 (×16 tris)</option>
                  <option value="3">3 (×64 tris)</option>
                </select>
              </label>
              <button
                type="button"
                data-modeling-subdivide-run
                disabled={disabled}
                onClick={() => ops.runSubdivide()}
              >
                Subdivide (Loop)
              </button>
            </div>
          </div>

          <div className="op" data-modeling-hollow-section>
            <span className="muted">5. Print Prep — Hollow Mesh</span>
            <div className="op-row">
              <label className="field">
                <span>Wall</span>
                <LengthInput
                  mm={prep.hollowWallMm}
                  unit={units}
                  data-modeling-hollow-thickness
                  onMm={(mm) => prep.setHollowWallMm(Math.max(0.1, mm))}
                />
              </label>
              <label className="field">
                <span>Backend</span>
                <select
                  value={prep.hollowBackend}
                  data-modeling-hollow-backend
                  onChange={(e) => prep.setHollowBackend(e.target.value as HollowBackend)}
                >
                  <option value="bvhSdf">BVH SDF</option>
                  <option value="fhEdt">Fast EDT</option>
                </select>
              </label>
            </div>
            <div className="op-row">
              <label className="field">
                <span>Voxel (mm)</span>
                <input
                  type="number"
                  min={0.05}
                  step={0.1}
                  placeholder="Auto"
                  value={prep.hollowVoxelSizeText}
                  data-modeling-hollow-voxel-size
                  onChange={(e) => prep.setHollowVoxelSizeText(e.target.value)}
                />
              </label>
              <label className="field">
                <span>Max Voxels</span>
                <select
                  value={prep.hollowMaxVoxels}
                  data-modeling-hollow-max-voxels
                  onChange={(e) => prep.setHollowMaxVoxels(Number(e.target.value))}
                >
                  <option value={262144}>256k</option>
                  <option value={524288}>512k</option>
                  <option value={1048576}>1M</option>
                  <option value={2097152}>2M</option>
                </select>
              </label>
            </div>
            <div className="op-row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={prep.hollowCapInnerShell}
                  data-modeling-hollow-inner-shell
                  onChange={(e) => prep.setHollowCapInnerShell(e.target.checked)}
                />
                <span>Cap inner shell (≤12k tris)</span>
              </label>
              <button
                type="button"
                className="primary"
                disabled={disabled || prep.prepBusy !== null}
                data-modeling-hollow-run
                onClick={() => void prep.runHollow(activeItem)}
              >
                Hollow Mesh
              </button>
            </div>
          </div>

          <div className="op" data-modeling-drain-section>
            <span className="muted">6. Print Prep — Drain Hole</span>
            <div className="op-row">
              <label className="field">
                <span>Radius</span>
                <LengthInput
                  mm={prep.printPrep.drainParams.diameter * 0.5}
                  unit={units}
                  data-modeling-drain-radius
                  onMm={(mm) => prep.updateDrainParams({ diameter: Math.max(0.2, mm * 2) })}
                />
              </label>
              <label className="field">
                <span>Depth</span>
                <LengthInput
                  mm={prep.drainThrough ? prep.drainSavedDepthMm : prep.printPrep.drainParams.depth}
                  unit={units}
                  disabled={prep.drainThrough}
                  data-modeling-drain-depth
                  onMm={prep.setDrainDepth}
                />
              </label>
            </div>
            <div className="op-row">
              <label className="field">
                <span>Chamfer</span>
                <LengthInput
                  mm={prep.printPrep.drainParams.outerChamfer}
                  unit={units}
                  data-modeling-drain-chamfer
                  onMm={(mm) => prep.updateDrainParams({ outerChamfer: Math.max(0, mm) })}
                />
              </label>
              <label className="field">
                <span>Taper (°)</span>
                <input
                  type="number"
                  min={0}
                  max={45}
                  step={1}
                  value={prep.printPrep.drainParams.taperDeg}
                  data-modeling-drain-taper
                  onChange={(e) => prep.updateDrainParams({ taperDeg: taperInput(e.target.value) })}
                />
              </label>
            </div>
            <div className="op-row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={prep.drainThrough}
                  data-modeling-drain-through
                  onChange={(e) => prep.setDrainThrough(e.target.checked)}
                />
                <span>Through wall</span>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={prep.printPrep.drainParams.generatePlug}
                  data-modeling-drain-plug
                  onChange={(e) => prep.updateDrainParams({ generatePlug: e.target.checked })}
                />
                <span>Create plug</span>
              </label>
            </div>
            <div className="op-row">
              <button
                type="button"
                className={prep.printPrep.activeTool === "drainHole" ? "primary" : undefined}
                aria-pressed={prep.printPrep.activeTool === "drainHole"}
                disabled={disabled || prep.prepBusy !== null}
                data-modeling-drain-tool
                onClick={() => prep.toggleTool("drainHole")}
              >
                Pick Surface Point to Punch Hole
              </button>
              <button
                type="button"
                disabled={disabled || prep.prepBusy !== null}
                data-modeling-drain-center
                onClick={() => void prep.punchAtCenter(activeItem)}
              >
                Punch at Center
              </button>
            </div>
          </div>

          <div className="op" data-modeling-place-section>
            <span className="muted">7. Print Prep — Surface Stamp (Place Mesh)</span>
            <div className="op-row">
              <label className="field">
                <span>Stamp</span>
                <select
                  value={prep.printPrep.placeParams.stampSource}
                  data-modeling-place-source
                  onChange={(e) =>
                    prep.updatePlaceParams({
                      stampSource: e.target.value as typeof prep.printPrep.placeParams.stampSource,
                    })
                  }
                >
                  <option value="peg">Peg</option>
                  <option value="hexBoss">Hex Boss</option>
                  <option value="dome">Dome</option>
                  <option value="cube">Cube</option>
                  <option value="sceneMesh">Scene Mesh</option>
                </select>
              </label>
              <label className="field">
                <span>Scene Stamp</span>
                <select
                  value={
                    prep.printPrep.placeParams.stampMeshId === null ? "" : String(prep.printPrep.placeParams.stampMeshId)
                  }
                  data-modeling-place-stamp-mesh
                  onChange={(e) => prep.pickSceneStamp(e.target.value)}
                >
                  <option value="">Built-in preset</option>
                  {items
                    .filter((item) => item.mesh !== activeItem.mesh)
                    .map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                </select>
              </label>
            </div>
            <div className="op-row">
              <label className="field">
                <span>Mode</span>
                <select
                  value={prep.printPrep.placeParams.mode}
                  data-modeling-place-mode
                  onChange={(e) =>
                    prep.updatePlaceParams({
                      mode: e.target.value as typeof prep.printPrep.placeParams.mode,
                    })
                  }
                >
                  <option value="embedUnion">Embed & Union</option>
                  <option value="engraveSubtract">Engrave</option>
                  <option value="keepSeparate">Keep Separate</option>
                </select>
              </label>
              <label className="field">
                <span>Up Axis</span>
                <select
                  value={prep.printPrep.placeParams.upAxis}
                  data-modeling-place-up-axis
                  onChange={(e) =>
                    prep.updatePlaceParams({ upAxis: e.target.value as PlaceUpAxis })
                  }
                >
                  <option value="posZ">+Z (posZ)</option>
                  <option value="negZ">-Z (negZ)</option>
                  <option value="posY">+Y (posY)</option>
                  <option value="negY">-Y (negY)</option>
                </select>
              </label>
            </div>
            <div className="op-row">
              <label className="field">
                <span>Scale</span>
                <input
                  type="number"
                  min={0.05}
                  step={0.1}
                  value={prep.printPrep.placeParams.uniformScale}
                  data-modeling-place-scale
                  onChange={(e) => prep.updatePlaceParams({ uniformScale: stampScaleInput(e.target.value) })}
                />
              </label>
              <label className="field">
                <span>Embed</span>
                <LengthInput
                  mm={prep.printPrep.placeParams.embedDepth}
                  unit={units}
                  data-modeling-place-embed-depth
                  onMm={(mm) => prep.updatePlaceParams({ embedDepth: mm })}
                />
              </label>
              <label className="field">
                <span>Spin (°)</span>
                <input
                  type="number"
                  step={5}
                  value={prep.printPrep.placeParams.twistDeg}
                  data-modeling-place-spin
                  onChange={(e) => prep.updatePlaceParams({ twistDeg: spinInput(e.target.value) })}
                />
              </label>
            </div>
            <div className="op-row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={prep.placeAlignToNormal}
                  data-modeling-place-align-normal
                  onChange={(e) => prep.setAlignToNormal(e.target.checked)}
                />
                <span>Align to normal</span>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={prep.printPrep.placeParams.conformToSurface}
                  data-modeling-place-conform
                  onChange={(e) =>
                    prep.updatePlaceParams({ conformToSurface: e.target.checked })
                  }
                />
                <span>Conform to surface</span>
              </label>
            </div>
            <div className="op-row">
              <button
                type="button"
                className={prep.printPrep.activeTool === "placeStamp" ? "primary" : undefined}
                aria-pressed={prep.printPrep.activeTool === "placeStamp"}
                disabled={disabled || prep.prepBusy !== null}
                data-modeling-place-tool
                onClick={() => prep.toggleTool("placeStamp")}
              >
                Click Surface to Place Stamp
              </button>
            </div>
          </div>

          {prep.prepBusy !== null ? <p className="muted busy">{prep.prepBusy}...</p> : null}
          {prep.prepReport !== null ? (
            <p className={prep.prepReport.error ? "report error" : "report"} data-modeling-print-prep-report>
              <strong>{prep.prepReport.label}:</strong> {prep.prepReport.text}
            </p>
          ) : null}
        </>
      )}

      {ops.busy !== null ? <p className="muted busy">{ops.busy}...</p> : null}
      {ops.report !== null ? (
        <p className={ops.report.error ? "report error" : "report"} data-modeling-report>
          <strong>{ops.report.label}:</strong> {ops.report.text}
        </p>
      ) : null}
    </section>
  );
}
