// Modeling & Resampling panel (#195): Boolean CSG (with Keep originals), destructive Plane Cut,
// Offset & Shell (Solidify), Quadric Simplify, and Loop Subdivide.
import type {
  CutReport,
  HollowBackend,
  HollowOptions,
  MeshHandle,
  OffsetReport,
  PlaceUpAxis,
  ShellReport,
  SimplifyReport,
  SubdivideReport,
} from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useEffect, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import {
  type BooleanOperation,
  type CutAxis,
  type CutKeepSide,
  extractLiveMeshData,
  type OffsetModeAlgorithm,
  runBooleanCsg,
  runOffset,
  runPlaneCut,
  runShell,
  runSimplify,
  runSubdivide,
} from "../../../engine/workbench-ops";
import { computeLocalExtents } from "../../transform/ux/transform-math";
import {
  executeDrainHoleAtHit,
  findCenterSurfaceHit,
  getPlaceAlignToNormal,
  runHollowMeshOp,
  setPlaceAlignToNormal,
} from "./HollowPlaceToolsLayer";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { describeReport } from "../../mesh-repair/ux/repair-summary";
import { formatLength, fromDisplayLength, lengthInputValue } from "../../../shared/ux/units";

function pickOtherMesh(items: readonly Item[], primary: MeshHandle | null, current: MeshHandle | null): MeshHandle | null {
  if (current !== null && items.some((i) => i.mesh === current) && current !== primary) {
    return current;
  }
  return items.find((i) => i.mesh !== primary)?.mesh ?? items[0]?.mesh ?? null;
}

export function ModelingPanel(props: { app: AppState }) {
  const prepStore = useScene();
  const prepUnit = useApp(props.app, (s) => s.units);
  const printPrep = useApp(props.app, (s) => s.wave6.printPrep);
  const [hollowWallMm, setHollowWallMm] = useState(1.5);
  const [hollowBackend, setHollowBackend] = useState<HollowBackend>("bvhSdf");
  const [hollowVoxelSizeText, setHollowVoxelSizeText] = useState("");
  const [hollowMaxVoxels, setHollowMaxVoxels] = useState(524288);
  const [hollowCapInnerShell, setHollowCapInnerShell] = useState(false);
  const [drainSavedDepthMm, setDrainSavedDepthMm] = useState(12.0);
  const [placeAlignToNormal, setPlaceAlignState] = useState(() => getPlaceAlignToNormal(props.app));
  const [prepBusy, setPrepBusy] = useState<string | null>(null);
  const [prepReport, setPrepReport] = useState<{ label: string; text: string; error?: boolean } | null>(null);

  const drainThrough = printPrep.drainParams.depth <= 0;

  const updatePrintPrep = (patch: Partial<typeof printPrep>) => {
    const current = props.app.getSnapshot().wave6.printPrep;
    props.app.updateWave6({ printPrep: { ...current, ...patch } });
  };

  const updateDrainParams = (patch: Partial<typeof printPrep.drainParams>) => {
    const current = props.app.getSnapshot().wave6.printPrep;
    props.app.updateWave6({
      printPrep: { ...current, drainParams: { ...current.drainParams, ...patch } },
    });
  };

  const updatePlaceParams = (patch: Partial<typeof printPrep.placeParams>) => {
    const current = props.app.getSnapshot().wave6.printPrep;
    props.app.updateWave6({
      printPrep: { ...current, placeParams: { ...current.placeParams, ...patch } },
    });
  };

  const runHollowAction = async (target: Item) => {
    setPrepBusy("Hollow Mesh");
    try {
      const parsedVoxel = hollowVoxelSizeText.trim() === "" ? undefined : Number(hollowVoxelSizeText);
      const options: HollowOptions = {
        wallThickness: Math.max(0.1, hollowWallMm),
        backend: hollowBackend,
        ...(parsedVoxel !== undefined && Number.isFinite(parsedVoxel) && parsedVoxel > 0
          ? { voxelSize: parsedVoxel }
          : {}),
        maxVoxels: hollowMaxVoxels,
        ...(hollowCapInnerShell ? { maxCavityTriangles: 12000 } : {}),
      };
      const res = await runHollowMeshOp({ store: prepStore, mesh: target.mesh, options });
      setPrepReport({
        label: "Hollow Mesh",
        text: `${res.beforeTriangles} -> ${res.afterTriangles} tris · ${res.report.cavitiesCreated} cavity (${res.report.cavityTriangles} cavity tris, ${res.report.cavityVolume.toFixed(1)} mm3)`,
        error: false,
      });
    } catch (err) {
      setPrepReport({
        label: "Hollow Mesh",
        text: err instanceof Error ? err.message : String(err),
        error: true,
      });
    } finally {
      setPrepBusy(null);
    }
  };

  const runPunchCenterAction = async (target: Item) => {
    setPrepBusy("Drain Hole");
    try {
      const hit = await findCenterSurfaceHit(prepStore.scene, target.mesh);
      if (hit === null) {
        throw new Error("Could not find a surface point on the active mesh.");
      }
      const res = await executeDrainHoleAtHit({
        store: prepStore,
        app: props.app,
        target: target.mesh,
        targetName: target.name,
        hitPoint: hit.point,
        hitNormal: hit.normal,
        drainParams: printPrep.drainParams,
        through: drainThrough,
      });
      const plugText = res.plugs.length > 0 ? ` · ${res.plugs.length} plug adopted` : "";
      setPrepReport({
        label: "Drain Hole",
        text: `${res.beforeTriangles} -> ${res.afterTriangles} tris (${res.report.holesPunched} hole${plugText})`,
        error: false,
      });
    } catch (err) {
      setPrepReport({
        label: "Drain Hole",
        text: err instanceof Error ? err.message : String(err),
        error: true,
      });
    } finally {
      setPrepBusy(null);
    }
  };

  const togglePrintPrepTool = (tool: "drainHole" | "placeStamp") => {
    const next = printPrep.activeTool === tool ? null : tool;
    updatePrintPrep({ activeTool: next });
    void prepStore.commitFrame();
  };
  const { app } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const units = useApp(app, (s) => s.units);

  const selectedItem = selection === null ? null : (items.find((i) => i.mesh === selection) ?? null);
  const activeItem = selectedItem ?? items[0] ?? null;

  // Boolean CSG state
  const [boolTarget, setBoolTarget] = useState<MeshHandle | null>(null);
  const [boolSource, setBoolSource] = useState<MeshHandle | null>(null);
  const [boolOp, setBoolOp] = useState<BooleanOperation>("union");
  const [keepOriginals, setKeepOriginals] = useState(true);

  // Plane Cut state
  const [cutAxis, setCutAxis] = useState<CutAxis>("y");
  const [cutCustomNormal, setCutCustomNormal] = useState<[number, number, number]>([0, 1, 0]);
  const [cutOffset, setCutOffset] = useState(0);
  const [cutCap, setCutCap] = useState(true);
  const [cutKeepSide, setCutKeepSide] = useState<CutKeepSide>("+");

  // Offset & Shell state
  const [offsetAlgorithm, setOffsetAlgorithm] = useState<OffsetModeAlgorithm>("auto");
  const [offsetDistance, setOffsetDistance] = useState(0.5);
  const [offsetVoxelSize, setOffsetVoxelSize] = useState("");
  const [shellThickness, setShellThickness] = useState(1);

  // Resampling state
  const [lockBoundaries, setLockBoundaries] = useState(true);
  const [simplifyRatio, setSimplifyRatio] = useState(50);
  const [simplifyTarget, setSimplifyTarget] = useState(500);
  const [simplifyUseRatio, setSimplifyUseRatio] = useState(true);
  const [subdivideLevels, setSubdivideLevels] = useState(1);

  // Live triangle count & axis bounds of active mesh
  const [liveTris, setLiveTris] = useState(0);
  const [cutBounds, setCutBounds] = useState<[number, number]>([-50, 50]);

  const [busy, setBusy] = useState<string | null>(null);
  const [report, setReport] = useState<{ label: string; text: string; error?: boolean } | null>(null);

  const resolvedTarget =
    (boolTarget !== null && items.some((i) => i.mesh === boolTarget) ? boolTarget : null) ??
    activeItem?.mesh ??
    null;
  const resolvedSource = pickOtherMesh(items, resolvedTarget, boolSource);

  useEffect(() => {
    if (activeItem === null) {
      setLiveTris(0);
      setCutBounds([-50, 50]);
      return;
    }
    const updateStats = () => {
      try {
        const snap = extractLiveMeshData(store.scene, activeItem.mesh);
        const tris = Math.floor(snap.indices.length / 3);
        setLiveTris(tris);
        if (simplifyUseRatio) {
          setSimplifyTarget(Math.max(1, Math.round((tris * simplifyRatio) / 100)));
        }
        const ext = computeLocalExtents(snap.positions, snap.indices);
        const idx = cutAxis === "x" ? 0 : cutAxis === "y" ? 1 : 2;
        const pad = Math.max(ext.extents[idx] * 0.25, 1);
        setCutBounds([ext.min[idx] - pad, ext.max[idx] + pad]);
      } catch {
        setLiveTris(activeItem.triangles);
      }
    };
    updateStats();
    const unsubStore = store.subscribe(updateStats);
    const unsubScene = store.scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === activeItem.mesh)) {
        updateStats();
      }
    });
    return () => {
      unsubStore();
      unsubScene();
    };
  }, [activeItem, store, cutAxis, simplifyUseRatio, simplifyRatio]);

  const runAction = async (label: string, fn: () => Promise<string>) => {
    setBusy(label);
    setReport(null);
    try {
      const text = await fn();
      setReport({ label, text });
    } catch (e) {
      setReport({ label, text: e instanceof Error ? e.message : String(e), error: true });
    } finally {
      setBusy(null);
    }
  };

  const targetItem = resolvedTarget === null ? null : (items.find((i) => i.mesh === resolvedTarget) ?? null);
  const sourceItem = resolvedSource === null ? null : (items.find((i) => i.mesh === resolvedSource) ?? null);
  const disabled = busy !== null || activeItem === null;

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
                value={resolvedTarget === null ? "" : String(items.findIndex((i) => i.mesh === resolvedTarget))}
                disabled={busy !== null}
                onChange={(e) => {
                  const idx = Number(e.currentTarget.value);
                  setBoolTarget(items[idx]?.mesh ?? null);
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
                value={resolvedSource === null ? "" : String(items.findIndex((i) => i.mesh === resolvedSource))}
                disabled={busy !== null}
                onChange={(e) => {
                  const idx = Number(e.currentTarget.value);
                  setBoolSource(items[idx]?.mesh ?? null);
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
              {(
                [
                  ["union", "Union"],
                  ["intersect", "Intersection"],
                  ["subtract", "Difference"],
                ] as const
              ).map(([op, label]) => (
                <button
                  key={op}
                  type="button"
                  data-modeling-boolean-op={op}
                  aria-pressed={boolOp === op}
                  disabled={busy !== null}
                  onClick={() => setBoolOp(op)}
                >
                  {label}
                </button>
              ))}
            </div>
            <label className="check">
              <input
                type="checkbox"
                data-modeling-keep-originals
                checked={keepOriginals}
                disabled={busy !== null}
                onChange={(e) => setKeepOriginals(e.currentTarget.checked)}
              />
              Keep originals
            </label>
            <button
              type="button"
              data-modeling-boolean-run
              disabled={busy !== null || targetItem === null || sourceItem === null || items.length < 2}
              onClick={() => {
                if (!targetItem || !sourceItem) return;
                void runAction("Boolean", async () => {
                  const res = await runBooleanCsg({
                    store,
                    app,
                    target: targetItem,
                    source: sourceItem,
                    op: boolOp,
                    keepOriginals,
                  });
                  return `${res.name}: ${res.triangles.toLocaleString()} triangles`;
                });
              }}
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
                  aria-pressed={cutAxis === ax}
                  disabled={disabled}
                  onClick={() => setCutAxis(ax)}
                >
                  {ax === "custom" ? "Custom" : ax.toUpperCase()}
                </button>
              ))}
            </div>
            {cutAxis === "custom" ? (
              <div className="op-row">
                {(["nx", "ny", "nz"] as const).map((label, idx) => (
                  <label key={label} className="field">
                    {label}
                    <input
                      type="number"
                      step="any"
                      aria-label={`Custom cut normal ${label}`}
                      value={cutCustomNormal[idx]}
                      onChange={(e) => {
                        const next: [number, number, number] = [...cutCustomNormal];
                        next[idx] = Number(e.currentTarget.value) || 0;
                        setCutCustomNormal(next);
                      }}
                    />
                  </label>
                ))}
              </div>
            ) : null}
            <label className="slider">
              <div className="slider-head">
                <span>Offset</span>
                <output>{formatLength(cutOffset, units)}</output>
              </div>
              <input
                type="range"
                aria-label="Cut plane offset slider"
                min={cutBounds[0]}
                max={cutBounds[1]}
                step={(cutBounds[1] - cutBounds[0]) / 100 || 0.1}
                value={Math.max(cutBounds[0], Math.min(cutBounds[1], cutOffset))}
                disabled={disabled}
                onChange={(e) => setCutOffset(Number(e.currentTarget.value))}
              />
            </label>
            <div className="op-row">
              <label className="field">
                offset ({units})
                <LengthInput
                  aria-label="Cut plane offset"
                  data-modeling-cut-offset
                  mm={cutOffset}
                  unit={units}
                  onMm={setCutOffset}
                />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  data-modeling-cut-cap
                  checked={cutCap}
                  disabled={disabled}
                  onChange={(e) => setCutCap(e.currentTarget.checked)}
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
                    aria-pressed={cutKeepSide === side}
                    disabled={disabled}
                    onClick={() => setCutKeepSide(side)}
                  >
                    {side} side
                  </button>
                ))}
              </div>
              <button
                type="button"
                data-modeling-cut-run
                disabled={disabled}
                onClick={() => {
                  if (!activeItem) return;
                  void runAction("Plane cut", async () => {
                    const r: CutReport = await runPlaneCut({
                      store,
                      mesh: activeItem.mesh,
                      axis: cutAxis,
                      customNormal: cutCustomNormal,
                      offset: cutOffset,
                      cap: cutCap,
                      keepSide: cutKeepSide,
                    });
                    return `Kept ${r.keptTriangles.toLocaleString()} tris (removed ${r.removedTriangles.toLocaleString()}, cap ${r.capTriangles.toLocaleString()})`;
                  });
                }}
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
                  value={offsetAlgorithm}
                  disabled={disabled}
                  onChange={(e) => setOffsetAlgorithm(e.currentTarget.value as OffsetModeAlgorithm)}
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
                  mm={offsetDistance}
                  unit={units}
                  onMm={setOffsetDistance}
                />
              </label>
              {offsetAlgorithm === "bvhSdf" || offsetAlgorithm === "fhEdt" ? (
                <label className="field">
                  voxel ({units})
                  <input
                    type="number"
                    min={0}
                    step="any"
                    aria-label="Offset voxel size"
                    data-modeling-offset-voxel-size
                    placeholder="auto"
                    value={offsetVoxelSize === "" ? "" : lengthInputValue(Number(offsetVoxelSize), units)}
                    onChange={(e) => {
                      const typed = e.currentTarget.value;
                      setOffsetVoxelSize(typed === "" ? "" : String(fromDisplayLength(Number(typed), units)));
                    }}
                  />
                </label>
              ) : null}
              <button
                type="button"
                data-modeling-offset-run
                disabled={
                  disabled ||
                  (offsetAlgorithm !== "bvhSdf" && offsetAlgorithm !== "fhEdt" && offsetDistance === 0) ||
                  ((offsetAlgorithm === "bvhSdf" || offsetAlgorithm === "fhEdt") &&
                    offsetVoxelSize !== "" &&
                    !(Number(offsetVoxelSize) > 0))
                }
                onClick={() => {
                  if (!activeItem) return;
                  void runAction("Offset", async () => {
                    const r: OffsetReport = await runOffset({
                      store,
                      mesh: activeItem.mesh,
                      distance: offsetDistance,
                      algorithm: offsetAlgorithm,
                      voxelSize: offsetVoxelSize === "" ? undefined : Number(offsetVoxelSize),
                    });
                    return describeReport("offset", r, units);
                  });
                }}
              >
                {(offsetAlgorithm === "bvhSdf" || offsetAlgorithm === "fhEdt") && offsetDistance === 0
                  ? "Voxel remesh"
                  : "Offset"}
              </button>
            </div>
            <div className="op-row">
              <label className="field">
                thickness ({units})
                <LengthInput
                  aria-label="Shell wall thickness"
                  data-modeling-shell-thickness
                  min={0}
                  mm={shellThickness}
                  unit={units}
                  onMm={(mm) => setShellThickness(Math.max(0, mm))}
                />
              </label>
              <button
                type="button"
                data-modeling-shell-run
                disabled={disabled || !(shellThickness > 0)}
                onClick={() => {
                  if (!activeItem) return;
                  void runAction("Shell", async () => {
                    const r: ShellReport = await runShell({
                      store,
                      mesh: activeItem.mesh,
                      thickness: shellThickness,
                    });
                    return describeReport("shell", r, units);
                  });
                }}
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
                checked={lockBoundaries}
                disabled={disabled}
                onChange={(e) => setLockBoundaries(e.currentTarget.checked)}
              />
              Lock boundaries
            </label>
            <div className="op-row">
              <label className="field">
                {simplifyUseRatio ? "ratio (%)" : "target tris"}
                {simplifyUseRatio ? (
                  <input
                    type="number"
                    min={1}
                    max={100}
                    step={5}
                    aria-label="Simplify percentage"
                    data-modeling-simplify-ratio
                    value={simplifyRatio}
                    disabled={disabled}
                    onChange={(e) => {
                      const pct = Math.max(1, Math.min(100, Number(e.currentTarget.value) || 50));
                      setSimplifyRatio(pct);
                      if (liveTris > 0) {
                        setSimplifyTarget(Math.max(1, Math.round((liveTris * pct) / 100)));
                      }
                    }}
                  />
                ) : (
                  <input
                    type="number"
                    min={1}
                    step={1}
                    aria-label="Simplify target triangles"
                    data-modeling-simplify-target
                    value={simplifyTarget}
                    disabled={disabled}
                    onChange={(e) => setSimplifyTarget(Math.max(1, Math.round(Number(e.currentTarget.value) || 1)))}
                  />
                )}
              </label>
              <button
                type="button"
                data-modeling-simplify-mode
                disabled={disabled}
                onClick={() => setSimplifyUseRatio((v) => !v)}
                title="Switch between triangle percentage ratio and exact count"
              >
                {simplifyUseRatio ? "Count" : "Ratio"}
              </button>
              <button
                type="button"
                data-modeling-simplify-run
                disabled={disabled || liveTris <= 0}
                onClick={() => {
                  if (!activeItem) return;
                  const targetCount = simplifyUseRatio
                    ? Math.max(1, Math.round((liveTris * simplifyRatio) / 100))
                    : simplifyTarget;
                  void runAction("Simplify", async () => {
                    const r: SimplifyReport = await runSimplify({
                      store,
                      mesh: activeItem.mesh,
                      targetTriangles: targetCount,
                      preserveBoundary: lockBoundaries,
                    });
                    return describeReport("simplify", r, units);
                  });
                }}
              >
                Simplify ({simplifyUseRatio ? Math.max(1, Math.round((liveTris * simplifyRatio) / 100)) : simplifyTarget})
              </button>
            </div>
            <div className="op-row">
              <label className="field">
                levels
                <select
                  aria-label="Subdivide levels"
                  data-modeling-subdivide-levels
                  value={String(subdivideLevels)}
                  disabled={disabled}
                  onChange={(e) => setSubdivideLevels(Number(e.currentTarget.value))}
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
                onClick={() => {
                  if (!activeItem) return;
                  void runAction("Subdivide", async () => {
                    const r: SubdivideReport = await runSubdivide({
                      store,
                      mesh: activeItem.mesh,
                      levels: subdivideLevels,
                    });
                    return describeReport("subdivide", r, units);
                  });
                }}
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
                  mm={hollowWallMm}
                  unit={prepUnit}
                  data-modeling-hollow-thickness
                  onMm={(mm) => setHollowWallMm(Math.max(0.1, mm))}
                />
              </label>
              <label className="field">
                <span>Backend</span>
                <select
                  value={hollowBackend}
                  data-modeling-hollow-backend
                  onChange={(e) => setHollowBackend(e.target.value as HollowBackend)}
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
                  value={hollowVoxelSizeText}
                  data-modeling-hollow-voxel-size
                  onChange={(e) => setHollowVoxelSizeText(e.target.value)}
                />
              </label>
              <label className="field">
                <span>Max Voxels</span>
                <select
                  value={hollowMaxVoxels}
                  data-modeling-hollow-max-voxels
                  onChange={(e) => setHollowMaxVoxels(Number(e.target.value))}
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
                  checked={hollowCapInnerShell}
                  data-modeling-hollow-inner-shell
                  onChange={(e) => setHollowCapInnerShell(e.target.checked)}
                />
                <span>Cap inner shell (≤12k tris)</span>
              </label>
              <button
                type="button"
                className="primary"
                disabled={disabled || prepBusy !== null}
                data-modeling-hollow-run
                onClick={() => void runHollowAction(activeItem)}
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
                  mm={printPrep.drainParams.diameter * 0.5}
                  unit={prepUnit}
                  data-modeling-drain-radius
                  onMm={(mm) => updateDrainParams({ diameter: Math.max(0.2, mm * 2) })}
                />
              </label>
              <label className="field">
                <span>Depth</span>
                <LengthInput
                  mm={drainThrough ? drainSavedDepthMm : printPrep.drainParams.depth}
                  unit={prepUnit}
                  disabled={drainThrough}
                  data-modeling-drain-depth
                  onMm={(mm) => {
                    const next = Math.max(0.2, mm);
                    setDrainSavedDepthMm(next);
                    updateDrainParams({ depth: next });
                  }}
                />
              </label>
            </div>
            <div className="op-row">
              <label className="field">
                <span>Chamfer</span>
                <LengthInput
                  mm={printPrep.drainParams.outerChamfer}
                  unit={prepUnit}
                  data-modeling-drain-chamfer
                  onMm={(mm) => updateDrainParams({ outerChamfer: Math.max(0, mm) })}
                />
              </label>
              <label className="field">
                <span>Taper (°)</span>
                <input
                  type="number"
                  min={0}
                  max={45}
                  step={1}
                  value={printPrep.drainParams.taperDeg}
                  data-modeling-drain-taper
                  onChange={(e) =>
                    updateDrainParams({
                      taperDeg: Math.max(0, Math.min(45, Number(e.target.value) || 0)),
                    })
                  }
                />
              </label>
            </div>
            <div className="op-row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={drainThrough}
                  data-modeling-drain-through
                  onChange={(e) => {
                    if (e.target.checked) {
                      if (printPrep.drainParams.depth > 0) {
                        setDrainSavedDepthMm(printPrep.drainParams.depth);
                      }
                      updateDrainParams({ depth: 0 });
                    } else {
                      updateDrainParams({ depth: Math.max(0.5, drainSavedDepthMm) });
                    }
                  }}
                />
                <span>Through wall</span>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={printPrep.drainParams.generatePlug}
                  data-modeling-drain-plug
                  onChange={(e) => updateDrainParams({ generatePlug: e.target.checked })}
                />
                <span>Create plug</span>
              </label>
            </div>
            <div className="op-row">
              <button
                type="button"
                className={printPrep.activeTool === "drainHole" ? "primary" : undefined}
                aria-pressed={printPrep.activeTool === "drainHole"}
                disabled={disabled || prepBusy !== null}
                data-modeling-drain-tool
                onClick={() => togglePrintPrepTool("drainHole")}
              >
                Pick Surface Point to Punch Hole
              </button>
              <button
                type="button"
                disabled={disabled || prepBusy !== null}
                data-modeling-drain-center
                onClick={() => void runPunchCenterAction(activeItem)}
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
                  value={printPrep.placeParams.stampSource}
                  data-modeling-place-source
                  onChange={(e) =>
                    updatePlaceParams({
                      stampSource: e.target.value as typeof printPrep.placeParams.stampSource,
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
                  value={printPrep.placeParams.stampMeshId === null ? "" : String(printPrep.placeParams.stampMeshId)}
                  data-modeling-place-stamp-mesh
                  onChange={(e) => {
                    const val = e.target.value;
                    if (val === "") {
                      updatePlaceParams({
                        stampMeshId: null,
                        ...(printPrep.placeParams.stampSource === "sceneMesh"
                          ? { stampSource: "peg" }
                          : {}),
                      });
                    } else {
                      updatePlaceParams({
                        stampSource: "sceneMesh",
                        stampMeshId: Number(val),
                      });
                    }
                  }}
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
                  value={printPrep.placeParams.mode}
                  data-modeling-place-mode
                  onChange={(e) =>
                    updatePlaceParams({
                      mode: e.target.value as typeof printPrep.placeParams.mode,
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
                  value={printPrep.placeParams.upAxis}
                  data-modeling-place-up-axis
                  onChange={(e) =>
                    updatePlaceParams({ upAxis: e.target.value as PlaceUpAxis })
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
                  value={printPrep.placeParams.uniformScale}
                  data-modeling-place-scale
                  onChange={(e) =>
                    updatePlaceParams({
                      uniformScale: Math.max(0.05, Number(e.target.value) || 1),
                    })
                  }
                />
              </label>
              <label className="field">
                <span>Embed</span>
                <LengthInput
                  mm={printPrep.placeParams.embedDepth}
                  unit={prepUnit}
                  data-modeling-place-embed-depth
                  onMm={(mm) => updatePlaceParams({ embedDepth: mm })}
                />
              </label>
              <label className="field">
                <span>Spin (°)</span>
                <input
                  type="number"
                  step={5}
                  value={printPrep.placeParams.twistDeg}
                  data-modeling-place-spin
                  onChange={(e) =>
                    updatePlaceParams({ twistDeg: Number(e.target.value) || 0 })
                  }
                />
              </label>
            </div>
            <div className="op-row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={placeAlignToNormal}
                  data-modeling-place-align-normal
                  onChange={(e) => {
                    const checked = e.target.checked;
                    setPlaceAlignState(checked);
                    setPlaceAlignToNormal(props.app, checked);
                  }}
                />
                <span>Align to normal</span>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={printPrep.placeParams.conformToSurface}
                  data-modeling-place-conform
                  onChange={(e) =>
                    updatePlaceParams({ conformToSurface: e.target.checked })
                  }
                />
                <span>Conform to surface</span>
              </label>
            </div>
            <div className="op-row">
              <button
                type="button"
                className={printPrep.activeTool === "placeStamp" ? "primary" : undefined}
                aria-pressed={printPrep.activeTool === "placeStamp"}
                disabled={disabled || prepBusy !== null}
                data-modeling-place-tool
                onClick={() => togglePrintPrepTool("placeStamp")}
              >
                Click Surface to Place Stamp
              </button>
            </div>
          </div>

          {prepBusy !== null ? <p className="muted busy">{prepBusy}...</p> : null}
          {prepReport !== null ? (
            <p className={prepReport.error ? "report error" : "report"} data-modeling-print-prep-report>
              <strong>{prepReport.label}:</strong> {prepReport.text}
            </p>
          ) : null}
        </>
      )}

      {busy !== null ? <p className="muted busy">{busy}...</p> : null}
      {report !== null ? (
        <p className={report.error ? "report error" : "report"} data-modeling-report>
          <strong>{report.label}:</strong> {report.text}
        </p>
      ) : null}
    </section>
  );
}
