// Modeling & Resampling panel (#195): Boolean CSG (with Keep originals), destructive Plane Cut,
// Offset & Shell (Solidify), Quadric Simplify, and Loop Subdivide.
import type { CutReport, MeshHandle, OffsetReport, ShellReport, SimplifyReport, SubdivideReport } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useEffect, useState } from "react";
import { type AppState, type Item, useApp } from "../scene/app-state";
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
} from "../scene/workbench-ops";
import { computeLocalExtents } from "../scene/transform-math";
import { LengthInput } from "./LengthInput";
import { describeReport } from "./repair-summary";
import { formatLength, fromDisplayLength, lengthInputValue } from "./units";

function pickOtherMesh(items: readonly Item[], primary: MeshHandle | null, current: MeshHandle | null): MeshHandle | null {
  if (current !== null && items.some((i) => i.mesh === current) && current !== primary) {
    return current;
  }
  return items.find((i) => i.mesh !== primary)?.mesh ?? items[0]?.mesh ?? null;
}

export function ModelingPanel(props: { app: AppState }) {
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
                disabled={disabled || (offsetAlgorithm === "normal" && offsetDistance === 0)}
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
                Offset
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
