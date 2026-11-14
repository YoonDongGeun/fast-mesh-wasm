// Rigid ICP Registration / Alignment panel (#195): exposes `scene.align.register`
// with Moving + Fixed pickers, maxIterations / overlap / maxDistance options,
// single-step undoable placement commit, and RMSE / iteration / status readouts.
import type { MeshHandle, RegisterResult } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useState } from "react";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { runIcpAlign } from "../scene/workbench-ops";
import { LengthInput } from "./LengthInput";
import { formatLength } from "./units";

function pickDistinctMesh(items: readonly Item[], exclude: MeshHandle | null, current: MeshHandle | null): MeshHandle | null {
  if (current !== null && items.some((i) => i.mesh === current) && current !== exclude) {
    return current;
  }
  return items.find((i) => i.mesh !== exclude)?.mesh ?? null;
}

export function AlignPanel(props: { app: AppState }) {
  const { app } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const units = useApp(app, (s) => s.units);

  const [movingPick, setMovingPick] = useState<MeshHandle | null>(null);
  const [fixedPick, setFixedPick] = useState<MeshHandle | null>(null);
  const [maxIterations, setMaxIterations] = useState(50);
  const [overlap, setOverlap] = useState(1);
  const [maxDistance, setMaxDistance] = useState(0);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegisterResult | null>(null);

  const resolvedMoving =
    (movingPick !== null && items.some((i) => i.mesh === movingPick) ? movingPick : null) ??
    (selection !== null && items.some((i) => i.mesh === selection) ? selection : null) ??
    items[0]?.mesh ??
    null;
  const resolvedFixed = pickDistinctMesh(items, resolvedMoving, fixedPick);

  const canRun = !busy && resolvedMoving !== null && resolvedFixed !== null && resolvedMoving !== resolvedFixed;

  const onRun = async () => {
    if (!canRun || resolvedMoving === null || resolvedFixed === null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await runIcpAlign({
        store,
        moving: resolvedMoving,
        fixed: resolvedFixed,
        maxIterations: Math.max(1, Math.round(maxIterations)),
        overlap: Math.max(0.05, Math.min(1, overlap)),
        maxDistance: maxDistance > 0 ? maxDistance : undefined,
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel align-panel" aria-label="Registration" data-panel="align">
      <header className="panel-head">
        <h2>Align (ICP)</h2>
      </header>

      {items.length < 2 ? (
        <p className="muted">Load at least two meshes to run rigid ICP registration.</p>
      ) : (
        <>
          <div className="picker">
            <span>M</span>
            <select
              aria-label="Moving mesh"
              data-align-moving
              value={resolvedMoving === null ? "" : String(items.findIndex((i) => i.mesh === resolvedMoving))}
              disabled={busy}
              onChange={(e) => setMovingPick(items[Number(e.currentTarget.value)]?.mesh ?? null)}
            >
              {items.map((it, idx) => (
                <option key={it.id} value={String(idx)}>
                  {it.name} (moving)
                </option>
              ))}
            </select>
          </div>
          <div className="picker">
            <span>F</span>
            <select
              aria-label="Fixed mesh"
              data-align-fixed
              value={resolvedFixed === null ? "" : String(items.findIndex((i) => i.mesh === resolvedFixed))}
              disabled={busy}
              onChange={(e) => setFixedPick(items[Number(e.currentTarget.value)]?.mesh ?? null)}
            >
              {items.map((it, idx) => (
                <option key={it.id} value={String(idx)}>
                  {it.name} (fixed)
                </option>
              ))}
            </select>
          </div>

          <div className="op-row">
            <label className="field">
              maxIterations
              <input
                type="number"
                min={1}
                max={500}
                step={5}
                aria-label="Max iterations"
                data-align-iterations
                value={maxIterations}
                disabled={busy}
                onChange={(e) => setMaxIterations(Math.max(1, Math.round(Number(e.currentTarget.value) || 1)))}
              />
            </label>
            <label className="field">
              overlap
              <input
                type="number"
                min={0.05}
                max={1}
                step={0.05}
                aria-label="Expected overlap"
                data-align-overlap
                value={overlap}
                disabled={busy}
                onChange={(e) => setOverlap(Math.max(0.05, Math.min(1, Number(e.currentTarget.value) || 1)))}
              />
            </label>
          </div>

          <div className="op-row">
            <label className="field">
              maxDistance ({units})
              <LengthInput
                aria-label="Max correspondence distance"
                data-align-max-distance
                min={0}
                mm={maxDistance}
                unit={units}
                onMm={(mm) => setMaxDistance(Math.max(0, mm))}
              />
            </label>
            <button type="button" data-align-run disabled={!canRun} onClick={() => void onRun()}>
              {busy ? "Aligning..." : "Align (ICP)"}
            </button>
          </div>
        </>
      )}

      {error !== null ? (
        <p className="report error" data-align-error>
          {error}
        </p>
      ) : null}

      {result !== null ? (
        <div className="report" data-align-report>
          <div>
            <strong>Status:</strong> <span data-align-status>{result.status}</span> (
            {result.converged ? "converged" : "not converged"})
          </div>
          <div>
            <strong>RMSE:</strong>{" "}
            <span data-align-rmse>
              {Number.isFinite(result.rmse) ? formatLength(result.rmse, units) : "Infinity"}
            </span>
            {" · "}
            <strong>Iterations:</strong> <span data-align-iters>{result.iterations}</span>
          </div>
        </div>
      ) : null}
    </section>
  );
}
