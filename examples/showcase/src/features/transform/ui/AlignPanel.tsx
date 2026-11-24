// Rigid ICP Registration / Alignment panel (#195): exposes `scene.align.register`
// with Moving + Fixed pickers, maxIterations / overlap / maxDistance options,
// single-step undoable placement commit, and RMSE / iteration / status readouts.
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { formatLength } from "../../../shared/ux/units";
import { useIcpAlign } from "../ux/use-icp-align";

export function AlignPanel(props: { app: AppState }) {
  const { app } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const units = useApp(app, (s) => s.units);

  const align = useIcpAlign(store, items, selection);
  const { resolvedMoving, resolvedFixed, canRun, maxIterations, overlap, maxDistance, busy, error, result } = align;

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
              onChange={(e) => align.chooseMoving(Number(e.currentTarget.value))}
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
              onChange={(e) => align.chooseFixed(Number(e.currentTarget.value))}
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
                onChange={(e) => align.setMaxIterations(Number(e.currentTarget.value))}
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
                onChange={(e) => align.setOverlap(Number(e.currentTarget.value))}
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
                onMm={(mm) => align.setMaxDistance(mm)}
              />
            </label>
            <button type="button" data-align-run disabled={!canRun} onClick={() => void align.onRun()}>
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
