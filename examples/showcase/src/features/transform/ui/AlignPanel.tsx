// Registration / Alignment panel. Rigid ICP (#195): `scene.align.register` with
// Moving + Fixed pickers, maxIterations / overlap / maxDistance options, a
// single-step undoable placement commit, and RMSE / iteration / status
// readouts. Landmark pairs, coarse + ICP and the symmetry plane (#253): each
// alignment, and the mirror across the plane, is one undo step.
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { COARSE_METHODS, formatNormal, ICP_OBJECTIVES, landmarkPrompt, MIN_LANDMARK_PAIRS } from "../ux/align-tools";
import { useLandmarkAlign, useMultistageAlign, useSymmetryPlane } from "../ux/use-align-tools";
import { useIcpAlign } from "../ux/use-icp-align";

const length = (mm: number, units: LengthUnit) => (Number.isFinite(mm) ? formatLength(mm, units) : "Infinity");

export function AlignPanel(props: { app: AppState }) {
  const { app } = props;
  const store = useScene();
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const units = useApp(app, (s) => s.units);

  const align = useIcpAlign(store, items, selection);
  const { resolvedMoving, resolvedFixed, canRun, maxIterations, overlap, maxDistance, busy, error, result } = align;
  const multi = useMultistageAlign(store, resolvedMoving, resolvedFixed, { maxIterations, overlap, maxDistance });
  const marks = useLandmarkAlign(app, store, resolvedMoving, resolvedFixed);
  const sym = useSymmetryPlane(app, store, items, selection);
  const anyBusy = busy || multi.busy || marks.busy || sym.busy;

  const nameOf = (mesh: unknown) => items.find((i) => i.mesh === mesh)?.name ?? "";
  const indexOf = (mesh: unknown) => items.findIndex((i) => i.mesh === mesh);
  const set = marks.set;
  const picking = set?.picking === true;

  /** Starts picking pairs, pinning the moving and fixed meshes so a selection change keeps them. */
  const pickPairs = () => {
    if (picking) {
      marks.stop();
      return;
    }
    align.chooseMoving(indexOf(resolvedMoving));
    align.chooseFixed(indexOf(resolvedFixed));
    marks.pick();
  };

  return (
    <section className="panel align-panel" aria-label="Registration" data-panel="align">
      <header className="panel-head">
        <h2>Align</h2>
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
              value={resolvedMoving === null ? "" : String(indexOf(resolvedMoving))}
              disabled={anyBusy}
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
              value={resolvedFixed === null ? "" : String(indexOf(resolvedFixed))}
              disabled={anyBusy}
              onChange={(e) => align.chooseFixed(Number(e.currentTarget.value))}
            >
              {items.map((it, idx) => (
                <option key={it.id} value={String(idx)}>
                  {it.name} (fixed)
                </option>
              ))}
            </select>
          </div>

          <h3>ICP</h3>
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
                disabled={anyBusy}
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
                disabled={anyBusy}
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
            <button type="button" data-align-run disabled={!canRun || anyBusy} onClick={() => void align.onRun()}>
              {busy ? "Aligning..." : "Align (ICP)"}
            </button>
          </div>

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
                <strong>RMSE:</strong> <span data-align-rmse>{length(result.rmse, units)}</span>
                {" · "}
                <strong>Iterations:</strong> <span data-align-iters>{result.iterations}</span>
              </div>
            </div>
          ) : null}

          <div className="op" data-align-multistage>
            <h3>Coarse + ICP</h3>
            <label className="field">
              Coarse
              <select
                aria-label="Coarse pre-alignment"
                data-align-coarse
                value={multi.coarseMethod}
                disabled={anyBusy}
                onChange={(e) => multi.setCoarseMethod(e.currentTarget.value as typeof multi.coarseMethod)}
              >
                {COARSE_METHODS.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Metric
              <select
                aria-label="ICP metric"
                data-align-objective
                value={multi.objective}
                disabled={anyBusy}
                onChange={(e) => multi.setObjective(e.currentTarget.value as typeof multi.objective)}
              >
                {ICP_OBJECTIVES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              data-align-action="multistage-run"
              disabled={!multi.canRun || anyBusy}
              title="Pre-align by the meshes' shapes, then refine with ICP (the ICP options above), as one undo step"
              onClick={() => void multi.run()}
            >
              {multi.busy ? "Aligning..." : "Coarse + ICP"}
            </button>
            {multi.error !== null ? (
              <p className="report error" data-align-multistage-error>
                {multi.error}
              </p>
            ) : null}
            {multi.stages !== null ? (
              <div className="report align-report" data-align-multistage-report>
                <dl>
                  {multi.stages.map((s) => (
                    <div key={s.label} style={{ display: "contents" }} data-align-stage={s.status === null ? "coarse" : "icp"}>
                      <dt>{s.label}</dt>
                      <dd>
                        RMSE <span data-align-stage-rmse>{length(s.rmse, units)}</span> · {s.iterations} iterations
                        {s.status !== null ? ` · ${s.status}` : ""}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            ) : null}
          </div>

          <div className="op" data-align-landmarks>
            <h3>Landmarks</h3>
            <button
              type="button"
              data-align-pick
              data-align-action="landmarks-pick"
              aria-pressed={picking}
              disabled={anyBusy || resolvedMoving === null || resolvedFixed === null}
              title="Click a point on the moving mesh, then the matching point on the fixed mesh, for each pair"
              onClick={pickPairs}
            >
              {picking ? "Stop Picking" : set !== null && set.movingPoints.length > 0 ? "Pick More" : "Pick Pairs"}
            </button>
            <button
              type="button"
              data-align-action="landmarks-undo"
              disabled={anyBusy || set === null || set.movingPoints.length === 0}
              onClick={marks.undoPoint}
            >
              Undo Point
            </button>
            <button
              type="button"
              data-align-action="landmarks-clear"
              disabled={anyBusy || set === null}
              onClick={marks.clear}
            >
              Clear
            </button>
            <label className="field">
              <input
                type="checkbox"
                data-align-landmark-scale
                checked={marks.allowScaling}
                disabled={anyBusy}
                onChange={(e) => marks.setAllowScaling(e.currentTarget.checked)}
              />
              Uniform scale
            </label>
            <button
              type="button"
              data-align-action="landmarks-run"
              disabled={!marks.canAlign || anyBusy}
              title={`Move the moving mesh so its points meet their matches (at least ${MIN_LANDMARK_PAIRS} pairs), as one undo step`}
              onClick={() => void marks.align()}
            >
              {marks.busy ? "Aligning..." : `Align ${marks.pairs} Pairs`}
            </button>
            {set !== null ? (
              <p className="muted" data-align-landmark-count>
                {marks.pairs} {marks.pairs === 1 ? "pair" : "pairs"}
                {set.movingPoints.length > set.fixedPoints.length ? " + 1 point" : ""}
                {marks.pairs < MIN_LANDMARK_PAIRS ? ` (at least ${MIN_LANDMARK_PAIRS} to align)` : ""}
              </p>
            ) : null}
            {picking && set !== null ? (
              <p className="muted" data-align-landmark-prompt>
                {landmarkPrompt(set, nameOf(set.moving), nameOf(set.fixed))} Esc stops.
              </p>
            ) : null}
            {marks.error !== null ? (
              <p className="report error" data-align-landmark-error>
                {marks.error}
              </p>
            ) : null}
            {marks.result !== null ? (
              <div className="report" data-align-landmark-report>
                <div>
                  <strong>RMSE:</strong> <span data-align-landmark-rmse>{length(marks.result.rmse, units)}</span>
                  {marks.result.scale !== 1 ? (
                    <>
                      {" · "}
                      <strong>Scale:</strong> <span data-align-landmark-scale-factor>{marks.result.scale.toFixed(4)}</span>
                    </>
                  ) : null}
                </div>
                <ol className="align-residuals" data-align-landmark-residuals>
                  {marks.result.residuals.map((r, i) => (
                    <li key={i}>{length(r, units)}</li>
                  ))}
                </ol>
              </div>
            ) : null}
          </div>
        </>
      )}

      {items.length > 0 ? (
        <div className="op" data-align-symmetry>
          <h3>Symmetry Plane</h3>
          <label className="field">
            Mesh
            <select
              aria-label="Symmetry mesh"
              data-align-symmetry-mesh
              value={sym.mesh === null ? "" : String(indexOf(sym.mesh))}
              disabled={anyBusy}
              onChange={(e) => sym.chooseMesh(Number(e.currentTarget.value))}
            >
              {items.map((it, idx) => (
                <option key={it.id} value={String(idx)}>
                  {it.name}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            data-align-action="symmetry-detect"
            disabled={anyBusy || sym.mesh === null}
            title="Find the plane the mesh is most nearly a mirror image across, and draw it"
            onClick={() => void sym.detect()}
          >
            {sym.running === "detect" ? "Detecting..." : "Detect"}
          </button>
          <button
            type="button"
            data-align-action="symmetry-mirror"
            disabled={anyBusy || sym.shown === null}
            title="Mirror the plane's mesh across the plane, as one undo step"
            onClick={() => void sym.mirror()}
          >
            {sym.running === "mirror" ? "Mirroring..." : "Mirror Across Plane"}
          </button>
          <button type="button" data-align-action="symmetry-clear" disabled={sym.shown === null} onClick={sym.clear}>
            Hide
          </button>
          {sym.error !== null ? (
            <p className="report error" data-align-symmetry-error>
              {sym.error}
            </p>
          ) : null}
          {sym.shown !== null ? (
            <div className="report" data-align-symmetry-report>
              <div>
                <strong>Plane on {nameOf(sym.shown.mesh)}:</strong> normal{" "}
                <span data-align-symmetry-normal>{formatNormal(sym.worldNormal ?? sym.shown.normal)}</span>
              </div>
              <div>
                <strong>Fit error:</strong> <span data-align-symmetry-error-rms>{length(sym.shown.errorRms, units)}</span>
                {" · "}
                <strong>Score:</strong> <span data-align-symmetry-score>{sym.shown.score.toFixed(3)}</span>
              </div>
              {sym.mirrored ? <div data-align-symmetry-mirrored>Mirrored across the plane.</div> : null}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
