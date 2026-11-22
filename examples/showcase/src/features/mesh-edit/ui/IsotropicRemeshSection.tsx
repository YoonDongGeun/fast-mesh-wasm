// The Modeling panel's isotropic remesh section (#254), beside simplify and
// subdivide: the target edge length (the mesh's mean edge length until one
// is typed), passes, Lock boundaries, Preserve creases and their angle, the
// triangle count now and about how many the target gives, and the run.
// What it does lives in `ux/use-isotropic-remesh.ts`.
import type { SceneStore } from "fast-mesh-wasm/store";
import type { Item } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { MAX_REMESH_ITERATIONS } from "../ux/isotropic-remesh";
import { useIsotropicRemesh } from "../ux/use-isotropic-remesh";

export function IsotropicRemeshSection(props: {
  store: SceneStore;
  /** The selected mesh, else the first one. */
  activeItem: Item | null;
  units: LengthUnit;
}) {
  const { units } = props;
  const r = useIsotropicRemesh(props);
  const disabled = r.busy !== null || props.activeItem === null;

  return (
    <div className="op" data-isotropic-section>
      <h3>Isotropic Remesh</h3>
      <p className="muted" data-isotropic-stats>
        {props.activeItem?.name ?? "No mesh"}: {r.stats.triangles.toLocaleString("en")} triangles, mean edge{" "}
        {formatLength(r.stats.meanEdge, units)}
        {r.estimate > 0 ? ` → about ${r.estimate.toLocaleString("en")} triangles` : ""}
      </p>
      <div className="op-row">
        <label className="field">
          edge ({units})
          <LengthInput
            aria-label="Isotropic remesh target edge length"
            data-isotropic-edge
            min={0}
            mm={r.target}
            unit={units}
            disabled={disabled}
            onMm={r.setTarget}
          />
        </label>
        <button
          type="button"
          data-isotropic-edge-mean
          aria-pressed={r.targetIsMean}
          disabled={disabled}
          title="Use the mesh's mean edge length"
          onClick={r.resetTargetToMean}
        >
          Mean
        </button>
        <label className="field">
          passes
          <input
            type="number"
            min={1}
            max={MAX_REMESH_ITERATIONS}
            step={1}
            aria-label="Isotropic remesh passes"
            data-isotropic-iterations
            value={r.iterations}
            disabled={disabled}
            onChange={(e) => r.setIterationsText(e.currentTarget.value)}
          />
        </label>
      </div>
      <div className="op-row">
        <label className="check">
          <input
            type="checkbox"
            data-isotropic-lock-boundaries
            checked={r.lockBoundaries}
            disabled={disabled}
            onChange={(e) => r.setLockBoundaries(e.currentTarget.checked)}
          />
          Lock boundaries
        </label>
        <label className="check">
          <input
            type="checkbox"
            data-isotropic-creases
            checked={r.preserveCreases}
            disabled={disabled}
            onChange={(e) => r.setPreserveCreases(e.currentTarget.checked)}
          />
          Preserve creases
        </label>
        {r.preserveCreases ? (
          <label className="field">
            angle (°)
            <input
              type="number"
              min={0}
              max={180}
              step={1}
              aria-label="Crease angle"
              data-isotropic-crease-angle
              value={r.creaseAngle}
              disabled={disabled}
              onChange={(e) => r.setCreaseAngleText(e.currentTarget.value)}
            />
          </label>
        ) : null}
      </div>
      <button type="button" data-isotropic-run disabled={disabled || r.blocked} onClick={() => void r.run()}>
        Remesh (Isotropic)
      </button>
      {r.busy !== null ? <p className="muted busy">{r.busy}...</p> : null}
      {r.report !== null ? (
        <p className={r.report.error ? "report error" : "report"} data-isotropic-report>
          <strong>{r.report.label}:</strong> {r.report.text}
        </p>
      ) : null}
    </div>
  );
}
