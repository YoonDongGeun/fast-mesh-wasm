// The repair panel's surface cleanup section (#264): Denoise and Delaunay
// flips with their settings, limited to the selected faces when asked, and
// the surface's readouts before and after the last run.
import type { SurfaceCleanup } from "../ux/use-surface-cleanup";

export function SurfaceCleanupSection(props: { cleanup: SurfaceCleanup; disabled: boolean }) {
  const { cleanup, disabled } = props;
  return (
    <div className="ops" data-surface-cleanup>
      <div className="op">
        <button
          type="button"
          data-repair-op="denoise"
          disabled={disabled || cleanup.blocked}
          title="Smooth noise out of the surface and keep its sharp edges"
          onClick={() => void cleanup.denoise()}
        >
          Denoise
        </button>
        <label className="field" title="Rounds of smoothing, 1 to 100">
          rounds
          <input
            type="number"
            data-denoise="iterations"
            min={1}
            max={100}
            step={1}
            value={cleanup.iterations}
            onChange={(e) => cleanup.setIterationsText(e.currentTarget.value)}
          />
        </label>
        <label className="field" title="How much each round smooths">
          strength
          <input
            type="range"
            data-denoise="strength"
            min={0.05}
            max={1}
            step={0.05}
            value={cleanup.strength}
            onChange={(e) => cleanup.setStrengthText(e.currentTarget.value)}
          />
          <span className="muted small">{cleanup.strength.toFixed(2)}</span>
        </label>
        <label className="field" title="Edges sharper than this angle are kept, in degrees">
          keep edges over
          <input
            type="number"
            data-denoise="featureAngle"
            min={1}
            max={180}
            step={1}
            value={cleanup.featureAngle}
            onChange={(e) => cleanup.setFeatureAngleText(e.currentTarget.value)}
          />
          °
        </label>
      </div>
      <div className="op">
        <button
          type="button"
          data-repair-op="delaunayFlips"
          disabled={disabled || cleanup.blocked}
          title="Flip edges for better triangles without moving a vertex"
          onClick={() => void cleanup.delaunayFlips()}
        >
          Delaunay flips
        </button>
        <label className="field" title="The surface turns at most this far from what it was, in degrees">
          within
          <input
            type="number"
            data-delaunay="dihedralLimit"
            min={0}
            max={90}
            step={1}
            value={cleanup.dihedralLimit}
            onChange={(e) => cleanup.setDihedralLimitText(e.currentTarget.value)}
          />
          °
        </label>
      </div>
      <label className="check" title="Run on the faces selected with the Select tools; everything else stays as it is">
        <input
          type="checkbox"
          data-cleanup-selected
          checked={cleanup.selectedOnly}
          onChange={(e) => cleanup.setSelectedOnly(e.currentTarget.checked)}
        />
        Selected faces only ({cleanup.selectedCount.toLocaleString("en")})
      </label>
      {cleanup.readouts !== null ? (
        <dl className="analysis" data-surface-readout={cleanup.readouts.label}>
          {cleanup.readouts.rows.map((row) => (
            <div key={row.key} data-readout={row.key} style={{ display: "contents" }}>
              <dt>{row.label}</dt>
              <dd>
                {row.before} → {row.after}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
    </div>
  );
}
