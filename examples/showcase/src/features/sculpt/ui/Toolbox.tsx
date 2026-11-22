// The sculpt toolbox: a floating strip over the viewport for the selected
// mesh. Twelve tools arm the sculpt brush, with per-brush algorithm pickers,
// composable surface feature filters, mirror symmetry toggles, brush radius
// and per-tool strength sliders, falloff, and stretch-split / stroke-end
// remesh toggles; the viewport paints the strokes.
// The mask group paints the protected region the brushes hold (invert and
// clear included); mask painting disarms the sculpt brush and vice versa.
import type { DyntopoMode, MeshHandle } from "fast-mesh-wasm";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { SculptFilterKind } from "../../../shared/ux/settings";
import { formatLength } from "../../../shared/ux/units";
import type { AppState } from "../../../app/state/app-state";
import {
  DYNTOPO_MODE_OPTIONS,
  FILTER_KIND_OPTIONS,
  REMESH_TRI_LIMIT,
  remeshTooBig,
  type SculptTool,
} from "../ux/sculpt";
import {
  formatReadout,
  methodConfigFor,
  SCULPT_TOOLS,
  useMaskTools,
  useSculptToolbox,
} from "../ux/use-sculpt-toolbox";

export type { SculptTool };

export function SculptToolbox(props: { app: AppState; selection?: MeshHandle | null }) {
  const { app } = props;
  const {
    eligibleItems,
    targetItem,
    targetMesh,
    armed,
    activeTool,
    radius,
    setRadius,
    strengths,
    setStrength,
    falloff,
    setFalloff,
    pinToActive,
    remesh,
    liveRemesh,
    dyntopo,
    dyntopoMode,
    dyntopoDetail,
    methods,
    grabPoisson,
    filter,
    symmetry,
    units,
    pickTool,
    remeshRoi,
  } = useSculptToolbox(app, props.selection);

  if (eligibleItems.length === 0) {
    return null;
  }
  const targetRadius = targetItem?.radius ?? 1;
  // The optional remesh stands down on huge meshes, where it would stall the worker.
  const tooBig = targetItem !== null && remeshTooBig(targetItem.triangles);
  const remeshTitle = tooBig
    ? `Stands down past ${REMESH_TRI_LIMIT.toLocaleString("en")} triangles: a remesh would stall the worker`
    : null;
  const strength = strengths[activeTool];
  const methodUi = methodConfigFor(activeTool, methods, app);

  return (
    <div className="toolbox" role="toolbar" aria-label="Sculpt tools">
      <div className="group">
        {SCULPT_TOOLS.map((t) => (
          <button
            key={t.tool}
            type="button"
            data-action={`sculpt-tool-${t.tool}`}
            aria-pressed={armed && activeTool === t.tool}
            disabled={targetMesh === null}
            title={
              targetMesh === null
                ? "Select an unlocked visible mesh or uncheck Pin to active mesh to sculpt"
                : t.title
            }
            onClick={() => pickTool(t.tool)}
          >
            {t.label}
          </button>
        ))}
      </div>
      {methodUi !== null ? (
        <div className="toolbox-section">
          <label className="field" title="Brush kernel algorithm">
            method
            <select
              data-action="sculpt-method"
              value={methodUi.value}
              onChange={(e) => methodUi.onChange(e.currentTarget.value)}
            >
              {methodUi.options.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          <p className="method-desc" data-role="sculpt-method-description">
            {methodUi.description}
          </p>
          {activeTool === "grab" && methods.grab === "kelvinlet" ? (
            <label className="slider" title="Elastic Poisson ratio (0 compressible .. 0.5 incompressible)">
              poisson <output>{formatReadout(grabPoisson)}</output>
              <input
                type="range"
                data-action="sculpt-grab-poisson"
                min={0}
                max={0.5}
                step={0.01}
                value={grabPoisson}
                onChange={(e) => app.setSculptGrabPoisson(Number(e.currentTarget.value))}
              />
            </label>
          ) : null}
        </div>
      ) : null}
      <label className="slider" title="Brush radius slider">
        radius <output>{formatLength(radius, units)}</output>
        <input
          type="range"
          data-action="sculpt-radius"
          min={0.01}
          max={Math.max(targetRadius, 0.02)}
          step={0.01}
          value={Math.min(radius, Math.max(targetRadius, 0.02))}
          onChange={(e) => setRadius(Number(e.currentTarget.value))}
        />
      </label>
      <label className="field" title="Brush radius">
        radius ({units})
        <LengthInput min={0} mm={radius} unit={units} onMm={(mm) => setRadius(Math.max(0.01, mm))} />
      </label>
      <label
        className="slider"
        title={
          activeTool === "grab"
            ? "How closely the surface follows the pointer"
            : activeTool === "twist"
              ? "Twist angle per dab (radians)"
              : "Strength of one dab"
        }
      >
        strength <output>{formatReadout(strength)}</output>
        <input
          type="range"
          data-action="sculpt-strength"
          min={0}
          max={1}
          step={0.01}
          value={strength}
          onChange={(e) => setStrength(Number(e.currentTarget.value))}
        />
      </label>
      <label className="field" title="How the strength falls off with distance">
        falloff
        <select
          data-action="sculpt-falloff"
          value={falloff}
          onChange={(e) => setFalloff(e.currentTarget.value as "linear" | "smooth")}
        >
          <option value="smooth">smooth</option>
          <option value="linear">linear</option>
        </select>
      </label>
      <div className="toolbox-section">
        <label className="field" title="Composable per-vertex surface feature filter">
          filter
          <select
            data-action="sculpt-filter"
            value={filter.kind}
            onChange={(e) => app.setSculptFilter({ kind: e.currentTarget.value as SculptFilterKind })}
          >
            {FILTER_KIND_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
        {filter.kind === "peaksOnly" || filter.kind === "valleysOnly" ? (
          <>
            <label className="slider" title="Minimum prominence threshold before the brush engages">
              threshold <output>{formatReadout(filter.threshold)}</output>
              <input
                type="range"
                data-action="sculpt-filter-threshold"
                min={0}
                max={1}
                step={0.01}
                value={filter.threshold}
                onChange={(e) => app.setSculptFilter({ threshold: Number(e.currentTarget.value) })}
              />
            </label>
            <label className="slider" title="Sharpness of the filter ramp">
              contrast <output>{formatReadout(filter.contrast)}</output>
              <input
                type="range"
                data-action="sculpt-filter-contrast"
                min={0}
                max={1}
                step={0.01}
                value={filter.contrast}
                onChange={(e) => app.setSculptFilter({ contrast: Number(e.currentTarget.value) })}
              />
            </label>
          </>
        ) : null}
        {filter.kind === "normalAngle" ? (
          <label className="slider" title="Maximum surface normal cone half-angle (degrees)">
            angle <output>{formatReadout(filter.maxAngleDeg)}°</output>
            <input
              type="range"
              data-action="sculpt-filter-angle"
              min={5}
              max={180}
              step={1}
              value={filter.maxAngleDeg}
              onChange={(e) => app.setSculptFilter({ maxAngleDeg: Number(e.currentTarget.value) })}
            />
          </label>
        ) : null}
      </div>
      <div className="group-3" role="group" aria-label="Mirror symmetry">
        <button
          type="button"
          data-action="sculpt-sym-x"
          aria-pressed={symmetry.x}
          title="Mirror stroke across local X = 0 plane"
          onClick={() => app.setSculptSymmetry({ x: !symmetry.x })}
        >
          Sym X
        </button>
        <button
          type="button"
          data-action="sculpt-sym-y"
          aria-pressed={symmetry.y}
          title="Mirror stroke across local Y = 0 plane"
          onClick={() => app.setSculptSymmetry({ y: !symmetry.y })}
        >
          Sym Y
        </button>
        <button
          type="button"
          data-action="sculpt-sym-z"
          aria-pressed={symmetry.z}
          title="Mirror stroke across local Z = 0 plane"
          onClick={() => app.setSculptSymmetry({ z: !symmetry.z })}
        >
          Sym Z
        </button>
      </div>
      <label
        className="check"
        title={
          remeshTitle ??
          "Remesh the stroked region when the stroke ends. Rewrites everything in the region, including what the stroke did not touch"
        }
      >
        <input
          type="checkbox"
          data-action="sculpt-remesh"
          checked={remesh}
          disabled={tooBig}
          onChange={(e) => app.setSculptRemesh(e.currentTarget.checked)}
        />
        even out after stroke
      </label>
      <label
        className="check"
        title="Split triangles once the stroke stretches them past their limit. Everything else stays exactly as it is"
      >
        <input
          type="checkbox"
          data-action="sculpt-live-remesh"
          checked={liveRemesh}
          onChange={(e) => app.setSculptLiveRemesh(e.currentTarget.checked)}
        />
        split stretched triangles
      </label>
      <label
        className="check"
        title="Restrict sculpt and mask strokes to the selected mesh only (Alt+Click any mesh to quick-pin)"
      >
        <input
          type="checkbox"
          data-action="sculpt-pin-mesh"
          checked={pinToActive}
          onChange={(e) => app.setSculptPinToActive(e.currentTarget.checked)}
        />
        Pin to active mesh
      </label>
      <div className="toolbox-section" role="group" aria-label="Dynamic topology">
        <button
          type="button"
          data-action="sculpt-dyntopo"
          aria-pressed={dyntopo}
          title="Dynamic Topology: live edge splitting, collapsing, and valence equalization during sculpt strokes"
          onClick={() => app.setSculptDyntopo(!dyntopo)}
        >
          Dyntopo
        </button>
        {dyntopo ? (
          <>
            <label className="field" title="Dynamic topology detail policy">
              mode
              <select
                data-action="sculpt-dyntopo-mode"
                value={dyntopoMode}
                onChange={(e) => app.setSculptDyntopoMode(e.currentTarget.value as DyntopoMode)}
              >
                {DYNTOPO_MODE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="slider" title="Target detail size (fraction of brush or mesh radius)">
              detail{" "}
              <output>
                {dyntopoMode === "brush-relative"
                  ? `${Math.round(dyntopoDetail * 100)}%`
                  : formatLength(dyntopoDetail * targetRadius, units)}
              </output>
              <input
                type="range"
                data-action="sculpt-dyntopo-detail"
                min={0.04}
                max={0.8}
                step={0.01}
                value={Math.min(0.8, Math.max(0.04, dyntopoDetail))}
                onChange={(e) => app.setSculptDyntopoDetail(Number(e.currentTarget.value))}
              />
            </label>
            <button
              type="button"
              data-action="sculpt-dyntopo-apply"
              disabled={targetItem === null}
              title="Clean/subdivide local Dyntopo ROI around brush center or mesh center"
              onClick={remeshRoi}
            >
              Remesh ROI
            </button>
          </>
        ) : null}
      </div>
      <MaskTools app={app} targetMesh={targetMesh} radius={targetRadius} />
    </div>
  );
}

/**
 * The mask group: paint the protected region the deform brushes hold, with
 * invert and clear. Paint accumulates sphere-brush picks on the selected
 * mesh (a new mesh replaces the mask); invert and clear work in place over
 * the mirrored views, failing loudly when a topology edit took a masked
 * vertex. The counts come from the app; the notice is the loud half of the
 * mask contract (see `features/sculpt/ux/sculpt.ts`).
 */
function MaskTools(props: { app: AppState; targetMesh: MeshHandle | null; radius: number }) {
  const { app, targetMesh, radius } = props;
  const { mask, painting, maskRadius, notice, shown, units, onInvert, togglePaint } = useMaskTools(
    app,
    targetMesh,
    radius,
  );
  const sameMesh = mask !== null && targetMesh !== null && mask.mesh === targetMesh;

  return (
    <>
      <div className="group" aria-label="Mask tools">
        <button
          type="button"
          data-action="sculpt-mask-paint"
          aria-pressed={painting}
          disabled={targetMesh === null}
          title="Paint the protected region the brushes skip (disarms the sculpt brush)"
          onClick={togglePaint}
        >
          Mask
        </button>
        <button
          type="button"
          data-action="sculpt-mask-invert"
          title="Protect everything outside the mask instead"
          disabled={!sameMesh}
          onClick={onInvert}
        >
          Invert
        </button>
        <button
          type="button"
          data-action="sculpt-mask-clear"
          title="Drop the protected region"
          disabled={mask === null}
          onClick={() => app.setSculptMask(null)}
        >
          Clear
        </button>
        <button
          type="button"
          data-action="sculpt-mask-show"
          title="Show the mask's tint on the surface while a tool is armed"
          aria-pressed={shown}
          onClick={() => app.setSculptMaskShown(!shown)}
        >
          Show
        </button>
        <span title={sameMesh ? `${mask.vertices.length} protected vertices` : undefined}>
          {sameMesh ? `${mask.vertices.length} masked` : mask === null ? "no mask" : "masked elsewhere"}
        </span>
      </div>
      <label className="slider" title="Mask paint radius">
        mask radius <output>{formatLength(maskRadius, units)}</output>
        <input
          type="range"
          data-action="sculpt-mask-radius"
          min={0.01}
          max={Math.max(radius, 0.02)}
          step={0.01}
          value={Math.min(maskRadius, Math.max(radius, 0.02))}
          onChange={(e) => app.setMaskRadius(Number(e.currentTarget.value))}
        />
      </label>
      {notice !== null ? (
        <span role="status" title="The mask contract's loud half">
          {notice}
        </span>
      ) : null}
    </>
  );
}
