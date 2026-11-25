// The repair panel's wrap section (#263): the gap that closes, the voxel size
// (by detail or directly), projection onto the input, the grid and memory the
// settings take, and the run on the selected meshes as one new mesh. What it
// does lives in `ux/use-wrap.ts`.
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { LengthUnit } from "../../../shared/ux/units";
import { useWrap } from "../ux/use-wrap";
import { describeWrapPlan, MAX_WRAP_DETAIL, MIN_WRAP_DETAIL, wrapDetailInput, type WrapSizing } from "../ux/wrap";

export function WrapSection(props: {
  app: AppState;
  store: SceneStore;
  items: readonly Item[];
  /** The selected mesh. */
  activeItem: Item | null;
  units: LengthUnit;
  /** Whether another repair runs. */
  disabled: boolean;
}) {
  const { activeItem, units } = props;
  const w = useWrap(props);
  const disabled = props.disabled || w.busy;

  return (
    <div className="op" data-wrap-section>
      <h3>Wrap</h3>
      <p className="muted small" data-wrap-sources>
        {w.sources.length > 1
          ? `A closed envelope around ${w.sources.length} selected meshes, as a new mesh.`
          : `A closed envelope around ${activeItem?.name ?? "the selection"}, as a new mesh. Shift-click to add meshes.`}
      </p>
      <div className="op-row">
        <label className="field" title="Gaps and holes narrower than this are closed">
          gap ({units})
          <LengthInput
            aria-label="Wrap gap"
            data-wrap-gap
            min={0}
            mm={w.gap}
            unit={units}
            disabled={disabled}
            onMm={(mm) => w.update({ gap: Math.max(0, mm) })}
          />
        </label>
        <button
          type="button"
          data-wrap-gap-auto
          aria-pressed={w.gapIsAuto}
          disabled={disabled}
          title="Two percent of the selection's size"
          onClick={() => w.update({ gap: null })}
        >
          Auto
        </button>
      </div>
      <div className="op-row">
        <label className="field">
          size by
          <select
            aria-label="Wrap voxel sizing"
            data-wrap-sizing
            value={w.settings.sizing}
            disabled={disabled}
            onChange={(e) => {
              const sizing = e.currentTarget.value as WrapSizing;
              // A direct voxel size starts at the one the detail gave.
              w.update(sizing === "voxel" ? { sizing, voxelSize: w.voxelSize } : { sizing });
            }}
          >
            <option value="detail">Detail</option>
            <option value="voxel">Voxel size</option>
          </select>
        </label>
        {w.settings.sizing === "detail" ? (
          <label className="field" title="Voxels along the longest side of the selection">
            detail
            <input
              type="number"
              aria-label="Wrap detail"
              data-wrap-detail
              min={MIN_WRAP_DETAIL}
              max={MAX_WRAP_DETAIL}
              step={8}
              value={w.settings.detail}
              disabled={disabled}
              onChange={(e) => w.update({ detail: wrapDetailInput(e.currentTarget.value) })}
            />
          </label>
        ) : (
          <label className="field">
            voxel ({units})
            <LengthInput
              aria-label="Wrap voxel size"
              data-wrap-voxel
              min={0}
              mm={w.voxelSize}
              unit={units}
              disabled={disabled}
              onMm={(mm) => w.update({ voxelSize: mm > 0 ? mm : null })}
            />
          </label>
        )}
        <label className="check" title="Move the result back onto the meshes where that keeps it clean">
          <input
            type="checkbox"
            data-wrap-project
            checked={w.settings.project}
            disabled={disabled}
            onChange={(e) => w.update({ project: e.currentTarget.checked })}
          />
          Project onto input
        </label>
      </div>
      {w.planError !== null ? (
        <p className="report error" data-wrap-plan role="alert">
          {w.planError}
        </p>
      ) : w.plan !== null ? (
        <p className={w.plan.fits ? "muted small" : "report error"} data-wrap-plan>
          {describeWrapPlan(w.plan, units)}
        </p>
      ) : null}
      <button
        type="button"
        data-wrap-run
        disabled={disabled || w.blocked}
        title="Build a closed, manifold envelope of the selected meshes as a new mesh (one undo step)"
        onClick={() => void w.run()}
      >
        Wrap
      </button>
      {w.busy ? (
        <p className="busy small" data-wrap-busy>
          Wrap: computing… <span className="progress" aria-hidden="true" />
        </p>
      ) : w.report !== null ? (
        <p className={w.report.error ? "report error" : "report"} data-wrap-report role="status">
          {w.report.text}
        </p>
      ) : null}
    </div>
  );
}
