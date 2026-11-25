// The select panel's "Select by Criteria" block (#258): a criterion, its
// settings (lengths and areas in the display unit), and the button that
// selects its faces by the combine mode.
import { LengthInput } from "../../../shared/ui/LengthInput";
import { fromDisplayArea, type LengthUnit, toDisplayArea } from "../../../shared/ux/units";
import {
  BUILD_AXES,
  type BuildAxis,
  type CriteriaParams,
  CRITERION_OPTIONS,
  type FaceCriterion,
} from "../ux/select-criteria";

export interface SelectCriteriaSectionProps {
  criterion: FaceCriterion;
  params: CriteriaParams;
  unit: LengthUnit;
  disabled: boolean;
  onCriterion: (criterion: FaceCriterion) => void;
  onParams: (patch: Partial<CriteriaParams>) => void;
  onSelect: () => void;
}

function AngleInput(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  dataAttr: string;
  disabled: boolean;
  onChange: (deg: number) => void;
}) {
  return (
    <label className="field">
      {props.label} (deg)
      <input
        type="number"
        step="any"
        min={props.min}
        max={props.max}
        value={props.value}
        disabled={props.disabled}
        {...{ [props.dataAttr]: "" }}
        onChange={(e) => {
          const v = Number(e.currentTarget.value);
          if (Number.isFinite(v)) {
            props.onChange(Math.min(props.max, Math.max(props.min, v)));
          }
        }}
      />
    </label>
  );
}

export function SelectCriteriaSection(props: SelectCriteriaSectionProps) {
  const { criterion, params, unit, disabled, onParams } = props;
  return (
    <div className="op" data-select-criteria>
      <span>Select by Criteria</span>
      <label className="field">
        <span>Criterion</span>
        <select
          data-select-criterion
          value={criterion}
          disabled={disabled}
          onChange={(e) => props.onCriterion(e.currentTarget.value as FaceCriterion)}
        >
          {CRITERION_OPTIONS.map((opt) => (
            <option key={opt.id} value={opt.id}>
              {opt.label}
            </option>
          ))}
        </select>
      </label>

      {criterion === "area" ? (
        <>
          <label className="field">
            <span>Faces</span>
            <select
              data-select-area-mode
              value={params.areaMode}
              disabled={disabled}
              onChange={(e) => onParams({ areaMode: e.currentTarget.value as CriteriaParams["areaMode"] })}
            >
              <option value="below">smaller than</option>
              <option value="above">larger than</option>
            </select>
          </label>
          <label className="field">
            ({unit}²)
            <input
              type="number"
              step="any"
              min={0}
              aria-label="Area threshold"
              data-select-area-threshold
              disabled={disabled}
              value={unit === "mm" ? params.areaThreshold : Number(toDisplayArea(params.areaThreshold, unit).toPrecision(12))}
              onChange={(e) => onParams({ areaThreshold: fromDisplayArea(Math.max(0, Number(e.currentTarget.value) || 0), unit) })}
            />
          </label>
        </>
      ) : null}

      {criterion === "facing" ? (
        <AngleInput
          label="Within"
          value={params.facingAngleDeg}
          min={0}
          max={180}
          dataAttr="data-select-facing-angle"
          disabled={disabled}
          onChange={(facingAngleDeg) => onParams({ facingAngleDeg })}
        />
      ) : null}

      {criterion === "overhang" ? (
        <>
          <AngleInput
            label="Past"
            value={params.overhangAngleDeg}
            min={0}
            max={90}
            dataAttr="data-select-overhang-angle"
            disabled={disabled}
            onChange={(overhangAngleDeg) => onParams({ overhangAngleDeg })}
          />
          <label className="field">
            <span>Build up</span>
            <select
              data-select-build-axis
              value={params.buildAxis}
              disabled={disabled}
              onChange={(e) => onParams({ buildAxis: e.currentTarget.value as BuildAxis })}
            >
              {BUILD_AXES.map((axis) => (
                <option key={axis} value={axis}>
                  {axis.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
        </>
      ) : null}

      {criterion === "degenerate" ? (
        <label className="field">
          Sliver aspect at least
          <input
            type="number"
            step="any"
            min={1}
            data-select-sliver-aspect
            disabled={disabled}
            value={params.sliverAspect}
            onChange={(e) => {
              const v = Number(e.currentTarget.value);
              if (Number.isFinite(v) && v >= 1) {
                onParams({ sliverAspect: v });
              }
            }}
          />
        </label>
      ) : null}

      {criterion === "sharp" ? (
        <AngleInput
          label="Dihedral at least"
          value={params.sharpAngleDeg}
          min={0}
          max={180}
          dataAttr="data-select-sharp-angle"
          disabled={disabled}
          onChange={(sharpAngleDeg) => onParams({ sharpAngleDeg })}
        />
      ) : null}

      {criterion === "contact" ? (
        <label className="field">
          Within ({unit})
          <LengthInput
            aria-label="Contact margin"
            data-select-contact-margin
            min={0}
            mm={params.contactMargin}
            unit={unit}
            disabled={disabled}
            onMm={(mm) => onParams({ contactMargin: Math.max(0, mm) })}
          />
        </label>
      ) : null}

      {criterion === "boundary" ? <span className="muted">Uses the adjacency above.</span> : null}

      <button type="button" className="primary" data-select-action="by-criterion" disabled={disabled} onClick={props.onSelect}>
        Select Faces
      </button>
    </div>
  );
}
