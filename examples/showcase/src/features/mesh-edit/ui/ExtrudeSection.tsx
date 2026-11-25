// The Modeling panel's extrude section (#262): Extrude Faces (the active
// mesh's face selection, by a distance along normals or an axis) and Extrude
// Base (the active mesh's boundary, to a flat, capped base). What it does
// lives in `ux/use-extrude.ts`.
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import type { LengthUnit } from "../../../shared/ux/units";
import type { SignedAxis } from "../../transform/ux/placement-math";
import { BASE_DIRECTIONS, type BasePlaneMode, FACE_DIRECTIONS } from "../ux/extrude";
import { useExtrude } from "../ux/use-extrude";

export function ExtrudeSection(props: {
  app: AppState;
  store: SceneStore;
  /** The selected mesh, else the first one. */
  activeItem: Item | null;
  units: LengthUnit;
}) {
  const { activeItem, units } = props;
  const x = useExtrude(props);
  const disabled = x.busy !== null || activeItem === null;
  const axis = x.baseDirection[1].toUpperCase();

  return (
    <>
      <div className="op" data-extrude-faces-section>
        <h3>Extrude Faces</h3>
        <p className="muted" data-extrude-faces-selection>
          {x.selectedCount > 0
            ? `${x.selectedCount.toLocaleString("en")} selected ${x.selectedCount === 1 ? "face" : "faces"} of ${activeItem?.name ?? "the mesh"}.`
            : `No faces selected on ${activeItem?.name ?? "the mesh"}. Pick some with the Select tools.`}
        </p>
        <div className="modes" role="group" aria-label="Extrude direction">
          {FACE_DIRECTIONS.map(([id, label, title]) => (
            <button
              key={id}
              type="button"
              data-extrude-faces-direction={id}
              aria-pressed={x.faceDirection === id}
              disabled={disabled}
              title={title}
              onClick={() => x.setFaceDirection(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="op-row">
          <label className="field">
            distance ({units})
            <LengthInput
              aria-label="Extrude distance"
              data-extrude-faces-distance
              mm={x.distance}
              unit={units}
              disabled={disabled}
              onMm={x.setDistance}
            />
          </label>
          <button
            type="button"
            data-extrude-faces-run
            disabled={disabled || x.facesBlocked}
            title="Negative distances push the faces in"
            onClick={() => void x.runFaces()}
          >
            Extrude Faces
          </button>
        </div>
      </div>

      <div className="op" data-extrude-base-section>
        <h3>Extrude Base</h3>
        <p className="muted">
          Extends {activeItem?.name ?? "the mesh"}&apos;s open boundary to a flat base and caps it.
        </p>
        <div className="op-row">
          <label className="field">
            direction
            <select
              aria-label="Extrude base direction"
              data-extrude-base-direction
              value={x.baseDirection}
              disabled={disabled}
              onChange={(e) => x.setBaseDirection(e.currentTarget.value as SignedAxis)}
            >
              {BASE_DIRECTIONS.map((d) => (
                <option key={d} value={d}>
                  {d.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            base
            <select
              aria-label="Extrude base plane"
              data-extrude-base-plane
              value={x.plane}
              disabled={disabled}
              onChange={(e) => x.setPlane(e.currentTarget.value as BasePlaneMode)}
            >
              <option value="below">Past the furthest point</option>
              <option value="at">At a coordinate</option>
            </select>
          </label>
          {x.plane === "below" ? (
            <label className="field">
              by ({units})
              <LengthInput
                aria-label="Base distance past the mesh"
                data-extrude-base-offset
                min={0}
                mm={x.offset}
                unit={units}
                disabled={disabled}
                onMm={(mm) => x.setOffset(Math.max(0, mm))}
              />
            </label>
          ) : (
            <label className="field">
              {axis} ({units})
              <LengthInput
                aria-label="Base coordinate"
                data-extrude-base-at
                mm={x.at}
                unit={units}
                disabled={disabled}
                onMm={x.setAt}
              />
            </label>
          )}
        </div>
        <div className="op-row">
          <label className="check">
            <input
              type="checkbox"
              data-extrude-base-all
              checked={x.allLoops}
              disabled={disabled}
              onChange={(e) => x.setAllLoops(e.currentTarget.checked)}
            />
            All boundary loops
          </label>
          <label className="check">
            <input
              type="checkbox"
              data-extrude-base-refine
              checked={x.refine}
              disabled={disabled}
              onChange={(e) => x.setRefine(e.currentTarget.checked)}
            />
            Refine cap
          </label>
          <button type="button" data-extrude-base-run disabled={disabled} onClick={() => void x.runBase()}>
            Extrude Base
          </button>
        </div>
        {x.busy !== null ? <p className="muted busy">{x.busy}...</p> : null}
        {x.report !== null ? (
          <p className={x.report.error ? "report error" : "report"} data-extrude-report>
            <strong>{x.report.label}:</strong> {x.report.text}
          </p>
        ) : null}
        {x.crossingCount > 0 ? (
          <button type="button" data-extrude-select-crossing disabled={disabled} onClick={x.selectCrossing}>
            Select {x.crossingCount.toLocaleString("en")} crossing {x.crossingCount === 1 ? "face" : "faces"}
          </button>
        ) : null}
      </div>
    </>
  );
}
