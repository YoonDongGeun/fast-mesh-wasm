// The Surface Stamp section's text fields (#260), shown while the stamp is
// Text: the text, its letter height, depth and letter spacing in the
// display unit (the same text the Text dialog edits), and a line saying
// what the stamp will do to the surface with the section's settings.
import { type AppState, useApp } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { textParamsProblem } from "../ux/text-mesh";
import { textStampEffect } from "../ux/text-stamp";
import { textDialogOf } from "../ux/text-dialog-state";
import { updateText } from "../ux/use-text-tool";

export function TextStampFields(props: { app: AppState }) {
  const { app } = props;
  const place = useApp(app, (s) => s.wave6.printPrep.placeParams);
  const units = useApp(app, (s) => s.units);
  const t = place.text;
  const problem = textParamsProblem(t);
  const effect = problem === null ? textStampEffect(place, units) : null;
  const note = problem ?? effect?.text ?? null;
  return (
    <div data-modeling-place-text>
      <div className="op-row">
        <label className="field">
          <span>Text</span>
          <textarea
            rows={1}
            spellCheck={false}
            value={t.text}
            data-modeling-place-text-value
            onChange={(e) => updateText(app, { text: e.currentTarget.value })}
          />
        </label>
        <label className="field">
          <span>Letter height</span>
          <LengthInput
            mm={t.size}
            unit={units}
            min={0}
            data-modeling-place-text-size
            onMm={(mm) => updateText(app, { size: mm })}
          />
        </label>
      </div>
      <div className="op-row">
        <label className="field">
          <span>Depth</span>
          <LengthInput
            mm={t.depth}
            unit={units}
            min={0}
            data-modeling-place-text-depth
            onMm={(mm) => updateText(app, { depth: mm })}
          />
        </label>
        <label className="field">
          <span>Spacing</span>
          <LengthInput
            mm={t.spacing}
            unit={units}
            min={0}
            data-modeling-place-text-spacing
            onMm={(mm) => updateText(app, { spacing: mm })}
          />
        </label>
        <button type="button" data-modeling-place-text-dialog onClick={() => textDialogOf(app).open()}>
          Text…
        </button>
      </div>
      {note !== null ? (
        <p
          className={problem !== null || effect?.problem === true ? "report error" : "muted small"}
          data-modeling-place-text-note
        >
          {note}
        </p>
      ) : null}
    </div>
  );
}
