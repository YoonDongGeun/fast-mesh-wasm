// The Text dialog (#260): the text, its letter height, depth and letter
// spacing in the display unit, and its curve detail; the name, centre and
// rotation of a created mesh; Create, and Emboss or Engrave on a surface
// (which arm Place Mesh with the text and close the dialog). It floats over
// the viewport like the Create dialog, whose look it shares, so the live
// preview (TextPreviewLayer) stays in view. Opened from the Home and Mesh
// Edit shelves and the command palette; Esc inside it, Cancel or × close it.
import type { AppState } from "../../../app/state/app-state";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { useTextDialog } from "../ux/text-dialog-state";
import { MAX_CURVE_SEGMENTS } from "../ux/text-mesh";
import { useTextTool } from "../ux/use-text-tool";
import "../../home/ui/create-dialog.css";
import "./text-dialog.css";

const AXES = ["X", "Y", "Z"] as const;

export function TextDialog(props: { app: AppState }) {
  const { app } = props;
  const open = useTextDialog(app, (s) => s.open);
  const session = useTextDialog(app, (s) => s.session);
  if (!open) {
    return null;
  }
  // A fresh form per opening.
  return <TextForm key={session} app={app} />;
}

function TextForm(props: { app: AppState }) {
  const f = useTextTool(props.app);
  const busy = f.busy || !f.ready;
  return (
    <section
      className="create-dialog text-dialog"
      role="dialog"
      aria-modal="false"
      aria-label="Text"
      data-panel="text-tool"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          f.cancel();
        }
      }}
    >
      <header className="panel-head">
        <h2>Text</h2>
        <button type="button" className="icon" aria-label="Close the Text dialog" onClick={f.cancel}>
          ×
        </button>
      </header>
      <div className="create-fields" aria-label="Text parameters">
        <label className="field text-field">
          <span>Text</span>
          <textarea
            data-text-param="text"
            rows={2}
            spellCheck={false}
            value={f.text.text}
            onChange={(e) => f.setText({ text: e.currentTarget.value })}
          />
        </label>
        <label className="field">
          <span>Letter height ({f.units})</span>
          <LengthInput data-text-param="size" mm={f.text.size} unit={f.units} min={0} onMm={(mm) => f.setText({ size: mm })} />
        </label>
        <label className="field">
          <span>Depth ({f.units})</span>
          <LengthInput data-text-param="depth" mm={f.text.depth} unit={f.units} min={0} onMm={(mm) => f.setText({ depth: mm })} />
        </label>
        <label className="field">
          <span>Letter spacing ({f.units})</span>
          <LengthInput
            data-text-param="spacing"
            mm={f.text.spacing}
            unit={f.units}
            min={0}
            onMm={(mm) => f.setText({ spacing: mm })}
          />
        </label>
        <label className="field">
          <span>Curve segments</span>
          <input
            type="number"
            data-text-param="curveSegments"
            min={1}
            max={MAX_CURVE_SEGMENTS}
            step={1}
            value={f.text.curveSegments}
            onChange={(e) => f.setText({ curveSegments: Number(e.currentTarget.value) })}
          />
        </label>
      </div>
      <div className="create-place">
        <label className="field create-name">
          <span>Name</span>
          <input type="text" data-text-name value={f.name} onChange={(e) => f.setName(e.currentTarget.value)} />
        </label>
        <div className="create-vec" role="group" aria-label={`Centre (${f.units})`}>
          <span>Centre ({f.units})</span>
          {AXES.map((a, k) => (
            <LengthInput
              key={a}
              aria-label={`Centre ${a}`}
              data-text-centre={a.toLowerCase()}
              mm={f.centre[k]}
              unit={f.units}
              onMm={(mm) => f.setCentre(k, mm)}
            />
          ))}
        </div>
        <div className="create-vec" role="group" aria-label="Rotation (degrees)">
          <span>Rotation (°)</span>
          {AXES.map((a, k) => (
            <input
              key={a}
              type="number"
              step="any"
              aria-label={`Rotation ${a}`}
              data-text-rotation={a.toLowerCase()}
              value={f.rotationDeg[k]}
              onChange={(e) => f.setRotation(k, Number(e.currentTarget.value) || 0)}
            />
          ))}
        </div>
      </div>
      <p className={f.status.problem ? "create-problem small" : "muted small"} data-text-status role="status">
        {f.status.text}
      </p>
      <div className="op-row">
        <button type="button" className="primary" data-action="text-create" disabled={busy} onClick={() => void f.create()}>
          {f.busy ? "Creating…" : "Create"}
        </button>
        <button
          type="button"
          data-action="text-emboss"
          title="Raise the text out of a surface: click the surface to place it"
          disabled={busy}
          onClick={() => f.placeOnSurface("embedUnion")}
        >
          Emboss on Surface
        </button>
        <button
          type="button"
          data-action="text-engrave"
          title="Cut the text into a surface: click the surface to place it"
          disabled={busy}
          onClick={() => f.placeOnSurface("engraveSubtract")}
        >
          Engrave on Surface
        </button>
        <button type="button" data-action="text-cancel" onClick={f.cancel}>
          Cancel
        </button>
      </div>
    </section>
  );
}
