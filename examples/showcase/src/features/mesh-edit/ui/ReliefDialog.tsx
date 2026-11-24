// The Image to Relief dialog (#267): the image (a file picker, or a drop
// anywhere on the page) with a small picture of it, the relief's size, height,
// base, samples and direction, its name, centre and rotation, and Create. It
// floats over the viewport like the Create and Text dialogs, whose look it
// shares, so the live preview (ReliefPreviewLayer) stays in view. Opened from
// the Mesh Edit shelf, the command palette or by dropping an image; Esc inside
// it, Cancel or × close it.
import { useEffect, useRef } from "react";
import type { AppState } from "../../../app/state/app-state";
import { IMAGE_ACCEPT } from "../../../io/height-image";
import { LengthInput } from "../../../shared/ui/LengthInput";
import { useReliefDialog } from "../ux/relief-dialog-state";
import { useReliefTool } from "../ux/use-relief-tool";
import "../../home/ui/create-dialog.css";
import "./relief-dialog.css";

const AXES = ["X", "Y", "Z"] as const;

export function ReliefDialog(props: { app: AppState }) {
  const { app } = props;
  const open = useReliefDialog(app, (s) => s.open);
  const session = useReliefDialog(app, (s) => s.session);
  if (!open) {
    return null;
  }
  // A fresh form per opening.
  return <ReliefForm key={session} app={app} />;
}

/** The image's brightness, small. */
function Thumbnail(props: { grey: Float32Array; width: number; height: number }) {
  const { grey, width, height } = props;
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (ctx === null || ctx === undefined) {
      return;
    }
    const data = ctx.createImageData(width, height);
    for (let i = 0; i < grey.length; i++) {
      data.data.fill(grey[i], 4 * i, 4 * i + 3);
      data.data[4 * i + 3] = 255;
    }
    ctx.putImageData(data, 0, 0);
  }, [grey, width, height]);
  return <canvas ref={canvas} className="relief-thumbnail" width={width} height={height} aria-hidden="true" />;
}

function ReliefForm(props: { app: AppState }) {
  const f = useReliefTool(props.app);
  const { image, params } = f;
  return (
    <section
      className="create-dialog relief-dialog"
      role="dialog"
      aria-modal="false"
      aria-label="Image to relief"
      data-panel="image-to-relief"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          f.cancel();
        }
      }}
    >
      <header className="panel-head">
        <h2>Image to relief</h2>
        <button type="button" className="icon" aria-label="Close the Image to Relief dialog" onClick={f.cancel}>
          ×
        </button>
      </header>
      <div className="relief-image">
        {image === null ? null : <Thumbnail {...image.thumbnail} />}
        <label className="relief-file">
          <span data-relief-image>
            {image === null ? "Drop an image here, or choose one" : `${image.name} · ${image.width} × ${image.height} px · ${image.depth}-bit`}
          </span>
          <input
            type="file"
            data-relief-file
            accept={`${IMAGE_ACCEPT},.json`}
            multiple
            onChange={(e) => {
              const { files } = e.currentTarget;
              if (files !== null && files.length > 0) {
                f.pick([...files]);
              }
              // The same file again is a new choice.
              e.currentTarget.value = "";
            }}
          />
        </label>
      </div>
      {f.note === null ? null : (
        <p className="muted small" data-relief-note>
          {f.note}
        </p>
      )}
      {params === null ? null : (
        <>
          <div className="create-fields" aria-label="Relief parameters">
            <label className="field">
              <span>Width ({f.units})</span>
              <LengthInput
                data-relief-param="width"
                mm={params.width}
                unit={f.units}
                min={0}
                onMm={(width) => f.setParams({ width })}
              />
            </label>
            <label className="field">
              <span>Height of white ({f.units})</span>
              <LengthInput
                data-relief-param="maxHeight"
                mm={params.maxHeight}
                unit={f.units}
                min={0}
                onMm={(maxHeight) => f.setParams({ maxHeight })}
              />
            </label>
            <label className="field">
              <span>Base thickness ({f.units})</span>
              <LengthInput
                data-relief-param="base"
                mm={params.base}
                unit={f.units}
                min={0}
                onMm={(base) => f.setParams({ base })}
              />
            </label>
            <label className="field">
              <span>Samples, longer side</span>
              <input
                type="number"
                data-relief-param="samples"
                min={2}
                max={f.maxSamples}
                step={1}
                value={params.samples}
                onChange={(e) => f.setParams({ samples: Number(e.currentTarget.value) })}
              />
            </label>
            <label className="check">
              <input
                type="checkbox"
                data-relief-param="invert"
                checked={params.invert}
                onChange={(e) => f.setParams({ invert: e.currentTarget.checked })}
              />
              Dark is high
            </label>
          </div>
          <div className="create-place">
            <label className="field create-name">
              <span>Name</span>
              <input type="text" data-relief-name value={f.name} onChange={(e) => f.setName(e.currentTarget.value)} />
            </label>
            <div className="create-vec" role="group" aria-label={`Centre (${f.units})`}>
              <span>Centre ({f.units})</span>
              {AXES.map((a, k) => (
                <LengthInput
                  key={a}
                  aria-label={`Centre ${a}`}
                  data-relief-centre={a.toLowerCase()}
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
                  data-relief-rotation={a.toLowerCase()}
                  value={f.rotationDeg[k]}
                  onChange={(e) => f.setRotation(k, Number(e.currentTarget.value) || 0)}
                />
              ))}
            </div>
          </div>
        </>
      )}
      <p className={f.status.problem ? "create-problem small" : "muted small"} data-relief-status role="status">
        {f.status.text}
      </p>
      <div className="op-row">
        <button
          type="button"
          className="primary"
          data-action="relief-create"
          disabled={!f.ready || f.busy}
          onClick={() => void f.create()}
        >
          {f.busy ? "Creating…" : "Create"}
        </button>
        <button type="button" data-action="relief-cancel" onClick={f.cancel}>
          Cancel
        </button>
      </div>
    </section>
  );
}
