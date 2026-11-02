// The shortcut overlay (`?`) and the first-run tour card: text-first, no new
// dependencies. Anchors are `[data-tour]` attributes on the panels and the
// viewport toolbar; the active one gets `.tour-spot` while its step shows.
import { useEffect } from "react";
import { TOUR_STEPS } from "./tour";

// Exported for tests: the overlay must name every binding in App's keyboard
// handler (T/R, Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y, F, W, Esc, ?).
export const SHORTCUTS: Array<[string, string]> = [
  ["T / R", "Move / rotate gizmo"],
  ["Ctrl+Z · Ctrl+Shift+Z (Ctrl+Y)", "Undo / redo"],
  ["F", "Frame the selection"],
  ["W", "Wireframe every mesh"],
  ["Esc", "Disarm, cancel the pick, deselect"],
  ["?", "This help"],
];

export function HelpOverlay(props: { onClose: () => void; onStartTour: () => void }) {
  const { onClose, onStartTour } = props;
  return (
    <div
      className="help-scrim"
      data-help-overlay
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <section className="help-card" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
        <header className="panel-head">
          <h2>Shortcuts</h2>
          <button type="button" data-action="help-close" onClick={onClose}>
            Close
          </button>
        </header>
        <dl className="keys">
          {SHORTCUTS.map(([keys, what]) => (
            <div key={keys}>
              <dt>{keys}</dt>
              <dd>{what}</dd>
            </div>
          ))}
        </dl>
        <p className="muted small">New here? The tour walks the mesh list, the gizmo, repair and collision.</p>
        <button type="button" className="primary" data-action="tour-start" onClick={onStartTour}>
          Take the tour
        </button>
      </section>
    </div>
  );
}

export function TourCard(props: { index: number; onNext: () => void; onBack: () => void; onDone: () => void }) {
  const { index, onNext, onBack, onDone } = props;
  const step = TOUR_STEPS[index];
  const last = index === TOUR_STEPS.length - 1;

  // Ring the step's anchor while it shows.
  useEffect(() => {
    const el = document.querySelector(`[data-tour="${step.target}"]`);
    el?.classList.add("tour-spot");
    el?.scrollIntoView({ block: "nearest" });
    return () => el?.classList.remove("tour-spot");
  }, [step]);

  if (step === undefined) {
    return null;
  }
  return (
    <section className="tour-card" role="dialog" aria-label={`Tour: ${step.title}`} data-tour-card data-tour-step={index}>
      <header className="panel-head">
        <h2>
          {index + 1} of {TOUR_STEPS.length} · {step.title}
        </h2>
        <button type="button" data-action="tour-skip" onClick={onDone} title="Dismiss the tour">
          Skip
        </button>
      </header>
      <p>{step.body}</p>
      <div className="tour-nav">
        <button type="button" data-action="tour-back" onClick={onBack} disabled={index === 0}>
          Back
        </button>
        <button type="button" className="primary" data-action={last ? "tour-done" : "tour-next"} onClick={onNext}>
          {last ? "Done" : "Next"}
        </button>
      </div>
    </section>
  );
}
