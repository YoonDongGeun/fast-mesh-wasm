// The transform panel's placement helpers (#256): centre or bottom centre
// to the origin, lay flat on a face, align a direction to an axis, the 3-2-1
// datum alignment and apply transform, each one undo step. The surface
// picks arm here and land in the viewport (PlacementPicksLayer). The panel
// owns the behaviour (`usePlacementHelpers`), so a pick stands down even
// while no mesh is selected and this section is gone.
import type { Item } from "../../../app/state/app-state";
import type { SignedAxis, UpAxis } from "../ux/placement-math";
import { type PlacementPickKind, pickPrompt } from "../ux/placement-picks";
import type { PlacementHelpersState } from "../ux/use-placement-helpers";

const TARGETS: readonly SignedAxis[] = ["+x", "-x", "+y", "-y", "+z", "-z"];

export function PlacementHelpers(props: { helpers: PlacementHelpersState; selectedItem: Item }) {
  const { helpers: h, selectedItem } = props;
  const off = h.mesh === null || h.busy;
  const armButton = (kind: PlacementPickKind, action: string, label: string, title: string) => (
    <button
      type="button"
      data-placement-action={action}
      aria-pressed={h.pick?.kind === kind}
      disabled={off}
      title={title}
      onClick={() => (h.pick?.kind === kind ? h.cancel() : h.arm(kind))}
    >
      {label}
    </button>
  );

  return (
    <div className="op placement-helpers" data-placement>
      <h3>Placement</h3>
      <label className="field">
        Up
        <select
          data-placement-up
          aria-label="Up axis"
          value={h.up}
          disabled={h.pick !== null}
          onChange={(e) => h.setUp(e.currentTarget.value as UpAxis)}
        >
          <option value="z">+Z</option>
          <option value="y">+Y</option>
        </select>
      </label>
      <button
        type="button"
        data-placement-action="origin-center"
        disabled={off}
        title="Move the bounding box centre to the origin"
        onClick={() => h.toOrigin("center")}
      >
        Centre to Origin
      </button>
      <button
        type="button"
        data-placement-action="origin-bottom"
        disabled={off}
        title="Move the bounding box's bottom centre to the origin"
        onClick={() => h.toOrigin("bottom")}
      >
        Bottom to Origin
      </button>
      {armButton("layFlat", "lay-flat", "Lay Flat", "Click a face: it turns to face down and the mesh rests on the floor")}
      <label className="field">
        Align to
        <select
          data-placement-target
          aria-label="Align to axis"
          value={h.target}
          disabled={h.pick !== null}
          onChange={(e) => h.setTarget(e.currentTarget.value as SignedAxis)}
        >
          {TARGETS.map((t) => (
            <option key={t} value={t}>
              {t.toUpperCase()}
            </option>
          ))}
        </select>
      </label>
      {armButton("alignPoints", "align-points", "Two Points", "Click two points: their direction turns onto the axis")}
      {armButton("alignFace", "align-face", "Face Normal", "Click a face: its normal turns onto the axis")}
      {armButton(
        "datum321",
        "datum-321",
        "3-2-1",
        "Click three points on the primary plane, two along the secondary direction, then the origin",
      )}
      <button
        type="button"
        data-placement-action="apply-transform"
        disabled={off}
        title="Bake the placement into the vertices, so the placement becomes the identity"
        onClick={() => void h.bake()}
      >
        {h.busy ? "Applying..." : "Apply Transform"}
      </button>
      {selectedItem.locked ? <p className="muted">Unlock the mesh to place it.</p> : null}
      {h.pick !== null ? (
        <p className="muted" data-placement-prompt>
          {pickPrompt(h.pick)}{" "}
          <button type="button" data-placement-cancel onClick={h.cancel}>
            Cancel
          </button>
        </p>
      ) : null}
      {h.report !== null ? (
        <p className={h.report.error ? "report error" : "report"} data-placement-report>
          {h.report.text}
        </p>
      ) : null}
    </div>
  );
}
