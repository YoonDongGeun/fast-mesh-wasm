// The angle measure's readout in the inspect panel (#257): the hint while
// the tool is armed, the points picked so far, the angle and the arm
// lengths (in the unit setting), and Clear.
import { type AppState, type InspectTool, useApp } from "../../../app/state/app-state";
import { formatLength } from "../../../shared/ux/units";
import { angleDegrees, formatAngle } from "../ux/angle-measure";
import { rulerDistance } from "../ux/inspect-tools";

export function AngleReadout(props: { app: AppState; tool: InspectTool | null }) {
  const { app, tool } = props;
  const points = useApp(app, (s) => s.objectInfo.angle);
  const unit = useApp(app, (s) => s.units);
  if (tool !== "angle" && points.length === 0) {
    return null;
  }
  const done = points.length === 3;
  const degrees = done ? angleDegrees(points[0], points[1], points[2]) : null;
  return (
    <div className="readout" data-readout="angle">
      {tool === "angle" ? (
        <p className="muted small">Click three vertices: the second one is the angle's vertex. A fourth click starts again.</p>
      ) : null}
      {points.length > 0 ? (
        <p>
          {done ? (
            <>
              <strong data-angle-degrees>{degrees === null ? "no angle (an arm has no length)" : formatAngle(degrees)}</strong>
              {" · arms "}
              {formatLength(rulerDistance(points[0], points[1]), unit)}, {formatLength(rulerDistance(points[2], points[1]), unit)}
            </>
          ) : (
            `${points.length} of 3 points · click the ${points.length === 1 ? "vertex" : "second arm"}`
          )}{" "}
          <button type="button" data-action="angle-clear" title="Clear the angle" onClick={() => app.setObjectInfo({ angle: [] })}>
            Clear
          </button>
        </p>
      ) : null}
    </div>
  );
}
