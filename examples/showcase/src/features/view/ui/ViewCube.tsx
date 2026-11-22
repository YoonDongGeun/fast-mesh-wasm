// Interactive 3D ViewCube (faces/edges/corners + Persp/Ortho), synced XYZ basis
// triad, live metric/imperial scale bar, and 6-mode shading selector (#232).
import { useRef } from "react";
import { type AppState, type DiagnosticShadingMode, useApp } from "../../../app/state/app-state";
import { VIEW_CUBE_CORNERS, VIEW_CUBE_EDGES, VIEW_CUBE_FACES } from "../ux/navigation-hud";
import { snapView, useNavigationHud } from "../ux/use-navigation-hud";
import { DIAGNOSTIC_SHADING_OPTIONS } from "./shading";

/**
 * Corner 3D ViewCube + Persp/Ortho toggle + bottom-left XYZ triad & live metric scale bar (#232).
 */
export function ViewportNavigationHud(props: { app: AppState }) {
  const { app } = props;
  const projection = useApp(app, (s) => s.cameraProjection);
  const units = useApp(app, (s) => s.units);
  const shadingMode = useApp(app, (s) => s.shadingMode);
  const creaseAngleDeg = useApp(app, (s) => s.creaseAngleDeg);
  const cubeRef = useRef<HTMLDivElement>(null);

  const { triad, scaleBar } = useNavigationHud(units, cubeRef);

  return (
    <>
      <div className="view-cube" aria-label="3D Navigation Cube">
        <div className="view-cube-stage">
          <div className="view-cube-box" ref={cubeRef}>
            {VIEW_CUBE_FACES.map((face) => (
              <button
                key={face.id}
                type="button"
                className={`view-cube-face face-${face.id}`}
                data-viewcube-face={face.id}
                data-camera-preset={face.preset}
                title={face.title}
                onClick={() => snapView(app, face)}
              >
                {face.label}
              </button>
            ))}
          </div>
        </div>
        <div className="view-cube-corners" role="group" aria-label="Isometric view corners and edges">
          {VIEW_CUBE_CORNERS.slice(0, 4).map((corner) => (
            <button
              key={corner.id}
              type="button"
              className="view-cube-chip"
              data-viewcube-corner={corner.id}
              title={corner.title}
              onClick={() => snapView(app, corner)}
            >
              {corner.label}
            </button>
          ))}
          {VIEW_CUBE_EDGES.slice(0, 4).map((edge) => (
            <button
              key={edge.id}
              type="button"
              className="view-cube-chip edge"
              data-viewcube-edge={edge.id}
              title={edge.title}
              onClick={() => snapView(app, edge)}
            >
              {edge.label.replace("-", "/")}
            </button>
          ))}
        </div>
        <div className="camera-bar" role="toolbar" aria-label="Camera projection and diagnostic shading">
          <div className="group camera-group">
            <button
              type="button"
              data-action="camera-projection"
              aria-pressed={projection === "orthographic"}
              onClick={() => app.toggleCameraProjection()}
              title="Toggle Perspective / Orthographic projection"
            >
              {projection === "perspective" ? "Persp" : "Ortho"}
            </button>
            <label className="shading-select-label" title="Diagnostic viewport shading mode">
              <select
                data-action="shading-mode"
                value={shadingMode}
                aria-label="Diagnostic shading mode"
                onChange={(e) => app.setShadingMode(e.currentTarget.value as DiagnosticShadingMode)}
              >
                {DIAGNOSTIC_SHADING_OPTIONS.map((opt) => (
                  <option key={opt.mode} value={opt.mode}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {shadingMode === "autoCrease" ? (
            <label className="group crease-angle-control" title="Dihedral crease threshold angle">
              <span>Crease {Math.round(creaseAngleDeg)}°</span>
              <input
                type="range"
                data-action="crease-angle"
                min={5}
                max={90}
                step={1}
                value={creaseAngleDeg}
                onChange={(e) => app.setCreaseAngleDeg(Number(e.currentTarget.value))}
              />
            </label>
          ) : null}
        </div>
      </div>

      <div className="xyz-scale-hud" aria-label="Coordinate triad and scale bar">
        <div className="xyz-triad" title="World XYZ basis orientation">
          <svg viewBox="-30 -30 60 60" width="52" height="52" aria-hidden="true">
            <circle cx="0" cy="0" r="2.2" fill="#e6edf3" />
            {triad.map((t) => (
              <g key={t.axis}>
                <line x1="0" y1="0" x2={t.x} y2={t.y} stroke={t.color} strokeWidth="2.2" strokeLinecap="round" />
                <text
                  x={t.x * 1.24}
                  y={t.y * 1.24 + 3}
                  fill={t.color}
                  fontSize="9"
                  fontWeight="700"
                  textAnchor="middle"
                >
                  {t.axis}
                </text>
              </g>
            ))}
          </svg>
        </div>
        <div className="scale-bar" title={`Live scale bar (${scaleBar.label})`}>
          <span className="scale-bar-label">{scaleBar.label}</span>
          <span className="scale-bar-rule" style={{ width: `${Math.round(scaleBar.widthPx)}px` }} />
        </div>
      </div>
    </>
  );
}
