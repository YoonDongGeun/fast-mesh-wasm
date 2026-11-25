// The navigation cube box in the viewport's bottom-right corner (top right on
// a phone) (#232, #255): the corner and edge views, the live cube (faces snap
// to the six views) with orbit arrows around it and roll arrows above it, and
// the display toggles under it (scale indicator, axis basis, projection); the
// shading picker below. Bottom left (top left on a phone): the synced XYZ basis
// triad and the live metric/imperial scale bar, or a screen grid of the same step.
import { useRef } from "react";
import { type AppState, type DiagnosticShadingMode, useApp } from "../../../app/state/app-state";
import {
  nextScaleIndicator,
  PRESET_SHORTCUTS,
  scaleGridBackground,
  VIEW_CUBE_CORNERS,
  VIEW_CUBE_EDGES,
  VIEW_CUBE_FACES,
} from "../ux/navigation-hud";
import { orbitView, rollView, snapView, useNavigationHud, useViewCubeOpen } from "../ux/use-navigation-hud";
import { ORBIT_FINE_STEP_DEG, ORBIT_STEP_DEG, type OrbitDirection, type RollDirection } from "../ux/view-steps";
import { DIAGNOSTIC_SHADING_OPTIONS } from "./shading";

const ORBIT_ARROWS: Readonly<Record<OrbitDirection, { label: string; rotate: number }>> = {
  up: { label: "Orbit up", rotate: 0 },
  left: { label: "Orbit left", rotate: -90 },
  right: { label: "Orbit right", rotate: 90 },
  down: { label: "Orbit down", rotate: 180 },
};

const ROLL_LABELS: Readonly<Record<RollDirection, string>> = {
  ccw: "Roll 90° counter-clockwise",
  cw: "Roll 90° clockwise",
};

const SCALE_LABELS = { ruler: "Ruler", grid: "Grid", off: "Scale" } as const;
const SCALE_TITLES = {
  ruler: "Scale indicator: ruler (click for a grid)",
  grid: "Scale indicator: grid (click to hide)",
  off: "Scale indicator: hidden (click for a ruler)",
} as const;

function OrbitArrowIcon(props: { rotate: number }) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true" style={{ transform: `rotate(${props.rotate}deg)` }}>
      <path d="M8 3.5 L13 11.5 H3 Z" fill="currentColor" />
    </svg>
  );
}

function RollArrowIcon(props: { direction: RollDirection }) {
  return (
    <svg
      viewBox="0 0 16 16"
      aria-hidden="true"
      style={props.direction === "ccw" ? { transform: "scaleX(-1)" } : undefined}
    >
      <path d="M3.2 10 A5 5 0 1 1 12.8 10" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <path d="M10.4 9 L15.2 9 L12.8 13 Z" fill="currentColor" />
    </svg>
  );
}

function CubeIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path
        d="M8 1.8 L13.6 4.9 V11.1 L8 14.2 L2.4 11.1 V4.9 Z M2.4 4.9 L8 8 L13.6 4.9 M8 8 V14.2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** An orbit arrow: a quarter turn the way it points, 15° with Shift held. */
function OrbitButton(props: { app: AppState; direction: OrbitDirection }) {
  const { app, direction } = props;
  const { label, rotate } = ORBIT_ARROWS[direction];
  const title = `${label} ${ORBIT_STEP_DEG}° (Shift: ${ORBIT_FINE_STEP_DEG}°)`;
  return (
    <button
      type="button"
      className={`view-nav-btn orbit-${direction}`}
      data-view-orbit={direction}
      title={title}
      aria-label={title}
      onClick={(e) => orbitView(app, direction, e.shiftKey)}
    >
      <OrbitArrowIcon rotate={rotate} />
    </button>
  );
}

/** A roll arrow: a quarter turn of the picture about the view direction. */
function RollButton(props: { app: AppState; direction: RollDirection }) {
  const { app, direction } = props;
  return (
    <button
      type="button"
      className={`view-nav-btn roll-${direction}`}
      data-view-roll={direction}
      title={ROLL_LABELS[direction]}
      aria-label={ROLL_LABELS[direction]}
      onClick={() => rollView(app, direction)}
    >
      <RollArrowIcon direction={direction} />
    </button>
  );
}

/**
 * The navigation cube box (corner and edge views, cube, orbit and roll
 * arrows, display toggles) + shading picker, and the bottom-left XYZ triad
 * and scale indicator (#232, #255).
 */
export function ViewportNavigationHud(props: { app: AppState }) {
  const { app } = props;
  const projection = useApp(app, (s) => s.cameraProjection);
  const units = useApp(app, (s) => s.units);
  const shadingMode = useApp(app, (s) => s.shadingMode);
  const creaseAngleDeg = useApp(app, (s) => s.creaseAngleDeg);
  const scaleIndicator = useApp(app, (s) => s.scaleIndicator);
  const showAxisBasis = useApp(app, (s) => s.showAxisBasis);
  const cubeRef = useRef<HTMLDivElement>(null);
  const [open, toggleOpen] = useViewCubeOpen();

  const { triad, scaleBar } = useNavigationHud(units, cubeRef);

  return (
    <>
      {scaleIndicator === "grid" ? (
        <div
          className="scale-grid"
          data-scale-grid
          aria-hidden="true"
          style={scaleGridBackground(scaleBar.widthPx)}
        />
      ) : null}
      <div className="view-cube" aria-label="3D Navigation Cube">
        <div className="view-cube-panel" data-open={open}>
          <div className="view-cube-body">
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
            <div className="view-nav" role="group" aria-label="Orbit and roll the view">
              <RollButton app={app} direction="ccw" />
              <OrbitButton app={app} direction="up" />
              <RollButton app={app} direction="cw" />
              <OrbitButton app={app} direction="left" />
              <div className="view-cube-stage">
                <div className="view-cube-box" ref={cubeRef}>
                  {VIEW_CUBE_FACES.map((face) => (
                    <button
                      key={face.id}
                      type="button"
                      className={`view-cube-face face-${face.id}`}
                      data-viewcube-face={face.id}
                      data-camera-preset={face.preset}
                      title={face.preset === undefined ? face.title : `${face.title} (${PRESET_SHORTCUTS[face.preset]})`}
                      onClick={() => snapView(app, face)}
                    >
                      {face.label}
                    </button>
                  ))}
                </div>
              </div>
              <OrbitButton app={app} direction="right" />
              <OrbitButton app={app} direction="down" />
            </div>
          </div>
          <div className="view-toggles" role="toolbar" aria-label="View display">
            <div className="view-cube-body view-toggle-group">
              <button
                type="button"
                data-action="view-scale"
                data-scale-indicator={scaleIndicator}
                aria-pressed={scaleIndicator !== "off"}
                title={SCALE_TITLES[scaleIndicator]}
                onClick={() => app.setScaleIndicator(nextScaleIndicator(scaleIndicator))}
              >
                {SCALE_LABELS[scaleIndicator]}
              </button>
              <button
                type="button"
                data-action="view-axes"
                aria-pressed={showAxisBasis}
                title={showAxisBasis ? "Hide the XYZ axis basis" : "Show the XYZ axis basis"}
                onClick={() => app.setShowAxisBasis(!showAxisBasis)}
              >
                XYZ
              </button>
              <button
                type="button"
                data-action="camera-projection"
                aria-pressed={projection === "orthographic"}
                onClick={() => app.toggleCameraProjection()}
                title="Toggle Perspective / Orthographic projection"
              >
                {projection === "perspective" ? "Persp" : "Ortho"}
              </button>
            </div>
            <button
              type="button"
              className="view-cube-toggle"
              data-action="view-cube-toggle"
              aria-expanded={open}
              title={open ? "Fold the view cube" : "Show the view cube"}
              aria-label={open ? "Fold the view cube" : "Show the view cube"}
              onClick={toggleOpen}
            >
              <CubeIcon />
            </button>
          </div>
        </div>
        <div className="camera-bar" role="toolbar" aria-label="Diagnostic shading">
          <div className="group camera-group">
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

      {showAxisBasis || scaleIndicator !== "off" ? (
        <div className="xyz-scale-hud" aria-label="Coordinate triad and scale indicator">
          {showAxisBasis ? (
            <div className="xyz-triad" data-axis-basis title="World XYZ basis orientation">
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
          ) : null}
          {scaleIndicator === "ruler" ? (
            <div className="scale-bar" data-scale-bar title={`Live scale bar (${scaleBar.label})`}>
              <span className="scale-bar-label">{scaleBar.label}</span>
              <span className="scale-bar-rule" style={{ width: `${Math.round(scaleBar.widthPx)}px` }} />
            </div>
          ) : null}
          {scaleIndicator === "grid" ? (
            <div className="scale-bar" data-scale-bar title={`Screen grid: ${scaleBar.label} per cell at the orbit target`}>
              <span className="scale-bar-label">Grid {scaleBar.label}</span>
            </div>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
