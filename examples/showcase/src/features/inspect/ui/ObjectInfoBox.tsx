// The object info box (#257) at the bottom of the left sidebar, collapsible:
// the selected mesh's triangles, vertices, edges, components, area, volume,
// average edge length and bounding box (world- or local-aligned), the
// bounding box and normals overlays, and the quality report export. Lengths
// follow the unit setting; the numbers come from `useObjectInfo`.
import type { Scene } from "fast-mesh-wasm";
import { useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { Fragment, useEffect, useState } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import {
  formatArea,
  formatLength,
  formatLengthValue,
  formatPoint,
  formatVolume,
  type LengthUnit,
} from "../../../shared/ux/units";
import { boxSize, type MeshInfo } from "../ux/object-info";
import { NORMALS_SCALE_RANGE, type NormalsMode } from "../ux/object-info-state";
import { MAX_NORMAL_LINES } from "../ux/overlay-geometry";
import { exportQualityReport, LARGE_MESH_TRIANGLES, objectInfoProbe, useObjectInfo } from "../ux/use-object-info";

const NORMALS_MODES: readonly { mode: NormalsMode; label: string; title: string }[] = [
  { mode: "off", label: "Off", title: "No normals overlay" },
  { mode: "face", label: "Face", title: "One line per face, from its centroid" },
  { mode: "vertex", label: "Vertex", title: "One line per vertex, along its normal" },
];

export function ObjectInfoBox(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const settings = useApp(app, (s) => s.objectInfo);
  const unit = useApp(app, (s) => s.units);
  const item = items.find((i) => i.mesh === selection) ?? null;
  const brush = useApp(app, (s) => s.sculpt !== null || s.maskPaint);
  const { info, measuring } = useObjectInfo(app, scene, item?.mesh ?? null, settings.open);
  const [message, setMessage] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    objectInfoProbe.mesh = info === null ? null : (item?.name ?? null);
    objectInfoProbe.info = info;
  });

  const exportReport = (format: "html" | "csv") => {
    if (item === null) {
      return;
    }
    setExporting(true);
    void exportQualityReport(app, scene, item.mesh, format).then((error) => {
      setMessage(error);
      setExporting(false);
    });
  };

  return (
    <section className="panel object-info" aria-label="Object info" data-panel="object-info">
      <header className="panel-head">
        <h2>Object info</h2>
        <button
          type="button"
          className="small"
          data-action="object-info-toggle"
          aria-expanded={settings.open}
          title={settings.open ? "Collapse the object info" : "Show the selected mesh's numbers"}
          onClick={() => app.setObjectInfo({ open: !settings.open })}
        >
          {settings.open ? "Hide" : "Show"}
        </button>
      </header>
      {!settings.open ? null : item === null ? (
        <p className="muted small">Select a mesh to see its numbers.</p>
      ) : (
        <>
          <div className="modes" role="group" aria-label="Info frame">
            {(["world", "local"] as const).map((space) => (
              <button
                key={space}
                type="button"
                data-info-space={space}
                aria-pressed={settings.space === space}
                title={space === "world" ? "Bounding box along the world axes" : "Bounding box along the mesh's own axes"}
                onClick={() => app.setObjectInfo({ space })}
              >
                {space === "world" ? "World" : "Local"}
              </button>
            ))}
          </div>
          {info === null ? (
            <p className="muted small">Reading the mesh…</p>
          ) : (
            <InfoReadouts info={info} measuring={measuring} space={settings.space} unit={unit} />
          )}
          {brush && info !== null && info.triangles > LARGE_MESH_TRIANGLES ? (
            <p className="muted small">Large mesh: the numbers refresh once the brush is put down.</p>
          ) : null}
          <div className="object-info-overlays">
            <label className="check">
              <input
                type="checkbox"
                data-object-info-box
                checked={settings.showBox}
                onChange={(e) => app.setObjectInfo({ showBox: e.currentTarget.checked })}
              />
              Bounding box
            </label>
            <div className="modes" role="group" aria-label="Normals overlay">
              {NORMALS_MODES.map((m) => (
                <button
                  key={m.mode}
                  type="button"
                  data-normals-mode={m.mode}
                  aria-pressed={settings.normals === m.mode}
                  title={m.title}
                  onClick={() => app.setObjectInfo({ normals: m.mode })}
                >
                  {m.mode === "off" ? "Normals off" : m.label}
                </button>
              ))}
            </div>
            {settings.normals !== "off" ? (
              <label className="slider">
                <span className="slider-head">
                  normal length <output>{formatLength(settings.normalsScale * item.radius, unit)}</output>
                </span>
                <input
                  type="range"
                  aria-label="Normal length"
                  data-normals-length
                  min={NORMALS_SCALE_RANGE.min}
                  max={NORMALS_SCALE_RANGE.max}
                  step={NORMALS_SCALE_RANGE.step}
                  value={settings.normalsScale}
                  onChange={(e) => app.setObjectInfo({ normalsScale: Number(e.currentTarget.value) })}
                />
              </label>
            ) : null}
            {settings.normals !== "off" && info !== null && (settings.normals === "face" ? info.triangles : info.vertices) > MAX_NORMAL_LINES ? (
              <p className="muted small">Large mesh: a sample of {MAX_NORMAL_LINES.toLocaleString("en")} normals shows.</p>
            ) : null}
          </div>
          <div className="row">
            <button
              type="button"
              data-action="object-info-report-html"
              disabled={exporting}
              title="Export the info, the repair analysis and the shown map's readouts as an HTML page"
              onClick={() => exportReport("html")}
            >
              Report (HTML)
            </button>
            <button
              type="button"
              data-action="object-info-report-csv"
              disabled={exporting}
              title="Export the info, the repair analysis and the shown map's readouts as CSV"
              onClick={() => exportReport("csv")}
            >
              Report (CSV)
            </button>
          </div>
          {message !== null ? (
            <p className="error" role="alert">
              {message}
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}

/** A size vector as `10 × 20 × 5 mm`. */
function formatSize(size: readonly number[], unit: LengthUnit): string {
  return `${size.map((x) => formatLengthValue(x, unit)).join(" × ")} ${unit}`;
}

function InfoReadouts(props: { info: MeshInfo; measuring: boolean; space: "world" | "local"; unit: LengthUnit }) {
  const { info, measuring, space, unit } = props;
  const box = space === "world" ? info.world : info.local;
  const pending = measuring ? "measuring…" : "—";
  const volume =
    info.volume !== null
      ? formatVolume(info.volume, unit)
      : measuring
        ? pending
        : info.volumeReason === "open"
          ? "none (open mesh)"
          : info.volumeReason === "non-manifold"
            ? "none (non-manifold)"
            : "—";
  const rows: [string, string, string][] = [
    ["triangles", "Triangles", info.triangles.toLocaleString("en")],
    ["vertices", "Vertices", info.vertices.toLocaleString("en")],
    ["edges", "Edges", info.edges.toLocaleString("en")],
    ["components", "Components", info.components.toLocaleString("en")],
    ["area", "Area", info.area === null ? pending : formatArea(info.area, unit)],
    ["volume", "Volume", volume],
    ["mean-edge", "Avg edge", formatLength(info.meanEdge, unit)],
  ];
  if (box !== null) {
    rows.push(
      ["size", "Size", formatSize(boxSize(box), unit)],
      ["min", "Min", formatPoint(box.min, unit)],
      ["max", "Max", formatPoint(box.max, unit)],
    );
  }
  return (
    <dl className="object-info-readouts" data-object-info={space} aria-busy={measuring}>
      {rows.map(([key, label, value]) => (
        <Fragment key={key}>
          <dt>{label}</dt>
          {/* Until the measurement lands, the last one shows dimmed. */}
          <dd data-info={key} className={measuring && (key === "area" || key === "volume") ? "stale" : undefined}>
            {value}
          </dd>
        </Fragment>
      ))}
    </dl>
  );
}
