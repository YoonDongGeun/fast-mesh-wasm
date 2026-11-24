// The export panel (issue #127): the selected mesh out to STL/OBJ/PLY/glTF,
// the inspect section's contours out to DXF/SVG, the collision summary
// out to CSV (the actions in `ux/export-actions`), and a height map of meshes
// out to a 16-bit PNG (#267, `HeightMapSection`). Errors show in the panel;
// cancelling never applies (downloads have no picker to cancel).
import type { Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { AppState } from "../../../app/state/app-state";
import { exportFileName, type MeshExportFormat } from "../../../io/export";
import { MESH_FORMATS, useExportActions } from "../ux/export-actions";
import { HeightMapSection } from "./HeightMapSection";

export function ExportPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const { items, section, picked, format, setFormat, setMeshId, message, exportMesh, exportSection, exportReport } =
    useExportActions(app, scene);

  return (
    <section className="panel" aria-label="Export">
      <header className="panel-head">
        <h2>Export</h2>
      </header>
      <div className="row">
        <label className="picker">
          <span>Mesh</span>
          <select
            data-pick="export-mesh"
            value={picked === null ? "" : String(picked.id)}
            onChange={(e) => setMeshId(e.currentTarget.value || null)}
          >
            {items.map((i) => (
              <option key={i.id} value={String(i.id)}>
                {i.name}
              </option>
            ))}
          </select>
        </label>
        <label className="picker">
          <span>Format</span>
          <select data-format="export-mesh" value={format} onChange={(e) => setFormat(e.currentTarget.value as MeshExportFormat)}>
            {MESH_FORMATS.map((f) => (
              <option key={f.format} value={f.format}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          data-action="export-mesh"
          disabled={picked === null}
          onClick={exportMesh}
          title={`Download the mesh as ${exportFileName("mesh", format)}`}
        >
          Export mesh
        </button>
      </div>
      <div className="row">
        <button
          type="button"
          data-action="export-section-dxf"
          disabled={section === null}
          onClick={() => exportSection("dxf")}
          title="Download the section contours as DXF"
        >
          Section DXF
        </button>
        <button
          type="button"
          data-action="export-section-svg"
          disabled={section === null}
          onClick={() => exportSection("svg")}
          title="Download the section contours as SVG"
        >
          Section SVG
        </button>
      </div>
      <div className="row">
        <button type="button" data-action="export-report-csv" onClick={exportReport} title="Download the collision summary as CSV">
          Collision CSV
        </button>
      </div>
      <HeightMapSection app={app} scene={scene} picked={picked} />
      {message !== null ? (
        <p className="error" role="alert">
          {message}
        </p>
      ) : null}
    </section>
  );
}
