// The export panel (issue #127): the selected mesh out to STL/OBJ/PLY/glTF,
// the inspect section's contours out to DXF/SVG, and the collision summary
// out to CSV.
//
// Mesh encoding reuses `load/export`'s `exportMeshBytes`/`downloadMesh` (the
// #104 encoders); contours and the report reuse `load/report`'s pure builders
// (unit-tested, no `document`). Errors show in the panel; cancelling never
// applies (downloads have no picker to cancel).
import type { MeshHandle, Scene, SectionPlane } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useState } from "react";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { downloadMesh, exportFileName, type MeshExportFormat } from "../load/export";
import {
  type CollisionCsvRow,
  collisionRowsToCsv,
  type ContourLike,
  contoursToDxf,
  contoursToSvg,
  downloadReportText,
  reportFileName,
  toContourLikes,
} from "../load/report";
import { type GroupReadouts, inspect, type Readouts } from "./inspect";

const MESH_FORMATS: { format: MeshExportFormat; label: string }[] = [
  { format: "stl", label: "STL" },
  { format: "obj", label: "OBJ" },
  { format: "ply", label: "PLY" },
  { format: "gltf", label: "glTF" },
  { format: "glb", label: "GLB" },
];

/**
 * The collision summary as named CSV rows: the pair mode's single row (under
 * the pair as picked now) and the group mode's per-pair rows. Empty until a
 * readout has landed. Exported for the test hook's headless `exportCsv`.
 */
export function buildCollisionRows(
  items: readonly Item[],
  pair: { a: MeshHandle | null; b: MeshHandle | null },
  collision: { readouts: Readouts | null },
  group: { readouts: GroupReadouts | null },
): CollisionCsvRow[] {
  const nameOf = (m: MeshHandle): string => items.find((i) => i.mesh === m)?.name ?? "?";
  const rows: CollisionCsvRow[] = [];
  const r = collision.readouts;
  if (r !== null && pair.a !== null && pair.b !== null) {
    rows.push({
      a: nameOf(pair.a),
      b: nameOf(pair.b),
      gap: r.gap,
      intersects: r.intersects,
      depth: r.penetration?.depth ?? null,
    });
  }
  const g = group.readouts;
  if (g !== null) {
    for (const p of g.pairs) {
      rows.push({ a: nameOf(p.a), b: nameOf(p.b), gap: p.gap, intersects: p.intersects, depth: p.depth });
    }
  }
  return rows;
}

/**
 * The inspect section's contours in one plane: every cut mesh's contours
 * concatenated, with the plane they were cut with. `null` without a section
 * (or when it cuts nothing). Exported for the test hook's headless section
 * export. Meshes the section rejects (removed meanwhile) are skipped.
 */
export function readSectionContours(
  store: SceneStore,
  scene: Scene | WorkerScene,
  handle: unknown,
  plane: SectionPlane,
): { origin: SectionPlane["origin"]; normal: SectionPlane["normal"]; contours: ContourLike[] } | null {
  const section = handle as import("fast-mesh-wasm").SectionHandle;
  const meshes = store.getSnapshot().sections.get(section)?.desired.meshes ?? [];
  const contours: ContourLike[] = [];
  for (const mesh of meshes) {
    try {
      contours.push(...toContourLikes(scene.section(section, mesh)));
    } catch {
      // Removed meanwhile (or never cut): that mesh contributes nothing.
    }
  }
  return { origin: plane.origin, normal: plane.normal, contours };
}

export function ExportPanel(props: { app: AppState; scene: Scene | WorkerScene }) {
  const { app, scene } = props;
  const store = useScene();
  const items = useApp(app, (s) => s.items);
  const section = useApp(app, (s) => s.inspectSection);
  const pair = useApp(app, (s) => s.collision);
  const [selection] = useSelection();
  const [format, setFormat] = useState<MeshExportFormat>("stl");
  const [meshId, setMeshId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const picked = items.find((i) => String(i.id) === meshId) ?? items.find((i) => i.mesh === selection) ?? items[0] ?? null;

  const onExportMesh = (): void => {
    if (picked === null) {
      setMessage("There is no mesh to export yet.");
      return;
    }
    try {
      downloadMesh(scene, picked.mesh, picked.name, format);
      setMessage(null);
    } catch (error) {
      setMessage(`Could not export the mesh: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const onExportSection = (kind: "dxf" | "svg"): void => {
    if (section === null) {
      setMessage("There is no section to export yet: arm the Section tool first.");
      return;
    }
    try {
      const read = readSectionContours(store, scene, section.handle, section.plane);
      if (read === null || read.contours.length === 0) {
        setMessage("The section has no contours to export yet.");
        return;
      }
      const text = kind === "dxf" ? contoursToDxf(read.contours, read.origin, read.normal) : contoursToSvg(read.contours, read.origin, read.normal);
      downloadReportText(text, reportFileName("section", kind), kind === "dxf" ? "application/dxf" : "image/svg+xml");
      setMessage(null);
    } catch (error) {
      setMessage(`Could not export the section: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const onExportReport = (): void => {
    try {
      const rows = buildCollisionRows(items, pair, inspect.collision, inspect.group);
      if (rows.length === 0) {
        setMessage("There is no collision summary yet: pick a pair or tick a group first.");
        return;
      }
      downloadReportText(collisionRowsToCsv(rows), reportFileName("collision-summary", "csv"), "text/csv");
      setMessage(null);
    } catch (error) {
      setMessage(`Could not export the report: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

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
          onClick={onExportMesh}
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
          onClick={() => onExportSection("dxf")}
          title="Download the section contours as DXF"
        >
          Section DXF
        </button>
        <button
          type="button"
          data-action="export-section-svg"
          disabled={section === null}
          onClick={() => onExportSection("svg")}
          title="Download the section contours as SVG"
        >
          Section SVG
        </button>
      </div>
      <div className="row">
        <button type="button" data-action="export-report-csv" onClick={onExportReport} title="Download the collision summary as CSV">
          Collision CSV
        </button>
      </div>
      {message !== null ? (
        <p className="error" role="alert">
          {message}
        </p>
      ) : null}
    </section>
  );
}
