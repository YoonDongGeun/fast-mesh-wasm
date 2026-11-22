// The export panel's actions (issue #127): the mesh picked (else the
// selected one, else the first) out to STL/OBJ/PLY/3MF/glTF/GLB, the inspect
// section's contours out to DXF/SVG and the collision summary out to CSV,
// each failure kept as the panel's message.
//
// Mesh encoding reuses `load/export`'s `exportMeshBytes`/`downloadMesh` (the
// #104 encoders); contours and the report reuse `load/report`'s pure builders
// (unit-tested, no `document`). The row and contour builders here are shared
// with the test hook's headless exports.
import type { MeshHandle, Scene, SectionPlane } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useState } from "react";
import { type AppState, type InspectSection, type Item, useApp } from "../../../app/state/app-state";
import { downloadMesh, type MeshExportFormat } from "../../../io/export";
import {
  type CollisionCsvRow,
  collisionRowsToCsv,
  type ContourLike,
  contoursToDxf,
  contoursToSvg,
  downloadReportText,
  reportFileName,
  toContourLikes,
} from "../../../io/report";
import { type GroupReadouts, inspect, type Readouts } from "../../../app/state/readouts";

export const MESH_FORMATS: { format: MeshExportFormat; label: string }[] = [
  { format: "stl", label: "STL" },
  { format: "obj", label: "OBJ" },
  { format: "ply", label: "PLY" },
  { format: "3mf", label: "3MF" },
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

/**
 * The panel's state and actions. `picked` is the mesh to export: the one
 * chosen (`setMeshId`), else the selected one, else the first listed.
 * `message` holds the last action's error, cleared by a success.
 */
export function useExportActions(
  app: AppState,
  scene: Scene | WorkerScene,
): {
  items: readonly Item[];
  section: InspectSection | null;
  picked: Item | null;
  format: MeshExportFormat;
  setFormat: (format: MeshExportFormat) => void;
  setMeshId: (id: string | null) => void;
  message: string | null;
  exportMesh: () => void;
  exportSection: (kind: "dxf" | "svg") => void;
  exportReport: () => void;
} {
  const store = useScene();
  const items = useApp(app, (s) => s.items);
  const section = useApp(app, (s) => s.inspectSection);
  const pair = useApp(app, (s) => s.collision);
  const units = useApp(app, (s) => s.units);
  const [selection] = useSelection();
  const [format, setFormat] = useState<MeshExportFormat>("stl");
  const [meshId, setMeshId] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const picked = items.find((i) => String(i.id) === meshId) ?? items.find((i) => i.mesh === selection) ?? items[0] ?? null;

  const exportMesh = (): void => {
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

  const exportSection = (kind: "dxf" | "svg"): void => {
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

  const exportReport = (): void => {
    try {
      const rows = buildCollisionRows(items, pair, inspect.collision, inspect.group);
      if (rows.length === 0) {
        setMessage("There is no collision summary yet: pick a pair or tick a group first.");
        return;
      }
      downloadReportText(collisionRowsToCsv(rows, units), reportFileName("collision-summary", "csv"), "text/csv");
      setMessage(null);
    } catch (error) {
      setMessage(`Could not export the report: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  return { items, section, picked, format, setFormat, setMeshId, message, exportMesh, exportSection, exportReport };
}
