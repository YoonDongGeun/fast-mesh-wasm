// The info box's behaviour (#257): the selected mesh's info, kept current
// as it changes, the quality report export, and what the test hook reads.
// Edits, undo and redo of the mesh refresh it once they settle (a held
// sculpt stroke or gizmo drag is not settled, as for the inspect maps): a
// geometry change builds the topology again and measures, a placement change
// only reads the boxes again (and measures when the placement's scale
// changed). Nothing runs while the box is collapsed.
import type { MeshChange, MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useState } from "react";
import type { AppState } from "../../../app/state/app-state";
import type { Vec3 } from "../../../io/mesh-data";
import { downloadReportText, reportFileName } from "../../../io/report";
import { summarize } from "../../mesh-repair/ux/repair-summary";
import type { LengthUnit } from "../../../shared/ux/units";
import { view } from "../../../viewport/ux/view";
import {
  type Box,
  type InfoScene,
  type MeasureInfo,
  type MeshInfo,
  type MeshTopology,
  meshTopology,
  NO_MEASURE,
  readMeasureInfo,
  readMeshInfo,
  viewInfo,
} from "./object-info";
import { type QualityReportInput, qualityReportCsv, qualityReportHtml } from "./quality-report";
import { inspectMapProbe } from "./use-inspect-maps";

/** How long a burst of changes must pause before the info (and the overlays) follow. */
export const OBJECT_INFO_REFRESH_MS = 150;

/** What the test hook reads: the info box, the overlays and the angle (names, not handles). */
export const objectInfoProbe: {
  /** The mesh the info box shows, and its numbers. */
  mesh: string | null;
  info: MeshInfo | null;
  /** Info refreshes that landed. */
  refreshes: number;
  /** The bounding box overlay's box as drawn (in its frame), `null` when off. */
  box: { space: "world" | "local"; box: Box } | null;
  /** Normal lines drawn, and how many times they were built. */
  normalLines: number;
  normalBuilds: number;
  /** The angle measure's points and angle in degrees (`null` until three points). */
  angle: { points: Vec3[]; degrees: number | null };
  /** The last quality report exported: its format, file name and length. */
  lastReport: { format: "html" | "csv"; fileName: string; chars: number } | null;
} = {
  mesh: null,
  info: null,
  refreshes: 0,
  box: null,
  normalLines: 0,
  normalBuilds: 0,
  angle: { points: [], degrees: null },
  lastReport: null,
};

/** Whether a change record's buffer ranges say something changed. */
function touched(r: MeshChange["positions"]): boolean {
  return r === "all" || r.length > 0;
}

/** What a batch of changes did to `mesh`: removed it, changed its geometry, moved it, or nothing. */
export function meshChangeKind(changes: readonly MeshChange[], mesh: MeshHandle): "removed" | "geometry" | "placement" | null {
  let kind: "geometry" | "placement" | null = null;
  for (const c of changes) {
    if (c.mesh !== mesh) {
      continue;
    }
    if (c.removed) {
      return "removed";
    }
    if (c.capacityChanged || touched(c.positions) || touched(c.indices)) {
      kind = "geometry";
    } else if (c.placementChanged && kind === null) {
      kind = "placement";
    }
  }
  return kind;
}

/**
 * Calls `onSettled` once a burst of changes to `mesh` has paused for
 * `OBJECT_INFO_REFRESH_MS` and no stroke or drag holds it (`geometry` when
 * any change in it touched the buffers), and `onRemoved` at once when the mesh
 * goes. Returns the function that stops watching.
 */
export function watchMesh(
  scene: { onChanges(listener: (c: MeshChange[]) => void): () => void },
  mesh: MeshHandle,
  onSettled: (geometry: boolean) => void,
  onRemoved: () => void,
): () => void {
  let alive = true;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let geometry = false;
  const schedule = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (!alive) {
        return;
      }
      if (view.stroking || view.dragging) {
        schedule();
      } else if (view.strokeLanding !== null) {
        void view.strokeLanding.then(() => alive && schedule());
      } else {
        const g = geometry;
        geometry = false;
        onSettled(g);
      }
    }, OBJECT_INFO_REFRESH_MS);
  };
  const off = scene.onChanges((changes) => {
    const kind = meshChangeKind(changes, mesh);
    if (kind === "removed") {
      clearTimeout(timer);
      onRemoved();
    } else if (kind !== null) {
      geometry ||= kind === "geometry";
      schedule();
    }
  });
  return () => {
    alive = false;
    off();
    clearTimeout(timer);
  };
}

/** The scale of a column-major 4x4's linear part (its column lengths), to 9 digits: rotations keep it. */
function scaleKey(m: ArrayLike<number>): string {
  return [0, 4, 8].map((c) => Math.hypot(m[c], m[c + 1], m[c + 2]).toPrecision(9)).join(",");
}

/**
 * The info of `mesh` while `enabled`, kept current: `info` is `null` without
 * a mesh, while collapsed, and once the mesh is removed; `measuring` while
 * the SDK's `measure` runs (the other numbers show meanwhile).
 */
export function useObjectInfo(
  scene: Scene | WorkerScene,
  mesh: MeshHandle | null,
  enabled: boolean,
): { info: MeshInfo | null; measuring: boolean } {
  const store = useScene();
  const [state, setState] = useState<{ mesh: MeshHandle; info: MeshInfo; measuring: boolean } | null>(null);

  useEffect(() => {
    if (!enabled || mesh === null) {
      setState(null);
      return;
    }
    const s = scene as unknown as InfoScene;
    let alive = true;
    let seq = 0;
    let topology: MeshTopology | null = null;
    let measured: MeasureInfo = NO_MEASURE;
    let measuredKey: string | null = null;

    const refresh = (geometry: boolean) => {
      const n = ++seq;
      let base: ReturnType<typeof viewInfo>;
      let key: string;
      try {
        const views = s.views(mesh);
        const positions = views.positions();
        if (geometry || topology === null) {
          topology = meshTopology(views.indices(), positions.length / 3);
        }
        const m = s.matrix4(mesh);
        key = scaleKey(m);
        base = viewInfo(positions, topology, m, s.origin(mesh));
      } catch {
        return; // Removed meanwhile: the watcher clears it.
      }
      if (!geometry && key === measuredKey) {
        setState({ mesh, info: { ...base, ...measured }, measuring: false });
        objectInfoProbe.refreshes++;
        return;
      }
      setState({ mesh, info: { ...base, ...measured }, measuring: true });
      void readMeasureInfo(s, mesh).then((mi) => {
        if (!alive || n !== seq) {
          return;
        }
        measured = mi;
        measuredKey = key;
        setState({ mesh, info: { ...base, ...mi }, measuring: false });
        objectInfoProbe.refreshes++;
      });
    };

    refresh(true);
    const stop = watchMesh(
      store.scene,
      mesh,
      (geometry) => alive && refresh(geometry),
      () => {
        seq++;
        setState(null);
      },
    );
    return () => {
      alive = false;
      stop();
    };
  }, [store, scene, mesh, enabled]);

  const current = state !== null && state.mesh === mesh ? state : null;
  return { info: current?.info ?? null, measuring: current?.measuring ?? false };
}

/** The scene a report reads: the info box's, and the analysis the repair panel runs. */
type ReportScene = InfoScene & {
  readonly inspect?: InfoScene["inspect"] & { analyze(mesh: MeshHandle, options?: { maxListed?: number }): unknown };
};

/**
 * The quality report's inputs for `mesh`, read now: its info, the repair
 * panel's analysis when it is of this mesh (else a fresh one), and the
 * shown inspect map's readouts.
 */
export async function readQualityReport(app: AppState, scene: Scene | WorkerScene, mesh: MeshHandle): Promise<QualityReportInput> {
  const s = scene as unknown as ReportScene;
  const snap = app.getSnapshot();
  const name = snap.items.find((i) => i.mesh === mesh)?.name ?? "mesh";
  const info = await readMeshInfo(s, mesh);
  let repair: QualityReportInput["repair"] = null;
  const shared = app.getSnapshot().repairAnalysis;
  if (shared !== null && shared.mesh === mesh) {
    repair = { summary: summarize(shared.analysis), selfIntersects: shared.analysis.selfIntersects };
  } else if (s.inspect !== undefined) {
    try {
      const a = (await s.inspect.analyze(mesh, { maxListed: 64 })) as Parameters<typeof summarize>[0];
      repair = { summary: summarize(a), selfIntersects: a.selfIntersects };
    } catch {
      // Removed meanwhile: no analysis section.
    }
  }
  const map = inspectMapProbe.shown;
  return { name, info, repair, map: map === null ? null : { ...map }, generated: new Date() };
}

/** The report's text in `format`, and its download name. */
export function qualityReportFile(
  input: QualityReportInput,
  format: "html" | "csv",
  unit: LengthUnit,
): { text: string; fileName: string; mime: string } {
  const text = format === "html" ? qualityReportHtml(input, unit) : qualityReportCsv(input, unit);
  return {
    text,
    fileName: reportFileName(`${input.name}-quality`, format),
    mime: format === "html" ? "text/html" : "text/csv",
  };
}

/** Reads the report for `mesh` and downloads it; returns the error message, `null` on success. */
export async function exportQualityReport(
  app: AppState,
  scene: Scene | WorkerScene,
  mesh: MeshHandle,
  format: "html" | "csv",
): Promise<string | null> {
  try {
    const input = await readQualityReport(app, scene, mesh);
    const file = qualityReportFile(input, format, app.getSnapshot().units);
    downloadReportText(file.text, file.fileName, file.mime);
    objectInfoProbe.lastReport = { format, fileName: file.fileName, chars: file.text.length };
    return null;
  } catch (error) {
    return `Could not export the report: ${error instanceof Error ? error.message : String(error)}`;
  }
}
