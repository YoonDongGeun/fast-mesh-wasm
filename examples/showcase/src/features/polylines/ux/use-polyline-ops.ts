// The polyline panel's behaviour (#268): the fields of each tool with
// defaults that follow the size of what is selected, and running a tool on
// the selected polylines (or the only one) with a one-line report.
import { useScene } from "fast-mesh-wasm/react";
import { useMemo, useState } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { downloadReportText, reportFileName } from "../../../io/report";
import { formatLength } from "../../../shared/ux/units";
import { niceSize } from "../../home/ux/create-primitive";
import type { ExtrudeSide } from "./extrude-curves";
import type { OffsetCorners } from "./offset";
import { planarFit } from "./planar";
import { curveLength, curveReadout, curvesExtent, pointCount, type PolylineDisplay } from "./polyline";
import {
  activePolylines,
  type EditCounts,
  extrudePolylines,
  offsetPolylines,
  removePolylines,
  resamplePolylines,
  simplifyPolylines,
  tubePolylines,
} from "./polyline-actions";
import { type ExportPlane, polylinesFile } from "./polyline-export";

/** The lengths the panel takes, all in millimetres. */
export interface PolylineLengths {
  tolerance: number;
  spacing: number;
  smoothing: number;
  offset: number;
  radius: number;
  depth: number;
}

/** The defaults for curves of size `extent` (the diagonal of their box) and length `length` each, in round numbers. */
export function defaultLengths(extent: number, length: number): PolylineLengths {
  return {
    tolerance: niceSize(extent / 1000),
    spacing: niceSize(length / 100),
    smoothing: 0,
    offset: niceSize(extent / 50),
    radius: niceSize(extent / 100),
    depth: niceSize(extent / 10),
  };
}

/** A tool's one-line report. */
export interface PolylineReport {
  text: string;
  error: boolean;
}

/** An edit's report: its points before and after. */
export function describeEdit(label: string, c: EditCounts): string {
  return c.unchanged
    ? `${label}: nothing to change.`
    : `${label}: ${c.before.toLocaleString("en")} points → ${c.after.toLocaleString("en")}.`;
}

const count = (n: number, one: string, more = `${one}s`) => `${n.toLocaleString("en")} ${n === 1 ? one : more}`;

export function usePolylineOps(app: AppState) {
  const store = useScene();
  const polylines = useApp(app, (s) => s.polylines);
  const selection = useApp(app, (s) => s.polylineSelection);
  const unit = useApp(app, (s) => s.units);
  const active = useMemo(() => activePolylines({ polylines, polylineSelection: selection }), [polylines, selection]);

  /** The lengths typed; the rest follow the selection's size. */
  const [typed, setTyped] = useState<Partial<PolylineLengths>>({});
  const [corners, setCorners] = useState<OffsetCorners>("round");
  const [sides, setSides] = useState(16);
  const [side, setSide] = useState<ExtrudeSide>("one");
  const [plane, setPlane] = useState<ExportPlane>("fit");
  const [report, setReport] = useState<PolylineReport | null>(null);

  const totals = useMemo(() => {
    const length = active.reduce((sum, p) => sum + curveLength(p), 0);
    return {
      length,
      points: active.reduce((sum, p) => sum + pointCount(p), 0),
      defaults: defaultLengths(curvesExtent(active), active.length === 0 ? 0 : length / active.length),
      fit: active.length === 1 ? planarFit(active) : null,
    };
  }, [active]);
  const lengths: PolylineLengths = { ...totals.defaults, ...typed };

  /** What is worked on, in a line or two. */
  const readout: string[] =
    active.length === 0
      ? []
      : active.length === 1
        ? [
            curveReadout(active[0], unit),
            totals.fit === null || "plane" in totals.fit
              ? "planar"
              : totals.fit.problem === "bent"
                ? `not planar: strays ${formatLength(totals.fit.deviation, unit)} from its plane`
                : "a straight line",
          ]
        : [
            `${active.length} polylines · ${formatLength(totals.length, unit)} in all · ${totals.points.toLocaleString("en")} points`,
            `${active.filter((p) => p.closed).length} closed`,
          ];

  /** Runs a tool on the active polylines; its report, or why it did nothing, replaces the last one. */
  const run = (tool: () => string) => {
    if (active.length === 0) {
      return;
    }
    try {
      setReport({ text: tool(), error: false });
    } catch (e) {
      setReport({ text: e instanceof Error ? e.message : String(e), error: true });
    }
  };

  /** A mesh the engine refuses after all is reported when it does. */
  const landing = (done: Promise<void>) => {
    done.catch((e: unknown) => setReport({ text: e instanceof Error ? e.message : String(e), error: true }));
  };

  const simplify = () =>
    run(() => describeEdit("Simplify", simplifyPolylines(app, store, active, lengths.tolerance)));

  const resample = () =>
    run(() =>
      describeEdit("Resample", resamplePolylines(app, store, active, { spacing: lengths.spacing, smoothing: lengths.smoothing })),
    );

  const offset = () =>
    run(() => {
      const made = offsetPolylines(app, store, active, lengths.offset, corners, unit);
      return `Offset by ${formatLength(lengths.offset, unit)}: ${count(made.length, "polyline")}.`;
    });

  const tube = () =>
    run(() => {
      const r = tubePolylines(app, store, active, lengths.radius, sides);
      landing(r.done);
      const folds =
        r.folds === 0 ? "" : ` It overlaps itself where the curve bends tighter than the radius (${count(r.folds, "place")}).`;
      return `${count(r.items.length, "tube")}, ${count(r.triangles, "triangle")}.${folds}`;
    });

  const extrude = () =>
    run(() => {
      const r = extrudePolylines(app, store, active, lengths.depth, side, unit);
      landing(r.done);
      return `${count(r.items.length, "solid")}, ${count(r.triangles, "triangle")}.`;
    });

  const remove = () =>
    run(() => {
      const n = active.length;
      removePolylines(
        app,
        store,
        active.map((p) => p.id),
      );
      return `Removed ${count(n, "polyline")}.`;
    });

  /** Downloads the active polylines, or with none active every visible one. */
  const exportAs = (format: "svg" | "dxf") => {
    const curves = active.length > 0 ? active : polylines.filter((p) => p.visible);
    if (curves.length === 0) {
      return;
    }
    const file = polylinesFile(curves, format, plane);
    const name = reportFileName(curves.length === 1 ? curves[0].name : "polylines", format);
    if (typeof URL.createObjectURL === "function") {
      downloadReportText(file.text, name, format === "svg" ? "image/svg+xml" : "application/dxf");
    }
    const flat = file.flattened === 0 ? "" : ` Flattened: they span ${formatLength(file.flattened, unit)} across the plane.`;
    setReport({ text: `Exported ${count(curves.length, "polyline")} to ${name} (mm).${flat}`, error: false });
  };

  return {
    polylines,
    active,
    unit,
    lengths,
    setLength: (key: keyof PolylineLengths, mm: number) => setTyped((prev) => ({ ...prev, [key]: mm })),
    corners,
    setCorners,
    sides,
    setSides,
    side,
    setSide,
    plane,
    setPlane,
    readout,
    report,
    selectAll: () => app.selectPolylines(polylines.map((p) => p.id)),
    setDisplay: (patch: Partial<PolylineDisplay>) =>
      app.setPolylineDisplay(
        active.map((p) => p.id),
        patch,
      ),
    simplify,
    resample,
    offset,
    tube,
    extrude,
    remove,
    exportAs,
  };
}
