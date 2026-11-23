// The inspect maps' state and handlers (#251): the controls' settings, the
// shown map's report, and the calls that make, restyle, refresh and clear
// it. A map draws through the surface field overlay (`surfaceFieldValues`),
// so one map at a time shows on one mesh: a map is shown while the overlay
// is the one it set, and anything else that takes the overlay (a surface
// field mode, the mesh's removal) clears it. Edits, undo and redo of the
// mapped mesh (or the deviation reference) run its query again once they
// settle; removing either clears it.
import type { MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, type SurfaceFieldOverlayData, useApp } from "../../../app/state/app-state";
import { view } from "../../../viewport/ux/view";
import {
  buildInspectMapRequest,
  DEFAULT_INSPECT_MAP_SETTINGS,
  type InspectMapReadout,
  type InspectMapReport,
  type InspectMapRequest,
  type InspectMapSettings,
  inspectMapChangeAction,
  type QualityMapMetric,
  inspectMapOverlay,
  QUALITY_METRICS,
  resolveBuildDirection,
  resolveReference,
  runInspectMap,
  styleInspectMap,
} from "./inspect-maps";

/** How long a burst of edits must pause before the map's query runs again. */
export const INSPECT_MAP_REFRESH_MS = 150;

/** What the test hook and the quality report (#257) read: the shown map (names, not handles) and how many queries landed. */
export const inspectMapProbe: {
  shown: {
    kind: InspectMapReport["kind"];
    mesh: string | null;
    reference: string | null;
    readout: InspectMapReadout;
    /** What the readouts were measured with: the request's parameters, without its handles (#257's report). */
    params:
      | { signed: boolean }
      | { criticalAngleDeg: number; direction: [number, number, number] }
      | { metric: QualityMapMetric; threshold: number };
    range: [number, number];
    band: [number, number] | null;
  } | null;
  /** Queries that landed on the overlay (a first show, a parameter change, a refresh). */
  runs: number;
} = { shown: null, runs: 0 };

/** The settings a change only restyles: the tolerance band needs no new query. */
const RESTYLE_ONLY: readonly (keyof InspectMapSettings)[] = ["toleranceOn", "tolerance"];

export function useInspectMaps(
  app: AppState,
  items: readonly Item[],
  selection: MeshHandle | null,
  /** Turns the surface field off before a map takes the overlay (stable or not: read at call time). */
  onTakeOverlay: () => void,
) {
  const store = useScene();
  const overlay = useApp(app, (s) => s.surfaceFieldValues);
  const [settings, setSettings] = useState<InspectMapSettings>(DEFAULT_INSPECT_MAP_SETTINGS);
  const [report, setReport] = useState<InspectMapReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** The overlay data this map set last: the map shows while the overlay is still this object. */
  const shownRef = useRef<SurfaceFieldOverlayData | null>(null);
  const seq = useRef(0);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const meshes = items.map((i) => i.mesh);
  const target = selection !== null && meshes.includes(selection) ? selection : null;
  const mappedMesh = report?.request.mesh ?? null;
  const reference = resolveReference(settings.reference, mappedMesh ?? target, meshes);
  const active = report !== null && overlay !== null && overlay === shownRef.current;

  // Something else took the overlay (a surface field, the mesh's removal): the
  // map is gone. Read live: a render can see a new report before the overlay.
  useEffect(() => {
    if (report !== null && app.getSnapshot().surfaceFieldValues !== shownRef.current) {
      seq.current++;
      shownRef.current = null;
      setReport(null);
      setBusy(false);
    }
  }, [app, report, overlay]);

  /** Runs `request` and shows its report, unless a newer run, a clear or another overlay came first. */
  const run = async (request: InspectMapRequest) => {
    const n = ++seq.current;
    const snap = app.getSnapshot();
    const mine = shownRef.current !== null && snap.surfaceFieldValues === shownRef.current;
    if (!mine) {
      onTakeOverlay();
    }
    const before = mine ? shownRef.current : null;
    setBusy(true);
    setError(null);
    try {
      const next = await runInspectMap(store.scene, request);
      if (n !== seq.current) {
        return;
      }
      const now = app.getSnapshot();
      if (
        now.surfaceFieldValues !== before ||
        now.surfaceField.mode !== "none" ||
        !now.items.some((i) => i.mesh === request.mesh)
      ) {
        return; // Another overlay or the mesh's removal came first.
      }
      const data = inspectMapOverlay(next, styleInspectMap(next, settingsRef.current));
      shownRef.current = data;
      app.setSurfaceFieldValues(data);
      setReport(next);
      inspectMapProbe.runs++;
    } catch (e) {
      if (n === seq.current) {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (n === seq.current) {
        setBusy(false);
      }
    }
  };

  /** The request for `next` on `mesh`, with the build direction read now; sets the error when there is none. */
  const requestFor = (next: InspectMapSettings, mesh: MeshHandle): InspectMapRequest | null => {
    const ref = resolveReference(next.reference, mesh, meshes);
    const direction = resolveBuildDirection(next.buildFrom, view.camera?.matrixWorld.elements ?? null);
    const request = buildInspectMapRequest(next, mesh, ref, direction);
    if (request === null) {
      setError("Load a second mesh to measure the deviation against.");
    }
    return request;
  };

  /** Shows the current map on the selected mesh (or moves it there). */
  const show = () => {
    if (target === null) {
      setError("Select a mesh to map.");
      return;
    }
    const request = requestFor(settings, target);
    if (request !== null) {
      void run(request);
    }
  };

  /** Removes the map from its mesh. */
  const clear = () => {
    seq.current++;
    if (shownRef.current !== null && app.getSnapshot().surfaceFieldValues === shownRef.current) {
      app.setSurfaceFieldValues(null);
    }
    shownRef.current = null;
    setReport(null);
    setBusy(false);
    setError(null);
  };

  /**
   * Changes the settings. A shown map follows on its mesh: the tolerance
   * restyles it, anything else runs its query again.
   */
  const update = (patch: Partial<InspectMapSettings>) => {
    const next = { ...settingsRef.current, ...patch };
    if (patch.metric !== undefined && patch.threshold === undefined && patch.metric !== settingsRef.current.metric) {
      next.threshold = QUALITY_METRICS.find((m) => m.metric === patch.metric)?.threshold ?? next.threshold;
    }
    settingsRef.current = next;
    setSettings(next);
    if (!active || report === null) {
      return;
    }
    const keys = Object.keys(patch) as (keyof InspectMapSettings)[];
    if (keys.every((k) => RESTYLE_ONLY.includes(k))) {
      const data = inspectMapOverlay(report, styleInspectMap(report, next));
      shownRef.current = data;
      app.setSurfaceFieldValues(data);
      return;
    }
    const request = requestFor(next, report.request.mesh);
    if (request !== null) {
      void run(request);
    }
  };

  // While a map shows, edits of its mesh (or the reference) run it again once
  // they settle, with the same inputs; removing either clears it. A held
  // sculpt stroke or gizmo drag is not settled: a large mesh's query would
  // queue ahead of its next dabs, so the refresh waits for the release (and
  // a released stroke's landing).
  const runRef = useRef(run);
  runRef.current = run;
  const clearRef = useRef(clear);
  clearRef.current = clear;
  useEffect(() => {
    if (!active || report === null) {
      return;
    }
    const request = report.request;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
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
          void runRef.current(request);
        }
      }, INSPECT_MAP_REFRESH_MS);
    };
    const off = store.scene.onChanges((changes) => {
      const action = inspectMapChangeAction(changes, request);
      if (action === "clear") {
        clearTimeout(timer);
        clearRef.current();
      } else if (action === "refresh") {
        schedule();
      }
    });
    return () => {
      alive = false;
      off();
      clearTimeout(timer);
    };
  }, [store, active, report]);

  // The test hook's view, after every render.
  useEffect(() => {
    const nameOf = (m: MeshHandle) => items.find((i) => i.mesh === m)?.name ?? null;
    const data = shownRef.current;
    const r = report?.request ?? null;
    inspectMapProbe.shown =
      !active || report === null || r === null || data === null
        ? null
        : {
            kind: report.kind,
            mesh: nameOf(report.request.mesh),
            reference: report.kind === "deviation" ? nameOf(report.request.reference) : null,
            readout: { ...report.readout },
            params:
              r.kind === "deviation"
                ? { signed: r.signed }
                : r.kind === "overhang"
                  ? { criticalAngleDeg: r.criticalAngleDeg, direction: [r.direction[0], r.direction[1], r.direction[2]] }
                  : { metric: r.metric, threshold: r.threshold },
            range: [data.min, data.max],
            band: data.band == null ? null : [data.band.lo, data.band.hi],
          };
  });

  return {
    settings,
    report: active ? report : null,
    style: active && report !== null ? styleInspectMap(report, settings) : null,
    busy,
    error,
    target,
    reference,
    show,
    clear,
    update,
  };
}
