// The geodesic map's state and handlers (#252): the map follows the picked
// sources and the method (each change runs the query again, the latest one
// only: a request that lands after a newer one was asked for is dropped),
// and its iso-lines follow the map and their spacing. The map draws through
// the surface field overlay like the inspect maps, so one colouring shows at
// a time: it shows while the overlay is the one it set, and anything else
// that takes the overlay hides it (the sources stay; Show map brings it
// back). Removing the sources, or the mesh changing, removes it.
import type { MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, type SurfaceFieldOverlayData, useApp } from "../../../app/state/app-state";
import {
  autoIsoSpacing,
  type GeodesicMapReport,
  type GeodesicMapRequest,
  geodesicOverlay,
  isoLevelSpacing,
  isoLineSegments,
  runGeodesicMap,
  sameGeodesicRequest,
} from "./geodesic-map";
import { SURFACE_LINE_OFFSET, type SurfacePathToolState } from "./surface-paths";

/** The overlay data the geodesic map set: another colouring releases it with {@link releaseGeodesicOverlay}. */
const geodesicOverlays = new WeakSet<SurfaceFieldOverlayData>();

/** Removes the geodesic map from the overlay, if it holds it: another map is about to take it. */
export function releaseGeodesicOverlay(app: AppState): void {
  const shown = app.getSnapshot().surfaceFieldValues;
  if (shown !== null && geodesicOverlays.has(shown)) {
    app.setSurfaceFieldValues(null);
  }
}

/** The map the state's picks ask for, or `null` without sources. */
export function geodesicRequestOf(state: SurfacePathToolState): GeodesicMapRequest | null {
  return state.mesh === null || state.sources.length === 0
    ? null
    : { mesh: state.mesh, sources: state.sources.map((s) => s.slot), method: state.method };
}

/** What the test hook reads: the shown map (its mesh by name) and how many queries landed. */
export const geodesicMapProbe: {
  shown: {
    mesh: string | null;
    sources: number[];
    method: GeodesicMapRequest["method"];
    max: number;
    mean: number;
    reached: number;
    unreached: number;
    /** The iso-line spacing drawn, or `null` with iso-lines off. */
    isoSpacing: number | null;
  } | null;
  runs: number;
} = { shown: null, runs: 0 };

/** The iso-line controls: on or off, and the spacing (`null`: a round tenth of the farthest distance). */
export interface IsoSettings {
  on: boolean;
  spacing: number | null;
}

export function useGeodesicMap(
  app: AppState,
  items: readonly Item[],
  /** Turns the surface field off before the map takes the overlay (read at call time). */
  onTakeOverlay: () => void,
) {
  const store = useScene();
  const tool = useApp(app, (s) => s.surfacePath);
  const overlay = useApp(app, (s) => s.surfaceFieldValues);
  const [iso, setIso] = useState<IsoSettings>({ on: false, spacing: null });
  const [report, setReport] = useState<GeodesicMapReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** The overlay data this map set last: the map shows while the overlay is still this object. */
  const shownRef = useRef<SurfaceFieldOverlayData | null>(null);
  const reportRef = useRef<GeodesicMapReport | null>(null);
  reportRef.current = report;
  /** The map asked for and not yet landed (the latest ask wins). */
  const wantRef = useRef<GeodesicMapRequest | null>(null);
  const runningRef = useRef(false);
  const takeRef = useRef(onTakeOverlay);
  takeRef.current = onTakeOverlay;
  const active = report !== null && overlay !== null && overlay === shownRef.current;

  // Something else took the overlay (a surface field, an inspect map, the
  // mesh's removal): the map is gone. Read live: a render can see a new report before the overlay.
  useEffect(() => {
    if (report !== null && app.getSnapshot().surfaceFieldValues !== shownRef.current) {
      shownRef.current = null;
      setReport(null);
    }
  }, [app, report, overlay]);

  /** Runs the wanted map until none is left, showing each that is still wanted when it lands. */
  const pump = async () => {
    if (runningRef.current) {
      return;
    }
    runningRef.current = true;
    setBusy(true);
    try {
      while (wantRef.current !== null) {
        const request = wantRef.current;
        const mine = shownRef.current !== null && app.getSnapshot().surfaceFieldValues === shownRef.current;
        if (!mine) {
          takeRef.current();
        }
        const before = app.getSnapshot().surfaceFieldValues;
        let next: GeodesicMapReport;
        try {
          next = await runGeodesicMap(store.scene, request);
        } catch (e) {
          if (wantRef.current === request) {
            wantRef.current = null;
            setError(e instanceof Error ? e.message : String(e));
          }
          continue;
        }
        if (wantRef.current !== request) {
          continue; // A newer ask, or none: run that one, or stop.
        }
        wantRef.current = null;
        const now = app.getSnapshot();
        if (
          now.surfaceFieldValues !== before ||
          now.surfaceField.mode !== "none" ||
          !now.items.some((i) => i.mesh === request.mesh)
        ) {
          continue; // Another colouring, or the mesh's removal, came first.
        }
        const data = geodesicOverlay(next);
        geodesicOverlays.add(data);
        shownRef.current = data;
        app.setSurfaceFieldValues(data);
        setReport(next);
        setError(null);
        geodesicMapProbe.runs++;
      }
    } finally {
      runningRef.current = false;
      setBusy(false);
    }
  };

  /** Removes the map (and any ask still running). */
  const hide = () => {
    wantRef.current = null;
    if (shownRef.current !== null && app.getSnapshot().surfaceFieldValues === shownRef.current) {
      app.setSurfaceFieldValues(null);
    }
    shownRef.current = null;
    setReport(null);
  };

  // The map follows the sources and the method.
  const pumpRef = useRef(pump);
  pumpRef.current = pump;
  const hideRef = useRef(hide);
  hideRef.current = hide;
  useEffect(() => {
    const request = geodesicRequestOf(app.getSnapshot().surfacePath);
    if (request === null) {
      if (reportRef.current !== null || wantRef.current !== null) {
        hideRef.current();
      }
      setError(null);
      return;
    }
    const shown = reportRef.current;
    if (
      shown !== null &&
      shownRef.current !== null &&
      app.getSnapshot().surfaceFieldValues === shownRef.current &&
      sameGeodesicRequest(shown.request, request)
    ) {
      return;
    }
    setError(null);
    wantRef.current = request;
    void pumpRef.current();
  }, [app, tool.mesh, tool.sources, tool.method]);

  /** Shows the map of the current sources again (after another colouring took the overlay). */
  const show = () => {
    const request = geodesicRequestOf(app.getSnapshot().surfacePath);
    if (request === null) {
      setError("Pick a source on the selected mesh first.");
      return;
    }
    setError(null);
    wantRef.current = request;
    void pump();
  };

  // The iso-lines follow the shown map and their spacing.
  const mappedMesh: MeshHandle | null = report?.request.mesh ?? null;
  const radius = items.find((i) => i.mesh === mappedMesh)?.radius ?? 1;
  const spacing = report === null ? null : isoLevelSpacing(iso.spacing ?? autoIsoSpacing(report.max), report.max);
  useEffect(() => {
    const clear = () => {
      if (app.getSnapshot().surfacePath.isoLines !== null) {
        app.setSurfacePath({ isoLines: null });
      }
    };
    if (!active || report === null || !iso.on || spacing === null) {
      clear();
      return;
    }
    try {
      const mesh = report.request.mesh;
      const views = store.scene.views(mesh);
      app.setSurfacePath({
        isoLines: isoLineSegments(
          views.positions(),
          views.indices(),
          store.scene.matrix4(mesh),
          report.values,
          spacing,
          radius * SURFACE_LINE_OFFSET,
        ),
      });
    } catch {
      clear(); // removed meanwhile
    }
  }, [app, store, active, report, iso.on, spacing, radius]);

  // The test hook's view, after every render.
  useEffect(() => {
    geodesicMapProbe.shown =
      !active || report === null
        ? null
        : {
            mesh: items.find((i) => i.mesh === report.request.mesh)?.name ?? null,
            sources: [...report.request.sources],
            method: report.request.method,
            max: report.max,
            mean: report.mean,
            reached: report.reached,
            unreached: report.unreached,
            isoSpacing: iso.on ? spacing : null,
          };
  });

  return {
    report: active ? report : null,
    busy,
    error,
    iso,
    /** The spacing iso-lines are drawn at (or would be), in mm. */
    spacing: spacing ?? iso.spacing ?? 1,
    setIso: (patch: Partial<IsoSettings>) => setIso((prev) => ({ ...prev, ...patch })),
    show,
  };
}
