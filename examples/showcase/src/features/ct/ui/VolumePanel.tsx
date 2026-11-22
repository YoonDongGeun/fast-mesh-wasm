// The volume view: our own direct volume rendering (a three.js
// raycaster) of the selected mesh's source DICOM volume, with HU-tuned
// presets, a MIP toggle, and the slice planes overlaid in 3D — plus
// linked axial/coronal/sagittal slice views on 2D canvas with window
// presets, crosshairs, and click/drag navigation, like a PACS viewer.
// The loader retains each series' volume; this panel fetches it on
// selection.
import { useSelection } from "fast-mesh-wasm/react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Loader } from "../../../io/loader";
import type { VolumePayload } from "../../../io/protocol";
import { type AppState, useApp } from "../../../app/state/app-state";
import { inspect } from "../../../app/state/readouts";
import { normalizeVolume, presetLut } from "../ux/volume-lut";
import { VOLUME_PRESETS, WINDOW_PRESETS } from "../ux/volume-presets";
import { crosshairLines, extractSlice, SLICE_AXES, slicePlaneImages, sliceCount, windowToGray, type SliceAxis } from "../ux/volume-slice";
import { useVolumeMpr } from "../ux/volume-mpr";
import { sliceNavigation, usePanelVolume, useRevealTabs, useSliceViewsProbe, useVolumeSources } from "../ux/use-volume-panel";
import { VolumeView3D } from "./volume-three";

export function VolumePanel(props: { app: AppState; loader: Loader }) {
  const { app, loader } = props;
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const item = items.find((i) => i.mesh === selection) ?? null;
  if (item === null || item.volumeId === null) {
    return null;
  }
  // A sample's suggested preset (by name), else Bone.
  const suggested = item.preset === null ? -1 : VOLUME_PRESETS.findIndex((p) => p.name === item.preset);
  const initialPreset = suggested < 0 ? 0 : suggested;
  return (
    <VolumeWork
      key={item.volumeId}
      app={app}
      loader={loader}
      volumeId={item.volumeId}
      name={item.name}
      initialPreset={initialPreset}
    />
  );
}

function VolumeWork(props: { app: AppState; loader: Loader; volumeId: string; name: string; initialPreset: number }) {
  const { app, loader, volumeId, name, initialPreset } = props;
  const { payload, failed } = usePanelVolume(loader, volumeId);

  if (failed) {
    return (
      <section className="panel" aria-label="Volume">
        <header className="panel-head">
          <h2>Volume</h2>
          <span className="muted small">{name}</span>
        </header>
        <p className="muted">The volume is no longer retained.</p>
      </section>
    );
  }
  if (payload === null) {
    return (
      <section className="panel" aria-label="Volume">
        <header className="panel-head">
          <h2>Volume</h2>
          <span className="muted small">{name}</span>
        </header>
        <p className="muted">Loading the volume…</p>
      </section>
    );
  }
  return <VolumeTabs app={app} payload={payload} volumeId={volumeId} name={name} initialPreset={initialPreset} />;
}

/** The tabs share the MPR state through the app, so the scene overlay stays linked too. */
function VolumeTabs(props: {
  app: AppState;
  payload: VolumePayload;
  volumeId: string;
  name: string;
  initialPreset: number;
}) {
  const { app, payload, volumeId, name, initialPreset } = props;
  const tabsRef = useRef<HTMLDivElement>(null);
  useRevealTabs(tabsRef, volumeId);
  const {
    live,
    tab,
    display,
    range,
    windowName,
    colorWindow,
    colorLevel,
    setTab,
    setPreset,
    setWindow,
    setInScene,
    setSlices,
  } = useVolumeMpr(app, payload, volumeId, initialPreset);
  const { sources, chooseSource } = useVolumeSources(app);

  return (
    <section className="panel" aria-label="Volume">
      <header className="panel-head">
        <h2>Volume</h2>
        <span className="muted small">{name}</span>
      </header>
      <div ref={tabsRef} className="row" role="tablist" aria-label="Volume views">
        <button type="button" role="tab" aria-selected={tab === "volume"} data-action="volume-tab-3d" onClick={() => setTab("volume")}>
          3D
        </button>
        <button type="button" role="tab" aria-selected={tab === "slices"} data-action="volume-tab-slices" onClick={() => setTab("slices")}>
          Slices
        </button>
        <label className="slider" title="Which loaded volume to view">
          source{" "}
          <select
            data-action="volume-source"
            value={volumeId}
            onChange={(e) => chooseSource(e.currentTarget.value)}
          >
            {sources.map((i) => (
              <option key={i.volumeId} value={i.volumeId ?? ""}>
                {i.name}
              </option>
            ))}
          </select>
        </label>
        <label className="check" title="Show the slices in the main scene too">
          <input
            type="checkbox"
            data-action="volume-scene"
            checked={live?.inScene ?? true}
            onChange={(e) => setInScene(e.currentTarget.checked)}
          />
          Scene
        </label>
      </div>
      {live === null ? null : tab === "volume" ? (
        <VolumeCanvas
          payload={payload}
          volumeId={volumeId}
          display={display}
          range={range}
          slices={live.slices}
          colorWindow={colorWindow}
          colorLevel={colorLevel}
          preset={live.preset}
          setPreset={setPreset}
        />
      ) : (
        <SliceWork
          payload={payload}
          volumeId={volumeId}
          display={display}
          slices={live.slices}
          setSlices={setSlices}
          windowIdx={live.window}
          setWindowIdx={setWindow}
          windowName={windowName}
          colorWindow={colorWindow}
          colorLevel={colorLevel}
        />
      )}
    </section>
  );
}

function VolumeCanvas(props: {
  payload: VolumePayload;
  volumeId: string;
  display: Float32Array;
  range: [number, number];
  slices: Record<SliceAxis, number>;
  colorWindow: number;
  colorLevel: number;
  preset: number;
  setPreset: (i: number) => void;
}) {
  const { payload, volumeId, display, range, slices, colorWindow, colorLevel, preset, setPreset } = props;
  const divRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewRef = useRef<VolumeView3D | null>(null);
  const [mip, setMip] = useState(false);
  const [planesOn, setPlanesOn] = useState(true);
  const [failed, setFailed] = useState(false);
  const prepared = useMemo(() => normalizeVolume(display, range[0], range[1]), [display, range]);

  useEffect(() => {
    const div = divRef.current;
    const canvas = canvasRef.current;
    if (div === null || canvas === null) {
      return;
    }
    let view: VolumeView3D | null = null;
    try {
      view = new VolumeView3D(
        canvas,
        prepared,
        { columns: payload.columns, rows: payload.rows, slices: payload.slices, spacing: payload.spacing },
        presetLut(VOLUME_PRESETS[preset], range[0], range[1]),
      );
    } catch {
      setFailed(true);
      return;
    }
    view.setMip(mip);
    view.setSize(div.clientWidth, div.clientHeight);
    const resize = new ResizeObserver(() => view?.setSize(div.clientWidth, div.clientHeight));
    resize.observe(div);
    viewRef.current = view;
    inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: 0, ready: true };
    return () => {
      resize.disconnect();
      viewRef.current = null;
      view?.dispose();
      if (inspect.volume.id === volumeId) {
        inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: 0, ready: false };
      }
    };
    // The view belongs to the payload; preset/mip/planes steer it below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload, volumeId]);

  useEffect(() => {
    const view = viewRef.current;
    if (view !== null) {
      view.setLut(presetLut(VOLUME_PRESETS[preset], range[0], range[1]));
      view.setMip(mip);
      inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: planesOn ? 3 : 0, ready: true };
    }
  }, [preset, mip, planesOn, prepared, range, volumeId]);

  useEffect(() => {
    const view = viewRef.current;
    if (view === null) {
      return;
    }
    if (!planesOn) {
      view.setPlanes(null);
      inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: 0, ready: true };
      return;
    }
    view.setPlanes(
      slicePlaneImages(
        display,
        { columns: payload.columns, rows: payload.rows, slices: payload.slices },
        slices,
        colorWindow,
        colorLevel,
      ),
    );
    inspect.volume = { id: volumeId, preset: VOLUME_PRESETS[preset].name, mip, planes: 3, ready: true };
  }, [slices, colorWindow, colorLevel, planesOn, display, payload, preset, mip, volumeId]);

  return (
    <>
      <div className="row">
        <label className="slider">
          preset{" "}
          <select data-action="volume-preset" value={preset} onChange={(e) => setPreset(Number(e.currentTarget.value))}>
            {VOLUME_PRESETS.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="check" title="Maximum intensity projection (the classic angio look)">
          <input
            type="checkbox"
            data-action="volume-mip"
            checked={mip}
            onChange={(e) => setMip(e.currentTarget.checked)}
          />
          MIP
        </label>
        <label className="check" title="Show the slice positions as planes in 3D">
          <input
            type="checkbox"
            data-action="volume-planes"
            checked={planesOn}
            onChange={(e) => setPlanesOn(e.currentTarget.checked)}
          />
          Planes
        </label>
      </div>
      {failed ? <p className="muted">WebGL is unavailable: the volume cannot render here.</p> : null}
      <div ref={divRef} className="volume-canvas" data-volume={volumeId}>
        <canvas ref={canvasRef} />
      </div>
    </>
  );
}

function SliceWork(props: {
  payload: VolumePayload;
  volumeId: string;
  display: Float32Array;
  slices: Record<SliceAxis, number>;
  setSlices: React.Dispatch<React.SetStateAction<Record<SliceAxis, number>>>;
  windowIdx: number;
  setWindowIdx: (i: number) => void;
  windowName: string;
  colorWindow: number;
  colorLevel: number;
}) {
  const { payload, volumeId, display, slices, setSlices, windowIdx, setWindowIdx, windowName, colorWindow, colorLevel } = props;
  const markReady = useSliceViewsProbe(volumeId, windowName, slices);

  return (
    <>
      <div className="row">
        <label className="slider">
          window{" "}
          <select data-action="volume-window" value={windowIdx} onChange={(e) => setWindowIdx(Number(e.currentTarget.value))}>
            {WINDOW_PRESETS.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      {SLICE_AXES.map((axis) => (
        <div key={axis} className="slice-row">
          <div className="row">
            <span className="slice-label">{axis}</span>
            <label className="slider">
              slice{" "}
              <input
                type="range"
                data-action={`slice-${axis}`}
                min={0}
                max={sliceCount(axis, payload) - 1}
                value={slices[axis]}
                onChange={(e) => {
                  // Read the event now: the updater runs after dispatch,
                  // when currentTarget is null.
                  const v = Number(e.currentTarget.value);
                  setSlices((s) => ({ ...s, [axis]: v }));
                }}
              />
              <span className="muted small">
                {slices[axis]}/{sliceCount(axis, payload) - 1}
              </span>
            </label>
          </div>
          <SliceView
            display={display}
            payload={payload}
            axis={axis}
            slices={slices}
            colorWindow={colorWindow}
            colorLevel={colorLevel}
            onNavigate={(part) => setSlices((s) => ({ ...s, ...part }))}
            onReady={() => markReady(axis)}
          />
        </div>
      ))}
    </>
  );
}

function SliceView(props: {
  display: Float32Array;
  payload: VolumePayload;
  axis: SliceAxis;
  slices: Record<SliceAxis, number>;
  colorWindow: number;
  colorLevel: number;
  onNavigate: (part: Partial<Record<SliceAxis, number>>) => void;
  onReady: () => void;
}) {
  const { display, payload, axis, slices, colorWindow, colorLevel, onNavigate, onReady } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const image = extractSlice(display, payload.columns, payload.rows, payload.slices, axis, slices[axis]);
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    if (ctx === null) {
      return;
    }
    ctx.putImageData(new ImageData(windowToGray(image.data, colorWindow, colorLevel), image.width, image.height), 0, 0);
    // The crosshair: the other two slice positions, colored by axis.
    const lines = crosshairLines(axis, slices, payload);
    for (const line of lines) {
      ctx.fillStyle = line.color;
      if (line.vertical) {
        ctx.fillRect(line.at, 0, 1, image.height);
      } else {
        ctx.fillRect(0, line.at, image.width, 1);
      }
    }
    onReady();
  }, [display, payload, axis, slices, colorWindow, colorLevel, onReady]);

  const { onPointerDown, onPointerMove } = sliceNavigation(canvasRef, axis, payload, onNavigate);

  return (
    <div className="slice-canvas" data-slice={axis}>
      <canvas ref={canvasRef} onPointerDown={onPointerDown} onPointerMove={onPointerMove} />
    </div>
  );
}
