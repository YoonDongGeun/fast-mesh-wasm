// Height map export (#267): the views it can look from, the checks of its
// settings, sampling meshes in bands of rows (so the engine answers other
// requests in between, and the export can be cancelled), and the two files it
// makes: a 16-bit PNG and a JSON with the image's scale and place.
import type { HeightMap, HeightMapFrame, HeightMapOptions, MeshHandle, Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { CameraPreset } from "../../../app/state/app-state";
import { heightMapFile } from "../../../io/height-image";
import { encodeHeightPng } from "../../../io/height-image-client";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { presetFrame } from "../../view/ux/camera";

type Vec3 = [number, number, number];

/** Where the map looks from: a standard view, or the viewport's camera. */
export type HeightMapView = CameraPreset | "camera";

/** The views, in the order the panel lists them; heights grow towards the viewer. */
export const HEIGHT_MAP_VIEWS: readonly (readonly [HeightMapView, string])[] = [
  ["top", "Top (+Y)"],
  ["bottom", "Bottom (−Y)"],
  ["front", "Front (+Z)"],
  ["back", "Back (−Z)"],
  ["left", "Left (−X)"],
  ["right", "Right (+X)"],
  ["camera", "Current view"],
];

/** The smallest and largest resolution offered, in pixels along the longer side. */
export const MIN_RESOLUTION = 2;
export const MAX_RESOLUTION = 8192;

/** The export's settings; lengths in millimetres. */
export interface HeightMapParams {
  view: HeightMapView;
  /** Pixels along the longer side. */
  resolution: number;
  /** The heights shown, as coordinates along the view; `null` for the meshes' own. */
  range: readonly [low: number, high: number] | null;
}

export const DEFAULT_HEIGHT_MAP: HeightMapParams = { view: "top", resolution: 1024, range: null };

/** What a camera gives: its world matrix, column-major. */
export interface CameraLike {
  matrixWorld: { elements: ArrayLike<number> };
}

/** The directions an image looks along and has up, or `null` for the current view without a camera. */
export function viewAxes(view: HeightMapView, camera: CameraLike | null): { direction: Vec3; up: Vec3 } | null {
  if (view === "camera") {
    if (camera === null) {
      return null;
    }
    const m = camera.matrixWorld.elements;
    return { direction: [-m[8], -m[9], -m[10]], up: [m[4], m[5], m[6]] };
  }
  const { dir, up } = presetFrame(view);
  return { direction: [0 - dir[0], 0 - dir[1], 0 - dir[2]], up };
}

/** Why the settings cannot be exported, in words, or `null` when they can. */
export function heightMapProblem(p: HeightMapParams): string | null {
  if (!Number.isInteger(p.resolution) || p.resolution < MIN_RESOLUTION || p.resolution > MAX_RESOLUTION) {
    return `The resolution must be a whole number from ${MIN_RESOLUTION} to ${MAX_RESOLUTION}.`;
  }
  if (p.range !== null && !(Number.isFinite(p.range[0]) && Number.isFinite(p.range[1]) && p.range[0] < p.range[1])) {
    return "The lowest height must be below the highest.";
  }
  return null;
}

/** The engine's options for the settings, seen along `axes`. */
export function frameOptions(p: HeightMapParams, axes: { direction: Vec3; up: Vec3 }): HeightMapOptions {
  return {
    direction: axes.direction,
    up: axes.up,
    resolution: p.resolution,
    ...(p.range === null ? {} : { range: [p.range[0], p.range[1]] }),
  };
}

/** A frame's heights as coordinates along its view: `[low, high]`. */
export function frameRange(f: HeightMapFrame): [number, number] {
  const low = f.origin[0] * f.normal[0] + f.origin[1] * f.normal[1] + f.origin[2] * f.normal[2];
  return [low, low + f.range];
}

/** A frame in words: its pixels, their size and the height range. */
export function describeFrame(f: HeightMapFrame, unit: LengthUnit): string {
  return `${f.width} × ${f.height} px · ${formatLength(f.pixelSize, unit)} per pixel · heights 0 to ${formatLength(f.range, unit)}`;
}

/** About how many pixels one request samples. */
export const BAND_PIXELS = 1 << 20;

/** Thrown when an export is cancelled. */
export class CancelledError extends Error {
  constructor() {
    super("cancelled");
  }
}

/** How an export reports and is stopped. */
export interface SampleControl {
  /** Called with the share of rows done. */
  onProgress?: (fraction: number) => void;
  signal?: AbortSignal;
}

/** Lets the page (and the Worker's other requests) in between two bands. */
function pause(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Samples `meshes` in bands of rows, about {@link BAND_PIXELS} each.
 *
 * @throws {CancelledError} once `signal` aborts; an error when one of the
 *   meshes changes meanwhile, which would leave a seam in the image.
 */
export async function sampleHeightMap(
  scene: Scene | WorkerScene,
  meshes: readonly MeshHandle[],
  options: HeightMapOptions,
  control: SampleControl = {},
): Promise<{ frame: HeightMapFrame; heights: Float32Array }> {
  let changed = false;
  const shown = new Set(meshes);
  const stop = scene.onChanges((changes) => {
    changed ||= changes.some(
      (c) => shown.has(c.mesh) && (c.removed || c.placementChanged || c.positions.length > 0 || c.indices.length > 0),
    );
  });
  // Changes reach listeners with a frame: ask for the ones already here.
  const changedSince = () => {
    scene.commitFrame();
    return changed;
  };
  const checkCancelled = () => {
    if (control.signal?.aborted === true) {
      throw new CancelledError();
    }
  };
  try {
    const frame = await scene.heightMapFrame(meshes, options);
    // What changed before the frame was fitted is in it.
    changedSince();
    changed = false;
    const { width, height } = frame;
    const heights = new Float32Array(width * height);
    const rows = Math.max(1, Math.floor(BAND_PIXELS / width));
    for (let first = 0; first < height; first += rows) {
      const end = Math.min(first + rows, height);
      const band: HeightMap = await scene.heightMap(meshes, frame, [first, end]);
      checkCancelled();
      if (changedSince()) {
        throw new Error("The meshes changed while the height map was sampled. Export it again.");
      }
      heights.set(band.heights, first * width);
      control.onProgress?.(end / height);
      await pause();
      checkCancelled();
    }
    return { frame, heights };
  } finally {
    stop();
  }
}

/** The two files' names for an export of `name`. */
export function heightMapFileNames(name: string): { png: string; json: string } {
  const stem = name.replace(/[\\/:*?"<>|]/g, "_").trim() || "mesh";
  return { png: `${stem}-height-map.png`, json: `${stem}-height-map.json` };
}

/** An exported height map: its two files, and the frame they show. */
export interface HeightMapExport {
  png: { name: string; bytes: Uint8Array };
  json: { name: string; text: string };
  frame: HeightMapFrame;
  /** Pixels that show a surface. */
  seen: number;
}

/** Samples meshes and makes the two files; see {@link sampleHeightMap} for its errors. */
export async function exportHeightMap(
  scene: Scene | WorkerScene,
  meshes: readonly { mesh: MeshHandle; name: string }[],
  options: HeightMapOptions,
  control: SampleControl = {},
): Promise<HeightMapExport> {
  const { frame, heights } = await sampleHeightMap(
    scene,
    meshes.map((m) => m.mesh),
    options,
    control,
  );
  let seen = 0;
  for (let i = 0; i < heights.length; i++) {
    seen += Number.isNaN(heights[i]) ? 0 : 1;
  }
  const names = heightMapFileNames(meshes.length === 1 ? meshes[0].name : "meshes");
  const bytes = await encodeHeightPng(heights, frame.width, frame.height, frame.range);
  const doc = heightMapFile(
    frame,
    names.png,
    meshes.map((m) => m.name),
  );
  return {
    png: { name: names.png, bytes },
    json: { name: names.json, text: `${JSON.stringify(doc, null, 2)}\n` },
    frame,
    seen,
  };
}
