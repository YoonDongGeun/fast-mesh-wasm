// Height maps and reliefs: the types `Scene.heightMap`, `Scene.addHeightfield`
// (and their `WorkerScene` twins) and `generateHeightfield` take and return,
// and the argument checks, encoding and decoding they share. Wasm-free, so
// that `WorkerScene` can check its arguments on the main thread.
import type { Mat3, Placement, PlacementInput, Vec3 } from "./api.js";
import { FastMeshError } from "./core.js";
import { rotationMatrix } from "./handles.js";
import type { MeshHandle } from "./scene.js";

/** The most pixels of a height map: 2^28. */
export const MAX_HEIGHT_MAP_PIXELS = 2 ** 28;

/** How `heightMapFrame` (and `heightMap` without a frame) fits an image to the meshes. */
export interface HeightMapOptions {
  /**
   * The world direction the image looks along, of any length; heights are
   * measured against it. Default `[0, 0, -1]`, down the Z axis.
   */
  direction?: Readonly<Vec3>;
  /**
   * The world direction that is up in the image, made perpendicular to
   * `direction`. By default `+Y`, or looking along the Y axis, `-Z` from above
   * and `+Z` from below (`+X` is right in both).
   */
  up?: Readonly<Vec3>;
  /**
   * Pixels along the longer side of the image, a whole number of at least 2
   * (default `1024`). Pixels are square, so the other side follows from the
   * meshes' proportions; it has at least 2 pixels.
   */
  resolution?: number;
  /** The world size of a pixel, instead of `resolution`. */
  pixelSize?: number;
  /** World distance to leave free around the meshes (default `0`). */
  margin?: number;
  /**
   * The heights the image covers, `[low, high]`, as world coordinates along
   * `-direction`: `low` is height 0, and only surfaces between the two are
   * seen. By default the meshes' own range.
   */
  range?: readonly [number, number];
}

/**
 * Where a height map's pixels are in the world: pixel `(col, row)`, rows from
 * the top, at height `h` is the world point
 * `origin + col * pixelSize * right - row * pixelSize * up + h * normal`.
 */
export interface HeightMapFrame {
  /** Pixels along a row. */
  width: number;
  /** Rows. */
  height: number;
  /** The centre of the top left pixel, at height 0. */
  origin: Vec3;
  /** The image's right, a unit vector. */
  right: Vec3;
  /** The image's up, a unit vector perpendicular to `right`. */
  up: Vec3;
  /** The direction heights grow in, towards the viewer: `right × up`. */
  normal: Vec3;
  /** The world distance between the centres of neighbouring pixels. */
  pixelSize: number;
  /** The largest height: the image covers heights from 0 to this. */
  range: number;
}

/** The heights of a height map, or of a band of its rows. */
export interface HeightMap {
  /** Where the pixels are. */
  frame: HeightMapFrame;
  /** The rows sampled: from the first to before the second. */
  rows: [number, number];
  /**
   * The heights of the rows, `frame.width` to a row from the left, in
   * `[0, frame.range]`; NaN where no surface is seen.
   */
  heights: Float32Array;
}

/** A grid of heights, for `generateHeightfield` and `addHeightfield`. */
export interface Heightfield {
  /** `width * height` heights, row by row from the top left; NaN stands at 0. Each must be above `-base`. */
  heights: Float32Array | ArrayLike<number>;
  /** Samples along a row, at least 2. */
  width: number;
  /** Rows, at least 2. */
  height: number;
  /** The distance between samples: one number, or `[along a row, between rows]`. Default `1`. */
  pixelSize?: number | readonly [number, number];
  /** The thickness of the base below height 0. Default `1`. */
  base?: number;
}

/** Options for `addHeightfield`. */
export interface AddHeightfieldOptions {
  /** Where the relief goes (default: the identity). See {@link heightMapPlacement}. */
  placement?: PlacementInput;
  /** Turn on the new mesh's smooth vertex normals, as `addMesh` does (default `true`). */
  normals?: boolean;
}

/** What `addHeightfield` added. */
export interface HeightfieldResult {
  /** The new mesh holding the relief: closed, outward-wound, one undo step. */
  mesh: MeshHandle;
  /** Vertices of the relief. */
  vertices: number;
  /** Triangles of the relief. */
  triangles: number;
}

/** Internal: {@link Heightfield} checked, with its defaults; `heights` may be the caller's array. */
export interface CheckedHeightfield {
  heights: Float32Array;
  width: number;
  height: number;
  pixelSize: [number, number];
  base: number;
}

/** Internal: {@link AddHeightfieldOptions} checked, with their defaults; a plain object, safe to post to a Worker. */
export interface CheckedHeightfieldPlacement {
  rotation: number[];
  translation: Vec3;
  scale?: Vec3;
  normals: boolean;
}

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/** The cross product, without -0. */
function cross(a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 {
  return [a[1] * b[2] - a[2] * b[1] + 0, a[2] * b[0] - a[0] * b[2] + 0, a[0] * b[1] - a[1] * b[0] + 0];
}

/**
 * The placement that puts a relief of a map's heights (`addHeightfield` with
 * the frame's `pixelSize`) where the map saw them.
 *
 * @example
 * ```ts
 * const map = scene.heightMap(mesh, { resolution: 512 });
 * scene.addHeightfield(
 *   { ...map, width: map.frame.width, height: map.frame.height, pixelSize: map.frame.pixelSize },
 *   { placement: heightMapPlacement(map.frame) },
 * );
 * ```
 */
export function heightMapPlacement(frame: HeightMapFrame): Placement {
  const { right, up, origin, pixelSize } = frame;
  const n = cross(right, up);
  const u = 0.5 * (frame.width - 1) * pixelSize;
  const v = -0.5 * (frame.height - 1) * pixelSize;
  const at = (k: number) => origin[k] + u * right[k] + v * up[k];
  return {
    rotation: [right[0], up[0], n[0], right[1], up[1], n[1], right[2], up[2], n[2]],
    translation: [at(0), at(1), at(2)],
    scale: [1, 1, 1],
  };
}

/** Internal: one mesh or several, as a non-empty list. */
export function heightMapMeshList(meshes: MeshHandle | readonly MeshHandle[]): readonly MeshHandle[] {
  const list = Array.isArray(meshes) ? (meshes as readonly MeshHandle[]) : [meshes as MeshHandle];
  if (list.length === 0) {
    throw invalid("heightMap needs at least one mesh");
  }
  return list;
}

function checkObject(value: unknown, name: string): void {
  if (typeof value !== "object" || value === null) {
    throw invalid(`${name} must be an object`);
  }
}

function vec3(value: unknown, name: string): Vec3 {
  const v = value as ArrayLike<unknown> | null | undefined;
  if (v == null || typeof v !== "object" || v.length !== 3) {
    throw invalid(`${name} must be 3 numbers`);
  }
  const out: Vec3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const c = v[k];
    if (typeof c !== "number" || !Number.isFinite(c)) {
      throw invalid(`${name} must be 3 finite numbers`);
    }
    out[k] = c;
  }
  return out;
}

function finite(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw invalid(`${name} must be a finite number`);
  }
  return value;
}

function positive(value: unknown, name: string): number {
  const v = finite(value, name);
  if (v <= 0) {
    throw invalid(`${name} must be above zero`);
  }
  return v;
}

function whole(value: unknown, min: number, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min || value > 0xffff_ffff) {
    throw invalid(`${name} must be a whole number of at least ${min}`);
  }
  return value;
}

/** Internal: whether `arg` is a frame rather than options to fit one. */
export function isHeightMapFrame(arg: HeightMapFrame | HeightMapOptions | undefined): arg is HeightMapFrame {
  return typeof arg === "object" && arg !== null && "origin" in arg;
}

/** Internal: checked options as the binding takes them (NaN for a default); plain numbers, safe to post. */
export function encodeHeightMapOptions(options: HeightMapOptions = {}): number[] {
  checkObject(options, "options");
  const direction = options.direction === undefined ? [0, 0, -1] : vec3(options.direction, "options.direction");
  const up = options.up === undefined ? [Number.NaN, 0, 0] : vec3(options.up, "options.up");
  const resolution = options.resolution === undefined ? 1024 : whole(options.resolution, 2, "options.resolution");
  const pixelSize = options.pixelSize === undefined ? Number.NaN : positive(options.pixelSize, "options.pixelSize");
  const margin = options.margin === undefined ? 0 : finite(options.margin, "options.margin");
  let range: readonly number[] = [Number.NaN, Number.NaN];
  if (options.range !== undefined) {
    const r = options.range as ArrayLike<unknown> | null;
    if (r === null || typeof r !== "object" || r.length !== 2) {
      throw invalid("options.range must be [low, high]");
    }
    range = [finite(r[0], "options.range"), finite(r[1], "options.range")];
  }
  return [...direction, ...up, resolution, pixelSize, margin, ...range];
}

/** Internal: {@link encodeHeightMapOptions} undone, for the Worker engine. */
export function decodeHeightMapOptions(raw: readonly number[]): HeightMapOptions {
  return {
    direction: [raw[0], raw[1], raw[2]],
    ...(Number.isNaN(raw[3]) ? {} : { up: [raw[3], raw[4], raw[5]] }),
    resolution: raw[6],
    ...(Number.isNaN(raw[7]) ? {} : { pixelSize: raw[7] }),
    margin: raw[8],
    ...(Number.isNaN(raw[9]) ? {} : { range: [raw[9], raw[10]] }),
  };
}

/** Internal: a frame as the binding takes it; plain numbers, safe to post. */
export function encodeHeightMapFrame(frame: HeightMapFrame): number[] {
  checkObject(frame, "frame");
  return [
    whole(frame.width, 1, "frame.width"),
    whole(frame.height, 1, "frame.height"),
    ...vec3(frame.origin, "frame.origin"),
    ...vec3(frame.right, "frame.right"),
    ...vec3(frame.up, "frame.up"),
    finite(frame.pixelSize, "frame.pixelSize"),
    finite(frame.range, "frame.range"),
  ];
}

/** Internal: a frame from the binding's 13 values. */
export function decodeHeightMapFrame(raw: ArrayLike<number>): HeightMapFrame {
  // `+ 0` turns -0 into 0.
  const at = (k: number): Vec3 => [raw[k] + 0, raw[k + 1] + 0, raw[k + 2] + 0];
  const right = at(5);
  const up = at(8);
  return {
    width: raw[0],
    height: raw[1],
    origin: at(2),
    right,
    up,
    normal: cross(right, up),
    pixelSize: raw[11],
    range: raw[12],
  };
}

/** Internal: rows `[first, end)` of a map of `height` rows, all of them by default. */
export function checkHeightMapRows(rows: readonly [number, number] | undefined, height: number): [number, number] {
  if (rows === undefined) {
    return [0, height];
  }
  if (!Array.isArray(rows) || rows.length !== 2) {
    throw invalid("rows must be [first, end]");
  }
  const first = whole(rows[0], 0, "rows[0]");
  const end = whole(rows[1], 0, "rows[1]");
  if (first > end || end > height) {
    throw invalid("rows must lie within the frame");
  }
  return [first, end];
}

/** Internal: a grid of heights, checked by type and size; its values and ranges are checked by the generator. */
export function checkHeightfield(field: Heightfield): CheckedHeightfield {
  checkObject(field, "a height field");
  const width = whole(field.width, 2, "width");
  const height = whole(field.height, 2, "height");
  const src = field.heights;
  if (src == null || typeof src !== "object" || typeof src.length !== "number") {
    throw invalid("heights must be an array of numbers");
  }
  if (src.length !== width * height) {
    throw invalid("heights must hold width times height values");
  }
  const heights = src instanceof Float32Array ? src : Float32Array.from(src);
  const size = field.pixelSize ?? 1;
  let pixelSize: [number, number];
  if (typeof size === "number") {
    pixelSize = [positive(size, "pixelSize"), size];
  } else {
    if (!Array.isArray(size) || size.length !== 2) {
      throw invalid("pixelSize must be a number or 2 numbers");
    }
    pixelSize = [positive(size[0], "pixelSize"), positive(size[1], "pixelSize")];
  }
  return { heights, width, height, pixelSize, base: positive(field.base ?? 1, "base") };
}

/** Internal: {@link AddHeightfieldOptions}, checked. */
export function checkHeightfieldPlacement(options: AddHeightfieldOptions = {}): CheckedHeightfieldPlacement {
  checkObject(options, "options");
  const normals = options.normals ?? true;
  if (typeof normals !== "boolean") {
    throw invalid("options.normals must be a boolean");
  }
  const p = options.placement;
  if (p === undefined) {
    const identity: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    return { rotation: [...identity], translation: [0, 0, 0], normals };
  }
  checkObject(p, "options.placement");
  return {
    rotation: Array.from(rotationMatrix(p.rotation)),
    translation: vec3(p.translation, "options.placement.translation"),
    ...(p.scale === undefined ? {} : { scale: vec3(p.scale, "options.placement.scale") }),
    normals,
  };
}

/** Internal: the binding's options of a checked grid. */
export function encodeHeightfield(f: CheckedHeightfield): Float64Array {
  return Float64Array.of(f.pixelSize[0], f.pixelSize[1], f.base);
}
