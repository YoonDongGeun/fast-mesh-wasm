// Primitive shapes: their options, their encoding for the binding and the
// Worker protocol, and the decoding of the generated buffers. Wasm-free, so
// that `WorkerScene` can check and encode a shape on the main thread; the
// generator itself is `generatePrimitive` (`primitives-wasm.ts`).
import type { Vec3 } from "./api.js";
import { FastMeshError } from "./core.js";

/** The axis a shape of revolution turns around, or a plane's normal. */
export type PrimitiveAxis = "x" | "y" | "z";

/**
 * A box with a size per axis, its faces split into grids. Exact: its volume
 * is the product of its sizes.
 */
export interface BoxShape {
  kind: "box";
  /** Extent along X, Y and Z, or one number for a cube. Default `1`. */
  size?: number | Readonly<Vec3>;
  /** Grid cells along X, Y and Z (whole numbers, at least 1), or one number for all three. Default `1`. */
  segments?: number | Readonly<Vec3>;
}

/** A capped cylinder around `axis`. */
export interface CylinderShape {
  kind: "cylinder";
  /** Default `0.5`. */
  radius?: number;
  /** Extent along the axis. Default `1`. */
  height?: number;
  /** Sides around the axis, at least 3. Default `32`. */
  segments?: number;
  /** Bands along the axis, at least 1. Default `1`. */
  rings?: number;
  /** Concentric bands of each cap, at least 1. Default `1` (a fan). */
  capRings?: number;
  /** Default `"y"`. */
  axis?: PrimitiveAxis;
}

/** A cone around `axis`, or a frustum when both radii are positive. */
export interface ConeShape {
  kind: "cone";
  /** Radius at `-axis`; `0` puts the apex there. Default `0.5`. */
  radius?: number;
  /** Radius at `+axis`; `0` (the default) puts the apex there. */
  topRadius?: number;
  /** Extent along the axis. Default `1`. */
  height?: number;
  /** Sides around the axis, at least 3. Default `32`. */
  segments?: number;
  /** Bands along the axis, at least 1. Default `1`. */
  rings?: number;
  /** Concentric bands of each cap, at least 1. Default `1` (a fan). */
  capRings?: number;
  /** Default `"y"`. */
  axis?: PrimitiveAxis;
}

/** A latitude-longitude sphere with a vertex at each pole, on `axis`. */
export interface UvSphereShape {
  kind: "uvSphere";
  /** Default `0.5`. */
  radius?: number;
  /** Meridians, at least 3. Default `32`. */
  segments?: number;
  /** Latitude bands from pole to pole, at least 2. Default `16`. */
  rings?: number;
  /** The pole axis. Default `"y"`. */
  axis?: PrimitiveAxis;
}

/**
 * A geodesic sphere: an icosahedron whose faces are split into triangular
 * grids and pushed onto the sphere. Its triangles are near equal in size.
 */
export interface IcosphereShape {
  kind: "icosphere";
  /** Default `0.5`. */
  radius?: number;
  /** Each level splits every triangle into four: `20 · 4^subdivisions` triangles. At most 16. Default `3`. */
  subdivisions?: number;
}

/** A ring torus around `axis`. */
export interface TorusShape {
  kind: "torus";
  /** From the axis to the tube's centre. Default `0.4`. */
  majorRadius?: number;
  /** The tube's radius, below `majorRadius`. Default `0.1`. */
  minorRadius?: number;
  /** Sides around the axis, at least 3. Default `48`. */
  segments?: number;
  /** Sides around the tube, at least 3. Default `24`. */
  tubeSegments?: number;
  /** Default `"y"`. */
  axis?: PrimitiveAxis;
}

/** An arrow, a capped shaft and a cone head, pointing to `+axis`. */
export interface ArrowShape {
  kind: "arrow";
  /** From the tail to the tip. Default `1`. */
  length?: number;
  /** Default `0.05`. */
  shaftRadius?: number;
  /** The head's base radius, above `shaftRadius`. Default `0.1`. */
  headRadius?: number;
  /** The head's extent along the axis, below `length`. Default `0.25`. */
  headLength?: number;
  /** Sides around the axis, at least 3. Default `32`. */
  segments?: number;
  /** Default `"y"`. */
  axis?: PrimitiveAxis;
}

/**
 * A right prism over a regular polygon, along `axis`. Exact. One corner
 * lies on `+z` for the `"y"` axis, on `+x` for `"z"` and on `+y` for `"x"`.
 */
export interface PrismShape {
  kind: "prism";
  /** Corners of the polygon, at least 3. Default `6`. */
  sides?: number;
  /** From the axis to each corner (the circumradius). Default `0.5`. */
  radius?: number;
  /** Extent along the axis. Default `1`. */
  height?: number;
  /** Default `"y"`. */
  axis?: PrimitiveAxis;
}

/**
 * A flat rectangular grid facing `+axis`: open, the one shape with a
 * boundary.
 */
export interface PlaneShape {
  kind: "plane";
  /**
   * Extents along the other two axes in `x, y, z` order (for the default
   * `"y"` normal: X, then Z), or one number for a square. Default `1`.
   */
  size?: number | readonly [number, number];
  /** Grid cells along the same two axes (at least 1), or one number for both. Default `1`. */
  segments?: number | readonly [number, number];
  /** The normal. Default `"y"`. */
  axis?: PrimitiveAxis;
}

/**
 * A primitive shape and its parameters, for `generatePrimitive`. Every shape
 * is centred on the origin, welded (one vertex per point of the surface),
 * closed (except the plane: no boundary edges) and wound counter-clockwise
 * seen from outside, so its normals point outward and its signed volume is
 * positive. Curved shapes are inscribed polygons: their volume approaches
 * the analytic one from below as the resolution grows.
 *
 * Lengths must be finite and in `[1e-30, 1e30]`; resolutions are whole
 * numbers with the lower bound each one names. A shape with more vertices
 * or triangles than `u32` ids can name is refused.
 */
export type PrimitiveShape =
  | BoxShape
  | CylinderShape
  | ConeShape
  | UvSphereShape
  | IcosphereShape
  | TorusShape
  | ArrowShape
  | PrismShape
  | PlaneShape;

/** The kinds of {@link PrimitiveShape}. */
export type PrimitiveShapeKind = PrimitiveShape["kind"];

/** Every {@link PrimitiveShapeKind}, in a stable order (for pickers). */
export const PRIMITIVE_SHAPE_KINDS: readonly PrimitiveShapeKind[] = [
  "box",
  "cylinder",
  "cone",
  "uvSphere",
  "icosphere",
  "torus",
  "arrow",
  "prism",
  "plane",
];

/** A generated primitive: flat positions and triangle indices, ready for `addMesh` or a `BufferGeometry`. */
export interface PrimitiveMesh {
  /** 3 per vertex, centred on the origin. */
  positions: Float32Array;
  /** 3 per triangle, counter-clockwise seen from outside. */
  indices: Uint32Array;
}

/**
 * Internal: a shape as the binding takes it, the kind's code and its
 * parameters (NaN for a default). Structured-cloneable, for the Worker.
 */
export interface EncodedPrimitive {
  kind: number;
  params: number[];
}

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

const AXES: Record<PrimitiveAxis, number> = { x: 0, y: 1, z: 2 };

/** One shape's parameter reader: each value or NaN, checked by type. */
function reader(kind: string, shape: object) {
  const get = (name: string): unknown => (shape as Record<string, unknown>)[name];
  const length = (name: string, value: unknown = get(name)): number => {
    if (value === undefined) {
      return Number.NaN;
    }
    if (typeof value !== "number" || Number.isNaN(value)) {
      throw invalid(`${kind} ${name} must be a number`);
    }
    return value;
  };
  const count = (name: string, value: unknown = get(name)): number => {
    if (value === undefined) {
      return Number.NaN;
    }
    if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
      throw invalid(`${kind} ${name} must be a whole number`);
    }
    return value;
  };
  const axis = (): number => {
    const value = get("axis");
    if (value === undefined) {
      return Number.NaN;
    }
    const code = AXES[value as PrimitiveAxis];
    if (code === undefined) {
      throw invalid(`${kind} axis must be "x", "y" or "z"`);
    }
    return code;
  };
  /** A number for every component, or `n` components. */
  const tuple = (name: string, n: number, each: typeof length): number[] => {
    const value = get(name);
    if (value === undefined || typeof value === "number") {
      return Array.from({ length: n }, () => each(name, value));
    }
    const a = value as readonly unknown[] | null;
    if (!Array.isArray(a) || a.length !== n) {
      throw invalid(`${kind} ${name} must be a number or ${n} numbers`);
    }
    return a.map((v) => each(name, v));
  };
  return { length, count, axis, tuple };
}

/**
 * Internal: checks `shape` by type and encodes it for the binding (see
 * `crates/fast-mesh-wasm/src/primitives.rs`). Ranges are checked by the
 * generator.
 *
 * @throws `INVALID_ARGUMENT` for an unknown kind, a value of the wrong type
 * or a resolution that is not a whole number.
 */
export function encodePrimitive(shape: PrimitiveShape): EncodedPrimitive {
  if (typeof shape !== "object" || shape === null) {
    throw invalid("a primitive shape must be an object with a kind");
  }
  const kind = (shape as { kind?: unknown }).kind;
  const r = reader(String(kind), shape);
  switch (kind) {
    case "box":
      return { kind: 0, params: [...r.tuple("size", 3, r.length), ...r.tuple("segments", 3, r.count)] };
    case "cylinder":
      return {
        kind: 1,
        params: [r.length("radius"), r.length("height"), r.count("segments"), r.count("rings"), r.count("capRings"), r.axis()],
      };
    case "cone":
      return {
        kind: 2,
        params: [
          r.length("radius"),
          r.length("topRadius"),
          r.length("height"),
          r.count("segments"),
          r.count("rings"),
          r.count("capRings"),
          r.axis(),
        ],
      };
    case "uvSphere":
      return { kind: 3, params: [r.length("radius"), r.count("segments"), r.count("rings"), r.axis()] };
    case "icosphere":
      return { kind: 4, params: [r.length("radius"), r.count("subdivisions")] };
    case "torus":
      return {
        kind: 5,
        params: [r.length("majorRadius"), r.length("minorRadius"), r.count("segments"), r.count("tubeSegments"), r.axis()],
      };
    case "arrow":
      return {
        kind: 6,
        params: [
          r.length("length"),
          r.length("shaftRadius"),
          r.length("headRadius"),
          r.length("headLength"),
          r.count("segments"),
          r.axis(),
        ],
      };
    case "prism":
      return { kind: 7, params: [r.count("sides"), r.length("radius"), r.length("height"), r.axis()] };
    case "plane":
      return { kind: 8, params: [...r.tuple("size", 2, r.length), ...r.tuple("segments", 2, r.count), r.axis()] };
    default:
      throw invalid(`unknown primitive kind "${String(kind)}"`);
  }
}

/**
 * Internal: the binding's result, `[vertices, triangles, positions as f32
 * bits..., indices...]`, as fresh arrays (each its own buffer, so a Worker
 * can transfer them).
 */
export function decodePrimitiveMesh(raw: Uint32Array): PrimitiveMesh {
  const v = raw[0];
  const t = raw[1];
  const start = raw.byteOffset + 8;
  const positions = new Float32Array(raw.buffer.slice(start, start + 12 * v));
  const indices = raw.slice(2 + 3 * v, 2 + 3 * v + 3 * t);
  return { positions, indices };
}
