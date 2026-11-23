// Object info (#257): the numbers the info box shows for one mesh, and the
// boxes the bounding box overlay draws. Counts, edges, components, the mean
// edge length and the boxes come from the render buffers in one pass over the
// index buffer (`meshTopology`, built once per geometry change, and checked
// against the SDK's `analyze` and `bounds` in the tests); area and volume are
// the SDK's own `measure`. Pure (no three, no DOM, no React), so vitest
// covers it against the engine.
import type { MeshHandle } from "fast-mesh-wasm";
import type { Vec3 } from "../../../io/mesh-data";
import { usedVertices } from "./overlay-geometry";

/** An axis-aligned box. */
export interface Box {
  min: Vec3;
  max: Vec3;
}

/** Which frame the info box and the bounding box overlay read: the world's axes or the mesh's own. */
export type InfoSpace = "world" | "local";

/**
 * What one pass over the index buffer finds: the live triangles, the vertex
 * slots they use, their unique edges and their edge-connected components. A
 * removed triangle slot reads `0, 0, 0` and is skipped.
 */
export interface MeshTopology {
  /** Live triangles. */
  triangles: number;
  /** The vertex slots the live triangles use, ascending. */
  used: Uint32Array;
  /** The unique undirected edges, as `lo, hi` pairs with `lo < hi`, ascending. */
  edges: Uint32Array;
  /** Groups of live triangles connected through shared edges (as the SDK's `analyze` counts them). */
  components: number;
}

/** Whether triangle slot `t` is live: a removed slot reads `0, 0, 0`. */
function liveTriangle(a: number, b: number, c: number, n: number): boolean {
  return (a !== 0 || b !== 0 || c !== 0) && a < n && b < n && c < n;
}

/** The root of `x`, halving the path on the way. */
function root(parent: Uint32Array, x: number): number {
  while (parent[x] !== x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

/**
 * The live triangles, used vertices, unique edges and components of an
 * index buffer. Linear time and memory: each edge is bucketed by its lower
 * vertex (a counting sort) with the triangle it came from, each small bucket
 * is sorted, its repeats are one edge and union their triangles. A million
 * triangles take tens of milliseconds.
 */
export function meshTopology(indices: ArrayLike<number>, vertexSlots: number): MeshTopology {
  const triSlots = Math.floor(indices.length / 3);
  const n = Math.max(0, vertexSlots);
  // Edges per lower vertex, then their start offsets (exclusive prefix sum).
  const start = new Uint32Array(n + 1);
  let triangles = 0;
  for (let t = 0; t < triSlots; t++) {
    const a = indices[3 * t];
    const b = indices[3 * t + 1];
    const c = indices[3 * t + 2];
    if (!liveTriangle(a, b, c, n)) {
      continue;
    }
    triangles++;
    if (a !== b) start[a < b ? a : b]++;
    if (b !== c) start[b < c ? b : c]++;
    if (c !== a) start[c < a ? c : a]++;
  }
  let sum = 0;
  for (let v = 0; v <= n; v++) {
    const k = start[v];
    start[v] = sum;
    sum += k;
  }
  const his = new Uint32Array(sum);
  const tris = new Uint32Array(sum);
  const cursor = start.slice(0, n);
  const put = (u: number, w: number, t: number) => {
    const lo = u < w ? u : w;
    const at = cursor[lo]++;
    his[at] = u < w ? w : u;
    tris[at] = t;
  };
  const parent = new Uint32Array(triSlots);
  for (let t = 0; t < triSlots; t++) {
    parent[t] = t;
    const a = indices[3 * t];
    const b = indices[3 * t + 1];
    const c = indices[3 * t + 2];
    if (!liveTriangle(a, b, c, n)) {
      continue;
    }
    if (a !== b) put(a, b, t);
    if (b !== c) put(b, c, t);
    if (c !== a) put(c, a, t);
  }
  // Sort each bucket by its upper vertex (a vertex has a handful of edges):
  // a run of one upper vertex is one edge, and its triangles are connected.
  let edgeCount = 0;
  for (let v = 0; v < n; v++) {
    const s = start[v];
    const e = start[v + 1];
    for (let i = s + 1; i < e; i++) {
      const x = his[i];
      const xt = tris[i];
      let j = i - 1;
      while (j >= s && his[j] > x) {
        his[j + 1] = his[j];
        tris[j + 1] = tris[j];
        j--;
      }
      his[j + 1] = x;
      tris[j + 1] = xt;
    }
    for (let i = s; i < e; i++) {
      if (i === s || his[i] !== his[i - 1]) {
        edgeCount++;
      } else {
        const ra = root(parent, tris[i - 1]);
        const rb = root(parent, tris[i]);
        if (ra !== rb) {
          parent[ra < rb ? rb : ra] = ra < rb ? ra : rb;
        }
      }
    }
  }
  const edges = new Uint32Array(2 * edgeCount);
  let k = 0;
  for (let v = 0; v < n; v++) {
    const s = start[v];
    const e = start[v + 1];
    for (let i = s; i < e; i++) {
      if (i === s || his[i] !== his[i - 1]) {
        edges[k++] = v;
        edges[k++] = his[i];
      }
    }
  }
  let components = 0;
  for (let t = 0; t < triSlots; t++) {
    if (parent[t] === t && liveTriangle(indices[3 * t], indices[3 * t + 1], indices[3 * t + 2], n)) {
      components++;
    }
  }
  return { triangles, used: usedVertices(indices, n), edges, components };
}

/**
 * The mean length of the edges, through the linear part of the column-major
 * 4x4 `m` (its scale applies; the translation does not matter), or of the
 * local positions with `m` `null`. 0 without edges.
 */
export function meanEdgeLength(
  positions: ArrayLike<number>,
  edges: ArrayLike<number>,
  m: ArrayLike<number> | null,
): number {
  const count = Math.floor(edges.length / 2);
  if (count === 0) {
    return 0;
  }
  const [m0, m1, m2, m4, m5, m6, m8, m9, m10] =
    m === null ? [1, 0, 0, 0, 1, 0, 0, 0, 1] : [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const a = 3 * edges[2 * i];
    const b = 3 * edges[2 * i + 1];
    const dx = positions[b] - positions[a];
    const dy = positions[b + 1] - positions[a + 1];
    const dz = positions[b + 2] - positions[a + 2];
    const x = m0 * dx + m4 * dy + m8 * dz;
    const y = m1 * dx + m5 * dy + m9 * dz;
    const z = m2 * dx + m6 * dy + m10 * dz;
    sum += Math.sqrt(x * x + y * y + z * z);
  }
  return sum / count;
}

/**
 * The box around the used vertices through the linear part of `m` (no
 * translation), or around the local positions with `m` `null`. `null`
 * without used vertices.
 */
export function linearBox(positions: ArrayLike<number>, used: ArrayLike<number>, m: ArrayLike<number> | null): Box | null {
  if (used.length === 0) {
    return null;
  }
  const [m0, m1, m2, m4, m5, m6, m8, m9, m10] =
    m === null ? [1, 0, 0, 0, 1, 0, 0, 0, 1] : [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  let x0 = Infinity;
  let y0 = Infinity;
  let z0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  let z1 = -Infinity;
  for (let i = 0; i < used.length; i++) {
    const v = 3 * used[i];
    const px = positions[v];
    const py = positions[v + 1];
    const pz = positions[v + 2];
    const x = m0 * px + m4 * py + m8 * pz;
    const y = m1 * px + m5 * py + m9 * pz;
    const z = m2 * px + m6 * py + m10 * pz;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
    if (z < z0) z0 = z;
    if (z > z1) z1 = z;
  }
  return { min: [x0, y0, z0], max: [x1, y1, z1] };
}

/** `box` moved by `t`. */
export function offsetBox(box: Box, t: readonly [number, number, number]): Box {
  return {
    min: [box.min[0] + t[0], box.min[1] + t[1], box.min[2] + t[2]],
    max: [box.max[0] + t[0], box.max[1] + t[1], box.max[2] + t[2]],
  };
}

/** The box's size along each axis. */
export function boxSize(box: Box): Vec3 {
  return [box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]];
}

/** The box's centre. */
export function boxCenter(box: Box): Vec3 {
  return [(box.min[0] + box.max[0]) / 2, (box.min[1] + box.max[1]) / 2, (box.min[2] + box.max[2]) / 2];
}

/**
 * The world-aligned box of a mesh as its matrix changes, without walking
 * every vertex on every move: the box through the matrix's linear part is
 * kept, so a move that only translates (a translate drag) shifts it, and
 * only a rotation or scale walks the used vertices again.
 */
export class WorldBoxTracker {
  #key: string | null = null;
  #linear: Box | null = null;
  /** How many times the used vertices were walked (for the tests). */
  walks = 0;

  constructor(
    private readonly positions: ArrayLike<number>,
    private readonly used: ArrayLike<number>,
  ) {}

  /** The world-aligned box under the column-major 4x4 `m`, `null` without used vertices. */
  box(m: ArrayLike<number>): Box | null {
    const key = `${m[0]},${m[1]},${m[2]},${m[4]},${m[5]},${m[6]},${m[8]},${m[9]},${m[10]}`;
    if (key !== this.#key) {
      this.#key = key;
      this.#linear = linearBox(this.positions, this.used, m);
      this.walks++;
    }
    return this.#linear === null ? null : offsetBox(this.#linear, [m[12], m[13], m[14]]);
  }
}

/** The info box's numbers for one mesh. Lengths in mm, areas in mm², volumes in mm³. */
export interface MeshInfo {
  /** Live triangles. */
  triangles: number;
  /** Vertices the live triangles use. */
  vertices: number;
  /** Unique edges of the live triangles. */
  edges: number;
  /** Edge-connected triangle components (as the SDK's `analyze` counts them). */
  components: number;
  /** Surface area in the world (the SDK's `measure`), `null` when the query failed. */
  area: number | null;
  /** Enclosed volume (the SDK's `measure`), `null` when the mesh is not watertight or the query failed. */
  volume: number | null;
  /** Why `volume` is `null` for a measured mesh. */
  volumeReason: "open" | "non-manifold" | null;
  /** Mean unique-edge length in the world (the placement's scale applies). */
  meanEdge: number;
  /** The world-aligned box, `null` without live triangles. */
  world: Box | null;
  /** The box in the mesh's own frame (its placement undone), `null` without live triangles. */
  local: Box | null;
}

/** What `readMeshInfo` needs of a scene: a local `Scene` and a `WorkerScene` both fit. */
export interface InfoScene {
  views(mesh: MeshHandle): { positions(): Float32Array; indices(): Uint32Array };
  matrix4(mesh: MeshHandle): ArrayLike<number>;
  origin(mesh: MeshHandle): readonly [number, number, number];
  readonly inspect?: { measure(mesh: MeshHandle): unknown };
}

/** The topology's view-derived numbers: counts, mean edge, boxes. Reads the views now (before any other scene call). */
export function viewInfo(
  positions: ArrayLike<number>,
  topology: MeshTopology,
  matrix: ArrayLike<number>,
  origin: readonly [number, number, number],
): Pick<MeshInfo, "triangles" | "vertices" | "edges" | "components" | "meanEdge" | "world" | "local"> {
  const local = linearBox(positions, topology.used, null);
  const world = linearBox(positions, topology.used, matrix);
  return {
    triangles: topology.triangles,
    vertices: topology.used.length,
    edges: topology.edges.length / 2,
    components: topology.components,
    meanEdge: meanEdgeLength(positions, topology.edges, matrix),
    world: world === null ? null : offsetBox(world, [matrix[12], matrix[13], matrix[14]]),
    local: local === null ? null : offsetBox(local, origin),
  };
}

/** Area and volume from the SDK's `measure`, as `MeshInfo` holds them. */
export type MeasureInfo = Pick<MeshInfo, "area" | "volume" | "volumeReason">;

/** No measurement (the query failed, or the scene has none). */
export const NO_MEASURE: MeasureInfo = { area: null, volume: null, volumeReason: null };

/**
 * The SDK's `measure` of the mesh (world coordinates), which a `WorkerScene`
 * answers with a promise. A failed query (the mesh removed meanwhile) reads
 * as no measurement.
 */
export async function readMeasureInfo(scene: Pick<InfoScene, "inspect">, mesh: MeshHandle): Promise<MeasureInfo> {
  const inspect = scene.inspect;
  if (inspect === undefined) {
    return NO_MEASURE;
  }
  try {
    const m = (await inspect.measure(mesh)) as { area: number; volume: number | null; volumeReason: MeshInfo["volumeReason"] };
    return { area: m.area, volume: m.volume, volumeReason: m.volumeReason };
  } catch {
    return NO_MEASURE;
  }
}

/**
 * The mesh's info: the buffers' numbers read at once (`topology` when it is
 * already built for the current buffers), then the SDK's `measure`. Throws
 * when the mesh is gone before the buffers are read.
 */
export async function readMeshInfo(scene: InfoScene, mesh: MeshHandle, topology?: MeshTopology): Promise<MeshInfo> {
  const views = scene.views(mesh);
  const positions = views.positions();
  const topo = topology ?? meshTopology(views.indices(), positions.length / 3);
  const base = viewInfo(positions, topo, scene.matrix4(mesh), scene.origin(mesh));
  return { ...base, ...(await readMeasureInfo(scene, mesh)) };
}
