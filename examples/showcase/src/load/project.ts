// Showcase project save + reopen (issue #126): the session (meshes +
// placements + coloring/section state) to a file and back.
//
// The file is JSON (`PROJECT_EXTENSION`, `PROJECT_MIME`), forward-compatible:
// `version` numbers the format and the loader is tolerant — unknown fields
// are ignored, missing optionals take defaults, a newer `version` still loads
// what this code understands, and entries that fail validation are dropped
// (a coloring with no live targets, a section with a bad plane) instead of
// failing the whole file. Mesh geometry rides as base64 little-endian bytes
// (f64 positions, u32 indices), so a round trip is bit-exact.
//
// Units (issue #145): positions are in the world unit, millimetres, whatever
// the display unit. `units` names how the project displays them; a mesh's
// `importScale` records how its file's numbers became millimetres. That
// scale is already in the positions and is never applied here, so a reload
// cannot scale a mesh twice. Both are optional: files without them load as
// before, on the global default unit (`version` stays 1, the fields are
// additive and older builds ignore them).
//
// The pure core (`encodeF64`/`decodeF64`, `encodeU32`/`decodeU32`,
// `buildProjectDoc`, `parseProjectDoc`) touches no scene and is unit-tested;
// `collectProject`/`restoreProject` are the thin store/app adapters, and
// `saveProjectFile`/`openProjectFile` wrap the File System Access API with an
// anchor/input fallback.
import type { MeshHandle } from "fast-mesh-wasm";
import type { DistanceColoring } from "fast-mesh-wasm/store";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState } from "../scene/app-state";
import { type ImportScale, isImportUnit, type LengthUnit, NO_IMPORT_SCALE } from "../panels/units";
import { bounds } from "./mesh-data";

/** The project format version written by `buildProjectDoc`. */
export const PROJECT_VERSION = 1;
/** The `app` tag written into project files (informational: never required). */
export const PROJECT_APP = "fast-mesh-showcase";
/** The project file extension and download MIME type. */
export const PROJECT_EXTENSION = ".fmsproj.json";
export const PROJECT_MIME = "application/json";

/** A placement as stored: row-major rotation, translation, optional scale. */
export interface ProjectPlacement {
  rotation: number[];
  translation: [x: number, y: number, z: number];
  scale?: [x: number, y: number, z: number];
}

/**
 * A mesh's distance coloring as stored: the store's rule with its target
 * meshes as indices into the doc's `meshes`. `maxDistance` is `null` for
 * `Infinity` (JSON has no infinities); the remaining rule fields ride along
 * opaquely and are validated lightly on load (`map` a string, `range`
 * `[lo, hi]` with `lo < hi`).
 */
export interface ProjectColoring {
  targets: number[];
  maxDistance: number | null;
  sign?: unknown;
  map: unknown;
  range: unknown;
  below?: unknown;
  above?: unknown;
  center?: unknown;
  base?: unknown;
  nan?: unknown;
}

/** One mesh as stored: geometry as base64 little-endian bytes. */
export interface ProjectMesh {
  name: string;
  color: number;
  opacity: number;
  visible: boolean;
  /** Base64 of the local positions as little-endian f64. */
  positions: string;
  /** Base64 of the indices as little-endian u32. */
  indices: string;
  placement: ProjectPlacement;
  coloring: ProjectColoring | null;
  /** How the positions became mm on load (absent: as they were). Already applied: provenance only. */
  importScale?: ImportScale;
}

/** One section as stored: meshes as indices into the doc's `meshes`. */
export interface ProjectSection {
  plane: { origin: [x: number, y: number, z: number]; normal: [x: number, y: number, z: number] };
  meshes: number[];
  fill: boolean;
}

/** A parsed project file. Unknown fields are ignored by the loader. */
export interface ProjectDoc {
  version: number;
  app: string;
  meshes: ProjectMesh[];
  sections: ProjectSection[];
  selection: number | null;
  wireframe: boolean;
  edgeSoftness: number;
  /** The project's display unit (absent in older files: the global default). */
  units?: LengthUnit;
}

// ---- Base64 over raw bytes (Buffer in Node/tests, btoa/atob in browsers) ----

function bytesToB64(bytes: Uint8Array): string {
  const bufferOf = (globalThis as { Buffer?: { from(b: Uint8Array): { toString(e: string): string } } }).Buffer;
  if (bufferOf !== undefined) {
    return bufferOf.from(bytes).toString("base64");
  }
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function b64ToBytes(b64: string, what: string): Uint8Array {
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(b64) || b64.length % 4 !== 0) {
    throw new Error(`the project file's ${what} is not valid base64`);
  }
  const bufferFrom = (globalThis as { Buffer?: { from(s: string, e: string): Uint8Array } }).Buffer;
  if (bufferFrom !== undefined) {
    const copy = Uint8Array.from(bufferFrom.from(b64, "base64"));
    return copy;
  }
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

const LITTLE_ENDIAN: boolean = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** Encodes numbers as little-endian f64 bytes, base64ed. */
export function encodeF64(values: ArrayLike<number>): string {
  const n = values.length;
  const bytes = new Uint8Array(8 * n);
  if (values instanceof Float64Array && LITTLE_ENDIAN) {
    bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  } else {
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < n; i++) {
      view.setFloat64(8 * i, Number(values[i]), true);
    }
  }
  return bytesToB64(bytes);
}

/** Decodes `encodeF64`. @throws when the bytes are not whole f64s. */
export function decodeF64(b64: string, what: string): Float64Array {
  const bytes = b64ToBytes(b64, what);
  if (bytes.byteLength % 8 !== 0) {
    throw new Error(`the project file's ${what} is not whole float64 values`);
  }
  // Copy onto an aligned buffer first: the decoded offset is arbitrary.
  const aligned = new Uint8Array(bytes.byteLength);
  aligned.set(bytes);
  const out = new Float64Array(aligned.buffer);
  if (!LITTLE_ENDIAN) {
    const view = new DataView(aligned.buffer);
    for (let i = 0; i < out.length; i++) {
      out[i] = view.getFloat64(8 * i, true);
    }
  }
  return out;
}

/** Encodes numbers (truncated to uint32) as little-endian u32 bytes, base64ed. */
export function encodeU32(values: ArrayLike<number>): string {
  const n = values.length;
  const bytes = new Uint8Array(4 * n);
  if (values instanceof Uint32Array && LITTLE_ENDIAN) {
    bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  } else {
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < n; i++) {
      view.setUint32(4 * i, Number(values[i]) >>> 0, true);
    }
  }
  return bytesToB64(bytes);
}

/** Decodes `encodeU32`. @throws when the bytes are not whole u32s. */
export function decodeU32(b64: string, what: string): Uint32Array {
  const bytes = b64ToBytes(b64, what);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`the project file's ${what} is not whole uint32 values`);
  }
  const aligned = new Uint8Array(bytes.byteLength);
  aligned.set(bytes);
  const out = new Uint32Array(aligned.buffer);
  if (!LITTLE_ENDIAN) {
    const view = new DataView(aligned.buffer);
    for (let i = 0; i < out.length; i++) {
      out[i] = view.getUint32(4 * i, true);
    }
  }
  return out;
}

// ---- Building (pure: no scene) ----

/** A mesh's serializable state, with coloring targets as mesh indices. */
export interface ProjectMeshInput {
  name: string;
  color: number;
  opacity: number;
  visible: boolean;
  positions: ArrayLike<number>;
  indices: ArrayLike<number>;
  rotation: ArrayLike<number>;
  translation: [x: number, y: number, z: number];
  scale?: [x: number, y: number, z: number];
  coloring: (Omit<ProjectColoring, "targets"> & { targets: number[] }) | null;
  importScale?: ImportScale;
}

/** A section's serializable state, with meshes as mesh indices. */
export interface ProjectSectionInput {
  plane: { origin: [x: number, y: number, z: number]; normal: [x: number, y: number, z: number] };
  meshes: number[];
  fill: boolean;
}

/** Serializes meshes + scene state into a project doc (compact-JSON ready). */
export function buildProjectDoc(input: {
  meshes: ProjectMeshInput[];
  sections: ProjectSectionInput[];
  selection: number | null;
  wireframe: boolean;
  edgeSoftness: number;
  units?: LengthUnit;
}): ProjectDoc {
  return {
    version: PROJECT_VERSION,
    app: PROJECT_APP,
    meshes: input.meshes.map((m) => ({
      name: m.name,
      color: m.color,
      opacity: m.opacity,
      visible: m.visible,
      positions: encodeF64(m.positions),
      indices: encodeU32(m.indices),
      placement: {
        rotation: Array.from(m.rotation),
        translation: [...m.translation] as [number, number, number],
        ...(m.scale === undefined ? {} : { scale: [...m.scale] as [number, number, number] }),
      },
      coloring:
        m.coloring === null ? null : { ...m.coloring, targets: [...m.coloring.targets].sort((a, b) => a - b) },
      ...(m.importScale === undefined ? {} : { importScale: { ...m.importScale } }),
    })),
    sections: input.sections.map((s) => ({
      plane: { origin: [...s.plane.origin] as [number, number, number], normal: [...s.plane.normal] as [number, number, number] },
      meshes: [...s.meshes],
      fill: s.fill,
    })),
    selection: input.selection,
    wireframe: input.wireframe,
    edgeSoftness: input.edgeSoftness,
    ...(input.units === undefined ? {} : { units: input.units }),
  };
}

// ---- Parsing (pure, tolerant) ----

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && Object.getPrototypeOf(value) === Object.prototype;
}

function numTriple(value: unknown): [number, number, number] | null {
  if (!Array.isArray(value) || value.length !== 3) {
    return null;
  }
  const out = value.map(Number);
  if (!out.every((n) => Number.isFinite(n))) {
    return null;
  }
  return out as [number, number, number];
}

function readColoring(value: unknown, meshCount: number): ProjectColoring | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (!isRecord(value)) {
    return null;
  }
  const targets = Array.isArray(value.targets)
    ? value.targets.filter((t): t is number => Number.isInteger(t) && (t as number) >= 0 && (t as number) < meshCount)
    : [];
  if (targets.length === 0) {
    return null;
  }
  const maxDistance = value.maxDistance === null || value.maxDistance === undefined ? null : Number(value.maxDistance);
  if (!(maxDistance === null || (Number.isFinite(maxDistance) && (maxDistance as number) > 0))) {
    return null;
  }
  if (typeof value.map !== "string") {
    return null;
  }
  const range = value.range;
  if (
    !Array.isArray(range) ||
    range.length !== 2 ||
    !Number.isFinite(Number(range[0])) ||
    !Number.isFinite(Number(range[1])) ||
    !(Number(range[0]) < Number(range[1]))
  ) {
    return null;
  }
  return {
    targets,
    maxDistance,
    ...(value.sign === undefined ? {} : { sign: value.sign }),
    map: value.map,
    range: [Number(range[0]), Number(range[1])],
    ...(value.below === undefined ? {} : { below: value.below }),
    ...(value.above === undefined ? {} : { above: value.above }),
    ...(value.center === undefined ? {} : { center: value.center }),
    ...(value.base === undefined ? {} : { base: value.base }),
    ...(value.nan === undefined ? {} : { nan: value.nan }),
  };
}

/** A mesh's stored import scale; `undefined` for a missing or unusable one. An unknown unit reads as custom. */
function readImportScale(value: unknown): ImportScale | undefined {
  if (!isRecord(value) || typeof value.factor !== "number" || !Number.isFinite(value.factor) || value.factor <= 0) {
    return undefined;
  }
  return { factor: value.factor, units: isImportUnit(value.units) ? value.units : null };
}

/**
 * Parses project JSON into a doc, tolerantly: unknown fields are ignored, a
 * newer `version` still loads what is understood, and invalid entries fall
 * back to defaults (or are dropped, for colorings and sections).
 *
 * @throws when the text is not JSON, not an object, or holds no usable meshes.
 */
export function parseProjectDoc(text: string): ProjectDoc {
  let root: unknown;
  try {
    root = JSON.parse(text) as unknown;
  } catch {
    throw new Error("this is not a showcase project file (it is not JSON)");
  }
  if (!isRecord(root)) {
    throw new Error("this is not a showcase project file (it has no project object)");
  }
  const version = Number(root.version);
  if (!Number.isFinite(version)) {
    throw new Error("this is not a showcase project file (it has no version)");
  }
  if (!Array.isArray(root.meshes)) {
    throw new Error("this is not a showcase project file (it has no meshes)");
  }
  const rawMeshes = root.meshes as unknown[];
  const meshCount = rawMeshes.length;
  const meshes: ProjectMesh[] = [];
  for (let i = 0; i < rawMeshes.length; i++) {
    const raw = rawMeshes[i];
    if (!isRecord(raw)) {
      continue;
    }
    // Geometry is the one hard requirement: without it there is no mesh.
    if (typeof raw.positions !== "string" || typeof raw.indices !== "string") {
      continue;
    }
    let positions: Float64Array;
    let indices: Uint32Array;
    try {
      positions = decodeF64(raw.positions, `mesh ${i} positions`);
      indices = decodeU32(raw.indices, `mesh ${i} indices`);
    } catch {
      continue;
    }
    if (positions.length % 3 !== 0 || indices.length % 3 !== 0 || indices.length === 0) {
      continue;
    }
    const placement = isRecord(raw.placement) ? raw.placement : {};
    const rotation =
      Array.isArray(placement.rotation) && placement.rotation.length === 9 && placement.rotation.every((n) => Number.isFinite(Number(n)))
        ? placement.rotation.map(Number)
        : [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const translation = numTriple(placement.translation) ?? [0, 0, 0];
    const scaleRaw = numTriple(placement.scale);
    const scale = scaleRaw !== null && scaleRaw.every((n) => n > 0) ? (scaleRaw as [number, number, number]) : undefined;
    const importScale = readImportScale(raw.importScale);
    meshes.push({
      name: typeof raw.name === "string" && raw.name !== "" ? raw.name : `Mesh ${meshes.length + 1}`,
      color: Number.isFinite(Number(raw.color)) ? Number(raw.color) : 0x999999,
      opacity: typeof raw.opacity === "number" && raw.opacity >= 0 && raw.opacity <= 1 ? raw.opacity : 1,
      visible: raw.visible === undefined ? true : raw.visible === true,
      positions: encodeF64(positions),
      indices: encodeU32(indices),
      placement: {
        rotation,
        translation,
        ...(scale === undefined ? {} : { scale }),
      },
      coloring: readColoring(raw.coloring, meshCount),
      ...(importScale === undefined ? {} : { importScale }),
    });
  }
  if (meshes.length === 0) {
    throw new Error("this project file has no readable meshes");
  }
  // Targets were validated against the raw count; remap onto the kept meshes.
  // (Dropped meshes shift nothing: kept order matches raw order, and targets
  // past the kept end are filtered here.)
  const kept = meshes.length;
  for (const mesh of meshes) {
    if (mesh.coloring !== null) {
      const targets = mesh.coloring.targets.filter((t) => t < kept);
      mesh.coloring = targets.length === 0 ? null : { ...mesh.coloring, targets };
    }
  }
  const sections: ProjectSection[] = [];
  if (Array.isArray(root.sections)) {
    for (const raw of root.sections as unknown[]) {
      if (!isRecord(raw)) {
        continue;
      }
      const plane = isRecord(raw.plane) ? raw.plane : null;
      const origin = plane === null ? null : numTriple(plane.origin);
      const normal = plane === null ? null : numTriple(plane.normal);
      if (origin === null || normal === null) {
        continue;
      }
      const member = (m: unknown): m is number => Number.isInteger(m) && (m as number) >= 0 && (m as number) < kept;
      sections.push({
        plane: { origin, normal },
        meshes: Array.isArray(raw.meshes) ? (raw.meshes as unknown[]).filter(member) : [],
        fill: raw.fill === true,
      });
    }
  }
  const selection = Number.isInteger(root.selection) && (root.selection as number) >= 0 && (root.selection as number) < kept ? (root.selection as number) : null;
  return {
    version,
    app: typeof root.app === "string" ? root.app : "",
    meshes,
    sections,
    selection,
    wireframe: root.wireframe === true,
    edgeSoftness: typeof root.edgeSoftness === "number" && Number.isFinite(root.edgeSoftness) && root.edgeSoftness >= 0 ? root.edgeSoftness : 0,
    ...(root.units === "mm" || root.units === "in" ? { units: root.units } : {}),
  };
}

// ---- Store/app adapters ----

/** Collects the live session into a project doc. */
export function collectProject(store: SceneStore, app: AppState): ProjectDoc {
  const items = app.getSnapshot().items;
  const snap = store.getSnapshot();
  const indexOf = new Map<MeshHandle, number>(items.map((item, i) => [item.mesh, i]));
  return buildProjectDoc({
    meshes: items.map((item) => {
      const display = snap.meshes.get(item.mesh)?.desired ?? { coloring: null, opacity: 1, visible: true };
      const views = store.scene.views(item.mesh);
      const placement = store.scene.placement(item.mesh);
      const coloring = display.coloring;
      const targets =
        coloring === null
          ? []
          : (Array.isArray(coloring.distanceTo) ? coloring.distanceTo : [coloring.distanceTo])
              .map((m) => indexOf.get(m))
              .filter((t): t is number => t !== undefined);
      return {
        name: item.name,
        color: item.color,
        opacity: display.opacity,
        visible: display.visible,
        positions: views.positions(),
        indices: views.indices(),
        rotation: placement.rotation,
        translation: [placement.translation[0], placement.translation[1], placement.translation[2]],
        ...(placement.scale === undefined ? {} : { scale: [placement.scale[0], placement.scale[1], placement.scale[2]] as [number, number, number] }),
        coloring:
          coloring === null || targets.length === 0
            ? null
            : {
                targets,
                maxDistance: Number.isFinite(coloring.maxDistance) ? coloring.maxDistance : null,
                ...(coloring.sign === undefined ? {} : { sign: coloring.sign }),
                map: coloring.map,
                range: [...coloring.range] as [number, number],
                ...(coloring.below === undefined ? {} : { below: coloring.below }),
                ...(coloring.above === undefined ? {} : { above: coloring.above }),
                ...(coloring.center === undefined ? {} : { center: coloring.center }),
                ...(coloring.base === undefined ? {} : { base: coloring.base }),
                ...(coloring.nan === undefined ? {} : { nan: coloring.nan }),
              },
        ...(item.importScale.factor === 1 ? {} : { importScale: item.importScale }),
      };
    }),
    sections: [...snap.sections.values()].map((s) => ({
      plane: {
        origin: [s.desired.plane.origin[0], s.desired.plane.origin[1], s.desired.plane.origin[2]] as [number, number, number],
        normal: [s.desired.plane.normal[0], s.desired.plane.normal[1], s.desired.plane.normal[2]] as [number, number, number],
      },
      meshes: s.desired.meshes.map((m) => indexOf.get(m)).filter((t): t is number => t !== undefined),
      fill: s.desired.fill,
    })),
    selection: snap.selection === null ? null : (indexOf.get(snap.selection) ?? null),
    wireframe: app.getSnapshot().wireframe,
    edgeSoftness: app.getSnapshot().edgeSoftness,
    units: app.getSnapshot().units,
  });
}

/**
 * Reopens a project doc: replaces every mesh and section with the doc's.
 * Source volumes are not stored (tens of megabytes), so reopened meshes keep
 * their isosurface geometry but lose their volume link (`volumeId`).
 */
export function restoreProject(store: SceneStore, app: AppState, doc: ProjectDoc): void {
  for (const item of [...app.getSnapshot().items]) {
    app.removeMesh(store, item);
  }
  for (const section of [...store.getSnapshot().sections.keys()]) {
    store.removeSection(section);
  }
  store.select(null);
  const handles: MeshHandle[] = [];
  const colorings: ({ handle: MeshHandle; coloring: ProjectColoring } | null)[] = [];
  for (const mesh of doc.meshes) {
    const positions = decodeF64(mesh.positions, "positions");
    const indices = decodeU32(mesh.indices, "indices");
    const { center, radius } = bounds(positions);
    // The positions as stored, already in mm: `importScale` only rides along.
    const item = app.addMesh(store, {
      name: mesh.name,
      positions,
      indices,
      center,
      radius,
      volumeId: null,
      preset: null,
      importScale: mesh.importScale ?? NO_IMPORT_SCALE,
    });
    item.name = mesh.name;
    item.color = mesh.color;
    item.material.color.setHex(mesh.color);
    store.setPlacement(item.mesh, {
      rotation: [...mesh.placement.rotation],
      translation: [...mesh.placement.translation] as [number, number, number],
      ...(mesh.placement.scale === undefined ? {} : { scale: [...mesh.placement.scale] as [number, number, number] }),
    });
    store.setDisplay(item.mesh, { opacity: mesh.opacity, visible: mesh.visible });
    handles.push(item.mesh);
    colorings.push(mesh.coloring === null ? null : { handle: item.mesh, coloring: mesh.coloring });
  }
  for (const entry of colorings) {
    if (entry === null) {
      continue;
    }
    const { coloring } = entry;
    const distanceTo = coloring.targets.map((t) => handles[t]).filter((h): h is MeshHandle => h !== undefined);
    if (distanceTo.length === 0) {
      continue;
    }
    store.setDisplay(entry.handle, {
      coloring: {
        distanceTo: distanceTo.length === 1 ? distanceTo[0] : distanceTo,
        maxDistance: coloring.maxDistance ?? Infinity,
        ...(coloring.sign === undefined ? {} : { sign: coloring.sign as DistanceColoring["sign"] }),
        map: coloring.map as DistanceColoring["map"],
        range: [...(coloring.range as [number, number])] as [number, number],
        ...(coloring.below === undefined ? {} : { below: coloring.below as DistanceColoring["below"] }),
        ...(coloring.above === undefined ? {} : { above: coloring.above as DistanceColoring["above"] }),
        ...(coloring.center === undefined ? {} : { center: coloring.center as number }),
        ...(coloring.base === undefined ? {} : { base: coloring.base as DistanceColoring["base"] }),
        ...(coloring.nan === undefined ? {} : { nan: coloring.nan as DistanceColoring["nan"] }),
      },
    });
  }
  for (const section of doc.sections) {
    store.addSection({
      plane: { origin: [...section.plane.origin] as [number, number, number], normal: [...section.plane.normal] as [number, number, number] },
      meshes: section.meshes.map((i) => handles[i]).filter((h): h is MeshHandle => h !== undefined),
      fill: section.fill,
    });
  }
  const selected = doc.selection === null ? null : (handles[doc.selection] ?? null);
  store.select(selected);
  app.setMulti(selected === null ? [] : [selected]);
  app.setWireframe(doc.wireframe);
  app.setEdgeSoftness(doc.edgeSoftness);
  app.setProjectUnits(doc.units);
}

// ---- Files: File System Access with a download/input fallback ----

/** The suggested file name for a saved project. */
export function projectFileName(when = new Date()): string {
  const stamp = `${when.getFullYear()}${String(when.getMonth() + 1).padStart(2, "0")}${String(when.getDate()).padStart(2, "0")}-${String(when.getHours()).padStart(2, "0")}${String(when.getMinutes()).padStart(2, "0")}`;
  return `showcase-project-${stamp}${PROJECT_EXTENSION}`;
}

interface SavePicker {
  showSaveFilePicker?: (options: unknown) => Promise<{
    createWritable: () => Promise<{ write(data: Uint8Array): Promise<void>; close(): Promise<void> }>;
  }>;
  showOpenFilePicker?: (options: unknown) => Promise<{ getFile: () => Promise<File> }[]>;
}

function picker(): SavePicker | null {
  if (typeof window === "undefined") {
    return null;
  }
  const win = window as unknown as SavePicker;
  return typeof win.showSaveFilePicker === "function" || typeof win.showOpenFilePicker === "function" ? win : null;
}

/** Triggers a download of `text` (the fallback when no save picker exists). */
export function downloadText(text: string, fileName: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Saves the project JSON: the File System Access save picker where
 * available, a download otherwise. Resolves `"cancelled"` when the user
 * dismisses the picker. @throws when the write itself fails.
 */
export async function saveProjectFile(text: string, fileName: string): Promise<"picker" | "download" | "cancelled"> {
  const via = picker();
  if (via?.showSaveFilePicker !== undefined) {
    let handle;
    try {
      handle = await via.showSaveFilePicker({
        suggestedName: fileName,
        types: [{ description: "Showcase project", accept: { [PROJECT_MIME]: [PROJECT_EXTENSION, ".json"] } }],
      });
    } catch (error) {
      if ((error as { name?: string } | null)?.name === "AbortError") {
        return "cancelled";
      }
      throw error;
    }
    const writable = await handle.createWritable();
    try {
      await writable.write(new TextEncoder().encode(text));
    } finally {
      await writable.close();
    }
    return "picker";
  }
  downloadText(text, fileName, PROJECT_MIME);
  return "download";
}

/**
 * Opens a project file: the File System Access open picker where available,
 * a hidden file input otherwise. Resolves `null` when the user cancels.
 */
export function openProjectFile(): Promise<File | null> {
  const showOpen = picker()?.showOpenFilePicker;
  if (showOpen !== undefined) {
    return (async () => {
      try {
        const [handle] = await showOpen({
          multiple: false,
          types: [{ description: "Showcase project", accept: { [PROJECT_MIME]: [PROJECT_EXTENSION, ".json"] } }],
        });
        return handle === undefined ? null : await handle.getFile();
      } catch (error) {
        if ((error as { name?: string } | null)?.name === "AbortError") {
          return null;
        }
        throw error;
      }
    })();
  }
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = [PROJECT_EXTENSION, ".json"].join(",");
    input.hidden = true;
    const done = (file: File | null): void => {
      input.remove();
      resolve(file);
    };
    input.onchange = () => done(input.files?.[0] ?? null);
    input.oncancel = () => done(null);
    document.body.appendChild(input);
    input.click();
  });
}
