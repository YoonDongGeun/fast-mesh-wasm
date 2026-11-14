/**
 * 3MF (3D Manufacturing Format) OPC importer and exporter.
 *
 * Implements OPC archive resolution (`_rels/.rels`, `3D/3dmodel.model`),
 * Production Extension external `.model` references (`p:path`), unit
 * normalization to millimeters, recursive 3x4 affine transforms with mirrored
 * winding correction, `<basematerials>` / `<m:colorgroup>` RGBA8 colors, and
 * strict XXE / `<!DOCTYPE` / `<!ENTITY` guards.
 */
import type { MeshHandle } from "../handles.js";
import type { SceneLike } from "../scene-like.js";
import type { ExportOptions, ImportedMesh } from "./index.js";
import { readZip, writeZip, writeZipSync } from "./zip.js";

/** Supported 3MF `<model unit="...">` values. */
export type ThreeMfUnit =
  | "micron"
  | "millimeter"
  | "centimeter"
  | "inch"
  | "foot"
  | "meter";

const UNIT_TO_MM: Record<ThreeMfUnit, number> = {
  micron: 0.001,
  millimeter: 1,
  centimeter: 10,
  inch: 25.4,
  foot: 304.8,
  meter: 1000,
};

/** Options for {@link import3MF}. */
export interface ThreeMfImportOptions {
  /** Maximum accepted ZIP archive / uncompressed payload size in bytes. Default: 512 MiB. */
  maxFileBytes?: number;
  /** Override millimeter scale factor instead of reading `<model unit="...">`. */
  unitScale?: number;
  /** Optional abort signal. */
  signal?: AbortSignal;
}

/** Options for {@link export3MF} and {@link export3MFSync}. */
export interface ThreeMfExportOptions extends ExportOptions {
  /** Model coordinate unit stored in `<model unit="...">`. Defaults to `"millimeter"`. */
  unit?: ThreeMfUnit;
  /** Write per-vertex/per-triangle RGBA colors when present on the mesh. Defaults to `true`. */
  vertexColors?: boolean;
  /** Deflate-compress ZIP entries in async {@link export3MF}. Defaults to `true`. */
  compress?: boolean;
  /** Object name (or prefix when exporting multiple meshes). Defaults to `"mesh"`. */
  name?: string;
}

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8");

function toUint8(bytes: Uint8Array | ArrayBuffer): Uint8Array {
  return bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    const reason = signal.reason;
    if (reason instanceof Error) {
      throw reason;
    }
    throw new DOMException("The operation was aborted.", "AbortError");
  }
}

function decodeXmlUtf8(data: Uint8Array, label: string): string {
  let xml = textDecoder.decode(data);
  if (xml.charCodeAt(0) === 0xfeff) {
    xml = xml.slice(1);
  }
  if (/<!(?:DOCTYPE|ENTITY)\b/i.test(xml)) {
    throw new Error(
      `3MF XML "${label}" contains forbidden DOCTYPE or ENTITY declaration`,
    );
  }
  return xml;
}

function xmlUnescape(text: string): string {
  if (!text.includes("&")) return text;
  return text.replace(
    /&(#x[0-9a-fA-F]+|#[0-9]+|lt|gt|amp|quot|apos);/g,
    (match, body: string) => {
      switch (body) {
        case "lt":
          return "<";
        case "gt":
          return ">";
        case "amp":
          return "&";
        case "quot":
          return '"';
        case "apos":
          return "'";
        default: {
          const code =
            body.charCodeAt(1) === 120 || body.charCodeAt(1) === 88
              ? Number.parseInt(body.slice(2), 16)
              : Number.parseInt(body.slice(1), 10);
          return Number.isFinite(code) && code >= 0 && code <= 0x10ffff
            ? String.fromCodePoint(code)
            : match;
        }
      }
    },
  );
}

function xmlEscapeAttr(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function localTagName(qName: string): string {
  const idx = qName.indexOf(":");
  return (idx >= 0 ? qName.slice(idx + 1) : qName).toLowerCase();
}

/**
 * Scan attributes from a tag's raw attribute slice without allocating per-element
 * objects unless needed. Matches by exact attribute name OR suffix after `:`
 * when `matchLocal` is needed via a callback.
 */
function parseTagAttrs(
  attrText: string,
  cb: (rawName: string, value: string) => void,
): void {
  const len = attrText.length;
  let i = 0;
  while (i < len) {
    while (i < len && attrText.charCodeAt(i) <= 32) i++;
    if (i >= len) break;
    const nameStart = i;
    while (
      i < len &&
      attrText.charCodeAt(i) > 32 &&
      attrText[i] !== "=" &&
      attrText[i] !== "/"
    ) {
      i++;
    }
    const rawName = attrText.slice(nameStart, i);
    while (i < len && attrText.charCodeAt(i) <= 32) i++;
    if (i >= len || attrText[i] !== "=") {
      continue;
    }
    i++; // '='
    while (i < len && attrText.charCodeAt(i) <= 32) i++;
    if (i >= len) break;
    const quote = attrText[i];
    if (quote === '"' || quote === "'") {
      i++;
      const valStart = i;
      const close = attrText.indexOf(quote, valStart);
      if (close < 0) {
        cb(rawName, xmlUnescape(attrText.slice(valStart)));
        break;
      }
      cb(rawName, xmlUnescape(attrText.slice(valStart, close)));
      i = close + 1;
    } else {
      const valStart = i;
      while (i < len && attrText.charCodeAt(i) > 32) i++;
      cb(rawName, xmlUnescape(attrText.slice(valStart, i)));
    }
  }
}

interface XmlScannerCallbacks {
  onOpen: (localName: string, attrText: string, selfClosing: boolean) => void;
  onClose: (localName: string) => void;
}

/**
 * Streaming, non-backtracking XML tag scanner tailored for 3MF `.model` and `.rels` files.
 */
function scanXml(xml: string, cb: XmlScannerCallbacks, signal?: AbortSignal): void {
  const len = xml.length;
  let pos = 0;
  let iter = 0;

  while (pos < len) {
    if ((iter++ & 0x3fff) === 0) {
      throwIfAborted(signal);
    }
    const lt = xml.indexOf("<", pos);
    if (lt < 0) break;

    const next = xml.charCodeAt(lt + 1);
    if (next === 63 /* ? */) {
      const endPi = xml.indexOf("?>", lt + 2);
      pos = endPi < 0 ? len : endPi + 2;
      continue;
    }
    if (next === 33 /* ! */) {
      if (xml.startsWith("<!--", lt)) {
        const endComment = xml.indexOf("-->", lt + 4);
        if (endComment < 0) {
          throw new Error("Invalid 3MF XML: unclosed comment");
        }
        pos = endComment + 3;
        continue;
      }
      if (xml.startsWith("<![CDATA[", lt)) {
        const endCdata = xml.indexOf("]]>", lt + 9);
        if (endCdata < 0) {
          throw new Error("Invalid 3MF XML: unclosed CDATA section");
        }
        pos = endCdata + 3;
        continue;
      }
      throw new Error("Invalid 3MF XML: unsupported declaration");
    }

    // Find tag closing '>' while ignoring '>' inside quotes
    let i = lt + 1;
    let quoteChar = 0;
    let tagEnd = -1;
    while (i < len) {
      const ch = xml.charCodeAt(i);
      if (quoteChar !== 0) {
        if (ch === quoteChar) quoteChar = 0;
      } else {
        if (ch === 34 /* " */ || ch === 39 /* ' */) {
          quoteChar = ch;
        } else if (ch === 62 /* > */) {
          tagEnd = i;
          break;
        }
      }
      i++;
    }
    if (tagEnd < 0) {
      throw new Error("Invalid 3MF XML: unclosed tag");
    }

    if (next === 47 /* / */) {
      const rawClose = xml.slice(lt + 2, tagEnd).trim();
      cb.onClose(localTagName(rawClose));
      pos = tagEnd + 1;
      continue;
    }

    let selfClosing = false;
    let contentEnd = tagEnd;
    if (tagEnd > lt + 1 && xml.charCodeAt(tagEnd - 1) === 47 /* / */) {
      selfClosing = true;
      contentEnd = tagEnd - 1;
    }

    let nameEnd = lt + 1;
    while (nameEnd < contentEnd && xml.charCodeAt(nameEnd) > 32) {
      nameEnd++;
    }
    const qName = xml.slice(lt + 1, nameEnd);
    const attrText = xml.slice(nameEnd, contentEnd);
    const lname = localTagName(qName);
    cb.onOpen(lname, attrText, selfClosing);
    if (selfClosing) {
      cb.onClose(lname);
    }
    pos = tagEnd + 1;
  }
}

function normalizeOpcPath(baseFile: string, target: string): string {
  const cleaned = target.replaceAll("\\", "/");
  const isRooted = cleaned.startsWith("/");
  const baseParts = isRooted
    ? []
    : baseFile.split("/").slice(0, -1);
  const relParts = (isRooted ? cleaned.slice(1) : cleaned).split("/");
  const out: string[] = [...baseParts];
  for (let i = 0; i < relParts.length; i++) {
    const seg = relParts[i];
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      if (out.length === 0) {
        throw new Error(`Invalid 3MF OPC path "${target}" escapes archive root`);
      }
      out.pop();
      continue;
    }
    out.push(seg);
  }
  return out.join("/");
}

function findZipEntryCaseFallback(
  entries: ReadonlyMap<string, Uint8Array>,
  path: string,
): { path: string; data: Uint8Array } | null {
  const direct = entries.get(path);
  if (direct !== undefined) {
    return { path, data: direct };
  }
  const lower = path.toLowerCase();
  for (const [k, v] of entries) {
    if (k.toLowerCase() === lower) {
      return { path: k, data: v };
    }
  }
  return null;
}

function parseHexColor(hexRaw: string): number {
  const s = hexRaw.trim();
  if (!s.startsWith("#")) {
    throw new Error(`Invalid 3MF color "${hexRaw}": expected #RRGGBB or #RRGGBBAA`);
  }
  const h = s.slice(1);
  let r: number;
  let g: number;
  let b: number;
  let a = 255;
  if (h.length === 6) {
    const val = Number.parseInt(h, 16);
    if (!Number.isFinite(val)) {
      throw new Error(`Invalid 3MF color "${hexRaw}"`);
    }
    r = (val >>> 16) & 0xff;
    g = (val >>> 8) & 0xff;
    b = val & 0xff;
  } else if (h.length === 8) {
    const val = Number.parseInt(h, 16);
    if (!Number.isFinite(val)) {
      throw new Error(`Invalid 3MF color "${hexRaw}"`);
    }
    r = (val >>> 24) & 0xff;
    g = (val >>> 16) & 0xff;
    b = (val >>> 8) & 0xff;
    a = val & 0xff;
  } else if (h.length === 3 || h.length === 4) {
    const rN = Number.parseInt(h[0], 16);
    const gN = Number.parseInt(h[1], 16);
    const bN = Number.parseInt(h[2], 16);
    const aN = h.length === 4 ? Number.parseInt(h[3], 16) : 15;
    if (
      !Number.isFinite(rN) ||
      !Number.isFinite(gN) ||
      !Number.isFinite(bN) ||
      !Number.isFinite(aN)
    ) {
      throw new Error(`Invalid 3MF color "${hexRaw}"`);
    }
    r = rN * 17;
    g = gN * 17;
    b = bN * 17;
    a = aN * 17;
  } else {
    throw new Error(`Invalid 3MF color "${hexRaw}": unsupported hex length`);
  }
  return (((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff)) >>> 0;
}

/** 3x4 row-vector affine matrix: [m00 m01 m02  m10 m11 m12  m20 m21 m22  m30 m31 m32]. */
type RowAffine3x4 = Float64Array;

const IDENTITY_3X4 = new Float64Array([
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
  0, 0, 0,
]);

function parseTransform3x4(raw: string | undefined): RowAffine3x4 {
  if (raw === undefined || raw.trim() === "") {
    return IDENTITY_3X4;
  }
  const parts = raw.trim().split(/\s+/);
  if (parts.length !== 12) {
    throw new Error(
      `Invalid 3MF transform "${raw}": expected 12 numbers, got ${parts.length}`,
    );
  }
  const out = new Float64Array(12);
  for (let i = 0; i < 12; i++) {
    const n = Number(parts[i]);
    if (!Number.isFinite(n)) {
      throw new Error(`Invalid 3MF transform: non-finite number "${parts[i]}"`);
    }
    out[i] = n;
  }
  return out;
}

/** Compose affine matrices `child` then `parent`: `(v * child) * parent`. */
function composeRowAffine(child: RowAffine3x4, parent: RowAffine3x4): RowAffine3x4 {
  const r = new Float64Array(12);
  for (let row = 0; row < 3; row++) {
    const a0 = child[row * 3];
    const a1 = child[row * 3 + 1];
    const a2 = child[row * 3 + 2];
    for (let col = 0; col < 3; col++) {
      r[row * 3 + col] = a0 * parent[col] + a1 * parent[3 + col] + a2 * parent[6 + col];
    }
  }
  const t0 = child[9];
  const t1 = child[10];
  const t2 = child[11];
  for (let col = 0; col < 3; col++) {
    r[9 + col] =
      t0 * parent[col] + t1 * parent[3 + col] + t2 * parent[6 + col] + parent[9 + col];
  }
  return r;
}

function det3x3(m: RowAffine3x4): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

interface ParsedMeshData {
  vertices: number[]; // flat x,y,z
  triangles: number[]; // flat v1,v2,v3
  triPids: (string | undefined)[];
  triP1: Int32Array;
  triP2: Int32Array;
  triP3: Int32Array;
}

interface ParsedComponentRef {
  objectId: string;
  transform: RowAffine3x4;
  path?: string;
}

interface ParsedObject {
  id: string;
  name: string;
  defaultPid?: string;
  defaultPIndex?: number;
  mesh?: ParsedMeshData;
  components?: ParsedComponentRef[];
}

interface ParsedBuildItem {
  objectId: string;
  transform: RowAffine3x4;
  path?: string;
}

interface ParsedModelFile {
  path: string;
  unit?: ThreeMfUnit;
  materials: Map<string, number[]>; // groupId -> array of packed RGBA u32
  objects: Map<string, ParsedObject>;
  buildItems: ParsedBuildItem[];
}

function parseIntStrict(raw: string, label: string): number {
  const n = Number(raw);
  if (!Number.isInteger(n)) {
    throw new Error(`Invalid 3MF integer for ${label}: "${raw}"`);
  }
  return n;
}

function parseFloatFinite(raw: string, label: string): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) {
    throw new Error(`Invalid 3MF number for ${label}: "${raw}"`);
  }
  return n;
}

function parseModelXml(
  modelPath: string,
  xml: string,
  signal?: AbortSignal,
): ParsedModelFile {
  const materials = new Map<string, number[]>();
  const objects = new Map<string, ParsedObject>();
  const buildItems: ParsedBuildItem[] = [];
  let unit: ThreeMfUnit | undefined;

  let activeMaterialGroupId: string | null = null;
  let activeMaterialColors: number[] | null = null;

  let currentObject: ParsedObject | null = null;
  let inResources = false;
  let inBuild = false;
  let inMesh = false;
  let inVertices = false;
  let inTriangles = false;
  let inComponents = false;

  // Dynamic arrays for triangles so we do not allocate objects per triangle
  let triPids: (string | undefined)[] = [];
  let triP1List: number[] = [];
  let triP2List: number[] = [];
  let triP3List: number[] = [];

  scanXml(
    xml,
    {
      onOpen(tag, attrText) {
        if (tag === "model") {
          parseTagAttrs(attrText, (name, value) => {
            if (name === "unit") {
              const lower = value.trim().toLowerCase();
              if (lower in UNIT_TO_MM) {
                unit = lower as ThreeMfUnit;
              }
            }
          });
          return;
        }
        if (tag === "resources") {
          inResources = true;
          return;
        }
        if (tag === "build") {
          inBuild = true;
          return;
        }
        if (inResources && (tag === "basematerials" || tag === "colorgroup")) {
          let id = "";
          parseTagAttrs(attrText, (name, value) => {
            if (name === "id") id = value;
          });
          if (id === "") {
            throw new Error(`Invalid 3MF <${tag}> missing id attribute`);
          }
          activeMaterialGroupId = id;
          activeMaterialColors = [];
          materials.set(id, activeMaterialColors);
          return;
        }
        if (activeMaterialColors !== null && (tag === "base" || tag === "color")) {
          let hex: string | undefined;
          parseTagAttrs(attrText, (name, value) => {
            if (name === "displaycolor" || name === "color") {
              hex = value;
            }
          });
          if (hex !== undefined) {
            activeMaterialColors.push(parseHexColor(hex));
          } else {
            activeMaterialColors.push(0xffffffff);
          }
          return;
        }
        if (inResources && tag === "object") {
          let id = "";
          let name = "";
          let pid: string | undefined;
          let pindex: number | undefined;
          parseTagAttrs(attrText, (attrName, value) => {
            if (attrName === "id") id = value;
            else if (attrName === "name" || attrName === "partnumber") {
              if (name === "" || attrName === "name") name = value;
            } else if (attrName === "pid") pid = value;
            else if (attrName === "pindex") {
              pindex = parseIntStrict(value, "object pindex");
            }
          });
          if (id === "") {
            throw new Error("Invalid 3MF <object> missing id attribute");
          }
          currentObject = {
            id,
            name: name || `object-${id}`,
            defaultPid: pid,
            defaultPIndex: pindex,
          };
          objects.set(id, currentObject);
          return;
        }
        if (currentObject !== null && tag === "mesh") {
          inMesh = true;
          triPids = [];
          triP1List = [];
          triP2List = [];
          triP3List = [];
          currentObject.mesh = {
            vertices: [],
            triangles: [],
            triPids,
            triP1: new Int32Array(0),
            triP2: new Int32Array(0),
            triP3: new Int32Array(0),
          };
          return;
        }
        if (inMesh && currentObject?.mesh && tag === "vertices") {
          inVertices = true;
          return;
        }
        if (inMesh && currentObject?.mesh && tag === "triangles") {
          inTriangles = true;
          return;
        }
        if (inVertices && currentObject?.mesh && tag === "vertex") {
          let x: number | undefined;
          let y: number | undefined;
          let z: number | undefined;
          parseTagAttrs(attrText, (name, value) => {
            if (name === "x") x = parseFloatFinite(value, "vertex.x");
            else if (name === "y") y = parseFloatFinite(value, "vertex.y");
            else if (name === "z") z = parseFloatFinite(value, "vertex.z");
          });
          if (x === undefined || y === undefined || z === undefined) {
            throw new Error("Invalid 3MF <vertex> missing x, y, or z attribute");
          }
          currentObject.mesh.vertices.push(x, y, z);
          return;
        }
        if (inTriangles && currentObject?.mesh && tag === "triangle") {
          let v1 = -1;
          let v2 = -1;
          let v3 = -1;
          let pid: string | undefined;
          let p1 = -1;
          let p2 = -1;
          let p3 = -1;
          parseTagAttrs(attrText, (name, value) => {
            if (name === "v1") v1 = parseIntStrict(value, "triangle.v1");
            else if (name === "v2") v2 = parseIntStrict(value, "triangle.v2");
            else if (name === "v3") v3 = parseIntStrict(value, "triangle.v3");
            else if (name === "pid") pid = value;
            else if (name === "p1") p1 = parseIntStrict(value, "triangle.p1");
            else if (name === "p2") p2 = parseIntStrict(value, "triangle.p2");
            else if (name === "p3") p3 = parseIntStrict(value, "triangle.p3");
          });
          if (v1 < 0 || v2 < 0 || v3 < 0) {
            throw new Error(
              `Invalid 3MF <triangle> indices (${v1}, ${v2}, ${v3})`,
            );
          }
          currentObject.mesh.triangles.push(v1, v2, v3);
          triPids.push(pid);
          triP1List.push(p1);
          triP2List.push(p2 >= 0 ? p2 : p1);
          triP3List.push(p3 >= 0 ? p3 : p1);
          return;
        }
        if (currentObject !== null && tag === "components") {
          inComponents = true;
          if (!currentObject.components) {
            currentObject.components = [];
          }
          return;
        }
        if (inComponents && currentObject?.components && tag === "component") {
          let objectId = "";
          let transformRaw: string | undefined;
          let path: string | undefined;
          parseTagAttrs(attrText, (name, value) => {
            const lname = localTagName(name);
            if (lname === "objectid") objectId = value;
            else if (lname === "transform") transformRaw = value;
            else if (lname === "path") path = value;
          });
          if (objectId === "") {
            throw new Error("Invalid 3MF <component> missing objectid");
          }
          currentObject.components.push({
            objectId,
            transform: parseTransform3x4(transformRaw),
            path,
          });
          return;
        }
        if (inBuild && tag === "item") {
          let objectId = "";
          let transformRaw: string | undefined;
          let path: string | undefined;
          parseTagAttrs(attrText, (name, value) => {
            const lname = localTagName(name);
            if (lname === "objectid") objectId = value;
            else if (lname === "transform") transformRaw = value;
            else if (lname === "path") path = value;
          });
          if (objectId === "") {
            throw new Error("Invalid 3MF <item> missing objectid");
          }
          buildItems.push({
            objectId,
            transform: parseTransform3x4(transformRaw),
            path,
          });
        }
      },
      onClose(tag) {
        if (tag === "resources") {
          inResources = false;
        } else if (tag === "build") {
          inBuild = false;
        } else if (tag === "basematerials" || tag === "colorgroup") {
          if (activeMaterialGroupId !== null) {
            activeMaterialGroupId = null;
            activeMaterialColors = null;
          }
        } else if (tag === "object") {
          currentObject = null;
          inMesh = false;
          inVertices = false;
          inTriangles = false;
          inComponents = false;
        } else if (tag === "mesh") {
          if (currentObject?.mesh) {
            currentObject.mesh.triP1 = Int32Array.from(triP1List);
            currentObject.mesh.triP2 = Int32Array.from(triP2List);
            currentObject.mesh.triP3 = Int32Array.from(triP3List);
          }
          inMesh = false;
          inVertices = false;
          inTriangles = false;
        } else if (tag === "vertices") {
          inVertices = false;
        } else if (tag === "triangles") {
          inTriangles = false;
        } else if (tag === "components") {
          inComponents = false;
        }
      },
    },
    signal,
  );

  // Validate triangle vertex index ranges eagerly
  for (const obj of objects.values()) {
    const mesh = obj.mesh;
    if (!mesh) continue;
    const vertCount = mesh.vertices.length / 3;
    for (let i = 0; i < mesh.triangles.length; i++) {
      const idx = mesh.triangles[i];
      if (idx < 0 || idx >= vertCount) {
        throw new Error(
          `Invalid 3MF mesh "${obj.name}": triangle vertex index ${idx} out of range (0..${vertCount - 1})`,
        );
      }
    }
  }

  return { path: modelPath, unit, materials, objects, buildItems };
}

function findRootModelPath(entries: ReadonlyMap<string, Uint8Array>): string {
  const relsEntry = findZipEntryCaseFallback(entries, "_rels/.rels");
  if (relsEntry) {
    const xml = decodeXmlUtf8(relsEntry.data, relsEntry.path);
    let targetPath: string | null = null;
    scanXml(xml, {
      onOpen(tag, attrText) {
        if (tag !== "relationship" || targetPath !== null) return;
        let relType = "";
        let target = "";
        parseTagAttrs(attrText, (name, value) => {
          const lname = localTagName(name);
          if (lname === "type") relType = value.trim().toLowerCase();
          else if (lname === "target") target = value.trim();
        });
        if (relType.endsWith("/3dmodel") && target !== "") {
          targetPath = normalizeOpcPath(relsEntry.path, target);
        }
      },
      onClose() {
        // no-op
      },
    });
    if (targetPath !== null) {
      return targetPath;
    }
  }
  return "3D/3dmodel.model";
}

function instantiateMeshObject(
  model: ParsedModelFile,
  obj: ParsedObject,
  mesh: ParsedMeshData,
  transform: RowAffine3x4,
  scaleToMm: number,
  name: string,
): ImportedMesh {
  const origVertCount = mesh.vertices.length / 3;
  const triCount = mesh.triangles.length / 3;
  const flipWinding = det3x3(transform) < 0;

  // Determine per-triangle corner packed RGBA colors if any material references exist
  const cornerColors = new Uint32Array(triCount * 3);
  const cornerHasColor = new Uint8Array(triCount * 3);
  let anyColor = false;

  for (let t = 0; t < triCount; t++) {
    const pid = mesh.triPids[t] ?? obj.defaultPid;
    if (pid === undefined) continue;
    const group = model.materials.get(pid);
    if (!group || group.length === 0) continue;

    const p1 = mesh.triP1[t] >= 0 ? mesh.triP1[t] : obj.defaultPIndex ?? -1;
    const p2 = mesh.triP2[t] >= 0 ? mesh.triP2[t] : p1;
    const p3 = mesh.triP3[t] >= 0 ? mesh.triP3[t] : p1;

    const cIndices = flipWinding ? [p1, p3, p2] : [p1, p2, p3];
    for (let k = 0; k < 3; k++) {
      const pi = cIndices[k];
      if (pi >= 0 && pi < group.length) {
        cornerColors[t * 3 + k] = group[pi];
        cornerHasColor[t * 3 + k] = 1;
        anyColor = true;
      }
    }
  }

  let outPositions: Float64Array;
  let outIndices: Uint32Array;
  let outColors: Uint8Array | null = null;

  if (!anyColor) {
    outPositions = new Float64Array(origVertCount * 3);
    for (let v = 0; v < origVertCount; v++) {
      const x = mesh.vertices[v * 3];
      const y = mesh.vertices[v * 3 + 1];
      const z = mesh.vertices[v * 3 + 2];
      outPositions[v * 3] =
        (transform[0] * x + transform[3] * y + transform[6] * z + transform[9]) * scaleToMm;
      outPositions[v * 3 + 1] =
        (transform[1] * x + transform[4] * y + transform[7] * z + transform[10]) * scaleToMm;
      outPositions[v * 3 + 2] =
        (transform[2] * x + transform[5] * y + transform[8] * z + transform[11]) * scaleToMm;
    }
    outIndices = new Uint32Array(triCount * 3);
    for (let t = 0; t < triCount; t++) {
      const a = mesh.triangles[t * 3];
      const b = mesh.triangles[t * 3 + 1];
      const c = mesh.triangles[t * 3 + 2];
      if (flipWinding) {
        outIndices[t * 3] = a;
        outIndices[t * 3 + 1] = c;
        outIndices[t * 3 + 2] = b;
      } else {
        outIndices[t * 3] = a;
        outIndices[t * 3 + 1] = b;
        outIndices[t * 3 + 2] = c;
      }
    }
  } else {
    // Only clone a vertex when adjacent corners assign different colors.
    const vertAssigned = new Uint8Array(origVertCount);
    const vertColor = new Uint32Array(origVertCount);
    const cloneLookup = new Map<number, Map<number, number>>();
    const extraSrcVert: number[] = [];
    const extraColors: number[] = [];

    outIndices = new Uint32Array(triCount * 3);
    for (let t = 0; t < triCount; t++) {
      const a = mesh.triangles[t * 3];
      const b = mesh.triangles[t * 3 + 1];
      const c = mesh.triangles[t * 3 + 2];
      const triVerts = flipWinding ? [a, c, b] : [a, b, c];

      for (let k = 0; k < 3; k++) {
        const origV = triVerts[k];
        const col = cornerHasColor[t * 3 + k] !== 0 ? cornerColors[t * 3 + k] : 0xffffffff;

        if (vertAssigned[origV] === 0) {
          vertAssigned[origV] = 1;
          vertColor[origV] = col;
          outIndices[t * 3 + k] = origV;
        } else if (vertColor[origV] === col) {
          outIndices[t * 3 + k] = origV;
        } else {
          let mapForV = cloneLookup.get(origV);
          if (!mapForV) {
            mapForV = new Map<number, number>();
            cloneLookup.set(origV, mapForV);
          }
          let clonedIdx = mapForV.get(col);
          if (clonedIdx === undefined) {
            clonedIdx = origVertCount + extraSrcVert.length;
            extraSrcVert.push(origV);
            extraColors.push(col);
            mapForV.set(col, clonedIdx);
          }
          outIndices[t * 3 + k] = clonedIdx;
        }
      }
    }

    const totalVerts = origVertCount + extraSrcVert.length;
    outPositions = new Float64Array(totalVerts * 3);
    outColors = new Uint8Array(totalVerts * 4);

    const writeTransformedVert = (dstIdx: number, srcOrigIdx: number, rgba: number) => {
      const x = mesh.vertices[srcOrigIdx * 3];
      const y = mesh.vertices[srcOrigIdx * 3 + 1];
      const z = mesh.vertices[srcOrigIdx * 3 + 2];
      outPositions[dstIdx * 3] =
        (transform[0] * x + transform[3] * y + transform[6] * z + transform[9]) * scaleToMm;
      outPositions[dstIdx * 3 + 1] =
        (transform[1] * x + transform[4] * y + transform[7] * z + transform[10]) * scaleToMm;
      outPositions[dstIdx * 3 + 2] =
        (transform[2] * x + transform[5] * y + transform[8] * z + transform[11]) * scaleToMm;
      outColors![dstIdx * 4] = (rgba >>> 24) & 0xff;
      outColors![dstIdx * 4 + 1] = (rgba >>> 16) & 0xff;
      outColors![dstIdx * 4 + 2] = (rgba >>> 8) & 0xff;
      outColors![dstIdx * 4 + 3] = rgba & 0xff;
    };

    for (let v = 0; v < origVertCount; v++) {
      const rgba = vertAssigned[v] !== 0 ? vertColor[v] : 0xffffffff;
      writeTransformedVert(v, v, rgba);
    }
    for (let e = 0; e < extraSrcVert.length; e++) {
      writeTransformedVert(origVertCount + e, extraSrcVert[e], extraColors[e]);
    }
  }

  return {
    name,
    positions: outPositions,
    indices: outIndices,
    placement: new Float64Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]),
    colors: outColors,
    uvs: null,
  };
}

/**
 * Parse a 3MF archive into {@link ImportedMesh} records scaled to millimeters.
 */
export async function import3MF(
  bytes: Uint8Array | ArrayBuffer,
  options?: ThreeMfImportOptions,
): Promise<ImportedMesh[]> {
  throwIfAborted(options?.signal);
  const archive = await readZip(toUint8(bytes), {
    maxFileBytes: options?.maxFileBytes,
    signal: options?.signal,
  });

  // Check Content_Types XML too so hostile DOCTYPE/ENTITY declarations cannot slip past.
  const contentTypes = findZipEntryCaseFallback(archive, "[Content_Types].xml");
  if (contentTypes) {
    decodeXmlUtf8(contentTypes.data, contentTypes.path);
  }

  const rootPath = findRootModelPath(archive);
  const modelCache = new Map<string, ParsedModelFile>();

  const loadModel = (modelPath: string): ParsedModelFile => {
    const cached = modelCache.get(modelPath);
    if (cached) return cached;
    const found = findZipEntryCaseFallback(archive, modelPath);
    if (!found) {
      throw new Error(`3MF archive is missing model file "${modelPath}"`);
    }
    const xml = decodeXmlUtf8(found.data, found.path);
    const parsed = parseModelXml(found.path, xml, options?.signal);
    modelCache.set(modelPath, parsed);
    return parsed;
  };

  const rootModel = loadModel(rootPath);
  const rootUnitScale = UNIT_TO_MM[rootModel.unit ?? "millimeter"];

  const results: ImportedMesh[] = [];
  const stackKeys = new Set<string>();

  const visitObject = (
    model: ParsedModelFile,
    objectId: string,
    worldTransform: RowAffine3x4,
    inheritedScale: number,
  ): void => {
    throwIfAborted(options?.signal);
    if (stackKeys.size >= 256) {
      throw new Error("3MF component hierarchy exceeds maximum recursion depth (256)");
    }
    const key = `${model.path}#${objectId}`;
    if (stackKeys.has(key)) {
      throw new Error(`3MF cyclic component reference detected at "${key}"`);
    }
    const obj = model.objects.get(objectId);
    if (!obj) {
      throw new Error(
        `3MF references unknown object id "${objectId}" in "${model.path}"`,
      );
    }
    const modelScale =
      options?.unitScale ??
      (model.unit !== undefined ? UNIT_TO_MM[model.unit] : inheritedScale);

    stackKeys.add(key);
    try {
      if (obj.mesh) {
        results.push(
          instantiateMeshObject(
            model,
            obj,
            obj.mesh,
            worldTransform,
            modelScale,
            obj.name,
          ),
        );
      }
      if (obj.components) {
        for (let i = 0; i < obj.components.length; i++) {
          const comp = obj.components[i];
          const targetModel = comp.path
            ? loadModel(normalizeOpcPath(model.path, comp.path))
            : model;
          const combined = composeRowAffine(comp.transform, worldTransform);
          visitObject(targetModel, comp.objectId, combined, modelScale);
        }
      }
    } finally {
      stackKeys.delete(key);
    }
  };

  for (let i = 0; i < rootModel.buildItems.length; i++) {
    const item = rootModel.buildItems[i];
    const targetModel = item.path
      ? loadModel(normalizeOpcPath(rootModel.path, item.path))
      : rootModel;
    visitObject(targetModel, item.objectId, item.transform, rootUnitScale);
  }

  return results;
}

const CONTENT_TYPES_XML = textEncoder.encode(
  '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n' +
    '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n' +
    '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n' +
    "</Types>\n",
);

const ROOT_RELS_XML = textEncoder.encode(
  '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n' +
    '  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n' +
    "</Relationships>\n",
);

function isIdentityMatrix4(m: ArrayLike<number>): boolean {
  return (
    m[0] === 1 && m[1] === 0 && m[2] === 0 && m[3] === 0 &&
    m[4] === 0 && m[5] === 1 && m[6] === 0 && m[7] === 0 &&
    m[8] === 0 && m[9] === 0 && m[10] === 1 && m[11] === 0 &&
    m[12] === 0 && m[13] === 0 && m[14] === 0 && m[15] === 1
  );
}

function isAffineMatrix4(m: ArrayLike<number>): boolean {
  return (
    Math.abs(m[3]) <= 1e-12 &&
    Math.abs(m[7]) <= 1e-12 &&
    Math.abs(m[11]) <= 1e-12 &&
    Math.abs(m[15] - 1) <= 1e-12
  );
}

function formatHexColor(r: number, g: number, b: number, a: number): string {
  const hex2 = (n: number) => (n & 0xff).toString(16).toUpperCase().padStart(2, "0");
  return a === 255
    ? `#${hex2(r)}${hex2(g)}${hex2(b)}`
    : `#${hex2(r)}${hex2(g)}${hex2(b)}${hex2(a)}`;
}

function buildThreeMfEntries(
  scene: SceneLike,
  meshes: MeshHandle | readonly MeshHandle[],
  options?: ThreeMfExportOptions,
): { name: string; data: Uint8Array; compress: boolean }[] {
  const list = Array.isArray(meshes) ? meshes : [meshes as MeshHandle];
  const unit: ThreeMfUnit = options?.unit ?? "millimeter";
  const unitScale = UNIT_TO_MM[unit];
  const invUnitScale = 1 / unitScale;
  const applyPlacement = options?.applyPlacement !== false;
  const includeColors = options?.vertexColors !== false;
  const compress = options?.compress !== false;
  const baseName = options?.name ?? "mesh";

  const colorList: string[] = [];
  const colorToIndex = new Map<number, number>();
  const getColorIndex = (r: number, g: number, b: number, a: number): number => {
    const key = (((r & 0xff) << 24) | ((g & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff)) >>> 0;
    let idx = colorToIndex.get(key);
    if (idx === undefined) {
      idx = colorList.length;
      colorList.push(formatHexColor(r, g, b, a));
      colorToIndex.set(key, idx);
    }
    return idx;
  };

  interface ExtractedObject {
    id: number;
    name: string;
    positions: Float32Array;
    colors: Uint8Array | null;
    indices: Uint32Array;
    itemTransformAttr: string;
  }

  const objects: ExtractedObject[] = [];
  let hasAnyColors = false;
  // Color group takes id=1 when needed; objects start at id=2 (or 1 when uncolored).
  // Assign ids after collecting objects so we know if any colors exist.
  for (let mIdx = 0; mIdx < list.length; mIdx++) {
    const handle = list[mIdx];
    const views = scene.views(handle);
    const rawPos = views.positions().slice();
    const rawIdx = views.indices().slice();
    const rawCol = views.colors();
    const vertexSlots = rawPos.length / 3;
    const triSlots = rawIdx.length / 3;
    const hasColors =
      includeColors &&
      rawCol !== null &&
      rawCol !== undefined &&
      rawCol.length >= (4 / 3) * rawPos.length;
    const colorsCopy = hasColors ? rawCol.slice() : null;

    const remap = new Int32Array(vertexSlots).fill(-1);
    const kept = new Uint32Array(rawIdx.length);
    let triangles = 0;
    let vertices = 0;

    for (let t = 0; t < triSlots; t++) {
      const a = rawIdx[3 * t];
      const b = rawIdx[3 * t + 1];
      const c = rawIdx[3 * t + 2];
      if (a === 0 && b === 0 && c === 0) {
        continue;
      }
      for (const v of [a, b, c]) {
        if (v >= vertexSlots) {
          throw new Error(
            `fast-mesh-wasm/io: triangle ${t} indexes missing vertex ${v}`,
          );
        }
        if (remap[v] === -1) {
          remap[v] = vertices++;
        }
      }
      kept[3 * triangles] = remap[a];
      kept[3 * triangles + 1] = remap[b];
      kept[3 * triangles + 2] = remap[c];
      triangles++;
    }

    const outPos = new Float32Array(3 * vertices);
    const outColors = colorsCopy !== null ? new Uint8Array(4 * vertices) : null;
    for (let v = 0; v < vertexSlots; v++) {
      const to = remap[v];
      if (to === -1) continue;
      outPos[3 * to] = rawPos[3 * v];
      outPos[3 * to + 1] = rawPos[3 * v + 1];
      outPos[3 * to + 2] = rawPos[3 * v + 2];
      if (outColors !== null && colorsCopy !== null) {
        outColors[4 * to] = colorsCopy[4 * v];
        outColors[4 * to + 1] = colorsCopy[4 * v + 1];
        outColors[4 * to + 2] = colorsCopy[4 * v + 2];
        outColors[4 * to + 3] = colorsCopy[4 * v + 3];
      }
    }

    let itemTransformAttr = "";
    if (applyPlacement) {
      const m = scene.matrix4(handle);
      if (!isIdentityMatrix4(m)) {
        if (isAffineMatrix4(m)) {
          const vals = [
            m[0], m[1], m[2],
            m[4], m[5], m[6],
            m[8], m[9], m[10],
            m[12] * invUnitScale,
            m[13] * invUnitScale,
            m[14] * invUnitScale,
          ];
          itemTransformAttr = ` transform="${vals.join(" ")}"`;
        } else {
          for (let i = 0; i < outPos.length; i += 3) {
            const x = outPos[i];
            const y = outPos[i + 1];
            const z = outPos[i + 2];
            outPos[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
            outPos[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
            outPos[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
          }
        }
      }
    }

    if (outColors !== null && triangles > 0) {
      hasAnyColors = true;
    }

    const objName = list.length === 1 ? baseName : `${baseName}-${mIdx + 1}`;
    objects.push({
      id: mIdx + 1,
      name: objName,
      positions: outPos,
      colors: outColors,
      indices: kept.subarray(0, 3 * triangles),
      itemTransformAttr,
    });
  }

  // Build XML string; reserve id=1 for <m:colorgroup> when colors are written.
  const colorGroupId = 1;
  if (hasAnyColors) {
    for (let i = 0; i < objects.length; i++) {
      objects[i].id = i + 2;
    }
  }

  const bodyParts: string[] = [];
  for (let i = 0; i < objects.length; i++) {
    const obj = objects[i];
    const vertCount = obj.positions.length / 3;
    const triCount = obj.indices.length / 3;

    // Pre-register vertex color indices so `<m:colorgroup id="1">` can be emitted before objects
    let vertColorIds: Uint32Array | null = null;
    if (obj.colors !== null && triCount > 0) {
      vertColorIds = new Uint32Array(vertCount);
      for (let v = 0; v < vertCount; v++) {
        vertColorIds[v] = getColorIndex(
          obj.colors[4 * v],
          obj.colors[4 * v + 1],
          obj.colors[4 * v + 2],
          obj.colors[4 * v + 3],
        );
      }
    }

    bodyParts.push(
      `    <object id="${obj.id}" name="${xmlEscapeAttr(obj.name)}" type="model">\n` +
        "      <mesh>\n" +
        "        <vertices>\n",
    );
    for (let v = 0; v < vertCount; v++) {
      const x = obj.positions[3 * v] * invUnitScale;
      const y = obj.positions[3 * v + 1] * invUnitScale;
      const z = obj.positions[3 * v + 2] * invUnitScale;
      bodyParts.push(`          <vertex x="${x}" y="${y}" z="${z}"/>\n`);
    }
    bodyParts.push("        </vertices>\n        <triangles>\n");
    for (let t = 0; t < triCount; t++) {
      const a = obj.indices[3 * t];
      const b = obj.indices[3 * t + 1];
      const c = obj.indices[3 * t + 2];
      if (vertColorIds !== null) {
        const ca = vertColorIds[a];
        const cb = vertColorIds[b];
        const cc = vertColorIds[c];
        if (ca === cb && cb === cc) {
          bodyParts.push(
            `          <triangle v1="${a}" v2="${b}" v3="${c}" pid="${colorGroupId}" p1="${ca}"/>\n`,
          );
        } else {
          bodyParts.push(
            `          <triangle v1="${a}" v2="${b}" v3="${c}" pid="${colorGroupId}" p1="${ca}" p2="${cb}" p3="${cc}"/>\n`,
          );
        }
      } else {
        bodyParts.push(`          <triangle v1="${a}" v2="${b}" v3="${c}"/>\n`);
      }
    }
    bodyParts.push("        </triangles>\n      </mesh>\n    </object>\n");
  }

  const xmlParts: string[] = [
    '<?xml version="1.0" encoding="UTF-8"?>\n',
    `<model unit="${unit}" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"` +
      (hasAnyColors
        ? ' xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02"'
        : "") +
      ">\n  <resources>\n",
  ];

  if (hasAnyColors) {
    xmlParts.push(`    <m:colorgroup id="${colorGroupId}">\n`);
    for (let i = 0; i < colorList.length; i++) {
      xmlParts.push(`      <m:color color="${colorList[i]}"/>\n`);
    }
    xmlParts.push("    </m:colorgroup>\n");
  }

  for (let i = 0; i < bodyParts.length; i++) {
    xmlParts.push(bodyParts[i]);
  }
  xmlParts.push("  </resources>\n  <build>\n");
  for (let i = 0; i < objects.length; i++) {
    xmlParts.push(
      `    <item objectid="${objects[i].id}"${objects[i].itemTransformAttr}/>\n`,
    );
  }
  xmlParts.push("  </build>\n</model>\n");

  const modelBytes = textEncoder.encode(xmlParts.join(""));
  return [
    { name: "[Content_Types].xml", data: CONTENT_TYPES_XML, compress },
    { name: "_rels/.rels", data: ROOT_RELS_XML, compress },
    { name: "3D/3dmodel.model", data: modelBytes, compress },
  ];
}

/**
 * Export one or more scene meshes as a compressed 3MF archive.
 */
export async function export3MF(
  scene: SceneLike,
  meshes: MeshHandle | readonly MeshHandle[],
  options?: ThreeMfExportOptions,
): Promise<Uint8Array> {
  const entries = buildThreeMfEntries(scene, meshes, options);
  return writeZip(entries, { compress: options?.compress ?? true });
}

/**
 * Export one or more scene meshes synchronously as an uncompressed (`STORE`) 3MF archive.
 */
export function export3MFSync(
  scene: SceneLike,
  meshes: MeshHandle | readonly MeshHandle[],
  options?: ThreeMfExportOptions,
): Uint8Array {
  const entries = buildThreeMfEntries(scene, meshes, options);
  return writeZipSync(entries);
}
