// PLY, ASCII and binary little-endian: the `vertex` element's x, y, z and the
// `face` element's index list (`vertex_indices` or `vertex_index`, any
// polygon, fan-triangulated). Other properties and elements are skipped.
import { checkMesh, type MeshData } from "./mesh-data";

type Scalar = "int8" | "uint8" | "int16" | "uint16" | "int32" | "uint32" | "float32" | "float64";

const SCALARS: Record<string, Scalar> = {
  char: "int8",
  int8: "int8",
  uchar: "uint8",
  uint8: "uint8",
  short: "int16",
  int16: "int16",
  ushort: "uint16",
  uint16: "uint16",
  int: "int32",
  int32: "int32",
  uint: "uint32",
  uint32: "uint32",
  float: "float32",
  float32: "float32",
  double: "float64",
  float64: "float64",
};

const SIZE: Record<Scalar, number> = {
  int8: 1,
  uint8: 1,
  int16: 2,
  uint16: 2,
  int32: 4,
  uint32: 4,
  float32: 4,
  float64: 8,
};

type Property = { name: string; type: Scalar; count?: undefined } | { name: string; type: Scalar; count: Scalar };

interface Element {
  name: string;
  size: number;
  properties: Property[];
}

interface Header {
  format: "ascii" | "binary_little_endian";
  elements: Element[];
  /** The byte offset of the body. */
  body: number;
}

/** @throws on a malformed or big-endian file, a face index out of range, or no faces. */
export function parsePly(buffer: ArrayBuffer): MeshData {
  const header = readHeader(new Uint8Array(buffer));
  const vertex = header.elements.find((e) => e.name === "vertex");
  const face = header.elements.find((e) => e.name === "face");
  if (vertex === undefined || face === undefined) {
    throw new Error("PLY: needs a vertex and a face element");
  }
  const xyz = ["x", "y", "z"].map((n) => vertex.properties.findIndex((p) => p.name === n && p.count === undefined));
  if (xyz.includes(-1)) {
    throw new Error("PLY: the vertex element has no x, y, z");
  }
  let list = face.properties.findIndex((p) => p.name === "vertex_indices" || p.name === "vertex_index");
  if (list === -1) {
    list = face.properties.findIndex((p) => p.count !== undefined);
  }
  if (list === -1 || face.properties[list].count === undefined) {
    throw new Error("PLY: the face element has no vertex index list");
  }

  const positions = new Float64Array(3 * vertex.size);
  const indices: number[] = [];
  const reader = header.format === "ascii" ? asciiReader(buffer, header.body) : binaryReader(buffer, header.body);
  const corner: number[] = [];
  for (const element of header.elements) {
    for (let i = 0; i < element.size; i++) {
      for (let p = 0; p < element.properties.length; p++) {
        const prop = element.properties[p];
        if (prop.count === undefined) {
          const value = reader.read(prop.type);
          if (element === vertex) {
            const axis = xyz.indexOf(p);
            if (axis !== -1) {
              positions[3 * i + axis] = value;
            }
          }
          continue;
        }
        const n = reader.read(prop.count);
        if (element === face && p === list) {
          corner.length = 0;
          for (let k = 0; k < n; k++) {
            corner.push(reader.read(prop.type));
          }
          for (let k = 1; k + 1 < n; k++) {
            indices.push(corner[0], corner[k], corner[k + 1]);
          }
        } else {
          for (let k = 0; k < n; k++) {
            reader.read(prop.type);
          }
        }
      }
    }
  }
  for (const i of indices) {
    if (!Number.isInteger(i) || i < 0 || i >= vertex.size) {
      throw new Error(`PLY: face index ${i} is out of range (${vertex.size} vertices)`);
    }
  }
  return checkMesh({ positions, indices: Uint32Array.from(indices) }, "PLY");
}

function readHeader(bytes: Uint8Array): Header {
  const magic = new TextDecoder().decode(bytes.subarray(0, 4));
  if (!/^ply\s/.test(magic)) {
    throw new Error("not a PLY file");
  }
  // The header is ASCII and ends with the line "end_header".
  const end = indexOf(bytes, new TextEncoder().encode("end_header"));
  if (end === -1) {
    throw new Error("PLY: no end_header");
  }
  let body = end + "end_header".length;
  if (bytes[body] === 13) {
    body++;
  }
  if (bytes[body] === 10) {
    body++;
  }
  const lines = new TextDecoder().decode(bytes.subarray(0, end)).split(/\r?\n/);
  let format: Header["format"] | undefined;
  const elements: Element[] = [];
  for (const line of lines) {
    const w = line.trim().split(/\s+/);
    if (w[0] === "format") {
      if (w[1] === "binary_big_endian") {
        throw new Error("PLY: big-endian binary PLY is not supported");
      }
      if (w[1] !== "ascii" && w[1] !== "binary_little_endian") {
        throw new Error(`PLY: unknown format ${w[1]}`);
      }
      format = w[1];
    } else if (w[0] === "element") {
      const size = Number(w[2]);
      if (!Number.isInteger(size) || size < 0) {
        throw new Error(`PLY: bad element line "${line}"`);
      }
      elements.push({ name: w[1], size, properties: [] });
    } else if (w[0] === "property") {
      const element = elements.at(-1);
      if (element === undefined) {
        throw new Error("PLY: a property before any element");
      }
      if (w[1] === "list") {
        const count = SCALARS[w[2]];
        const type = SCALARS[w[3]];
        if (count === undefined || type === undefined) {
          throw new Error(`PLY: bad property line "${line}"`);
        }
        element.properties.push({ name: w[4], type, count });
      } else {
        const type = SCALARS[w[1]];
        if (type === undefined) {
          throw new Error(`PLY: bad property line "${line}"`);
        }
        element.properties.push({ name: w[2], type });
      }
    }
  }
  if (format === undefined) {
    throw new Error("PLY: no format line");
  }
  return { format, elements, body };
}

function indexOf(bytes: Uint8Array, needle: Uint8Array): number {
  outer: for (let i = 0; i + needle.length <= bytes.length; i++) {
    for (let k = 0; k < needle.length; k++) {
      if (bytes[i + k] !== needle[k]) {
        continue outer;
      }
    }
    return i;
  }
  return -1;
}

interface Reader {
  read(type: Scalar): number;
}

function binaryReader(buffer: ArrayBuffer, offset: number): Reader {
  const view = new DataView(buffer);
  let at = offset;
  return {
    read(type) {
      if (at + SIZE[type] > view.byteLength) {
        throw new Error("PLY: the file ends early");
      }
      let v: number;
      switch (type) {
        case "int8":
          v = view.getInt8(at);
          break;
        case "uint8":
          v = view.getUint8(at);
          break;
        case "int16":
          v = view.getInt16(at, true);
          break;
        case "uint16":
          v = view.getUint16(at, true);
          break;
        case "int32":
          v = view.getInt32(at, true);
          break;
        case "uint32":
          v = view.getUint32(at, true);
          break;
        case "float32":
          v = view.getFloat32(at, true);
          break;
        case "float64":
          v = view.getFloat64(at, true);
          break;
      }
      at += SIZE[type];
      return v;
    },
  };
}

function asciiReader(buffer: ArrayBuffer, offset: number): Reader {
  const src = new TextDecoder().decode(new Uint8Array(buffer, offset));
  const token = /\S+/g;
  return {
    read() {
      const m = token.exec(src);
      if (m === null) {
        throw new Error("PLY: the file ends early");
      }
      const v = Number(m[0]);
      if (Number.isNaN(v)) {
        throw new Error(`PLY: "${m[0]}" is not a number`);
      }
      return v;
    },
  };
}
