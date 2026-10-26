// A minimal DICOM Part-10 slice parser: explicit and implicit VR little
// endian, uncompressed. It reads the series, geometry, rescale and pixel
// tags a series stack needs and skips everything else (including undefined-
// length sequences); encapsulated transfer syntaxes fail with a clear
// message instead of misparsed pixels.
export interface DicomSlice {
  seriesUid: string;
  instanceUid: string;
  rows: number;
  columns: number;
  bitsAllocated: 8 | 16;
  pixelRepresentation: 0 | 1;
  rescaleSlope: number;
  rescaleIntercept: number;
  /** Row and column spacing in mm. */
  pixelSpacing: [number, number];
  /** The first voxel's centre in mm (patient coordinates). */
  position: [number, number, number];
  /** Row then column direction cosines. */
  orientation: [number, number, number, number, number, number];
  /** Raw samples in native order (MONOCHROME1 reads dark anatomy). */
  pixels: Uint8Array | Uint16Array | Int16Array;
  /** SeriesDescription, or "" when the files carry none. */
  description: string;
  /** MONOCHROME1: bright is air, so the isosurface flips sides. */
  inverted: boolean;
}

/** Whether the buffer carries the DICM prefix at byte 128. */
export function isDicom(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 132) {
    return false;
  }
  const b = new Uint8Array(buffer, 128, 4);
  return b[0] === 0x44 && b[1] === 0x49 && b[2] === 0x43 && b[3] === 0x4d;
}

const UNDEFINED = 0xffffffff;
const LONG_VR = new Set(["OB", "OD", "OF", "OL", "OV", "OW", "SQ", "UC", "UR", "UT", "UN"]);

/** Implicit-VR lookup for the tags read (lengths skip the rest). */
const IMPLICIT_VR = new Map<number, string>([
  [0x00080018, "UI"],
  [0x0008103e, "LO"],
  [0x0020000e, "UI"],
  [0x00280002, "US"],
  [0x00280004, "CS"],
  [0x00280010, "US"],
  [0x00280011, "US"],
  [0x00280100, "US"],
  [0x00280101, "US"],
  [0x00280102, "US"],
  [0x00280103, "US"],
  [0x00280030, "DS"],
  [0x00200032, "DS"],
  [0x00200037, "DS"],
  [0x00281052, "DS"],
  [0x00281053, "DS"],
  [0x7fe00010, "OW"],
]);

class Reader {
  readonly view: DataView;
  readonly bytes: Uint8Array;
  offset = 0;

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
    this.bytes = new Uint8Array(buffer);
  }

  get done(): boolean {
    return this.offset >= this.bytes.length;
  }

  tag(): number {
    const group = this.view.getUint16(this.offset, true);
    const element = this.view.getUint16(this.offset + 2, true);
    this.offset += 4;
    return group * 0x10000 + element;
  }

  u16(): number {
    const v = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return v;
  }

  u32(): number {
    const v = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return v;
  }

  ascii(length: number): string {
    const s = new TextDecoder("ascii").decode(this.bytes.subarray(this.offset, this.offset + length));
    this.offset += length;
    return s;
  }

  skip(length: number): void {
    this.offset += length;
  }
}

function fail(name: string, message: string): never {
  throw new Error(`${name}: ${message}`);
}

/** Skips an undefined-length sequence or item content (nested items recurse). */
function skipSequence(r: Reader, name: string): void {
  for (;;) {
    if (r.done) {
      fail(name, "DICOM ends inside an undefined-length sequence");
    }
    const tag = r.tag();
    const length = r.u32();
    if (tag === 0xfffee0dd) {
      // Sequence delimitation item: no value (a zero length follows).
      return;
    }
    if (tag === 0xfffee000) {
      if (length === UNDEFINED) {
        skipItems(r, name);
      } else {
        r.skip(length);
      }
      continue;
    }
    fail(name, `DICOM has an unexpected tag ${(tag >>> 0).toString(16)} inside a sequence`);
  }
}

/** Skips undefined-length item content up to its delimitation item. */
function skipItems(r: Reader, name: string): void {
  for (;;) {
    if (r.done) {
      fail(name, "DICOM ends inside an undefined-length item");
    }
    const tag = r.tag();
    const length = r.u32();
    if (tag === 0xfffee00d) {
      return;
    }
    if (tag === 0xfffee000) {
      if (length === UNDEFINED) {
        skipItems(r, name);
      } else {
        r.skip(length);
      }
      continue;
    }
    // A nested dataset element: implicit-style header (no VR here).
    if (length === UNDEFINED) {
      skipSequence(r, name);
    } else {
      r.skip(length);
    }
  }
}

interface Element {
  tag: number;
  vr: string;
  /** Value bytes (never the undefined length: sequences skip instead). */
  value: Uint8Array;
}

/** Reads one dataset element, skipping sequence contents. */
function readElement(r: Reader, name: string, implicit: boolean): Element | null {
  if (r.done) {
    return null;
  }
  if (r.offset + 8 > r.bytes.length) {
    fail(name, "DICOM ends inside an element header");
  }
  const tag = r.tag();
  let vr: string;
  let length: number;
  if (implicit) {
    vr = IMPLICIT_VR.get(tag) ?? "UN";
    length = r.u32();
  } else {
    vr = r.ascii(2);
    if (LONG_VR.has(vr)) {
      r.skip(2);
      length = r.u32();
    } else {
      length = r.u16();
    }
  }
  if (length === UNDEFINED) {
    if (tag === 0x7fe00010) {
      fail(name, "DICOM pixel data is encapsulated (compressed): only uncompressed transfer syntaxes load");
    }
    skipSequence(r, name);
    return { tag, vr, value: new Uint8Array(0) };
  }
  if (r.offset + length > r.bytes.length) {
    fail(name, "DICOM ends inside an element value");
  }
  const value = r.bytes.subarray(r.offset, r.offset + length);
  r.skip(length);
  return { tag, vr, value };
}

const text = (value: Uint8Array): string => new TextDecoder("ascii").decode(value).replace(/\0+$/, "").trim();

function numbers(name: string, label: string, value: Uint8Array, count: number): number[] {
  const parts = text(value).split("\\").map((p) => p.trim());
  if (parts.length !== count) {
    fail(name, `DICOM ${label} has ${parts.length} values, not ${count}`);
  }
  const nums = parts.map(Number);
  if (nums.some((n) => !Number.isFinite(n))) {
    fail(name, `DICOM ${label} is not numeric`);
  }
  return nums;
}

function usValue(value: Uint8Array): number {
  return new DataView(value.buffer, value.byteOffset, value.byteLength).getUint16(0, true);
}

/** Parses one uncompressed slice; `@throws` with the file name and the reason. */
export function parseDicom(name: string, buffer: ArrayBuffer): DicomSlice {
  if (!isDicom(buffer)) {
    fail(name, "not a DICOM Part-10 file (missing the DICM prefix)");
  }
  const r = new Reader(buffer);
  r.offset = 132;
  // The meta group is always explicit VR little endian; the dataset starts
  // past it (rewind to the first non-meta header: the walk consumes it).
  let transferSyntax = "";
  let datasetAt = r.bytes.length;
  for (;;) {
    const at = r.offset;
    const el = readElement(r, name, false);
    if (el === null) {
      break;
    }
    if ((el.tag >>> 16) !== 0x0002) {
      datasetAt = at;
      break;
    }
    if (el.tag === 0x00020010) {
      transferSyntax = text(el.value);
    }
  }
  if (transferSyntax !== "1.2.840.10008.1.2" && transferSyntax !== "1.2.840.10008.1.2.1") {
    fail(
      name,
      `DICOM transfer syntax ${transferSyntax === "" ? "(missing)" : transferSyntax} is not supported (uncompressed little endian only)`,
    );
  }
  const implicit = transferSyntax === "1.2.840.10008.1.2";
  r.offset = datasetAt;
  let seriesUid = "";
  let instanceUid = "";
  let description = "";
  let samples = 0;
  let photometric = "";
  let rows = 0;
  let columns = 0;
  let bitsAllocated = 0;
  let pixelRepresentation = 0;
  let pixelSpacing: [number, number] | null = null;
  let position: [number, number, number] | null = null;
  let orientation: [number, number, number, number, number, number] | null = null;
  let slope = 1;
  let intercept = 0;
  let pixels: Uint8Array | null = null;
  for (;;) {
    const el = readElement(r, name, implicit);
    if (el === null) {
      break;
    }
    switch (el.tag) {
      case 0x00080018:
        instanceUid = text(el.value);
        break;
      case 0x0008103e:
        description = text(el.value);
        break;
      case 0x0020000e:
        seriesUid = text(el.value);
        break;
      case 0x00280002:
        samples = usValue(el.value);
        break;
      case 0x00280004:
        photometric = text(el.value);
        break;
      case 0x00280010:
        rows = usValue(el.value);
        break;
      case 0x00280011:
        columns = usValue(el.value);
        break;
      case 0x00280100:
        bitsAllocated = usValue(el.value);
        break;
      case 0x00280103:
        pixelRepresentation = usValue(el.value);
        break;
      case 0x00280030: {
        const [dy, dx] = numbers(name, "PixelSpacing (0028,0030)", el.value, 2);
        pixelSpacing = [dy, dx];
        break;
      }
      case 0x00200032: {
        const [x, y, z] = numbers(name, "ImagePositionPatient (0020,0032)", el.value, 3);
        position = [x, y, z];
        break;
      }
      case 0x00200037: {
        const o = numbers(name, "ImageOrientationPatient (0020,0037)", el.value, 6);
        orientation = [o[0], o[1], o[2], o[3], o[4], o[5]];
        break;
      }
      case 0x00281052:
        intercept = numbers(name, "RescaleIntercept (0028,1052)", el.value, 1)[0];
        break;
      case 0x00281053:
        slope = numbers(name, "RescaleSlope (0028,1053)", el.value, 1)[0];
        break;
      case 0x7fe00010:
        pixels = el.value;
        break;
      default:
        break;
    }
  }
  if (seriesUid === "") {
    fail(name, "DICOM is missing SeriesInstanceUID (0020,000E)");
  }
  if (samples !== 1) {
    fail(name, `DICOM has ${samples} samples per pixel (grayscale only)`);
  }
  if (rows <= 0 || columns <= 0) {
    fail(name, "DICOM has no rows or columns");
  }
  if (bitsAllocated !== 8 && bitsAllocated !== 16) {
    fail(name, `DICOM has ${bitsAllocated} allocated bits (8 or 16 only)`);
  }
  if (pixelRepresentation !== 0 && pixelRepresentation !== 1) {
    fail(name, `DICOM pixel representation ${pixelRepresentation} is not 0 or 1`);
  }
  if (pixelSpacing === null) {
    fail(name, "DICOM is missing PixelSpacing (0028,0030)");
  }
  if (position === null) {
    fail(name, "DICOM is missing ImagePositionPatient (0020,0032)");
  }
  if (orientation === null) {
    fail(name, "DICOM is missing ImageOrientationPatient (0020,0037)");
  }
  if (pixels === null) {
    fail(name, "DICOM has no pixel data");
  }
  const needed = (rows * columns * bitsAllocated) / 8;
  if (pixels.length < needed) {
    fail(name, `DICOM pixel data holds ${pixels.length} bytes, not ${needed}`);
  }
  let view = pixels.subarray(0, needed);
  if (bitsAllocated === 16 && view.byteOffset % 2 !== 0) {
    // Malformed (odd value lengths upstream): copy to an aligned buffer.
    const aligned = new Uint8Array(needed);
    aligned.set(view);
    view = aligned;
  }
  const samplesOut: Uint8Array | Uint16Array | Int16Array =
    bitsAllocated === 8
      ? new Uint8Array(view.buffer, view.byteOffset, needed)
      : pixelRepresentation === 1
        ? new Int16Array(view.buffer, view.byteOffset, needed / 2)
        : new Uint16Array(view.buffer, view.byteOffset, needed / 2);
  return {
    seriesUid,
    instanceUid,
    description,
    rows,
    columns,
    bitsAllocated,
    pixelRepresentation: pixelRepresentation as 0 | 1,
    rescaleSlope: slope,
    rescaleIntercept: intercept,
    pixelSpacing,
    position,
    orientation,
    pixels: samplesOut,
    inverted: photometric === "MONOCHROME1",
  };
}
