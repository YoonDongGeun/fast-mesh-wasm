// Synthetic DICOM Part-10 files for tests: explicit or implicit VR little
// endian, a 2x2 16-bit slice (or custom pixels), and the geometry tags a
// series stack needs. Real files carry much more; the parser skips the rest.
export interface FixtureSlice {
  seriesUid?: string;
  instanceUid?: string;
  rows?: number;
  columns?: number;
  bitsAllocated?: 8 | 16;
  pixelRepresentation?: 0 | 1;
  slope?: number;
  intercept?: number;
  pixelSpacing?: [number, number];
  position?: [number, number, number];
  orientation?: [number, number, number, number, number, number];
  pixels?: number[];
  photometric?: string;
  description?: string;
  transferSyntax?: string;
  implicit?: boolean;
  /** Extra dataset elements (tag, VR, value bytes) the parser must skip. */
  extra?: { group: number; element: number; vr: string; value: Uint8Array }[];
}

function ascii(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

function element(
  out: number[],
  group: number,
  element: number,
  vr: string,
  value: Uint8Array,
  implicit: boolean,
): void {
  out.push(group & 0xff, group >> 8, element & 0xff, element >> 8);
  const longVr = vr === "OB" || vr === "OW" || vr === "SQ" || vr === "UC" || vr === "UR" || vr === "UT" || vr === "UN";
  if (!implicit) {
    out.push(vr.charCodeAt(0), vr.charCodeAt(1));
    if (longVr) {
      out.push(0, 0);
      const n = value.length;
      out.push(n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff, (n >> 24) & 0xff);
    } else {
      out.push(value.length & 0xff, value.length >> 8);
    }
  } else {
    const n = value.length;
    out.push(n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff, (n >> 24) & 0xff);
  }
  for (const b of value) {
    out.push(b);
  }
}

const us = (v: number): Uint8Array => new Uint8Array([v & 0xff, v >> 8]);
const usList = (...vs: number[]): Uint8Array => {
  const out = new Uint8Array(2 * vs.length);
  vs.forEach((v, i) => {
    out[2 * i] = v & 0xff;
    out[2 * i + 1] = v >> 8;
  });
  return out;
};
const padded = (s: string): Uint8Array => {
  const raw = ascii(s);
  if (raw.length % 2 === 0) {
    return raw;
  }
  const out = new Uint8Array(raw.length + 1);
  out.set(raw);
  out[raw.length] = 0x20;
  return out;
};
const uipad = (s: string): Uint8Array => {
  const raw = ascii(s);
  if (raw.length % 2 === 0) {
    return raw;
  }
  const out = new Uint8Array(raw.length + 1);
  out.set(raw);
  out[raw.length] = 0;
  return out;
};

/** A minimal Part-10 file for one slice. */
export function dicomFile(f: FixtureSlice = {}): ArrayBuffer {
  const rows = f.rows ?? 2;
  const columns = f.columns ?? 2;
  const bits = f.bitsAllocated ?? 16;
  const implicit = f.implicit ?? false;
  const ts = f.transferSyntax ?? (implicit ? "1.2.840.10008.1.2" : "1.2.840.10008.1.2.1");
  const out: number[] = new Array(128).fill(0);
  out.push(0x44, 0x49, 0x43, 0x4d); // DICM
  const meta: number[] = [];
  element(meta, 0x0002, 0x0001, "OB", new Uint8Array([0, 1]), false);
  element(meta, 0x0002, 0x0002, "UI", uipad("1.2.840.10008.5.1.4.1.1.2"), false);
  element(meta, 0x0002, 0x0003, "UI", uipad(f.instanceUid ?? "1.2.3.4"), false);
  element(meta, 0x0002, 0x0010, "UI", uipad(ts), false);
  element(meta, 0x0002, 0x0012, "UI", uipad("1.2.3.99"), false);
  const metaBytes = new Uint8Array(meta);
  out.push(0x02, 0x00, 0x00, 0x00, 0x55, 0x4c, 0x04, 0x00);
  const n = metaBytes.length;
  out.push(n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff, (n >> 24) & 0xff);
  for (const b of metaBytes) {
    out.push(b);
  }
  const el = (group: number, elem: number, vr: string, value: Uint8Array) =>
    element(out, group, elem, vr, value, implicit);
  el(0x0008, 0x0018, "UI", uipad(f.instanceUid ?? "1.2.3.4"));
  el(0x0020, 0x000e, "UI", uipad(f.seriesUid ?? "1.2.3"));
  if (f.description !== undefined) {
    el(0x0008, 0x103e, "LO", padded(f.description));
  }
  el(0x0028, 0x0002, "US", us(1));
  el(0x0028, 0x0004, "CS", padded(f.photometric ?? "MONOCHROME2"));
  el(0x0028, 0x0010, "US", us(rows));
  el(0x0028, 0x0011, "US", us(columns));
  el(0x0028, 0x0100, "US", us(bits));
  el(0x0028, 0x0101, "US", us(bits));
  el(0x0028, 0x0102, "US", us(bits - 1));
  el(0x0028, 0x0103, "US", us(f.pixelRepresentation ?? 0));
  const spacing = f.pixelSpacing ?? [1, 1];
  el(0x0028, 0x0030, "DS", padded(`${spacing[0]}\\${spacing[1]}`));
  const pos = f.position ?? [0, 0, 0];
  el(0x0020, 0x0032, "DS", padded(`${pos[0]}\\${pos[1]}\\${pos[2]}`));
  const ori = f.orientation ?? [1, 0, 0, 0, 1, 0];
  el(0x0020, 0x0037, "DS", padded(ori.join("\\")));
  el(0x0028, 0x1052, "DS", padded(String(f.intercept ?? 0)));
  el(0x0028, 0x1053, "DS", padded(String(f.slope ?? 1)));
  for (const x of f.extra ?? []) {
    el(x.group, x.element, x.vr, x.value);
  }
  const pixels = f.pixels ?? [0, 0, 0, 0];
  const raw = new Uint8Array((rows * columns * bits) / 8);
  const view = new DataView(raw.buffer);
  pixels.forEach((v, i) => {
    if (bits === 16) {
      view.setUint16(2 * i, v & 0xffff, true);
    } else {
      view.setUint8(i, v & 0xff);
    }
  });
  el(0x7fe0, 0x0010, bits === 16 ? "OW" : "OB", raw);
  return new Uint8Array(out).buffer as ArrayBuffer;
}
