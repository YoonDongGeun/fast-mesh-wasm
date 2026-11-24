// Height maps as images (#267): the 16-bit PNG encoder and decoder (against
// files built with Node's own zlib and checksum), the codes and their JSON,
// and the grids an image is sampled by.
import { deflateSync, inflateSync, crc32 as nodeCrc32 } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  gridSize,
  heightCodes,
  HeightImageEngine,
  type HeightImage,
  heightMapFile,
  imageGrid,
  isImageFile,
  MAX_CODE,
  parseHeightMapFile,
  pngHeights,
  readHeightImage,
} from "../src/io/height-image";
import { canDecodePng, decodePng, encodePng16, readPngHeader } from "../src/io/png";

/** A deterministic generator of 16-bit values. */
function noise(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s >>> 16;
  };
}

interface Chunk {
  type: string;
  data: Buffer;
  crc: number;
}

/** A PNG's chunks, read without the code under test. */
function chunks(bytes: Uint8Array): Chunk[] {
  const buf = Buffer.from(bytes);
  expect(buf.subarray(0, 8)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  const out: Chunk[] = [];
  for (let at = 8; at < buf.length; ) {
    const length = buf.readUInt32BE(at);
    const type = buf.toString("latin1", at + 4, at + 8);
    const data = buf.subarray(at + 8, at + 8 + length);
    out.push({ type, data, crc: buf.readUInt32BE(at + 8 + length) });
    at += 12 + length;
  }
  return out;
}

function chunk(type: string, data: Buffer): Buffer {
  const body = Buffer.concat([Buffer.from(type, "latin1"), data]);
  const out = Buffer.alloc(body.length + 8);
  out.writeUInt32BE(data.length, 0);
  body.copy(out, 4);
  out.writeUInt32BE(nodeCrc32(body), body.length + 4);
  return out;
}

const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

/** A PNG with the given samples, each row under the filter `filterOf` gives it. */
function makePng(
  width: number,
  height: number,
  depth: number,
  colorType: number,
  samples: readonly number[],
  options: { filterOf?: (row: number) => number; pieces?: number; interlaced?: boolean } = {},
): Uint8Array {
  const channels = CHANNELS[colorType];
  const bpp = (channels * depth) / 8;
  const stride = width * bpp;
  const plain = Buffer.alloc(stride * height);
  samples.forEach((v, i) => {
    if (depth === 16) {
      plain.writeUInt16BE(v, 2 * i);
    } else {
      plain[i] = v;
    }
  });
  const at = (row: number, i: number) => (row < 0 || i < 0 ? 0 : plain[row * stride + i]);
  const raw = Buffer.alloc((stride + 1) * height);
  for (let row = 0; row < height; row++) {
    const filter = options.filterOf?.(row) ?? 0;
    raw[row * (stride + 1)] = filter;
    for (let i = 0; i < stride; i++) {
      const a = at(row, i - bpp);
      const b = at(row - 1, i);
      const c = row > 0 ? at(row - 1, i - bpp) : 0;
      const p = a + b - c;
      const [pa, pb, pc] = [Math.abs(p - a), Math.abs(p - b), Math.abs(p - c)];
      const paeth = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      const predicted = [0, a, b, (a + b) >> 1, paeth][filter];
      raw[row * (stride + 1) + 1 + i] = at(row, i) - predicted;
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = depth;
  header[9] = colorType;
  header[12] = options.interlaced === true ? 1 : 0;
  const deflated = deflateSync(raw);
  const pieces = options.pieces ?? 1;
  const size = Math.ceil(deflated.length / pieces);
  const idat = Array.from({ length: pieces }, (_, k) => chunk("IDAT", deflated.subarray(k * size, (k + 1) * size)));
  return new Uint8Array(
    Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      chunk("IHDR", header),
      chunk("tEXt", Buffer.from("Comment\0a chunk to skip")),
      ...idat,
      chunk("IEND", Buffer.alloc(0)),
    ]),
  );
}

describe("16-bit PNG", () => {
  it("encodes grey images another decoder reads back exactly", async () => {
    for (const [w, h] of [
      [1, 1],
      [3, 2],
      [2, 5],
      [61, 47],
      [1000, 700],
    ]) {
      const next = noise(w * 31 + h);
      // Smooth with noise on top, so that both filters get used.
      const codes = Uint16Array.from({ length: w * h }, (_, i) => (i * 37 + (i % 5 === 0 ? next() : 0)) & 0xffff);
      const png = await encodePng16(codes, w, h);
      const list = chunks(png);
      expect(list[0].type).toBe("IHDR");
      expect(list.at(-1)?.type).toBe("IEND");
      for (const c of list) {
        expect(nodeCrc32(Buffer.concat([Buffer.from(c.type, "latin1"), c.data]))).toBe(c.crc);
      }
      const header = list[0].data;
      expect([header.readUInt32BE(0), header.readUInt32BE(4)]).toEqual([w, h]);
      // 16 bits, grey, deflate, adaptive filters, not interlaced.
      expect([...header.subarray(8)]).toEqual([16, 0, 0, 0, 0]);
      const idat = Buffer.concat(list.filter((c) => c.type === "IDAT").map((c) => c.data));
      const raw = inflateSync(idat);
      expect(raw.length).toBe((2 * w + 1) * h);
      // Undone here by hand: only Sub and Up are written.
      const bytes = Buffer.alloc(2 * w * h);
      for (let row = 0; row < h; row++) {
        const filter = raw[row * (2 * w + 1)];
        expect([1, 2]).toContain(filter);
        for (let i = 0; i < 2 * w; i++) {
          const left = i < 2 ? 0 : bytes[row * 2 * w + i - 2];
          const above = row === 0 ? 0 : bytes[(row - 1) * 2 * w + i];
          bytes[row * 2 * w + i] = raw[row * (2 * w + 1) + 1 + i] + (filter === 1 ? left : above);
        }
      }
      expect(Array.from(codes, (_, i) => bytes.readUInt16BE(2 * i))).toEqual(Array.from(codes));
      // And its own decoder.
      const back = await decodePng(png);
      expect([back.width, back.height, back.channels, back.depth]).toEqual([w, h, 1, 16]);
      expect(back.samples).toEqual(codes);
    }
  });

  it("refuses sizes that do not fit its codes", async () => {
    await expect(encodePng16(new Uint16Array(5), 2, 3)).rejects.toThrow(/width times height/);
    await expect(encodePng16(new Uint16Array(0), 0, 3)).rejects.toThrow(/width and a height/);
  });

  it("splits a large image's data over several chunks", async () => {
    const next = noise(5);
    const codes = Uint16Array.from({ length: 1500 * 900 }, next);
    const png = await encodePng16(codes, 1500, 900);
    expect(chunks(png).filter((c) => c.type === "IDAT").length).toBeGreaterThan(1);
    expect((await decodePng(png)).samples).toEqual(codes);
  });

  it("decodes every filter, depth and channel count", async () => {
    const next = noise(99);
    for (const depth of [8, 16]) {
      for (const colorType of [0, 2, 4, 6]) {
        const [w, h] = [7, 10];
        const max = depth === 16 ? 0xffff : 0xff;
        const samples = Array.from({ length: w * h * CHANNELS[colorType] }, () => next() & max);
        for (const filterOf of [(r: number) => r % 5, () => 3, () => 4]) {
          const png = makePng(w, h, depth, colorType, samples, { filterOf, pieces: 3 });
          const image = await decodePng(png);
          expect([image.width, image.height, image.channels, image.depth]).toEqual([w, h, CHANNELS[colorType], depth]);
          expect(Array.from(image.samples)).toEqual(samples);
        }
      }
    }
  });

  it("reports damaged files and leaves other kinds of PNG alone", async () => {
    const good = makePng(8, 8, 16, 0, Array.from({ length: 64 }, (_, i) => i * 1000));
    await expect(decodePng(good.subarray(0, good.length - 60))).rejects.toThrow(/cut short/);
    const broken = good.slice();
    broken[broken.length - 40] ^= 0xff;
    await expect(decodePng(broken)).rejects.toThrow();
    const filter = makePng(2, 2, 16, 0, [1, 2, 3, 4], { filterOf: () => 0 });
    await expect(decodePng(new Uint8Array([1, 2, 3]))).rejects.toThrow(/not a PNG/);
    expect(canDecodePng(readPngHeader(filter)!)).toBe(true);
    expect(readPngHeader(new Uint8Array(40))).toBeNull();
    const palette = makePng(2, 2, 8, 3, [0, 1, 1, 0]);
    const adam7 = makePng(2, 2, 16, 0, [1, 2, 3, 4], { interlaced: true });
    for (const png of [palette, adam7]) {
      expect(canDecodePng(readPngHeader(png)!)).toBe(false);
      await expect(decodePng(png)).rejects.toThrow(/not supported/);
    }
  });
});

describe("heights and codes", () => {
  it("rounds to the nearest code and puts missing pixels at 0", () => {
    const codes = heightCodes(Float32Array.of(0, 10, 5, Number.NaN, 10 / MAX_CODE, 11, -1, Infinity), 10);
    expect(Array.from(codes)).toEqual([0, 65535, 32768, 0, 1, 65535, 0, 65535]);
    expect(Array.from(heightCodes(Float32Array.of(0, Number.NaN), 0))).toEqual([0, 0]);
  });

  it("writes a JSON that reads back as the frame", () => {
    const frame = {
      width: 640,
      height: 480,
      origin: [1.5, -2, 3e-7] as [number, number, number],
      right: [1, 0, 0] as [number, number, number],
      up: [0, 0, -1] as [number, number, number],
      normal: [0, 1, 0] as [number, number, number],
      pixelSize: 0.05,
      range: 12.25,
    };
    const doc = heightMapFile(frame, "part-height-map.png", ["part"]);
    expect(doc).toMatchObject({ type: "height-map", units: "mm", noSurface: 0, meshes: ["part"] });
    // A code's height, and the pixel where the JSON says.
    expect(doc.heightStep * MAX_CODE).toBeCloseTo(12.25, 12);
    expect(parseHeightMapFile(JSON.stringify(doc))).toEqual(frame);
    expect(parseHeightMapFile("{")).toBeNull();
    expect(parseHeightMapFile("[1]")).toBeNull();
    expect(parseHeightMapFile('{"type":"mesh"}')).toBeNull();
    for (const broken of [
      { width: 0 },
      { width: 2.5 },
      { pixelSize: 0 },
      { pixelSize: "1" },
      { heightRange: -1 },
      { origin: [0, 0] },
      { up: [0, 0, null] },
    ]) {
      expect(parseHeightMapFile(JSON.stringify({ ...doc, ...broken })), JSON.stringify(broken)).toBeNull();
    }
  });
});

describe("images as heights", () => {
  it("takes 16 bits as they are, colours by their brightness, and fades transparent pixels", async () => {
    const grey = makePng(2, 2, 16, 0, [0, 1, 32768, 65535]);
    const image = await readHeightImage(new Blob([grey as Uint8Array<ArrayBuffer>]));
    expect([image.width, image.height, image.depth]).toEqual([2, 2, 16]);
    expect(Array.from(image.values)).toEqual(Array.from(Float32Array.of(0, 1 / 65535, 32768 / 65535, 1)));
    const rgb = await decodePng(makePng(3, 2, 8, 2, [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 128, 128, 128]));
    const luma = pngHeights(rgb).values;
    expect(luma[0]).toBeCloseTo(0.2126, 5);
    expect(luma[1]).toBeCloseTo(0.7152, 5);
    expect(luma[2]).toBeCloseTo(0.0722, 5);
    expect(luma[3]).toBeCloseTo(1, 6);
    expect(luma[5]).toBeCloseTo(128 / 255, 5);
    const alpha = pngHeights(await decodePng(makePng(2, 2, 8, 4, [255, 255, 255, 0, 255, 51, 100, 255]))).values;
    expect(Array.from(alpha, (v) => Number(v.toFixed(4)))).toEqual([1, 0, 0.2, Number((100 / 255).toFixed(4))]);
  });

  it("refuses images that are too small, too large or not images", async () => {
    const thin = makePng(1, 4, 16, 0, [1, 2, 3, 4]);
    await expect(readHeightImage(new Blob([thin as Uint8Array<ArrayBuffer>]))).rejects.toThrow(/at least 2 pixels/);
    // Only its header says how large it is: refused before its data is read.
    const huge = makePng(2, 2, 16, 0, [1, 2, 3, 4]);
    new DataView(huge.buffer).setUint32(16, 100_000);
    new DataView(huge.buffer).setUint32(20, 100_000);
    await expect(readHeightImage(new Blob([huge as Uint8Array<ArrayBuffer>]))).rejects.toThrow(/pixels: at most/);
    // Node has no image decoder: anything but these PNGs is not readable here.
    await expect(readHeightImage(new Blob(["not an image"]))).rejects.toThrow(/not an image/);
  });

  it("knows image files by type or name", () => {
    expect(isImageFile({ name: "a.PNG", type: "" })).toBe(true);
    expect(isImageFile({ name: "a.jpeg", type: "" })).toBe(true);
    expect(isImageFile({ name: "photo", type: "image/webp" })).toBe(true);
    expect(isImageFile({ name: "a.stl", type: "" })).toBe(false);
    expect(isImageFile({ name: "png", type: "model/stl" })).toBe(false);
  });

  it("sizes grids by the longer side and keeps proportions", () => {
    expect(gridSize(1024, 768, 1024)).toEqual([1024, 768]);
    expect(gridSize(1024, 768, 2000)).toEqual([1024, 768]);
    expect(gridSize(1025, 513, 257)).toEqual([257, 129]);
    expect(gridSize(513, 1025, 257)).toEqual([129, 257]);
    expect(gridSize(1000, 2, 10)).toEqual([10, 2]);
    expect(gridSize(5, 5, 1)).toEqual([2, 2]);
  });

  const ramp = (w: number, h: number): HeightImage => ({
    width: w,
    height: h,
    depth: 16,
    values: Float32Array.from({ length: w * h }, (_, i) => (i % w) / (w - 1)),
  });

  it("copies values at the image's size and scales or inverts them", () => {
    const image: HeightImage = { width: 2, height: 2, depth: 8, values: Float32Array.of(0, 0.25, 0.5, 1) };
    expect(Array.from(imageGrid(image, 2, 2, { scale: 1, invert: false }))).toEqual([0, 0.25, 0.5, 1]);
    expect(Array.from(imageGrid(image, 2, 2, { scale: 8, invert: false }))).toEqual([0, 2, 4, 8]);
    expect(Array.from(imageGrid(image, 2, 2, { scale: 8, invert: true }))).toEqual([8, 6, 4, 0]);
    expect(() => imageGrid(image, 1, 2, { scale: 1, invert: false })).toThrow(RangeError);
  });

  it("samples flat images exactly and ramps within an eighth of a pixel, corner to corner", () => {
    for (const [W, H, w, h] of [
      [9, 5, 5, 3],
      [100, 40, 7, 5],
      [64, 64, 33, 20],
      [10, 10, 19, 13],
      [2, 2, 5, 4],
    ]) {
      const flat: HeightImage = { width: W, height: H, depth: 16, values: new Float32Array(W * H).fill(0.375) };
      for (const v of imageGrid(flat, w, h, { scale: 2, invert: false })) {
        expect(v).toBeCloseTo(0.75, 6);
      }
      const grid = imageGrid(ramp(W, H), w, h, { scale: 1, invert: false });
      for (let row = 0; row < h; row++) {
        for (let col = 1; col < w - 1; col++) {
          // Interpolated exactly; averaged over whole pixels, whose middles
          // miss a fractional window's by up to an eighth of a pixel.
          const slack = W > w ? 0.125 / (W - 1) : 0;
          const off = Math.abs(grid[row * w + col] - col / (w - 1));
          expect(off, `${W}x${H} to ${w}x${h} at ${col}`).toBeLessThanOrEqual(slack + 1e-6);
        }
      }
      // Reading by rows gives what reading columns does, transposed.
      const turned: HeightImage = {
        width: H,
        height: W,
        depth: 16,
        values: Float32Array.from({ length: W * H }, (_, i) => Math.floor(i / H) / (W - 1)),
      };
      const other = imageGrid(turned, h, w, { scale: 1, invert: false });
      for (let col = 0; col < w; col++) {
        expect(other[col * h]).toBeCloseTo(grid[col], 6);
      }
    }
  });

  it("averages the pixels around a sample when there are fewer samples", () => {
    // 9 pixels to 3 samples: 4 pixels a step, so the middle one is the mean of
    // pixels 2 and 6 at half weight and 3, 4, 5 whole.
    const values = Float32Array.of(0, 0, 8, 1, 2, 3, 4, 0, 0, 0, 0, 8, 1, 2, 3, 4, 0, 0);
    const image: HeightImage = { width: 9, height: 2, depth: 16, values };
    const grid = imageGrid(image, 3, 2, { scale: 1, invert: false });
    expect(grid[1]).toBeCloseTo((4 + 1 + 2 + 3 + 2) / 4, 6);
    expect(grid[4]).toBeCloseTo(grid[1], 6);
  });

  it("keeps images open by id until they are closed", async () => {
    const engine = new HeightImageEngine();
    const png = await engine.encode(Float32Array.of(0, 1, 2, 4), 2, 2, 4);
    const info = await engine.open(new Blob([png as Uint8Array<ArrayBuffer>]));
    expect(info).toMatchObject({ width: 2, height: 2, depth: 16 });
    const heights = engine.grid(info.id, 2, 2, { scale: 4, invert: false });
    Array.from([0, 1, 2, 4]).forEach((h, i) => expect(heights[i]).toBeCloseTo(h, 3));
    engine.close(info.id);
    expect(() => engine.grid(info.id, 2, 2, { scale: 1, invert: false })).toThrow(/closed/);
  });
});
