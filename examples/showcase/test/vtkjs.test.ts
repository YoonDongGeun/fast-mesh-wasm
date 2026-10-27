// The VTK sample reader: vtk.js-format descriptors (dims, spacing,
// origin, direction, scalar ref) into float volumes, resampled
// axis-aligned, then sliced for the DICOM pipeline.
import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  alignVolume,
  loadSample,
  parseSampleMeta,
  parseVtkJsImage,
  toFloatVolume,
  volumeToSlices,
} from "../src/load/vtkjs";

const DESCRIPTOR = {
  origin: [0, 0, 0],
  spacing: [3.2, 3.2, 1.5],
  extent: [0, 63, 0, 63, 0, 92],
  direction: [0.866, 0.5, 0, -0.5, 0.866, 0, 0, 0, 1],
  pointData: {
    arrays: [
      {
        data: {
          name: "ImageFile",
          dataType: "Int16Array",
          size: 380928,
          ref: { basepath: "data", id: "1839938295fe4f0839c9df8d9fa67137" },
        },
      },
    ],
  },
};

describe("parseVtkJsImage", () => {
  it("reads_dims_spacing_and_scalars", () => {
    const parsed = parseVtkJsImage(DESCRIPTOR);
    expect(parsed.dims).toEqual([64, 64, 93]);
    expect(parsed.spacing).toEqual([3.2, 3.2, 1.5]);
    expect(parsed.origin).toEqual([0, 0, 0]);
    expect(parsed.direction).toEqual([0.866, 0.5, 0, -0.5, 0.866, 0, 0, 0, 1]);
    expect(parsed.dataType).toBe("int16");
    expect(parsed.refId).toBe("1839938295fe4f0839c9df8d9fa67137");
  });

  it("accepts_axis_aligned_uint8", () => {
    const parsed = parseVtkJsImage({
      origin: [0, 0, 0],
      spacing: [1, 1, 1],
      extent: [0, 1, 0, 1, 0, 0],
      pointData: { arrays: [{ data: { dataType: "Uint8Array", size: 4, ref: { id: "x" } } }] },
    });
    expect(parsed.direction).toBe(null);
    expect(parsed.dataType).toBe("uint8");
  });

  it("rejects_garbage", () => {
    expect(() => parseVtkJsImage(null)).toThrow();
    expect(() => parseVtkJsImage({ ...DESCRIPTOR, extent: [0, 63] })).toThrow();
    expect(() =>
      parseVtkJsImage({
        ...DESCRIPTOR,
        pointData: { arrays: [{ data: { dataType: "Float64Array", size: 1, ref: { id: "x" } } }] },
      }),
    ).toThrow();
  });
});

describe("toFloatVolume", () => {
  it("reads_little_endian_int16_and_uint8", () => {
    expect(Array.from(toFloatVolume(new Uint8Array([0xff, 0xff, 0x00, 0x04]), "int16"))).toEqual([-1, 1024]);
    expect(Array.from(toFloatVolume(new Uint8Array([0, 255]), "uint8"))).toEqual([0, 255]);
  });
});

describe("alignVolume", () => {
  it("passes_axis_aligned_volumes_through", () => {
    const values = new Float32Array([1, 2, 3, 4]);
    const out = alignVolume(values, [2, 2, 1], [1, 1, 1], [0, 0, 0], null);
    expect(out.dims).toEqual([2, 2, 1]);
    expect(Array.from(out.values)).toEqual([1, 2, 3, 4]);
  });

  it("unrotates_a_quarter_turn_exactly", () => {
    // The input's x axis reads (1, 2) along x and (10, 20) one row up;
    // a +90° z rotation turns its rows into columns.
    const values = new Float32Array([1, 2, 10, 20]);
    const out = alignVolume(values, [2, 2, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0, 1]);
    expect(out.dims).toEqual([2, 2, 1]);
    expect(Array.from(out.values)).toEqual([2, 20, 1, 10]);
  });
});

describe("parseSampleMeta", () => {
  it("reads_hu_mapping_and_ids", () => {
    expect(
      parseSampleMeta({
        ...DESCRIPTOR,
        hu: { slope: 8, intercept: -1000 },
        seriesUid: "9.9",
        description: "demo",
        iso: 300,
      }),
    ).toEqual({ slope: 8, intercept: -1000, seriesUid: "9.9", description: "demo", iso: 300 });
  });

  it("accepts_a_null_iso", () => {
    expect(
      parseSampleMeta({
        ...DESCRIPTOR,
        hu: { slope: 1, intercept: -1024 },
        seriesUid: "9.9",
        description: "demo",
        iso: null,
      }).iso,
    ).toBe(null);
  });

  it("rejects_garbage", () => {
    expect(() => parseSampleMeta(DESCRIPTOR)).toThrow();
    expect(() => parseSampleMeta({ ...DESCRIPTOR, hu: { slope: "8", intercept: -1000 } })).toThrow();
  });
});

describe("loadSample", () => {
  const TINY = {
    origin: [0, 0, 0],
    spacing: [1, 1, 1],
    extent: [0, 1, 0, 1, 0, 0],
    pointData: { arrays: [{ data: { dataType: "Uint8Array", size: 4, ref: { id: "blob.gz" } } }] },
    hu: { slope: 8, intercept: -1000 },
    seriesUid: "9.9",
    description: "tiny",
    iso: 300,
  };

  it("fetches_gunzips_and_slices", async () => {
    const seen: string[] = [];
    const fetchImpl = async (url: string): Promise<Response> => {
      seen.push(url);
      if (url.endsWith("index.json")) return new Response(JSON.stringify(TINY));
      return new Response(gzipSync(Buffer.from([0, 125, 200, 255])) as unknown as BodyInit);
    };
    const { slices, iso } = await loadSample("headsq", "/base/", fetchImpl as typeof fetch);
    expect(seen).toEqual(["/base/samples/headsq/index.json", "/base/samples/headsq/data/blob.gz"]);
    expect(iso).toBe(300);
    expect(slices.length).toBe(1);
    expect(slices[0].seriesUid).toBe("9.9");
    expect(Array.from(slices[0].pixels as Int16Array)).toEqual([-1000, 0, 600, 1040]);
  });

  it("fails_loudly_on_http_errors", async () => {
    const fetchImpl = async (): Promise<Response> => new Response("nope", { status: 404 });
    await expect(loadSample("lidc2", "/", fetchImpl as typeof fetch)).rejects.toThrow("404");
  });

  it("rejects_non_gzip_scalars", async () => {
    const fetchImpl = async (url: string): Promise<Response> => {
      if (url.endsWith("index.json")) return new Response(JSON.stringify(TINY));
      return new Response(new Uint8Array([1, 2, 3, 4]) as unknown as BodyInit);
    };
    await expect(loadSample("headsq", "/", fetchImpl as typeof fetch)).rejects.toThrow("gzip");
  });
});

describe("volumeToSlices", () => {
  it("stacks_rows_origin_and_hu", () => {
    const slices = volumeToSlices(new Float32Array([1, 2, 3, 4]), [2, 2, 1], [1, 1, 1], [5, 6, 7], {
      seriesUid: "9.9",
      description: "demo",
      slope: 8,
      intercept: -1000,
    });
    expect(slices.length).toBe(1);
    expect(slices[0].rows).toBe(2);
    expect(slices[0].columns).toBe(2);
    expect(slices[0].position).toEqual([5, 6, 7]);
    expect(slices[0].rescaleSlope).toBe(1);
    expect(Array.from(slices[0].pixels as Int16Array)).toEqual([-992, -984, -976, -968]);
  });
});
