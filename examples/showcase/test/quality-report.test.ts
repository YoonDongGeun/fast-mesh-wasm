// The quality report's builders (#257): the rows from the object info, the
// repair analysis and the shown map; the CSV at full precision in the display
// unit (a point split into x, y and z); the HTML with the panels' formatting,
// its sections, and the mesh name escaped.
import { describe, expect, it } from "vitest";
import type { AnalysisSummary } from "../src/features/mesh-repair/ux/repair-summary";
import type { MeshInfo } from "../src/features/inspect/ux/object-info";
import {
  qualityReportCsv,
  qualityReportHtml,
  type QualityReportInput,
  qualityReportRows,
  type ShownInspectMap,
} from "../src/features/inspect/ux/quality-report";
import { formatArea, formatLength, formatVolume } from "../src/shared/ux/units";

const INFO: MeshInfo = {
  triangles: 12,
  vertices: 8,
  edges: 18,
  components: 1,
  area: 24,
  volume: 8,
  volumeReason: null,
  meanEdge: 2.5,
  world: { min: [-1, -1, -1], max: [1, 1, 1] },
  local: { min: [0, 0, 0], max: [2, 2, 2] },
};

const SUMMARY: AnalysisSummary = {
  holes: [{ id: { minVertex: 0, len: 4, hash: 1n }, edges: 4, perimeter: 25.4 }],
  openChains: 0,
  orientationConflicts: 0,
  components: [
    { triangles: 10, closed: false },
    { triangles: 2, closed: true },
  ],
  nonManifoldEdges: 1,
  nonManifoldVertices: 0,
  pinchVertices: 0,
  degenerate: 2,
  duplicate: 0,
  cavities: 0,
  clean: false,
};

const DEVIATION: ShownInspectMap = {
  kind: "deviation",
  mesh: "scan",
  reference: "design",
  readout: { min: 0, max: 0.5, mean: 0.1, rms: 0.2, p95: 0.4, samples: 8 },
  params: { signed: false },
  range: [0, 0.5],
  band: null,
};

const generated = new Date("2026-11-23T10:00:00Z");

function input(patch: Partial<QualityReportInput> = {}): QualityReportInput {
  return { name: "part", info: INFO, repair: null, map: null, generated, ...patch };
}

/** The CSV as `[section, quantity, value, unit]` rows (no quoted commas in these fixtures). */
function csvRows(text: string): string[][] {
  return text.trim().split("\n").slice(1).map((l) => l.split(","));
}

describe("quality report rows", () => {
  it("lists the object info, then the repair analysis and the map when there are", () => {
    const only = qualityReportRows(input());
    expect(new Set(only.map((r) => r.section))).toEqual(new Set(["Object"]));
    expect(only.find((r) => r.quantity === "Edges")?.value).toBe(18);
    expect(only.find((r) => r.quantity === "World box size")?.value).toEqual([2, 2, 2]);

    const all = qualityReportRows(input({ repair: { summary: SUMMARY, selfIntersects: null }, map: DEVIATION }));
    const rep = all.filter((r) => r.section === "Repair analysis");
    expect(rep.find((r) => r.quantity === "Holes")?.value).toBe(1);
    expect(rep.find((r) => r.quantity === "Largest hole perimeter")?.value).toBe(25.4);
    expect(rep.find((r) => r.quantity === "Open components")?.value).toBe(1);
    expect(rep.find((r) => r.quantity === "Self-intersections")?.value).toBe("not checked");
    const map = all.filter((r) => r.section === "Inspect map");
    expect(map.find((r) => r.quantity === "Reference mesh")?.value).toBe("design");
    expect(map.find((r) => r.quantity === "RMS deviation")).toMatchObject({ value: 0.2, kind: "length" });
  });

  it("labels each map kind with its own readouts and units", () => {
    const overhang = qualityReportRows(
      input({
        map: {
          ...DEVIATION,
          kind: "overhang",
          reference: null,
          readout: { area: 3, faces: 2, maxAngleDeg: 80 },
          params: { criticalAngleDeg: 45, direction: [0, 0, 1] },
        },
      }),
    ).filter((r) => r.section === "Inspect map");
    expect(overhang.find((r) => r.quantity === "Overhang area")).toMatchObject({ value: 3, kind: "area" });
    expect(overhang.find((r) => r.quantity === "Critical angle")).toMatchObject({ value: 45, kind: "degrees" });
    expect(overhang.find((r) => r.quantity === "Build direction")?.value).toBe("(0, 0, 1)");

    const quality = qualityReportRows(
      input({
        map: {
          ...DEVIATION,
          kind: "quality",
          reference: null,
          readout: { past: 3, total: 12, min: 1.2, max: 30, mean: 4 },
          params: { metric: "minAngleDeg", threshold: 10 },
        },
      }),
    ).filter((r) => r.section === "Inspect map");
    expect(quality.find((r) => r.quantity === "Metric")?.value).toBe("Minimum angle");
    expect(quality.find((r) => r.quantity === "Triangles below the threshold")?.value).toBe(3);
    expect(quality.find((r) => r.quantity === "Maximum")).toMatchObject({ value: 30, kind: "degrees" });
  });
});

describe("quality report CSV", () => {
  it("writes every length in the display unit at full precision, points split into x, y and z", () => {
    const csv = qualityReportCsv(input({ repair: { summary: SUMMARY, selfIntersects: false }, map: DEVIATION }), "in");
    expect(csv.split("\n")[0]).toBe("section,quantity,value,unit");
    const rows = csvRows(csv);
    const find = (q: string) => rows.find((r) => r[1] === q);
    expect(find("Mesh")).toEqual(["Report", "Mesh", "part", ""]);
    expect(find("Length unit")).toEqual(["Report", "Length unit", "in", ""]);
    expect(find("Triangles")).toEqual(["Object", "Triangles", "12", ""]);
    expect(find("Surface area")).toEqual(["Object", "Surface area", String(24 / 25.4 ** 2), "in^2"]);
    expect(find("Volume")).toEqual(["Object", "Volume", String(8 / 25.4 ** 3), "in^3"]);
    expect(find("Average edge length")).toEqual(["Object", "Average edge length", String(2.5 / 25.4), "in"]);
    expect(find("Local box max x")).toEqual(["Object", "Local box max x", String(2 / 25.4), "in"]);
    expect(find("Local box max z")).toBeDefined();
    expect(find("Largest hole perimeter")).toEqual(["Repair analysis", "Largest hole perimeter", "1", "in"]);
    expect(find("Self-intersections")?.[2]).toBe("no");
    expect(find("Maximum deviation")).toEqual(["Inspect map", "Maximum deviation", String(0.5 / 25.4), "in"]);

    const mm = csvRows(qualityReportCsv(input(), "mm"));
    expect(mm.find((r) => r[1] === "Surface area")).toEqual(["Object", "Surface area", "24", "mm^2"]);
  });

  it("says why an open mesh has no volume, and quotes a name with a comma", () => {
    const open = qualityReportCsv(input({ name: 'a, "b"', info: { ...INFO, volume: null, volumeReason: "open" } }), "mm");
    expect(open).toContain('Report,Mesh,"a, ""b""",');
    expect(open).toContain("Object,Volume,open mesh (no volume),");
  });
});

describe("quality report HTML", () => {
  it("is a standalone page with one table per section, formatted like the panels", () => {
    const html = qualityReportHtml(input({ repair: { summary: SUMMARY, selfIntersects: true }, map: DEVIATION }), "mm");
    expect(html.startsWith("<!doctype html>")).toBe(true);
    expect(html).toContain("<title>Quality report: part</title>");
    for (const h of ["Object", "Repair analysis", "Inspect map"]) {
      expect(html).toContain(`<h2>${h}</h2>`);
    }
    expect(html).toContain(`<th scope="row">Surface area</th><td>${formatArea(24, "mm")}</td>`);
    expect(html).toContain(`<th scope="row">Volume</th><td>${formatVolume(8, "mm")}</td>`);
    expect(html).toContain(`<th scope="row">World box min</th><td>(-1, -1, -1) mm</td>`);
    expect(html).toContain(`<th scope="row">Self-intersections</th><td>yes</td>`);
    expect(html).toContain(`<th scope="row">Maximum deviation</th><td>${formatLength(0.5, "mm")}</td>`);
    expect(html).toContain("2026-11-23T10:00:00.000Z");
  });

  it("follows the unit and escapes the mesh name", () => {
    const html = qualityReportHtml(input({ name: "<b>&co</b>" }), "in");
    expect(html).toContain("<title>Quality report: &lt;b&gt;&amp;co&lt;/b&gt;</title>");
    expect(html).not.toContain("<b>&co</b>");
    expect(html).toContain(`<td>${formatArea(24, "in")}</td>`);
    expect(html).toContain("Length unit: in");
    expect(html).not.toContain("<h2>Repair analysis</h2>");
  });
});
