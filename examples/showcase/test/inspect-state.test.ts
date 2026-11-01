// The inspect tools' app state: arming, ruler endpoints, query hits, the
// section plane and the clip box.
import type { MeshHandle, SectionHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { AppState } from "../src/scene/app-state";

const mesh = 7 as unknown as MeshHandle;
const handle = {} as SectionHandle;

describe("inspect state", () => {
  it("starts_disarmed", () => {
    const snap = new AppState().getSnapshot();
    expect(snap.inspectTool).toBe(null);
    expect(snap.ruler).toBe(null);
    expect(snap.inspectQuery).toBe(null);
    expect(snap.inspectSection).toBe(null);
    expect(snap.clipBox).toBe(null);
  });

  it("arms_and_disarms_tools", () => {
    const app = new AppState();
    app.setInspectTool("ruler");
    expect(app.getSnapshot().inspectTool).toBe("ruler");
    app.setInspectTool(null);
    expect(app.getSnapshot().inspectTool).toBe(null);
  });

  it("holds_ruler_endpoints", () => {
    const app = new AppState();
    app.setRuler({ a: [0, 0, 0], b: null });
    expect(app.getSnapshot().ruler).toEqual({ a: [0, 0, 0], b: null });
    app.setRuler({ a: [0, 0, 0], b: [1, 0, 0] });
    expect(app.getSnapshot().ruler?.b).toEqual([1, 0, 0]);
    app.setRuler(null);
    expect(app.getSnapshot().ruler).toBe(null);
  });

  it("holds_query_section_and_clip", () => {
    const app = new AppState();
    app.setInspectQuery({ mesh, slot: 3, point: [1, 2, 3] });
    expect(app.getSnapshot().inspectQuery).toEqual({ mesh, slot: 3, point: [1, 2, 3] });
    const plane = { origin: [0, 0, 0] as [number, number, number], normal: [0, 0, 1] as [number, number, number] };
    app.setInspectSection({ handle, plane });
    expect(app.getSnapshot().inspectSection?.handle).toBe(handle);
    app.setClipBox({ min: [0, 0, 0], max: [1, 1, 1] });
    expect(app.getSnapshot().clipBox?.max).toEqual([1, 1, 1]);
    app.setInspectSection(null);
    app.setClipBox(null);
    app.setInspectQuery(null);
    expect(app.getSnapshot().inspectSection).toBe(null);
    expect(app.getSnapshot().clipBox).toBe(null);
    expect(app.getSnapshot().inspectQuery).toBe(null);
  });
});
