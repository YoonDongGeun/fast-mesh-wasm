// Text on a surface (#260): Place Mesh with the text as its stamp embosses
// and engraves it on a curved mesh (a sphere), following the curve, as one
// undo step each; and the words the Surface Stamp section shows for it,
// and arming the tool with a text.
import { generatePrimitive, Scene, type Vec3 } from "fast-mesh-wasm";
import { createSceneStore } from "fast-mesh-wasm/store";
import { afterEach, describe, expect, it } from "vitest";
import { AppState } from "../src/app/state/app-state";
import { createDefaultWave6State, type Wave6PrintPrepState } from "../src/app/state/wave6-state";
import { executePlaceStampAtHit } from "../src/features/mesh-edit/ux/print-prep-ops";
import { DEFAULT_TEXT_PARAMS, type TextParams } from "../src/features/mesh-edit/ux/text-mesh";
import { armTextStamp, textStampEffect, textStampGeometry } from "../src/features/mesh-edit/ux/text-stamp";

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const c of cleanups.splice(0).reverse()) c();
});

const RADIUS = 20;

/** A sphere of radius 20 to write on, fine enough that a 1 mm relief stands out of its facets. */
function sphereScene() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  const sphere = generatePrimitive({ kind: "uvSphere", radius: RADIUS, segments: 128, rings: 64 });
  const item = app.addMesh(store, {
    name: "Ball",
    positions: Float64Array.from(sphere.positions),
    indices: sphere.indices,
    center: [0, 0, 0],
    radius: RADIUS,
    volumeId: null,
    preset: null,
  });
  return { scene, store, app, item };
}

const TEXT: TextParams = { ...DEFAULT_TEXT_PARAMS, text: "AB8", size: 6, depth: 1.5 };

function placeParams(patch: Partial<Wave6PrintPrepState["placeParams"]>): Wave6PrintPrepState["placeParams"] {
  return {
    ...createDefaultWave6State().printPrep.placeParams,
    stampSource: "text",
    text: TEXT,
    embedDepth: 0.5,
    conformToSurface: true,
    ...patch,
  };
}

/** World distances of the mesh's vertices from the sphere's centre. */
function radii(scene: Scene, mesh: Parameters<Scene["views"]>[0]): number[] {
  const p = scene.views(mesh).positions();
  const out: number[] = [];
  for (let i = 0; i < p.length; i += 3) out.push(Math.hypot(p[i], p[i + 1], p[i + 2]));
  return out;
}

/** Triangles in use (a removed one reads 0, 0, 0 in the view). */
function liveTriangles(scene: Scene, mesh: Parameters<Scene["views"]>[0]): number {
  const idx = scene.views(mesh).indices();
  let n = 0;
  for (let t = 0; t < idx.length; t += 3) if (idx[t] !== idx[t + 1] || idx[t] !== idx[t + 2]) n++;
  return n;
}

/** Where on the sphere to write: off its poles and seams, the outward normal there. */
const DIR: Vec3 = (() => {
  const d = [0.6, 0.48, 0.64];
  const l = Math.hypot(...d);
  return [d[0] / l, d[1] / l, d[2] / l];
})();
const HIT: Vec3 = [DIR[0] * RADIUS, DIR[1] * RADIUS, DIR[2] * RADIUS];

describe("text on a curved surface", () => {
  it("embosses: raised by depth less embed along the curve, closed, one undo step", async () => {
    const { scene, store, app, item } = sphereScene();
    const before = scene.inspect.measure(item.mesh);
    const trianglesBefore = liveTriangles(scene, item.mesh);
    const res = await executePlaceStampAtHit({
      store,
      app,
      target: item.mesh,
      targetName: item.name,
      hitPoint: HIT,
      hitNormal: DIR,
      placeParams: placeParams({ mode: "embedUnion" }),
    });
    expect(res.added).toBeNull();
    expect(res.afterTriangles).toBeGreaterThan(res.beforeTriangles);
    const after = scene.inspect.measure(item.mesh);
    expect(after.closed).toBe(true);
    expect(after.orientationConflicts).toBe(0);
    expect(after.volume!).toBeGreaterThan(before.volume!);
    // The letters stand 1 mm proud of the surface all along the line: with
    // the stamp conformed, its top follows the sphere at radius 21.
    const top = Math.max(...radii(scene, item.mesh));
    expect(top).toBeGreaterThan(RADIUS + 0.95);
    expect(top).toBeLessThan(RADIUS + 1.05);
    // The temporary stamp is gone; only the ball is listed.
    expect(store.getSnapshot().meshes.size).toBe(1);
    expect(app.getSnapshot().items.map((i) => i.name)).toEqual(["Ball"]);

    expect(scene.canUndo()).toBe(true);
    store.undo();
    expect(scene.canUndo()).toBe(false);
    expect(liveTriangles(scene, item.mesh)).toBe(trianglesBefore);
    expect(scene.inspect.measure(item.mesh).volume).toBeCloseTo(before.volume!, 6);
    store.redo();
    expect(scene.inspect.measure(item.mesh).volume).toBeCloseTo(after.volume!, 6);
  });

  it("engraves: cut as deep as the embed, closed, one undo step", async () => {
    const { scene, store, app, item } = sphereScene();
    const before = scene.inspect.measure(item.mesh);
    await executePlaceStampAtHit({
      store,
      app,
      target: item.mesh,
      targetName: item.name,
      hitPoint: HIT,
      hitNormal: DIR,
      placeParams: placeParams({ mode: "engraveSubtract", embedDepth: 0.8 }),
    });
    const after = scene.inspect.measure(item.mesh);
    expect(after.closed).toBe(true);
    expect(after.orientationConflicts).toBe(0);
    expect(after.volume!).toBeLessThan(before.volume!);
    // The cut's floor lies 0.8 under the surface, and nothing stands out.
    const r = radii(scene, item.mesh);
    expect(Math.min(...r)).toBeGreaterThan(RADIUS - 0.85);
    expect(Math.min(...r)).toBeLessThan(RADIUS - 0.75);
    expect(Math.max(...r)).toBeLessThanOrEqual(RADIUS + 1e-3);
    store.undo();
    expect(scene.canUndo()).toBe(false);
    expect(scene.inspect.measure(item.mesh).volume).toBeCloseTo(before.volume!, 6);
  });

  it("rejects a text it cannot build before touching the scene", async () => {
    const { scene, store, app, item } = sphereScene();
    await expect(
      executePlaceStampAtHit({
        store,
        app,
        target: item.mesh,
        hitPoint: HIT,
        hitNormal: DIR,
        placeParams: placeParams({ text: { ...TEXT, text: "  " } }),
      }),
    ).rejects.toThrow(/Type some text/);
    expect(scene.canUndo()).toBe(false);
    expect(store.getSnapshot().meshes.size).toBe(1);
  });
});

describe("the text stamp", () => {
  it("stands on its base at z = 0, centred on the picked point", async () => {
    const g = await textStampGeometry(TEXT);
    let minZ = Infinity;
    let maxZ = -Infinity;
    let minX = Infinity;
    let maxX = -Infinity;
    for (let i = 0; i < g.positions.length; i += 3) {
      minX = Math.min(minX, g.positions[i]);
      maxX = Math.max(maxX, g.positions[i]);
      minZ = Math.min(minZ, g.positions[i + 2]);
      maxZ = Math.max(maxZ, g.positions[i + 2]);
    }
    expect([minZ, maxZ]).toEqual([0, 1.5]);
    expect(minX + maxX).toBeCloseTo(0, 4);
  });

  it("says what it will do to the surface", () => {
    const base = placeParams({ embedDepth: 0.5, uniformScale: 1 });
    expect(textStampEffect({ ...base, mode: "embedUnion" }, "mm")).toEqual({
      text: "Raises the text 1 mm out of the surface.",
      problem: false,
    });
    expect(textStampEffect({ ...base, mode: "engraveSubtract" }, "mm")?.text).toMatch(/Cuts the text 0\.5 mm into/);
    expect(textStampEffect({ ...base, mode: "engraveSubtract", embedDepth: 0 }, "mm")?.problem).toBe(true);
    expect(textStampEffect({ ...base, mode: "embedUnion", embedDepth: 2 }, "mm")?.problem).toBe(true);
    expect(textStampEffect({ ...base, mode: "engraveSubtract", embedDepth: 2 }, "mm")?.problem).toBe(true);
    // Scale scales the depth too.
    expect(textStampEffect({ ...base, mode: "embedUnion", uniformScale: 2 }, "mm")?.text).toMatch(/2\.5 mm/);
    expect(textStampEffect({ ...base, mode: "keepSeparate" }, "mm")).toBeNull();
  });

  it("arms Place Mesh with the text, conforming, on the Mesh Edit tab", () => {
    const app = new AppState();
    app.updateWave6((prev) => ({ ...prev, select: { ...prev.select, activeTool: "brush" } }));
    armTextStamp(app, { ...TEXT, text: "Hi" }, "engraveSubtract");
    const snap = app.getSnapshot();
    expect(snap.activeTab).toBe("edit");
    expect(snap.wave6.select.activeTool).toBeNull();
    expect(snap.wave6.printPrep.activeTool).toBe("placeStamp");
    expect(snap.wave6.printPrep.placeParams).toMatchObject({
      stampSource: "text",
      mode: "engraveSubtract",
      conformToSurface: true,
      text: { text: "Hi", size: 6 },
    });
  });
});
