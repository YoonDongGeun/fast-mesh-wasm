// Text on a surface (#260): Place Mesh with the text as its stamp embosses
// and engraves it on a curved mesh (a sphere), following the curve, as one
// undo step each; and the words the Surface Stamp section shows for it,
// and arming the tool with a text.
import { generatePrimitive, Scene, type Vec3 } from "fast-mesh-wasm";
import { createSceneStore } from "fast-mesh-wasm/store";
import * as THREE from "three";
import { afterEach, describe, expect, it } from "vitest";
import { AppState } from "../src/app/state/app-state";
import { createDefaultWave6State, type Wave6PrintPrepState } from "../src/app/state/wave6-state";
import { executePlaceStampAtHit } from "../src/features/mesh-edit/ux/print-prep-ops";
import { orientationFromSurfaceNormal } from "../src/features/mesh-edit/ux/surface-orientation";
import { type Contour, loadTextFont } from "../src/features/mesh-edit/ux/text-font";
import {
  DEFAULT_TEXT_PARAMS,
  glyphRegions,
  layoutText,
  type TextParams,
} from "../src/features/mesh-edit/ux/text-mesh";
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

/** Whether `(x, y)` is inside the closed polyline `c`. */
function inside(c: Contour, x: number, y: number): boolean {
  let hit = false;
  for (let i = 0, j = c.length - 2; i < c.length; j = i, i += 2) {
    if (c[i + 1] > y !== c[j + 1] > y && x < ((c[j] - c[i]) * (y - c[i + 1])) / (c[j + 1] - c[i + 1]) + c[i]) {
      hit = !hit;
    }
  }
  return hit;
}

/**
 * Places `text` on a sphere of `radius` at `DIR` (its outside, or with
 * `concave` the inside of a thick shell) and measures the relief inside the
 * letters, away from their edges: how far the surface then lies from the
 * sphere, along the placement normal, at a grid over each letter. A letter
 * that only its outline follows the curve with would dip in the middle.
 */
async function reliefInsideLetters(args: {
  radius: number;
  concave: boolean;
  mode: "embedUnion" | "engraveSubtract";
  text: TextParams;
  embedDepth: number;
}): Promise<{ relief: number[]; closed: boolean; orientationConflicts: number }> {
  const { radius, concave, mode, text, embedDepth } = args;
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  const ball = generatePrimitive({ kind: "uvSphere", radius, segments: 192, rings: 96 });
  let positions = Float64Array.from(ball.positions);
  let indices = ball.indices;
  if (concave) {
    // A thick shell: the ball turned inside out, in a ball half as large again.
    const n = ball.positions.length;
    const t = ball.indices.length;
    positions = new Float64Array(2 * n);
    indices = new Uint32Array(2 * t);
    for (let i = 0; i < n; i++) {
      positions[i] = ball.positions[i];
      positions[n + i] = 1.5 * ball.positions[i];
    }
    for (let i = 0; i < t; i += 3) {
      indices.set([ball.indices[i], ball.indices[i + 2], ball.indices[i + 1]], i);
      indices.set([ball.indices[i] + n / 3, ball.indices[i + 1] + n / 3, ball.indices[i + 2] + n / 3], t + i);
    }
  }
  const item = app.addMesh(store, {
    name: "Curved",
    positions,
    indices,
    center: [0, 0, 0],
    radius,
    volumeId: null,
    preset: null,
  });
  const sign = concave ? -1 : 1;
  const normal: Vec3 = [sign * DIR[0], sign * DIR[1], sign * DIR[2]];
  const hit: Vec3 = [DIR[0] * radius, DIR[1] * radius, DIR[2] * radius];
  await executePlaceStampAtHit({
    store,
    app,
    target: item.mesh,
    hitPoint: hit,
    hitNormal: normal,
    placeParams: placeParams({ mode, text, embedDepth }),
  });
  const measured = scene.inspect.measure(item.mesh);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(scene.views(item.mesh).positions()), 3));
  geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(scene.views(item.mesh).indices()), 1));
  const placed = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
  const q = new THREE.Quaternion(...orientationFromSurfaceNormal(normal, 0).quaternion);
  const n = new THREE.Vector3(...normal);
  const x = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
  const y = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
  const ray = new THREE.Raycaster();
  const relief: number[] = [];
  const GRID = 10;
  for (const g of layoutText(await loadTextFont(), text).glyphs) {
    for (const r of glyphRegions(g.contours)) {
      const xs = r.outer.filter((_, i) => i % 2 === 0);
      const ys = r.outer.filter((_, i) => i % 2 === 1);
      const solid = (u: number, v: number) => inside(r.outer, u, v) && !r.holes.some((h) => inside(h, u, v));
      for (let a = 0; a < GRID; a++) {
        for (let b = 0; b < GRID; b++) {
          const u = Math.min(...xs) + ((Math.max(...xs) - Math.min(...xs)) * (a + 0.5)) / GRID;
          const v = Math.min(...ys) + ((Math.max(...ys) - Math.min(...ys)) * (b + 0.5)) / GRID;
          const e = 0.02 * text.size;
          if (!solid(u, v) || !solid(u + e, v) || !solid(u - e, v) || !solid(u, v + e) || !solid(u, v - e)) {
            continue;
          }
          const at = new THREE.Vector3(...hit).addScaledVector(x, u).addScaledVector(y, v);
          ray.set(at.clone().addScaledVector(n, concave ? 0.9 * radius : radius), n.clone().negate());
          const [first] = ray.intersectObject(placed);
          const sphere = sign * (Math.sqrt(radius * radius - u * u - v * v) - radius);
          relief.push(first.point.sub(at).dot(n) - sphere);
        }
      }
    }
  }
  geometry.dispose();
  return { relief, closed: measured.closed, orientationConflicts: measured.orientationConflicts };
}

describe("text following a curve", () => {
  // Letters 14 mm tall and a third of the ball's radius: far too large to
  // follow it with their outline alone (the faces would cut 0.6 mm chords).
  const text: TextParams = { ...DEFAULT_TEXT_PARAMS, text: "AB8 R2", size: 14, depth: 1.5 };
  for (const concave of [false, true]) {
    for (const mode of ["embedUnion", "engraveSubtract"] as const) {
      it(`${mode === "embedUnion" ? "raises" : "cuts"} whole letters evenly on a ${concave ? "concave" : "convex"} surface`, async () => {
        const { relief, closed, orientationConflicts } = await reliefInsideLetters({
          radius: 40,
          concave,
          mode,
          text,
          embedDepth: 0.5,
        });
        expect(closed).toBe(true);
        expect(orientationConflicts).toBe(0);
        expect(relief.length).toBeGreaterThan(150);
        const want = mode === "embedUnion" ? 1 : -0.5;
        for (const r of relief) {
          expect(Math.abs(r - want)).toBeLessThan(0.05);
        }
      });
    }
  }
});

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
