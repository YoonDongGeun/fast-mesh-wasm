// @vitest-environment jsdom
// The panels' mesh-making buttons are one undo step each (#271): Extract and
// Split faces, Reconstruct surface and Join, with undo taking the new row out
// of the list (or bringing the joined one back) and redo putting it back; and
// the collision rules of a mesh that left and came back.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, type ReactElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => await import("../../../packages/fast-mesh-wasm/dist/web.js"));

import { init, Scene } from "fast-mesh-wasm";
import { AppState, type Item } from "../src/app/state/app-state";
import { CollisionPanel } from "../src/features/collision/ui/CollisionPanel";
import { GroupsBar } from "../src/features/collision/ui/GroupsBar";
import { RepairPanel } from "../src/features/mesh-repair/ui/RepairPanel";
import { PointsPanel } from "../src/features/points/ui/PointsPanel";
import { createSyntheticSphereCloud } from "../src/features/points/ux/point-cloud";
import { SelectPanel } from "../src/features/select/ui/SelectPanel";
import { writeMeshFaceSelection } from "../src/features/select/ux/face-selection-ops";
import { countLiveTriangles } from "../src/features/select/ux/face-sets";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => void)[] = [];
afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

interface Harness {
  scene: Scene;
  store: SceneStore;
  app: AppState;
}

function open(): Harness {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const off = app.follow(store);
  cleanups.push(() => {
    off();
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

function mount({ store }: Harness, element: ReactElement): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<SceneProvider store={store}>{element}</SceneProvider>));
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
  });
  return host;
}

function add(h: Harness, name: string, positions: number[], indices: number[]): Item {
  return h.app.addMesh(h.store, {
    name,
    positions: Float64Array.from(positions),
    indices: Uint32Array.from(indices),
    center: [0, 0, 0],
    radius: 1,
    volumeId: null,
    preset: null,
  });
}

function addBox(h: Harness, name: string, x = 0): Item {
  const p: number[] = [];
  for (let i = 0; i < 8; i++) {
    p.push(x + (i & 1), (i >> 1) & 1, i >> 2);
  }
  return add(h, name, p, [
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
  ]);
}

const live = (h: Harness, item: Item) => countLiveTriangles(h.store.scene.views(item.mesh).indices());
const names = (h: Harness) => h.app.getSnapshot().items.map((i) => i.name);

async function frame(h: Harness): Promise<void> {
  await act(async () => {
    await new Promise((done) => setTimeout(done, 0));
    h.store.commitFrame();
  });
}

async function click(h: Harness, el: Element | null): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => el!.dispatchEvent(new MouseEvent("click", { bubbles: true })));
  await frame(h);
}

async function history(h: Harness, step: "undo" | "redo"): Promise<void> {
  await act(async () => h.store[step]());
  await frame(h);
}

async function pick(h: Harness, el: HTMLSelectElement | null, value: string): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")!.set!.call(el, value);
    el!.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await frame(h);
}

describe("the panels' mesh-making buttons", () => {
  it("Extract and Split faces: undo removes the new row, redo brings it back", async () => {
    const h = open();
    const box = addBox(h, "Box");
    h.store.select(box.mesh);
    const host = mount(h, <SelectPanel app={h.app} store={h.store} />);
    for (const [action, name, left] of [
      ["extract", "Box (Extract)", 12],
      ["split", "Box (Split)", 10],
    ] as const) {
      act(() => writeMeshFaceSelection(h.app, box.id, Uint32Array.of(0, 1)));
      await click(h, host.querySelector(`[data-select-action="${action}"]`));
      expect(names(h)).toEqual(["Box", name]);
      const made = h.app.getSnapshot().items[1];
      expect(live(h, box)).toBe(left);
      await history(h, "undo");
      expect(h.app.getSnapshot().items).toEqual([box]);
      expect(live(h, box)).toBe(12);
      expect(h.scene.canUndo()).toBe(false);
      await history(h, "redo");
      expect(h.app.getSnapshot().items).toEqual([box, made]);
      expect(h.app.getSnapshot().items[1]).toBe(made);
      expect([live(h, box), live(h, made)]).toEqual([left, 2]);
      await history(h, "undo");
    }
  });

  it("Reconstruct surface: undo removes the new row, redo brings it back", async () => {
    const h = open();
    h.app.updateWave6((prev) => ({
      ...prev,
      metrology: { ...prev.metrology, pointCloud: createSyntheticSphereCloud(200, 1) },
    }));
    const host = mount(h, <PointsPanel app={h.app} store={h.store} />);
    await click(h, host.querySelector('[data-action="points-reconstruct"]'));
    await vi.waitFor(() => expect(h.app.getSnapshot().items).toHaveLength(1), { timeout: 5000 });
    await frame(h);
    const [made] = h.app.getSnapshot().items;
    const triangles = live(h, made);
    expect(triangles).toBeGreaterThan(0);
    expect(h.store.getSnapshot().selection).toBe(made.mesh);
    await history(h, "undo");
    expect(h.app.getSnapshot().items).toEqual([]);
    expect(h.scene.canUndo()).toBe(false);
    await history(h, "redo");
    expect(h.app.getSnapshot().items).toEqual([made]);
    expect(h.app.getSnapshot().items[0]).toBe(made);
    expect(live(h, made)).toBe(triangles);
    expect(h.store.getSnapshot().selection).toBe(made.mesh);
  });

  it("Join: undo brings the source's row back, redo takes it again", async () => {
    const h = open();
    const a = add(h, "A", [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], [0, 2, 1, 0, 3, 2]);
    const source = add(h, "Source", [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], [0, 1, 2, 0, 2, 3]);
    h.store.select(a.mesh);
    // The adds are reported before the panel listens: it analyses once.
    h.store.commitFrame();
    const host = mount(h, <RepairPanel app={h.app} scene={h.scene} />);
    const loop = async (side: "a" | "b") => {
      const select = () => host.querySelector<HTMLSelectElement>(`[data-join-loop="${side}"]`);
      await vi.waitFor(async () => {
        await frame(h);
        expect(select()?.options.length).toBeGreaterThan(1);
      }, { timeout: 5000 });
      await pick(h, select(), "0");
    };
    await pick(h, host.querySelector("[data-join-source]"), String(source.id));
    await loop("a");
    await loop("b");
    const join = host.querySelector<HTMLButtonElement>('[data-action="join"]')!;
    expect(join.disabled).toBe(false);
    await click(h, join);
    const report = () => host.querySelector("[data-repair-report]")?.textContent ?? "";
    await vi.waitFor(() => expect(report()).toContain("Source removed (undo brings it back)"), { timeout: 5000 });
    expect(h.app.getSnapshot().items).toEqual([a]);
    expect(live(h, a)).toBe(12);
    await history(h, "undo");
    expect(h.app.getSnapshot().items).toEqual([a, source]);
    expect([live(h, a), live(h, source)]).toEqual([2, 2]);
    expect(h.scene.canUndo()).toBe(false);
    expect([...host.querySelectorAll("[data-join-source] option")].map((o) => o.textContent)).toContain("Source");
    await history(h, "redo");
    expect(h.app.getSnapshot().items).toEqual([a]);
    expect(live(h, a)).toBe(12);
  });
});

describe("collision rules of a mesh that leaves and comes back", () => {
  const mode = (h: Harness, a: Item, b: Item) => h.scene.collisionRule(a.mesh, b.mesh).mode;

  it("a pair override comes back with its mesh, as the panel has it by then", async () => {
    const h = open();
    const [a, b] = [addBox(h, "A"), addBox(h, "B", 3)];
    const host = mount(h, <CollisionPanel app={h.app} scene={h.scene} />);
    await frame(h);
    const select = () => host.querySelector<HTMLSelectElement>("[data-collision-pair-mode]");
    await pick(h, select(), "rigid");
    expect(mode(h, a, b)).toBe("rigid");
    await act(async () => h.app.removeMesh(h.store, b));
    await frame(h);
    // The default response changes while it is out: no rule can be set then.
    await act(async () =>
      h.app.updateWave6((prev) => ({ ...prev, collision: { ...prev.collision, rigidSlide: !prev.collision.rigidSlide } })),
    );
    await frame(h);
    const { rigidSlide } = h.app.getSnapshot().wave6.collision;
    await history(h, "undo");
    expect(h.app.getSnapshot().items).toEqual([a, b]);
    expect(h.scene.collisionRule(a.mesh, b.mesh)).toMatchObject({ mode: "rigid", rigidSlide });
    expect(select()?.value).toBe("rigid");
    // Gone for good: its override goes.
    await act(async () => h.app.removeMesh(h.store, b, { undoable: false }));
    await frame(h);
    expect(h.store.isRestorable(b.mesh)).toBe(false);
  });

  it("a group's rule comes back with its meshes, removed one by one or with the group's button", async () => {
    const h = open();
    const [a, b] = [addBox(h, "A"), addBox(h, "B", 3)];
    const host = mount(h, <GroupsBar app={h.app} />);
    act(() => {
      h.app.toggleCheck(a.mesh, true);
      h.app.toggleCheck(b.mesh, true);
    });
    await click(h, host.querySelector('[data-action="group-checked"]'));
    const select = () => host.querySelector<HTMLSelectElement>('[data-group-collision="Group 1"]');
    await pick(h, select(), "rigid");
    expect(mode(h, a, b)).toBe("rigid");

    await act(async () => h.app.removeMesh(h.store, b));
    await frame(h);
    expect(h.app.getSnapshot().meshGroups[0].members).toEqual([a.mesh]);
    await act(async () =>
      h.app.updateWave6((prev) => ({ ...prev, collision: { ...prev.collision, rigidSlide: !prev.collision.rigidSlide } })),
    );
    await frame(h);
    const { rigidSlide } = h.app.getSnapshot().wave6.collision;
    await history(h, "undo");
    expect(h.scene.collisionRule(a.mesh, b.mesh)).toMatchObject({ mode: "rigid", rigidSlide });
    expect(select()?.value).toBe("rigid");

    const remove = [...host.querySelectorAll("button")].find((el) => el.textContent?.trim() === "Remove meshes");
    await click(h, remove ?? null);
    expect(h.app.getSnapshot().items).toEqual([]);
    await history(h, "undo");
    expect(h.app.getSnapshot().items).toEqual([a, b]);
    expect(h.scene.canUndo()).toBe(false);
    expect(mode(h, a, b)).toBe("rigid");
    expect(select()?.value).toBe("rigid");
    // A mesh in another's place in the group is another pair.
    await act(async () => h.app.removeMesh(h.store, b, { undoable: false }));
    expect(h.app.getSnapshot().meshGroups[0].members).toEqual([a.mesh]);
  });
});
