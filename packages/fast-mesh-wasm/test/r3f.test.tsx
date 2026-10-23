// React Three Fiber `<FastMesh>` and its frame driver (`fast-mesh-wasm/r3f`)
// over a local `SceneStore`, rendered headless with @react-three/test-renderer
// in vitest's default Node environment.
//
// The test renderer's `advanceFrames(n)` calls every `useFrame` subscriber
// synchronously with one unchanged root state, subscriber by subscriber
// (all n frames of the first, then all n of the second, ...), so these tests
// also pin that the driver's dedupe needs no clock or frame stamp.
import { existsSync, readFileSync } from "node:fs";
import { type RootState, useThree } from "@react-three/fiber";
import ReactThreeTestRenderer from "@react-three/test-renderer";
import * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FastMesh, useFrameDriver } from "../src/r3f/index.js";
import { SceneProvider } from "../src/react/index.js";
import { createSceneStore, type DistanceColoring, meshState, type SceneStore } from "../src/store/index.js";
import { listenerCount } from "../src/store/store.js";
import { bindSection, type MeshBinding } from "../src/three/index.js";
import type { BoundMesh } from "../src/three/mesh.js";
import { type MeshHandle, Scene } from "../src/node.js";

// Every `BoundMesh` that `<FastMesh>` creates, with a spy on `dispose`.
const bounds = vi.hoisted(() => [] as BoundMesh[]);
vi.mock("../src/three/mesh.js", async (importOriginal) => {
  const real = await importOriginal<typeof import("../src/three/mesh.js")>();
  return {
    ...real,
    bindMesh: (...args: Parameters<typeof real.bindMesh>) => {
      const bound = real.bindMesh(...args);
      vi.spyOn(bound, "dispose");
      bounds.push(bound);
      return bound;
    },
  };
});

const { create, act } = ReactThreeTestRenderer;

// React's act() environment flag, so state updates inside act() are not warned about.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** A closed tetrahedron offset by `dx` along x. */
function tetra(dx = 0) {
  const positions = Float64Array.from([dx, 0, 0, dx + 1, 0, 0, dx, 1, 0, dx, 0, 1]);
  const indices = Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
  return { positions, indices };
}

const disposers: (() => void | Promise<void>)[] = [];

afterEach(async () => {
  for (const d of disposers.splice(0).reverse()) {
    await d();
  }
  bounds.length = 0;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** A local store with meshes `a`, `b` and `c`, already in frame-loop mode. */
function setup(): { store: SceneStore; a: MeshHandle; b: MeshHandle; c: MeshHandle } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  disposers.push(() => {
    store.dispose();
    scene.free();
  });
  const [a, b, c] = [0, 3, 6].map((dx) => {
    const t = tetra(dx);
    return store.addMesh(t.positions, t.indices);
  });
  store.commitFrame();
  return { store, a, b, c };
}

type Renderer = Awaited<ReturnType<typeof create>>;

async function mount(el: React.ReactNode, options?: Parameters<typeof create>[1]): Promise<Renderer> {
  const r = await create(el, options);
  let mounted = true;
  disposers.push(async () => {
    if (mounted) {
      mounted = false;
      await r.unmount();
    }
  });
  const unmount = r.unmount;
  r.unmount = async () => {
    mounted = false;
    await unmount();
  };
  return r;
}

/** Runs `n` frames inside act(), so store updates re-render and run effects. */
async function frames(r: Renderer, n: number) {
  await act(async () => {
    await r.advanceFrames(n, 1 / 60);
  });
}

describe("FastMesh", () => {
  it("fastmesh_binds_and_follows_display", async () => {
    const { store, a } = setup();
    const r = await mount(
      <SceneProvider store={store}>
        <FastMesh mesh={a} />
      </SceneProvider>,
    );
    expect(r.scene.children).toHaveLength(1);
    expect(bounds).toHaveLength(1);
    const object = r.scene.children[0].instance as THREE.Mesh;
    expect(object).toBe(bounds[0].object);
    expect(object.visible).toBe(true);

    await act(async () => {
      store.setDisplay(a, { opacity: 0.4, visible: false });
    });
    // Frame-loop mode: the change waits for the frame.
    expect((object.material as THREE.Material).opacity).toBe(1);
    await frames(r, 1);
    const material = object.material as THREE.MeshStandardMaterial;
    expect(material.opacity).toBe(0.4);
    expect(material.transparent).toBe(true);
    expect(object.visible).toBe(false);

    await act(async () => {
      store.setDisplay(a, { opacity: 1, visible: true });
    });
    await frames(r, 1);
    expect(material.opacity).toBe(1);
    expect(material.transparent).toBe(false);
    expect(object.visible).toBe(true);
  });

  it("fastmesh_store_prop_without_provider", async () => {
    const { store, a } = setup();
    const spy = vi.spyOn(store, "commitFrame");
    const r = await mount(<FastMesh mesh={a} store={store} />);
    expect(r.scene.children).toHaveLength(1);
    await frames(r, 2);
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("fastmesh_outside_provider_throws", async () => {
    const { a } = setup();
    vi.spyOn(console, "error").mockImplementation(() => {});
    await expect(create(<FastMesh mesh={a} />)).rejects.toMatchObject({ code: "INVALID_ARGUMENT" });
  });

  it("one_commit_per_frame", async () => {
    const { store, a, b, c } = setup();
    const spy = vi.spyOn(store, "commitFrame");
    const r = await mount(
      <SceneProvider store={store}>
        <FastMesh mesh={a} />
        <FastMesh mesh={b} />
        <FastMesh mesh={c} />
      </SceneProvider>,
    );
    expect(r.scene.children).toHaveLength(3);
    await frames(r, 5);
    expect(spy).toHaveBeenCalledTimes(5);
    // Separate advanceFrames calls keep counting one per frame.
    await frames(r, 1);
    await frames(r, 2);
    expect(spy).toHaveBeenCalledTimes(8);

    // The driving instance unmounts: another takes over, still once per frame.
    await r.update(
      <SceneProvider store={store}>
        <FastMesh mesh={b} />
        <FastMesh mesh={c} />
      </SceneProvider>,
    );
    spy.mockClear();
    await frames(r, 3);
    expect(spy).toHaveBeenCalledTimes(3);
  });

  it("one_commit_per_frame_per_store", async () => {
    const one = setup();
    const two = setup();
    const spyOne = vi.spyOn(one.store, "commitFrame");
    const spyTwo = vi.spyOn(two.store, "commitFrame");
    const r = await mount(
      <>
        <FastMesh mesh={one.a} store={one.store} />
        <FastMesh mesh={one.b} store={one.store} />
        <FastMesh mesh={two.a} store={two.store} />
        <FastMesh mesh={two.b} store={two.store} />
      </>,
    );
    await frames(r, 4);
    expect(spyOne).toHaveBeenCalledTimes(4);
    expect(spyTwo).toHaveBeenCalledTimes(4);
  });

  it("use_frame_driver_alone", async () => {
    const { store } = setup();
    const spy = vi.spyOn(store, "commitFrame");
    function Driver() {
      useFrameDriver();
      return null;
    }
    const r = await mount(
      <SceneProvider store={store}>
        <Driver />
        <Driver />
      </SceneProvider>,
    );
    await frames(r, 3);
    expect(spy).toHaveBeenCalledTimes(3);
  });

  it("fastmesh_unmount_after_remove", async () => {
    const { store, a } = setup();
    const spy = vi.spyOn(store, "commitFrame");
    const errors = vi.spyOn(console, "error");
    const r = await mount(
      <SceneProvider store={store}>
        <FastMesh mesh={a} />
      </SceneProvider>,
    );
    expect(bounds).toHaveLength(1);
    const bound = bounds[0];

    await act(async () => {
      store.removeMesh(a);
    });
    await frames(r, 1);
    expect(spy).toHaveBeenCalledTimes(1);

    // Unmount the component while the root stays alive.
    await r.update(<SceneProvider store={store} />);
    expect(r.scene.children).toHaveLength(0);
    expect(bound.dispose).toHaveBeenCalledTimes(1);
    expect(errors).not.toHaveBeenCalled();

    // No frame-driver subscription is left behind.
    spy.mockClear();
    await frames(r, 3);
    expect(spy).not.toHaveBeenCalled();

    await r.unmount();
    expect(bound.dispose).toHaveBeenCalledTimes(1);
    expect(bounds).toHaveLength(1);
  });

  it("fastmesh_demand_invalidates", async () => {
    const { store, a } = setup();
    const before = listenerCount(store);
    let state: RootState | undefined;
    function Grab() {
      state = useThree();
      return null;
    }
    // No requestAnimationFrame in Node: R3F's demand loop is never started.
    vi.stubGlobal("requestAnimationFrame", () => 0);
    const r = await mount(<Grab />, { frameloop: "demand" });
    const invalidate = vi.spyOn(state!, "invalidate").mockImplementation(() => {});
    await r.update(
      <>
        <Grab />
        <SceneProvider store={store}>
          <FastMesh mesh={a} />
          <FastMesh mesh={a} />
        </SceneProvider>
      </>,
    );
    await frames(r, 1);
    invalidate.mockClear();

    await act(async () => {
      store.setDisplay(a, { opacity: 0.5 });
    });
    expect(invalidate).toHaveBeenCalled();
    const object = r.scene.children[0].instance as THREE.Mesh;
    expect((object.material as THREE.Material).opacity).toBe(1);
    await frames(r, 1);
    expect((object.material as THREE.Material).opacity).toBe(0.5);

    // A frame with no change asks for no other frame.
    invalidate.mockClear();
    await frames(r, 1);
    expect(invalidate).not.toHaveBeenCalled();

    // Unmounted: no subscription left on the store.
    await r.update(<Grab />);
    expect(listenerCount(store)).toBe(before);
  });

  it("fastmesh_unmount_restores_microtask_flush", async () => {
    const { store, a, b } = setup();
    const r = await mount(
      <SceneProvider store={store}>
        <FastMesh mesh={a} />
      </SceneProvider>,
    );
    await frames(r, 1);
    await r.update(<SceneProvider store={store} />);
    const rule: DistanceColoring = { distanceTo: b, maxDistance: 5, map: "turbo", range: [0, 2] };
    store.setDisplay(a, { coloring: rule });
    expect(meshState(store.getSnapshot(), a).display.coloring).toBeNull();
    await Promise.resolve();
    expect(meshState(store.getSnapshot(), a).display.coloring).toEqual(rule);
  });

  it("fastmesh_display_same_frame", async () => {
    const { store, a, b } = setup();
    const r = await mount(
      <SceneProvider store={store}>
        <FastMesh mesh={a} />
      </SceneProvider>,
    );
    const object = r.scene.children[0].instance as THREE.Mesh;
    await act(async () => {
      store.setDisplay(a, {
        coloring: { distanceTo: b, maxDistance: 5, map: "turbo", range: [0, 2] },
        opacity: 0.5,
        visible: false,
      });
    });
    expect(object.geometry.getAttribute("color")).toBeUndefined();
    await act(async () => {
      // Checked inside act(), before React re-renders: the frame alone applies both.
      void r.advanceFrames(1, 1 / 60);
      expect(object.geometry.getAttribute("color")).toBeDefined();
      expect((object.material as THREE.Material).opacity).toBe(0.5);
      expect(object.visible).toBe(false);
    });
  });

  it("fastmesh_on_bind_section", async () => {
    const { store, a } = setup();
    const section = store.addSection({ plane: { origin: [0, 0, 0.2], normal: [0, 0, 1] }, meshes: [a] });
    store.commitFrame();
    const onBind = vi.fn<(b: BoundMesh | null) => void>();
    const r = await mount(
      <SceneProvider store={store}>
        <FastMesh mesh={a} onBind={onBind} />
      </SceneProvider>,
    );
    expect(onBind).toHaveBeenCalledTimes(1);
    const binding: MeshBinding = onBind.mock.calls[0][0]!;
    expect(binding.object).toBe(r.scene.children[0].instance);
    const drawn = bindSection(store.scene, section, [binding]);
    const material = binding.object.material as THREE.Material;
    expect(material.clippingPlanes).toHaveLength(1);
    expect(drawn.group.children.length).toBeGreaterThan(0);
    drawn.dispose();

    // A new inline callback does not rebind.
    await r.update(
      <SceneProvider store={store}>
        <FastMesh mesh={a} onBind={(x) => onBind(x)} />
      </SceneProvider>,
    );
    expect(bounds).toHaveLength(1);
    await r.update(<SceneProvider store={store} />);
    expect(onBind).toHaveBeenCalledTimes(2);
    expect(onBind).toHaveBeenLastCalledWith(null);
  });

  it("fastmesh_shading_props", async () => {
    const { store, a } = setup();
    const el = (shading: "vertex" | "fragment", edgeSoftness: number) => (
      <SceneProvider store={store}>
        <FastMesh mesh={a} shading={shading} edgeSoftness={edgeSoftness} />
      </SceneProvider>
    );
    const r = await mount(el("fragment", 0.2));
    expect(bounds).toHaveLength(1);
    expect(bounds[0].shading).toBe("fragment");
    const soften = vi.spyOn(bounds[0], "setEdgeSoftness");
    // A new softness is set on the binding, without a rebind.
    await r.update(el("fragment", 0.5));
    expect(soften).toHaveBeenLastCalledWith(0.5);
    expect(bounds).toHaveLength(1);
    // A new shading rebinds.
    await r.update(el("vertex", 0.5));
    expect(bounds).toHaveLength(2);
    expect(bounds[0].dispose).toHaveBeenCalled();
    expect(bounds[1].shading).toBe("vertex");
  });

  it("fastmesh_bad_softness_is_ignored", async () => {
    const { store, a } = setup();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const el = (edgeSoftness: number) => (
      <SceneProvider store={store}>
        <FastMesh mesh={a} shading="fragment" edgeSoftness={edgeSoftness} />
      </SceneProvider>
    );
    // NaN or a negative width binds and draws as 0 (a hard edge), with a warning.
    const r = await mount(el(Number.NaN));
    expect(bounds).toHaveLength(1);
    const soften = vi.spyOn(bounds[0], "setEdgeSoftness");
    await r.update(el(0.3));
    expect(soften).toHaveBeenLastCalledWith(0.3);
    await r.update(el(-1));
    expect(soften).toHaveBeenLastCalledWith(0);
    expect(warn.mock.calls.some((c) => String(c[0]).includes("edgeSoftness"))).toBe(true);
    warn.mockRestore();
  });

  it("package_r3f_export", () => {
    const dist = new URL("../dist/", import.meta.url);
    expect(existsSync(new URL("r3f.js", dist))).toBe(true);
    expect(existsSync(new URL("r3f.d.ts", dist))).toBe(true);
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(pkg.exports["./r3f"]).toEqual({ types: "./dist/r3f.d.ts", default: "./dist/r3f.js" });
    expect(pkg.peerDependencies["@react-three/fiber"]).toBe(">=9");
    expect(pkg.peerDependenciesMeta["@react-three/fiber"]).toEqual({ optional: true });

    /** Every dist file a file imports, transitively. */
    const graph = (file: string, seen = new Set<string>()): Set<string> => {
      if (seen.has(file)) {
        return seen;
      }
      seen.add(file);
      const src = readFileSync(new URL(file, dist), "utf8");
      for (const m of src.matchAll(/(?:import|export)\s[^"']*?from\s*["']([^"']+)["']|import\s*["']([^"']+)["']/g)) {
        const spec = m[1] ?? m[2];
        if (spec.startsWith("./")) {
          graph(new URL(spec, new URL(file, dist)).href.slice(dist.href.length), seen);
        } else {
          seen.add(spec);
        }
      }
      return seen;
    };
    const files = [...graph("r3f.js")];
    expect(files).toEqual(expect.arrayContaining(["@react-three/fiber", "three", "react"]));
    expect(files.filter((f) => /fast_mesh_wasm\.js|^wasm\//.test(f))).toEqual([]);
    const code = files.filter((f) => f.endsWith(".js")).map((f) => readFileSync(new URL(f, dist), "utf8"));
    // Imported, never inlined: no copy of R3F's reconciler, three's classes or React's internals.
    for (const src of code) {
      expect(src).not.toMatch(/\bimport\(/);
      expect(src).not.toMatch(/__SECRET_INTERNALS|__CLIENT_INTERNALS|REVISION\s*=\s*["']\d+/);
      expect(src).not.toMatch(/function createRoot\b|class WebGLRenderer\b/);
    }
    // It shares the hooks' code with `./react` (one context), never a copy.
    const react = graph("react.js");
    expect(files.filter((f) => f !== "r3f.js" && f.endsWith(".js") && react.has(f)).length).toBeGreaterThan(0);
    // `./react` stays free of three and R3F.
    expect([...react].filter((f) => /^three|@react-three/.test(f))).toEqual([]);
  });
});
