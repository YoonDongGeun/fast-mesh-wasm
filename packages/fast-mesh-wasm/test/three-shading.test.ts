// Fragment shading (`bindMesh({ shading: "fragment" })`): the JavaScript
// reference of the shader's mapping against the engine's `color_of` (through
// the test build's `__test_color_of`), the vertex value encoding, edge
// softness, and the binding's attribute, uniforms and material patch. The
// GLSL itself runs in headless Chrome (scripts/showcase-check.mjs).
import * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  encodeValues,
  exactBytes,
  SHADING_FRAGMENT,
  SRGB_TO_LINEAR16,
  shadeReference,
  shadingOf,
  softBytes,
} from "../src/colormap.js";
import { type ColorRule, type MeshHandle, Scene, type Vec3 } from "../src/node.js";
import { encodeRule } from "../src/rule.js";
import type { SceneLike } from "../src/scene-like.js";
import { contactHeatmap } from "../src/store/index.js";
import { bindMesh, type BoundMesh } from "../src/three/index.js";
import * as raw from "../src/wasm/fast_mesh_wasm.js";

// Only the test build exports it; the cast keeps `tsc` happy on a release build.
const colorOf = (raw as unknown as { __test_color_of: (m: number, r: Float32Array, v: Float32Array) => Uint8Array })
  .__test_color_of;

/** The engine's colours of `values` under `rule`. */
function engineColors(rule: ColorRule, values: Float32Array): Uint8Array {
  const { map, values: encoded } = encodeRule(rule);
  return colorOf(map, encoded, values);
}

/** A deterministic PRNG in [0, 1). */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** An n×n vertex grid on the XY plane, `size` wide, offset by `at`. */
function grid(n: number, size: number, at: Vec3): { positions: Float64Array; indices: Uint32Array } {
  const positions = new Float64Array(n * n * 3);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      positions.set([(x * size) / (n - 1) + at[0], (y * size) / (n - 1) + at[1], at[2]], (y * n + x) * 3);
    }
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let k = 0;
  for (let y = 0; y < n - 1; y++) {
    for (let x = 0; x < n - 1; x++) {
      const a = y * n + x;
      indices.set([a, a + 1, a + n + 1, a, a + n + 1, a + n], k);
      k += 6;
    }
  }
  return { positions, indices };
}

const field = undefined as unknown as ColorRule["field"];
const STOPS = contactHeatmap({ depth: 1, gap: 2, tolerance: 0.25 });
/** Rules covering every map, a centre, and both out-of-range modes. */
const RULES: ColorRule[] = [
  { field, map: "turbo", range: [-1, 3] },
  { field, map: "viridis", range: [0, 1], below: "base", above: "clamp" },
  { field, map: "coolwarm", range: [-2, 1], center: 0, base: [10, 20, 30, 40], nan: [1, 2, 3, 4] },
  { field, map: "grayscale", range: [0.5, 0.75], center: 0.7, below: "base", above: "base" },
  { field, map: STOPS.map, range: STOPS.range, below: STOPS.below, above: STOPS.above },
  {
    field,
    map: { stops: [[0.2, [255, 0, 0, 255]], [0.5, [0, 255, 0, 128]], [0.9, [0, 0, 255, 0]]] },
    range: [-3, 3],
    center: -1,
    above: "clamp",
  },
];

/** Values around and inside `rule`'s range, its knots included. */
function samples(rule: ColorRule, random: () => number): Float32Array {
  const [lo, hi] = rule.range;
  const out: number[] = [lo, hi, rule.center ?? lo, lo - 1, hi + 1, Infinity, -Infinity, Number.NaN];
  const knots = typeof rule.map === "string" ? [] : rule.map.stops.map(([t]) => t);
  const c = rule.center;
  for (const t of knots) {
    // The value whose t is the knot, both sides of a centre.
    out.push(c === undefined ? lo + t * (hi - lo) : t < 0.5 ? lo + 2 * t * (c - lo) : c + (2 * t - 1) * (hi - c));
  }
  for (let k = 0; k < 2000; k++) {
    out.push(lo - 0.1 * (hi - lo) + random() * 1.2 * (hi - lo));
  }
  return Float32Array.from(out);
}

/** sRGB bytes to linear as the shader gives them (the table is f32 in the texture). */
function linearOf(bytes: readonly number[]): number[] {
  return [...bytes.slice(0, 3).map((b) => Math.fround(SRGB_TO_LINEAR16[b] / 65535)), bytes[3] / 255];
}

describe("fragment shading reference", () => {
  it("exact_colours_at_vertices_equal_the_engine_color_of", () => {
    const random = rng(5);
    for (const rule of RULES) {
      const s = shadingOf(rule);
      const values = samples(rule, random);
      const want = engineColors(rule, values);
      const encoded = new Float32Array(4 * values.length);
      encodeValues(values, encoded, 0, values.length);
      for (let i = 0; i < values.length; i++) {
        // Through the attribute encoding, as the shader sees the vertex.
        const got = shadeReference(s, encoded.subarray(4 * i, 4 * i + 4), 0);
        expect(got, `${JSON.stringify(rule.map).slice(0, 30)} at ${values[i]}`).toEqual(
          linearOf(Array.from(want.subarray(4 * i, 4 * i + 4))),
        );
      }
    }
  });

  it("far_and_nan_never_tint_their_neighbours", () => {
    // The fast contact heatmap: deep penetration sits next to far vertices.
    const heat = contactHeatmap({ depth: 1, gap: 2, tolerance: 0.25 });
    const rule: ColorRule = { field, map: heat.map, range: heat.range, below: heat.below, above: heat.above };
    const s = shadingOf(rule);
    const [lo] = s.range;
    const enc = (v: number) => {
      const out = new Float32Array(4);
      encodeValues([v], out, 0, 1);
      return out;
    };
    expect(Array.from(enc(Infinity))).toEqual([0, 0, 0, 1]);
    expect(Array.from(enc(-Infinity))).toEqual([0, 0, 0, -1]);
    expect(Array.from(enc(Number.NaN))).toEqual([0, 0, 1, 0]);
    expect(Array.from(enc(0.25))).toEqual([0.25, 1, 0, 0]);
    const base = linearOf([200, 200, 200, 255]);
    const nan = linearOf([128, 128, 128, 255]);
    const mix = (a: Float32Array, b: Float32Array, t: number) => a.map((x, k) => Math.fround((1 - t) * x + t * b[k]));
    for (const v0 of [Math.fround(lo - 1e-3), lo, Math.fround(lo + 1e-3)]) {
      const own = shadeReference(s, enc(v0), 0);
      for (const [other, otherColor] of [
        [Infinity, base],
        [Number.NaN, nan],
      ] as const) {
        for (let k = 0; k <= 100; k++) {
          const got = shadeReference(s, mix(enc(v0), enc(other), k / 100), 0);
          // Only the deep vertex's own colour or the far / NaN colour: no
          // contact white, no gap colours on the way.
          expect([own, otherColor], `${v0} to ${other} at ${k / 100}`).toContainEqual(got);
          expect(got).toEqual(k <= 50 ? own : otherColor);
        }
      }
    }
    // -Infinity follows `below` (clamp: the deepest colour).
    expect(shadeReference(s, enc(-Infinity), 0)).toEqual(linearOf(exactBytes(s, lo - 1)));
    // NaN beats far only where its weight is larger.
    expect(shadeReference(s, [0, 0.2, 0.5, 0.3], 0)).toEqual(nan);
    expect(shadeReference(s, [0, 0.2, 0.3, 0.5], 0)).toEqual(base);
    // Between finite vertices the value is the plain interpolation.
    const a = enc(-0.9);
    const b = enc(1.5);
    for (let k = 0; k <= 10; k++) {
      const m = mix(a, b, k / 10);
      expect(shadeReference(s, m, 0)).toEqual(linearOf(exactBytes(s, Math.fround(m[0] / m[1]))));
    }
  });

  it("fast_mode_outer_boundary_blends_under_softness", () => {
    // Fast mode: the field is bounded at 2 · max(depth, gap), so a vertex just
    // beyond the gap (the range end) still holds its value: the edge from a
    // coloured vertex to it crosses `gap` as a value boundary, which softness
    // blends. Only vertices beyond the bound are far.
    const heat = contactHeatmap({ depth: 1, gap: 2, tolerance: 0.25 });
    const rule: ColorRule = { field, map: heat.map, range: heat.range, below: heat.below, above: heat.above };
    const s = shadingOf(rule);
    const gap = heat.range[1];
    const inside = 1.5;
    const outside = 2.6;
    // Both within the field's bound: finite values, not far.
    expect(outside).toBeLessThanOrEqual(heat.maxDistance);
    const enc = (v: number) => {
      const out = new Float32Array(4);
      encodeValues([v], out, 0, 1);
      return out;
    };
    const a = enc(inside);
    const b = enc(outside);
    const at = (t: number) => a.map((x, k) => Math.fround((1 - t) * x + t * b[k]));
    /** Where along the edge the interpolated value is `v`. */
    const where = (v: number) => (v - inside) / (outside - inside);
    const base = linearOf([200, 200, 200, 255]);
    const farColour = shadeReference(s, enc(Math.fround(gap - 1e-4)), 0);
    // Softness 0: a hard step at the gap.
    expect(shadeReference(s, at(where(gap) - 1e-3), 0)).not.toEqual(base);
    expect(shadeReference(s, at(where(gap) + 1e-3), 0)).toEqual(base);
    // Softness 0.4: a ramp across [gap - 0.2, gap + 0.2], half way at the gap.
    const softness = 0.4;
    const mid = shadeReference(s, at(where(gap)), softness);
    for (let c = 0; c < 3; c++) {
      const lo = Math.min(farColour[c], base[c]);
      const hi = Math.max(farColour[c], base[c]);
      expect(mid[c]).toBeGreaterThan(lo + 0.2 * (hi - lo));
      expect(mid[c]).toBeLessThan(hi - 0.2 * (hi - lo));
    }
    expect(shadeReference(s, at(where(gap + 0.25)), softness)).toEqual(base);
    // Each step of the ramp moves towards the base colour.
    let prev = shadeReference(s, at(where(gap - 0.2)), softness);
    for (let k = 1; k <= 20; k++) {
      const now = shadeReference(s, at(where(gap - 0.2 + (0.4 * k) / 20)), softness);
      for (let c = 0; c < 3; c++) {
        expect(Math.sign(now[c] - prev[c]) * Math.sign(base[c] - farColour[c])).toBeGreaterThanOrEqual(0);
      }
      prev = now;
    }
    // With the old bound (max(depth, gap) = 2) the outside vertex would be far:
    // a hard switch half way, whatever the softness.
    const old = enc(Infinity);
    const pastHalf = a.map((x, k) => Math.fround(0.51 * old[k] + 0.49 * x));
    const beforeHalf = a.map((x, k) => Math.fround(0.49 * old[k] + 0.51 * x));
    expect(shadeReference(s, pastHalf, softness)).toEqual(base);
    expect(shadeReference(s, beforeHalf, softness)).toEqual(shadeReference(s, [inside, 1, 0, 0], softness));
  });

  it("edge_softness_zero_is_a_step_and_wider_a_ramp", () => {
    const rule: ColorRule = { field, map: STOPS.map, range: STOPS.range, below: STOPS.below, above: STOPS.above };
    const s = shadingOf(rule);
    const hi = s.range[1];
    const far = exactBytes(s, hi);
    const base = [200, 200, 200, 255];
    // Hard: the far colour up to hi, the base colour just beyond.
    expect(exactBytes(s, Math.fround(hi - 1e-6))).toEqual(far);
    expect(exactBytes(s, Math.fround(hi + 1e-6))).toEqual(base);
    // Soft (s = 0.4): base from hi + 0.2, a monotone blend across [hi - 0.2, hi + 0.2].
    const h = 0.2;
    expect(softBytes(s, hi + h + 1e-3, h)).toEqual(base);
    const mid = softBytes(s, hi, h);
    for (let k = 0; k < 3; k++) {
      expect(mid[k]).toBeGreaterThan(Math.min(far[k], base[k]) + 1);
      expect(mid[k]).toBeLessThan(Math.max(far[k], base[k]) - 1);
    }
    let prev = softBytes(s, hi - h, h);
    for (let k = 1; k <= 20; k++) {
      const now = softBytes(s, hi - h + (k * 2 * h) / 20, h);
      for (let c = 0; c < 3; c++) {
        // Towards the base colour, channel by channel.
        expect(Math.sign(now[c] - prev[c]) * Math.sign(base[c] - far[c])).toBeGreaterThanOrEqual(0);
      }
      prev = now;
    }
    // The tolerance band's edge (contact white to shallow red at -0.25) softens too.
    const inBand = exactBytes(s, 0);
    expect(inBand).toEqual([255, 255, 255, 255]);
    const edge = softBytes(s, -0.25, 0.05);
    expect(edge[1]).toBeLessThan(250);
    expect(edge[1]).toBeGreaterThan(110);
    // A linear stretch of the map is left as it is.
    const ramp = shadingOf({ field, map: "grayscale", range: [0, 1] });
    for (const v of [0.3, 0.5, 0.7]) {
      const soft = softBytes(ramp, v, 0.1);
      expect(soft[0]).toBeCloseTo(v * 255, 3);
    }
    // Wide windows over a built-in table use the running integral.
    const turbo = shadingOf({ field, map: "turbo", range: [0, 1] });
    const narrow = softBytes(turbo, 0.5, 16 / 255 / 2 - 1e-6);
    const wide = softBytes(turbo, 0.5, 16 / 255 / 2 + 1e-6);
    for (let c = 0; c < 4; c++) {
      expect(wide[c]).toBeCloseTo(narrow[c], 2);
    }
  });

  it("shader_mirrors_the_reference", () => {
    // Every reference step has its GLSL function.
    for (const fn of ["fmColorAt", "fmRound", "fmT", "fmExact", "fmPrefix", "fmIntegral", "fmSoft", "fmLinear", "fmShade"]) {
      expect(SHADING_FRAGMENT).toContain(`vec4 ${fn}(`.replace("vec4 fmT(", "float fmT("));
    }
    expect(SHADING_FRAGMENT).toContain("> 16.0");
  });
});

describe("bindMesh fragment shading", () => {
  const scenes: Scene[] = [];
  const bound: BoundMesh[] = [];
  afterEach(() => {
    for (const b of bound.splice(0)) {
      b.dispose();
    }
    for (const s of scenes.splice(0)) {
      s.free();
    }
  });

  function setup() {
    const s = new Scene();
    scenes.push(s);
    const sheet = grid(16, 4, [0, 0, 0]);
    const m = s.addMesh(sheet.positions, sheet.indices);
    const small = grid(4, 1, [1.5, 1.5, 0.3]);
    const n = s.addMesh(small.positions, small.indices);
    return { s, m, n };
  }

  function bind(s: SceneLike, m: MeshHandle, options: Parameters<typeof bindMesh>[2] = {}) {
    const b = bindMesh(s, m, options);
    bound.push(b);
    return b;
  }

  /** The values attribute equals a fresh encoding of the scene's values. */
  function expectValuesMirrored(s: Scene, m: MeshHandle, b: BoundMesh) {
    const values = s.views(m).values!()!;
    const want = new Float32Array(4 * values.length);
    encodeValues(values, want, 0, values.length);
    expect(Array.from(b.object.geometry.getAttribute("fmValue").array)).toEqual(Array.from(want));
  }

  it("fragment_colour_at_vertices_equals_vertex_colour", () => {
    const { s, m, n } = setup();
    const f = s.addDistanceField({ name: "d", on: m, to: n, maxDistance: 1.5 });
    for (const rule of [
      { field: f, ...STOPS, range: STOPS.range },
      { field: f, map: "coolwarm" as const, range: [-0.5, 1] as const, center: 0.4 },
      { field: f, map: "turbo" as const, range: [0, 0.8] as const, above: "clamp" as const },
    ]) {
      const { maxDistance: _m, sign: _s, ...colorRule } = rule as typeof rule & { maxDistance?: number; sign?: string };
      s.setColoring(m, colorRule as ColorRule);
      const vertex = bind(s, m);
      const fragment = bind(s, m, { shading: "fragment" });
      s.commitFrame();
      expect(fragment.shading).toBe("fragment");
      const color = vertex.object.geometry.getAttribute("color").array as Uint16Array;
      const value = fragment.object.geometry.getAttribute("fmValue").array as Float32Array;
      expect(fragment.object.geometry.getAttribute("color")).toBeUndefined();
      const sh = shadingOf(s.coloring(m)!);
      let far = 0;
      for (let v = 0; v < value.length / 4; v++) {
        if (s.views(m).values!()![v] === Infinity) {
          far++;
        }
        const got = shadeReference(sh, value.subarray(4 * v, 4 * v + 4), 0);
        for (let k = 0; k < 3; k++) {
          // The same 16-bit level: the texture holds the table as f32.
          expect(Math.round(got[k] * 65535), `vertex ${v}`).toBe(color[4 * v + k]);
        }
        expect(Math.round(got[3] * 65535)).toBe(color[4 * v + 3]);
      }
      expect(far).toBeGreaterThan(0);
      vertex.dispose();
      fragment.dispose();
    }
  });

  it("values_follow_the_colours_ranges_and_the_rule", () => {
    const { s, m, n } = setup();
    const f = s.addDistanceField({ name: "d", on: m, to: n, maxDistance: 1 });
    const b = bind(s, m, { shading: "fragment", edgeSoftness: 0.1 });
    const material = b.object.material as THREE.MeshStandardMaterial;
    const shader = {
      uniforms: {} as Record<string, { value: unknown }>,
      vertexShader: THREE.ShaderLib.standard.vertexShader,
      fragmentShader: THREE.ShaderLib.standard.fragmentShader,
    };
    material.onBeforeCompile(shader as never, undefined as never);
    expect(shader.vertexShader).toContain("vFmValue = fmValue;");
    expect(shader.fragmentShader).toContain("diffuseColor *= fmShade(vFmValue);");
    expect(material.customProgramCacheKey()).toContain("fast-mesh-fragment-shading");
    const u = shader.uniforms;
    expect(u.fmSoftness.value).toBe(0.1);
    expect(u.fmOn.value).toBe(false);
    // Colouring on: the values attribute, the uniforms, a white material.
    const rule: ColorRule = { field: f, map: "turbo", range: [0, 1] };
    s.setColoring(m, rule);
    s.commitFrame();
    expect(u.fmOn.value).toBe(true);
    expect(material.vertexColors).toBe(false);
    expect(material.color.getHex()).toBe(0xffffff);
    expect((u.fmLut.value as THREE.DataTexture).image.width).toBe(256);
    expectValuesMirrored(s, m, b);
    // A drag updates the colours' ranges only, and the values with them.
    const attribute = b.object.geometry.getAttribute("fmValue") as THREE.BufferAttribute;
    attribute.clearUpdateRanges();
    s.setPlacement(n, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0.3, 0, -0.1] });
    s.commitFrame();
    expect(attribute.updateRanges.length).toBeGreaterThan(0);
    expectValuesMirrored(s, m, b);
    // A new rule (another range) encodes every value again.
    const wider: ColorRule = { field: f, map: { stops: [[0, [0, 0, 0, 255]], [1, [255, 255, 255, 255]]] }, range: [-1, 2] };
    s.setColoring(m, wider);
    s.commitFrame();
    expect(u.fmBuiltin.value).toBe(false);
    expect(u.fmCount.value).toBe(2);
    expect((u.fmRange.value as THREE.Vector2).toArray()).toEqual([-1, 2]);
    expectValuesMirrored(s, m, b);
    // Softness: a uniform, checked.
    b.setEdgeSoftness(0);
    expect(u.fmSoftness.value).toBe(0);
    expect(() => b.setEdgeSoftness(-1)).toThrow(/edgeSoftness/);
    expect(() => b.setEdgeSoftness(Number.NaN)).toThrow(/edgeSoftness/);
    // Colouring off: no attribute, the base colour back.
    s.setColoring(m, null);
    s.commitFrame();
    expect(b.object.geometry.getAttribute("fmValue")).toBeUndefined();
    expect(u.fmOn.value).toBe(false);
    expect(material.color.getHex()).toBe(0xcccccc);
  });

  it("a_rebound_material_compiles_with_the_new_bindings_uniforms", () => {
    const { s, m } = setup();
    const material = new THREE.MeshStandardMaterial();
    let disposed = 0;
    material.addEventListener("dispose", () => disposed++);
    /** The uniforms the renderer would compile the material with now. */
    const compiled = () => {
      const shader = {
        uniforms: {} as Record<string, { value: unknown }>,
        vertexShader: THREE.ShaderLib.standard.vertexShader,
        fragmentShader: THREE.ShaderLib.standard.fragmentShader,
      };
      material.onBeforeCompile(shader as never, undefined as never);
      return shader;
    };
    const first = bindMesh(s, m, { material, shading: "fragment", edgeSoftness: 0.1 });
    // Patching drops any program compiled before, so three recompiles.
    expect(disposed).toBe(1);
    expect(compiled().uniforms.fmSoftness.value).toBe(0.1);
    // One material, one fragment-shaded binding.
    expect(() => bindMesh(s, m, { material, shading: "fragment" })).toThrow(/material.clone/);
    first.dispose();
    expect(disposed).toBe(2);
    const second = bindMesh(s, m, { material, shading: "fragment", edgeSoftness: 0.3 });
    expect(disposed).toBe(3);
    const shader = compiled();
    expect(shader.uniforms.fmSoftness.value).toBe(0.3);
    second.setEdgeSoftness(0.4);
    expect(shader.uniforms.fmSoftness.value).toBe(0.4);
    second.dispose();
    // Fragment, then vertex: the material is back as it was, recompiled.
    const plain = bindMesh(s, m, { material });
    expect(disposed).toBe(4);
    expect(compiled().fragmentShader).not.toContain("fmShade");
    plain.dispose();
    material.dispose();
  });

  it("unsupported_materials_fall_back_with_a_warning", () => {
    const { s, m } = setup();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const toon = new THREE.MeshToonMaterial() as unknown as THREE.MeshStandardMaterial;
    expect(bind(s, m, { material: toon, shading: "fragment" }).shading).toBe("vertex");
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/MeshToonMaterial is drawn with vertex colours/));
    const node = new THREE.MeshStandardMaterial();
    (node as unknown as { isNodeMaterial: boolean }).isNodeMaterial = true;
    expect(bind(s, m, { material: node, shading: "fragment" }).shading).toBe("vertex");
    expect(bind(s, m, { material: new THREE.MeshBasicMaterial(), shading: "fragment" }).shading).toBe("fragment");
    // A chunk another handler removed: warned once, the shader left alone.
    const odd = new THREE.MeshStandardMaterial();
    const b = bind(s, m, { material: odd, shading: "fragment" });
    expect(b.shading).toBe("fragment");
    const shader = { uniforms: {}, vertexShader: "void main() {}", fragmentShader: "void main() {}" };
    odd.onBeforeCompile(shader as never, undefined as never);
    expect(shader.fragmentShader).toBe("void main() {}");
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/could not find #include <common>/));
    warn.mockRestore();
    toon.dispose();
    node.dispose();
    odd.dispose();
  });

  it("given_material_is_restored_and_hosts_without_values_fall_back", () => {
    const { s, m } = setup();
    const material = new THREE.MeshLambertMaterial();
    const before = material.onBeforeCompile;
    const b = bindMesh(s, m, { material, shading: "fragment" });
    expect(material.onBeforeCompile).not.toBe(before);
    b.dispose();
    expect(material.onBeforeCompile).toBe(before);
    material.dispose();
    // A host without `coloring` draws vertex colours.
    const host: SceneLike = {
      views: (h) => {
        const v = s.views(h);
        return { positions: v.positions, normals: v.normals, colors: v.colors, indices: v.indices };
      },
      matrix4: (h) => s.matrix4(h),
      placement: (h) => s.placement(h),
      section: (x, h) => s.section(x, h),
      sectionPlane: (x) => s.sectionPlane(x),
      onChanges: (l) => s.onChanges(l),
      commitFrame: () => s.commitFrame(),
      drag: (h) => s.drag(h),
    };
    expect(bind(host, m, { shading: "fragment" }).shading).toBe("vertex");
    expect(() => bind(s, m, { shading: "pixel" as never })).toThrow(/shading/);
    expect(() => bind(s, m, { edgeSoftness: -0.5 })).toThrow(/edgeSoftness/);
  });
});
