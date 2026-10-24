// `bindMesh`: a THREE.Mesh whose BufferGeometry follows a scene mesh through
// the change stream. Attributes own JS-side mirror arrays (spec §11.2): three
// uploads lazily, so handing it wasm views would leave it reading detached or
// moved memory at render time. Each frame copies only the dirty ranges from
// fresh views into the mirrors.
import * as THREE from "three";
import {
  encodeValues,
  LUT_ROWS,
  SHADING_FRAGMENT,
  SHADING_VERTEX,
  SRGB_TO_LINEAR16,
  type Shading,
  shadingOf,
} from "../colormap.js";
import { FastMeshError } from "../core.js";
import type { ColorRule } from "../scene.js";
import type { MeshChange, MeshHandle, Range } from "../scene.js";
import type { SceneLike } from "../scene-like.js";

/** Options of {@link bindMesh}. */
export interface BindMeshOptions {
  /**
   * The material to draw with. Defaults to a new `MeshStandardMaterial`, which
   * {@link BoundMesh.dispose} then disposes; a material passed here is used as
   * given and left to the caller to dispose.
   *
   * The binding controls the material's `color`, `vertexColors`, `opacity`,
   * `transparent` and `depthWrite` (colouring and {@link BoundMesh.setOpacity}
   * change them), so do not share one material between bound meshes: give
   * each its own (`material.clone()`). With `shading: "fragment"` the
   * binding calls the material's `dispose()` when it binds and again when it
   * is disposed and puts the material back (dropping its compiled program so
   * it is rebuilt with, then without, the shading); the material stays usable.
   */
  material?: THREE.MeshStandardMaterial | THREE.MeshPhongMaterial | THREE.MeshLambertMaterial | THREE.MeshBasicMaterial;
  /**
   * The colour shown while the mesh has no colouring. Defaults to the given
   * material's colour (left untouched), or light grey for the default material.
   */
  color?: THREE.ColorRepresentation;
  /**
   * Initial opacity in [0, 1], applied with {@link BoundMesh.setOpacity}.
   * Without it, the default material is opaque and a given material keeps
   * its `opacity`, `transparent` and `depthWrite`.
   */
  opacity?: number;
  /**
   * How a colouring is drawn.
   *
   * - `"vertex"` (the default): the engine's colours as a vertex attribute,
   *   interpolated across each triangle, so the edge between coloured and
   *   base-coloured vertices blurs over a triangle's width.
   * - `"fragment"`: every fragment is coloured from the interpolated field
   *   value through the colouring's rule (a lookup texture and a shader chunk
   *   added to the material with `onBeforeCompile`, so `WebGLRenderer` only).
   *   Range edges and colour steps (a contact heatmap's tolerance band) are
   *   crisp iso-lines of the value, softened with `edgeSoftness`, and
   *   gradients are smooth per pixel. At a vertex the colour equals its
   *   vertex colour. Far (`Infinity`) and NaN values never tint their
   *   neighbours. Needs a scene with `coloring(mesh)` and
   *   `views(mesh).values` (`Scene` and `WorkerScene` have both); on another
   *   host the binding falls back to `"vertex"` ({@link BoundMesh.shading}),
   *   and so does a material three's `WebGLRenderer` does not build from its
   *   own shader chunks (a node material for `WebGPURenderer`, say), with a
   *   warning.
   *
   *   The binding patches the material it draws with, so each fragment-shaded
   *   binding needs its own material (`material.clone()`): binding a material
   *   another live fragment-shaded binding uses throws `INVALID_ARGUMENT`.
   *   Disposing the binding puts the material back as it was (and disposes
   *   its compiled program, so a later binding recompiles with its own
   *   uniforms).
   */
  shading?: "vertex" | "fragment";
  /**
   * Fragment shading: the width, in the field's distance units, of the blend
   * at each colour step (a range edge to the base colour, the edges of a
   * contact band). 0 (the default) is a hard edge. Ignored by vertex shading.
   * See {@link BoundMesh.setEdgeSoftness}.
   */
  edgeSoftness?: number;
}

/** A scene mesh bound to a THREE.Mesh by {@link bindMesh}. */
export interface BoundMesh {
  /**
   * The mesh to add to a THREE.Scene. Its geometry has `position`, `normal`
   * (when the scene mesh keeps normals), `color` (while colouring is on) and
   * an index; its `matrix` is the scene mesh's world matrix
   * (`matrixAutoUpdate` is off). The colour attribute is normalized `Uint16`
   * RGBA: the engine's sRGB colours converted to three's linear working space.
   */
  readonly object: THREE.Mesh;
  readonly mesh: MeshHandle;
  /** The shading in use: `"fragment"` only when asked for and the scene supports it. */
  readonly shading: "vertex" | "fragment";
  /**
   * Sets the fragment shading's edge softness (see
   * {@link BindMeshOptions.edgeSoftness}): a finite number `>= 0`. Takes
   * effect on the next render, with no recompile. A display setting: never
   * recorded in history.
   */
  setEdgeSoftness(width: number): void;
  /**
   * Sets the opacity in [0, 1]. Below 1 the material is transparent and does
   * not write depth. A display setting: never recorded in history.
   */
  setOpacity(alpha: number): void;
  /**
   * Stops following the scene and disposes the geometry (and the material when
   * `bindMesh` created it). The object stays wherever it was added. Safe to
   * call more than once, before or after the scene mesh is removed.
   */
  dispose(): void;
}

/**
 * The part of a {@link BoundMesh} that `attachTransformControls` and
 * `bindSection` use: the scene mesh and its THREE.Mesh. A full `BoundMesh`
 * is one, and so is the binding `<FastMesh onBind>` delivers.
 */
export type MeshBinding = Pick<BoundMesh, "mesh" | "object">;

type Material = NonNullable<BindMeshOptions["material"]>;
type Mirror = Float32Array | Uint8Array | Uint16Array | Uint32Array;

const DEFAULT_COLOR = 0xcccccc;
const WHITE = new THREE.Color(0xffffff);

/**
 * Update ranges listed on one attribute before they are folded into a single
 * full range. Ranges pile up while a bound mesh is not rendered (culled,
 * hidden, on-demand rendering), and three uploads each one separately.
 */
const MAX_UPDATE_RANGES = 64;

/** sRGB byte to linear, as a normalized `Uint16` (see `colormap.ts`). */
export { SRGB_TO_LINEAR16 };

/**
 * Copies RGBA8 sRGB colours into the `Uint16` mirror at `at`: colour channels
 * through {@link SRGB_TO_LINEAR16}, alpha (already linear) scaled from 255 to
 * 65535. `at` is a multiple of 4.
 */
const copyColors = (mirror: Uint16Array, src: Uint8Array, at: number) => {
  for (let i = 0; i < src.length; i++) {
    const j = at + i;
    mirror[j] = (j & 3) === 3 ? src[i] * 257 : SRGB_TO_LINEAR16[src[i]];
  }
};

/** How each buffer maps to an attribute: components per element and the mirror type. */
const BUFFERS = {
  positions: { name: "position", itemSize: 3, normalized: false, make: (n: number) => new Float32Array(n) },
  normals: { name: "normal", itemSize: 3, normalized: false, make: (n: number) => new Float32Array(n) },
  colors: { name: "color", itemSize: 4, normalized: true, make: (n: number) => new Uint16Array(n) },
  // Fragment shading: (v · m, m, nan, far) per vertex, encoded by `encodeValues`.
  values: { name: "fmValue", itemSize: 4, normalized: false, make: (n: number) => new Float32Array(n) },
  indices: { name: "index", itemSize: 3, normalized: false, make: (n: number) => new Uint32Array(n) },
} as const;

type BufferKey = keyof typeof BUFFERS;

/**
 * Binds a scene mesh to a new THREE.Mesh and keeps it in sync on every
 * {@link SceneLike.commitFrame}: dirty ranges are copied into the attributes'
 * mirrors and marked with `addUpdateRange`, attributes are recreated when the
 * mesh's slot counts change, the colour attribute comes and goes with the
 * mesh's colouring, and `object.matrix` follows its placement.
 *
 * Dead vertex slots keep stale positions, so the geometry's bounding box and
 * sphere are computed from the vertices of live triangles only (a removed
 * triangle reads `0, 0, 0` in the index). They are recomputed when positions
 * or indices change, never on placement-only frames.
 *
 * With `shading: "fragment"` the colour attribute is replaced by the
 * colouring's values (`fmValue`), coloured per fragment by a shader chunk the
 * binding adds to the material (see {@link BindMeshOptions.shading}).
 *
 * When the scene mesh is removed, the binding detaches: it stops following
 * the scene and leaves the object as it was (remove it from your THREE.Scene
 * and call {@link BoundMesh.dispose}).
 *
 * The object must sit under an untransformed parent (its `matrix` is a world
 * matrix); a transformed ancestor is detected and warned about once.
 *
 * With a debug scene (`new Scene({ debug: true })`), the views the binding
 * copies from are checked `Proxy` objects (see `SceneOptions.debug` for
 * their cost); the mirrors it uploads are always plain typed arrays.
 *
 * The adapter only listens through {@link SceneLike.onChanges}: call
 * `scene.commitFrame()` once per animation frame, before rendering.
 */
export function bindMesh(scene: SceneLike, mesh: MeshHandle, options: BindMeshOptions = {}): BoundMesh {
  if (options.shading !== undefined && options.shading !== "vertex" && options.shading !== "fragment") {
    throw new FastMeshError("INVALID_ARGUMENT", 'shading must be "vertex" or "fragment"');
  }
  checkSoftness(options.edgeSoftness ?? 0);
  const ownsMaterial = options.material === undefined;
  const material: Material = options.material ?? new THREE.MeshStandardMaterial({ color: DEFAULT_COLOR });
  // Fragment shading needs the rule and the values (other hosts get vertex
  // colours) and a material built from three's shader chunks.
  const fragment =
    options.shading === "fragment" &&
    typeof scene.coloring === "function" &&
    typeof scene.views(mesh).values === "function" &&
    patchable(material);
  if (fragment && patched.has(material)) {
    throw new FastMeshError(
      "INVALID_ARGUMENT",
      "the material is already drawn by another fragment-shaded binding: give each binding its own (material.clone())",
    );
  }
  const keys: readonly BufferKey[] = fragment
    ? ["positions", "normals", "values", "indices"]
    : ["positions", "normals", "colors", "indices"];
  // The colour shown without colouring; vertex colours are multiplied by the
  // material colour, so it is white while colouring is on.
  if (options.color !== undefined) {
    material.color.set(options.color);
  }
  const baseColor = material.color.clone();

  const geometry = new THREE.BufferGeometry();
  const object = new THREE.Mesh(geometry, material);
  object.matrixAutoUpdate = false;

  // Slot counts the attributes were sized for.
  let vertexSlots = -1;
  let triangleSlots = -1;
  let colored = false;
  // Fragment shading: the rule drawn, its shading data, and the uniforms.
  let rule: ColorRule | null = null;
  let shading: Shading | undefined;
  const uniforms = fragment ? shadingUniforms(options.edgeSoftness ?? 0) : undefined;
  const restoreMaterial = uniforms === undefined ? undefined : patchMaterial(material, uniforms);

  /** The attribute of a buffer, or undefined when it is off. */
  const attributeOf = (key: BufferKey): THREE.BufferAttribute | undefined =>
    key === "indices"
      ? (geometry.getIndex() ?? undefined)
      : (geometry.getAttribute(BUFFERS[key].name) as THREE.BufferAttribute | undefined);

  /** A fresh view of a buffer, or null when it is off. */
  const viewOf = (key: BufferKey): Mirror | null => {
    const v = scene.views(mesh);
    switch (key) {
      case "positions":
        return v.positions();
      case "normals":
        return v.normals();
      case "colors":
        return v.colors();
      case "values":
        // Off without a rule to draw them by.
        return shading === undefined ? null : (v.values?.() ?? null);
      case "indices":
        return v.indices();
    }
  };

  /** Mirror length of a buffer at the current slot counts. */
  const lengthOf = (key: BufferKey): number =>
    key === "indices" ? 3 * triangleSlots : BUFFERS[key].itemSize * vertexSlots;

  /**
   * Copies `view[from, to)` into `mirror`, clamped to both. A lazy mesh's
   * normals can be shorter than its slots until it is updated; the rest of the
   * mirror then keeps its data. `subarray` gives a plain typed array even for
   * a debug-mode Proxy view. Colours are converted from sRGB to linear, and
   * values (one per slot, four mirror elements each) encoded.
   */
  const copy = (key: BufferKey, mirror: Mirror, view: Mirror, from: number, to: number) => {
    if (key === "values") {
      const end = Math.min(to / 4, view.length, mirror.length / 4);
      if (end > from / 4) {
        encodeValues(view as Float32Array, mirror as Float32Array, from / 4, end);
      }
      return;
    }
    const end = Math.min(to, view.length, mirror.length);
    if (end <= from) {
      return;
    }
    if (key === "colors") {
      copyColors(mirror as Uint16Array, view.subarray(from, end) as Uint8Array, from);
      return;
    }
    (mirror as Float32Array).set(view.subarray(from, end) as ArrayLike<number>, from);
  };

  /** A new attribute for `key` holding the whole view, installed in the geometry. */
  const create = (key: BufferKey, view: Mirror) => {
    const spec = BUFFERS[key];
    const mirror = spec.make(lengthOf(key));
    copy(key, mirror, view, 0, mirror.length);
    if (key === "indices") {
      geometry.setIndex(new THREE.BufferAttribute(mirror, 1));
    } else {
      geometry.setAttribute(spec.name, new THREE.BufferAttribute(mirror, spec.itemSize, spec.normalized));
    }
  };

  const remove = (key: BufferKey) => {
    if (key === "indices") {
      geometry.setIndex(null);
    } else {
      geometry.deleteAttribute(BUFFERS[key].name);
    }
  };

  const setColored = (on: boolean) => {
    if (on === colored) {
      return;
    }
    colored = on;
    material.color.copy(on ? WHITE : baseColor);
    if (uniforms !== undefined) {
      // The shader multiplies by the fragment's colour only while on: no recompile.
      uniforms.fmOn.value = on;
      return;
    }
    material.vertexColors = on;
    material.needsUpdate = true;
  };

  /**
   * Fragment shading: follows the scene's rule for the mesh. Returns whether
   * it changed, in which case every value is encoded again.
   */
  const followRule = (): boolean => {
    if (uniforms === undefined) {
      return false;
    }
    const now = scene.coloring?.(mesh) ?? null;
    if (now === rule) {
      return false;
    }
    rule = now;
    shading = now === null ? undefined : shadingOf(now);
    if (shading !== undefined) {
      setShadingUniforms(uniforms, shading);
    }
    return true;
  };

  /**
   * Live triangles' vertices only: dead slots keep stale positions. From the
   * scene's bounds when the host has them (one engine pass instead of two
   * walks over every slot), else walked from the mirrors.
   */
  const computeBounds = () => {
    const position = attributeOf("positions");
    const index = attributeOf("indices");
    if (position === undefined || index === undefined) {
      return;
    }
    const box = (geometry.boundingBox ??= new THREE.Box3());
    const sphere = (geometry.boundingSphere ??= new THREE.Sphere());
    const fromScene = typeof scene.bounds === "function" ? scene.bounds(mesh) : undefined;
    if (fromScene !== undefined) {
      if (fromScene === null) {
        // No live triangle: three's convention for an empty sphere.
        box.makeEmpty();
        sphere.set(new THREE.Vector3(), -1);
        return;
      }
      box.min.fromArray(fromScene.min);
      box.max.fromArray(fromScene.max);
      box.getCenter(sphere.center);
      sphere.radius = box.min.distanceTo(box.max) / 2;
      return;
    }
    const p = position.array as Float32Array;
    const t = index.array as Uint32Array;
    box.makeEmpty();
    const point = new THREE.Vector3();
    const live = (k: number) => t[k] !== 0 || t[k + 1] !== 0 || t[k + 2] !== 0;
    for (let k = 0; k < t.length; k += 3) {
      if (live(k)) {
        for (let c = 0; c < 3; c++) {
          box.expandByPoint(point.fromArray(p, 3 * t[k + c]));
        }
      }
    }
    if (box.isEmpty()) {
      // No live triangle: three's convention for an empty sphere.
      sphere.set(new THREE.Vector3(), -1);
      return;
    }
    box.getCenter(sphere.center);
    let r2 = 0;
    for (let k = 0; k < t.length; k += 3) {
      if (live(k)) {
        for (let c = 0; c < 3; c++) {
          r2 = Math.max(r2, sphere.center.distanceToSquared(point.fromArray(p, 3 * t[k + c])));
        }
      }
    }
    sphere.radius = Math.sqrt(r2);
  };

  const updateMatrix = () => {
    object.matrix.fromArray(scene.matrix4(mesh));
    object.matrixWorldNeedsUpdate = true;
  };

  /**
   * Brings every buffer of `change` (all of them when it is undefined) into
   * the geometry. Returns whether positions or indices changed.
   */
  const apply = (change: MeshChange | undefined): boolean => {
    const v = scene.views(mesh);
    const vs = v.positions().length / 3;
    const ts = v.indices().length / 3;
    const resized = vs !== vertexSlots || ts !== triangleSlots;
    vertexSlots = vs;
    triangleSlots = ts;
    let structural = false;
    let geometryChanged = false;
    const pending: (() => void)[] = [];
    const ruleChanged = followRule();
    for (const key of keys) {
      // Values change with the colours (all of them with the rule).
      const dirty: Range[] | "all" =
        change === undefined || (key === "values" && ruleChanged)
          ? "all"
          : change[key === "values" ? "colors" : key];
      if (dirty !== "all" && dirty.length === 0 && !resized) {
        continue;
      }
      const view = viewOf(key);
      const attribute = attributeOf(key);
      if (view === null) {
        if (attribute !== undefined) {
          structural = true;
          pending.push(() => remove(key));
        }
        continue;
      }
      if (key === "positions" || key === "indices") {
        geometryChanged = true;
      }
      if (attribute === undefined || attribute.array.length !== lengthOf(key)) {
        // Turned on, or the slot counts changed: a new attribute and mirror.
        structural = true;
        pending.push(() => create(key, view));
        continue;
      }
      const mirror = attribute.array as Mirror;
      const itemSize = BUFFERS[key].itemSize;
      if (dirty === "all") {
        // Same length: copy in place and mark the whole buffer. A range, not
        // an empty list: three uploads only the listed ranges once any exist,
        // so a later frame's ranges would otherwise drop this full update.
        copy(key, mirror, view, 0, mirror.length);
        attribute.addUpdateRange(0, mirror.length);
      } else {
        for (const [start, count] of dirty) {
          copy(key, mirror, view, start * itemSize, (start + count) * itemSize);
          attribute.addUpdateRange(start * itemSize, count * itemSize);
        }
      }
      if (attribute.updateRanges.length > MAX_UPDATE_RANGES) {
        // Not uploaded for a while: one full range instead of a long list.
        attribute.clearUpdateRanges();
        attribute.addUpdateRange(0, mirror.length);
      }
      attribute.needsUpdate = true;
    }
    if (structural) {
      // The attribute set changes: release the renderer's buffers and binding
      // states first, so replaced attributes do not leak GPU memory. Kept
      // attributes are uploaded whole on the next render.
      geometry.dispose();
      for (const run of pending) {
        run();
      }
      for (const key of keys) {
        attributeOf(key)?.clearUpdateRanges();
      }
    }
    setColored(attributeOf(fragment ? "values" : "colors") !== undefined);
    return geometryChanged;
  };

  apply(undefined);
  computeBounds();
  updateMatrix();
  checkUntransformedParent(object);

  let unsubscribe: (() => void) | undefined;
  const detach = () => {
    unsubscribe?.();
    unsubscribe = undefined;
  };
  unsubscribe = scene.onChanges((changes) => {
    // A parent may have moved with no change to this mesh.
    checkUntransformedParent(object);
    for (const c of changes) {
      if (c.mesh !== mesh) {
        continue;
      }
      if (c.removed) {
        // Detach: the mesh is gone and its handle is no longer valid.
        detach();
        return;
      }
      if (apply(c)) {
        computeBounds();
      }
      if (c.placementChanged) {
        updateMatrix();
      }
    }
  });

  let disposed = false;
  const bound: BoundMesh = {
    object,
    mesh,
    shading: fragment ? "fragment" : "vertex",
    setEdgeSoftness(width: number) {
      checkSoftness(width);
      if (uniforms !== undefined) {
        uniforms.fmSoftness.value = width;
      }
    },
    setOpacity(alpha: number) {
      if (typeof alpha !== "number" || !(alpha >= 0 && alpha <= 1)) {
        throw new FastMeshError("INVALID_ARGUMENT", "opacity must be a number in [0, 1]");
      }
      const transparent = alpha < 1;
      if (material.transparent !== transparent) {
        material.needsUpdate = true;
      }
      material.opacity = alpha;
      material.transparent = transparent;
      material.depthWrite = !transparent;
    },
    dispose() {
      detach();
      if (disposed) {
        return;
      }
      disposed = true;
      geometry.dispose();
      uniforms?.fmLut.value?.dispose();
      if (ownsMaterial) {
        material.dispose();
      } else {
        restoreMaterial?.();
      }
    },
  };
  if (options.opacity !== undefined) {
    bound.setOpacity(options.opacity);
  }
  return bound;
}

/** Objects already warned about a transformed ancestor. */
const warnedParents = new WeakSet<object>();

/** Whether the object's own transform is the identity. */
function isIdentityTransform(o: THREE.Object3D): boolean {
  if (!o.matrixAutoUpdate) {
    // Column-major: the diagonal is elements 0, 5, 10, 15.
    return o.matrix.elements.every((v, i) => v === (i % 5 === 0 ? 1 : 0));
  }
  const { position: p, quaternion: q, scale: s } = o;
  return (
    p.x === 0 &&
    p.y === 0 &&
    p.z === 0 &&
    q.x === 0 &&
    q.y === 0 &&
    q.z === 0 &&
    q.w === 1 &&
    s.x === 1 &&
    s.y === 1 &&
    s.z === 1
  );
}

/**
 * Warns once per object when it sits under a transformed ancestor: the
 * adapter drives its matrix as a world matrix, so a transformed parent
 * offsets it. Internal: shared with the controls and section bindings.
 */
export function checkUntransformedParent(object: THREE.Object3D): void {
  if (warnedParents.has(object)) {
    return;
  }
  for (let p = object.parent; p !== null; p = p.parent) {
    if (!isIdentityTransform(p)) {
      warnedParents.add(object);
      console.warn(
        "fast-mesh-wasm/three: the bound object has a transformed ancestor; " +
          "its matrix is driven as a world matrix, so add it under the scene or an untransformed group.",
      );
      return;
    }
  }
}

/** `INVALID_ARGUMENT` unless `width` is a finite number `>= 0`. */
function checkSoftness(width: number): void {
  if (typeof width !== "number" || !Number.isFinite(width) || width < 0) {
    throw new FastMeshError("INVALID_ARGUMENT", "edgeSoftness must be a finite number >= 0");
  }
}

/** The fragment shading's uniforms (see `SHADING_FRAGMENT`), shared with the material's shader. */
function shadingUniforms(softness: number) {
  return {
    fmLut: { value: null as THREE.DataTexture | null },
    fmCount: { value: 0 },
    fmBuiltin: { value: true },
    fmRange: { value: new THREE.Vector2() },
    fmCenter: { value: 0 },
    fmHasCenter: { value: false },
    fmBelowBase: { value: false },
    fmAboveBase: { value: true },
    fmBase: { value: new THREE.Vector4() },
    fmNan: { value: new THREE.Vector4() },
    fmSoftness: { value: softness },
    fmOn: { value: false },
  };
}

type ShadingUniforms = ReturnType<typeof shadingUniforms>;

/** Loads a rule's shading data into the uniforms; the texture is reused while its size holds. */
function setShadingUniforms(u: ShadingUniforms, s: Shading): void {
  let lut = u.fmLut.value;
  if (lut === null || lut.image.width !== s.width) {
    lut?.dispose();
    lut = new THREE.DataTexture(s.lut, s.width, LUT_ROWS, THREE.RGBAFormat, THREE.FloatType);
    // Read with texelFetch: no filtering, no mipmaps.
    lut.minFilter = THREE.NearestFilter;
    lut.magFilter = THREE.NearestFilter;
    lut.generateMipmaps = false;
    u.fmLut.value = lut;
  } else {
    (lut.image.data as Float32Array).set(s.lut);
  }
  lut.needsUpdate = true;
  u.fmCount.value = s.count;
  u.fmBuiltin.value = s.builtin;
  u.fmRange.value.set(s.range[0], s.range[1]);
  u.fmHasCenter.value = !Number.isNaN(s.center);
  u.fmCenter.value = Number.isNaN(s.center) ? 0 : s.center;
  u.fmBelowBase.value = s.belowBase;
  u.fmAboveBase.value = s.aboveBase;
  u.fmBase.value.fromArray(s.base);
  u.fmNan.value.fromArray(s.nan);
}

/** Materials a live fragment-shaded binding has patched. */
const patched = new WeakSet<object>();

/** The shader chunks fragment shading hooks into, in each stage. */
const HOOKS = {
  vertex: ["#include <common>", "#include <begin_vertex>"],
  fragment: ["#include <common>", "#include <color_fragment>"],
} as const;

/** `ShaderLib` entries of the materials fragment shading supports, by `material.type`. */
const SHADER_LIB: Record<string, string> = {
  MeshStandardMaterial: "standard",
  MeshPhysicalMaterial: "physical",
  MeshPhongMaterial: "phong",
  MeshLambertMaterial: "lambert",
  MeshBasicMaterial: "basic",
};

/**
 * Whether `WebGLRenderer` builds the material from three's chunks with the
 * hooks fragment shading needs; else warns that the binding uses vertex
 * colours.
 */
function patchable(material: Material): boolean {
  const lib = (THREE.ShaderLib as Record<string, { vertexShader: string; fragmentShader: string } | undefined>)[
    SHADER_LIB[material.type] ?? ""
  ];
  const ok =
    (material as { isNodeMaterial?: boolean }).isNodeMaterial !== true &&
    lib !== undefined &&
    HOOKS.vertex.every((h) => lib.vertexShader.includes(h)) &&
    HOOKS.fragment.every((h) => lib.fragmentShader.includes(h));
  if (!ok) {
    console.warn(
      `fast-mesh-wasm: shading "fragment" needs a WebGLRenderer material built from three's shader chunks ` +
        `(MeshStandardMaterial, MeshPhongMaterial, MeshLambertMaterial or MeshBasicMaterial); ` +
        `${material.type} is drawn with vertex colours.`,
    );
  }
  return ok;
}

/**
 * Adds the fragment shading to a material's shaders (`onBeforeCompile`, after
 * any handler it had) under its own program cache key, and disposes the
 * material's compiled program so the next render compiles it with these
 * uniforms; turns its `vertexColors` off meanwhile. Returns the function
 * that puts the material's handler, key and `vertexColors` back (disposing
 * the program again).
 */
function patchMaterial(material: Material, uniforms: ShadingUniforms): () => void {
  const before = material.onBeforeCompile;
  const key = material.customProgramCacheKey;
  // The shading colours each fragment from its value: vertex colours (an
  // attribute the binding may not even keep) would multiply in on top.
  const vertexColors = material.vertexColors;
  material.vertexColors = false;
  let warned = false;
  const hook = (source: string, chunk: string, add: string): string => {
    if (!source.includes(chunk)) {
      if (!warned) {
        warned = true;
        console.warn(`fast-mesh-wasm: fragment shading could not find ${chunk} in the material's shader; colours are not shown.`);
      }
      return source;
    }
    return source.replace(chunk, `${chunk}\n${add}`);
  };
  material.onBeforeCompile = function (this: Material, shader, renderer) {
    before.call(this, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = hook(
      hook(shader.vertexShader, HOOKS.vertex[0], SHADING_VERTEX),
      HOOKS.vertex[1],
      "\tvFmValue = fmValue;",
    );
    shader.fragmentShader = hook(
      hook(shader.fragmentShader, HOOKS.fragment[0], SHADING_FRAGMENT),
      HOOKS.fragment[1],
      "\tif (fmOn) diffuseColor *= fmShade(vFmValue);",
    );
  };
  material.customProgramCacheKey = function (this: Material) {
    return `${key.call(this)}|fast-mesh-fragment-shading`;
  };
  patched.add(material);
  // A program compiled before (by an earlier binding, or none) is dropped:
  // three runs onBeforeCompile again only for a new program.
  material.dispose();
  material.needsUpdate = true;
  return () => {
    material.onBeforeCompile = before;
    material.customProgramCacheKey = key;
    material.vertexColors = vertexColors;
    patched.delete(material);
    material.dispose();
    material.needsUpdate = true;
  };
}
