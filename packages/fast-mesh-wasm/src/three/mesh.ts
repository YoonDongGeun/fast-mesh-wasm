// `bindMesh`: a THREE.Mesh whose BufferGeometry follows a scene mesh through
// the change stream. Attributes own JS-side mirror arrays (spec §11.2): three
// uploads lazily, so handing it wasm views would leave it reading detached or
// moved memory at render time. Each frame copies only the dirty ranges from
// fresh views into the mirrors.
import * as THREE from "three";
import { FastMeshError } from "../core.js";
import type { MeshChange, MeshHandle, Range, Scene } from "../scene.js";

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
   * each its own (`material.clone()`).
   */
  material?: THREE.MeshStandardMaterial | THREE.MeshPhongMaterial | THREE.MeshLambertMaterial;
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
}

/** A scene mesh bound to a THREE.Mesh by {@link bindMesh}. */
export interface BoundMesh {
  /**
   * The mesh to add to a THREE.Scene. Its geometry has `position`, `normal`
   * (when the scene mesh keeps normals), `color` (while colouring is on) and
   * an index; its `matrix` is the scene mesh's world matrix
   * (`matrixAutoUpdate` is off).
   */
  readonly object: THREE.Mesh;
  readonly mesh: MeshHandle;
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

type Material = NonNullable<BindMeshOptions["material"]>;
type Mirror = Float32Array | Uint8Array | Uint32Array;

const DEFAULT_COLOR = 0xcccccc;
const WHITE = new THREE.Color(0xffffff);

/** How each buffer maps to an attribute: components per element and the mirror type. */
const BUFFERS = {
  positions: { name: "position", itemSize: 3, normalized: false, make: (n: number) => new Float32Array(n) },
  normals: { name: "normal", itemSize: 3, normalized: false, make: (n: number) => new Float32Array(n) },
  colors: { name: "color", itemSize: 4, normalized: true, make: (n: number) => new Uint8Array(n) },
  indices: { name: "index", itemSize: 3, normalized: false, make: (n: number) => new Uint32Array(n) },
} as const;

type BufferKey = keyof typeof BUFFERS;

/**
 * Binds a scene mesh to a new THREE.Mesh and keeps it in sync on every
 * {@link Scene.commitFrame}: dirty ranges are copied into the attributes'
 * mirrors and marked with `addUpdateRange`, attributes are recreated when the
 * mesh's slot counts change, the colour attribute comes and goes with the
 * mesh's colouring, and `object.matrix` follows its placement.
 *
 * Dead vertex slots keep stale positions, so the geometry's bounding box and
 * sphere are computed from the vertices of live triangles only (a removed
 * triangle reads `0, 0, 0` in the index). They are recomputed when positions
 * or indices change, never on placement-only frames.
 *
 * When the scene mesh is removed, the binding detaches: it stops following
 * the scene and leaves the object as it was (remove it from your THREE.Scene
 * and call {@link BoundMesh.dispose}).
 *
 * The adapter only listens through {@link Scene.onChanges}: call
 * `scene.commitFrame()` once per animation frame, before rendering.
 */
export function bindMesh(scene: Scene, mesh: MeshHandle, options: BindMeshOptions = {}): BoundMesh {
  const ownsMaterial = options.material === undefined;
  const material: Material = options.material ?? new THREE.MeshStandardMaterial({ color: DEFAULT_COLOR });
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
   * a debug-mode Proxy view.
   */
  const copy = (mirror: Mirror, view: Mirror, from: number, to: number) => {
    const end = Math.min(to, view.length, mirror.length);
    if (end > from) {
      (mirror as Float32Array).set(view.subarray(from, end) as ArrayLike<number>, from);
    }
  };

  /** A new attribute for `key` holding the whole view, installed in the geometry. */
  const create = (key: BufferKey, view: Mirror) => {
    const spec = BUFFERS[key];
    const mirror = spec.make(lengthOf(key));
    copy(mirror, view, 0, mirror.length);
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
    material.vertexColors = on;
    material.color.copy(on ? WHITE : baseColor);
    material.needsUpdate = true;
  };

  /** Live triangles' vertices only: dead slots keep stale positions. */
  const computeBounds = () => {
    const position = attributeOf("positions");
    const index = attributeOf("indices");
    if (position === undefined || index === undefined) {
      return;
    }
    const p = position.array as Float32Array;
    const t = index.array as Uint32Array;
    const box = (geometry.boundingBox ??= new THREE.Box3());
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
    const sphere = (geometry.boundingSphere ??= new THREE.Sphere());
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
    const vs = scene.vertexSlots(mesh);
    const ts = scene.triangleSlots(mesh);
    const resized = vs !== vertexSlots || ts !== triangleSlots;
    vertexSlots = vs;
    triangleSlots = ts;
    let structural = false;
    let geometryChanged = false;
    const pending: (() => void)[] = [];
    for (const key of Object.keys(BUFFERS) as BufferKey[]) {
      const dirty: Range[] | "all" = change === undefined ? "all" : change[key];
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
        copy(mirror, view, 0, mirror.length);
        attribute.addUpdateRange(0, mirror.length);
      } else {
        for (const [start, count] of dirty) {
          copy(mirror, view, start * itemSize, (start + count) * itemSize);
          attribute.addUpdateRange(start * itemSize, count * itemSize);
        }
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
      for (const key of Object.keys(BUFFERS) as BufferKey[]) {
        attributeOf(key)?.clearUpdateRanges();
      }
    }
    setColored(attributeOf("colors") !== undefined);
    return geometryChanged;
  };

  apply(undefined);
  computeBounds();
  updateMatrix();

  let unsubscribe: (() => void) | undefined;
  const detach = () => {
    unsubscribe?.();
    unsubscribe = undefined;
  };
  unsubscribe = scene.onChanges((changes) => {
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
      if (ownsMaterial) {
        material.dispose();
      }
    },
  };
  if (options.opacity !== undefined) {
    bound.setOpacity(options.opacity);
  }
  return bound;
}
