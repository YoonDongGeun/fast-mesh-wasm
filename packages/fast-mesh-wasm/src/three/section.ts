// `bindSection`: draws a scene section (contour lines and fill) and clips the
// bound meshes at its plane, following the change stream.
import * as THREE from "three";
import { FastMeshError } from "../core.js";
import type { SectionHandle, SectionPlane, SectionResult } from "../scene.js";
import type { SceneLike } from "../scene-like.js";
import { checkUntransformedParent, type MeshBinding } from "./mesh.js";

/** Options of {@link bindSection}. */
export interface BindSectionOptions {
  /** The contour line colour. Defaults to black. */
  lineColor?: THREE.ColorRepresentation;
  /** The fill colour. Defaults to a mid grey. */
  fillColor?: THREE.ColorRepresentation;
  /**
   * Whether to add {@link BoundSection.plane} to the bound meshes' materials'
   * `clippingPlanes`. Defaults to `true`.
   */
  clip?: boolean;
}

/** A section drawn by {@link bindSection}. */
export interface BoundSection {
  /**
   * Add it to the THREE.Scene, untransformed (results are in world
   * coordinates; a transformed ancestor is detected and warned about once).
   * Per bound mesh the section cuts, it holds a `LineSegments` of the contours
   * and a `Mesh` of the fill (empty while the section has no fill).
   */
  readonly group: THREE.Group;
  /**
   * The section plane, kept equal to the scene's. Three clips away what lies
   * on the plane's negative side, so the part on the side the section normal
   * points to stays visible.
   */
  readonly plane: THREE.Plane;
  /**
   * Stops following the scene, removes the plane from the materials (if the
   * section's removal has not already) and
   * disposes the geometries and the line and fill materials. Safe to call
   * more than once. The group stays wherever it was added.
   */
  dispose(): void;
}

const DEFAULT_LINE = 0x000000;
const DEFAULT_FILL = 0x888888;

/** One bound mesh's drawing: its contour lines and fill. */
interface Entry {
  bound: MeshBinding;
  lines: THREE.LineSegments;
  fill: THREE.Mesh;
}

/**
 * Line segment endpoints of a result's contours: consecutive point pairs,
 * plus the last-to-first pair of a closed contour.
 */
function segments(result: SectionResult): Float32Array {
  let count = 0;
  for (const c of result.contours) {
    const n = c.points.length / 3;
    count += n < 2 ? 0 : c.closed ? n : n - 1;
  }
  const out = new Float32Array(count * 6);
  let at = 0;
  for (const { points, closed } of result.contours) {
    const n = points.length / 3;
    if (n < 2) {
      continue;
    }
    const last = closed ? n : n - 1;
    for (let k = 0; k < last; k++) {
      const next = (k + 1) % n;
      out.set(points.subarray(3 * k, 3 * k + 3), at);
      out.set(points.subarray(3 * next, 3 * next + 3), at + 3);
      at += 6;
    }
  }
  return out;
}

/** The result for a mesh, or `null` when the section or mesh is gone or the section does not cut it. */
function resultOf(scene: SceneLike, section: SectionHandle, bound: MeshBinding): SectionResult | null {
  try {
    return scene.section(section, bound.mesh);
  } catch (error) {
    if (
      error instanceof FastMeshError &&
      (error.code === "INVALID_ARGUMENT" || error.code === "UNKNOWN_MESH" || error.code === "UNKNOWN_SECTION")
    ) {
      return null;
    }
    throw error;
  }
}

/**
 * Draws a section of the scene and clips the bound meshes at its plane.
 *
 * The plane is added to each bound mesh's `material.clippingPlanes` (unless
 * `clip` is `false`), which three applies only when the renderer has
 * `renderer.localClippingEnabled = true`. A mesh with an array of materials
 * has the plane added to each; a material shared by several bound meshes
 * gets it once.
 *
 * The lines and fill of a bound mesh are rebuilt when a frame's
 * `sectionsChanged` lists the section for it; a mesh the section does not
 * cut (or that is removed) draws nothing. Results are shown as last
 * computed: a `"lazy"` or `"manual"` mesh's lag until the scene computes
 * them. The plane follows `Scene.setSectionPlane` on the next frame.
 * When the section is removed, the drawing is emptied and stops following
 * the scene, and the plane is taken off the materials.
 *
 * Like `bindMesh`, it only listens through {@link SceneLike.onChanges}: call
 * `scene.commitFrame()` once per animation frame.
 */
export function bindSection(
  scene: SceneLike,
  section: SectionHandle,
  bound: readonly MeshBinding[],
  options: BindSectionOptions = {},
): BoundSection {
  const group = new THREE.Group();
  const plane = new THREE.Plane();
  const lineMaterial = new THREE.LineBasicMaterial({ color: options.lineColor ?? DEFAULT_LINE });
  const fillMaterial = new THREE.MeshBasicMaterial({
    color: options.fillColor ?? DEFAULT_FILL,
    side: THREE.DoubleSide,
  });

  // Last plane read, to touch `plane` only when it moved.
  let last = "";
  const updatePlane = (p: SectionPlane) => {
    const key = `${p.origin.join()},${p.normal.join()}`;
    if (key === last) {
      return;
    }
    last = key;
    plane.setFromNormalAndCoplanarPoint(new THREE.Vector3(...p.normal), new THREE.Vector3(...p.origin));
  };
  updatePlane(scene.sectionPlane(section));

  const entries: Entry[] = bound.map((b) => {
    const lines = new THREE.LineSegments(new THREE.BufferGeometry(), lineMaterial);
    const fill = new THREE.Mesh(new THREE.BufferGeometry(), fillMaterial);
    lines.matrixAutoUpdate = false;
    fill.matrixAutoUpdate = false;
    group.add(lines, fill);
    return { bound: b, lines, fill };
  });

  const rebuild = (e: Entry) => {
    const result = resultOf(scene, section, e.bound);
    const lines = new THREE.BufferGeometry();
    const fill = new THREE.BufferGeometry();
    if (result !== null) {
      lines.setAttribute("position", new THREE.BufferAttribute(segments(result), 3));
      if (result.fill !== undefined) {
        fill.setAttribute("position", new THREE.BufferAttribute(result.fill.positions, 3));
        fill.setIndex(new THREE.BufferAttribute(result.fill.indices, 1));
      }
    }
    lines.computeBoundingSphere();
    fill.computeBoundingSphere();
    e.lines.geometry.dispose();
    e.fill.geometry.dispose();
    e.lines.geometry = lines;
    e.fill.geometry = fill;
  };
  for (const e of entries) {
    rebuild(e);
  }

  // Clipping: add the plane once per material, remembering what to undo.
  const clipped: { material: THREE.Material; created: boolean }[] = [];
  if (options.clip ?? true) {
    for (const b of bound) {
      const materials = b.object.material;
      for (const material of Array.isArray(materials) ? materials : [materials]) {
        if (clipped.some((c) => c.material === material)) {
          continue;
        }
        const created = material.clippingPlanes === null;
        material.clippingPlanes = [...(material.clippingPlanes ?? []), plane];
        material.needsUpdate = true;
        clipped.push({ material, created });
      }
    }
  }
  /** Takes the plane off the materials it was added to. */
  const unclip = () => {
    for (const { material, created } of clipped.splice(0)) {
      const rest = (material.clippingPlanes ?? []).filter((p) => p !== plane);
      material.clippingPlanes = rest.length === 0 && created ? null : rest;
      material.needsUpdate = true;
    }
  };

  let unsubscribe: (() => void) | undefined;
  const detach = () => {
    unsubscribe?.();
    unsubscribe = undefined;
  };
  checkUntransformedParent(group);
  unsubscribe = scene.onChanges((changes) => {
    checkUntransformedParent(group);
    let plane: SectionPlane;
    try {
      plane = scene.sectionPlane(section);
    } catch (error) {
      if (error instanceof FastMeshError && error.code === "UNKNOWN_SECTION") {
        // Removed: nothing left to draw, clip or follow.
        for (const e of entries) {
          rebuild(e);
        }
        unclip();
        detach();
        return;
      }
      throw error;
    }
    updatePlane(plane);
    for (const c of changes) {
      if (!c.removed && !c.sectionsChanged.includes(section)) {
        continue;
      }
      for (const e of entries) {
        if (e.bound.mesh === c.mesh) {
          rebuild(e);
        }
      }
    }
  });

  let disposed = false;
  return {
    group,
    plane,
    dispose() {
      detach();
      if (disposed) {
        return;
      }
      disposed = true;
      unclip();
      for (const e of entries) {
        e.lines.geometry.dispose();
        e.fill.geometry.dispose();
      }
      lineMaterial.dispose();
      fillMaterial.dispose();
    },
  };
}
