// The owner's scene in raw WebGPU: three meshes you can pick, drag, rotate,
// colour by distance, make transparent and cut with a section plane. The
// scene (fast-mesh-wasm) owns geometry, placements, history, fields and
// sections; this file only turns input into scene calls, and once per frame
// hands `scene.commitFrame()` to the renderer.
import {
  type DragSession,
  type FieldHandle,
  init,
  type MeshHandle,
  type Placement,
  type PlacementInput,
  Scene,
  type SectionHandle,
  type Vec3,
} from "fast-mesh-wasm";
import { add, type CameraBasis, cross, dot, normalize, OrbitCamera, scale, sub } from "./camera";
import { type DrawItem, Renderer } from "./gpu";
import { type MeshData, parseBinaryStl, sphere, torus, wavyPlate } from "./meshes";

const canvas = document.querySelector<HTMLCanvasElement>("#view")!;
const statusLine = document.querySelector<HTMLElement>("#status")!;
const message = document.querySelector<HTMLElement>("#message")!;

const renderer = await Renderer.create(canvas).catch((error: unknown) => {
  console.error(error);
  return null;
});
if (renderer === null) {
  message.textContent =
    "WebGPU is not available in this browser. Use a recent Chrome, Edge or Safari " +
    "(or enable WebGPU), then reload.";
  message.hidden = false;
  statusLine.hidden = true;
} else {
  await init();
  run(renderer);
}

/** Per-mesh display settings: never part of the scene's history. */
interface Look {
  name: string;
  color: [number, number, number];
  opacity: number;
}

/** A pointer gesture in progress. The drag session starts on the first real move. */
type Gesture =
  | { kind: "orbit"; x: number; y: number; yaw: number; pitch: number }
  | {
      kind: "move" | "rotate";
      mesh: MeshHandle;
      x: number;
      y: number;
      start: Placement;
      /** move: the picked surface point; rotate: the mesh's world centre. */
      anchor: Vec3;
      camera: CameraBasis;
      session?: DragSession;
    };
type MeshGesture = Exclude<Gesture, { kind: "orbit" }>;

function run(renderer: Renderer): void {
  const scene = new Scene();
  const camera = new OrbitCamera();
  const looks = new Map<MeshHandle, Look>();
  const bounds = { min: [Infinity, Infinity, Infinity] as Vec3, max: [-Infinity, -Infinity, -Infinity] as Vec3 };
  const centre = (): Vec3 => scale(add(bounds.min, bounds.max), 0.5);
  const diagonal = () => Math.hypot(...sub(bounds.max, bounds.min));

  let picked: MeshHandle | null = null;
  let gesture: Gesture | null = null;
  /** The latest drag pose, applied once at the start of the next frame. */
  let pendingPose: PlacementInput | null = null;
  let coloring: { mesh: MeshHandle; field: FieldHandle } | null = null;
  let section: { handle: SectionHandle; origin: Vec3; dirty: boolean } | null = null;
  let note = "";

  function addMesh(name: string, data: MeshData, color: Look["color"]): void {
    const mesh = scene.addMesh(data.positions, data.indices);
    looks.set(mesh, { name, color, opacity: 1 });
    const p = data.positions;
    for (let i = 0; i < p.length; i++) {
      bounds.min[i % 3] = Math.min(bounds.min[i % 3], p[i]);
      bounds.max[i % 3] = Math.max(bounds.max[i % 3], p[i]);
    }
    camera.target = centre();
    camera.distance = 1.1 * diagonal();
  }

  addMesh("sphere", sphere([-2.4, 0, 0], 1), [0.85, 0.45, 0.4]);
  addMesh("torus", torus([0, 0, 0], 0.9, 0.35), [0.4, 0.7, 0.85]);
  addMesh("wavy plate", wavyPlate([2.6, 0, 0], 2, 0.15), [0.6, 0.8, 0.45]);

  // ---- Per frame: drag pose → commitFrame → GPU buffers → draw ----

  function frame(): void {
    const session = gesture?.kind === "orbit" ? undefined : gesture?.session;
    if (pendingPose !== null && session?.active) {
      session.move(pendingPose); // one move per frame, before commitFrame
    }
    pendingPose = null;

    for (const change of scene.commitFrame()) {
      renderer.applyChange(scene, change);
      if (section !== null && change.sectionsChanged.includes(section.handle)) {
        section.dirty = true;
      }
    }
    if (section?.dirty) {
      section.dirty = false;
      renderer.setLines(contourSegments(section.handle));
    }

    const aspect = renderer.resize();
    const { eye } = camera.basis();
    const items: DrawItem[] = [...looks].map(([mesh, look]) => ({
      mesh,
      baseColor: look.color,
      opacity: look.opacity,
      picked: mesh === picked,
    }));
    const clip = section === null ? null : ([0, 0, 1, -section.origin[2]] as const);
    renderer.render(scene, { viewProj: camera.viewProj(aspect), eye, clip }, items);
    showStatus();
    requestAnimationFrame(frame);
  }

  /** A section's contours as line-list segments (world xyz pairs). */
  function contourSegments(handle: SectionHandle): Float32Array {
    const meshes = [...looks.keys()];
    if (meshes.some((m) => scene.section(handle, m).status === "stale")) {
      scene.updateSections();
    }
    const out: number[] = [];
    for (const mesh of meshes) {
      for (const { points, closed } of scene.section(handle, mesh).contours) {
        const n = points.length / 3;
        for (let i = 0; i < (closed ? n : n - 1); i++) {
          const a = 3 * i;
          const b = 3 * ((i + 1) % n);
          out.push(points[a], points[a + 1], points[a + 2], points[b], points[b + 1], points[b + 2]);
        }
      }
    }
    return Float32Array.from(out);
  }

  // ---- Pointer: click picks, drag moves, Shift+drag rotates, empty space orbits ----

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const ray = camera.ray(e.offsetX, e.offsetY, canvas.clientWidth, canvas.clientHeight);
    const hit = scene.raycast(ray.origin, ray.direction);
    picked = hit?.mesh ?? null;
    if (hit === null) {
      gesture = { kind: "orbit", x: e.offsetX, y: e.offsetY, yaw: camera.yaw, pitch: camera.pitch };
      return;
    }
    const m = scene.matrix4(hit.mesh);
    gesture = {
      kind: e.shiftKey ? "rotate" : "move",
      mesh: hit.mesh,
      x: e.offsetX,
      y: e.offsetY,
      start: scene.placement(hit.mesh),
      anchor: e.shiftKey ? [m[12], m[13], m[14]] : hit.point,
      camera: camera.basis(),
    };
  });

  canvas.addEventListener("pointermove", (e) => {
    if (gesture === null) {
      return;
    }
    if (gesture.kind === "orbit") {
      camera.yaw = gesture.yaw - 0.01 * (e.offsetX - gesture.x);
      camera.pitch = Math.max(-1.5, Math.min(1.5, gesture.pitch + 0.01 * (e.offsetY - gesture.y)));
      return;
    }
    if (gesture.session === undefined) {
      if (Math.hypot(e.offsetX - gesture.x, e.offsetY - gesture.y) < 3) {
        return; // still a click
      }
      gesture.session = scene.drag(gesture.mesh); // the whole drag is one undo step
    }
    pendingPose = gesture.kind === "move" ? movePose(gesture, e) : rotatePose(gesture, e);
  });

  const endGesture = () => {
    if (gesture !== null && gesture.kind !== "orbit" && gesture.session?.active) {
      if (pendingPose !== null) {
        gesture.session.move(pendingPose);
      }
      gesture.session.end();
    }
    gesture = null;
    pendingPose = null;
  };
  canvas.addEventListener("pointerup", endGesture);
  canvas.addEventListener("pointercancel", endGesture);
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      camera.distance *= Math.exp(0.001 * e.deltaY);
    },
    { passive: false },
  );

  /** Translate in the camera-facing plane through the picked point. */
  function movePose(g: MeshGesture, e: PointerEvent): PlacementInput {
    const ray = camera.ray(e.offsetX, e.offsetY, canvas.clientWidth, canvas.clientHeight);
    const n = g.camera.forward;
    const t = dot(sub(g.anchor, ray.origin), n) / dot(ray.direction, n);
    const delta = sub(add(ray.origin, scale(ray.direction, t)), g.anchor);
    return { rotation: g.start.rotation, translation: add(g.start.translation, delta) };
  }

  /** Arcball: rotate about the mesh's world centre by the arc between two sphere points. */
  function rotatePose(g: MeshGesture, e: PointerEvent): PlacementInput {
    const a = arcball(g.x, g.y);
    const b = arcball(e.offsetX, e.offsetY);
    const angle = Math.acos(Math.max(-1, Math.min(1, dot(a, b))));
    const axisCamera = cross(a, b);
    if (angle < 1e-9 || dot(axisCamera, axisCamera) < 1e-18) {
      return g.start;
    }
    // Camera space (x right, y up, z towards the viewer) to world.
    const { right, up, forward } = g.camera;
    const axis = normalize(
      add(add(scale(right, axisCamera[0]), scale(up, axisCamera[1])), scale(forward, -axisCamera[2])),
    );
    const q = axisAngle(axis, angle);
    // world = R·p + t, rotated by Q about c: (Q·R)·p + Q·(t − c) + c.
    const r = g.start.rotation;
    const rotation: number[] = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        rotation.push(q[3 * i] * r[j] + q[3 * i + 1] * r[3 + j] + q[3 * i + 2] * r[6 + j]);
      }
    }
    const d = sub(g.start.translation, g.anchor);
    const qd: Vec3 = [dot([q[0], q[1], q[2]], d), dot([q[3], q[4], q[5]], d), dot([q[6], q[7], q[8]], d)];
    return { rotation, translation: add(qd, g.anchor) };
  }

  /** A pixel on the virtual trackball, as a unit vector in camera space. */
  function arcball(x: number, y: number): Vec3 {
    const size = Math.min(canvas.clientWidth, canvas.clientHeight);
    const nx = (2 * x - canvas.clientWidth) / size;
    const ny = (canvas.clientHeight - 2 * y) / size;
    const d2 = nx * nx + ny * ny;
    return d2 <= 1 ? [nx, ny, Math.sqrt(1 - d2)] : normalize([nx, ny, 0]);
  }

  // ---- Keys ----

  window.addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;
    note = "";
    if (ctrl && key === "z") {
      e.preventDefault();
      // An active drag is ended first, then undone.
      if (e.shiftKey) {
        scene.redo();
      } else {
        scene.undo();
      }
    } else if (ctrl && key === "y") {
      e.preventDefault();
      scene.redo();
    } else if (ctrl || e.altKey) {
      return;
    } else if (key === "c") {
      toggleColoring();
    } else if (key === "s") {
      toggleSection();
    } else if (/^[0-9]$/.test(key)) {
      if (picked === null) {
        note = "pick a mesh first";
      } else {
        looks.get(picked)!.opacity = key === "0" ? 1 : Number(key) / 10;
      }
    }
  });

  /** Distance colouring of the picked mesh against its nearest other mesh. */
  function toggleColoring(): void {
    if (coloring !== null) {
      scene.setColoring(coloring.mesh, null);
      scene.removeField(coloring.field);
      coloring = null;
      return;
    }
    if (picked === null) {
      note = "pick a mesh first";
      return;
    }
    let nearest: MeshHandle | null = null;
    let best = Infinity;
    for (const other of looks.keys()) {
      const d = other === picked ? null : scene.meshDistance(picked, other);
      if (d !== null && d.distance < best) {
        best = d.distance;
        nearest = other;
      }
    }
    if (nearest === null) {
      return;
    }
    const maxDistance = 0.1 * diagonal();
    const field = scene.addDistanceField({ name: "distance", on: picked, to: nearest, maxDistance });
    // Farther than maxDistance (Infinity) shows the base grey.
    scene.setColoring(picked, { field, map: "turbo", range: [0, maxDistance] });
    coloring = { mesh: picked, field };
  }

  /** A section plane through the scene centre (normal +z), cutting every mesh. */
  function toggleSection(): void {
    if (section !== null) {
      scene.removeSection(section.handle);
      section = null;
      renderer.setLines(new Float32Array());
      return;
    }
    const origin = centre();
    const handle = scene.addSection({ plane: { origin, normal: [0, 0, 1] }, meshes: [...looks.keys()] });
    section = { handle, origin, dirty: true };
  }

  // ---- Drop a binary STL to add it ----

  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0];
    if (file === undefined) {
      return;
    }
    void file.arrayBuffer().then((buffer) => {
      const data = parseBinaryStl(buffer);
      if (data === null) {
        note = `${file.name} is not a binary STL`;
        return;
      }
      addMesh(file.name, data, [0.8, 0.8, 0.75]);
      if (section !== null) {
        section.origin = centre();
        scene.setSectionMeshes(section.handle, [...looks.keys()]);
        scene.setSectionPlane(section.handle, { origin: section.origin, normal: [0, 0, 1] });
        section.dirty = true;
      }
    });
  });

  // ---- Status line ----

  let shown = "";
  function showStatus(): void {
    const look = picked === null ? undefined : looks.get(picked);
    const text = [
      look === undefined ? "nothing picked" : `picked: ${look.name} (opacity ${look.opacity})`,
      coloring === null ? "" : "distance colouring on",
      section === null ? "" : "section on",
      `undo ${scene.canUndo() ? "yes" : "no"} / redo ${scene.canRedo() ? "yes" : "no"}`,
      note,
    ]
      .filter((s) => s !== "")
      .join("  ·  ");
    if (text !== shown) {
      shown = text;
      statusLine.textContent = text;
    }
  }

  requestAnimationFrame(frame);
}

/** The row-major rotation by `angle` about the unit `axis` (Rodrigues). */
function axisAngle(axis: Vec3, angle: number): number[] {
  const [x, y, z] = axis;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const k = 1 - c;
  // prettier-ignore
  return [
    c + x * x * k, x * y * k - z * s, x * z * k + y * s,
    y * x * k + z * s, c + y * y * k, y * z * k - x * s,
    z * x * k - y * s, z * y * k + x * s, c + z * z * k,
  ];
}
