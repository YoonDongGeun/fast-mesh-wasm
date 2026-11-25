// The Create dialog's form (#259): the shape, its parameters (kept per
// shape while the dialog is open, defaults scaled to the scene), the name
// (the shape's label made unique, until typed), centre and rotation; the
// live preview, regenerated once edits settle and posed as the centre and
// rotation change; and Create, which adds the mesh as one undo step, selects
// it and closes the dialog.
import type { PrimitiveMesh, PrimitiveShapeKind } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useEffect, useMemo, useState } from "react";
import type { Vec3 } from "../../../io/mesh-data";
import { type AppState, useApp } from "../../../app/state/app-state";
import { selectMesh } from "../../select/ux/selection";
import { createDialogOf } from "./create-dialog-state";
import {
  addPrimitive,
  defaultBaseSize,
  defaultParams,
  estimateTriangles,
  generateShape,
  paramsProblem,
  PREVIEW_DEBOUNCE_MS,
  type ShapeParams,
  shapeSpec,
  toShape,
  uniqueName,
} from "./create-primitive";

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** One component of `v` replaced. */
function withComponent(v: Vec3, axis: number, value: number): Vec3 {
  const out: Vec3 = [v[0], v[1], v[2]];
  out[axis] = value;
  return out;
}

export function useCreatePrimitive(app: AppState, initialKind: PrimitiveShapeKind) {
  const store = useScene();
  const [, select] = useSelection();
  const dialog = createDialogOf(app);
  const units = useApp(app, (s) => s.units);
  const items = useApp(app, (s) => s.items);
  // Sized once per opening, from the meshes listed then.
  const [baseSize] = useState(() => defaultBaseSize(items));
  const [kind, setKindState] = useState(initialKind);
  const [edited, setEdited] = useState<Partial<Record<PrimitiveShapeKind, ShapeParams>>>({});
  const params = useMemo(() => edited[kind] ?? defaultParams(kind, baseSize), [edited, kind, baseSize]);
  const spec = shapeSpec(kind);
  const [nameDraft, setNameDraft] = useState<string | null>(null);
  const name = nameDraft ?? uniqueName(spec.label, items.map((i) => i.name));
  const [centre, setCentre] = useState<Vec3>([0, 0, 0]);
  const [rotationDeg, setRotationDeg] = useState<Vec3>([0, 0, 0]);
  const problem = paramsProblem(kind, params);
  const triangles = estimateTriangles(kind, params);
  const shape = useMemo(() => toShape(kind, params), [kind, params]);
  const shapeKey = JSON.stringify(shape);
  const [generated, setGenerated] = useState<{ key: string; mesh: PrimitiveMesh } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Regenerate the preview once the shape settles; a later shape drops an
  // earlier one's result.
  useEffect(() => {
    if (problem !== null) {
      return;
    }
    let stale = false;
    const timer = setTimeout(() => {
      generateShape(store.scene, shape).then(
        (mesh) => {
          if (!stale) {
            setGenerated({ key: shapeKey, mesh });
            setError(null);
          }
        },
        (e: unknown) => {
          if (!stale) {
            setError(message(e));
          }
        },
      );
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
    // `shape` is what `shapeKey` spells.
  }, [store, shapeKey, problem]);

  // The last generated mesh stays up while the next one is made.
  const shown = problem === null ? (generated?.mesh ?? null) : null;
  useEffect(() => {
    dialog.setPreview(shown === null ? null : { mesh: shown, centre, rotationDeg });
  }, [dialog, shown, centre, rotationDeg]);
  useEffect(() => () => dialog.setPreview(null), [dialog]);

  const setKind = (next: PrimitiveShapeKind) => {
    setKindState(next);
    dialog.setKind(next);
  };

  const create = async () => {
    if (problem !== null || busy) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const mesh = generated?.key === shapeKey ? generated.mesh : await generateShape(store.scene, shape);
      const { item, done } = addPrimitive(store, app, mesh, {
        name: name.trim() === "" ? spec.label : name.trim(),
        centre,
        rotationDeg,
      });
      await done;
      selectMesh(app, select, store.getSnapshot().selection, item.mesh, false);
      dialog.close();
    } catch (e) {
      setError(message(e));
      setBusy(false);
    }
  };

  return {
    kind,
    setKind,
    spec,
    params,
    setParam: (key: string, value: number) => setEdited((prev) => ({ ...prev, [kind]: { ...params, [key]: value } })),
    resetParams: () =>
      setEdited((prev) => {
        const next = { ...prev };
        delete next[kind];
        return next;
      }),
    name,
    setName: setNameDraft,
    centre,
    setCentre: (axis: number, mm: number) => setCentre((c) => withComponent(c, axis, mm)),
    rotationDeg,
    setRotation: (axis: number, deg: number) => setRotationDeg((r) => withComponent(r, axis, deg)),
    units,
    triangles,
    problem,
    error,
    /** The preview shows an earlier shape (or none yet) while this one is made. */
    pending: problem === null && generated?.key !== shapeKey,
    busy,
    create,
    cancel: () => dialog.close(),
  };
}
