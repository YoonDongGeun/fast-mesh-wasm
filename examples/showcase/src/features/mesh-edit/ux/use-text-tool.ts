// The Text dialog's form (#260): the text and its letter height, depth,
// spacing and curve detail (kept in the app's Place Mesh settings, so the
// dialog, the Surface Stamp section and the viewport tool share one text),
// the name, centre and rotation of a created mesh; the live preview, rebuilt
// once edits settle; Create, which adds the text as a mesh in one undo step
// (as the Create dialog adds a shape), selects it and closes the dialog; and
// Emboss or Engrave, which arm Place Mesh with the text as its stamp.
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useEffect, useState } from "react";
import type { Vec3 } from "../../../io/mesh-data";
import { type AppState, useApp } from "../../../app/state/app-state";
import { addPrimitive, PREVIEW_DEBOUNCE_MS, uniqueName } from "../../home/ux/create-primitive";
import { useCreateDialog } from "../../home/ux/create-dialog-state";
import { selectMesh } from "../../select/ux/selection";
import { formatLength } from "../../../shared/ux/units";
import { textDialogOf } from "./text-dialog-state";
import { loadTextFont, type TypefaceFont } from "./text-font";
import { buildTextMesh, type TextMesh, type TextParams, textParamsProblem } from "./text-mesh";
import { armTextStamp, type TextStampMode } from "./text-stamp";

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** One component of `v` replaced. */
function withComponent(v: Vec3, axis: number, value: number): Vec3 {
  const out: Vec3 = [v[0], v[1], v[2]];
  out[axis] = value;
  return out;
}

/** The longest default name, in characters, before it is cut with an ellipsis. */
const NAME_CHARS = 24;

/** A created text's default name: its first line with words, cut to {@link NAME_CHARS}; `Text` when blank. */
export function defaultTextName(text: string): string {
  const line = text
    .split(/\r?\n|\r/)
    .map((l) => l.trim())
    .find((l) => l !== "");
  if (line === undefined) {
    return "Text";
  }
  const chars = Array.from(line);
  return chars.length > NAME_CHARS ? `${chars.slice(0, NAME_CHARS - 1).join("")}…` : line;
}

/** What the status line says about a built text: triangles and size, and any letters the font lacks. */
export function describeTextMesh(mesh: TextMesh, unit: Parameters<typeof formatLength>[1]): string {
  const [w, h, d] = mesh.size;
  const parts = [
    `${(mesh.indices.length / 3).toLocaleString("en")} triangles · ${formatLength(w, unit)} × ${formatLength(h, unit)} × ${formatLength(d, unit)}`,
  ];
  if (mesh.missing.length > 0) {
    parts.push(`no letter for ${mesh.missing.join(" ")} (drawn as ?)`);
  }
  if (mesh.skipped.length > 0) {
    parts.push(`left out ${mesh.skipped.join(" ")}`);
  }
  return parts.join(" · ");
}

/** Updates `app`'s text (the Place Mesh text) with `patch`. */
export function updateText(app: AppState, patch: Partial<TextParams>): void {
  app.updateWave6((prev) => ({
    ...prev,
    printPrep: {
      ...prev.printPrep,
      placeParams: { ...prev.printPrep.placeParams, text: { ...prev.printPrep.placeParams.text, ...patch } },
    },
  }));
}

export function useTextTool(app: AppState) {
  const store = useScene();
  const [, select] = useSelection();
  const dialog = textDialogOf(app);
  const units = useApp(app, (s) => s.units);
  const items = useApp(app, (s) => s.items);
  const text = useApp(app, (s) => s.wave6.printPrep.placeParams.text);
  const createOpen = useCreateDialog(app, (s) => s.open);
  const [font, setFont] = useState<TypefaceFont | null>(null);
  const [nameDraft, setNameDraft] = useState<string | null>(null);
  const name = nameDraft ?? uniqueName(defaultTextName(text.text), items.map((i) => i.name));
  const [centre, setCentre] = useState<Vec3>([0, 0, 0]);
  const [rotationDeg, setRotationDeg] = useState<Vec3>([0, 0, 0]);
  const problem = textParamsProblem(text);
  const key = JSON.stringify(text);
  const [built, setBuilt] = useState<{ key: string; mesh: TextMesh } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // The Create dialog floats in the same place: opening it closes this one.
  useEffect(() => {
    if (createOpen) {
      dialog.close();
    }
  }, [createOpen, dialog]);

  // The font loads when the dialog first opens.
  useEffect(() => {
    let live = true;
    loadTextFont().then(
      (f) => {
        if (live) {
          setFont(f);
        }
      },
      (e: unknown) => {
        if (live) {
          setError(`The font did not load: ${message(e)}`);
        }
      },
    );
    return () => {
      live = false;
    };
  }, []);

  // Rebuild once the text settles (at once for the first one).
  const firstBuild = built === null;
  useEffect(() => {
    if (font === null || problem !== null) {
      return;
    }
    const params = JSON.parse(key) as TextParams;
    const timer = setTimeout(
      () => {
        try {
          setBuilt({ key, mesh: buildTextMesh(font, params, { centreZ: true }) });
          setError(null);
        } catch (e) {
          setError(message(e));
        }
      },
      firstBuild ? 0 : PREVIEW_DEBOUNCE_MS,
    );
    return () => clearTimeout(timer);
  }, [font, key, problem, firstBuild]);

  // The last built text stays up while the next one is made.
  const shown = problem === null ? (built?.mesh ?? null) : null;
  useEffect(() => {
    dialog.setPreview(shown === null ? null : { mesh: shown, centre, rotationDeg });
  }, [dialog, shown, centre, rotationDeg]);
  useEffect(() => () => dialog.setPreview(null), [dialog]);

  const create = async () => {
    if (problem !== null || busy || font === null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const mesh = built?.key === key ? built.mesh : buildTextMesh(font, text, { centreZ: true });
      if (mesh.indices.length === 0) {
        throw new Error("The text has no letters to build.");
      }
      const { item, done } = addPrimitive(store, app, mesh, {
        name: name.trim() === "" ? defaultTextName(text.text) : name.trim(),
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

  /** Arms Place Mesh with this text, to emboss or engrave it where the surface is clicked. */
  const placeOnSurface = (mode: TextStampMode) => {
    if (problem !== null) {
      return;
    }
    armTextStamp(app, text, mode);
    dialog.close();
  };

  const pending = problem === null && (font === null || built?.key !== key);
  let status: { text: string; problem: boolean };
  if (problem !== null || error !== null) {
    status = { text: (problem ?? error) as string, problem: true };
  } else if (font === null) {
    status = { text: "Loading the font…", problem: false };
  } else if (built === null) {
    status = { text: "Building the text…", problem: false };
  } else {
    status = {
      text: `${describeTextMesh(built.mesh, units)}${pending ? " · updating the preview…" : ""}`,
      problem: false,
    };
  }

  return {
    text,
    setText: (patch: Partial<TextParams>) => updateText(app, patch),
    name,
    setName: setNameDraft,
    centre,
    setCentre: (axis: number, mm: number) => setCentre((c) => withComponent(c, axis, mm)),
    rotationDeg,
    setRotation: (axis: number, deg: number) => setRotationDeg((r) => withComponent(r, axis, deg)),
    units,
    status,
    /** Whether the text can be created or placed now. */
    ready: problem === null && font !== null,
    pending,
    busy,
    create,
    placeOnSurface,
    cancel: () => dialog.close(),
  };
}
