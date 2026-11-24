// The Image to Relief dialog's form (#267): the image (picked, or dropped on
// the page), read in the image Worker; the relief's settings, which a height
// map's JSON fills in; the live preview, a coarse relief generated in the
// scene's Worker once edits settle; and Create, which samples the image in
// full and adds the relief as one undo step, selects it and closes the dialog.
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { generateRelief, runAddRelief } from "../../../engine/workbench-ops";
import { type HeightImageInfo, parseHeightMapFile } from "../../../io/height-image";
import { closeHeightImage, heightImageGrid, openHeightImage } from "../../../io/height-image-client";
import type { Vec3 } from "../../../io/mesh-data";
import { useCreateDialog } from "../../home/ux/create-dialog-state";
import { defaultBaseSize, PREVIEW_DEBOUNCE_MS, uniqueName } from "../../home/ux/create-primitive";
import { selectMesh } from "../../select/ux/selection";
import { eulerDegToMat3 } from "../../transform/ux/transform-math";
import {
  defaultRelief,
  describeRelief,
  maxSamples,
  PREVIEW_SAMPLES,
  type ReliefFiles,
  reliefGrid,
  reliefName,
  reliefOfFrame,
  type ReliefParams,
  reliefProblem,
  splitReliefFiles,
} from "./relief";
import { reliefDialogOf, type ReliefPreview, useReliefDialog } from "./relief-dialog-state";
import { useTextDialog } from "./text-dialog-state";

/** The thumbnail's longer side, in samples. */
const THUMBNAIL = 120;

/** An open image: what the Worker keeps, its file, and a small picture of it. */
interface OpenImage extends HeightImageInfo {
  name: string;
  thumbnail: { width: number; height: number; grey: Float32Array };
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** One component of `v` replaced. */
function withComponent(v: Vec3, axis: number, value: number): Vec3 {
  const out: Vec3 = [v[0], v[1], v[2]];
  out[axis] = value;
  return out;
}

export function useReliefTool(app: AppState) {
  const store = useScene();
  const [, select] = useSelection();
  const dialog = reliefDialogOf(app);
  const units = useApp(app, (s) => s.units);
  const items = useApp(app, (s) => s.items);
  const files = useReliefDialog(app, (s) => s.files);
  const createOpen = useCreateDialog(app, (s) => s.open);
  const textOpen = useTextDialog(app, (s) => s.open);
  // Sized once per opening, from the meshes listed then.
  const [baseSize] = useState(() => defaultBaseSize(items));
  const [image, setImage] = useState<OpenImage | null>(null);
  const [params, setParams] = useState<ReliefParams | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [nameDraft, setNameDraft] = useState<string | null>(null);
  const [centre, setCentre] = useState<Vec3>([0, 0, 0]);
  const [rotationDeg, setRotationDeg] = useState<Vec3>([0, 0, 0]);
  const [preview, setPreview] = useState<{ key: string; mesh: ReliefPreview["mesh"] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reading, setReading] = useState(false);
  const [busy, setBusy] = useState(false);
  /** Counts the images asked for: a later one drops an earlier one's result. */
  const asked = useRef(0);
  const open = useRef<OpenImage | null>(null);

  const read = useCallback(
    async (taken: ReliefFiles) => {
      const mine = ++asked.current;
      setReading(true);
      setError(null);
      try {
        const info = await openHeightImage(taken.image);
        if (mine !== asked.current) {
          closeHeightImage(info.id);
          return;
        }
        const k = Math.min(1, THUMBNAIL / Math.max(info.width, info.height));
        const tw = Math.max(2, Math.round(info.width * k));
        const th = Math.max(2, Math.round(info.height * k));
        const grey = await heightImageGrid(info.id, tw, th, { scale: 255, invert: false });
        const text = taken.json === null ? null : await taken.json.text();
        if (mine !== asked.current) {
          closeHeightImage(info.id);
          return;
        }
        const next: OpenImage = {
          ...info,
          name: taken.image.name,
          thumbnail: { width: tw, height: th, grey },
        };
        if (open.current !== null) {
          closeHeightImage(open.current.id);
        }
        open.current = next;
        setImage(next);
        const fallback = defaultRelief(next, baseSize);
        const frame = text === null ? null : parseHeightMapFile(text);
        const fitted = frame === null ? null : reliefOfFrame(frame, next, fallback.base);
        setParams(fitted?.params ?? fallback);
        setCentre(fitted?.place.centre ?? [0, 0, 0]);
        setRotationDeg(fitted?.place.rotationDeg ?? [0, 0, 0]);
        if (fitted !== null) {
          setNote("Size, height and place are from the height map's JSON.");
        } else if (taken.json !== null) {
          setNote(
            frame === null
              ? `${taken.json.name} is not a height map's JSON: it was left out.`
              : `${taken.json.name} is for another image (${frame.width} × ${frame.height} px): it was left out.`,
          );
        } else {
          setNote(null);
        }
        setNameDraft(null);
      } catch (e) {
        if (mine === asked.current) {
          setError(message(e));
        }
      } finally {
        if (mine === asked.current) {
          setReading(false);
        }
      }
    },
    [baseSize],
  );

  // The files the dialog was opened with, or that were dropped while it is open.
  useEffect(() => {
    if (files !== null) {
      void read(files);
    }
  }, [files, read]);

  // Closing the dialog drops the image.
  useEffect(
    () => () => {
      asked.current++;
      if (open.current !== null) {
        closeHeightImage(open.current.id);
        open.current = null;
      }
    },
    [],
  );

  // The other dialogs float in the same place: opening one closes this one.
  useEffect(() => {
    if (createOpen || textOpen) {
      dialog.close();
    }
  }, [createOpen, textOpen, dialog]);

  const problem = image === null || params === null ? null : reliefProblem(params, image);
  const key = image === null || params === null ? "" : JSON.stringify([image.id, params]);

  // Regenerate the preview once the settings settle; later settings drop an
  // earlier result.
  useEffect(() => {
    if (image === null || params === null || problem !== null) {
      return;
    }
    let stale = false;
    const timer = setTimeout(() => {
      const { w, h, pixelSize } = reliefGrid(params, image, PREVIEW_SAMPLES);
      heightImageGrid(image.id, w, h, { scale: params.maxHeight, invert: params.invert })
        .then((heights) => generateRelief(store.scene, { heights, width: w, height: h, pixelSize, base: params.base }))
        .then(
          (mesh) => {
            if (!stale) {
              setPreview({ key, mesh });
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
    // `image` and `params` are what `key` spells.
  }, [store, key, problem]);

  // The last generated relief stays up while the next one is made.
  const shown = image !== null && problem === null ? (preview?.mesh ?? null) : null;
  useEffect(() => {
    dialog.setPreview(shown === null ? null : { mesh: shown, centre, rotationDeg });
  }, [dialog, shown, centre, rotationDeg]);
  useEffect(() => () => dialog.setPreview(null), [dialog]);

  const name = nameDraft ?? uniqueName(image === null ? "Relief" : reliefName(image.name), items.map((i) => i.name));

  const create = async () => {
    if (image === null || params === null || problem !== null || busy || reading) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { w, h, pixelSize } = reliefGrid(params, image);
      const heights = await heightImageGrid(image.id, w, h, { scale: params.maxHeight, invert: params.invert });
      const { item } = await runAddRelief({
        store,
        app,
        name: name.trim() === "" ? "Relief" : name.trim(),
        field: { heights, width: w, height: h, pixelSize, base: params.base },
        placement: { rotation: eulerDegToMat3(rotationDeg), translation: centre },
      });
      selectMesh(app, select, store.getSnapshot().selection, item.mesh, false);
      dialog.close();
    } catch (e) {
      setError(message(e));
      setBusy(false);
    }
  };

  const pending = image !== null && problem === null && preview?.key !== key;
  let status: { text: string; problem: boolean };
  if (reading) {
    status = { text: "Reading the image…", problem: false };
  } else if (problem !== null || error !== null) {
    status = { text: (problem ?? error) as string, problem: true };
  } else if (image === null || params === null) {
    status = { text: "Choose an image, or drop one on the page.", problem: false };
  } else {
    status = {
      text: `${describeRelief(params, image, units)}${pending ? " · updating the preview…" : ""}`,
      problem: false,
    };
  }

  return {
    units,
    image,
    /** The settings, once an image is open. */
    params,
    setParams: (patch: Partial<ReliefParams>) => setParams((p) => (p === null ? p : { ...p, ...patch })),
    /** The most samples along the image's longer side that a relief may take. */
    maxSamples: image === null ? 2 : maxSamples(image),
    note,
    name,
    setName: setNameDraft,
    centre,
    setCentre: (axis: number, mm: number) => setCentre((c) => withComponent(c, axis, mm)),
    rotationDeg,
    setRotation: (axis: number, deg: number) => setRotationDeg((r) => withComponent(r, axis, deg)),
    status,
    ready: image !== null && params !== null && problem === null && !reading,
    busy,
    /** Takes the files of a file input: an image, and the JSON that may come with it. */
    pick: (picked: Iterable<File>) => {
      const { relief } = splitReliefFiles(picked);
      if (relief === null) {
        setError("Choose an image (PNG, JPEG, WebP, …).");
      } else {
        void read(relief);
      }
    },
    create,
    cancel: () => dialog.close(),
  };
}
