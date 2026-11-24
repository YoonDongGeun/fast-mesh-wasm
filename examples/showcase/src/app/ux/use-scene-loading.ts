// How meshes reach the scene from the shell: dropped or picked files,
// samples, phantoms and the demo stage, each framed once it lands (a file
// can be anywhere, at any scale), and the boot landing view.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useCallback, useEffect, useMemo, useRef } from "react";
import { selectMesh } from "../../features/select/ux/selection";
import { parseBigFileParams } from "../../io/big-file";
import type { Loader } from "../../io/loader";
import type { PhantomKind } from "../../io/phantom";
import type { VolumePayload } from "../../io/protocol";
import type { SampleKind } from "../../io/vtkjs";
import type { AppState } from "../state/app-state";

type Vec3 = readonly [number, number, number];

/** A world box for the camera to frame. */
export interface FrameBox {
  center: Vec3;
  radius: number;
}

/**
 * The boot landing frame for a volume: the middle of its voxel box (taken
 * relative to the mesh's `center`, then through the mesh's column-major
 * world matrix `m`) and half the box's diagonal. The box spans past the
 * mesh, so framing it keeps the slice quads in view.
 */
export function volumeLandingBox(
  payload: Pick<VolumePayload, "columns" | "rows" | "slices" | "spacing" | "origin">,
  center: Vec3,
  m: ArrayLike<number>,
): FrameBox {
  const ex = payload.columns * payload.spacing[0];
  const ey = payload.rows * payload.spacing[1];
  const ez = payload.slices * payload.spacing[2];
  const lx = payload.origin[0] + ex / 2 - center[0];
  const ly = payload.origin[1] + ey / 2 - center[1];
  const lz = payload.origin[2] + ez / 2 - center[2];
  return {
    center: [
      m[0] * lx + m[4] * ly + m[8] * lz + m[12],
      m[1] * lx + m[5] * ly + m[9] * lz + m[13],
      m[2] * lx + m[6] * ly + m[10] * lz + m[14],
    ],
    radius: Math.hypot(ex, ey, ez) / 2,
  };
}

export function useSceneLoading(opts: {
  app: AppState;
  scene: Scene | WorkerScene;
  loader: Loader;
  select: (mesh: MeshHandle | null) => void;
  focusOn: (mesh: MeshHandle, dir?: Vec3, box?: FrameBox) => void;
}) {
  const { app, scene, loader, select, focusOn } = opts;

  // `?big` loads the ~1M-triangle demo variants (see io/big-file.ts).
  const demoBig = useMemo(() => {
    try {
      return parseBigFileParams(location.search).demoBig;
    } catch {
      return false;
    }
  }, []);
  const cancelLoad = useCallback((id: number) => loader.cancelJob(id), [loader]);

  const load = useCallback(
    (files: Iterable<File>) => {
      void loader.loadFiles(files).then((added) => {
        // Point the camera at what was just loaded: a file can be anywhere, at any scale.
        const last = added.at(-1);
        if (last !== undefined) {
          focusOn(last.mesh);
        }
      });
    },
    [loader, focusOn],
  );

  // ---- Boot: the CT head, selected and framed ----
  // Once per page (StrictMode runs effects twice in development): the
  // medical landing view (the sample suggests its Vessels preset).
  // Everything else loads from its button.
  const booted = useRef(false);
  useEffect(() => {
    if (booted.current) {
      return;
    }
    booted.current = true;
    void loader.loadSample("headsq").then((added) => {
      const last = added.at(-1);
      if (last === undefined || last.volumeId === null) {
        return;
      }
      selectMesh(app, select, null, last.mesh, false);
      // A three-quarter view of the volume box (the quads span it, past the
      // mesh): the cutaway and its slice quads read best at an angle.
      loader
        .fetchVolume(last.volumeId)
        .then((payload) => {
          focusOn(last.mesh, [1, 0.55, 1.15], volumeLandingBox(payload, last.center, scene.matrix4(last.mesh)));
        })
        .catch(() => {
          focusOn(last.mesh);
        });
    });
  }, [app, scene, loader, focusOn, select]);

  const loadPhantom = (phantom: PhantomKind) => {
    void loader.loadDicomDemo(phantom).then((added) => {
      const last = added.at(-1);
      if (last !== undefined) {
        focusOn(last.mesh);
      }
    });
  };
  const loadSample = (sample: SampleKind) => {
    void loader.loadSample(sample).then((added) => {
      const last = added.at(-1);
      if (last !== undefined) {
        focusOn(last.mesh);
      }
    });
  };
  const loadDemo = () => {
    // No focus: the meshes join the current view (Reset view restores the overview).
    void loader.loadDemo(demoBig);
  };

  return { load, cancelLoad, loadPhantom, loadSample, loadDemo };
}
