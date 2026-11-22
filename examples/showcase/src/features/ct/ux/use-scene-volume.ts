// The volume the main scene shows, for its slice quads and its volume box:
// the selected mesh's, while the linked MPR state points at the same series,
// shows it in the scene and the mesh is not being sculpted; its retained
// volume fetched for it, and the display values with their range.
import { useSelection } from "fast-mesh-wasm/react";
import { useEffect, useMemo, useState } from "react";
import type { Loader } from "../../../io/loader";
import type { VolumePayload } from "../../../io/protocol";
import { type AppState, type Item, useApp, type VolumeMpr } from "../../../app/state/app-state";
import { dataRange } from "./volume-lut";
import { displayValues } from "./volume-data";
import { matchVolume } from "./volume-box";

/**
 * `item` is the matched mesh whether or not the scene shows its volume
 * (`active`); `volumeId` is set only while it does. `payload`, `display` and
 * `range` are `null` until the volume arrives, and again when it goes.
 */
export function useSceneVolume(
  app: AppState,
  loader: Loader,
): {
  item: Item | null;
  active: boolean;
  volumeId: string | null;
  mpr: VolumeMpr | null;
  payload: VolumePayload | null;
  display: Float32Array | null;
  range: [number, number] | null;
} {
  const [selection] = useSelection();
  const items = useApp(app, (s) => s.items);
  const mpr = useApp(app, (s) => s.volumeMpr);
  const [payload, setPayload] = useState<VolumePayload | null>(null);
  const found = matchVolume(items, selection, mpr);
  const sculpt = useApp(app, (s) => s.sculpt);
  const item: Item | null = found?.item ?? null;
  const active = found !== null && found.mpr.inScene && found.item.mesh !== (sculpt?.mesh ?? null);
  const volumeId = active && found !== null ? found.volumeId : null;

  useEffect(() => {
    setPayload(null);
    if (volumeId === null) {
      return;
    }
    let alive = true;
    loader
      .fetchVolume(volumeId)
      .then((got: VolumePayload) => {
        if (alive) {
          setPayload(got);
        }
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [loader, volumeId]);

  const display = useMemo(
    () => (payload === null ? null : displayValues(payload.values, payload.pseudoHu)),
    [payload],
  );
  const range = useMemo(() => (display === null ? null : dataRange(display)), [display]);
  return { item, active, volumeId, mpr, payload, display, range };
}
