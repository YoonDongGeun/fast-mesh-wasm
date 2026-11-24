// The Group mode's hover tip: over a coloured member, which partner the
// vertex under the pointer is nearest to and how far it is (see
// `useHoverPartner`; the tip may lag a move).
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { view } from "../../../viewport/ux/view";
import { formatLength } from "../../../shared/ux/units";
import { useHoverPartner } from "../ux/use-hover-partner";

export function HoverTip(props: { app: AppState; scene: Scene | WorkerScene; items: readonly Item[]; members: readonly MeshHandle[] }) {
  const { app, scene, items, members } = props;
  const units = useApp(app, (s) => s.units);
  const { hover, member, data, hit } = useHoverPartner(scene, members);
  if (hover === null || member === null) {
    return null;
  }
  const nameOf = (m: MeshHandle) => items.find((i) => i.mesh === m)?.name ?? "a mesh";
  return (
    <div
      className="hover-tip"
      data-hover-tip
      // A tap lands anywhere on a phone: the tip stays on screen (its max-width in styles.css).
      style={{ left: view.pointerType === "touch" ? `min(${hover.x + 14}px, calc(100vw - 236px))` : hover.x + 14, top: hover.y + 14 }}
    >
      <strong>{nameOf(member)}</strong>
      <br />
      {data === null
        ? "reading…"
        : hit === null
          ? "no partner within range"
          : `${hit.distance < 0 ? "inside" : "nearest to"} ${nameOf(hit.partner)}: ${formatLength(Math.abs(hit.distance), units)}`}
    </div>
  );
}
