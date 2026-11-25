// The polyline rows' behaviour (#268): what a click on a name selects, removal,
// and which polylines the tools work on (the command palette asks too).
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppSnapshot, AppState } from "../../../app/state/app-state";
import type { Polyline } from "./polyline";

/** The selection after a click on `id`: it alone (or none, when it was alone); with shift, it added or taken out. */
export function clickedSelection(selection: readonly number[], id: number, shift: boolean): readonly number[] {
  if (shift) {
    return selection.includes(id) ? selection.filter((s) => s !== id) : [...selection, id];
  }
  return selection.length === 1 && selection[0] === id ? [] : [id];
}

/** A click on a polyline's name. */
export function clickPolyline(app: AppState, id: number, shift: boolean): void {
  app.selectPolylines(clickedSelection(app.getSnapshot().polylineSelection, id, shift));
}

/** A colour as CSS, and as a colour input's value. */
export function cssColor(color: number): string {
  return `#${color.toString(16).padStart(6, "0")}`;
}

/** A colour input's value as a colour, or `null` for anything else. */
export function parseColor(value: string): number | null {
  return /^#[0-9a-f]{6}$/i.test(value) ? Number.parseInt(value.slice(1), 16) : null;
}

/** The polylines the tools work on: the selected ones, else the only one. */
export function activePolylines(s: Pick<AppSnapshot, "polylines" | "polylineSelection">): readonly Polyline[] {
  if (s.polylineSelection.length > 0) {
    return s.polylines.filter((p) => s.polylineSelection.includes(p.id));
  }
  return s.polylines.length === 1 ? s.polylines : [];
}

/** Removes polylines, as one undo step. */
export function removePolylines(app: AppState, store: SceneStore, ids: readonly number[]): void {
  const { polylines } = app.getSnapshot();
  const left = polylines.filter((p) => !ids.includes(p.id));
  if (left.length < polylines.length) {
    app.editPolylines(store, left);
  }
}
