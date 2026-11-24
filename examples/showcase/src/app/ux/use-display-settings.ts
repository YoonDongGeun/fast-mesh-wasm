// The persisted display settings the shell applies: the theme (as
// `data-theme` on the root, for the light overrides) and the sidebars
// (shown, widths, and the edges that drag to resize them).
import { type CSSProperties, type PointerEvent, type RefObject, useEffect, useRef } from "react";
import { DEFAULT_SIDEBAR_WIDTH } from "../../shared/ux/settings";
import { type AppState, useApp } from "../state/app-state";

/**
 * Applies the theme and lays the sidebars out: `shellStyle` sets their
 * widths as CSS variables on the shell, and `shellRef` is the shell the
 * resize handles measure from. A hidden sidebar keeps its panels mounted
 * (their effects, like the volume view's linked MPR state, stay alive): it
 * just takes no room.
 */
export function useDisplaySettings(app: AppState) {
  const theme = useApp(app, (s) => s.theme);
  const leftVisible = useApp(app, (s) => s.leftVisible);
  const rightVisible = useApp(app, (s) => s.rightVisible);
  const leftWidth = useApp(app, (s) => s.leftWidth);
  const rightWidth = useApp(app, (s) => s.rightWidth);
  const shellRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const shellStyle = {
    "--left-w": leftVisible ? `${leftWidth}px` : "0px",
    "--right-w": rightVisible ? `${rightWidth}px` : "0px",
  } as CSSProperties;
  return { shellRef, shellStyle, leftVisible, rightVisible };
}

/**
 * A sidebar edge's handlers: a primary-button drag sets the sidebar's
 * width from the shell's outer edge to the pointer (persisted), and a
 * double-click restores the default width.
 */
export function sidebarResizeHandlers(
  side: "left" | "right",
  shell: RefObject<HTMLDivElement | null>,
  app: AppState,
) {
  const setWidth = side === "left" ? (w: number) => app.setLeftWidth(w) : (w: number) => app.setRightWidth(w);
  return {
    onDoubleClick: () => setWidth(DEFAULT_SIDEBAR_WIDTH),
    onPointerDown: (e: PointerEvent<HTMLDivElement>) => e.currentTarget.setPointerCapture(e.pointerId),
    onPointerMove: (e: PointerEvent<HTMLDivElement>) => {
      if (e.buttons !== 1 || !e.isPrimary) {
        return;
      }
      const rect = shell.current?.getBoundingClientRect();
      if (rect === undefined) {
        return;
      }
      setWidth(side === "left" ? e.clientX - rect.left : rect.right - e.clientX);
    },
  };
}
