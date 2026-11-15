/**
 * Ctrl+K / `/` Command Search Palette (#235).
 * Fuzzy-searches workspace tabs, sculpt/edit/repair/inspect tools, 6 diagnostic
 * shading modes, camera presets, and loaded scene objects with full keyboard
 * navigation (ArrowUp / ArrowDown / Enter / Escape).
 */
import type { MeshHandle } from "fast-mesh-wasm";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  type AppState,
  type CameraPreset,
  type DiagnosticShadingMode,
  type Item,
  useApp,
  type WorkbenchTab,
} from "../scene/app-state";
import { sculptBrush, type SculptTool } from "../viewport/sculpt";
import type { GizmoMode } from "../viewport/Viewport";

export interface CommandPaletteItem {
  id: string;
  title: string;
  category: "Tab" | "Tool" | "Shading" | "Camera" | "Object" | "Action";
  keywords: string;
  shortcut?: string;
  disabled?: boolean;
  run: () => void;
}

export interface CommandPaletteContext {
  app: AppState;
  items: readonly Item[];
  eligibleItems?: readonly Item[];
  selection: MeshHandle | null;
  canUndo: boolean;
  canRedo: boolean;
  ortho: boolean;
  units: "mm" | "in";
  onSelectItem: (mesh: MeshHandle) => void;
  onSetMode: (mode: GizmoMode) => void;
  onResetView: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onOpenHelp: () => void;
}

/**
 * Scores a fuzzy subsequence / substring query against `text`.
 * Returns `null` when `query` does not match, otherwise higher scores are better.
 */
export function fuzzyMatchCommand(query: string, text: string): number | null {
  const q = query.trim().toLowerCase();
  if (q === "") return 1;
  const hay = text.toLowerCase();
  const idx = hay.indexOf(q);
  if (idx >= 0) {
    const wordStart = idx === 0 || /[\s\-/_:&]/.test(hay[idx - 1] ?? "");
    return 1000 - idx * 2 + (wordStart ? 200 : 0);
  }
  let qi = 0;
  let gap = 0;
  let lastMatch = -1;
  for (let hi = 0; hi < hay.length && qi < q.length; hi++) {
    if (hay[hi] === q[qi]) {
      if (lastMatch >= 0) gap += hi - lastMatch - 1;
      lastMatch = hi;
      qi++;
    }
  }
  if (qi < q.length) return null;
  return Math.max(1, 300 - gap * 3 - (lastMatch - q.length));
}

export const TAB_LABELS: readonly { tab: WorkbenchTab; label: string; keywords: string }[] = [
  { tab: "home", label: "Home Workspace", keywords: "tab home open recent project overview" },
  { tab: "select", label: "Select & Transform", keywords: "tab select gizmo move rotate scale groups" },
  { tab: "repair", label: "Mesh Repair & Healer", keywords: "tab heal repair holes manifold slivers intersections" },
  { tab: "edit", label: "Mesh Edit & Modeling", keywords: "tab modeling offset shell decimate boolean remesh" },
  { tab: "sculpt", label: "Sculpt & Deform", keywords: "tab sculpt inflate smooth flatten crease grab clay twist mask" },
  { tab: "align", label: "Points & Align", keywords: "tab align register collision distance gap pick" },
  { tab: "volume", label: "Voxel / CT Volume", keywords: "tab volume dicom mpr hu window slice" },
  { tab: "inspect", label: "Inspect & Measure", keywords: "tab inspect ruler measure section clip curvature thickness" },
];

const SHADING_COMMANDS: readonly { mode: DiagnosticShadingMode; label: string; keywords: string }[] = [
  { mode: "smooth", label: "Shading: Smooth", keywords: "shading smooth phong normal" },
  { mode: "flat", label: "Shading: Flat", keywords: "shading flat faceted triangle normals" },
  { mode: "autoCrease", label: "Shading: Auto-Crease", keywords: "shading auto crease sharp dihedral angle edges" },
  { mode: "wireframeShaded", label: "Shading: Wireframe-on-Shaded", keywords: "shading wireframe shaded overlay mesh topology" },
  { mode: "matcap", label: "Shading: MatCap", keywords: "shading matcap clay sculpt studio" },
  { mode: "backfaceWarning", label: "Shading: Backface Warning", keywords: "shading backface warning inverted flipped winding red pink" },
];

const CAMERA_PRESETS: readonly { preset: CameraPreset; label: string; shortcut: string }[] = [
  { preset: "front", label: "Camera: Front View", shortcut: "1" },
  { preset: "back", label: "Camera: Back View", shortcut: "Shift+1" },
  { preset: "left", label: "Camera: Left View", shortcut: "3" },
  { preset: "right", label: "Camera: Right View", shortcut: "Shift+3" },
  { preset: "top", label: "Camera: Top View", shortcut: "7" },
  { preset: "bottom", label: "Camera: Bottom View", shortcut: "Shift+7" },
];

const SCULPT_TOOLS: readonly { tool: SculptTool; label: string; keywords: string }[] = [
  { tool: "inflate", label: "Sculpt: Inflate Brush", keywords: "sculpt inflate push dent brush" },
  { tool: "smooth", label: "Sculpt: Smooth Brush", keywords: "sculpt smooth taubin relax brush" },
  { tool: "flatten", label: "Sculpt: Flatten Brush", keywords: "sculpt flatten planar brush" },
  { tool: "crease", label: "Sculpt: Crease Brush", keywords: "sculpt crease pinch sharp groove brush" },
  { tool: "grab", label: "Sculpt: Grab Brush", keywords: "sculpt grab move drag deform" },
  { tool: "clay", label: "Sculpt: Clay Build Brush", keywords: "sculpt clay buildup brush" },
  { tool: "twist", label: "Sculpt: Twist Brush", keywords: "sculpt twist vortex rotate brush" },
];

export function buildCommandPaletteItems(ctx: CommandPaletteContext): CommandPaletteItem[] {
  const { app, items, selection } = ctx;
  const cmds: CommandPaletteItem[] = [];

  for (const t of TAB_LABELS) {
    cmds.push({
      id: `tab:${t.tab}`,
      title: t.label,
      category: "Tab",
      keywords: t.keywords,
      run: () => app.setActiveTab(t.tab),
    });
  }

  for (const s of SHADING_COMMANDS) {
    cmds.push({
      id: `shading:${s.mode}`,
      title: s.label,
      category: "Shading",
      keywords: s.keywords,
      run: () => app.setShadingMode(s.mode),
    });
  }

  for (const c of CAMERA_PRESETS) {
    cmds.push({
      id: `camera:${c.preset}`,
      title: c.label,
      category: "Camera",
      keywords: `camera view ${c.preset} preset ortho focus`,
      shortcut: c.shortcut,
      run: () => app.requestCameraPreset(c.preset),
    });
  }

  const eligible = ctx.eligibleItems ?? items.filter((i) => !i.locked);
  const sculptTarget =
    (selection !== null ? eligible.find((i) => i.mesh === selection) : undefined) ?? eligible[0] ?? null;

  for (const st of SCULPT_TOOLS) {
    cmds.push({
      id: `sculpt:${st.tool}`,
      title: st.label,
      category: "Tool",
      keywords: st.keywords,
      disabled: sculptTarget === null,
      run: () => {
        if (sculptTarget === null) return;
        app.setActiveTab("sculpt");
        const snap = app.getSnapshot();
        const cur = snap.sculpt;
        const radius = cur?.mesh === sculptTarget.mesh ? cur.radius : sculptTarget.radius * 0.15;
        const strength = st.tool === "smooth" ? 0.5 : radius * 0.2;
        app.setSculpt(
          sculptBrush(st.tool, sculptTarget.mesh, radius, strength, "smooth", {
            methods: snap.sculptMethods,
            grabPoisson: snap.sculptGrabPoisson,
            filter: snap.sculptFilter,
            symmetry: snap.sculptSymmetry,
          }),
        );
      },
    });
  }

  cmds.push(
    {
      id: "tool:sculpt-disarm",
      title: "Exit Sculpt / Return to Select",
      category: "Tool",
      keywords: "sculpt exit stop disarm select pointer gizmo",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:gizmo-translate",
      title: "Transform: Move Gizmo",
      category: "Tool",
      keywords: "move translate gizmo transform position",
      shortcut: "T",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        ctx.onSetMode("translate");
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:gizmo-rotate",
      title: "Transform: Rotate Gizmo",
      category: "Tool",
      keywords: "rotate spin angle gizmo transform",
      shortcut: "R",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        ctx.onSetMode("rotate");
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:gizmo-scale",
      title: "Transform: Scale Gizmo",
      category: "Tool",
      keywords: "scale resize gizmo transform",
      shortcut: "S",
      run: () => {
        app.setSculpt(null);
        app.setMaskPaint(false);
        ctx.onSetMode("scale");
        app.setActiveTab("select");
      },
    },
    {
      id: "tool:inspect-ruler",
      title: "Inspect: Measure Ruler",
      category: "Tool",
      keywords: "inspect ruler measure distance two points",
      run: () => {
        app.setActiveTab("inspect");
        app.setInspectTool("ruler");
      },
    },
    {
      id: "tool:inspect-section",
      title: "Inspect: Section Cut Plane",
      category: "Tool",
      keywords: "inspect section cut slice cross",
      run: () => {
        app.setActiveTab("inspect");
        app.setInspectTool("section");
      },
    },
    {
      id: "action:reset-view",
      title: "Reset Camera View",
      category: "Action",
      keywords: "camera reset view fit home frame all",
      run: ctx.onResetView,
    },
    {
      id: "action:toggle-ortho",
      title: ctx.ortho ? "Camera: Switch to Perspective" : "Camera: Switch to Orthographic",
      category: "Camera",
      keywords: "camera projection orthographic perspective persp ortho",
      run: () => {
        app.toggleCameraProjection();
      },
    },
    {
      id: "action:toggle-units",
      title: `Units: Switch to ${ctx.units === "mm" ? "Inches (in)" : "Millimetres (mm)"}`,
      category: "Action",
      keywords: "units mm in millimeters inches scale",
      run: () => app.setUnits(ctx.units === "mm" ? "in" : "mm"),
    },
    {
      id: "action:undo",
      title: "Undo Last Action",
      category: "Action",
      keywords: "undo revert history",
      shortcut: "Ctrl+Z",
      disabled: !ctx.canUndo,
      run: ctx.onUndo,
    },
    {
      id: "action:redo",
      title: "Redo Action",
      category: "Action",
      keywords: "redo repeat history",
      shortcut: "Ctrl+Y",
      disabled: !ctx.canRedo,
      run: ctx.onRedo,
    },
    {
      id: "action:help",
      title: "Open Keyboard Shortcuts & Help",
      category: "Action",
      keywords: "help shortcuts controls guide",
      shortcut: "?",
      run: ctx.onOpenHelp,
    },
  );

  for (const item of items) {
    cmds.push({
      id: `object:${item.id}`,
      title: `Select Object: ${item.name}`,
      category: "Object",
      keywords: `object mesh ${item.name} ${item.volumeId !== null ? "ct volume" : ""} ${item.locked ? "locked" : ""}`,
      run: () => {
        ctx.onSelectItem(item.mesh);
        if (item.volumeId !== null) {
          app.setActiveTab("volume");
        }
      },
    });
    cmds.push({
      id: `lock:${item.id}`,
      title: `${item.locked ? "Unlock" : "Lock"} Object: ${item.name}`,
      category: "Object",
      keywords: `lock unlock protect freeze ${item.name}`,
      run: () => app.setMeshLocked(item.mesh, !item.locked),
    });
  }

  return cmds;
}

export function CommandPalette(props: CommandPaletteContext) {
  const { app } = props;
  const open = useApp(app, (s) => s.commandPaletteOpen);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Global Ctrl+K / Cmd+K and `/` launcher.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const inInput =
        target !== null &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable);

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        app.setCommandPaletteOpen(!app.getSnapshot().commandPaletteOpen);
        return;
      }
      if (!inInput && e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        app.setCommandPaletteOpen(true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [app]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const allCommands = useMemo(
    () => buildCommandPaletteItems(props),
    [
      props.app,
      props.items,
      props.eligibleItems,
      props.selection,
      props.canUndo,
      props.canRedo,
      props.ortho,
      props.units,
    ],
  );

  const filtered = useMemo(() => {
    const scored: { item: CommandPaletteItem; score: number }[] = [];
    for (const item of allCommands) {
      const s = fuzzyMatchCommand(query, `${item.title} ${item.category} ${item.keywords}`);
      if (s !== null) {
        scored.push({ item, score: s });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 24).map((e) => e.item);
  }, [allCommands, query]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  if (!open) return null;

  const execute = (cmd: CommandPaletteItem | undefined) => {
    if (!cmd || cmd.disabled) return;
    app.setCommandPaletteOpen(false);
    cmd.run();
  };

  return (
    <div
      className="command-palette-backdrop"
      data-command-palette
      role="dialog"
      aria-modal="true"
      aria-label="Command search"
      onClick={() => app.setCommandPaletteOpen(false)}
    >
      <div className="command-palette" onClick={(e) => e.stopPropagation()}>
        <div className="command-palette-head">
          <input
            ref={inputRef}
            type="search"
            className="command-palette-input"
            data-command-search-input
            aria-label="Search tools, tabs, shading modes, objects"
            placeholder="Search tools, tabs, shading, camera, meshes… (Esc to close)"
            value={query}
            onChange={(e) => setQuery(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                app.setCommandPaletteOpen(false);
              } else if (e.key === "ArrowDown") {
                e.preventDefault();
                setActiveIndex((i) => (filtered.length === 0 ? 0 : (i + 1) % filtered.length));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setActiveIndex((i) => (filtered.length === 0 ? 0 : (i - 1 + filtered.length) % filtered.length));
              } else if (e.key === "Enter") {
                e.preventDefault();
                execute(filtered[activeIndex]);
              }
            }}
          />
          <button
            type="button"
            className="icon"
            aria-label="Close command search"
            onClick={() => app.setCommandPaletteOpen(false)}
          >
            ×
          </button>
        </div>
        <ul className="command-palette-list" role="listbox">
          {filtered.length === 0 ? (
            <li className="command-palette-empty muted small">No matching commands</li>
          ) : (
            filtered.map((cmd, idx) => (
              <li key={cmd.id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={idx === activeIndex}
                  className={idx === activeIndex ? "command-palette-item active" : "command-palette-item"}
                  data-command-id={cmd.id}
                  disabled={cmd.disabled}
                  onMouseEnter={() => setActiveIndex(idx)}
                  onClick={() => execute(cmd)}
                >
                  <span className="command-palette-cat">{cmd.category}</span>
                  <span className="command-palette-title">{cmd.title}</span>
                  {cmd.shortcut ? <kbd className="command-palette-kbd">{cmd.shortcut}</kbd> : null}
                </button>
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
}
