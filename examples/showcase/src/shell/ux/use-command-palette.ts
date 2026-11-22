// The command palette's behaviour (#235): the global launcher (Ctrl+K /
// Cmd+K toggles it anywhere, `/` opens it outside text fields), a fresh
// query and the search field focused each time it opens, the commands
// ranked for the query, and keyboard navigation in the field (ArrowUp /
// ArrowDown wrap, Enter runs the active command, Escape closes).
import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../../app/state/app-state";
import { buildCommandPaletteItems, type CommandPaletteContext, type CommandPaletteItem } from "./command-registry";
import { filterCommands } from "./command-search";

export function useCommandPalette(ctx: CommandPaletteContext) {
  const { app } = ctx;
  const open = useApp(app, (s) => s.commandPaletteOpen);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Global Ctrl+K / Cmd+K and `/` launcher.
  useEffect(() => {
    const onKeyDown = (e: globalThis.KeyboardEvent) => {
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
    () => buildCommandPaletteItems(ctx),
    [ctx.app, ctx.items, ctx.eligibleItems, ctx.selection, ctx.canUndo, ctx.canRedo, ctx.ortho, ctx.units],
  );

  const filtered = useMemo(() => filterCommands(allCommands, query), [allCommands, query]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  const close = () => app.setCommandPaletteOpen(false);

  /** Closes the palette and runs `cmd`, unless it is missing or disabled. */
  const execute = (cmd: CommandPaletteItem | undefined) => {
    if (!cmd || cmd.disabled) return;
    app.setCommandPaletteOpen(false);
    cmd.run();
  };

  const onInputKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
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
  };

  return { open, query, setQuery, activeIndex, setActiveIndex, inputRef, filtered, close, execute, onInputKeyDown };
}
