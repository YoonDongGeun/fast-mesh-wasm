/**
 * Ctrl+K / `/` Command Search Palette (#235).
 * Fuzzy-searches workspace tabs, sculpt/edit/repair/inspect tools, 6 diagnostic
 * shading modes, camera presets, and loaded scene objects with full keyboard
 * navigation (ArrowUp / ArrowDown / Enter / Escape).
 */
import type { CommandPaletteContext } from "../ux/command-registry";
import { useCommandPalette } from "../ux/use-command-palette";

export function CommandPalette(props: CommandPaletteContext) {
  const { open, query, setQuery, activeIndex, setActiveIndex, inputRef, filtered, close, execute, onInputKeyDown } =
    useCommandPalette(props);

  if (!open) return null;

  return (
    <div
      className="command-palette-backdrop"
      data-command-palette
      role="dialog"
      aria-modal="true"
      aria-label="Command search"
      onClick={close}
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
            onKeyDown={onInputKeyDown}
          />
          <button
            type="button"
            className="icon"
            aria-label="Close command search"
            onClick={close}
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
