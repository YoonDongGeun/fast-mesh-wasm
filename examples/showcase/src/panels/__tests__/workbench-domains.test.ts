// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  domainMatchesPanel,
  WORKBENCH_DOMAINS,
  WorkbenchDomains,
  type WorkbenchDomainId,
} from "../WorkbenchDomains";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const roots: { root: Root; container: HTMLDivElement }[] = [];

function mount(element: React.ReactElement): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(element);
  });
  roots.push({ root, container });
  return container;
}

afterEach(() => {
  for (const { root, container } of roots.splice(0).reverse()) {
    act(() => {
      root.unmount();
    });
    container.remove();
  }
});

describe("WorkbenchDomains", () => {
  it("lists all 7 cohesive architecture domains", () => {
    const ids = WORKBENCH_DOMAINS.map((d) => d.id);
    expect(ids).toEqual(["repair", "sculpt", "remesh", "boolean", "deform", "inspect", "align"]);

    const el = mount(createElement(WorkbenchDomains));
    for (const id of ids) {
      const btn = el.querySelector<HTMLButtonElement>(`[data-domain-tab="${id}"]`);
      expect(btn).not.toBeNull();
      expect(btn?.getAttribute("aria-pressed")).toBe("false");
    }
    expect(el.querySelector("[data-workbench-domains]")?.getAttribute("data-active-domain")).toBe("all");
  });

  it("toggles active domain, renders API summary, and notifies callback", () => {
    const picked: (WorkbenchDomainId | null)[] = [];
    const el = mount(createElement(WorkbenchDomains, { onSelectDomain: (d) => picked.push(d) }));

    const inspectBtn = el.querySelector<HTMLButtonElement>('[data-domain-tab="inspect"]');
    expect(inspectBtn).not.toBeNull();
    act(() => {
      inspectBtn?.click();
    });

    expect(picked).toEqual(["inspect"]);
    expect(inspectBtn?.getAttribute("aria-pressed")).toBe("true");
    const summary = el.querySelector('[data-domain-summary="inspect"]');
    expect(summary?.textContent).toContain("scene.inspect");
    expect(summary?.textContent).toContain("measure");

    // Clicking the active domain again resets to all
    act(() => {
      inspectBtn?.click();
    });
    expect(picked).toEqual(["inspect", null]);
    expect(el.querySelector("[data-workbench-domains]")?.getAttribute("data-active-domain")).toBe("all");
  });

  it("expands the API and CLI map drawer cleanly", () => {
    const el = mount(createElement(WorkbenchDomains));
    const toggle = el.querySelector<HTMLButtonElement>('[data-domain-details-toggle="true"]');
    expect(el.querySelector('[data-domain-map="true"]')).toBeNull();

    act(() => {
      toggle?.click();
    });

    const map = el.querySelector('[data-domain-map="true"]');
    expect(map).not.toBeNull();
    expect(map?.querySelectorAll("[data-domain-entry]").length).toBe(7);
    expect(map?.textContent).toContain("fast-mesh repair auto");
    expect(map?.textContent).toContain("fast-mesh remesh simplify");
  });

  it("maps domain selections to showcase workbench panels without hiding controls", () => {
    expect(domainMatchesPanel(null, "repair")).toBe(true);
    expect(domainMatchesPanel(null, "collision")).toBe(true);
    expect(domainMatchesPanel("repair", "repair")).toBe(true);
    expect(domainMatchesPanel("repair", "collision")).toBe(false);
    expect(domainMatchesPanel("inspect", "inspect")).toBe(true);
    expect(domainMatchesPanel("align", "collision")).toBe(true);
    vi.restoreAllMocks();
  });
});
