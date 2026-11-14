import { Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { SceneStore } from "fast-mesh-wasm/store";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { describeReport } from "../src/panels/repair-summary";
import { AppState } from "../src/scene/app-state";
import { selectMesh } from "../src/scene/selection";
import { SculptToolbox } from "../src/viewport/Toolbox";
import { sculptBrush, type SculptTool } from "../src/viewport/sculpt";

function makeSelectedApp() {
  const scene = new Scene();
  const store = new SceneStore(scene);
  const app = new AppState();
  const item = app.addMesh(store, {
    name: "cube",
    positions: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2]),
    center: [0.25, 0.25, 0.25],
    radius: 1,
    volumeId: null,
    preset: null,
  });
  selectMesh(app, (m) => store.select(m), null, item.mesh, false);
  return { scene, store, app, mesh: item.mesh };
}

describe("SculptToolbox UI controls (#177, #192)", () => {
  const ALL_TOOLS: readonly SculptTool[] = [
    "inflate",
    "deflate",
    "smooth",
    "grab",
    "flatten",
    "scrape",
    "fill",
    "crease",
    "pinch",
    "clay",
    "clayStrips",
    "twist",
  ];

  it("renders_all_12_tool_buttons_filter_picker_and_symmetry_toggles", () => {
    const { store, app, mesh } = makeSelectedApp();
    app.setSculptSymmetry({ x: true, y: false, z: true });
    const html = renderToStaticMarkup(
      <SceneProvider store={store}>
        <SculptToolbox app={app} selection={mesh} />
      </SceneProvider>,
    );
    for (const tool of ALL_TOOLS) {
      expect(html).toContain(`data-action="sculpt-tool-${tool}"`);
    }
    expect(html).toContain(`data-action="sculpt-method"`);
    expect(html).toContain(`data-action="sculpt-filter"`);
    expect(html).toContain(`data-action="sculpt-sym-x"`);
    expect(html).toContain(`data-action="sculpt-sym-y"`);
    expect(html).toContain(`data-action="sculpt-sym-z"`);
    expect(html).toMatch(/data-action="sculpt-sym-x"[^>]*aria-pressed="true"/);
    expect(html).toMatch(/data-action="sculpt-sym-y"[^>]*aria-pressed="false"/);
    expect(html).toMatch(/data-action="sculpt-sym-z"[^>]*aria-pressed="true"/);
    store.dispose();
  });

  it("renders_per_brush_method_descriptions_poisson_slider_and_active_filter_controls", () => {
    const { store, app, mesh } = makeSelectedApp();

    // Smooth -> method picker with 4 algorithms + one-line description
    app.setSculptMethods({ smooth: "cotangent" });
    app.setSculpt(sculptBrush("smooth", mesh, 0.25, 0.5, "smooth", { methods: app.getSnapshot().sculptMethods }));
    const smoothHtml = renderToStaticMarkup(
      <SceneProvider store={store}>
        <SculptToolbox app={app} selection={mesh} />
      </SceneProvider>,
    );
    expect(smoothHtml).toContain(`data-action="sculpt-method"`);
    expect(smoothHtml).toContain(`value="taubin"`);
    expect(smoothHtml).toContain(`value="laplacian"`);
    expect(smoothHtml).toContain(`value="hcLaplacian"`);
    expect(smoothHtml).toContain(`value="cotangent"`);
    expect(smoothHtml).toContain(`data-role="sculpt-method-description"`);

    // Grab with kelvinlet -> Poisson slider visible
    app.setSculptMethods({ grab: "kelvinlet" });
    app.setSculptGrabPoisson(0.35);
    app.setSculpt(
      sculptBrush("grab", mesh, 0.25, 1, "smooth", {
        methods: app.getSnapshot().sculptMethods,
        grabPoisson: 0.35,
      }),
    );
    const grabHtml = renderToStaticMarkup(
      <SceneProvider store={store}>
        <SculptToolbox app={app} selection={mesh} />
      </SceneProvider>,
    );
    expect(grabHtml).toContain(`data-action="sculpt-grab-poisson"`);

    // Twist -> vortex / kelvinlet options
    app.setSculpt(sculptBrush("twist", mesh, 0.25, 0.5, "smooth"));
    const twistHtml = renderToStaticMarkup(
      <SceneProvider store={store}>
        <SculptToolbox app={app} selection={mesh} />
      </SceneProvider>,
    );
    expect(twistHtml).toContain(`value="vortex"`);
    expect(twistHtml).toContain(`value="kelvinlet"`);

    // Flatten -> no method picker, active peaksOnly filter shows threshold/contrast sliders
    app.setSculptFilter({ kind: "peaksOnly", threshold: 0.1, contrast: 0.6, maxAngleDeg: 45 });
    app.setSculpt(sculptBrush("flatten", mesh, 0.25, 0.5, "smooth", { filter: app.getSnapshot().sculptFilter }));
    const flattenHtml = renderToStaticMarkup(
      <SceneProvider store={store}>
        <SculptToolbox app={app} selection={mesh} />
      </SceneProvider>,
    );
    expect(flattenHtml).not.toContain(`data-action="sculpt-method"`);
    expect(flattenHtml).toContain(`data-action="sculpt-filter-threshold"`);
    expect(flattenHtml).toContain(`data-action="sculpt-filter-contrast"`);

    // normalAngle filter shows max angle slider
    app.setSculptFilter({ kind: "normalAngle", maxAngleDeg: 30 });
    const angleHtml = renderToStaticMarkup(
      <SceneProvider store={store}>
        <SculptToolbox app={app} selection={mesh} />
      </SceneProvider>,
    );
    expect(angleHtml).toContain(`data-action="sculpt-filter-angle"`);

    store.dispose();
  });
});

describe("describeReport for new sculpt ops", () => {
  it("formats_flatten_crease_clay_twist_and_captureStrokeState", () => {
    expect(describeReport("flatten", { moved: 12, maxDisplacement: 0.25 }, "mm")).toBe(
      "Flatten: moved 12 vertices (max 0.25 mm)",
    );
    expect(describeReport("crease", { moved: 1, maxDisplacement: 0.1 })).toBe("Crease: moved 1 vertex (max 0.1)");
    expect(describeReport("clay", { moved: 8, maxDisplacement: 0.5 })).toBe("Clay: moved 8 vertices (max 0.5)");
    expect(describeReport("twist", { moved: 5, maxDisplacement: 0.3 })).toBe("Twist: moved 5 vertices (max 0.3)");
    expect(describeReport("captureStrokeState", {})).toBe("Captured stroke state");
  });
});
