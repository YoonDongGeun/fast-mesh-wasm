// Seven-domain architecture navigator and summary drawer for the showcase
// right sidebar: organizes workbench workflows under the same 7 domain
// namespaces shared by Rust (`fast_mesh_ops`), TypeScript (`fast-mesh-wasm`),
// and the CLI (`fast-mesh`).
import { useState } from "react";

export type WorkbenchDomainId =
  | "repair"
  | "sculpt"
  | "remesh"
  | "boolean"
  | "deform"
  | "inspect"
  | "align";

export type WorkbenchPanelKey = "repair" | "collision" | "inspect" | "export" | "volume";

export interface WorkbenchDomainMeta {
  readonly id: WorkbenchDomainId;
  readonly label: string;
  readonly title: string;
  readonly api: string;
  readonly cli: string;
  readonly operations: readonly string[];
  readonly panels: readonly WorkbenchPanelKey[];
}

export const WORKBENCH_DOMAINS: readonly WorkbenchDomainMeta[] = [
  {
    id: "repair",
    label: "Repair",
    title: "Repair & Healing",
    api: "scene.repair",
    cli: "fast-mesh repair auto",
    operations: ["autoRepair", "weld", "orient", "fillHoles", "bridge", "makeManifold", "resolveSelfIntersections", "cleanupSlivers"],
    panels: ["repair"],
  },
  {
    id: "sculpt",
    label: "Sculpt",
    title: "Sculpt & Dynamic Topology",
    api: "scene.sculpt",
    cli: "fast_mesh_ops::sculpt",
    operations: ["beginStroke", "dab", "endStroke", "filter", "inflate", "smooth", "grab", "flatten", "crease", "clay", "twist", "splitStretched"],
    panels: ["repair"],
  },
  {
    id: "remesh",
    label: "Remesh",
    title: "Remesh & Decimation",
    api: "scene.remesh",
    cli: "fast-mesh remesh simplify|subdivide",
    operations: ["region", "simplify", "subdivide", "voxel"],
    panels: ["repair"],
  },
  {
    id: "boolean",
    label: "Boolean",
    title: "Booleans & Cutting",
    api: "scene.boolean",
    cli: "fast-mesh boolean union|intersection|difference",
    operations: ["union", "difference", "intersection", "cut", "offset", "shell"],
    panels: ["repair"],
  },
  {
    id: "deform",
    label: "Deform",
    title: "Deformation & Physics",
    api: "scene.deform",
    cli: "fast_mesh_ops::deform",
    operations: ["inflate", "smooth", "relax", "grab", "twist", "protect"],
    panels: ["repair"],
  },
  {
    id: "inspect",
    label: "Inspect",
    title: "Inspection & Metrology",
    api: "scene.inspect",
    cli: "fast-mesh inspect measure|analyze",
    operations: ["analyze", "measure", "raycast", "closestPoint", "addDistanceField", "addThicknessField", "addCurvatureField", "section"],
    panels: ["inspect", "collision", "volume", "export"],
  },
  {
    id: "align",
    label: "Align",
    title: "Registration & Alignment",
    api: "scene.align",
    cli: "fast_mesh_ops::align",
    operations: ["register"],
    panels: ["collision"],
  },
];

export function domainMatchesPanel(active: WorkbenchDomainId | null, panel: WorkbenchPanelKey): boolean {
  if (active === null) {
    return true;
  }
  const meta = WORKBENCH_DOMAINS.find((d) => d.id === active);
  return meta?.panels.includes(panel) ?? true;
}

export interface WorkbenchDomainsProps {
  activeDomain?: WorkbenchDomainId | null;
  onSelectDomain?: (domain: WorkbenchDomainId | null) => void;
}

export function WorkbenchDomains(props: WorkbenchDomainsProps) {
  const { activeDomain: controlled, onSelectDomain } = props;
  const [uncontrolled, setUncontrolled] = useState<WorkbenchDomainId | null>(null);
  const [expanded, setExpanded] = useState(false);

  const active = controlled !== undefined ? controlled : uncontrolled;
  const selectDomain = (next: WorkbenchDomainId | null) => {
    const toggled = next !== null && active === next ? null : next;
    if (controlled === undefined) {
      setUncontrolled(toggled);
    }
    onSelectDomain?.(toggled);
  };

  const activeMeta = active === null ? null : WORKBENCH_DOMAINS.find((d) => d.id === active) ?? null;

  return (
    <section
      className="panel workbench-domains"
      aria-label="Workbench domains"
      data-workbench-domains="true"
      data-active-domain={active ?? "all"}
    >
      <div className="row domains-header">
        <h2>Domains</h2>
        <div className="domains-header-actions">
          {active !== null ? (
            <button
              type="button"
              className="small"
              data-domain-reset="true"
              onClick={() => selectDomain(null)}
            >
              All
            </button>
          ) : null}
          <button
            type="button"
            className="small"
            data-domain-details-toggle="true"
            aria-expanded={expanded}
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "Hide map" : "API map"}
          </button>
        </div>
      </div>

      <div className="domain-tabs" role="group" aria-label="Domain filter">
        {WORKBENCH_DOMAINS.map((domain) => (
          <button
            key={domain.id}
            type="button"
            data-domain-tab={domain.id}
            aria-pressed={active === domain.id}
            title={`${domain.title} (${domain.api})`}
            onClick={() => selectDomain(domain.id)}
          >
            {domain.label}
          </button>
        ))}
      </div>

      {activeMeta !== null ? (
        <div className="domain-summary" data-domain-summary={activeMeta.id}>
          <div className="domain-summary-head">
            <strong>{activeMeta.title}</strong>
            <code>{activeMeta.api}</code>
          </div>
          <p className="muted small">
            {activeMeta.operations.join(" · ")}
          </p>
        </div>
      ) : null}

      {expanded ? (
        <ul className="domain-map" data-domain-map="true">
          {WORKBENCH_DOMAINS.map((domain) => (
            <li key={domain.id} data-domain-entry={domain.id}>
              <div>
                <strong>{domain.title}</strong>{" "}
                <code>{domain.api}</code>
              </div>
              <div className="muted small">
                CLI: <code>{domain.cli}</code>
              </div>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
