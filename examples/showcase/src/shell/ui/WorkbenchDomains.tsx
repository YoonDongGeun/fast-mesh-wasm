// Seven-domain architecture navigator and summary drawer for the showcase
// right sidebar: organizes workbench workflows under the same 7 domain
// namespaces shared by Rust (`fast_mesh_ops`), TypeScript (`fast-mesh-wasm`),
// and the CLI (`fast-mesh`).
import { useState } from "react";
import { useDomainFilter, WORKBENCH_DOMAINS, type WorkbenchDomainId } from "../ux/workbench-domains";

export interface WorkbenchDomainsProps {
  activeDomain?: WorkbenchDomainId | null;
  onSelectDomain?: (domain: WorkbenchDomainId | null) => void;
}

export function WorkbenchDomains(props: WorkbenchDomainsProps) {
  const { activeDomain: controlled, onSelectDomain } = props;
  const { active, activeMeta, selectDomain } = useDomainFilter(controlled, onSelectDomain);
  const [expanded, setExpanded] = useState(false);

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
