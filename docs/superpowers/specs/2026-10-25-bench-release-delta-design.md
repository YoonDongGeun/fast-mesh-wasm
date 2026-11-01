# Bench deltas on the release PR — short design

**Date:** 2026-10-25. **Status:** draft (my calls; user opted out of questions).

## Intent and why

Every release should show what got faster/slower: when the bot opens or updates
the Version Packages PR, CI runs the benchmark suites on the PR head and on the
previous release, and posts a per-case time/memory delta table as a PR comment.
The user reads the comment (and later the release notes derived from it) to see
what each release improved.

## Goals

- `bench/*.bench.mjs` can emit machine-readable JSON (`--json out.json`) with
  zero change to their console output.
- A compare step diffs head vs the previous `fast-mesh-wasm@*` release tag and
  renders a markdown delta table (median, p95, heap), marking |Δ| < 5% as noise.
- A workflow posts/refreshes one sticky comment on the Version Packages PR.
- Informational only: never blocks merging or publishing.

## Non-goals (draft)

- Rust/criterion benches (none exist; Rust perf stays ad-hoc).
- `--worker` / `--10m` / `--wide` suites (nightly follow-up, not the PR flow).
- Historical dashboard, trend storage, release-notes injection.
- Bench runs on Windows/macOS (ubuntu-latest only).

## Anti-goals

- Presenting runner noise as real change (hence the threshold + noise label).
- Slowing the publish path: benches run on the release PR, not in `release.yml`.

## Constraints

- GitHub-hosted runners are noisy: run baseline and head back-to-back in one
  job on the same runner; treat |Δ| < 5% as noise (const, tunable).
- Benches need the release `dist/` build (existing `__test_` guard stays).
- Heap numbers need `--expose-gc`; without it the column is omitted, not faked.

## Approaches

1. **Separate `bench-release.yml` on `changeset-release/*` PRs (chosen).**
   Right trigger (PR opened/updated), isolated from publish, easy to re-run.
2. Extend `release.yml`. Rejected: it runs on push to main and owns publish;
   benching there delays releases and has no PR to comment on.
3. Nightly benches + dashboard. Rejected for the draft: the user wants
   per-release deltas where the release is prepared, not a separate site.

## Decided

- JSON schema v1: `{schema, suite, node, platform, quick, cases[]}`,
  case = `{name, unit ("ms"|"µs"), median, p95, min?, max?, n, heapMB?}`.
- Scene bench units are all ms; store bench mixes ms (bursts) and µs (rows).
- Baseline = latest `fast-mesh-wasm@*` tag reachable from the PR base.
- Statuses: 🟢 ≥5% faster, 🔴 ≥5% slower, ➖ noise; added/removed cases listed.
- Sticky comment via `actions/github-script` (no third-party action).
- Compare script lives in `scripts/` (repo convention for CI tooling) as
  dependency-free ESM, tested with `node --test`; the reporter helper lives
  next to the benches and is covered by the package's vitest run.

## Left to the builder

- Comment body polish (exact wording, emoji set, details/summary folding).
- Sampling detail: which raw `console.log` measurements become JSON cases
  (all of them; names verbatim).
