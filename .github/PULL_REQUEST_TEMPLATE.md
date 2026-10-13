## What and why

<!-- What does this change, and which issue does it address? e.g. "Closes #12" -->

## Checklist

- [ ] `pnpm lint` and `pnpm test` pass locally
- [ ] New behavior is covered by a Rust test in `crates/fast-mesh` (and a JS test if the API changed)
- [ ] Public API changes have `///` doc comments (they become the TypeScript docs)
- [ ] Added a changeset with `pnpm changeset` (skip for docs/CI/test-only changes)
