# Changesets

Every pull request that changes what users get from `fast-mesh-wasm` should include a changeset.
Run `pnpm changeset`, pick the bump type and write one line for the changelog.

| Change | Before 1.0 | After 1.0 |
| --- | --- | --- |
| Breaking API change | `minor` | `major` |
| New feature | `patch` | `minor` |
| Bug fix / performance | `patch` | `patch` |

Docs, CI and test-only changes do not need a changeset.
See [CONTRIBUTING.md](../CONTRIBUTING.md#releasing) for how releases are published.
