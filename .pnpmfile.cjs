// pnpm install hook. TypeDoc 0.28 only supports TypeScript <= 6, while the repo
// itself compiles with TS 7, so TypeDoc gets its own TypeScript 6. A
// `typedoc>typescript` override cannot express this split (pnpm still resolves
// TypeDoc's peer from the workspace's own TypeScript), hence this hook.
// Re-check when TypeDoc supports TS 7, then delete this file.
module.exports = {
  hooks: {
    readPackage(pkg) {
      if (pkg.name === "typedoc") {
        pkg.dependencies = { ...pkg.dependencies, typescript: "6.0.3" };
        if (pkg.peerDependencies) {
          delete pkg.peerDependencies.typescript;
        }
      }
      return pkg;
    },
  },
};
