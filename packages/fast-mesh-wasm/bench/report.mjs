// Shared bench reporter: console passthrough plus an optional schema-v1 JSON
// document (`--json <path>`). Console text is byte-identical with or without
// the flag; without it, cases are not accumulated and `save()` is a no-op.
import { writeFileSync } from "node:fs";

const fmt = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));

function jsonPath(argv) {
  const i = argv.indexOf("--json");
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : undefined;
}

export function createReporter(argv, { suite } = {}) {
  const path = jsonPath(argv);
  const cases = [];
  const reporter = {
    report(name, stats, unit = "ms", extra = "") {
      console.log(
        `${name}: median ${fmt(stats.median)} ms, p95 ${fmt(stats.p95)} ms, min ${fmt(stats.min ?? NaN)}, max ${fmt(stats.max ?? NaN)} (n=${stats.n})${extra}`,
      );
      reporter.record(name, stats, unit);
    },
    // A case without the stats-line print: for benches whose console format is
    // not the stats line (the store bench prints table rows via `log()`).
    record(name, stats, unit = "ms") {
      if (path === undefined) {
        return;
      }
      const entry = { name, unit, median: stats.median, p95: stats.p95, n: stats.n };
      if (stats.min !== undefined) {
        entry.min = stats.min;
      }
      if (stats.max !== undefined) {
        entry.max = stats.max;
      }
      cases.push(entry);
    },
    log(line) {
      console.log(line);
    },
    document() {
      return { schema: 1, suite, quick: argv.includes("--quick"), cases: cases.map((c) => ({ ...c })) };
    },
    save() {
      if (path === undefined) {
        return;
      }
      writeFileSync(path, `${JSON.stringify(reporter.document(), null, 2)}\n`);
    },
  };
  return reporter;
}
