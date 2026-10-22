// Shared by browser-check.mjs and bundler-check.mjs: a headless Chrome driven
// over the DevTools protocol (no dependencies: Node's WebSocket), and a static
// file server that sends the cross-origin isolation headers.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, normalize, resolve } from "node:path";

/** The COOP/COEP headers that make a page cross-origin isolated. */
export const ISOLATION = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Chrome or Chromium: `CHROME_PATH`, else the usual install places. */
export function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ];
  const found = candidates.find((p) => p && existsSync(p));
  if (!found) {
    throw new Error("Chrome not found: set CHROME_PATH");
  }
  return found;
}

/** A DevTools protocol connection: `send` on the browser or a flat session, events by name. */
class Cdp {
  #ws;
  #id = 0;
  #pending = new Map();
  #handlers = new Map();

  static async connect(url) {
    const cdp = new Cdp();
    cdp.#ws = new WebSocket(url);
    await new Promise((resolve, reject) => {
      cdp.#ws.addEventListener("open", resolve, { once: true });
      cdp.#ws.addEventListener("error", () => reject(new Error(`cannot connect to ${url}`)), { once: true });
    });
    cdp.#ws.addEventListener("message", (e) => cdp.#receive(JSON.parse(e.data)));
    return cdp;
  }

  #receive(msg) {
    if (msg.id !== undefined) {
      const p = this.#pending.get(msg.id);
      this.#pending.delete(msg.id);
      if (msg.error) {
        p?.reject(new Error(`${p.method}: ${msg.error.message}`));
      } else {
        p?.resolve(msg.result);
      }
      return;
    }
    for (const h of this.#handlers.get(msg.method) ?? []) {
      h(msg.params, msg.sessionId);
    }
  }

  send(method, params = {}, sessionId) {
    const id = ++this.#id;
    this.#ws.send(JSON.stringify({ id, method, params, sessionId }));
    return new Promise((resolve, reject) => this.#pending.set(id, { resolve, reject, method }));
  }

  on(method, handler) {
    if (!this.#handlers.has(method)) {
      this.#handlers.set(method, []);
    }
    this.#handlers.get(method).push(handler);
  }

  close() {
    this.#ws.close();
  }
}

/**
 * Launches headless Chrome and opens a page. Returns `{ goto, evaluate,
 * workers, consoleLines, close }`; `workers()` lists the live dedicated
 * Workers of the page, nested ones included (auto-attached).
 */
export async function launchChrome({ verbose = false, args = [] } = {}) {
  const profile = mkdtempSync(join(tmpdir(), "fast-mesh-chrome-"));
  const chrome = spawn(
    findChrome(),
    [
      "--headless=new",
      "--remote-debugging-port=0",
      `--user-data-dir=${profile}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      "--disable-background-timer-throttling",
      ...(process.platform === "linux" ? ["--no-sandbox"] : []),
      ...args,
      "about:blank",
    ],
    { stdio: "ignore" },
  );
  const portFile = join(profile, "DevToolsActivePort");
  const t0 = Date.now();
  let lines = [];
  for (;;) {
    try {
      // Read while Chrome writes it: on Windows that can fail (EBUSY) or be partial.
      lines = readFileSync(portFile, "utf8").trim().split("\n");
    } catch {
      // not there yet
    }
    if (lines.length >= 2) {
      break;
    }
    if (Date.now() - t0 > 30_000 || chrome.exitCode !== null) {
      chrome.kill();
      throw new Error("Chrome did not start");
    }
    await sleep(50);
  }
  const [port, path] = lines;
  const cdp = await Cdp.connect(`ws://127.0.0.1:${port}${path}`);
  const version = await cdp.send("Browser.getVersion");
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
  const { sessionId: page } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });

  const workers = new Map(); // sessionId -> url
  const consoleLines = [];
  const log = (where, text) => {
    consoleLines.push(`${where}: ${text}`);
    if (verbose) {
      console.log(`  [${where}] ${text}`);
    }
  };
  const autoAttach = (sessionId) =>
    cdp.send("Target.setAutoAttach", { autoAttach: true, waitForDebuggerOnStart: false, flatten: true }, sessionId);
  cdp.on("Target.attachedToTarget", ({ sessionId, targetInfo }) => {
    if (targetInfo.type === "worker") {
      workers.set(sessionId, targetInfo.url);
      void autoAttach(sessionId).catch(() => {});
      void cdp.send("Runtime.enable", {}, sessionId).catch(() => {});
    }
  });
  cdp.on("Target.detachedFromTarget", ({ sessionId }) => workers.delete(sessionId));
  cdp.on("Runtime.consoleAPICalled", ({ type, args: a }, sessionId) =>
    log(sessionId === page ? "page" : "worker", `${type}: ${a.map((x) => x.value ?? x.description ?? "").join(" ")}`),
  );
  cdp.on("Runtime.exceptionThrown", ({ exceptionDetails: d }, sessionId) =>
    log(sessionId === page ? "page" : "worker", `exception: ${d.exception?.description ?? d.text}`),
  );
  await cdp.send("Runtime.enable", {}, page);
  await cdp.send("Page.enable", {}, page);
  await autoAttach(page);

  async function evaluate(expression) {
    const r = await cdp.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }, page);
    if (r.exceptionDetails) {
      throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    }
    return r.result.value;
  }

  return {
    name: version.product,
    async goto(url, ready = "true") {
      await cdp.send("Page.navigate", { url }, page);
      const t = Date.now();
      for (;;) {
        try {
          if (await evaluate(ready)) {
            return;
          }
        } catch {
          // navigating
        }
        if (Date.now() - t > 30_000) {
          throw new Error(`the page did not get ready: ${url}\n${consoleLines.join("\n")}`);
        }
        await sleep(100);
      }
    },
    evaluate,
    workers: () => [...workers.values()],
    consoleLines,
    async close() {
      try {
        await cdp.send("Browser.close");
      } catch {
        // already gone
      }
      cdp.close();
      chrome.kill();
      for (let i = 0; i < 20; i++) {
        try {
          rmSync(profile, { recursive: true, force: true });
          return;
        } catch {
          await sleep(250); // Windows: Chrome still holds its profile
        }
      }
    },
  };
}

const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".css": "text/css",
  ".map": "application/json",
};

/** Serves `dir` on a free port with `headers` (the isolation ones by default); resolves `{ url, close }`. */
export function serveStatic(dir, headers = ISOLATION) {
  const root = resolve(dir);
  const server = createServer((req, res) => {
    const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    let file = normalize(join(root, path));
    if (!file.startsWith(root)) {
      res.writeHead(403).end();
      return;
    }
    if (existsSync(file) && statSync(file).isDirectory()) {
      file = join(file, "index.html");
    }
    if (!existsSync(file)) {
      res.writeHead(404, headers).end("not found");
      return;
    }
    res.writeHead(200, { ...headers, "Content-Type": TYPES[extname(file)] ?? "application/octet-stream" });
    res.end(readFileSync(file));
  });
  return new Promise((resolveServer) =>
    server.listen(0, "127.0.0.1", () =>
      resolveServer({
        url: `http://127.0.0.1:${server.address().port}`,
        close: () => new Promise((r) => server.close(r)),
      }),
    ),
  );
}

/** Starts a server process (vite) and resolves once `url` answers; `stop()` ends it. */
export async function startServer(command, args, { cwd, url }) {
  const child = spawn(command, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  child.stdout.on("data", (d) => (output += d));
  child.stderr.on("data", (d) => (output += d));
  const t0 = Date.now();
  for (;;) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        break;
      }
    } catch {
      // not up yet
    }
    if (child.exitCode !== null || Date.now() - t0 > 60_000) {
      child.kill();
      throw new Error(`${command} ${args.join(" ")} did not start:\n${output}`);
    }
    await sleep(200);
  }
  return {
    output: () => output,
    stop: () => {
      child.kill();
    },
  };
}
