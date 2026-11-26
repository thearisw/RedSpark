/**
 * Web worker for the static demo: loads Pyodide and the redspark package
 * (redspark/browser.py), then passes JSON messages to and from it.
 * The files it loads are listed in py/manifest.json, written by vite.config.ts.
 */
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const HOME = "/home/pyodide";

let receive: ((text: string) => Promise<void>) | null = null;
let chain: Promise<void> = Promise.resolve(); // handle messages one at a time, in order
const early: string[] = [];

const post = (text: string) => (self as unknown as Worker).postMessage(text);

function handle(text: string): void {
  chain = chain.then(() => receive!(text)).catch((err) => post(JSON.stringify({ type: "error", message: String(err) })));
}

async function init(base: string): Promise<void> {
  const { loadPyodide } = await import(/* @vite-ignore */ `${PYODIDE}pyodide.mjs`);
  const [py, manifest] = await Promise.all([
    loadPyodide({ indexURL: PYODIDE }),
    fetch(new URL("py/manifest.json", base)).then((r) => r.json() as Promise<{ files: string[] }>),
  ]);
  await py.loadPackage("numpy");
  await Promise.all(
    manifest.files.map(async (path) => {
      const data = new Uint8Array(await (await fetch(new URL(`py/${path}`, base))).arrayBuffer());
      py.FS.mkdirTree(`${HOME}/${path.slice(0, path.lastIndexOf("/"))}`);
      py.FS.writeFile(`${HOME}/${path}`, data);
    }),
  );
  const browser = py.pyimport("redspark.browser");
  browser.start(post);
  receive = (text) => browser.receive(text);
  post(JSON.stringify({ type: "worker_ready" }));
  early.splice(0).forEach(handle);
}

self.onmessage = (ev: MessageEvent<string>) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "worker_init") {
    init(msg.base).catch((err) => post(JSON.stringify({ type: "worker_failed", message: String(err) })));
  } else if (receive) {
    handle(ev.data);
  } else {
    early.push(ev.data);
  }
};
