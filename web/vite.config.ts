import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));

/**
 * Static demo build: copies the Python package and example circuits into py/ next to the
 * page, with py/manifest.json listing them, for the Pyodide worker (src/worker.ts) to load.
 */
function pythonFiles(): Plugin {
  const PY = ["__init__", "sim", "tasks", "designer", "session", "browser"].map((m) => `redspark/${m}.py`);
  return {
    name: "redspark-python-files",
    generateBundle() {
      const circuits = readdirSync(`${root}circuits`).filter((f) => f.endsWith(".json")).map((f) => `circuits/${f}`);
      const files = [...PY, ...circuits];
      for (const f of files) this.emitFile({ type: "asset", fileName: `py/${f}`, source: readFileSync(`${root}${f}`) });
      this.emitFile({ type: "asset", fileName: "py/manifest.json", source: JSON.stringify({ files }) });
    },
  };
}

// `npm run dev` serves on :5173 and forwards API/WebSocket calls to the Python server on :8000.
// `npm run build` writes the viewer into the Python package, so the RedSpark app can serve it
// without Node.
// `npm run build:static` writes the web demo to web/dist: the simulator runs in the browser
// (Pyodide), no server needed. Deployed to GitHub Pages by .github/workflows/pages.yml.
export default defineConfig(({ mode }) => {
  const isStatic = mode === "static";
  return {
    base: "./",
    resolve: {
      alias: { "#transport": fileURLToPath(new URL(`./src/transport-${isStatic ? "worker" : "ws"}.ts`, import.meta.url)) },
    },
    plugins: isStatic ? [pythonFiles()] : [],
    build: {
      outDir: isStatic ? "dist" : "../redspark/viewer",
      emptyOutDir: true,
      // stable file names, so rebuilding overwrites the same files instead of piling up new ones
      rolldownOptions: { output: { entryFileNames: "assets/index.js", chunkFileNames: "assets/[name].js", assetFileNames: "assets/[name][extname]" } },
    },
    server: {
      port: 5173,
      proxy: {
        "/api": "http://localhost:8000",
        "/ws": { target: "ws://localhost:8000", ws: true },
      },
    },
  };
});
