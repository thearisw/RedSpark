import { defineConfig } from "vite";

// `npm run dev` serves on :5173 and forwards API/WebSocket calls to the Python server on :8000.
// `npm run build` writes the viewer into the Python package, so the RedSpark app can serve it
// without Node.
export default defineConfig({
  base: "./",
  build: {
    outDir: "../redspark/viewer",
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
});
