/**
 * The connection to the simulator. "#transport" is picked per build in vite.config.ts:
 * - `npm run build` / `npm run dev`: transport-ws.ts, a WebSocket to the Python server.
 * - `npm run build:static`: transport-worker.ts, the same Python code run in the browser
 *   with Pyodide (the web demo, no server).
 */
export { SimSocket } from "#transport";
