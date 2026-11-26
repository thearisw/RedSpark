import type { ServerMessage } from "./types";

/**
 * Same interface as the WebSocket client, but the simulator runs in a web worker
 * (worker.ts) with Pyodide, so the static web demo needs no server.
 */
export class SimSocket {
  private worker: Worker;
  private ready = false;
  private queue: string[] = [];
  private lists: ((names: string[]) => void)[] = [];

  constructor(
    private onMessage: (msg: ServerMessage) => void,
    private onStatus: (text: string, isError?: boolean) => void,
    private onOpen: () => void,
  ) {
    this.onStatus("Loading the simulator (Python in your browser, first visit takes a few seconds)...");
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<string>) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "worker_ready") {
        this.ready = true;
        this.onStatus("Running in your browser");
        this.queue.splice(0).forEach((m) => this.worker.postMessage(m));
        this.onOpen();
      } else if (msg.type === "worker_failed") {
        this.onStatus(`Could not start the simulator: ${msg.message}`, true);
      } else if (msg.type === "circuits") {
        this.lists.shift()?.(msg.names);
      } else {
        this.onMessage(msg as ServerMessage);
      }
    };
    this.worker.postMessage(JSON.stringify({ type: "worker_init", base: document.baseURI }));
  }

  send(msg: object): void {
    const text = JSON.stringify(msg);
    if (this.ready) this.worker.postMessage(text);
    else this.queue.push(text);
  }

  /** Names of the example circuits, plus any saved from training on this page. */
  listCircuits(): Promise<string[]> {
    return new Promise((resolve) => {
      this.lists.push(resolve);
      this.send({ type: "list" });
    });
  }
}
