import type { ServerMessage } from "./types";

/** Thin WebSocket client that reconnects if the backend restarts (the app build; see net.ts). */
export class SimSocket {
  private ws: WebSocket | null = null;
  private queue: string[] = [];

  constructor(
    private onMessage: (msg: ServerMessage) => void,
    private onStatus: (text: string, isError?: boolean) => void,
    private onOpen: () => void,
  ) {
    this.connect();
  }

  private connect(): void {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => {
      this.onStatus("Connected");
      this.queue.splice(0).forEach((m) => ws.send(m));
      this.onOpen();
    };
    ws.onmessage = (ev) => this.onMessage(JSON.parse(ev.data) as ServerMessage);
    ws.onclose = () => {
      this.onStatus("Backend not reachable, retrying...", true);
      setTimeout(() => this.connect(), 1500);
    };
  }

  send(msg: object): void {
    const text = JSON.stringify(msg);
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(text);
    else this.queue.push(text);
  }

  /** Names of the circuits in circuits/ (including ones saved from training). */
  async listCircuits(): Promise<string[]> {
    return (await fetch("/api/circuits")).json();
  }
}
