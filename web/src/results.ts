import { setPlayIcon, setResetIcon } from "./icons";
import type { SimSocket } from "./net";
import type { CircuitView } from "./render";
import type { ServerMessage, TrainResult } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/**
 * Train > Results: every new best circuit of the current run, newest first. Picking one
 * plays it like a saved circuit (view mode): Play / Step / Reset, and left click flips levers.
 */
export class ResultsPanel {
  active = false;
  private results: TrainResult[] = [];
  private selected: number | null = null; // id of the result being played
  private playing = false;

  constructor(private view: CircuitView, private socket: SimSocket) {
    setPlayIcon($("resPlay"), false);
    setResetIcon($("resReset"));
    $("resPlay").addEventListener("click", () => this.socket.send({ type: "play", on: !this.playing, tps: this.tps() }));
    $("resStep").addEventListener("click", () => this.socket.send({ type: "step" }));
    $("resReset").addEventListener("click", () => this.socket.send({ type: "reset" }));
    const tps = $<HTMLInputElement>("resTps");
    tps.addEventListener("input", () => {
      $("resTpsLabel").textContent = tps.value;
      if (this.playing) this.socket.send({ type: "play", on: true, tps: this.tps() });
    });
  }

  private tps = () => Number($<HTMLInputElement>("resTps").value);

  enter(): void {
    this.active = true;
    this.view.cellPicker = null; // clicks flip levers
    this.socket.send({ type: "train_results" });
    if (this.selected !== null) this.play(this.selected);
    else this.view.show({ size: [5, 1, 3], ticks: 0, blocks: [] });
    this.render();
  }

  leave(): void {
    this.active = false;
    if (this.playing) this.socket.send({ type: "play", on: false });
  }

  /** Handles results and (while open) simulation state. Returns true if it used the message. */
  handle(msg: ServerMessage): boolean {
    if (msg.type === "train_results") {
      this.results = msg.results;
      if (this.selected !== null && !this.results.some((r) => r.id === this.selected)) this.selected = null;
      if (this.active) this.render();
      return true;
    }
    if (msg.type === "state" && this.active) {
      this.playing = msg.playing;
      setPlayIcon($("resPlay"), msg.playing);
      if (msg.circuit && this.selected !== null) {
        this.view.show(msg.circuit);
        $("resTicks").textContent = String(msg.circuit.ticks);
      }
      return true;
    }
    return false;
  }

  private play(id: number): void {
    const r = this.results.find((x) => x.id === id);
    if (!r) return;
    this.selected = id;
    this.socket.send({ type: "load_data", circuit: r.circuit, name: `Result ${r.id}` });
    if (this.playing) this.socket.send({ type: "play", on: true, tps: this.tps() });
    this.render();
  }

  private render(): void {
    const list = $("resList");
    $("resPlayer").hidden = this.selected === null;
    if (!this.results.length) {
      list.innerHTML = `<p class="muted">No results yet. Run training in the Setup tab; every new best circuit shows up here.</p>`;
      return;
    }
    list.innerHTML = [...this.results].reverse().map((r) => {
      const solved = r.rows === r.rows_total;
      return `<button class="result${r.id === this.selected ? " on" : ""}" data-id="${r.id}">` +
        `<span class="${solved ? "ok" : "partial"}">${solved ? "✓" : `${r.rows}/${r.rows_total}`}</span>` +
        `<span>Gen ${r.gen}</span><span class="muted">${r.blocks} blocks · ${r.dims.join("×")} · ${r.delay} gt</span></button>`;
    }).join("");
    list.querySelectorAll<HTMLButtonElement>("button.result").forEach((b) =>
      b.addEventListener("click", () => this.play(Number(b.dataset.id))));
    const r = this.results.find((x) => x.id === this.selected);
    $("resCaption").textContent = r
      ? `Gen ${r.gen}: ${r.rows} / ${r.rows_total} rows, ${r.blocks} blocks, ${r.dims.join(" × ")}, settles in ${r.delay} game ticks.`
      : "";
  }
}
