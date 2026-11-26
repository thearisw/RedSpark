import { ProgressChart } from "./chart";
import type { SimSocket } from "./net";
import type { CircuitView } from "./render";
import type { BlockState, CircuitState, ServerMessage, ShowMode } from "./types";

type Pos = [number, number, number];
type Out = boolean | null; // null = don't care

const INPUT_NAMES = ["A", "B", "C", "D"];
const OUTPUT_NAMES = ["OUT1", "OUT2", "OUT3", "OUT4"];
const PRESETS: Record<string, (v: boolean[]) => boolean> = {
  "NOT A": (v) => !v[0],
  "Copy A": (v) => v[0],
  "A AND B": (v) => v[0] && v[1],
  "A OR B": (v) => v[0] || v[1],
  "A XOR B": (v) => v[0] !== v[1],
  "A NAND B": (v) => !(v[0] && v[1]),
  "A NOR B": (v) => !(v[0] || v[1]),
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
/** Bounding box as "W × H × D" (x, y, z). */
const dimsText = (d: number[]) => `${d[0]} × ${d[1]} × ${d[2]}`;
/** Activation time in game ticks (20 per second). */
const delayText = (gt: number) => `${gt} gt to activate`;
const samePos = (a: Pos, b: Pos) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
type TrainCircuit = Extract<ServerMessage, { type: "train_circuit" }>;
const KIND_NAME = (k: TrainCircuit["kind"]) =>
  k === "best_ever" ? "Best so far" : k === "success" ? "Success" : k === "best" ? "New best" : k === "samples" ? "Sample" : "Newest attempt";
const fmt = (n: number) => n.toLocaleString("en-US");
/** "gen 12" or, for a tested sample, "sample 431 (gen 12)". */
const genText = (c: TrainCircuit) => (c.sample != null ? `sample ${fmt(c.sample)} (gen ${c.gen})` : `gen ${c.gen}`);

/** Truth-table training: place inputs/outputs, fill the table, run and watch generations. */
export class TrainMode {
  active = false;
  private size: Pos = [5, 1, 5];
  private layer = 0;
  private inputs = new Map<string, Pos>([["A", [0, 0, 1]]]);
  private outputs = new Map<string, Pos>([["OUT1", [4, 0, 1]]]);
  private table = new Map<string, Out>(); // key: "<inputs bits>|<output label>"
  private running = false;
  private startedTask: string | null = null; // task JSON the server is training on
  private rowsTotal = 0;
  private shown: TrainCircuit | null = null;
  /** 2x2 multi-view: the four latest circuits the Show filter let through, newest first. */
  private multi = false;
  private panes: TrainCircuit[] = [];
  private selected: TrainCircuit | null = null; // pane whose truth table is shown
  private paneLabels: HTMLDivElement[] = [];
  private frameIdx = 0;
  private chart: ProgressChart;
  /** Recent (time ms, samples) readings for the samples/s figure. */
  private rate: [number, number][] = [];

  constructor(private view: CircuitView, private socket: SimSocket) {
    this.size = [5, 1, 3];
    this.chart = new ProgressChart($("chart"), $("chartTip"));
    this.applyPreset("NOT A");

    for (const [id, axis] of [["sizeX", 0], ["sizeY", 1], ["sizeZ", 2]] as const) {
      const el = $<HTMLInputElement>(id);
      el.value = String(this.size[axis]);
      el.addEventListener("change", () => {
        this.resize(axis, Math.round(Number(el.value) || 1) - this.size[axis], 1);
        el.value = String(this.size[axis]);
      });
    }
    $<HTMLSelectElement>("layer").addEventListener("change", (e) => {
      this.layer = Number((e.target as HTMLSelectElement).value);
      this.showEditor();
    });
    const preset = $<HTMLSelectElement>("preset");
    preset.innerHTML = `<option value="">Fill OUT1 with…</option>` +
      Object.keys(PRESETS).map((k) => `<option>${k}</option>`).join("");
    preset.addEventListener("change", () => {
      if (preset.value) this.applyPreset(preset.value);
      preset.value = "";
    });

    $("run").addEventListener("click", () => this.toggleRun());
    $("restart").addEventListener("click", () => this.reset());
    $("showBest").addEventListener("click", () => socket.send({ type: "train_show_best" }));
    $("editGrid").addEventListener("click", () => {
      this.shown = null;
      this.showEditor();
    });
    $("save").addEventListener("click", () =>
      socket.send({ type: "train_save", name: $<HTMLInputElement>("saveName").value || "designed" }));
    for (const id of ["gps", "showMode", "renderHz"]) $(id).addEventListener("input", () => this.sendConfig());
    const showSamples = $<HTMLInputElement>("showSamples");
    showSamples.checked = localStorage.getItem("redspark.showSamples") !== "0";
    const applySamples = () => {
      document.querySelectorAll<HTMLElement>(".sampleStat").forEach((el) => (el.hidden = !showSamples.checked));
      localStorage.setItem("redspark.showSamples", showSamples.checked ? "1" : "0");
    };
    showSamples.addEventListener("change", applySamples);
    applySamples();
    $("renderHz").addEventListener("input", () => ($("renderHzLabel").textContent = $<HTMLInputElement>("renderHz").value));

    const overlay = $("multiPanes");
    this.paneLabels = [0, 1, 2, 3].map(() => overlay.appendChild(document.createElement("div")));
    $<HTMLInputElement>("multiView").addEventListener("change", (e) => this.setMulti((e.target as HTMLInputElement).checked));
    this.view.onPaneClick = (i) => {
      if (!this.panes[i]) return;
      this.selected = this.panes[i];
      this.drawPanes();
    };
    window.addEventListener("resize", () => this.layoutPanes());

    // Cycle the shown circuit through its truth-table rows.
    // Cycle the shown circuit through its truth-table rows while training runs.
    setInterval(() => {
      if (!this.active || !this.shown || !this.running) return;
      this.frameIdx = (this.frameIdx + 1) % this.shown.frames.length;
      this.drawShown();
    }, 700);
  }

  // ------------------------------------------------------------------ mode
  enter(): void {
    this.active = true;
    this.view.cellPicker = { layer: this.layer, onPick: (p, button) => this.pick(p, button === "left" ? "input" : "output") };
    if (this.shown) this.drawShown();
    else this.showEditor();
  }

  leave(): void {
    this.view.setStretch(null);
    this.active = false;
    this.view.cellPicker = null;
    this.view.showLayer(null);
    this.view.setMulti(null);
    this.layoutPanes();
  }

  // ------------------------------------------------------------ multi-view
  private setMulti(on: boolean): void {
    this.multi = on;
    this.panes = on && this.shown ? [this.shown] : [];
    this.selected = this.panes[0] ?? null;
    if (!this.active) return;
    if (on && this.panes.length) this.drawPanes();
    else if (on) this.status("Multi-view: the next circuits from training fill the four panes.");
    else if (this.shown) this.drawShown();
    else this.showEditor();
  }

  /** The space between the two side panels, where the 2x2 grid goes. */
  private paneArea = () => {
    const W = window.innerWidth;
    const H = window.innerHeight;
    const left = $("panel").getBoundingClientRect().right + 12;
    const side = $("side");
    const right = side.hidden ? W - 12 : side.getBoundingClientRect().left - 12;
    return right - left >= 360 ? { left, top: 12, width: right - left, height: H - 24 } : { left: 0, top: 0, width: W, height: H };
  };

  /** Put a caption over each pane (hidden when multi-view is not on screen). */
  private layoutPanes(): void {
    const rects = this.view.paneRects();
    this.paneLabels.forEach((el, i) => {
      const p = this.panes[i];
      const r = rects[i];
      el.hidden = !r;
      if (!r) return;
      Object.assign(el.style, { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` });
      el.classList.toggle("selected", !!p && p === this.selected);
      el.textContent = p
        ? `${KIND_NAME(p.kind)}${i === 0 && p.kind !== "all" && p.kind !== "samples" ? " (newest)" : ""} · ${genText(p)} · ${p.rows_passed} / ${p.rows_total} rows · ${p.blocks} blocks`
        : "Waiting for a circuit…";
    });
  }

  private drawPanes(): void {
    if (!this.active) return;
    this.view.setStretch(null);
    this.view.showLayer(null);
    this.view.setMulti(this.paneArea);
    for (let i = 0; i < 4; i++) {
      const p = this.panes[i];
      this.view.showPane(i, p ? p.frames[this.frameIdx % p.frames.length] : null);
    }
    this.layoutPanes();
    const s = this.selected;
    $("shownCaption").textContent = s
      ? `Table: ${KIND_NAME(s.kind)}, ${genText(s)}, ${s.rows_passed} / ${s.rows_total} rows, ${s.blocks} blocks, ${dimsText(s.dims)}, ${delayText(s.delay)}. ` +
        `Showing row ${this.frameIdx + 1} in every pane. Click a pane to see its table.`
      : "";
    this.renderTable();
  }

  /** The circuit whose results the truth table shows. */
  private current(): TrainCircuit | null {
    return this.multi && this.view.multi ? this.selected : this.shown;
  }

  /** The backend restarted, so any training run is gone. */
  disconnected(): void {
    this.running = false;
    this.startedTask = null;
    $("run").textContent = "Run";
  }

  // ---------------------------------------------------------------- editing
  private sortedInputs(): [string, Pos][] {
    return [...this.inputs].sort(([a], [b]) => a.localeCompare(b));
  }

  private sortedOutputs(): [string, Pos][] {
    return [...this.outputs].sort(([a], [b]) => a.localeCompare(b));
  }

  private combos(): boolean[][] {
    const n = this.inputs.size;
    return Array.from({ length: 2 ** n }, (_, r) =>
      Array.from({ length: n }, (_, i) => Boolean((r >> (n - 1 - i)) & 1)));
  }

  private bits = (combo: boolean[]) => combo.map((b) => (b ? "1" : "0")).join("");

  /**
   * Left click places an input, right click an output. Clicking a cell that already holds the
   * same kind removes it; clicking an input with the output button (or the other way round) swaps it.
   */
  private pick(p: Pos, kind: "input" | "output"): void {
    this.shown = null; // placing happens on the empty grid
    const [names, map] = kind === "input" ? [INPUT_NAMES, this.inputs] : [OUTPUT_NAMES, this.outputs];
    const same = [...map].find(([, q]) => samePos(p, q));
    if (same) {
      map.delete(same[0]);
      return this.edited();
    }
    const name = names.find((n) => !map.has(n));
    if (!name) {
      this.status(`At most ${names.length} ${kind}s. Click one to remove it first.`, true);
      return this.edited();
    }
    const other = kind === "input" ? this.outputs : this.inputs;
    for (const [k, q] of other) if (samePos(p, q)) other.delete(k);
    map.set(name, p);
    this.edited();
  }

  /**
   * Grow (cells > 0) or shrink one side of the grid; side -1 is the low end, which moves every
   * input and output along. Anything that falls off the grid is removed. Returns the change made.
   */
  private resize(axis: 0 | 1 | 2, cells: number, side: 1 | -1): number {
    const old = this.size[axis];
    const v = Math.max(1, Math.min(axis === 1 ? 3 : 9, old + cells));
    const change = v - old;
    if (!change) return 0;
    this.size[axis] = v;
    const shift = side < 0 ? change : 0;
    for (const m of [this.inputs, this.outputs])
      for (const [k, p] of m) {
        const q: Pos = [p[0], p[1], p[2]];
        q[axis] += shift;
        if (q[axis] < 0 || q[axis] >= v) m.delete(k);
        else m.set(k, q);
      }
    this.layer = Math.min(this.layer, this.size[1] - 1);
    $<HTMLInputElement>(["sizeX", "sizeY", "sizeZ"][axis]).value = String(v);
    this.edited();
    return change;
  }

  private applyPreset(name: string): void {
    const fn = PRESETS[name];
    const needs = name.includes("B") ? 2 : 1;
    if (this.inputs.size < needs) return this.status(`${name} needs ${needs} inputs.`, true);
    if (!this.outputs.has("OUT1")) return this.status("Place OUT1 first.", true);
    for (const combo of this.combos()) this.table.set(`${this.bits(combo)}|OUT1`, fn(combo));
    this.edited();
  }

  private edited(): void {
    this.renderTable();
    this.showEditor();
    const fresh = this.startedTask !== null && this.startedTask !== JSON.stringify(this.task());
    $("taskNote").textContent = fresh ? "Task changed: Run starts a new population." : "";
  }

  private task() {
    return {
      size: this.size,
      inputs: this.sortedInputs().map(([label, pos]) => ({ label, pos })),
      outputs: this.sortedOutputs().map(([label, pos]) => ({ label, pos })),
      rows: this.combos().map((c) => ({
        in: c,
        out: this.sortedOutputs().map(([o]) => this.table.get(`${this.bits(c)}|${o}`) ?? null),
      })),
    };
  }

  private showEditor(): void {
    if (!this.active) return;
    // placing needs the single view; multi-view comes back with the next circuit from training
    this.view.setMulti(null);
    this.layoutPanes();
    const blocks: BlockState[] = [
      ...[...this.inputs].map(([label, pos]): BlockState => ({ pos, type: "lever", label, active: false, on: false })),
      ...[...this.outputs].map(([label, pos]): BlockState => ({ pos, type: "lamp", label, active: false })),
    ];
    this.view.show({ size: this.size, ticks: 0, blocks });
    this.view.showLayer(this.layer);
    this.view.setStretch((axis, side, cells) => this.resize(axis, cells, side)); // drag handles resize the grid
    if (this.view.cellPicker) this.view.cellPicker.layer = this.layer;
    const layerSel = $<HTMLSelectElement>("layer");
    layerSel.innerHTML = Array.from({ length: this.size[1] }, (_, y) => `<option value="${y}">${y}</option>`).join("");
    layerSel.value = String(this.layer);
    layerSel.disabled = this.size[1] === 1;
    $("shownCaption").textContent = "Editing: left click places an input, right click an output. Click one again to remove it.";
  }

  private renderTable(): void {
    const ins = this.sortedInputs().map(([n]) => n);
    const outs = this.sortedOutputs().map(([n]) => n);
    const el = $("truth");
    if (!ins.length || !outs.length) {
      el.innerHTML = `<p class="muted">Place at least one input and one output.</p>`;
      return;
    }
    const shown = this.current();
    const rowClass = (r: number) => {
      if (!shown || shown.passed.length !== 2 ** ins.length) return "";
      return (shown.passed[r] ? "pass" : "fail") + (r === this.frameIdx ? " current" : "");
    };
    const head = `<tr>${ins.map((n) => `<th>${n}</th>`).join("")}<th class="sep"></th>${outs.map((n) => `<th>${n}</th>`).join("")}${shown ? "<th>Got</th>" : ""}</tr>`;
    const body = this.combos().map((c, r) => {
      const cells = outs.map((o) => {
        const v = this.table.get(`${this.bits(c)}|${o}`) ?? null;
        return `<td><button class="cell" data-key="${this.bits(c)}|${o}">${v === null ? "–" : v ? "1" : "0"}</button></td>`;
      }).join("");
      const got = shown ? `<td class="got">${shown.got[r]?.map((g) => (g ? "1" : "0")).join(" ") ?? ""} ${shown.passed[r] ? "✓" : "✗"}</td>` : "";
      return `<tr data-row="${r}" class="${rowClass(r)}">${c.map((b) => `<td>${b ? 1 : 0}</td>`).join("")}<td class="sep"></td>${cells}${got}</tr>`;
    }).join("");
    el.innerHTML = `<table>${head}${body}</table>`;
    el.querySelectorAll<HTMLTableRowElement>("tr[data-row]").forEach((tr) =>
      tr.addEventListener("click", (e) => {
        if ((e.target as HTMLElement).closest("button") || !this.current()) return;
        this.frameIdx = Number(tr.dataset.row);
        this.drawShown();
      }));
    el.querySelectorAll<HTMLButtonElement>("button.cell").forEach((b) =>
      b.addEventListener("click", () => {
        const k = b.dataset.key!;
        const v = this.table.get(k) ?? null;
        this.table.set(k, v === null ? false : v === false ? true : null); // 0 -> 1 -> don't care
        this.edited();
      }));
  }

  // --------------------------------------------------------------- training
  private config() {
    const gps = Number($<HTMLSelectElement>("gps").value);
    return {
      gps,
      show: $<HTMLSelectElement>("showMode").value as ShowMode,
      render_hz: Number($<HTMLInputElement>("renderHz").value),
    };
  }

  private sendConfig(): void {
    if (this.startedTask !== null) this.socket.send({ type: "train_config", ...this.config() });
  }

  private start(): void {
    const task = this.task();
    this.startedTask = JSON.stringify(task);
    this.rowsTotal = task.rows.length;
    this.chart.reset(this.rowsTotal);
    this.rate = [];
    this.shown = null;
    this.panes = [];
    this.selected = null;
    $("taskNote").textContent = "";
    this.socket.send({
      type: "train_start", task, max_blocks: Number($<HTMLInputElement>("maxBlocks").value),
      method: $<HTMLSelectElement>("trainMethod").value, ...this.config(),
    });
  }

  /** Stop, throw the population away and go back to the empty grid. */
  private reset(): void {
    if (this.startedTask !== null) this.socket.send({ type: "train_pause" });
    this.startedTask = null;
    this.running = false;
    this.shown = null;
    this.panes = [];
    this.selected = null;
    this.chart.reset(this.combos().length);
    for (const id of ["genBest", "genSolved", "bestEver", "sampleRate"]) $(id).textContent = "–";
    $("gen").textContent = "0";
    $("samples").textContent = "0";
    this.rate = [];
    $("run").textContent = "Run";
    $("taskNote").textContent = "";
    this.renderTable();
    this.showEditor();
    this.status("Reset. Run starts a new population.");
  }

  private toggleRun(): void {
    if (this.running) {
      $("run").textContent = "Pausing…";
      return this.socket.send({ type: "train_pause" });
    }
    if (this.startedTask === JSON.stringify(this.task())) this.socket.send({ type: "train_resume", ...this.config() });
    else this.start();
  }

  handle(msg: ServerMessage): boolean {
    if (msg.type === "train_progress") {
      if (this.startedTask === null) return true; // left over from a run that was reset
      if (this.running && !msg.running) this.status(`Paused at generation ${msg.gen}.`);
      this.running = msg.running;
      $("run").textContent = msg.running ? "Pause" : "Run";
      this.chart.add(msg.stats);
      const last = msg.stats[msg.stats.length - 1];
      const restarts = msg.stats.filter((x) => x.restarted).length;
      const budget = $<HTMLInputElement>("maxBlocks");
      if (msg.max_blocks && Number(budget.value) < msg.max_blocks) {
        budget.value = String(msg.max_blocks);
        this.status(`Max blocks raised to ${msg.max_blocks}: a wire from the inputs to the outputs needs more than you set.`);
      }
      if (restarts) this.status(`Stuck for a while, so training started over with fresh networks (gen ${msg.stats.filter((x) => x.restarted).pop()!.gen}). The best circuit so far is kept.`);
      const t = msg.rows_total;
      $("gen").textContent = String(msg.gen);
      this.showSampleCount(msg.samples, msg.simulated, msg.running);
      if (last) {
        $("genBest").textContent = `${last.best} / ${t} rows`;
        $("genSolved").textContent = String(last.solved);
      }
      const b = msg.best_ever;
      $("bestEver").textContent = b
        ? b.rows === t
          ? `✓ solved: ${b.blocks} blocks, ${dimsText(b.dims)}, ${delayText(b.delay)} (gen ${b.gen})`
          : `${b.rows} / ${t} rows (gen ${b.gen})`
        : "–";
      return true;
    }
    if (msg.type === "train_circuit") {
      if (this.startedTask === null) return true;
      // Keep the row counter going, so a stream of new circuits still steps through every row.
      this.frameIdx = this.frameIdx % msg.frames.length;
      this.shown = msg;
      if (this.multi) {
        // newest goes top-left and the rest move along; the draw limit still caps how often this happens
        this.panes = [msg, ...this.panes].slice(0, 4);
        if (!this.selected || !this.panes.includes(this.selected)) this.selected = msg;
      }
      if (this.active) this.drawShown();
      return true;
    }
    if (msg.type === "saved") {
      this.status(`Saved as circuits/${msg.name}.json. Open it from the View tab.`);
      return true;
    }
    return false;
  }

  /** Samples tested so far, and how many per second over the last couple of seconds. */
  private showSampleCount(samples: number, simulated: number | null, running: boolean): void {
    const cached = simulated == null ? null : samples - simulated;
    $("samples").textContent = fmt(samples) + (cached ? ` (${fmt(cached)} from cache)` : "");
    $("samples").title = simulated == null ? "" : `${fmt(simulated)} simulated, ${fmt(cached ?? 0)} seen before and taken from the cache`;
    const now = performance.now();
    if (!running) {
      this.rate = [];
      $("sampleRate").textContent = "–";
      return;
    }
    this.rate.push([now, samples]);
    while (this.rate.length > 2 && now - this.rate[0][0] > 2000) this.rate.shift();
    const [t0, s0] = this.rate[0];
    if (now - t0 >= 300) $("sampleRate").textContent = fmt(Math.round(((samples - s0) * 1000) / (now - t0)));
  }

  private drawShown(): void {
    if (this.multi && this.panes.length) return this.drawPanes();
    const s = this.shown;
    if (!s || !this.active) return;
    this.view.setStretch(null);
    this.view.setMulti(null);
    this.layoutPanes();
    const frame: CircuitState = s.frames[this.frameIdx % s.frames.length];
    this.view.show(frame);
    this.view.showLayer(null);
    const what = KIND_NAME(s.kind);
    $("shownCaption").textContent =
      `${what}: ${genText(s)}, ${s.rows_passed} / ${s.rows_total} rows, ${s.blocks} blocks, ${dimsText(s.dims)}, ${delayText(s.delay)}. Showing row ${this.frameIdx + 1}.`;
    this.renderTable();
  }

  private status(text: string, isError = false): void {
    const el = $("status");
    el.textContent = text;
    el.classList.toggle("error", isError);
  }
}
