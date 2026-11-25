import type { GenStats } from "./types";

// Validated (dark surface) categorical slots 1 and 2.
const BEST = "#3987e5";
const MEAN = "#d95926";
const INK = "#c3c2b7";
const MUTED = "#898781";
const GRID = "#2c2c2a";
const AXIS = "#383835";

/**
 * Rows passed per generation: the best network (line) and the population mean.
 * Long runs are bucketed to one point per pixel column (max of best, average of mean),
 * so drawing stays cheap at thousands of generations.
 */
export class ProgressChart {
  private ctx: CanvasRenderingContext2D;
  private best: number[] = [];
  private mean: number[] = [];
  private total = 1;
  private hoverX: number | null = null;
  private dirty = true;
  private pad = { l: 26, r: 8, t: 8, b: 18 };

  constructor(private canvas: HTMLCanvasElement, private tooltip: HTMLElement) {
    this.ctx = canvas.getContext("2d")!;
    canvas.addEventListener("pointermove", (e) => {
      this.hoverX = e.offsetX;
      this.dirty = true;
    });
    canvas.addEventListener("pointerleave", () => {
      this.hoverX = null;
      this.tooltip.hidden = true;
      this.dirty = true;
    });
    const loop = () => {
      if (this.dirty) this.draw();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  reset(total: number): void {
    this.best = [];
    this.mean = [];
    this.total = Math.max(1, total);
    this.dirty = true;
  }

  add(stats: GenStats[]): void {
    for (const s of stats) {
      this.best[s.gen] = s.best;
      this.mean[s.gen] = s.mean;
    }
    if (stats.length) this.dirty = true;
  }

  private draw(): void {
    this.dirty = false;
    const { canvas, ctx, pad } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;
    const n = this.best.length;
    const y = (v: number) => pad.t + ph - (v / this.total) * ph;

    // Recessive grid: one hairline per row count (or a few, for big tables).
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const stepRows = Math.max(1, Math.ceil(this.total / 4));
    for (let r = 0; r <= this.total; r += stepRows) {
      ctx.strokeStyle = r === 0 ? AXIS : GRID;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, Math.round(y(r)) + 0.5);
      ctx.lineTo(w - pad.r, Math.round(y(r)) + 0.5);
      ctx.stroke();
      ctx.fillStyle = MUTED;
      ctx.fillText(String(r), pad.l - 5, y(r));
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("gen 0", pad.l, h - 4);
    ctx.textAlign = "right";
    ctx.fillText(`gen ${Math.max(0, n - 1)}`, w - pad.r, h - 4);
    if (n === 0) return;

    // Bucket generations into pixel columns.
    const cols = Math.max(1, Math.min(n, Math.floor(pw)));
    const per = n / cols;
    const bx: number[] = [], bBest: number[] = [], bMean: number[] = [];
    for (let c = 0; c < cols; c++) {
      const a = Math.floor(c * per);
      const b = Math.max(a + 1, Math.floor((c + 1) * per));
      let mx = 0, sum = 0, cnt = 0;
      for (let g = a; g < b && g < n; g++) {
        if (this.best[g] === undefined) continue;
        mx = Math.max(mx, this.best[g]);
        sum += this.mean[g];
        cnt++;
      }
      if (!cnt) continue;
      bx.push(cols === 1 ? pad.l + pw / 2 : pad.l + (c / (cols - 1)) * pw);
      bBest.push(mx);
      bMean.push(sum / cnt);
    }
    const line = (vals: number[], color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      vals.forEach((v, i) => (i ? ctx.lineTo(bx[i], y(v)) : ctx.moveTo(bx[i], y(v))));
      ctx.stroke();
    };
    line(bMean, MEAN);
    line(bBest, BEST);

    if (this.hoverX !== null && bx.length) {
      let i = 0;
      for (let k = 1; k < bx.length; k++) if (Math.abs(bx[k] - this.hoverX) < Math.abs(bx[i] - this.hoverX)) i = k;
      ctx.strokeStyle = MUTED;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(bx[i]) + 0.5, pad.t);
      ctx.lineTo(Math.round(bx[i]) + 0.5, pad.t + ph);
      ctx.stroke();
      for (const [v, color] of [[bBest[i], BEST], [bMean[i], MEAN]] as const) {
        ctx.fillStyle = color;
        ctx.strokeStyle = "#14161c";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(bx[i], y(v), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      const g0 = Math.floor(i * per);
      const g1 = Math.max(g0, Math.floor((i + 1) * per) - 1);
      const gens = g0 === g1 ? `Gen ${g0}` : `Gens ${g0}–${g1}`;
      this.tooltip.innerHTML =
        `<b>${gens}</b><br><span class="key" style="background:${BEST}"></span>Best ${bBest[i]} / ${this.total} rows` +
        `<br><span class="key" style="background:${MEAN}"></span>Mean ${bMean[i].toFixed(2)}`;
      this.tooltip.hidden = false;
      this.tooltip.style.left = `${Math.min(bx[i] + 10, w - 150)}px`;
    }
    ctx.fillStyle = INK;
  }
}
