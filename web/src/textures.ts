/**
 * Original pixel-art textures for RedSpark, drawn in code at 16x16 (or smaller).
 * Nothing here is copied or traced from any game: every pixel comes from the
 * palettes and rules below, with a seeded random generator so the art is the
 * same on every run.
 */
import * as THREE from "three";

type Draw = (px: Painter) => void;

/** Small deterministic RNG (mulberry32) so textures never change between runs. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Painter {
  readonly ctx: CanvasRenderingContext2D;
  readonly rand: () => number;
  constructor(readonly w: number, readonly h: number, seed: number) {
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    this.ctx = canvas.getContext("2d")!;
    this.rand = rng(seed);
  }
  set(x: number, y: number, color: string): void {
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return;
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x, y, 1, 1);
  }
  rect(x: number, y: number, w: number, h: number, color: string): void {
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x, y, w, h);
  }
  /** Fill an area, picking each pixel from a weighted palette. */
  noise(x: number, y: number, w: number, h: number, palette: [string, number][]): void {
    const total = palette.reduce((s, [, wt]) => s + wt, 0);
    for (let j = y; j < y + h; j++)
      for (let i = x; i < x + w; i++) {
        let r = this.rand() * total;
        for (const [c, wt] of palette) {
          if ((r -= wt) <= 0) {
            this.set(i, j, c);
            break;
          }
        }
      }
  }
  pick<T>(items: T[]): T {
    return items[Math.floor(this.rand() * items.length)];
  }
}

const cache = new Map<string, THREE.Texture>();

function texture(key: string, w: number, h: number, seed: number, draw: Draw): THREE.Texture {
  let tex = cache.get(key);
  if (!tex) {
    const p = new Painter(w, h, seed);
    draw(p);
    tex = new THREE.CanvasTexture(p.ctx.canvas);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    tex.colorSpace = THREE.SRGBColorSpace;
    cache.set(key, tex);
  }
  return tex;
}

// ------------------------------------------------------------------ palettes
const STONE: [string, number][] = [["#7b7e86", 6], ["#868991", 4], ["#70737b", 3], ["#92959c", 1]];
const SLAB: [string, number][] = [["#b4b6bb", 6], ["#a9abb1", 3], ["#bfc1c5", 1]];
const WOOD = ["#7a5230", "#694627", "#8a5f38"];
const RED_ON = { core: "#ff3b1f", hi: "#ffb08a", mid: "#e0220f" };
const RED_OFF = { core: "#6a1e15", hi: "#8a3326", mid: "#4f150f" };

// --------------------------------------------------------------------- stone
export const stoneTexture = () =>
  texture("stone", 16, 16, 11, (p) => {
    p.noise(0, 0, 16, 16, STONE);
    // a few darker chips and short cracks
    for (let k = 0; k < 5; k++) {
      let x = Math.floor(p.rand() * 16);
      let y = Math.floor(p.rand() * 16);
      for (let s = 0; s < 3; s++) {
        p.set(x, y, "#5f626a");
        x += p.pick([-1, 0, 1]);
        y += 1;
      }
    }
  });

/** Rounded stones with dark mortar, for lever bases. */
export const cobbleTexture = () =>
  texture("cobble", 16, 16, 23, (p) => {
    p.rect(0, 0, 16, 16, "#4c4e55");
    const stones = [[1, 1, 6, 4], [8, 1, 7, 5], [1, 6, 4, 5], [6, 7, 5, 4], [12, 7, 3, 4], [1, 12, 7, 3], [9, 12, 6, 3]];
    for (const [x, y, w, h] of stones) {
      p.noise(x, y, w, h, [["#8b8e95", 3], ["#7c7f87", 3], ["#9a9da3", 1]]);
      p.rect(x, y + h - 1, w, 1, "#6a6d74"); // shaded lower edge
      p.set(x, y, "#4c4e55"); // round the corners
      p.set(x + w - 1, y, "#4c4e55");
    }
  });

// --------------------------------------------------------------------- lamps
export const lampTexture = (lit: boolean) =>
  texture(`lamp-${lit}`, 16, 16, 31, (p) => {
    const frame = lit ? "#9a6a2a" : "#3d2b1b";
    const pane: [string, number][] = lit
      ? [["#ffd36b", 5], ["#ffe49a", 3], ["#fff4cf", 1]]
      : [["#6b4a28", 5], ["#7a5732", 3], ["#5c3f22", 2]];
    p.rect(0, 0, 16, 16, frame);
    for (const [x, y] of [[1, 1], [9, 1], [1, 9], [9, 9]]) {
      p.noise(x, y, 6, 6, pane);
      p.set(x + 1, y + 1, lit ? "#ffffff" : "#8c6a40"); // glint
    }
    p.rect(7, 0, 2, 16, frame);
    p.rect(0, 7, 16, 2, frame);
  });

// --------------------------------------------------------------------- torch
/** 2 x 10 strip wrapped around a thin box: wooden stick with a glowing head. */
export const torchTexture = (lit: boolean) =>
  texture(`torch-${lit}`, 2, 10, 41, (p) => {
    for (let y = 2; y < 10; y++) for (let x = 0; x < 2; x++) p.set(x, y, p.pick(WOOD));
    const c = lit ? RED_ON : RED_OFF;
    p.set(0, 0, c.hi);
    p.set(1, 0, c.core);
    p.set(0, 1, c.core);
    p.set(1, 1, c.mid);
  });

/** Same strip, only the head: used as the emissive map so the stick does not glow. */
export const torchGlowTexture = () =>
  texture("torch-glow", 2, 10, 41, (p) => {
    p.rect(0, 0, 2, 10, "#000000");
    p.rect(0, 0, 2, 2, "#ffffff");
  });

export const torchTopTexture = (lit: boolean) =>
  texture(`torch-top-${lit}`, 2, 2, 43, (p) => {
    const c = lit ? RED_ON : RED_OFF;
    p.set(0, 0, c.hi);
    p.set(1, 0, c.core);
    p.set(0, 1, c.core);
    p.set(1, 1, c.mid);
  });

export const woodTexture = () =>
  texture("wood", 2, 8, 47, (p) => {
    for (let y = 0; y < 8; y++) for (let x = 0; x < 2; x++) p.set(x, y, p.pick(WOOD));
  });

// ------------------------------------------------------------------ repeater
/** Top face, drawn pointing +x (right): a signal trace with a chevron at the front. */
export const repeaterTopTexture = (on: boolean, delay: number) =>
  texture(`repeater-${on}-${delay}`, 16, 16, 53, (p) => {
    p.noise(0, 0, 16, 16, SLAB);
    p.rect(0, 0, 16, 1, "#9a9ca2");
    p.rect(0, 15, 16, 1, "#9a9ca2");
    const c = on ? RED_ON : RED_OFF;
    for (let x = 2; x < 11; x++) {
      p.set(x, 7, c.core);
      p.set(x, 8, x % 3 === 0 ? c.hi : c.mid);
    }
    // chevron
    for (let i = 0; i < 4; i++) {
      p.set(10 + i, 4 + i, c.core);
      p.set(10 + i, 11 - i, c.core);
      p.set(11 + i, 4 + i, c.mid);
      p.set(11 + i, 11 - i, c.mid);
    }
    // one 2x2 pip per tick of delay, along the bottom edge
    for (let i = 0; i < delay; i++) {
      p.rect(2 + i * 3, 12, 2, 2, c.core);
      p.set(2 + i * 3, 12, c.hi);
    }
  });

export const repeaterGlowTexture = (delay: number) =>
  texture(`repeater-glow-${delay}`, 16, 16, 53, (p) => {
    p.rect(0, 0, 16, 16, "#000000");
    for (let i = 0; i < delay; i++) p.rect(2 + i * 3, 12, 2, 2, "#ffffff");
    p.rect(2, 7, 9, 2, "#ffffff");
    for (let i = 0; i < 4; i++) {
      p.rect(10 + i, 4 + i, 2, 1, "#ffffff");
      p.rect(10 + i, 11 - i, 2, 1, "#ffffff");
    }
  });

export const slabSideTexture = () =>
  texture("slab-side", 16, 3, 59, (p) => {
    p.noise(0, 0, 16, 3, SLAB);
    p.rect(0, 2, 16, 1, "#8f9197");
  });

// ---------------------------------------------------------------------- dust
/**
 * Dust drawn in light greys on a transparent background, tinted by the
 * material colour so one texture works for every power level.
 * `mask` bits: 1 = +x, 2 = -x, 4 = +z, 8 = -z (canvas right/left/down/up).
 */
export const dustTexture = (mask: number) =>
  texture(`dust-${mask}`, 16, 16, 61 + mask, (p) => {
    const shades = ["#ffffff", "#e2e2e2", "#c4c4c4"];
    const blob = (x: number, y: number) => p.set(x, y, p.pick(shades));
    // A lone connection still draws a straight line through the cell.
    let m = mask;
    if (m === 1 || m === 2) m = 3;
    if (m === 4 || m === 8) m = 12;
    if (m === 0) {
      // dot: a small rounded cluster
      for (let y = 5; y < 11; y++)
        for (let x = 5; x < 11; x++) {
          const edge = (x === 5 || x === 10) && (y === 5 || y === 10);
          if (!edge) blob(x, y);
        }
      return;
    }
    const arm = (dx: number, dy: number) => {
      for (let t = 0; t < 8; t++) {
        // three pixels wide, with ragged edges
        for (let k = -1; k <= 2; k++) {
          const edge = k === -1 || k === 2;
          if (edge && p.rand() < 0.6) continue;
          if (dx !== 0) blob(7 + dx * t + (dx > 0 ? 1 : 0), 7 + k);
          else blob(7 + k, 7 + dy * t + (dy > 0 ? 1 : 0));
        }
      }
    };
    if (m & 1) arm(1, 0);
    if (m & 2) arm(-1, 0);
    if (m & 4) arm(0, 1);
    if (m & 8) arm(0, -1);
    // junction knot where arms meet
    for (let y = 5; y < 11; y++)
      for (let x = 5; x < 11; x++) {
        const corner = (x === 5 || x === 10) && (y === 5 || y === 10);
        if (!corner && p.rand() < 0.9) blob(x, y);
      }
  });

/** Dust colour for a power level 0..15: dark maroon up to bright red. */
export function dustColor(level: number): THREE.Color {
  const t = level / 15;
  return new THREE.Color("#5e1710").lerp(new THREE.Color("#ff2a12"), t);
}

/** Floor plate: faint pixel checker so each grid cell reads clearly. */
export const floorTexture = () =>
  texture("floor", 16, 16, 71, (p) => {
    p.noise(0, 0, 16, 16, [["#24272e", 6], ["#272a32", 3], ["#21242a", 2]]);
    p.rect(0, 0, 16, 1, "#2e323b");
    p.rect(0, 0, 1, 16, "#2e323b");
  });

// ---------------------------------------------------------------- comparator
/**
 * Comparator top face, drawn pointing +x (right). Two input pads at the back,
 * a split trace that joins in the middle, and a round mode lamp at the front
 * that lights in subtract mode. Distinct from the repeater's chevron.
 */
export const comparatorTopTexture = (on: boolean, subtract: boolean) =>
  texture(`comparator-${on}-${subtract}`, 16, 16, 79, (p) => {
    p.noise(0, 0, 16, 16, [["#a7a3a0", 6], ["#9c9895", 3], ["#b3afac", 1]]);
    p.rect(0, 0, 16, 1, "#8e8a87");
    p.rect(0, 15, 16, 1, "#8e8a87");
    const c = on ? RED_ON : RED_OFF;
    // back input pads
    for (const y of [3, 11]) {
      p.rect(2, y, 2, 2, c.core);
      p.set(2, y, c.hi);
    }
    // two traces that merge at x = 9
    for (let x = 4; x < 9; x++) {
      const t = (x - 4) / 5;
      p.set(x, Math.round(4 + t * 3), c.mid);
      p.set(x, Math.round(11 - t * 3), c.mid);
    }
    p.rect(9, 7, 2, 2, c.core);
    // front mode lamp, ringed in dark stone
    const m = subtract ? RED_ON : RED_OFF;
    p.rect(11, 6, 4, 4, "#6d6966");
    p.rect(12, 7, 2, 2, m.core);
    p.set(12, 7, m.hi);
  });

export const comparatorGlowTexture = (on: boolean, subtract: boolean) =>
  texture(`comparator-glow-${on}-${subtract}`, 16, 16, 79, (p) => {
    p.rect(0, 0, 16, 16, "#000000");
    if (on) {
      for (const y of [3, 11]) p.rect(2, y, 2, 2, "#ffffff");
      for (let x = 4; x < 9; x++) {
        const t = (x - 4) / 5;
        p.set(x, Math.round(4 + t * 3), "#ffffff");
        p.set(x, Math.round(11 - t * 3), "#ffffff");
      }
      p.rect(9, 7, 2, 2, "#ffffff");
    }
    if (subtract) p.rect(12, 7, 2, 2, "#ffffff");
  });
