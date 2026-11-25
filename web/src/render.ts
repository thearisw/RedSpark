import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import type { BlockState, CircuitState, Facing } from "./types";

const FACING: Record<Facing, THREE.Vector3> = {
  "+x": new THREE.Vector3(1, 0, 0),
  "-x": new THREE.Vector3(-1, 0, 0),
  "+z": new THREE.Vector3(0, 0, 1),
  "-z": new THREE.Vector3(0, 0, -1),
  "+y": new THREE.Vector3(0, 1, 0),
  "-y": new THREE.Vector3(0, -1, 0),
};

import * as T from "./textures";

const PX = 1 / 16; // one texture pixel in world units

function stdMat(opts: THREE.MeshStandardMaterialParameters): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ roughness: 0.95, metalness: 0, ...opts });
}

const RED = new THREE.Color("#ff2a12");
const matCache = new Map<string, THREE.Material | THREE.Material[]>();
function cached<M extends THREE.Material | THREE.Material[]>(key: string, make: () => M): M {
  let m = matCache.get(key) as M | undefined;
  if (!m) {
    m = make();
    matCache.set(key, m);
  }
  return m;
}

const geo = {
  cube: new THREE.BoxGeometry(1, 1, 1),
  dust: new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
  torch: new THREE.BoxGeometry(2 * PX, 10 * PX, 2 * PX),
  smallTorch: new THREE.BoxGeometry(2 * PX, 5 * PX, 2 * PX),
  slab: new THREE.BoxGeometry(1, 3 * PX, 1),
  leverBase: new THREE.BoxGeometry(6 * PX, 3 * PX, 8 * PX),
  handle: new THREE.BoxGeometry(2 * PX, 8 * PX, 2 * PX),
  lockBar: new THREE.BoxGeometry(2 * PX, 2 * PX, 14 * PX),
};

const mats = {
  stone: (powered: boolean) =>
    cached(`stone-${powered}`, () =>
      stdMat({ map: T.stoneTexture(), emissive: powered ? new THREE.Color("#5a0c04") : new THREE.Color(0) })),
  lamp: (lit: boolean) =>
    cached(`lamp-${lit}`, () =>
      stdMat({
        map: T.lampTexture(lit),
        emissive: lit ? new THREE.Color("#ffffff") : new THREE.Color(0),
        emissiveMap: lit ? T.lampTexture(true) : null,
        emissiveIntensity: lit ? 0.75 : 0,
      })),
  torch: (lit: boolean, small = false) =>
    cached(`torch-${lit}-${small}`, () => {
      const side = stdMat({
        map: T.torchTexture(lit),
        emissive: lit ? RED : new THREE.Color(0),
        emissiveMap: T.torchGlowTexture(),
        emissiveIntensity: lit ? 2.5 : 0,
      });
      const top = stdMat({ map: T.torchTopTexture(lit), emissive: lit ? RED : new THREE.Color(0), emissiveIntensity: lit ? 2.5 : 0 });
      const bottom = stdMat({ map: T.woodTexture() });
      return [side, side, top, bottom, side, side];
    }),
  repeater: (on: boolean, delay: number) =>
    cached(`repeater-${on}-${delay}`, () => {
      const side = stdMat({ map: T.slabSideTexture() });
      const top = stdMat({
        map: T.repeaterTopTexture(on, delay),
        emissive: on ? RED : new THREE.Color(0),
        emissiveMap: T.repeaterGlowTexture(delay),
        emissiveIntensity: on ? 2.2 : 0,
      });
      return [side, side, top, side, side, side];
    }),
  comparator: (on: boolean, subtract: boolean) =>
    cached(`comparator-${on}-${subtract}`, () => {
      const side = stdMat({ map: T.slabSideTexture() });
      const top = stdMat({
        map: T.comparatorTopTexture(on, subtract),
        emissive: on || subtract ? RED : new THREE.Color(0),
        emissiveMap: T.comparatorGlowTexture(on, subtract),
        emissiveIntensity: on || subtract ? 2.2 : 0,
      });
      return [side, side, top, side, side, side];
    }),
  cobble: () => cached("cobble", () => stdMat({ map: T.cobbleTexture() })),
  wood: () => cached("wood", () => stdMat({ map: T.woodTexture() })),
  dust: (mask: number, level: number) =>
    cached(`dust-${mask}-${level}`, () => {
      const color = T.dustColor(level);
      return stdMat({
        map: T.dustTexture(mask),
        color,
        emissive: color,
        emissiveMap: T.dustTexture(mask),
        emissiveIntensity: level > 0 ? 0.6 + (level / 15) * 1.6 : 0,
        transparent: true,
        alphaTest: 0.5,
        depthWrite: false,
      });
    }),
};

const key = (x: number, y: number, z: number) => `${x},${y},${z}`;
type Lookup = (x: number, y: number, z: number) => BlockState | undefined;
export type Rect = { left: number; top: number; width: number; height: number };

/** Which sides a dust cell connects to: 1 = +x, 2 = -x, 4 = +z, 8 = -z. */
function dustMask(b: BlockState, at: Lookup): number {
  const [x, y, z] = b.pos;
  const sides: [number, number, number, "x" | "z"][] = [[1, 0, 1, "x"], [-1, 0, 2, "x"], [0, 1, 4, "z"], [0, -1, 8, "z"]];
  let mask = 0;
  for (const [dx, dz, bit, axis] of sides) {
    const n = at(x + dx, y, z + dz);
    let link = false;
    if (n) {
      if (n.type === "dust" || n.type === "lever" || n.type === "torch" || n.type === "lamp") link = true;
      if (n.type === "repeater" || n.type === "comparator") link = (n.facing ?? "+x").endsWith(axis);
    }
    if (!link && (at(x + dx, y + 1, z + dz)?.type === "dust" || at(x + dx, y - 1, z + dz)?.type === "dust")) link = true;
    if (link) mask |= bit;
  }
  return mask;
}

const YAW: Record<string, number> = { "+x": 0, "-x": Math.PI, "+z": -Math.PI / 2, "-z": Math.PI / 2 };

function labelSprite(text: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgba(15,17,22,0.85)";
  ctx.roundRect(4, 4, 120, 56, 12);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 34px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 64, 34);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
  sprite.scale.set(0.8, 0.4, 1);
  return sprite;
}

function mesh(g: THREE.BufferGeometry, m: THREE.Material | THREE.Material[], x = 0, y = 0, z = 0): THREE.Mesh {
  const out = new THREE.Mesh(g, m);
  out.position.set(x, y, z);
  out.castShadow = true;
  out.receiveShadow = true;
  return out;
}

/** Build the object for one block, centred on its cell. */
function buildBlock(b: BlockState, at: Lookup): THREE.Object3D {
  const g = new THREE.Group();
  switch (b.type) {
    case "solid":
      g.add(mesh(geo.cube, mats.stone(b.active)));
      break;
    case "dust": {
      const d = mesh(geo.dust, mats.dust(b.shape ?? dustMask(b, at), b.level ?? 0), 0, -0.5 + PX * 0.25, 0);
      d.castShadow = false;
      g.add(d);
      break;
    }
    case "lamp":
      g.add(mesh(geo.cube, mats.lamp(b.active)));
      break;
    case "torch": {
      const dir = FACING[b.facing ?? "-y"];
      const torch = mesh(geo.torch, mats.torch(b.active));
      if (dir.y < 0) {
        torch.position.y = -0.5 + 5 * PX; // standing on the block below
      } else {
        // leaning out from the block beside it
        torch.position.set(dir.x * 0.32, -0.5 + 7 * PX, dir.z * 0.32);
        torch.rotation.set(-dir.z * 0.4, 0, dir.x * 0.4);
      }
      g.add(torch);
      break;
    }
    case "repeater": {
      const body = new THREE.Group();
      body.add(mesh(geo.slab, mats.repeater(b.active, b.delay ?? 1), 0, -0.5 + 1.5 * PX, 0));
      if (b.locked) body.add(mesh(geo.lockBar, mats.cobble(), -1 * PX, -0.5 + 4 * PX, 0));
      body.rotation.y = YAW[b.facing ?? "+x"] ?? 0;
      g.add(body);
      break;
    }
    case "comparator": {
      const subtract = b.mode === "subtract";
      const body = new THREE.Group();
      body.add(mesh(geo.slab, mats.comparator(b.active, subtract), 0, -0.5 + 1.5 * PX, 0));
      // two input torches at the back, one shorter mode torch at the front
      for (const z of [-4, 4]) body.add(mesh(geo.smallTorch, mats.torch(b.active, true), -5 * PX, -0.5 + 5.5 * PX, z * PX));
      const modeTorch = mesh(geo.smallTorch, mats.torch(subtract, true), 5 * PX, -0.5 + 4.5 * PX, 0);
      modeTorch.scale.y = 0.6;
      body.add(modeTorch);
      body.rotation.y = YAW[b.facing ?? "+x"] ?? 0;
      g.add(body);
      g.userData.lever = b.pos; // clickable: switches compare/subtract mode
      break;
    }
    case "lever": {
      // Built standing on the floor, then turned to face the block it is attached to.
      const lever = new THREE.Group();
      lever.add(mesh(geo.leverBase, mats.cobble(), 0, -0.5 + 1.5 * PX, 0));
      const pivot = new THREE.Group();
      pivot.position.y = -0.5 + 3 * PX;
      pivot.rotation.z = b.active ? -0.65 : 0.65;
      pivot.add(mesh(geo.handle, mats.wood(), 0, 4 * PX, 0));
      // the tip lights up red when the lever is on
      const tip = mesh(geo.smallTorch, mats.torch(b.active, true), 0, 8 * PX, 0);
      tip.scale.set(1.4, 0.4, 1.4);
      pivot.add(tip);
      lever.add(pivot);
      const dir = FACING[b.facing ?? "-y"];
      if (dir.y === 0) {
        // on a wall: tip the lever over so its base sits against that wall
        lever.rotation.set(-dir.z * Math.PI / 2, 0, dir.x * Math.PI / 2);
      }
      g.add(lever);
      g.userData.lever = b.pos;
      break;
    }
  }
  if (b.label) {
    const s = labelSprite(b.label);
    s.position.y = 1.05;
    g.add(s);
  }
  const [x, y, z] = b.pos;
  g.position.set(x, y + 0.5, z);
  return g;
}

export class CircuitView {
  readonly renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
  private controls: OrbitControls;
  private world = new THREE.Group();
  private floor = new THREE.Group();
  private size: string | null = null;
  private raycaster = new THREE.Raycaster();
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private layerOutline = new THREE.Group();
  /** When set, left and right clicks pick a grid cell on this layer (y) instead of flipping levers. */
  cellPicker: { layer: number; onPick: (pos: [number, number, number], button: "left" | "right") => void } | null = null;
  /** 2x2 multi-view: one circuit per pane, drawn with this one renderer through scissor viewports. */
  private paneWorlds = Array.from({ length: 4 }, () => new THREE.Group());
  private multiArea: (() => Rect) | null = null;
  private paneSize = { w: 0, h: 0 };
  /**
   * Stretch handles: when set, coloured balls on the grid's sides can be dragged to resize it.
   * ``onStretch(axis, side, cells)`` grows (cells > 0) or shrinks that side and returns how many
   * cells it actually changed (after limits). side -1 is the low end (x = 0 or z = 0).
   */
  private stretch: ((axis: 0 | 1 | 2, side: 1 | -1, cells: number) => number) | null = null;
  private handles = new THREE.Group();
  private drag: { axis: 0 | 1 | 2; side: 1 | -1; origin: THREE.Vector3; dir: THREE.Vector3; applied: number; id: number } | null = null;
  /** Called with the pane index (0-3) when a pane is clicked in multi-view. */
  onPaneClick: ((pane: number) => void) | null = null;

  constructor(container: HTMLElement, onLeverClick: (pos: [number, number, number]) => void) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false }); // crisp pixel edges
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x15171c);
    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30281f, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(6, 12, 4);
    sun.castShadow = true;
    this.scene.add(sun, this.world, this.floor, this.layerOutline, this.handles);
    for (const g of this.paneWorlds) {
      g.visible = false;
      this.scene.add(g);
    }

    // Bloom makes the emissive (powered) pixels glow red; dark textures stay crisp.
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.6, 0.3, 0.85);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    // A click is a press and release of one button with (almost) no movement in between, so
    // orbiting (left drag) and panning (right drag) never count as clicks.
    let down: { x: number; y: number; button: number; moved: boolean } | null = null;
    const el = this.renderer.domElement;
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    // capture phase, so a press on a stretch handle runs before OrbitControls and can turn it off
    el.addEventListener("pointerdown", (e) => {
      down = { x: e.clientX, y: e.clientY, button: e.button, moved: false };
      if (e.button === 0 && this.startStretch(e)) {
        down = null;
        this.controls.enabled = false;
        el.setPointerCapture(e.pointerId);
      }
    }, { capture: true });
    el.addEventListener("pointermove", (e) => {
      if (this.drag) return this.moveStretch(e);
      if (this.stretch && !down) el.style.cursor = this.handleAt(e) ? "grab" : "";
      if (!down) return;
      // e.button >= 0 on a move means another button was pressed or released mid-gesture
      if (e.button >= 0 || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) down.moved = true;
    });
    const endStretch = () => {
      if (!this.drag) return false;
      this.drag = null;
      this.controls.enabled = true;
      el.style.cursor = "";
      return true;
    };
    el.addEventListener("pointercancel", () => {
      down = null;
      endStretch();
    });
    el.addEventListener("pointerup", (e) => {
      if (endStretch()) return;
      const d = down;
      down = null;
      if (!d || d.moved || d.button !== e.button) return; // it was a drag or a multi-button gesture
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return;
      if (e.button !== 0 && e.button !== 2) return;
      const button = e.button === 0 ? "left" : "right";
      if (this.multiArea) {
        const i = this.paneRects().findIndex((r) =>
          e.clientX >= r.x && e.clientX < r.x + r.w && e.clientY >= r.y && e.clientY < r.y + r.h);
        if (i >= 0 && button === "left") this.onPaneClick?.(i);
        return;
      }
      if (this.cellPicker) {
        const cell = this.pickCell(e, this.cellPicker.layer);
        if (cell) this.cellPicker.onPick(cell, button);
        return;
      }
      if (button !== "left") return;
      const lever = this.pick(e);
      if (lever) onLeverClick(lever);
    });

    window.addEventListener("resize", () => this.resize());
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      if (this.multiArea) this.renderPanes();
      else this.composer.render();
    });
  }

  /** Turn the 2x2 multi-view on (panes are laid out inside ``area()``, in CSS pixels) or off (null). */
  setMulti(area: (() => Rect) | null): void {
    if (!area === !this.multiArea) {
      this.multiArea = area;
      return;
    }
    this.multiArea = area;
    for (const g of this.paneWorlds) g.visible = false;
    this.world.visible = true;
    if (!area) {
      for (const g of this.paneWorlds) g.clear();
      this.resize();
    }
    this.paneSize = { w: 0, h: 0 };
  }

  get multi(): boolean {
    return this.multiArea !== null;
  }

  /** The four panes in CSS pixels from the top left: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right. */
  paneRects(): { x: number; y: number; w: number; h: number }[] {
    if (!this.multiArea) return [];
    const a = this.multiArea();
    const gap = 8;
    const w = Math.max(1, Math.floor((a.width - gap) / 2));
    const h = Math.max(1, Math.floor((a.height - gap) / 2));
    return [0, 1, 2, 3].map((i) => ({
      x: Math.round(a.left) + (i % 2) * (w + gap),
      y: Math.round(a.top) + Math.floor(i / 2) * (h + gap),
      w,
      h,
    }));
  }

  /** Draw ``circuit`` in multi-view pane ``i`` (null leaves it empty). */
  showPane(i: number, circuit: CircuitState | null): void {
    const g = this.paneWorlds[i];
    g.clear();
    if (!circuit) return;
    this.fit(circuit.size);
    const byPos = new Map(circuit.blocks.map((b) => [key(...b.pos), b]));
    const at: Lookup = (x, y, z) => byPos.get(key(x, y, z));
    for (const b of circuit.blocks) g.add(buildBlock(b, at));
  }

  private renderPanes(): void {
    const r = this.renderer;
    const H = window.innerHeight;
    const rects = this.paneRects();
    const { w, h } = rects[0];
    if (w !== this.paneSize.w || h !== this.paneSize.h) {
      // every pane has the same size, so one composer (and its bloom buffers) is reused for all four
      this.paneSize = { w, h };
      this.composer.setSize(w, h);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
    r.setScissorTest(false);
    r.setViewport(0, 0, window.innerWidth, H);
    r.setClearColor(0x0d0e12);
    r.clear();
    this.world.visible = false;
    rects.forEach((p, i) => {
      const y = H - p.y - p.h; // WebGL counts from the bottom
      r.setViewport(p.x, y, p.w, p.h);
      r.setScissor(p.x, y, p.w, p.h);
      r.setScissorTest(true);
      this.paneWorlds.forEach((g, j) => (g.visible = i === j));
      this.composer.render();
    });
    r.setScissorTest(false);
    r.setViewport(0, 0, window.innerWidth, H);
    for (const g of this.paneWorlds) g.visible = false;
    this.world.visible = true;
  }

  /** Show stretch handles on the grid (null hides them). */
  setStretch(onStretch: ((axis: 0 | 1 | 2, side: 1 | -1, cells: number) => number) | null): void {
    this.stretch = onStretch;
    if (!onStretch) this.drag = null;
    this.drawHandles();
  }

  private drawHandles(): void {
    this.handles.clear();
    if (!this.stretch || !this.size) return;
    const [X, Y, Z] = this.size.split("x").map(Number);
    const cx = (X - 1) / 2;
    const cz = (Z - 1) / 2;
    const gap = 0.75;
    const spots: [0 | 1 | 2, 1 | -1, number, number, number, number][] = [
      [0, 1, X - 0.5 + gap, 0.15, cz, 0xe5484d], [0, -1, -0.5 - gap, 0.15, cz, 0xe5484d],
      [2, 1, cx, 0.15, Z - 0.5 + gap, 0x3e63dd], [2, -1, cx, 0.15, -0.5 - gap, 0x3e63dd],
      [1, 1, cx, Y + 0.6, cz, 0x30a46c],
    ];
    for (const [axis, side, x, y, z, color] of spots) {
      const ball = new THREE.Mesh(new THREE.SphereGeometry(0.2, 20, 14),
        new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
      ball.renderOrder = 10; // stays visible behind blocks
      ball.position.set(x, y, z);
      ball.userData.stretch = { axis, side };
      this.handles.add(ball);
    }
  }

  private ray(e: PointerEvent): THREE.Ray {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    return this.raycaster.ray;
  }

  private handleAt(e: PointerEvent): THREE.Object3D | null {
    if (!this.stretch || this.multiArea || !this.handles.children.length) return null;
    this.ray(e);
    // generous hit radius: test a slightly bigger invisible sphere around each ball
    let best: THREE.Object3D | null = null;
    let bestD = Infinity;
    for (const h of this.handles.children) {
      const d = this.raycaster.ray.distanceSqToPoint(h.position);
      if (d < 0.35 * 0.35 && d < bestD) [best, bestD] = [h, d];
    }
    return best;
  }

  /** Distance along the drag axis (outwards) of the point on it closest to the pointer ray. */
  private along(e: PointerEvent): number | null {
    const d = this.drag!;
    const r = this.ray(e);
    const w = d.origin.clone().sub(r.origin);
    const b = d.dir.dot(r.direction);
    const denom = 1 - b * b;
    if (denom < 1e-4) return null; // looking straight down the axis
    return (b * r.direction.dot(w) - d.dir.dot(w)) / denom;
  }

  private startStretch(e: PointerEvent): boolean {
    const h = this.handleAt(e);
    if (!h) return false;
    const { axis, side } = h.userData.stretch;
    const dir = new THREE.Vector3();
    dir.setComponent(axis, side);
    this.drag = { axis, side, origin: h.position.clone(), dir, applied: 0, id: e.pointerId };
    const s = this.along(e);
    if (s !== null) this.drag.origin.addScaledVector(dir, s); // measure from where the ball was grabbed
    this.renderer.domElement.style.cursor = "grabbing";
    return true;
  }

  private moveStretch(e: PointerEvent): void {
    const d = this.drag!;
    const s = this.along(e);
    if (s === null || !this.stretch) return;
    const want = Math.round(s);
    if (want === d.applied) return;
    const done = this.stretch(d.axis, d.side, want - d.applied);
    if (!done) return;
    d.applied += done;
    if (d.side < 0) {
      // the grid grew or shrank at its low end, so everything moved +done along the axis:
      // move the camera and the drag line with it, so the grid seems to grow towards the pointer
      const shift = new THREE.Vector3();
      shift.setComponent(d.axis, done);
      this.camera.position.add(shift);
      this.controls.target.add(shift);
      d.origin.add(shift);
    }
  }

  private pick(e: PointerEvent): [number, number, number] | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    for (const hit of this.raycaster.intersectObjects(this.world.children, true)) {
      let o: THREE.Object3D | null = hit.object;
      while (o && !o.userData.lever) o = o.parent;
      if (o) return o.userData.lever;
    }
    return null;
  }

  /** The grid cell under the pointer on horizontal layer ``layer``, if inside the grid. */
  private pickCell(e: PointerEvent, layer: number): [number, number, number] | null {
    if (!this.size) return null;
    const [X, Y, Z] = this.size.split("x").map(Number);
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    if (!this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -layer), hit)) return null;
    const x = Math.round(hit.x);
    const z = Math.round(hit.z);
    if (x < 0 || z < 0 || x >= X || z >= Z || layer >= Y) return null;
    return [x, layer, z];
  }

  /** Outline the cells of the layer that clicks place on (null hides it). */
  showLayer(layer: number | null): void {
    this.layerOutline.clear();
    if (layer === null || !this.size) return;
    const [X, , Z] = this.size.split("x").map(Number);
    const y = layer + 0.003;
    const pts: THREE.Vector3[] = [];
    for (let x = 0; x <= X; x++) pts.push(new THREE.Vector3(x - 0.5, y, -0.5), new THREE.Vector3(x - 0.5, y, Z - 0.5));
    for (let z = 0; z <= Z; z++) pts.push(new THREE.Vector3(-0.5, y, z - 0.5), new THREE.Vector3(X - 0.5, y, z - 0.5));
    const lines = new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0xffb347, transparent: true, opacity: 0.55 }));
    this.layerOutline.add(lines);
  }

  private resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setSize(w, h);
    this.composer?.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.paneSize = { w: 0, h: 0 }; // multi-view re-fits on its next frame
  }

  private frame(size: [number, number, number], moveCamera = true): void {
    const [X, Y, Z] = size;
    this.floor.clear();
    const grid = new THREE.GridHelper(Math.max(X, Z) + 2, Math.max(X, Z) + 2, 0x3a3f4a, 0x262a32);
    grid.position.set((X - 1) / 2, 0, (Z - 1) / 2);
    this.floor.add(grid);
    const floorTex = T.floorTexture().clone();
    floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping;
    floorTex.repeat.set(X, Z);
    floorTex.needsUpdate = true;
    const plate = new THREE.Mesh(new THREE.PlaneGeometry(X, Z), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 1 }));
    plate.rotation.x = -Math.PI / 2;
    plate.position.set((X - 1) / 2, -0.001, (Z - 1) / 2);
    plate.receiveShadow = true;
    this.floor.add(plate);
    if (!moveCamera) return;
    const centre = new THREE.Vector3((X - 1) / 2, Y / 2, (Z - 1) / 2);
    const dist = Math.max(X, Y, Z) * 1.1 + 2;
    this.camera.position.set(centre.x + dist * 0.5, centre.y + dist * 0.8, centre.z + dist);
    this.controls.target.copy(centre);
  }

  /** Re-frame the floor and camera when the grid size changes. */
  private fit(size: [number, number, number]): void {
    const sizeKey = size.join("x");
    if (sizeKey !== this.size) {
      this.frame(size, !this.drag); // stretching keeps the camera where it is
      this.size = sizeKey;
      this.drawHandles();
    }
  }

  show(circuit: CircuitState): void {
    this.fit(circuit.size);
    // Small grids: rebuilding every tick is simpler than diffing and cheap enough.
    this.world.clear();
    const byPos = new Map(circuit.blocks.map((b) => [key(...b.pos), b]));
    const at: Lookup = (x, y, z) => byPos.get(key(x, y, z));
    for (const b of circuit.blocks) this.world.add(buildBlock(b, at));
  }
}
