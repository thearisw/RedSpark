import * as THREE from "three";

// BoxGeometry material order: +x, -x, +y, -y, +z, -z. The default camera looks from +z, so +z is FRONT.
const LABELS = ["RIGHT", "LEFT", "TOP", "BOTTOM", "FRONT", "BACK"];
const SIZE = 104; // CSS pixels

function faceTexture(label: string): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  g.fillStyle = "#2a2f3a";
  g.fillRect(0, 0, 128, 128);
  g.strokeStyle = "#4a505c";
  g.lineWidth = 6;
  g.strokeRect(3, 3, 122, 122);
  g.fillStyle = "#e8e8ea";
  g.font = "bold 22px system-ui, sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(label, 64, 66);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

/**
 * A small cube in the corner that turns with the main camera. Click a face, edge or corner to
 * snap the view to look from there; drag it to orbit.
 */
export class ViewCube {
  readonly canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  private cube: THREE.Mesh;
  private hover = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshBasicMaterial({ color: 0xffb347, transparent: true, opacity: 0.55, depthWrite: false }));
  private raycaster = new THREE.Raycaster();

  constructor(private onSnap: (dir: THREE.Vector3) => void, private onOrbit: (dx: number, dy: number) => void) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(SIZE, SIZE);
    this.canvas = this.renderer.domElement;
    this.canvas.id = "viewCube";
    this.canvas.title = "Click a face, edge or corner to look from there. Drag to orbit.";
    this.cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
      LABELS.map((l) => new THREE.MeshBasicMaterial({ map: faceTexture(l) })));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(this.cube.geometry),
      new THREE.LineBasicMaterial({ color: 0x8a909c }));
    this.hover.visible = false;
    this.scene.add(this.cube, edges, this.hover);

    let down: { x: number; y: number; moved: boolean } | null = null;
    const el = this.canvas;
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      down = { x: e.clientX, y: e.clientY, moved: false };
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", (e) => {
      if (!down) return this.setHover(this.regionAt(e));
      const dx = e.clientX - down.x;
      const dy = e.clientY - down.y;
      if (!down.moved && Math.hypot(dx, dy) < 3) return;
      down.moved = true;
      this.setHover(null);
      el.style.cursor = "grabbing";
      this.onOrbit(dx, dy);
      down.x = e.clientX;
      down.y = e.clientY;
    });
    el.addEventListener("pointerup", (e) => {
      const d = down;
      down = null;
      el.style.cursor = "";
      if (!d || d.moved) return;
      const dir = this.regionAt(e);
      if (dir) this.onSnap(dir.clone().normalize());
    });
    el.addEventListener("pointercancel", () => {
      down = null;
      el.style.cursor = "";
    });
    el.addEventListener("pointerleave", () => {
      if (!down) this.setHover(null);
    });
  }

  /**
   * Which face, edge or corner is under the pointer, as a direction with components -1, 0 or 1
   * (one non-zero: a face; two: an edge; three: a corner).
   */
  private regionAt(e: PointerEvent): THREE.Vector3 | null {
    const r = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = this.raycaster.intersectObject(this.cube)[0];
    if (!hit) return null;
    const p = hit.point;
    const s = (v: number) => (v > 0.3 ? 1 : v < -0.3 ? -1 : 0);
    return new THREE.Vector3(s(p.x), s(p.y), s(p.z));
  }

  private setHover(dir: THREE.Vector3 | null): void {
    this.hover.visible = !!dir;
    this.canvas.style.cursor = dir ? "pointer" : "";
    if (!dir) return;
    // a thin slab over the face's middle, a strip along an edge, or a little cube on a corner
    const part = (v: number) => (v === 0 ? 0.62 : 0.22);
    const at = (v: number) => v * (0.5 - 0.11 + 0.012);
    this.hover.scale.set(part(dir.x), part(dir.y), part(dir.z));
    this.hover.position.set(at(dir.x), at(dir.y), at(dir.z));
  }

  /** Turn the cube to match a camera looking from ``offset`` (camera position minus target). */
  render(offset: THREE.Vector3, up: THREE.Vector3): void {
    this.camera.position.copy(offset).normalize().multiplyScalar(4);
    this.camera.up.copy(up);
    this.camera.lookAt(0, 0, 0);
    this.renderer.render(this.scene, this.camera);
  }
}
