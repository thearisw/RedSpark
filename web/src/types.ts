export type BlockType = "solid" | "dust" | "torch" | "repeater" | "lever" | "lamp" | "comparator";
export type Facing = "+x" | "-x" | "+z" | "-z" | "+y" | "-y";

export interface BlockState {
  pos: [number, number, number];
  type: BlockType;
  facing?: Facing;
  delay?: number;
  on?: boolean;
  label?: string;
  active: boolean;
  level?: number;
  /** dust: sides it points to, 1 = +x, 2 = -x, 4 = +z, 8 = -z */
  shape?: number;
  locked?: boolean;
  /** comparator */
  mode?: "compare" | "subtract";
  output?: number;
}

export interface CircuitState {
  size: [number, number, number];
  ticks: number;
  blocks: BlockState[];
}

export interface GenStats {
  gen: number;
  best: number; // rows passed by the best network of this generation
  mean: number; // average rows passed across the population
  solved: number; // networks that passed every row
  new_best: boolean;
  blocks: number; // blocks used by the best network of this generation
  restarted: boolean; // the population was stuck and started over
}

export type ShowMode = "success" | "best" | "all";

export type ServerMessage =
  | { type: "state"; name: string | null; playing: boolean; circuit: CircuitState | null }
  | { type: "error"; message: string }
  | {
      type: "train_progress";
      running: boolean;
      gen: number;
      rows_total: number;
      stats: GenStats[];
      best_ever: { gen: number; rows: number; blocks: number; dims: number[]; delay: number } | null;
    }
  | {
      type: "train_circuit";
      gen: number;
      kind: ShowMode | "best_ever";
      rows_passed: number;
      rows_total: number;
      passed: boolean[];
      got: boolean[][];
      blocks: number;
      dims: number[]; // bounding box (x, y, z) of every block
      delay: number; // activation time: game ticks for the slowest row's outputs to settle
      frames: CircuitState[];
    }
  | { type: "saved"; name: string };
