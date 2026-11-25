"""Truth-table training: evolve a small neural network that builds a circuit.

The user places input levers and output lamps anywhere on a grid and fills in a
truth table (each output can be on, off or "don't care" per row). A population
of small NumPy networks then competes, one generation at a time. Each network
looks at the grid and picks the next block to place (or stops), until it has
used its block budget. The circuit it ends up with is scored against the truth
table, and the best networks are copied with small random changes into the next
generation (a simple genetic algorithm over the network weights).

This needs only NumPy, so it runs inside the app without PyTorch. It uses only
the public Circuit interface (place, set_lever, settle, powered, to_dict).
"""

from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context, shared_memory

import numpy as np

from . import sim
from .sim import DOWN, HORIZONTAL, OPPOSITE, UP, Block, Circuit, DIRS, add

# Which blocks a torch can hang on (wall), and what each part can stand on, as the
# simulator defines them. The mask below mirrors Circuit.can_place on whole grids.
WALL_SUPPORT = tuple(int(k) for k in sim.SUPPORTS_WALL_TORCH)
BELOW_SUPPORT = {kind: tuple(int(k) for k in blocks) for kind, blocks in sim.SUPPORTS_BELOW.items()}
STANDING_SUPPORT = BELOW_SUPPORT[Block.TORCH]
FLOOR_SUPPORT = bool(sim.FLOOR_IS_SUPPORT)

# (block, facing) choices a network can place. AIR clears a cell.
PLACEABLE: list[tuple[Block, int]] = (
    [(Block.AIR, 0), (Block.SOLID, 0), (Block.DUST, 0)]
    + [(Block.TORCH, d) for d in HORIZONTAL + [DOWN]]
    + [(Block.REPEATER, d) for d in HORIZONTAL]
    + [(Block.COMPARATOR, d) for d in HORIZONTAL]     # compare mode
    + [(Block.COMPARATOR, d) for d in HORIZONTAL]     # subtract mode (see PLACEABLE_SUBTRACT)
)
# Which choices put a comparator in subtract mode.
PLACEABLE_SUBTRACT: list[bool] = [False] * (len(PLACEABLE) - 4) + [True] * 4
FACING_KINDS = (Block.TORCH, Block.REPEATER, Block.COMPARATOR)
N_KINDS = len(Block)
MAX_INPUTS = 4
MAX_OUTPUTS = 4
MAX_CELLS = 9 * 3 * 9

Pos = tuple[int, int, int]


# --------------------------------------------------------------------- task
@dataclass
class Row:
    inputs: tuple[bool, ...]
    outputs: tuple[bool | None, ...]  # None = don't care


@dataclass
class TruthTableTask:
    size: tuple[int, int, int]
    inputs: dict[str, Pos]
    outputs: dict[str, Pos]
    rows: list[Row]

    @classmethod
    def from_dict(cls, data: dict) -> "TruthTableTask":
        size = tuple(int(v) for v in data["size"])
        if len(size) != 3 or not all(1 <= s <= 9 for s in size) or size[1] > 3:
            raise ValueError("grid size must be up to 9 x 3 x 9")
        ins = {str(d["label"]): tuple(int(v) for v in d["pos"]) for d in data["inputs"]}
        outs = {str(d["label"]): tuple(int(v) for v in d["pos"]) for d in data["outputs"]}
        if not 1 <= len(ins) <= MAX_INPUTS:
            raise ValueError(f"place 1 to {MAX_INPUTS} inputs")
        if not 1 <= len(outs) <= MAX_OUTPUTS:
            raise ValueError(f"place 1 to {MAX_OUTPUTS} outputs")
        cells = list(ins.values()) + list(outs.values())
        if len(set(cells)) != len(cells) or len(set(ins) | set(outs)) != len(cells):
            raise ValueError("inputs and outputs need their own cells and names")
        for p in cells:
            if not all(0 <= p[i] < size[i] for i in range(3)):
                raise ValueError(f"{list(p)} is outside the grid")
        rows = []
        for r in data["rows"]:
            i = tuple(bool(v) for v in r["in"])
            o = tuple(None if v is None else bool(v) for v in r["out"])
            if len(i) != len(ins) or len(o) != len(outs):
                raise ValueError("each truth-table row needs one value per input and output")
            rows.append(Row(i, o))
        if not rows:
            raise ValueError("the truth table is empty")
        if all(v is None for r in rows for v in r.outputs):
            raise ValueError("set at least one output in the truth table")
        return cls(size, ins, outs, rows)

    @staticmethod
    def all_rows(n_inputs: int) -> list[tuple[bool, ...]]:
        return list(itertools.product([False, True], repeat=n_inputs))

    def blank_circuit(self) -> Circuit:
        c = Circuit(self.size)
        for name, p in self.inputs.items():
            c.place(p, Block.LEVER, label=name)
        for name, p in self.outputs.items():
            c.place(p, Block.LAMP, label=name)
        return c

    @property
    def fixed(self) -> set[Pos]:
        return set(self.inputs.values()) | set(self.outputs.values())


@dataclass
class Evaluation:
    passed: list[bool]                    # per row: every output matched
    got: list[list[bool]]                 # per row, per output: lamp lit
    stable: bool
    frames: list[dict] = field(default_factory=list)  # settled state per row (if asked)
    reach: list[float] = field(default_factory=list)  # per output: how close input-driven signal gets (0..1)
    match: list[float] = field(default_factory=list)  # per output: how close the *right* signal gets (0..1)
    delays: list[int] = field(default_factory=list)   # per row: game ticks until the outputs stopped changing

    @property
    def rows_passed(self) -> int:
        return sum(self.passed)

    @property
    def delay(self) -> int:
        """Activation time: the slowest row's game ticks from flipping the inputs to the final outputs."""
        return max(self.delays, default=0)


def evaluate(circuit: Circuit, task: TruthTableTask, settle_ticks: int | None = None,
             frames: bool = False) -> Evaluation:
    """Run every truth-table row in order and compare each output lamp.

    A row whose circuit is still changing after ``settle_ticks`` fails.
    """
    c = circuit.copy()
    settle = settle_ticks if settle_ticks is not None else 8 * sum(c.size) + 40  # game ticks
    passed, got, shots, stable = [], [], [], True
    active_maps = []
    outs = list(task.outputs.values())
    lamps = lambda: tuple(bool(c.state[p]) for p in outs)  # noqa: E731
    delays = []
    for row in task.rows:
        for p, value in zip(task.inputs.values(), row.inputs):
            c.set_lever(p, value)
        # Step like Circuit.settle, but note the last tick on which any output changed:
        # that is how long this row takes to show its answer (0 = instantly).
        before, last, ok = lamps(), 0, False
        for tick in range(1, settle + 1):
            if c.is_settled():
                ok = True
                break
            c.step()
            now = lamps()
            if now != before:
                before, last = now, tick
        else:
            ok = c.is_settled()
        delays.append(last)
        stable &= ok
        lit = [bool(c.get(p) == Block.LAMP and c.state[p]) for p in task.outputs.values()]
        got.append(lit)
        passed.append(ok and all(e is None or e == g for e, g in zip(row.outputs, lit)))
        active_maps.append(active_cells(c))
        if frames:
            shots.append(c.to_dict())
    return Evaluation(passed, got, stable, shots, signal_reach(task, active_maps),
                      matching_reach(task, active_maps), delays)


def active_cells(c: Circuit) -> np.ndarray:
    """Cells that are on: levers, lit torches, active repeaters, lamps, dust and powered blocks."""
    active = c.state | (c.dust > 0)
    for idx in np.argwhere(c.block == Block.SOLID):
        p = tuple(int(i) for i in idx)
        active[p] = c.powered(p)
    return active


def signal_reach(task: TruthTableTask, active_maps: list[np.ndarray]) -> list[float]:
    """For each output, 1 - (distance from the nearest input-driven cell) / (grid span).

    A cell is input-driven if it is on in some rows and off in others. This gives
    training a hint when a signal path grows towards an output but has not
    reached it yet.
    """
    stack = np.stack(active_maps)
    driven = np.argwhere(stack.any(axis=0) & ~stack.all(axis=0))
    span = float(sum(task.size))
    out = []
    for p in task.outputs.values():
        if len(driven) == 0:
            out.append(0.0)
            continue
        d = np.abs(driven - np.array(p)).sum(axis=1).min()
        out.append(max(0.0, 1.0 - d / span))
    return out


def matching_reach(task: TruthTableTask, active_maps: list[np.ndarray]) -> list[float]:
    """For each output, how close a cell carrying (nearly) the *right* signal gets (0..1).

    A cell's signal is its on/off pattern over the truth-table rows (ignoring
    don't-care rows). A cell scores ``proximity * agreement``, where proximity is
    1 - distance / span and agreement is the fraction of rows whose pattern
    matches the output's, squared so an exact match counts much more. The
    opposite pattern also counts (a little less), because one torch turns it into
    the right one: for AND, a NAND signal next to the lamp is one inverter away
    from solving. For "OUT = B" the B lever itself already matches exactly, so a
    dust line growing from B towards the lamp scores better at every step, even
    while a wrong path (say from A) already reaches the lamp.
    """
    stack = np.stack(active_maps)
    span = float(sum(task.size))
    coords = np.indices(task.size)
    out = []
    for k, p in enumerate(task.outputs.values()):
        rows = [i for i, r in enumerate(task.rows) if r.outputs[k] is not None]
        want = np.array([task.rows[i].outputs[k] for i in rows], dtype=bool)
        if not rows or want.all() or not want.any():
            out.append(0.0)  # a constant output has no signal to follow
            continue
        pats = stack[rows]
        varies = pats.any(axis=0) & ~pats.all(axis=0)
        same = (pats == want[:, None, None, None]).mean(axis=0)
        agree = np.maximum(same ** 2, 0.85 * (1.0 - same) ** 2)
        dist = sum(np.abs(coords[i] - p[i]) for i in range(3))
        score = np.where(varies, agree * np.clip(1.0 - dist / span, 0.0, 1.0), 0.0)
        # Best cell, plus half the best cell carrying a *different* signal: a second
        # ingredient (say NOT B, when NOT A already reaches the lamp of a NAND) is
        # progress even before it is wired in.
        cells = np.argwhere(score > 0)
        best: dict[bytes, float] = {}
        for c in cells:
            col = pats[(slice(None), *c)]
            key = min(col.tobytes(), (~col).tobytes())  # a signal and its opposite are one ingredient
            v = float(score[tuple(c)])
            if v > best.get(key, 0.0):
                best[key] = v
        top = sorted(best.values(), reverse=True)[:2] + [0.0, 0.0]
        out.append((top[0] + 0.5 * top[1]) / 1.5)
    return out


def score_circuit(task: TruthTableTask, c: Circuit) -> tuple[tuple, Evaluation, int, int]:
    """A ranking key, compared element by element (bigger is better).

    Correctness always comes first: rows passed. Among circuits that pass
    every row, fewer blocks wins, then the smaller bounding box, then the
    faster one (lower activation time, in game ticks). Circuits
    that are not correct yet are ranked by hints instead (output bits right,
    how close an input-driven signal gets to the outputs), so a small but
    wrong circuit never beats a correct one.
    """
    res = evaluate(c, task)
    blocks = c.count_blocks() - len(task.fixed)
    volume = int(np.prod(bounding_dims(c)))
    if res.rows_passed == len(task.rows):
        return (res.rows_passed, 1, -blocks, -volume, -res.delay), res, blocks, volume
    bits = [(e == g) for row, got in zip(task.rows, res.got)
            for e, g in zip(row.outputs, got) if e is not None]
    bit_frac = sum(bits) / max(1, len(bits))
    # An output that reacts to the inputs at all is a step away from "always off".
    reach = float(np.mean(res.reach))
    match = float(np.mean(res.match))
    hint = 0.4 * bit_frac + 0.2 * reach + 0.4 * match - (0.5 if not res.stable else 0.0)
    return (res.rows_passed, 0, hint, -blocks, 0), res, blocks, volume


# ------------------------------------------------------------------- layout
SUPPORT_KINDS = frozenset(int(k) for k in sim.ALL_SUPPORTS) if hasattr(sim, "ALL_SUPPORTS") \
    else frozenset(WALL_SUPPORT)


class Layout:
    """Just the blocks and facings of a circuit being built, without simulating it.

    It follows the simulator's support rules (parts break off when what holds
    them goes away), which is all that building needs. ``to_circuit`` makes the
    real Circuit once building is done.
    """

    def __init__(self, task: TruthTableTask):
        self.task = task
        self.size = task.size
        self.block = np.zeros(self.size, dtype=np.int8)
        self.facing = np.zeros(self.size, dtype=np.int8)
        self.subtract = np.zeros(self.size, dtype=bool)  # comparators in subtract mode
        for p in task.inputs.values():
            self.block[p], self.facing[p] = Block.LEVER, DOWN
        for p in task.outputs.values():
            self.block[p] = Block.LAMP

    def copy(self) -> "Layout":
        new = Layout.__new__(Layout)
        new.task, new.size = self.task, self.size
        new.block, new.facing, new.subtract = self.block.copy(), self.facing.copy(), self.subtract.copy()
        return new

    def get(self, p: Pos) -> int:
        if all(0 <= p[i] < self.size[i] for i in range(3)):
            return int(self.block[p])
        return int(Block.AIR)

    def can_place(self, p: Pos, kind: int, facing: int) -> bool:
        """The simulator's support rules for putting ``kind`` at ``p``."""
        if kind == Block.TORCH and facing != DOWN:
            return self.get(add(p, facing)) in WALL_SUPPORT
        if kind in BELOW_SUPPORT:
            return FLOOR_SUPPORT if p[1] == 0 else self.get((p[0], p[1] - 1, p[2])) in BELOW_SUPPORT[kind]
        return True

    def supported(self, p: Pos) -> bool:
        kind = self.get(p)
        if kind == Block.TORCH and int(self.facing[p]) != DOWN:
            q = add(p, int(self.facing[p]))
            return self.get(q) in WALL_SUPPORT
        if kind in BELOW_SUPPORT:
            if p[1] == 0:
                return FLOOR_SUPPORT
            return self.get((p[0], p[1] - 1, p[2])) in BELOW_SUPPORT[kind]
        return True

    def place(self, p: Pos, kind: Block, facing: int, subtract: bool = False) -> None:
        self.block[p], self.facing[p] = int(kind), facing
        self.subtract[p] = bool(subtract) and int(kind) == Block.COMPARATOR
        if int(kind) in SUPPORT_KINDS:
            return
        # p may have held parts that now break off (and they may have held others).
        for d in range(6):
            q = add(p, d)
            k = self.get(q)
            if k == Block.AIR or q in self.task.fixed:
                continue
            rests = (k == Block.TORCH and add(q, int(self.facing[q])) == p) or (k in BELOW_SUPPORT and d == UP)
            if rests and not self.supported(q):
                self.place(q, Block.AIR, 0)

    def to_circuit(self) -> Circuit | None:
        """The simulated circuit for this layout, or None if the simulator disagrees."""
        c = self.task.blank_circuit()
        fixed = self.task.fixed
        cells = [tuple(int(v) for v in i) for i in np.argwhere(self.block != Block.AIR)]
        cells = [p for p in cells if p not in fixed]
        # Blocks that hold others first, then the parts, bottom layer up.
        cells.sort(key=lambda p: (int(self.block[p]) not in SUPPORT_KINDS, p[1]))
        try:
            for p in cells:
                kind = Block(int(self.block[p]))
                extra = {"subtract": bool(self.subtract[p])} if kind == Block.COMPARATOR else {}
                c.place(p, kind, facing=int(self.facing[p]), **extra)
        except ValueError:
            return None
        if hasattr(c, "subtract") and not np.array_equal(c.subtract & (c.block == Block.COMPARATOR),
                                                         self.subtract & (self.block == Block.COMPARATOR)):
            return None
        if not (np.array_equal(c.block, self.block) and np.array_equal(
                np.where(self.block == Block.AIR, 0, c.facing), np.where(self.block == Block.AIR, 0, self.facing))):
            return None
        return c


# ------------------------------------------------------------------ builder
class Builder:
    """Turns a network's choices into block placements, with legal-move masks."""

    def __init__(self, task: TruthTableTask):
        self.task = task
        self.size = task.size
        self.n_cells = int(np.prod(self.size))
        self.n_actions = self.n_cells * len(PLACEABLE) + 1  # last action: stop
        self.fixed_mask = np.zeros(self.size, dtype=bool)
        for p in task.fixed:
            self.fixed_mask[p] = True
        # Which input lever / output lamp each fixed cell is, so the network knows.
        self.role = np.zeros((MAX_INPUTS + MAX_OUTPUTS, *self.size), dtype=np.float32)
        for k, p in enumerate(task.inputs.values()):
            self.role[(k, *p)] = 1.0
        for k, p in enumerate(task.outputs.values()):
            self.role[(MAX_INPUTS + k, *p)] = 1.0
        self.obs_size = (N_KINDS + 6 + MAX_INPUTS + MAX_OUTPUTS) * self.n_cells + 1
        self._prepare_tables()

    def _prepare_tables(self) -> None:
        """Lookup tables so obs() and mask() are a handful of array operations per step."""
        n_types = max(int(k) for k in Block) + 1
        self._kinds = np.arange(N_KINDS)[:, None]
        self._dirs = np.arange(6)[:, None]
        self._has_facing = np.zeros(n_types, dtype=bool)
        self._has_facing[list(FACING_KINDS)] = True
        self._choice_kind = np.array([int(k) for k, _ in PLACEABLE])
        self._choice_facing = np.array([d for _, d in PLACEABLE])
        self._choice_uses_facing = np.isin(self._choice_kind, list(FACING_KINDS))
        self._choice_subtract = np.array(PLACEABLE_SUBTRACT)
        self._comparator_cols = np.flatnonzero(self._choice_kind == Block.COMPARATOR)
        self._free = ~self.fixed_mask.ravel()
        self._wall_lut = np.zeros(n_types, dtype=bool)
        self._wall_lut[list(WALL_SUPPORT)] = True
        # For each kind that needs a block beneath: its lookup table and which choices it covers.
        self._below = []
        for kind, blocks in BELOW_SUPPORT.items():
            cols = [i for i, (k, d) in enumerate(PLACEABLE) if k == kind and not (k == Block.TORCH and d != DOWN)]
            if cols:
                lut = np.zeros(n_types, dtype=bool)
                lut[list(blocks)] = True
                self._below.append((lut, cols))
        self._wall_cols = [(i, d) for i, (k, d) in enumerate(PLACEABLE) if k == Block.TORCH and d != DOWN]
        self._obs_buf = np.zeros(self.obs_size, dtype=np.float32)
        n = self.n_cells
        self._obs_buf[(N_KINDS + 6) * n:-1] = self.role.ravel()

    def obs(self, c: Circuit, progress: float) -> np.ndarray:
        n, out = self.n_cells, self._obs_buf
        block = c.block.ravel()
        out[:N_KINDS * n] = (block[None, :] == self._kinds).ravel()
        facing = (c.facing.ravel()[None, :] == self._dirs) & self._has_facing[block][None, :]
        out[N_KINDS * n:(N_KINDS + 6) * n] = facing.ravel()
        out[-1] = progress
        return out

    def mask(self, c: Circuit) -> np.ndarray:
        """Legal moves, mirroring Circuit.can_place over the whole grid at once."""
        X, Y, Z = self.size
        block3 = c.block
        block, facing = block3.ravel(), c.facing.ravel()
        same = (block[:, None] == self._choice_kind[None, :]) & (
            ~self._choice_uses_facing[None, :] | (facing[:, None] == self._choice_facing[None, :]))
        if len(self._comparator_cols):
            sub = getattr(c, "subtract", None)
            if sub is not None:
                cols = self._comparator_cols
                same[:, cols] &= sub.ravel()[:, None] == self._choice_subtract[None, cols]
        ok = self._free[:, None] & ~same
        if self._below:
            below = np.zeros(self.size, dtype=block3.dtype)
            below[:, 1:, :] = block3[:, :-1, :]
            for lut, cols in self._below:
                sup = lut[below]
                sup[:, 0, :] = FLOOR_SUPPORT  # the grid floor holds things on the bottom layer
                ok[:, cols] &= sup.ravel()[:, None]
        wall_ok = self._wall_lut[block3]
        for col, d in self._wall_cols:  # wall torch: needs a block beside it
            dx, dy, dz = DIRS[d]
            attach = np.zeros_like(wall_ok)
            attach[max(-dx, 0):X + min(-dx, 0), max(-dy, 0):Y + min(-dy, 0), max(-dz, 0):Z + min(-dz, 0)] = \
                wall_ok[max(dx, 0):X + min(dx, 0), max(dy, 0):Y + min(dy, 0), max(dz, 0):Z + min(dz, 0)]
            ok[:, col] &= attach.ravel()
        return np.concatenate([ok.ravel(), [True]])

    def decode(self, action: int) -> tuple[Pos, Block, int]:
        cell, choice = divmod(int(action), len(PLACEABLE))
        pos = tuple(int(i) for i in np.unravel_index(cell, self.size))
        return pos, *PLACEABLE[choice]

    def decode_full(self, action: int) -> tuple[Pos, Block, int, bool]:
        """Like decode, plus whether a comparator goes in subtract mode."""
        return (*self.decode(action), PLACEABLE_SUBTRACT[int(action) % len(PLACEABLE)])


# ------------------------------------------------------------------ network
class Policy:
    """One-hidden-layer MLP stored as a flat weight vector."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 32):
        self.shapes = [(n_in, hidden), (hidden,), (hidden, n_out), (n_out,)]
        self.size = sum(int(np.prod(s)) for s in self.shapes)

    def init(self, rng: np.random.Generator) -> np.ndarray:
        parts = []
        for s in self.shapes:
            fan_in = s[0] if len(s) == 2 else 1
            parts.append((rng.standard_normal(s) / np.sqrt(fan_in) if len(s) == 2
                          else np.zeros(s)).ravel())
        return np.concatenate(parts).astype(np.float32)

    def unpack(self, theta: np.ndarray) -> list[np.ndarray]:
        out, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(theta[i:i + n].reshape(s))
            i += n
        return out

    def logits(self, weights: list[np.ndarray], x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = weights
        return np.tanh(x @ w1 + b1) @ w2 + b2


# ---------------------------------------------------------------- training
def layout(c) -> bytes:
    """A circuit's (or Layout's) blocks and facings, for spotting identical designs.

    Facings only count for blocks that have one, so the key is the same either way.
    """
    has = np.isin(c.block, [*FACING_KINDS, Block.LEVER])
    sub = getattr(c, "subtract", None)
    sub = (sub & (c.block == Block.COMPARATOR)) if sub is not None else np.zeros(c.block.shape, bool)
    return c.block.tobytes() + np.where(has, c.facing, 0).astype(np.int8).tobytes() + sub.tobytes()


def bounding_dims(c: Circuit) -> list[int]:
    """Width, height and depth of the box around every block in the circuit."""
    idx = np.argwhere(c.block != Block.AIR)
    if len(idx) == 0:
        return [0, 0, 0]
    return [int(v) for v in idx.max(axis=0) - idx.min(axis=0) + 1]


@dataclass
class Individual:
    theta: np.ndarray | None    # None until a child is made from parent + noise
    fitness: tuple = ()
    circuit: Circuit | None = None
    result: Evaluation | None = None
    blocks: int = 0
    volume: int = 0             # bounding box of every block, inputs and outputs included
    parent: np.ndarray | None = None
    lay: "Layout | None" = None  # the layout itself (layout evolution)
    scale: float = 0.0          # mutation size for a child of `parent`
    seed: int = 0               # seeds that mutation's noise

    @property
    def dims(self) -> list[int]:
        return bounding_dims(self.circuit) if self.circuit is not None else [0, 0, 0]


@dataclass
class GenerationStats:
    gen: int
    best_rows: int
    mean_rows: float
    solved: int                 # individuals passing every row
    best: Individual
    new_best: bool
    newest: Individual          # best of the networks that were new this generation
    restarted: bool = False     # the population was stuck and has been replaced

    def to_dict(self) -> dict:
        return {"gen": self.gen, "best": self.best_rows, "mean": round(self.mean_rows, 3),
                "solved": self.solved, "new_best": self.new_best, "blocks": self.best.blocks,
                "restarted": self.restarted}


class Trainer:
    """Genetic algorithm over Policy weights. Call ``generation()`` repeatedly."""

    def __init__(self, task: TruthTableTask, max_blocks: int = 12, population: int = 32,
                 elites: int = 6, sigma: float = 0.08, seed: int | None = None, patience: int = 0,
                 workers: int | None = None):
        self.task = task
        self.builder = Builder(task)
        self.policy = Policy(self.builder.obs_size, self.builder.n_actions)
        self.max_blocks = int(max_blocks)
        self.population = int(population)
        self.elites = int(elites)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)
        self.pop = [Individual(self.policy.init(self.rng)) for _ in range(self.population)]
        self.gen = 0
        self.best: Individual | None = None
        self.best_gen = 0
        self.n_rows = len(task.rows)
        # If the population's best has not improved for `patience` generations, it is
        # stuck on a local optimum (say "OUT = A" when the task is "OUT = B"), so it
        # starts over from fresh random networks. The best circuit so far is kept.
        # 0 (the default) never starts over: progress is never thrown away.
        self.patience = int(patience)
        self.stale = 0
        self.pop_best: tuple = ()
        self.restarts = 0
        self._cache: dict[bytes, tuple] = {}  # layout -> (circuit, fitness, result, blocks, volume)
        # Scoring can run on several CPU cores. workers=None decides automatically:
        # after a few generations, if a generation takes long enough for the split
        # to pay off and the machine has the cores. workers=0 keeps it in-process.
        self.workers = workers
        self._pool: _Pool | None = None
        self._gen_times: list[float] = []

    def build(self, theta: np.ndarray) -> Circuit:
        """Let the network place blocks one by one, then return the finished circuit.

        The network only sees the layout, so building edits a plain layout (fast)
        and the real, simulated Circuit is made once at the end. If the simulator
        would end up with a different layout (a rule the layout model does not
        know), the slow path builds it directly in the simulator instead.
        """
        return self.to_circuit(self.build_layout(theta), theta)

    def build_layout(self, theta: np.ndarray) -> "Layout":
        b, lay = self.builder, Layout(self.task)
        weights = self.policy.unpack(theta)
        for step in range(self.max_blocks):
            mask = b.mask(lay)
            logits = self.policy.logits(weights, b.obs(lay, step / self.max_blocks))
            logits[~mask] = -np.inf
            action = int(np.argmax(logits))
            if action == b.n_actions - 1:
                break
            lay.place(*b.decode_full(action))
        return lay

    def to_circuit(self, lay: "Layout", theta: np.ndarray) -> Circuit:
        c = lay.to_circuit()
        if c is None:
            c = self._build_in_simulator(self.policy.unpack(theta))
        return c

    def _build_in_simulator(self, weights: list[np.ndarray]) -> Circuit:
        b, c = self.builder, self.task.blank_circuit()
        for step in range(self.max_blocks):
            mask = b.mask(c)
            logits = self.policy.logits(weights, b.obs(c, step / self.max_blocks))
            logits[~mask] = -np.inf
            action = int(np.argmax(logits))
            if action == b.n_actions - 1:
                break
            pos, kind, facing, subtract = b.decode_full(action)
            extra = {"subtract": subtract} if kind == Block.COMPARATOR else {}
            try:
                c.place(pos, kind, facing=facing, **extra)
            except ValueError:  # a move the simulator refuses: treat it as stopping
                break
        return c

    def fitness(self, c: Circuit) -> tuple[tuple, Evaluation, int, int]:
        return score_circuit(self.task, c)


    def score(self, ind: Individual) -> None:
        ind.circuit = self.build(ind.theta)
        ind.fitness, ind.result, ind.blocks, ind.volume = self.fitness(ind.circuit)

    def score_child(self, theta: np.ndarray | None, parent: np.ndarray | None, scale: float, seed: int,
                    seen: set[bytes]) -> tuple[np.ndarray, bytes, tuple]:
        """Make one network (parent + noise, or ``theta`` as given), build and score it.

        A small weight change often leaves the network's choices, and so the
        circuit, exactly as before. Evaluating a copy teaches nothing, so it
        pushes harder until the child builds something new (a few tries at most).
        """
        rng = np.random.default_rng(seed)
        if parent is not None:
            theta = parent + scale * rng.standard_normal(parent.shape).astype(np.float32)
        for attempt in range(4):
            lay = self.build_layout(theta)
            key = layout(lay)
            if key not in seen or parent is None or attempt == 3:
                break
            theta = parent + self.sigma * 2.0 ** (attempt + 1) * rng.standard_normal(parent.shape).astype(np.float32)
        cached = self._cache.get(key)
        if cached is None:  # simulating the truth table is the expensive part
            circuit = self.to_circuit(lay, theta)
            cached = (circuit, *self.fitness(circuit))
            self._cache[key] = cached
            if len(self._cache) > 20000:
                self._cache.pop(next(iter(self._cache)))  # forget the oldest
        return theta, key, cached

    def _maybe_start_pool(self) -> None:
        if self._pool is not None or self.workers == 0 or len(self._gen_times) < 3:
            return
        n = self.workers if self.workers else min(8, (os.cpu_count() or 1) - 1)
        avg = sum(self._gen_times[-3:]) / 3
        if n >= 2 and (self.workers or avg > 0.03):
            try:
                self._pool = _Pool(self, n)
            except Exception:  # no process pool here: keep going in-process
                self.workers = 0

    def close(self) -> None:
        """Stop the worker processes, if any."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def generation(self) -> GenerationStats:
        started = time.perf_counter()
        fresh = [ind for ind in self.pop if ind.result is None]
        seen = {layout(i.circuit) for i in self.pop if i.circuit is not None}
        results = None
        if self._pool is not None:
            try:
                results = self._pool.score(fresh, seen)
            except Exception:  # a worker died: carry on in-process
                self._pool.close()
                self._pool, self.workers = None, 0
        if results is None:
            results = []
            for ind in fresh:
                results.append(self.score_child(ind.theta, ind.parent, ind.scale, ind.seed, seen))
                seen.add(results[-1][1])
        for ind, (theta, key, cached) in zip(fresh, results):
            ind.theta = theta
            ind.circuit, ind.fitness, ind.result, ind.blocks, ind.volume = cached
        self._gen_times.append(time.perf_counter() - started)
        self._maybe_start_pool()
        self.pop.sort(key=lambda i: i.fitness, reverse=True)
        top = self.pop[0]
        newest = max(fresh, key=lambda i: i.fitness) if fresh else top
        if top.fitness > self.pop_best:
            self.pop_best, self.stale = top.fitness, 0
        else:
            self.stale += 1
        new_best = self.best is None or top.fitness > self.best.fitness
        if new_best:
            self.best, self.best_gen = top, self.gen
        rows = [i.result.rows_passed for i in self.pop]
        stats = GenerationStats(self.gen, top.result.rows_passed, float(np.mean(rows)),
                                sum(r == self.n_rows for r in rows), top, new_best, newest)
        self.gen += 1
        solved = top.result.rows_passed == self.n_rows
        # Once solved, give shrinking the circuit more time before starting over.
        if self.patience and self.stale >= self.patience * (3 if solved else 1):
            self.pop = [Individual(self.policy.init(self.rng)) for _ in range(self.population)]
            self.pop_best, self.stale = (), 0
            self.restarts += 1
            stats.restarted = True
            return stats
        # Next generation: keep the elites, fill the rest with mutated copies of them.
        parents = self.pop[:self.elites]
        children = []
        for _ in range(self.population - self.elites):
            p = parents[int(self.rng.integers(len(parents)))]
            # Mostly small tweaks, sometimes a big jump to escape a plateau. The noise
            # itself is drawn from `seed` when the child is scored (maybe in a worker).
            scale = self.sigma * float(self.rng.choice([0.25, 1.0, 4.0]))
            children.append(Individual(None, parent=p.theta, scale=scale,
                                       seed=int(self.rng.integers(2 ** 63))))
        self.pop = parents + children
        return stats


# ------------------------------------------------------------- worker pool
_worker: dict = {}


def _worker_init(task: TruthTableTask, max_blocks: int, sigma: float, n_params: int,
                 parents_name: str, children_name: str, rows: tuple[int, int]) -> None:
    trainer = Trainer(task, max_blocks=max_blocks, population=1, elites=1, sigma=sigma, workers=0)
    p_shm = shared_memory.SharedMemory(name=parents_name)
    c_shm = shared_memory.SharedMemory(name=children_name)
    _worker.update(trainer=trainer, shms=(p_shm, c_shm),
                   parents=np.ndarray((rows[0], n_params), np.float32, buffer=p_shm.buf),
                   children=np.ndarray((rows[1], n_params), np.float32, buffer=c_shm.buf))


def _worker_score(jobs: list[tuple[int, int, float, int]], seen: set[bytes]) -> list[tuple]:
    """Score children. Each job: (child row, parent row or -1, scale, seed).

    The weights travel through shared memory: parents are read from it and each
    finished child's weights are written back, so only small results are sent.
    """
    t, parents, children = _worker["trainer"], _worker["parents"], _worker["children"]
    out = []
    for row, prow, scale, seed in jobs:
        parent = parents[prow] if prow >= 0 else None
        theta, key, cached = t.score_child(children[row].copy(), parent, scale, seed, seen)
        children[row] = theta
        seen.add(key)
        out.append((row, key, cached))
    return out


class _Pool:
    """Worker processes that build and score a generation's new networks in parallel."""

    def __init__(self, trainer: Trainer, n: int):
        self.trainer, self.n = trainer, n
        size = trainer.policy.size
        self.rows = (trainer.population, trainer.population)
        self._shms = [shared_memory.SharedMemory(create=True, size=max(1, r * size * 4)) for r in self.rows]
        self.parents = np.ndarray((self.rows[0], size), np.float32, buffer=self._shms[0].buf)
        self.children = np.ndarray((self.rows[1], size), np.float32, buffer=self._shms[1].buf)
        self.executor = ProcessPoolExecutor(
            n, mp_context=get_context("spawn"), initializer=_worker_init,
            initargs=(trainer.task, trainer.max_blocks, trainer.sigma, size,
                      self._shms[0].name, self._shms[1].name, self.rows))

    def score(self, fresh: list[Individual], seen: set[bytes]) -> list[tuple]:
        parent_rows: dict[int, int] = {}
        jobs = []
        for row, ind in enumerate(fresh):
            if ind.parent is None:
                self.children[row] = ind.theta
                jobs.append((row, -1, 0.0, ind.seed))
            else:
                key = id(ind.parent)
                if key not in parent_rows:
                    parent_rows[key] = len(parent_rows)
                    self.parents[parent_rows[key]] = ind.parent
                jobs.append((row, parent_rows[key], ind.scale, ind.seed))
        chunks = [jobs[i::self.n] for i in range(self.n) if jobs[i::self.n]]
        results: dict[int, tuple] = {}
        for part in self.executor.map(_worker_score, chunks, [set(seen)] * len(chunks)):
            for row, key, cached in part:
                results[row] = (self.children[row].copy(), key, cached)
                self.trainer._cache.setdefault(key, cached)  # remember the workers' results too
        return [results[row] for row in range(len(fresh))]

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        for shm in self._shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass


# ======================================================== layout evolution
class LayoutTrainer:
    """Evolve circuit layouts directly. The default, and much faster than the network.

    Each individual *is* a layout. A child is a copy of a good parent with a few
    small edits: add a legal block (usually next to existing ones, so signal
    paths grow), remove one, swap one for a different block or facing, and
    sometimes it takes one part of the grid from a second parent (crossover).
    Every edit follows the simulator's placement rules. Ranking is the same as
    the network trainer's (score_circuit): rows correct first, with hints for
    getting closer, then fewer blocks, smaller size and faster activation.

    Same interface as Trainer: call generation(); read best, best_gen, gen.
    """

    def __init__(self, task: TruthTableTask, max_blocks: int = 12, population: int = 48,
                 elites: int = 8, seed: int | None = None, patience: int = 0,
                 workers: int | None = None):
        self.task = task
        self.builder = Builder(task)
        self.max_blocks = int(max_blocks)
        self.population = int(population)
        self.elites = int(elites)
        self.rng = np.random.default_rng(seed)
        self.n_rows = len(task.rows)
        self.gen = 0
        self.best: Individual | None = None
        self.best_gen = 0
        # Start over only when the population has not improved for `patience`
        # generations and nothing is solved yet. 0 (the default) never starts over.
        self.patience = int(patience)
        self.stale = 0
        self.pop_best: tuple = ()
        self.restarts = 0
        self.workers = workers
        self._pool: _LayoutPool | None = None
        self._gen_times: list[float] = []
        self._cache: dict[bytes, tuple] = {}
        self._fixed = self.builder.fixed_mask
        self.pop = self._random_population()

    # ------------------------------------------------------------ edits
    def _random_population(self) -> list[Individual]:
        out = []
        for _ in range(self.population):
            lay = Layout(self.task)
            for _ in range(int(self.rng.integers(1, max(2, self.max_blocks // 2) + 1))):
                self._add(lay)
            out.append(Individual(None, lay=lay))
        return out

    def _n_blocks(self, lay: Layout) -> int:
        return int(np.count_nonzero((lay.block != Block.AIR) & ~self._fixed))

    def _near(self, lay: Layout) -> np.ndarray:
        """Cells next to (or diagonal to) any existing block."""
        occ = lay.block != Block.AIR
        near = occ.copy()
        X, Y, Z = self.task.size
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    src = occ[max(dx, 0):X + min(dx, 0), max(dy, 0):Y + min(dy, 0), max(dz, 0):Z + min(dz, 0)]
                    near[max(-dx, 0):X + min(-dx, 0), max(-dy, 0):Y + min(-dy, 0), max(-dz, 0):Z + min(-dz, 0)] |= src
        return near & ~occ

    def _pick(self, lay: Layout, cells: np.ndarray | None) -> bool:
        """Place a random legal non-air choice, in one of ``cells`` (flat bool) if given."""
        n_choice = len(PLACEABLE)
        mask = self.builder.mask(lay)[:-1].reshape(-1, n_choice).copy()
        mask[:, 0] = False  # AIR is "remove", handled separately
        if cells is not None:
            mask &= cells[:, None]
        options = np.flatnonzero(mask)
        if len(options) == 0:
            return False
        a = int(options[int(self.rng.integers(len(options)))])
        lay.place(*self.builder.decode_full(a))
        return True

    def _add(self, lay: Layout) -> None:
        if self._n_blocks(lay) >= self.max_blocks:
            return self._change(lay)
        empty = (lay.block == Block.AIR).ravel()
        if self.rng.random() < 0.8 and self._pick(lay, self._near(lay).ravel()):
            return
        self._pick(lay, empty)

    def _remove(self, lay: Layout) -> None:
        cells = np.argwhere((lay.block != Block.AIR) & ~self._fixed)
        if len(cells):
            p = tuple(int(v) for v in cells[int(self.rng.integers(len(cells)))])
            lay.place(p, Block.AIR, 0)

    def _change(self, lay: Layout) -> None:
        """Swap one block for a different legal block or facing in the same cell."""
        cells = np.argwhere((lay.block != Block.AIR) & ~self._fixed)
        if not len(cells):
            return self._add(lay)
        p = tuple(int(v) for v in cells[int(self.rng.integers(len(cells)))])
        one = np.zeros(self.task.size, dtype=bool)
        one[p] = True
        self._pick(lay, one.ravel())

    def _crossover(self, a: Layout, b: Layout) -> Layout:
        """Left/right (or front/back) part from a, the rest from b, rebuilt legally."""
        axis = int(self.rng.choice([0, 2]))
        k = int(self.rng.integers(1, self.task.size[axis]))
        idx = np.indices(self.task.size)[axis]
        block = np.where(idx < k, a.block, b.block)
        facing = np.where(idx < k, a.facing, b.facing)
        sub = np.where(idx < k, a.subtract, b.subtract)
        child = Layout(self.task)
        cells = [tuple(int(v) for v in i) for i in np.argwhere((block != Block.AIR) & ~self._fixed)]
        cells.sort(key=lambda p: (int(block[p]) not in SUPPORT_KINDS, p[1]))
        for p in cells[:self.max_blocks]:
            if child.can_place(p, int(block[p]), int(facing[p])):
                child.place(p, Block(int(block[p])), int(facing[p]), bool(sub[p]))
        return child

    def _motif(self, lay: Layout) -> None:
        """Add a small known building block as one move, starting next to an existing block.

        Single random blocks rarely line up into something useful: an inverter on a
        one-layer grid needs dust, a block and a torch in a row. These pieces are
        added whole: a wire (1-4 dust), an inverter (block plus a torch on its far
        side, on a side, or on top), a repeater into a block, or a lone repeater.
        """
        cells = np.argwhere(lay.block != Block.AIR)
        q = tuple(int(v) for v in cells[int(self.rng.integers(len(cells)))])
        d = int(HORIZONTAL[int(self.rng.integers(4))])
        side = int(self.rng.choice([s for s in HORIZONTAL if s not in (d, OPPOSITE[d])]))
        p1 = add(q, d)
        kind = int(self.rng.integers(6))
        if kind == 0:     # wire
            plan, p = [], q
            for _ in range(int(self.rng.integers(1, 5))):
                p = add(p, d)
                plan.append((p, Block.DUST, 0))
        elif kind == 1:   # block, torch on its far side
            plan = [(p1, Block.SOLID, 0), (add(p1, d), Block.TORCH, OPPOSITE[d])]
        elif kind == 2:   # block, torch on one side
            plan = [(p1, Block.SOLID, 0), (add(p1, side), Block.TORCH, OPPOSITE[side])]
        elif kind == 3:   # block, torch on top
            plan = [(p1, Block.SOLID, 0), (add(p1, UP), Block.TORCH, DOWN)]
        elif kind == 4:   # repeater into a block
            plan = [(p1, Block.REPEATER, d), (add(p1, d), Block.SOLID, 0)]
        else:
            plan = [(p1, Block.REPEATER, d)]
        for p, block, facing in plan:
            if not all(0 <= p[i] < self.task.size[i] for i in range(3)) or p in self.task.fixed:
                return
            if self._n_blocks(lay) >= self.max_blocks and lay.get(p) == Block.AIR:
                return
            if not lay.can_place(p, int(block), facing):
                return
            lay.place(p, block, facing)

    def _mutate(self, lay: Layout) -> Layout:
        child = lay.copy()
        for _ in range(int(self.rng.geometric(0.55))):  # 1 edit usually, sometimes several
            r = self.rng.random()
            if r < 0.25:
                self._motif(child)
            elif r < 0.5:
                self._add(child)
            elif r < 0.72:
                self._remove(child)
            else:
                self._change(child)
        return child

    def _tournament(self) -> Individual:
        picks = self.rng.integers(len(self.pop), size=3)
        return self.pop[int(min(picks))]  # pop is sorted best first

    # ---------------------------------------------------------- scoring
    def _score_layout(self, lay: Layout) -> tuple:
        key = layout(lay)
        cached = self._cache.get(key)
        if cached is None:
            cached = score_layout(self.task, lay)
            self._cache[key] = cached
            if len(self._cache) > 50000:
                self._cache.pop(next(iter(self._cache)))
        return cached

    def _maybe_start_pool(self) -> None:
        if self._pool is not None or self.workers == 0 or len(self._gen_times) < 3:
            return
        n = self.workers if self.workers else min(8, (os.cpu_count() or 1) - 1)
        if n >= 2 and (self.workers or sum(self._gen_times[-3:]) / 3 > 0.03):
            try:
                self._pool = _LayoutPool(self.task, n)
            except Exception:
                self.workers = 0

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def generation(self) -> GenerationStats:
        started = time.perf_counter()
        fresh = [ind for ind in self.pop if ind.result is None]
        todo = [(i, layout(ind.lay)) for i, ind in enumerate(fresh)]
        missing = [(i, k) for i, k in todo if k not in self._cache]
        if self._pool is not None and missing:
            try:
                for (i, k), cached in zip(missing, self._pool.score([fresh[i].lay for i, _ in missing])):
                    self._cache[k] = cached
            except Exception:
                self._pool.close()
                self._pool, self.workers = None, 0
        for ind in fresh:
            ind.circuit, ind.fitness, ind.result, ind.blocks, ind.volume = self._score_layout(ind.lay)
        self.pop.sort(key=lambda i: i.fitness, reverse=True)
        top = self.pop[0]
        newest = max(fresh, key=lambda i: i.fitness) if fresh else top
        if top.fitness > self.pop_best:
            self.pop_best, self.stale = top.fitness, 0
        else:
            self.stale += 1
        new_best = self.best is None or top.fitness > self.best.fitness
        if new_best:
            self.best, self.best_gen = top, self.gen
        rows = [i.result.rows_passed for i in self.pop]
        stats = GenerationStats(self.gen, top.result.rows_passed, float(np.mean(rows)),
                                sum(r == self.n_rows for r in rows), top, new_best, newest)
        self.gen += 1
        self._gen_times.append(time.perf_counter() - started)
        self._maybe_start_pool()
        solved = top.result.rows_passed == self.n_rows
        if self.patience and not solved and self.stale >= self.patience:
            # Start completely fresh: the best design so far stays in self.best (and on
            # screen), but not in the breeding pool, where it would take over again.
            self.pop = self._random_population()
            self.pop_best, self.stale = (), 0
            self.restarts += 1
            stats.restarted = True
            return stats
        # Next generation: elites, then children of tournament winners (new designs only).
        seen = {layout(i.lay) for i in self.pop[:self.elites]}
        children = []
        while len(children) < self.population - self.elites:
            for _ in range(6):
                a = self._tournament()
                if self.rng.random() < 0.25:
                    child = self._mutate(self._crossover(a.lay, self._tournament().lay)) \
                        if self.rng.random() < 0.5 else self._crossover(a.lay, self._tournament().lay)
                else:
                    child = self._mutate(a.lay)
                key = layout(child)
                if key not in seen:
                    break
            seen.add(key)
            children.append(Individual(None, lay=child))
        self.pop = self.pop[:self.elites] + children
        return stats


def score_layout(task: TruthTableTask, lay: Layout) -> tuple:
    """(circuit, fitness key, evaluation, blocks, volume) for a layout."""
    c = lay.to_circuit()
    if c is None:  # the simulator would not build this exact layout: rank it last
        c = task.blank_circuit()
        key, res, blocks, volume = score_circuit(task, c)
        return (c, (-1,) + key[1:], res, blocks, volume)
    return (c, *score_circuit(task, c))


_layout_worker: dict = {}


def _layout_worker_init(task: TruthTableTask) -> None:
    _layout_worker["task"] = task


def _layout_worker_score(cells: list[tuple[bytes, bytes, bytes]]) -> list[tuple]:
    task = _layout_worker["task"]
    out = []
    for block, facing, sub in cells:
        lay = Layout(task)
        lay.block = np.frombuffer(block, np.int8).reshape(task.size).copy()
        lay.facing = np.frombuffer(facing, np.int8).reshape(task.size).copy()
        lay.subtract = np.frombuffer(sub, bool).reshape(task.size).copy()
        out.append(score_layout(task, lay))
    return out


class _LayoutPool:
    """Worker processes that simulate and score layouts in parallel (layouts are tiny to send)."""

    def __init__(self, task: TruthTableTask, n: int):
        self.n = n
        self.executor = ProcessPoolExecutor(n, mp_context=get_context("spawn"),
                                            initializer=_layout_worker_init, initargs=(task,))

    def score(self, lays: list[Layout]) -> list[tuple]:
        jobs = [(l.block.tobytes(), l.facing.tobytes(), l.subtract.tobytes()) for l in lays]
        chunks = [jobs[i::self.n] for i in range(self.n)]
        parts = list(self.executor.map(_layout_worker_score, [c for c in chunks if c]))
        out: list = [None] * len(lays)
        for i, part in enumerate(parts):
            for j, res in enumerate(part):
                out[i + j * self.n] = res
        return out

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
