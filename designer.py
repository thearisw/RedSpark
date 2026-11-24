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
from dataclasses import dataclass, field

import numpy as np

from .sim import DOWN, HORIZONTAL, Block, Circuit, DIRS

# (block, facing) choices a network can place. AIR clears a cell.
PLACEABLE: list[tuple[Block, int]] = (
    [(Block.AIR, 0), (Block.SOLID, 0), (Block.DUST, 0)]
    + [(Block.TORCH, d) for d in HORIZONTAL + [DOWN]]
    + [(Block.REPEATER, d) for d in HORIZONTAL]
)
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

    @property
    def rows_passed(self) -> int:
        return sum(self.passed)


def evaluate(circuit: Circuit, task: TruthTableTask, settle_ticks: int | None = None,
             frames: bool = False) -> Evaluation:
    """Run every truth-table row in order and compare each output lamp.

    A row whose circuit is still changing after ``settle_ticks`` fails.
    """
    c = circuit.copy()
    settle = settle_ticks if settle_ticks is not None else 8 * sum(c.size) + 40  # game ticks
    passed, got, shots, stable = [], [], [], True
    active_maps = []
    for row in task.rows:
        for p, value in zip(task.inputs.values(), row.inputs):
            c.set_lever(p, value)
        ok = c.settle(settle)
        stable &= ok
        lit = [bool(c.get(p) == Block.LAMP and c.state[p]) for p in task.outputs.values()]
        got.append(lit)
        passed.append(ok and all(e is None or e == g for e, g in zip(row.outputs, lit)))
        active_maps.append(active_cells(c))
        if frames:
            shots.append(c.to_dict())
    return Evaluation(passed, got, stable, shots, signal_reach(task, active_maps))


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

    def obs(self, c: Circuit, progress: float) -> np.ndarray:
        kinds = np.eye(N_KINDS, dtype=np.float32)[c.block.astype(np.int64)]       # (X,Y,Z,K)
        has_facing = (c.block == Block.TORCH) | (c.block == Block.REPEATER)
        facing = np.eye(6, dtype=np.float32)[c.facing.astype(np.int64)] * has_facing[..., None]
        parts = [np.moveaxis(kinds, -1, 0).ravel(), np.moveaxis(facing, -1, 0).ravel(),
                 self.role.ravel(), np.array([progress], dtype=np.float32)]
        return np.concatenate(parts)

    def mask(self, c: Circuit) -> np.ndarray:
        block, facing = c.block, c.facing
        free = ~self.fixed_mask
        solid = block == Block.SOLID
        cols = []
        for kind, d in PLACEABLE:
            if kind in (Block.TORCH, Block.REPEATER):
                same = (block == kind) & (facing == d)
            else:
                same = block == kind
            ok = free & ~same
            if kind == Block.TORCH:  # needs a solid block to attach to
                dx, dy, dz = DIRS[d]
                attach = np.zeros_like(solid)
                X, Y, Z = self.size
                src = solid[max(dx, 0):X + min(dx, 0), max(dy, 0):Y + min(dy, 0), max(dz, 0):Z + min(dz, 0)]
                attach[max(-dx, 0):X + min(-dx, 0), max(-dy, 0):Y + min(-dy, 0), max(-dz, 0):Z + min(-dz, 0)] = src
                ok &= attach
            cols.append(ok.ravel())
        return np.concatenate([np.stack(cols, axis=1).ravel(), [True]])

    def decode(self, action: int) -> tuple[Pos, Block, int]:
        cell, choice = divmod(int(action), len(PLACEABLE))
        pos = tuple(int(i) for i in np.unravel_index(cell, self.size))
        return pos, *PLACEABLE[choice]


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
@dataclass
class Individual:
    theta: np.ndarray
    fitness: float = float("-inf")
    circuit: Circuit | None = None
    result: Evaluation | None = None
    blocks: int = 0


@dataclass
class GenerationStats:
    gen: int
    best_rows: int
    mean_rows: float
    solved: int                 # individuals passing every row
    best: Individual
    new_best: bool

    def to_dict(self) -> dict:
        return {"gen": self.gen, "best": self.best_rows, "mean": round(self.mean_rows, 3),
                "solved": self.solved, "new_best": self.new_best}


class Trainer:
    """Genetic algorithm over Policy weights. Call ``generation()`` repeatedly."""

    def __init__(self, task: TruthTableTask, max_blocks: int = 12, population: int = 32,
                 elites: int = 6, sigma: float = 0.08, seed: int | None = None):
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

    def build(self, theta: np.ndarray) -> Circuit:
        b, c = self.builder, self.task.blank_circuit()
        weights = self.policy.unpack(theta)
        for step in range(self.max_blocks):
            mask = b.mask(c)
            logits = self.policy.logits(weights, b.obs(c, step / self.max_blocks))
            logits[~mask] = -np.inf
            action = int(np.argmax(logits))
            if action == b.n_actions - 1:
                break
            pos, kind, facing = b.decode(action)
            c.place(pos, kind, facing=facing)
        return c

    def fitness(self, c: Circuit) -> tuple[float, Evaluation, int]:
        res = evaluate(c, self.task)
        blocks = c.count_blocks() - len(self.task.fixed)
        bits = [(e == g) for row, got in zip(self.task.rows, res.got)
                for e, g in zip(row.outputs, got) if e is not None]
        bit_frac = sum(bits) / max(1, len(bits))
        # An output that reacts to the inputs at all is a step away from "always off".
        reach = float(np.mean(res.reach))
        score = res.rows_passed + 0.5 * bit_frac + 0.3 * reach - 0.005 * blocks
        if not res.stable:
            score -= 0.5
        return score, res, blocks

    def score(self, ind: Individual) -> None:
        ind.circuit = self.build(ind.theta)
        ind.fitness, ind.result, ind.blocks = self.fitness(ind.circuit)

    def generation(self) -> GenerationStats:
        for ind in self.pop:
            if ind.result is None:
                self.score(ind)
        self.pop.sort(key=lambda i: i.fitness, reverse=True)
        top = self.pop[0]
        new_best = self.best is None or top.fitness > self.best.fitness
        if new_best:
            self.best, self.best_gen = top, self.gen
        rows = [i.result.rows_passed for i in self.pop]
        stats = GenerationStats(self.gen, top.result.rows_passed, float(np.mean(rows)),
                                sum(r == self.n_rows for r in rows), top, new_best)
        # Next generation: keep the elites, fill the rest with mutated copies of them.
        parents = self.pop[:self.elites]
        children = []
        for _ in range(self.population - self.elites):
            p = parents[int(self.rng.integers(len(parents)))]
            noise = self.rng.standard_normal(p.theta.shape).astype(np.float32)
            # Mostly small tweaks, sometimes a big jump to escape a plateau.
            scale = self.sigma * float(self.rng.choice([0.25, 1.0, 4.0]))
            children.append(Individual(p.theta + scale * noise))
        self.pop = parents + children
        self.gen += 1
        return stats
