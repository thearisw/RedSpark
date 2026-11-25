"""Gymnasium environment: build a logic gate one block at a time.

The grid starts with the input levers (labelled A, B) on the left edge and the
output lamp (OUT) on the right edge. Each action places one block in one cell
(or clears it). The reward is the change in truth-table score after each
placement, so an episode's total is its final score minus the empty grid's
score. Solving the gate ends the episode with a bonus, minus a small cost per
block used. ``action_masks()`` lists the legal moves for MaskablePPO.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .sim import Block, Circuit, DOWN, HORIZONTAL
from .tasks import GATES, evaluate_gate

# (block, facing) choices the agent can place. AIR means "clear this cell".
PLACEABLE: list[tuple[Block, int]] = (
    [(Block.AIR, 0), (Block.SOLID, 0), (Block.DUST, 0)]
    + [(Block.TORCH, d) for d in HORIZONTAL + [DOWN]]
    + [(Block.REPEATER, d) for d in HORIZONTAL]
)
COMPARATOR_CHOICES: list[tuple[Block, int]] = [(Block.COMPARATOR, d) for d in HORIZONTAL]
N_BLOCK_TYPES = len(Block)


def default_size(gate: str) -> tuple[int, int, int]:
    return (5, 1, 3) if GATES[gate][0] == 1 else (5, 1, 5)


def io_layout(gate: str, size: tuple[int, int, int]) -> tuple[dict[str, tuple], tuple]:
    n_inputs = GATES[gate][0]
    X, _, Z = size
    if n_inputs == 1:
        inputs = {"A": (0, 0, Z // 2)}
    else:
        inputs = {"A": (0, 0, 0), "B": (0, 0, Z - 1)}
    return inputs, (X - 1, 0, Z // 2)


class RedstoneGateEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, gate: str = "NOT", size: tuple[int, int, int] | None = None,
                 max_steps: int = 16, block_cost: float = 0.01, solve_bonus: float = 1.0,
                 invalid_penalty: float = 0.02, render_mode: str | None = None,
                 allow_comparators: bool = False):
        if gate not in GATES:
            raise ValueError(f"unknown gate {gate}; choose from {sorted(GATES)}")
        self.gate = gate
        self.size = tuple(size or default_size(gate))
        self.max_steps = max_steps
        self.block_cost = block_cost
        self.solve_bonus = solve_bonus
        self.invalid_penalty = invalid_penalty
        self.render_mode = render_mode
        # Comparators are opt-in so existing runs keep the same action space.
        self.placeable = PLACEABLE + (COMPARATOR_CHOICES if allow_comparators else [])
        self.inputs, self.output = io_layout(gate, self.size)
        self.fixed = set(self.inputs.values()) | {self.output}

        self.n_cells = int(np.prod(self.size))
        self.action_space = spaces.Discrete(self.n_cells * len(self.placeable))
        # Channels: block one-hot, facing one-hot, fixed-cell mask.
        self.obs_shape = (N_BLOCK_TYPES + 6 + 1, *self.size)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(int(np.prod(self.obs_shape)),), dtype=np.float32)
        self.circuit: Circuit | None = None

    # ----------------------------------------------------------------- helpers
    def _new_circuit(self) -> Circuit:
        c = Circuit(self.size)
        for name, p in self.inputs.items():
            c.place(p, Block.LEVER, label=name)
        c.place(self.output, Block.LAMP, label="OUT")
        return c

    def _obs(self) -> np.ndarray:
        c = self.circuit
        obs = np.zeros(self.obs_shape, dtype=np.float32)
        for k in range(N_BLOCK_TYPES):
            obs[k] = c.block == k
        has_facing = np.isin(c.block, [Block.TORCH, Block.REPEATER, Block.LEVER, Block.COMPARATOR])
        for d in range(6):
            obs[N_BLOCK_TYPES + d] = has_facing & (c.facing == d)
        for p in self.fixed:
            obs[-1][p] = 1.0
        return obs.ravel()

    def decode(self, action: int) -> tuple[tuple[int, int, int], Block, int]:
        cell, choice = divmod(int(action), len(self.placeable))
        pos = tuple(int(i) for i in np.unravel_index(cell, self.size))
        kind, facing = self.placeable[choice]
        return pos, kind, facing

    def _valid(self, pos, kind: Block, facing: int) -> bool:
        c = self.circuit
        if pos in self.fixed:
            return False
        current = c.get(pos)
        if current == kind and (kind not in (Block.TORCH, Block.REPEATER) or c.facing[pos] == facing):
            return False  # would change nothing
        # Support rules live in the simulator (torches, dust, repeaters, comparators).
        return c.can_place(pos, kind, facing)

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            mask[a] = self._valid(*self.decode(a))
        return mask

    def _score(self) -> float:
        return evaluate_gate(self.circuit, self.gate).score

    def _used_blocks(self) -> int:
        return self.circuit.count_blocks() - len(self.fixed)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.circuit = self._new_circuit()
        self.steps = 0
        self.last_score = self._score()
        return self._obs(), {"score": self.last_score}

    def step(self, action: int):
        self.steps += 1
        reward = 0.0
        solved = False
        pos, kind, facing = self.decode(action)
        if self._valid(pos, kind, facing):
            self.circuit.place(pos, kind, facing=facing)
            result = evaluate_gate(self.circuit, self.gate)
            reward += result.score - self.last_score
            self.last_score = result.score
            solved = result.solved
        else:
            reward -= self.invalid_penalty
        terminated = solved
        truncated = not solved and self.steps >= self.max_steps
        if solved:
            reward += self.solve_bonus - self.block_cost * self._used_blocks()
        info = {"score": self.last_score, "solved": solved}
        if terminated or truncated:
            info["circuit"] = self.circuit.to_dict(include_state=False)
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        """Top-down text view of the y=0 layer."""
        chars = {Block.AIR: ".", Block.SOLID: "#", Block.DUST: "+", Block.TORCH: "i",
                 Block.REPEATER: ">", Block.LEVER: "L", Block.LAMP: "O",
                 Block.COMPARATOR: "C"}
        c = self.circuit
        lines = ["".join(chars[Block(c.block[x, 0, z])] for x in range(self.size[0]))
                 for z in range(self.size[2])]
        return "\n".join(lines)
