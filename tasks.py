"""Logic-gate tasks: truth tables and scoring a circuit against them."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

from .sim import Block, Circuit

GATES: dict[str, tuple[int, Callable[..., bool]]] = {
    "NOT": (1, lambda a: not a),
    "BUFFER": (1, lambda a: a),
    "AND": (2, lambda a, b: a and b),
    "OR": (2, lambda a, b: a or b),
    "NAND": (2, lambda a, b: not (a and b)),
    "NOR": (2, lambda a, b: not (a or b)),
    "XOR": (2, lambda a, b: a != b),
    "ANDNOT": (2, lambda a, b: a and not b),  # A and not B
}


def truth_table(gate: str) -> list[tuple[tuple[bool, ...], bool]]:
    n, fn = GATES[gate]
    return [(combo, bool(fn(*combo))) for combo in itertools.product([False, True], repeat=n)]


@dataclass
class GateResult:
    rows: list[tuple[tuple[bool, ...], bool, bool]]  # (inputs, expected, got)
    stable: bool

    @property
    def correct(self) -> int:
        return sum(exp == got for _, exp, got in self.rows)

    @property
    def score(self) -> float:
        return self.correct / len(self.rows)

    @property
    def solved(self) -> bool:
        return self.stable and self.correct == len(self.rows)


def lamp_on(c: Circuit, p) -> bool:
    return bool(p is not None and c.get(p) == Block.LAMP and c.state[p])


def evaluate_gate(circuit: Circuit, gate: str, inputs: list[str] | None = None,
                  output: str = "OUT", settle_ticks: int | None = None) -> GateResult:
    """Run every row of the gate's truth table and compare the output lamp.

    Inputs are the levers labelled ``A``, ``B``, ... and the output is the lamp
    labelled ``OUT``. A row counts as wrong if the circuit has not settled
    after ``settle_ticks`` ticks (for example a torch clock).
    """
    n, _ = GATES[gate]
    inputs = inputs or ["A", "B", "C"][:n]
    c = circuit.copy()
    settle = settle_ticks if settle_ticks is not None else 8 * sum(c.size) + 40  # game ticks
    out = c.labels.get(output)
    rows = []
    stable = True
    for combo, expected in truth_table(gate):
        for name, value in zip(inputs, combo):
            p = c.labels.get(name)
            if p is None or c.get(p) != Block.LEVER:
                raise ValueError(f"circuit has no lever labelled {name}")
            c.set_lever(p, value)
        settled = c.settle(settle)
        got = lamp_on(c, out)
        if not settled:  # still changing (e.g. a torch clock): never counts as correct
            stable = False
            got = not expected
        rows.append((combo, expected, got))
    return GateResult(rows=rows, stable=stable)
