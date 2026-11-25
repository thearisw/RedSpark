"""Regenerate the hand-built example circuits in this folder."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from redspark.sim import Block, Circuit, DOWN, EAST, NORTH, WEST  # noqa: E402

HERE = Path(__file__).parent


def not_gate() -> Circuit:
    # lever on the side of a block -> torch on the far side of that block -> dust -> lamp
    c = Circuit((5, 1, 3))
    c.place((1, 0, 1), Block.SOLID)
    c.place((0, 0, 1), Block.LEVER, facing=EAST, label="A")   # attached to the block
    c.place((2, 0, 1), Block.TORCH, facing=WEST)              # attached to the block
    c.place((3, 0, 1), Block.DUST)
    c.place((4, 0, 1), Block.LAMP, label="OUT")
    return c


def or_gate() -> Circuit:
    # two levers feed one dust line; a dust tail points into the lamp
    c = Circuit((4, 1, 3))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN, label="A")
    c.place((0, 0, 2), Block.LEVER, facing=DOWN, label="B")
    for z in range(3):
        c.place((1, 0, z), Block.DUST)
    c.place((2, 0, 1), Block.DUST)
    c.place((3, 0, 1), Block.LAMP, label="OUT")
    return c


def and_gate() -> Circuit:
    # Two inverters feed one dust line (OR of the inverted inputs = NAND), then a final inverter.
    c = Circuit((8, 1, 5))
    for z, name in ((0, "A"), (4, "B")):
        c.place((1, 0, z), Block.SOLID)
        c.place((0, 0, z), Block.LEVER, facing=EAST, label=name)
        c.place((2, 0, z), Block.TORCH, facing=WEST)
    for z in range(5):
        c.place((3, 0, z), Block.DUST)
    c.place((4, 0, 2), Block.DUST)
    c.place((5, 0, 2), Block.SOLID)
    c.place((6, 0, 2), Block.TORCH, facing=WEST)
    c.place((7, 0, 2), Block.LAMP, label="OUT")
    return c


def repeater_line() -> Circuit:
    c = Circuit((6, 1, 1))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN, label="A")
    c.place((1, 0, 0), Block.DUST)
    c.place((2, 0, 0), Block.REPEATER, facing=EAST, delay=2)
    c.place((3, 0, 0), Block.REPEATER, facing=EAST, delay=3)
    c.place((4, 0, 0), Block.DUST)
    c.place((5, 0, 0), Block.LAMP, label="OUT")
    return c


def tower_not() -> Circuit:
    # A vertical inverter: torch on top of a block, strongly powering the block above it.
    c = Circuit((3, 3, 1))
    c.place((1, 0, 0), Block.SOLID)
    c.place((0, 0, 0), Block.LEVER, facing=EAST, label="A")
    c.place((1, 1, 0), Block.TORCH, facing=DOWN)  # standing on the block below
    c.place((1, 2, 0), Block.SOLID)
    c.place((2, 2, 0), Block.LAMP, label="OUT")
    return c


def torch_clock() -> Circuit:
    # torch -> dust loop -> repeater back into the torch's block: blinks forever
    c = Circuit((3, 1, 4))
    c.place((1, 0, 0), Block.SOLID)
    c.place((2, 0, 0), Block.TORCH, facing=WEST)
    c.place((2, 0, 1), Block.DUST)
    c.place((2, 0, 2), Block.DUST)
    c.place((1, 0, 2), Block.DUST)
    c.place((1, 0, 1), Block.REPEATER, facing=NORTH, delay=3)
    c.place((0, 0, 0), Block.LAMP, label="OUT")
    c.place((0, 0, 3), Block.LEVER, facing=DOWN, label="A")
    return c


def comparator_and_not() -> Circuit:
    # A feeds the comparator's rear at strength 13; B feeds its side at 15, which wins.
    c = Circuit((7, 1, 3))
    c.place((0, 0, 2), Block.LEVER, facing=DOWN, label="A")
    for x in range(1, 4):
        c.place((x, 0, 2), Block.DUST)
    c.place((4, 0, 2), Block.COMPARATOR, facing=EAST)
    c.place((4, 0, 0), Block.LEVER, facing=DOWN, label="B")
    c.place((4, 0, 1), Block.DUST)
    c.place((5, 0, 2), Block.DUST)
    c.place((6, 0, 2), Block.LAMP, label="OUT")
    return c


EXAMPLES = {
    "not_gate": not_gate,
    "or_gate": or_gate,
    "and_gate": and_gate,
    "repeater_line": repeater_line,
    "tower_not": tower_not,
    "torch_clock": torch_clock,
    "comparator_and_not": comparator_and_not,
}

if __name__ == "__main__":
    for name, build in EXAMPLES.items():
        path = HERE / f"{name}.json"
        path.write_text(json.dumps(build().to_dict(include_state=False), indent=1) + "\n")
        print("wrote", path)
