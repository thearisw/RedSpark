"""Tick-based redstone-style circuit simulator.

This is an original, simplified model inspired by redstone logic. It is not a
faithful copy of any game's rules. The model in one paragraph:

* Levers, lit torches and active repeaters are the *stateful* parts. Everything
  else (dust levels, powered blocks, lamps) is recomputed instantly from them.
* A solid block is **strongly** powered by an adjacent lever that is on, a lit
  torch directly below it, or an active repeater facing into it. A strongly
  powered block feeds adjacent dust at level 15.
* A solid block is **weakly** powered by dust on top of it or dust next to it.
  Weak power affects torches, repeaters and lamps but does not feed dust.
* Dust takes level 15 from an adjacent source and loses 1 per dust block it
  travels. Dust connects horizontally and one step up or down diagonally.
* A torch is attached to a solid block (below it or beside it). Each tick it
  becomes unlit if that block is powered, and lit otherwise (1 tick delay).
* A repeater reads the cell behind it and outputs, ``delay`` ticks later, into
  the cell in front of it (a torch has a fixed delay of 1 tick).
* A lamp is lit while anything next to it supplies power.

Coordinates are ``(x, y, z)`` with ``y`` pointing up.
"""

from __future__ import annotations

import copy
from enum import IntEnum
from typing import Iterator

import numpy as np


class Block(IntEnum):
    AIR = 0
    SOLID = 1
    DUST = 2
    TORCH = 3
    REPEATER = 4
    LEVER = 5
    LAMP = 6


# Direction indices: 0:+x 1:-x 2:+z 3:-z 4:+y 5:-y
DIRS: list[tuple[int, int, int]] = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)]
FACING_NAMES = ["+x", "-x", "+z", "-z", "+y", "-y"]
OPPOSITE = [1, 0, 3, 2, 5, 4]
UP, DOWN = 4, 5
HORIZONTAL = [0, 1, 2, 3]

Pos = tuple[int, int, int]


def add(p: Pos, d: int) -> Pos:
    dx, dy, dz = DIRS[d]
    return (p[0] + dx, p[1] + dy, p[2] + dz)


class Circuit:
    """A 3D grid of blocks plus their simulation state."""

    def __init__(self, size: tuple[int, int, int]):
        self.size = tuple(int(s) for s in size)
        self.block = np.zeros(self.size, dtype=np.int8)
        # TORCH: direction towards the block it is attached to (never +y).
        # REPEATER: direction it outputs to (horizontal only).
        self.facing = np.zeros(self.size, dtype=np.int8)
        self.delay = np.ones(self.size, dtype=np.int8)
        # Stateful on/off: lever on, torch lit, repeater active, lamp lit.
        self.state = np.zeros(self.size, dtype=bool)
        # Derived fields, recomputed by propagate().
        self.dust = np.zeros(self.size, dtype=np.int8)
        self.strong = np.zeros(self.size, dtype=bool)
        self.weak = np.zeros(self.size, dtype=bool)
        self.labels: dict[str, Pos] = {}
        self._rep_buf: dict[Pos, list[bool]] = {}
        self._pos_cache: dict[Block, list[Pos]] = {}
        self.ticks = 0

    # ------------------------------------------------------------------ editing
    def in_bounds(self, p: Pos) -> bool:
        return all(0 <= p[i] < self.size[i] for i in range(3))

    def get(self, p: Pos) -> Block:
        return Block(self.block[p]) if self.in_bounds(p) else Block.AIR

    def place(self, p: Pos, kind: Block, facing: int = 0, delay: int = 1,
              on: bool = False, label: str | None = None) -> None:
        p = tuple(p)
        if not self.in_bounds(p):
            raise ValueError(f"{p} is outside the grid {self.size}")
        kind = Block(kind)
        if kind == Block.TORCH and facing == UP:
            raise ValueError("a torch cannot hang from the block above it")
        if kind == Block.REPEATER and facing not in HORIZONTAL:
            raise ValueError("a repeater must face horizontally")
        if not 1 <= delay <= 4:
            raise ValueError("repeater delay must be 1..4")
        self.block[p] = kind
        self._pos_cache.clear()
        self.facing[p] = facing
        self.delay[p] = delay
        # Torches start lit, like freshly placed ones; levers take `on`.
        self.state[p] = True if kind == Block.TORCH else (on if kind == Block.LEVER else False)
        self._rep_buf.pop(p, None)
        if kind == Block.REPEATER:
            self._rep_buf[p] = [False] * (delay - 1)
        for name, q in list(self.labels.items()):
            if q == p:
                del self.labels[name]
        if label:
            self.labels[label] = p
        self.propagate()

    def remove(self, p: Pos) -> None:
        self.place(p, Block.AIR)

    def set_lever(self, p: Pos, on: bool) -> None:
        p = tuple(p)
        if self.get(p) != Block.LEVER:
            raise ValueError(f"no lever at {p}")
        self.state[p] = bool(on)
        self.propagate()

    def toggle(self, p: Pos) -> None:
        self.set_lever(p, not self.state[tuple(p)])

    def copy(self) -> "Circuit":
        return copy.deepcopy(self)

    def positions(self, kind: Block) -> list[Pos]:
        cached = self._pos_cache.get(kind)
        if cached is None:
            cached = [tuple(int(i) for i in idx) for idx in np.argwhere(self.block == kind)]
            self._pos_cache[kind] = cached
        return cached

    def count_blocks(self) -> int:
        return int(np.count_nonzero(self.block))

    # --------------------------------------------------------------- simulation
    def powered(self, p: Pos) -> bool:
        """Whether a solid block is powered (strongly or weakly)."""
        return self.in_bounds(p) and bool(self.strong[p] or self.weak[p])

    def supplies(self, src: Pos, target: Pos) -> bool:
        """Whether the cell ``src`` delivers power into the adjacent cell ``target``."""
        if not self.in_bounds(src):
            return False
        kind = self.get(src)
        if kind == Block.LEVER:
            return bool(self.state[src])
        if kind == Block.TORCH:
            return bool(self.state[src]) and add(src, int(self.facing[src])) != target
        if kind == Block.REPEATER:
            return bool(self.state[src]) and add(src, int(self.facing[src])) == target
        if kind == Block.DUST:
            return self.dust[src] > 0
        if kind == Block.SOLID:
            return self.powered(src)
        return False

    def _dust_links(self, p: Pos) -> Iterator[Pos]:
        for d in HORIZONTAL:
            q = add(p, d)
            if self.get(q) == Block.DUST:
                yield q
            for vy in (1, -1):
                r = (q[0], q[1] + vy, q[2])
                if self.get(r) == Block.DUST:
                    yield r

    def propagate(self) -> None:
        """Recompute dust levels, block power and lamps from the stateful parts."""
        self.strong[:] = False
        self.weak[:] = False
        self.dust[:] = 0

        # 1. Strong power on solid blocks.
        for p in self.positions(Block.LEVER):
            if self.state[p]:
                for d in range(6):
                    q = add(p, d)
                    if self.get(q) == Block.SOLID:
                        self.strong[q] = True
        for p in self.positions(Block.TORCH):
            q = add(p, UP)
            if self.state[p] and self.get(q) == Block.SOLID:
                self.strong[q] = True
        for p in self.positions(Block.REPEATER):
            q = add(p, int(self.facing[p]))
            if self.state[p] and self.get(q) == Block.SOLID:
                self.strong[q] = True

        # 2. Dust: sources at 15, then spread with decay (bucketed by level).
        buckets: list[list[Pos]] = [[] for _ in range(16)]
        for p in self.positions(Block.DUST):
            for d in range(6):
                q = add(p, d)
                k = self.get(q)
                if k == Block.SOLID:
                    fed = bool(self.strong[q])
                elif k == Block.DUST:
                    fed = False
                else:
                    fed = self.supplies(q, p)
                if fed:
                    self.dust[p] = 15
                    buckets[15].append(p)
                    break
        for level in range(15, 1, -1):
            for p in buckets[level]:
                if self.dust[p] != level:
                    continue
                for q in self._dust_links(p):
                    if self.dust[q] < level - 1:
                        self.dust[q] = level - 1
                        buckets[level - 1].append(q)

        # 3. Weak power from dust: the block below it and blocks beside it.
        for p in self.positions(Block.DUST):
            if self.dust[p] > 0:
                for d in HORIZONTAL + [DOWN]:
                    q = add(p, d)
                    if self.get(q) == Block.SOLID:
                        self.weak[q] = True

        # 4. Lamps react instantly.
        for p in self.positions(Block.LAMP):
            self.state[p] = any(self.supplies(add(p, d), p) for d in range(6))

    def is_settled(self) -> bool:
        """True if the next tick cannot change anything (a fixed point)."""
        for p in self.positions(Block.TORCH):
            attached = add(p, int(self.facing[p]))
            off = self.get(attached) == Block.SOLID and self.powered(attached)
            if self.state[p] == off:
                return False
        for p in self.positions(Block.REPEATER):
            inp = self.supplies(add(p, OPPOSITE[int(self.facing[p])]), p)
            if self.state[p] != inp or any(b != inp for b in self._rep_buf.get(p, [])):
                return False
        return True

    def settle(self, max_ticks: int = 64) -> bool:
        """Step until nothing changes. Returns False if still changing after ``max_ticks``."""
        for _ in range(max_ticks):
            if self.is_settled():
                return True
            self.step()
        return self.is_settled()

    def step(self, n: int = 1) -> None:
        """Advance the simulation by ``n`` ticks."""
        for _ in range(n):
            self.propagate()
            new_state = self.state.copy()
            for p in self.positions(Block.TORCH):
                attached = add(p, int(self.facing[p]))
                off = self.get(attached) == Block.SOLID and self.powered(attached)
                new_state[p] = not off
            for p in self.positions(Block.REPEATER):
                back = add(p, OPPOSITE[int(self.facing[p])])
                buf = self._rep_buf.setdefault(p, [False] * (int(self.delay[p]) - 1))
                buf.append(self.supplies(back, p))
                new_state[p] = buf.pop(0)
            self.state = new_state
            self.propagate()
            self.ticks += 1

    # ------------------------------------------------------------ serialisation
    def to_dict(self, include_state: bool = True) -> dict:
        blocks = []
        for idx in np.argwhere(self.block != Block.AIR):
            p = tuple(int(i) for i in idx)
            kind = Block(self.block[p])
            entry: dict = {"pos": list(p), "type": kind.name.lower()}
            if kind in (Block.TORCH, Block.REPEATER):
                entry["facing"] = FACING_NAMES[int(self.facing[p])]
            if kind == Block.REPEATER:
                entry["delay"] = int(self.delay[p])
            if kind == Block.LEVER:
                entry["on"] = bool(self.state[p])
            label = next((n for n, q in self.labels.items() if q == p), None)
            if label:
                entry["label"] = label
            if include_state:
                entry["active"] = bool(self.state[p]) if kind != Block.SOLID else self.powered(p)
                if kind == Block.DUST:
                    entry["level"] = int(self.dust[p])
            blocks.append(entry)
        return {"size": list(self.size), "ticks": self.ticks, "blocks": blocks}

    @classmethod
    def from_dict(cls, data: dict) -> "Circuit":
        c = cls(tuple(data["size"]))
        for b in data["blocks"]:
            kind = Block[b["type"].upper()]
            facing = FACING_NAMES.index(b.get("facing", "+x"))
            c.place(tuple(b["pos"]), kind, facing=facing, delay=b.get("delay", 1),
                    on=b.get("on", False), label=b.get("label"))
        return c
