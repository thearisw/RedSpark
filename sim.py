"""Event-driven redstone-style simulator modelled on Java Edition's update order.

This is an original implementation written from public technical write-ups of
how Java Edition processes redstone. It does not use any game code.

Time
    The simulation advances in **game ticks** (20 per second). One redstone
    tick is 2 game ticks: a torch reacts after 2 gt and a repeater after 2, 4,
    6 or 8 gt.

What happens inside one game tick (``Circuit.step``)
    1. The game time advances.
    2. **Scheduled tile ticks** that are due run in order of (due time,
       priority, scheduling order). Priority: lower runs first. Repeaters use
       -3 when facing into another diode that does not face back, -2 when
       turning off and -1 when turning on. Comparators use -1 when facing into
       another diode that does not face back, else 0. Everything else uses 0.
    3. Block events and block entities (pistons, hoppers...) would run here.
       None of the current components use them.

How changes spread
    Nothing is recomputed globally. When a block changes it sends:

    * **shape updates** (post-placement updates) to its 6 neighbours, which
      only affect a block's own shape, such as repeater locking or dust
      connections. Observers will listen to these.
    * **block updates** (neighbour-changed updates). ``update_neighbors_at(p)``
      notifies the 6 neighbours of ``p`` in Java's order: west, east, down,
      up, north, south. A block only re-checks its power when it receives one,
      which is what makes budding and quasi-connectivity possible.

Components
    * **Lever**: attached to a block face. Weak power 15 to all neighbours and
      strong power into the block it is attached to.
    * **Torch**: attached below or beside. Only blocks in
      ``SUPPORTS_WALL_TORCH`` / ``SUPPORTS_STANDING_TORCH`` (solid blocks and
      lamps) can hold one, plus the floor for standing torches. A torch whose
      support is removed breaks off. Weak power 15 to all neighbours except
      its attachment block, and strong power into the block above. It turns off
      2 gt after its attachment block is powered. 8 toggles within 60 gt burn
      it out, and it rechecks 160 gt later.
    * **Repeater**: reads the block behind it (dust counts by its level). After
      its delay it gives strong power 15 to the block in front. Pulses shorter
      than the delay are extended. A powered diode facing into its side locks it.
    * **Dust**: power 0-15. It takes the strongest non-dust signal around it, or
      the strongest connected dust minus 1. It weakly powers the block below and
      the blocks it *points* into. A line points along itself, a lone dot points
      everywhere (the "cross"), and a dust with one connection points straight
      through. When its level changes, it block-updates around itself and every
      neighbour (7 positions, 42 updates). Java iterates those positions in hash
      order ("locational" behaviour); this simulator uses a fixed order.
    * **Comparator**: reads the block behind it (dust by its level) and the
      strongest *side* input. Only dust (by level) and the direct output of
      levers, torches, repeaters and comparators count as side inputs; plain
      blocks do not. Compare mode outputs the rear signal unless a side is
      stronger. Subtract mode outputs rear minus side. It updates 2 gt after
      its inputs change. Like a repeater it strongly powers the block in front,
      but at its output strength. ``comparator_update`` is the hook containers
      will call when their contents change.
    * **Lamp**: turns on as soon as it is powered and off 4 gt after power is lost.
      It is a conductor.
    * **Solid block**: a conductor. It is *strongly* powered by direct
      signals from levers, torches below and repeaters, which feed adjacent dust.
      It is *weakly* powered by dust pointing into it, which affects components
      but not dust.

Quasi-connectivity
    ``has_neighbor_signal_qc`` is the helper that pistons, droppers and
    dispensers will use: powered if they, *or the position above them*, has a
    neighbour signal. Because blocks only re-check on updates, those components
    can be "budded" (quasi-powered without noticing) exactly as in Java.

Simplifications: dust and repeaters do not need a block underneath (the grid
floor counts as support), and there are no observers or pistons
yet, nor containers for comparators to read. Coordinates are ``(x, y, z)`` with ``y`` up, +x = east and +z = south.
"""

from __future__ import annotations

import copy
import heapq
import sys
from enum import IntEnum

import numpy as np


class Block(IntEnum):
    AIR = 0
    SOLID = 1
    DUST = 2
    TORCH = 3
    REPEATER = 4
    LEVER = 5
    LAMP = 6
    COMPARATOR = 7


DIODES = (Block.REPEATER, Block.COMPARATOR)


# Direction indices: 0:+x(east) 1:-x(west) 2:+z(south) 3:-z(north) 4:+y(up) 5:-y(down)
DIRS: list[tuple[int, int, int]] = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)]
FACING_NAMES = ["+x", "-x", "+z", "-z", "+y", "-y"]
EAST, WEST, SOUTH, NORTH, UP, DOWN = 0, 1, 2, 3, 4, 5
OPPOSITE = [1, 0, 3, 2, 5, 4]
HORIZONTAL = [EAST, WEST, SOUTH, NORTH]
# Java's neighbour update order (Level.updateNeighborsAt): west, east, down, up, north, south.
UPDATE_ORDER = [WEST, EAST, DOWN, UP, NORTH, SOUTH]
# Java's Direction.values() order: down, up, north, south, west, east.
VALUES_ORDER = [DOWN, UP, NORTH, SOUTH, WEST, EAST]

# Delays in game ticks.
TORCH_DELAY = 2
COMPARATOR_DELAY = 2
LAMP_OFF_DELAY = 4
BURNOUT_TOGGLES, BURNOUT_WINDOW, BURNOUT_RECOVER = 8, 60, 160

# Tile tick priorities (lower runs first).
EXTREMELY_HIGH, VERY_HIGH, HIGH, NORMAL = -3, -2, -1, 0

CONDUCTORS = (Block.SOLID, Block.LAMP)

# Placement support, kept as data so new blocks only need adding here.
# A wall torch hangs on the side of one of these; a standing torch sits on top of one.
SUPPORTS_WALL_TORCH = frozenset({Block.SOLID, Block.LAMP})
SUPPORTS_STANDING_TORCH = frozenset({Block.SOLID, Block.LAMP})
FLOOR_SUPPORTS_STANDING_TORCH = True  # the floor under y = 0 counts as a block

Pos = tuple[int, int, int]


def add(p: Pos, d: int) -> Pos:
    dx, dy, dz = DIRS[d]
    return (p[0] + dx, p[1] + dy, p[2] + dz)


def direction_between(a: Pos, b: Pos) -> int:
    """Direction from a to the adjacent cell b."""
    return DIRS.index((b[0] - a[0], b[1] - a[1], b[2] - a[2]))


class Circuit:
    """A 3D grid of blocks with an event-driven simulation."""

    def __init__(self, size: tuple[int, int, int]):
        self.size = tuple(int(s) for s in size)
        self.block = np.zeros(self.size, dtype=np.int8)
        # TORCH / LEVER: direction towards the block it is attached to.
        # REPEATER / COMPARATOR: direction it outputs to (horizontal only).
        self.facing = np.zeros(self.size, dtype=np.int8)
        self.delay = np.ones(self.size, dtype=np.int8)       # repeater delay in redstone ticks
        self.state = np.zeros(self.size, dtype=bool)          # lever on, torch lit, repeater/lamp powered
        self.dust = np.zeros(self.size, dtype=np.int8)        # dust power level
        self.locked = np.zeros(self.size, dtype=bool)         # repeater locked
        self.output = np.zeros(self.size, dtype=np.int8)      # comparator output strength
        self.subtract = np.zeros(self.size, dtype=bool)       # comparator in subtract mode
        self.labels: dict[str, Pos] = {}
        self.ticks = 0                                        # game time
        self._queue: list[tuple[int, int, int, Pos]] = []     # (due, priority, order, pos)
        self._scheduled: dict[Pos, int] = {}                  # pos -> due tick
        self._order = 0                                       # tie-breaker: scheduling order
        self._burnout: dict[Pos, list[int]] = {}

    # ------------------------------------------------------------------ basics
    def in_bounds(self, p: Pos) -> bool:
        return 0 <= p[0] < self.size[0] and 0 <= p[1] < self.size[1] and 0 <= p[2] < self.size[2]

    def get(self, p: Pos) -> Block:
        return Block(int(self.block[p])) if self.in_bounds(p) else Block.AIR

    def is_conductor(self, p: Pos) -> bool:
        return self.get(p) in CONDUCTORS

    def count_blocks(self) -> int:
        return int(np.count_nonzero(self.block))

    def copy(self) -> "Circuit":
        clone = copy.copy(self)
        for name in ("block", "facing", "delay", "state", "dust", "locked", "output", "subtract"):
            setattr(clone, name, getattr(self, name).copy())
        clone.labels = dict(self.labels)
        clone._queue = list(self._queue)
        clone._scheduled = dict(self._scheduled)
        clone._burnout = {k: list(v) for k, v in self._burnout.items()}
        return clone

    # ---------------------------------------------------------------- signals
    def _emitted(self, p: Pos, toward: Pos, direct: bool, dust_signals: bool) -> int:
        """Signal the block at ``p`` sends into the adjacent cell ``toward``.

        ``direct`` asks for strong (direct) power only. With ``dust_signals``
        False, dust emits nothing (this is how dust reads its surroundings).
        """
        kind = self.get(p)
        if kind == Block.LEVER:
            if not self.state[p]:
                return 0
            return 15 if not direct or toward == add(p, int(self.facing[p])) else 0
        if kind == Block.TORCH:
            if not self.state[p]:
                return 0
            if direct:
                return 15 if toward == add(p, UP) else 0
            return 0 if toward == add(p, int(self.facing[p])) else 15
        if kind == Block.REPEATER:
            return 15 if self.state[p] and toward == add(p, int(self.facing[p])) else 0
        if kind == Block.COMPARATOR:
            return int(self.output[p]) if toward == add(p, int(self.facing[p])) else 0
        if kind == Block.DUST:
            if not dust_signals or self.dust[p] == 0:
                return 0
            d = direction_between(p, toward)
            if d == DOWN:
                return int(self.dust[p])
            if d == UP:
                return 0
            return int(self.dust[p]) if self.dust_points(p, d) else 0
        return 0

    def direct_signal_to(self, p: Pos, dust_signals: bool = True) -> int:
        """Strongest direct signal going into ``p`` (how conductors get powered)."""
        best = 0
        for d in range(6):
            q = add(p, d)
            if self.in_bounds(q):
                best = max(best, self._emitted(q, p, True, dust_signals))
                if best >= 15:
                    break
        return best

    def signal(self, p: Pos, toward: Pos, dust_signals: bool = True) -> int:
        """Signal a neighbour at ``toward`` receives from ``p`` (Level.getSignal)."""
        if not self.in_bounds(p):
            return 0
        s = self._emitted(p, toward, False, dust_signals)
        if self.is_conductor(p):
            s = max(s, self.direct_signal_to(p, dust_signals))
        return s

    def has_neighbor_signal(self, p: Pos) -> bool:
        return any(self.signal(add(p, d), p) > 0 for d in UPDATE_ORDER)

    def has_neighbor_signal_qc(self, p: Pos) -> bool:
        """Quasi-connectivity check for pistons, droppers and dispensers."""
        return self.has_neighbor_signal(p) or (self.in_bounds(add(p, UP)) and self.has_neighbor_signal(add(p, UP)))

    def powered(self, p: Pos) -> bool:
        """Whether a conductor is powered at all (strongly or weakly)."""
        return self.is_conductor(p) and self.direct_signal_to(p) > 0

    def strongly_powered(self, p: Pos) -> bool:
        """Powered by something other than dust, so it also feeds adjacent dust."""
        return self.is_conductor(p) and self.direct_signal_to(p, dust_signals=False) > 0

    # ------------------------------------------------------------------- dust
    def _connects(self, p: Pos, d: int) -> bool:
        """Whether dust at ``p`` connects towards horizontal direction ``d``."""
        q = add(p, d)
        kind = self.get(q)
        if kind == Block.DUST:
            return True
        if kind in (Block.LEVER, Block.TORCH):
            return True
        if kind in DIODES:
            return int(self.facing[q]) in (d, OPPOSITE[d])
        above, below = add(q, UP), add(q, DOWN)
        if self.get(above) == Block.DUST and not self.is_conductor(add(p, UP)):
            return True  # climbs onto the block beside it
        if not self.is_conductor(q) and self.get(below) == Block.DUST:
            return True  # steps down
        return False

    def dust_shape(self, p: Pos) -> int:
        """Bitmask of the sides the dust points to: 1 = +x, 2 = -x, 4 = +z, 8 = -z."""
        sides = [d for d in HORIZONTAL if self._connects(p, d)]
        if not sides:
            sides = list(HORIZONTAL)                 # dot acts as a cross
        elif len(sides) == 1:
            sides.append(OPPOSITE[sides[0]])         # one connection: straight line
        return sum(1 << d for d in sides)

    def dust_points(self, p: Pos, d: int) -> bool:
        return bool(self.dust_shape(p) & (1 << d))

    def _dust_target(self, p: Pos) -> int:
        strongest = max((self.signal(add(p, d), p, dust_signals=False) for d in range(6)), default=0)
        if strongest >= 15:
            return 15
        best_wire = 0
        up_conductor = self.is_conductor(add(p, UP))
        for d in HORIZONTAL:
            q = add(p, d)
            if self.get(q) == Block.DUST:
                best_wire = max(best_wire, int(self.dust[q]))
            if self.is_conductor(q):
                if not up_conductor and self.get(add(q, UP)) == Block.DUST:
                    best_wire = max(best_wire, int(self.dust[add(q, UP)]))
            elif self.get(add(q, DOWN)) == Block.DUST:
                best_wire = max(best_wire, int(self.dust[add(q, DOWN)]))
        return max(strongest, best_wire - 1)

    def _update_dust(self, p: Pos) -> None:
        target = self._dust_target(p)
        if target == self.dust[p]:
            return
        self.dust[p] = target
        for q in [p] + [add(p, d) for d in VALUES_ORDER]:
            self.update_neighbors_at(q)

    # ------------------------------------------------------------ scheduling
    def schedule(self, p: Pos, delay: int, priority: int = NORMAL) -> None:
        if p in self._scheduled:
            return
        due = self.ticks + delay
        self._scheduled[p] = due
        self._order += 1
        heapq.heappush(self._queue, (due, priority, self._order, p))

    def has_scheduled_tick(self, p: Pos) -> bool:
        return p in self._scheduled

    def will_tick_this_tick(self, p: Pos) -> bool:
        return self._scheduled.get(p) == self.ticks

    # ---------------------------------------------------------------- updates
    def update_neighbors_at(self, p: Pos) -> None:
        for d in UPDATE_ORDER:
            q = add(p, d)
            if self.in_bounds(q):
                self.neighbor_changed(q, p)

    def _shape_updates(self, p: Pos) -> None:
        for d in VALUES_ORDER:
            q = add(p, d)
            if self.get(q) == Block.REPEATER:
                self.locked[q] = self._repeater_locked(q)

    def neighbor_changed(self, p: Pos, source: Pos) -> None:
        kind = self.get(p)
        if kind == Block.DUST:
            self._update_dust(p)
        elif kind == Block.TORCH:
            powered = self._torch_input(p)
            if bool(self.state[p]) == powered and not self.will_tick_this_tick(p):
                self.schedule(p, TORCH_DELAY)
        elif kind == Block.REPEATER:
            self.locked[p] = self._repeater_locked(p)
            if not self.locked[p]:
                should = self._repeater_input(p) > 0
                if bool(self.state[p]) != should and not self.has_scheduled_tick(p):
                    if self._diode_prioritized(p):
                        priority = EXTREMELY_HIGH
                    elif self.state[p]:
                        priority = VERY_HIGH
                    else:
                        priority = HIGH
                    self.schedule(p, 2 * int(self.delay[p]), priority)
        elif kind == Block.COMPARATOR:
            if not self.has_scheduled_tick(p):
                if (self._comparator_output(p) != self.output[p]
                        or bool(self.state[p]) != self._comparator_should_power(p)):
                    priority = HIGH if self._diode_prioritized(p) else NORMAL
                    self.schedule(p, COMPARATOR_DELAY, priority)
        elif kind == Block.LAMP:
            powered = self.has_neighbor_signal(p)
            if self.state[p] and not powered:
                self.schedule(p, LAMP_OFF_DELAY)
            elif not self.state[p] and powered:
                self._set_state(p, True)

    def _tile_tick(self, p: Pos) -> None:
        kind = self.get(p)
        if kind == Block.TORCH:
            powered = self._torch_input(p)
            if self.state[p]:
                if powered:
                    self._set_state(p, False)
                    if self._record_toggle(p):  # burnt out: try again much later
                        self.schedule(p, BURNOUT_RECOVER)
            elif not powered and not self._toggled_too_often(p):
                self._set_state(p, True)
        elif kind == Block.REPEATER:
            if self.locked[p]:
                return
            should = self._repeater_input(p) > 0
            if self.state[p] and not should:
                self._set_state(p, False)
            elif not self.state[p]:
                self._set_state(p, True)
                if not should:  # extend short pulses to the full delay
                    self.schedule(p, 2 * int(self.delay[p]), VERY_HIGH)
        elif kind == Block.COMPARATOR:
            self._comparator_refresh(p)
        elif kind == Block.LAMP:
            if self.state[p] and not self.has_neighbor_signal(p):
                self._set_state(p, False)

    # --------------------------------------------------------- component logic
    def _torch_input(self, p: Pos) -> bool:
        attached = add(p, int(self.facing[p]))
        return self.signal(attached, p) > 0

    def _record_toggle(self, p: Pos) -> bool:
        history = [t for t in self._burnout.get(p, []) if self.ticks - t < BURNOUT_WINDOW]
        history.append(self.ticks)
        self._burnout[p] = history
        return len(history) >= BURNOUT_TOGGLES

    def _toggled_too_often(self, p: Pos) -> bool:
        return sum(self.ticks - t < BURNOUT_WINDOW for t in self._burnout.get(p, [])) >= BURNOUT_TOGGLES

    def _repeater_input(self, p: Pos) -> int:
        """Rear input of a diode: the signal from behind, with dust counted by its level."""
        back = add(p, OPPOSITE[int(self.facing[p])])
        s = self.signal(back, p)
        if s < 15 and self.get(back) == Block.DUST:
            s = max(s, int(self.dust[back]))
        return s

    def _side_input_at(self, p: Pos, side: int) -> int:
        q = add(p, side)
        kind = self.get(q)
        if kind == Block.DUST:
            return int(self.dust[q])
        if kind in (Block.LEVER, Block.TORCH, Block.REPEATER, Block.COMPARATOR):
            return self._emitted(q, p, True, True)  # signal sources count only with direct power
        return 0

    def _comparator_side(self, p: Pos) -> int:
        f = int(self.facing[p])
        return max(self._side_input_at(p, d) for d in HORIZONTAL if d not in (f, OPPOSITE[f]))

    def _comparator_output(self, p: Pos) -> int:
        rear = self._repeater_input(p)
        if rear == 0:
            return 0
        side = self._comparator_side(p)
        if side > rear:
            return 0
        return rear - side if self.subtract[p] else rear

    def _comparator_should_power(self, p: Pos) -> bool:
        rear = self._repeater_input(p)
        if rear == 0:
            return False
        side = self._comparator_side(p)
        return rear > side or (rear == side and not self.subtract[p])

    def _comparator_refresh(self, p: Pos) -> None:
        out = self._comparator_output(p)
        old = int(self.output[p])
        self.output[p] = out
        if old != out or not self.subtract[p]:
            self.state[p] = self._comparator_should_power(p)
            self._shape_updates(p)
            self._update_front(p)

    def comparator_update(self, p: Pos) -> None:
        """Notify comparators reading ``p`` directly or through one block.

        Containers will call this when their contents change, as in Java.
        """
        for d in HORIZONTAL:
            q = add(p, d)
            if self.get(q) == Block.COMPARATOR:
                self.neighbor_changed(q, p)
            elif self.is_conductor(q):
                r = add(q, d)
                if self.get(r) == Block.COMPARATOR and add(r, OPPOSITE[int(self.facing[r])]) == q:
                    self.neighbor_changed(r, q)

    def _update_front(self, p: Pos) -> None:
        """A diode updates only the block in front of it and that block's neighbours."""
        front = add(p, int(self.facing[p]))
        if self.in_bounds(front):
            self.neighbor_changed(front, p)
        for d in UPDATE_ORDER:
            q = add(front, d)
            if q != p and self.in_bounds(q):
                self.neighbor_changed(q, front)

    def _repeater_locked(self, p: Pos) -> bool:
        f = int(self.facing[p])
        for side in HORIZONTAL:
            if side in (f, OPPOSITE[f]):
                continue
            q = add(p, side)
            if (self.get(q) in DIODES and int(self.facing[q]) == OPPOSITE[side]
                    and self._emitted(q, p, True, True) > 0):
                return True
        return False

    def _diode_prioritized(self, p: Pos) -> bool:
        """Facing into another diode that does not face back at this one."""
        f = int(self.facing[p])
        front = add(p, f)
        return self.get(front) in DIODES and int(self.facing[front]) != OPPOSITE[f]

    def _set_state(self, p: Pos, on: bool) -> None:
        """Change a component's on/off state and send the updates Java would."""
        if bool(self.state[p]) == on:
            return
        self.state[p] = on
        kind = self.get(p)
        self._shape_updates(p)
        if kind == Block.TORCH:
            for d in VALUES_ORDER:          # onPlace: update around every neighbour
                self.update_neighbors_at(add(p, d))
            self.update_neighbors_at(p)
        elif kind == Block.REPEATER:        # only the block in front and its neighbours
            self._update_front(p)
        elif kind == Block.LEVER:
            self.update_neighbors_at(p)
            self.update_neighbors_at(add(p, int(self.facing[p])))
        # lamps change with flag 2 in Java: no block updates

    # ----------------------------------------------------------------- editing
    def place(self, p: Pos, kind: Block, facing: int | None = None, delay: int = 1,
              on: bool = False, label: str | None = None, subtract: bool = False) -> None:
        """Place (or with AIR, remove) a block and let the circuit react."""
        p = tuple(int(v) for v in p)
        if not self.in_bounds(p):
            raise ValueError(f"{p} is outside the grid {self.size}")
        kind = Block(kind)
        if facing is None:
            facing = DOWN if kind in (Block.TORCH, Block.LEVER) else EAST
        if kind == Block.TORCH and not self.can_support_torch(p, facing):
            raise ValueError(f"a torch at {p} needs a solid block or lamp to attach to "
                             f"(towards {FACING_NAMES[facing]})")
        if kind in DIODES and facing not in HORIZONTAL:
            raise ValueError(f"a {kind.name.lower()} must face horizontally")
        if not 1 <= delay <= 4:
            raise ValueError("repeater delay must be 1..4")

        old = self.get(p)
        old_state = (int(self.facing[p]), bool(self.state[p]) or bool(self.output[p]))
        self.block[p] = kind
        self.facing[p] = facing
        self.delay[p] = delay
        self.state[p] = True if kind == Block.TORCH else (on if kind == Block.LEVER else False)
        self.dust[p] = 0
        self.locked[p] = False
        self.output[p] = 0
        self.subtract[p] = subtract if kind == Block.COMPARATOR else False
        self._scheduled.pop(p, None)  # a stale tick is ignored when it comes up
        for name, q in list(self.labels.items()):
            if q == p:
                del self.labels[name]
        if label:
            self.labels[label] = p

        # Updates, in the order Java sends them for a block change.
        self._shape_updates(p)
        if old == Block.TORCH or kind == Block.TORCH or old == Block.DUST:
            for d in VALUES_ORDER:
                self.update_neighbors_at(add(p, d))
        if old == Block.LEVER:
            self.update_neighbors_at(add(p, old_state[0]))
        if kind == Block.LEVER:
            self.update_neighbors_at(add(p, facing))
        if old in DIODES and old_state[1]:
            self.update_neighbors_at(add(p, old_state[0]))
        self.update_neighbors_at(p)
        self.neighbor_changed(p, p)  # the new block checks its own inputs
        if kind not in SUPPORTS_WALL_TORCH | SUPPORTS_STANDING_TORCH:
            self._drop_unsupported_torches(p)

    def can_support_torch(self, p: Pos, facing: int) -> bool:
        """Whether a torch at ``p`` attached towards ``facing`` has something to hang on."""
        if facing == UP:
            return False
        support = add(p, facing)
        if facing == DOWN:
            if support[1] < 0:
                return FLOOR_SUPPORTS_STANDING_TORCH
            return self.get(support) in SUPPORTS_STANDING_TORCH
        return self.get(support) in SUPPORTS_WALL_TORCH

    def can_place(self, p: Pos, kind: Block, facing: int | None = None) -> bool:
        """Placement rules that the simulator enforces (training masks should use this)."""
        p = tuple(int(v) for v in p)
        if not self.in_bounds(p):
            return False
        if kind == Block.TORCH:
            return self.can_support_torch(p, DOWN if facing is None else facing)
        return True

    def _drop_unsupported_torches(self, p: Pos) -> None:
        """Torches next to ``p`` that lost their support break off, as in Java."""
        for d in VALUES_ORDER:
            q = add(p, d)
            if self.get(q) == Block.TORCH and add(q, int(self.facing[q])) == p \
                    and not self.can_support_torch(q, int(self.facing[q])):
                self.place(q, Block.AIR)

    def remove(self, p: Pos) -> None:
        self.place(p, Block.AIR)

    def set_lever(self, p: Pos, on: bool) -> None:
        p = tuple(p)
        if self.get(p) != Block.LEVER:
            raise ValueError(f"no lever at {p}")
        self._set_state(p, bool(on))

    def toggle(self, p: Pos) -> None:
        self.set_lever(p, not self.state[tuple(p)])

    def set_comparator_mode(self, p: Pos, subtract: bool) -> None:
        """Switch a comparator between compare and subtract mode (takes effect at once)."""
        p = tuple(p)
        if self.get(p) != Block.COMPARATOR:
            raise ValueError(f"no comparator at {p}")
        if bool(self.subtract[p]) != subtract:
            self.subtract[p] = subtract
            self._comparator_refresh(p)

    # -------------------------------------------------------------- simulation
    def step(self, n: int = 1) -> None:
        """Advance ``n`` game ticks (20 per second)."""
        for _ in range(n):
            self.ticks += 1
            ran = 0
            while self._queue and self._queue[0][0] <= self.ticks and ran < 65536:
                due, _, _, p = heapq.heappop(self._queue)
                if self._scheduled.get(p) != due:
                    continue  # cancelled by a block change
                del self._scheduled[p]
                self._tile_tick(p)
                ran += 1

    def is_settled(self) -> bool:
        """True when nothing is scheduled, so nothing will change on its own."""
        return not self._scheduled

    def settle(self, max_ticks: int = 200) -> bool:
        """Step until no ticks are pending. Returns False if still busy after ``max_ticks``."""
        for _ in range(max_ticks):
            if self.is_settled():
                return True
            self.step()
        return self.is_settled()

    # ----------------------------------------------------------- serialisation
    def to_dict(self, include_state: bool = True) -> dict:
        blocks = []
        for idx in np.argwhere(self.block != Block.AIR):
            p = tuple(int(i) for i in idx)
            kind = Block(int(self.block[p]))
            entry: dict = {"pos": list(p), "type": kind.name.lower()}
            if kind in (Block.TORCH, Block.REPEATER, Block.LEVER, Block.COMPARATOR):
                entry["facing"] = FACING_NAMES[int(self.facing[p])]
            if kind == Block.REPEATER:
                entry["delay"] = int(self.delay[p])
            if kind == Block.LEVER:
                entry["on"] = bool(self.state[p])
            if kind == Block.COMPARATOR:
                entry["mode"] = "subtract" if self.subtract[p] else "compare"
            label = next((n for n, q in self.labels.items() if q == p), None)
            if label:
                entry["label"] = label
            if include_state:
                entry["active"] = self.powered(p) if kind == Block.SOLID else bool(self.state[p])
                if kind == Block.DUST:
                    entry["level"] = int(self.dust[p])
                    entry["shape"] = self.dust_shape(p)
                if kind == Block.REPEATER and self.locked[p]:
                    entry["locked"] = True
                if kind == Block.COMPARATOR:
                    entry["output"] = int(self.output[p])
            blocks.append(entry)
        return {"size": list(self.size), "ticks": self.ticks, "blocks": blocks}

    @classmethod
    def from_dict(cls, data: dict, settle: bool = True) -> "Circuit":
        c = cls(tuple(data["size"]))
        for b in data["blocks"]:
            kind = Block[b["type"].upper()]
            facing = FACING_NAMES.index(b["facing"]) if "facing" in b else None
            c.place(tuple(b["pos"]), kind, facing=facing, delay=b.get("delay", 1),
                    on=b.get("on", False), label=b.get("label"),
                    subtract=b.get("mode") == "subtract")
        if settle:
            c.settle()
        return c


sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
