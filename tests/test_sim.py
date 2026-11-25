import json
from pathlib import Path

import pytest

from redspark.sim import (Block, Circuit, DOWN, EAST, EXTREMELY_HIGH, HIGH, NORTH, VERY_HIGH,
                          WEST)
from redspark.tasks import evaluate_gate

CIRCUITS = Path(__file__).resolve().parents[1] / "circuits"


def load(name: str) -> Circuit:
    return Circuit.from_dict(json.loads((CIRCUITS / f"{name}.json").read_text()))


def lamp(c: Circuit) -> bool:
    return bool(c.state[c.labels["OUT"]])


def line_of_dust(n: int) -> Circuit:
    c = Circuit((n + 1, 1, 1))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN, on=True)
    for x in range(1, n + 1):
        c.place((x, 0, 0), Block.DUST)
    return c


# ------------------------------------------------------------------- dust
def test_lever_powers_dust_with_decay_and_turns_off_again():
    c = line_of_dust(4)
    assert [int(c.dust[x, 0, 0]) for x in range(1, 5)] == [15, 14, 13, 12]
    c.toggle((0, 0, 0))
    assert c.dust.sum() == 0


def test_dust_runs_out_after_15_blocks():
    c = line_of_dust(17)
    assert c.dust[15, 0, 0] == 1
    assert c.dust[16, 0, 0] == 0


def test_dust_climbs_one_step():
    c = Circuit((3, 2, 1))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN, on=True)
    c.place((1, 0, 0), Block.DUST)
    c.place((2, 0, 0), Block.SOLID)
    c.place((2, 1, 0), Block.DUST)
    assert c.dust[2, 1, 0] == 14


def test_dust_shapes():
    c = Circuit((3, 1, 3))
    c.place((1, 0, 1), Block.DUST)
    assert c.dust_shape((1, 0, 1)) == 0b1111      # lone dot acts as a cross
    c.place((0, 0, 1), Block.DUST)
    assert c.dust_shape((1, 0, 1)) == 0b0011      # one connection: straight line
    c.place((1, 0, 0), Block.DUST)
    assert c.dust_shape((1, 0, 1)) == 0b1010      # corner: -x and -z


def test_dust_only_powers_what_it_points_at():
    # A lamp beside a dust line is not lit; a lamp at the end of the line is.
    c = Circuit((3, 1, 3))
    c.place((0, 0, 1), Block.LEVER, facing=DOWN, on=True)
    c.place((1, 0, 1), Block.DUST)
    c.place((2, 0, 1), Block.LAMP)        # end of the line
    c.place((1, 0, 0), Block.LAMP)        # beside the line
    assert c.state[2, 0, 1]
    assert not c.state[1, 0, 0]


def test_weak_power_does_not_feed_dust():
    c = Circuit((4, 1, 1))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN, on=True)
    c.place((1, 0, 0), Block.DUST)
    c.place((2, 0, 0), Block.SOLID)
    c.place((3, 0, 0), Block.DUST)
    assert c.powered((2, 0, 0)) and not c.strongly_powered((2, 0, 0))
    assert c.dust[3, 0, 0] == 0


def test_lever_strongly_powers_only_its_own_block():
    c = Circuit((3, 1, 3))
    c.place((1, 0, 1), Block.SOLID)
    c.place((1, 0, 0), Block.SOLID)
    c.place((0, 0, 1), Block.LEVER, facing=EAST, on=True)   # attached to (1,0,1)
    assert c.strongly_powered((1, 0, 1))
    c2 = Circuit((3, 1, 1))
    c2.place((1, 0, 0), Block.SOLID)
    c2.place((0, 0, 0), Block.LEVER, facing=DOWN, on=True)  # on the floor beside it
    assert not c2.powered((1, 0, 0))


# ------------------------------------------------------------ timing (game ticks)
def test_torch_takes_2_game_ticks_and_lamp_4_more_to_turn_off():
    c = load("not_gate")
    assert lamp(c)
    c.set_lever(c.labels["A"], True)
    c.step()
    assert c.state[2, 0, 1]          # torch still lit after 1 gt
    c.step()
    assert not c.state[2, 0, 1]      # off after 2 gt (1 redstone tick)
    c.step(3)
    assert lamp(c)                   # lamp waits 4 gt before turning off
    c.step()
    assert not lamp(c)
    c.set_lever(c.labels["A"], False)
    c.step(2)
    assert lamp(c)                   # lamps turn on instantly


def test_repeater_delays_add_up_in_game_ticks():
    c = load("repeater_line")        # delays 2 + 3 redstone ticks = 10 game ticks
    c.set_lever(c.labels["A"], True)
    c.step(9)
    assert not lamp(c)
    c.step()
    assert lamp(c)


def test_repeater_extends_short_pulses():
    c = Circuit((4, 1, 1))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN)
    c.place((1, 0, 0), Block.DUST)
    c.place((2, 0, 0), Block.REPEATER, facing=EAST, delay=4)
    c.place((3, 0, 0), Block.LAMP)
    c.toggle((0, 0, 0))
    c.step(2)
    c.toggle((0, 0, 0))              # a 2 gt pulse
    c.step(6)
    assert c.state[2, 0, 0]          # repeater turned on at 8 gt
    c.step(7)
    assert c.state[2, 0, 0]          # and stays on for its full delay
    c.step(1)
    assert not c.state[2, 0, 0]


def test_repeater_is_one_way_and_can_be_locked():
    c = Circuit((3, 1, 1))
    c.place((2, 0, 0), Block.LEVER, facing=DOWN, on=True)
    c.place((1, 0, 0), Block.REPEATER, facing=EAST)  # points at the lever
    c.place((0, 0, 0), Block.LAMP)
    c.step(5)
    assert not c.state[0, 0, 0]

    c2 = Circuit((3, 1, 3))
    c2.place((1, 0, 2), Block.LEVER, facing=DOWN, on=True)
    c2.place((1, 0, 1), Block.REPEATER, facing=NORTH)       # side repeater, powered
    c2.place((0, 0, 0), Block.LEVER, facing=DOWN)
    c2.place((1, 0, 0), Block.REPEATER, facing=EAST)        # locked by the one below it
    c2.settle()
    assert c2.locked[1, 0, 0]
    c2.toggle((0, 0, 0))
    c2.step(10)
    assert not c2.state[1, 0, 0]


def test_tile_tick_priorities():
    c = Circuit((4, 1, 1))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN)
    c.place((1, 0, 0), Block.REPEATER, facing=EAST)
    c.place((2, 0, 0), Block.REPEATER, facing=EAST)
    c.toggle((0, 0, 0))
    # the first repeater faces another repeater: highest priority
    assert [q[1] for q in c._queue] == [EXTREMELY_HIGH]
    c.step(2)
    assert [q[1] for q in c._queue] == [HIGH]                # second turns on: -1
    c.step(2)
    c.toggle((0, 0, 0))
    assert [q[1] for q in c._queue] == [EXTREMELY_HIGH]
    c.step(2)
    assert [q[1] for q in c._queue] == [VERY_HIGH]           # second turns off: -2


def test_torch_burns_out_and_recovers():
    c = load("torch_clock")
    c.place((1, 0, 1), Block.REPEATER, facing=NORTH, delay=1)  # make it too fast
    states = []
    for _ in range(100):
        c.step()
        states.append(bool(c.state[2, 0, 0]))
    assert not any(states[-30:])                                 # burnt out
    later = []
    for _ in range(160):
        c.step()
        later.append(bool(c.state[2, 0, 0]))
    assert any(later)                                            # relit after its cooldown


def test_quasi_connectivity_helper():
    c = Circuit((2, 3, 1))
    c.place((1, 1, 0), Block.LEVER, facing=DOWN, on=True)   # next to the cell above (0,0,0)
    assert not c.has_neighbor_signal((0, 0, 0))
    assert c.has_neighbor_signal_qc((0, 0, 0))


# ------------------------------------------------------------- whole circuits
def test_torch_clock_is_flagged_unstable():
    c = load("torch_clock")
    states = set()
    for _ in range(100):
        c.step()
        states.add(bool(c.state[2, 0, 0]))
    assert states == {True, False}
    assert not evaluate_gate(c, "BUFFER").stable


@pytest.mark.parametrize("name,gate", [
    ("not_gate", "NOT"),
    ("tower_not", "NOT"),
    ("or_gate", "OR"),
    ("and_gate", "AND"),
    ("repeater_line", "BUFFER"),
    ("comparator_and_not", "ANDNOT"),
])
def test_example_circuits_solve_their_gate(name, gate):
    result = evaluate_gate(load(name), gate)
    assert result.solved, result.rows


def test_wrong_gate_scores_partially():
    result = evaluate_gate(load("or_gate"), "AND")
    assert result.score == 0.5 and not result.solved


def test_round_trip_serialisation():
    c = load("and_gate")
    again = Circuit.from_dict(c.to_dict())
    assert (again.block == c.block).all() and (again.facing == c.facing).all() and again.labels == c.labels


# ------------------------------------------------------------ placement rules
@pytest.mark.parametrize("support,ok", [
    (Block.SOLID, True), (Block.LAMP, True), (Block.DUST, False),
    (Block.REPEATER, False), (Block.COMPARATOR, False), (Block.AIR, False),
])
def test_wall_torches_need_a_solid_block_or_lamp(support, ok):
    c = Circuit((3, 1, 1))
    if support != Block.AIR:
        c.place((1, 0, 0), support)
    assert c.can_place((2, 0, 0), Block.TORCH, WEST) is ok
    if ok:
        c.place((2, 0, 0), Block.TORCH, facing=WEST)
    else:
        with pytest.raises(ValueError):
            c.place((2, 0, 0), Block.TORCH, facing=WEST)


def test_standing_torches_need_a_block_below_or_the_floor():
    c = Circuit((1, 3, 1))
    assert c.can_place((0, 0, 0), Block.TORCH, DOWN)        # on the floor
    assert not c.can_place((0, 1, 0), Block.TORCH, DOWN)    # on air
    c.place((0, 0, 0), Block.LAMP)
    assert c.can_place((0, 1, 0), Block.TORCH, DOWN)
    assert not c.can_place((0, 1, 0), Block.TORCH, 4)       # never hangs from above


def test_torch_breaks_off_when_its_block_is_removed():
    c = load("not_gate")
    c.remove((1, 0, 1))                                    # the block holding the torch
    assert c.get((2, 0, 1)) == Block.AIR
    c2 = load("not_gate")
    c2.place((1, 0, 1), Block.DUST)                        # replaced by something non-solid
    assert c2.get((2, 0, 1)) == Block.AIR


@pytest.mark.parametrize("kind", [Block.DUST, Block.REPEATER, Block.COMPARATOR, Block.TORCH])
@pytest.mark.parametrize("below,ok", [
    (Block.SOLID, True), (Block.LAMP, True), (Block.AIR, False), (Block.DUST, False),
    (Block.REPEATER, False), (Block.COMPARATOR, False),
])
def test_parts_need_a_block_beneath_them(kind, below, ok):
    c = Circuit((1, 3, 1))
    assert c.can_place((0, 0, 0), kind, DOWN if kind == Block.TORCH else EAST)  # the floor holds them
    if below != Block.AIR:
        c.place((0, 0, 0), below, facing=EAST if below in (Block.REPEATER, Block.COMPARATOR) else None)
    facing = DOWN if kind == Block.TORCH else EAST
    assert c.can_place((0, 1, 0), kind, facing) is ok
    if ok:
        c.place((0, 1, 0), kind, facing=facing)
        assert c.get((0, 1, 0)) == kind
    else:
        with pytest.raises(ValueError):
            c.place((0, 1, 0), kind, facing=facing)


@pytest.mark.parametrize("kind", [Block.DUST, Block.REPEATER, Block.COMPARATOR, Block.TORCH])
def test_parts_break_off_when_the_block_beneath_goes(kind):
    facing = DOWN if kind == Block.TORCH else EAST
    c = Circuit((1, 2, 1))
    c.place((0, 0, 0), Block.SOLID)
    c.place((0, 1, 0), kind, facing=facing)
    c.place((0, 0, 0), Block.LAMP)                 # another support: the part stays
    assert c.get((0, 1, 0)) == kind
    c.remove((0, 0, 0))
    assert c.get((0, 1, 0)) == Block.AIR
    c.place((0, 0, 0), Block.SOLID)
    c.place((0, 1, 0), kind, facing=facing)
    c.place((0, 0, 0), Block.DUST)                 # replaced by something that cannot hold it
    assert c.get((0, 1, 0)) == Block.AIR


def test_breaking_cascades_up_a_tower():
    c = Circuit((3, 3, 1))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN, on=True)
    c.place((1, 0, 0), Block.SOLID)
    c.place((1, 1, 0), Block.SOLID)
    c.place((1, 2, 0), Block.DUST)
    c.place((2, 1, 0), Block.TORCH, facing=WEST)   # wall torch on the upper block
    c.remove((1, 0, 0))                            # the upper block floats, so it still holds
    assert c.get((1, 2, 0)) == Block.DUST and c.get((2, 1, 0)) == Block.TORCH
    c.remove((1, 1, 0))
    assert c.get((1, 2, 0)) == Block.AIR and c.get((2, 1, 0)) == Block.AIR


def test_broken_dust_stops_carrying_power():
    c = Circuit((4, 2, 1))
    c.place((0, 0, 0), Block.LEVER, facing=DOWN, on=True)
    c.place((1, 0, 0), Block.DUST)
    c.place((2, 0, 0), Block.SOLID)
    c.place((2, 1, 0), Block.DUST)
    c.place((3, 0, 0), Block.LAMP, label="OUT")
    c.place((3, 1, 0), Block.DUST)
    c.settle()
    assert c.dust[3, 1, 0] > 0
    c.remove((2, 0, 0))
    c.settle()
    assert c.get((2, 1, 0)) == Block.AIR and c.dust[3, 1, 0] == 0


def test_saved_circuits_load_whatever_the_block_order():
    c = Circuit((2, 3, 1))
    c.place((0, 0, 0), Block.SOLID)
    c.place((0, 1, 0), Block.SOLID)
    c.place((0, 2, 0), Block.REPEATER, facing=EAST, delay=3)
    c.place((1, 1, 0), Block.TORCH, facing=WEST)
    data = c.to_dict(include_state=False)
    data["blocks"].reverse()                       # parts listed before their supports
    again = Circuit.from_dict(data)
    assert (again.block == c.block).all() and again.delay[0, 2, 0] == 3
