from redspark.sim import Block, Circuit, DOWN, EAST, HIGH, NORMAL, NORTH, SOUTH, WEST


def rear_line(n_dust: int, width: int = 8, depth: int = 3) -> Circuit:
    """Lever -> n dust -> comparator at x = n + 1 facing east, on row z = 1."""
    c = Circuit((width, 1, depth))
    c.place((0, 0, 1), Block.LEVER, facing=DOWN)
    for x in range(1, n_dust + 1):
        c.place((x, 0, 1), Block.DUST)
    c.place((n_dust + 1, 0, 1), Block.COMPARATOR, facing=EAST)
    return c


def cmp_pos(c: Circuit) -> tuple:
    return next(tuple(int(v) for v in p) for p in zip(*((c.block == Block.COMPARATOR).nonzero())))


def test_compare_mode_passes_rear_strength_after_2_gt():
    c = rear_line(3)                       # dust levels 15, 14, 13
    p = cmp_pos(c)
    c.toggle((0, 0, 1))
    c.step()
    assert c.output[p] == 0
    c.step()
    assert c.output[p] == 13 and c.state[p]


def test_output_strength_carries_into_dust():
    c = rear_line(3)
    c.place((5, 0, 1), Block.DUST)
    c.place((6, 0, 1), Block.DUST)
    c.toggle((0, 0, 1))
    c.settle()
    assert c.dust[5, 0, 1] == 13 and c.dust[6, 0, 1] == 12


def side_lever_line(c: Circuit, p, side_level_dust: int) -> None:
    """Feed the comparator's north side with dust at a chosen level."""
    # lever at the far end of a dust run along z = 0 that ends beside the comparator
    x = p[0]
    c.place((x, 0, 0), Block.DUST)
    run = 15 - side_level_dust  # extra dust between lever and the side dust
    for i in range(1, run + 1):
        c.place((x + i, 0, 0), Block.DUST)
    c.place((x + run + 1, 0, 0), Block.LEVER, facing=DOWN, on=True)


def test_side_input_stronger_turns_compare_mode_off():
    c = rear_line(3, width=12)             # rear 13
    p = cmp_pos(c)
    c.toggle((0, 0, 1))
    c.settle()
    assert c.output[p] == 13
    side_lever_line(c, p, side_level_dust=14)   # side 14 > rear 13
    c.settle()
    assert c.output[p] == 0 and not c.state[p]


def test_equal_side_keeps_compare_mode_on():
    c = rear_line(3, width=12)
    p = cmp_pos(c)
    c.toggle((0, 0, 1))
    side_lever_line(c, p, side_level_dust=13)
    c.settle()
    assert c.output[p] == 13 and c.state[p]


def test_subtract_mode_and_instant_mode_switch():
    c = rear_line(1, width=16)             # rear 15
    p = cmp_pos(c)
    c.toggle((0, 0, 1))
    side_lever_line(c, p, side_level_dust=4)
    c.settle()
    assert c.output[p] == 15               # compare: 15 >= 4
    c.set_comparator_mode(p, True)
    assert c.output[p] == 11               # subtract applies at once
    assert c.dust[p[0], 0, 0] == 4


def subtracting_comparator() -> Circuit:
    """Rear 15 into a subtract-mode comparator at (2, 0, 1) facing east."""
    c = Circuit((5, 1, 4))
    c.place((0, 0, 1), Block.LEVER, facing=DOWN, on=True)
    c.place((1, 0, 1), Block.DUST)
    c.place((2, 0, 1), Block.COMPARATOR, facing=EAST, subtract=True)
    return c


def test_powered_blocks_and_torches_are_not_side_inputs():
    c = subtracting_comparator()
    c.place((2, 0, 0), Block.SOLID)
    c.place((3, 0, 0), Block.LEVER, facing=WEST, on=True)   # strongly powers that block
    c.place((2, 0, 3), Block.SOLID)
    c.place((2, 0, 2), Block.TORCH, facing=SOUTH)           # lit torch beside the comparator
    c.settle()
    assert c.strongly_powered((2, 0, 0)) and c.state[2, 0, 2]
    assert c.output[2, 0, 1] == 15


def test_a_repeater_into_the_side_counts():
    c = subtracting_comparator()
    c.place((2, 0, 3), Block.LEVER, facing=DOWN, on=True)
    c.place((2, 0, 2), Block.REPEATER, facing=NORTH)
    c.settle()
    assert c.output[2, 0, 1] == 0            # 15 - 15


def test_priority_depends_on_what_it_faces():
    c = rear_line(1)
    p = cmp_pos(c)
    c.toggle((0, 0, 1))
    assert [q[1] for q in c._queue] == [NORMAL]
    c2 = rear_line(1)
    c2.place((cmp_pos(c2)[0] + 1, 0, 1), Block.REPEATER, facing=EAST)
    c2.toggle((0, 0, 1))
    assert [q[1] for q in c2._queue] == [HIGH]


def test_comparator_locks_a_repeater_from_the_side():
    c = Circuit((3, 1, 3))
    c.place((1, 0, 2), Block.LEVER, facing=DOWN, on=True)
    c.place((1, 0, 1), Block.COMPARATOR, facing=NORTH)
    c.place((0, 0, 0), Block.LEVER, facing=DOWN)
    c.place((1, 0, 0), Block.REPEATER, facing=EAST)
    c.settle()
    assert c.locked[1, 0, 0]


def test_comparator_update_hook_rechecks_comparators():
    c = rear_line(1)
    p = cmp_pos(c)
    c.toggle((0, 0, 1))
    c.settle()
    c.output[p] = 3                        # pretend a container changed what it reads
    c.comparator_update((p[0] - 1, 0, 1))
    assert c.has_scheduled_tick(p)
    c.settle()
    assert c.output[p] == 15


def test_round_trip_keeps_mode():
    c = rear_line(1)
    c.set_comparator_mode(cmp_pos(c), True)
    again = Circuit.from_dict(c.to_dict())
    assert again.subtract[cmp_pos(again)]
