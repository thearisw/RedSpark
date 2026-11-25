import numpy as np
import pytest

from redspark.designer import PLACEABLE, Builder, Trainer, TruthTableTask, evaluate
from redspark.sim import Block


def not_task(**extra):
    return TruthTableTask.from_dict({
        "size": [5, 1, 3],
        "inputs": [{"label": "A", "pos": [0, 0, 1]}],
        "outputs": [{"label": "OUT1", "pos": [4, 0, 1]}],
        "rows": [{"in": [False], "out": [True]}, {"in": [True], "out": [False]}],
        **extra,
    })


def test_hand_built_not_passes_and_blank_grid_passes_half():
    task = not_task()
    c = task.blank_circuit()
    assert evaluate(c, task).rows_passed == 1  # lamp always off
    # The lever sits on the floor: it powers the dust beside it, which powers the block.
    c.place((1, 0, 1), Block.DUST)
    c.place((2, 0, 1), Block.SOLID)
    c.place((3, 0, 1), Block.TORCH, facing=1)
    res = evaluate(c, task, frames=True)
    assert res.passed == [True, True] and res.stable
    assert len(res.frames) == 2 and res.reach == [1.0]


def test_dont_care_outputs_always_pass():
    task = TruthTableTask.from_dict({
        "size": [5, 1, 5],
        "inputs": [{"label": "A", "pos": [0, 0, 0]}, {"label": "B", "pos": [0, 0, 4]}],
        "outputs": [{"label": "OUT1", "pos": [4, 0, 2]}, {"label": "OUT2", "pos": [2, 0, 2]}],
        "rows": [{"in": [a, b], "out": [None, False]} for a in (False, True) for b in (False, True)],
    })
    assert evaluate(task.blank_circuit(), task).rows_passed == 4


@pytest.mark.parametrize("bad", [
    {"inputs": []},
    {"outputs": [{"label": "OUT1", "pos": [0, 0, 1]}]},          # same cell as the input
    {"outputs": [{"label": "OUT1", "pos": [9, 0, 1]}]},          # outside the grid
    {"rows": [{"in": [False], "out": [None]}]},                  # nothing to learn
    {"rows": [{"in": [False, True], "out": [True]}]},            # wrong width
])
def test_bad_tasks_are_rejected(bad):
    with pytest.raises(ValueError):
        not_task(**bad)


def test_mask_matches_placement_rules():
    task = not_task()
    b = Builder(task)
    c = task.blank_circuit()

    def action(pos, kind, facing=0):
        cell = int(np.ravel_multi_index(pos, task.size))
        return cell * len(PLACEABLE) + PLACEABLE.index((kind, facing))

    mask = b.mask(c)
    assert mask[-1]                                           # stop is always allowed
    assert not mask[action((0, 0, 1), Block.SOLID)]           # input cell is fixed
    assert not mask[action((2, 0, 1), Block.TORCH, 1)]        # nothing to hang on
    assert not mask[action((1, 0, 1), Block.AIR)]             # already empty
    c.place((1, 0, 1), Block.SOLID)
    mask = b.mask(c)
    assert mask[action((2, 0, 1), Block.TORCH, 1)]            # torch on the -x side of the block
    assert not mask[action((2, 0, 1), Block.TORCH, 0)]
    assert b.decode(action((2, 0, 1), Block.TORCH, 1)) == ((2, 0, 1), Block.TORCH, 1)


def test_mask_needs_a_block_beneath_dust_repeaters_and_standing_torches():
    task = TruthTableTask.from_dict({
        "size": [5, 2, 3],
        "inputs": [{"label": "A", "pos": [0, 0, 1]}],
        "outputs": [{"label": "OUT1", "pos": [4, 0, 1]}],
        "rows": [{"in": [False], "out": [True]}, {"in": [True], "out": [False]}],
    })
    b = Builder(task)
    c = task.blank_circuit()

    def action(pos, kind, facing=0):
        cell = int(np.ravel_multi_index(pos, task.size))
        return cell * len(PLACEABLE) + PLACEABLE.index((kind, facing))

    upper = [((2, 1, 1), Block.DUST, 0), ((2, 1, 1), Block.REPEATER, 0), ((2, 1, 1), Block.TORCH, 5)]
    mask = b.mask(c)
    for pos, kind, facing in upper:
        assert not mask[action(pos, kind, facing)]            # nothing under them
        assert mask[action((2, 0, 1), kind, facing)]          # the floor holds them
    for support in (Block.SOLID, Block.LAMP):
        c.place((2, 0, 1), support)
        mask = b.mask(c)
        for pos, kind, facing in upper:
            assert mask[action(pos, kind, facing)]
    c.place((2, 0, 1), Block.DUST)
    mask = b.mask(c)
    for pos, kind, facing in upper:
        assert not mask[action(pos, kind, facing)]
    # The mask agrees with the simulator everywhere.
    c.place((1, 0, 1), Block.SOLID)
    mask = b.mask(c)
    for a in range(b.n_actions - 1):
        pos, kind, facing = b.decode(a)
        if pos not in task.fixed and kind != Block.AIR and mask[a]:
            assert c.can_place(pos, kind, facing)


def test_trainer_learns_a_not_gate():
    # Any single run can stall, so try a few seeds; most solve NOT within a few dozen generations.
    for seed in range(5):
        trainer = Trainer(not_task(), seed=seed)
        stats = [trainer.generation() for _ in range(80)]
        if trainer.best.result.rows_passed == 2:
            break
    assert max(s.best_rows for s in stats) == 2
    assert stats[-1].gen == 79 and trainer.gen == 80


def test_ranking_prefers_correct_then_fewer_blocks_then_smaller_box():
    task = TruthTableTask.from_dict({
        "size": [7, 1, 3],
        "inputs": [{"label": "A", "pos": [0, 0, 1]}],
        "outputs": [{"label": "OUT1", "pos": [4, 0, 1]}],
        "rows": [{"in": [False], "out": [True]}, {"in": [True], "out": [False]}],
    })
    trainer = Trainer(task, seed=0)

    def key(extra):
        c = task.blank_circuit()
        for pos, kind, facing in extra:
            c.place(pos, kind, facing=facing)
        return trainer.fitness(c)[0]

    nott = [((1, 0, 1), Block.DUST, 0), ((2, 0, 1), Block.SOLID, 0), ((3, 0, 1), Block.TORCH, 1)]
    wrong_small = key([((1, 0, 1), Block.DUST, 0)])
    correct = key(nott)
    padded_near = key(nott + [((2, 0, 0), Block.SOLID, 0)])   # same box, one more block
    padded_far = key(nott + [((6, 0, 0), Block.SOLID, 0)])    # same blocks, bigger box (7x1x2 vs 5x1x2)
    assert correct > padded_near > wrong_small
    assert padded_near > padded_far


def copy_b_task():
    return TruthTableTask.from_dict({
        "size": [5, 1, 5],
        "inputs": [{"label": "A", "pos": [0, 0, 0]}, {"label": "B", "pos": [0, 0, 4]}],
        "outputs": [{"label": "OUT1", "pos": [4, 0, 2]}],
        "rows": [{"in": [a, b], "out": [b]} for a in (False, True) for b in (False, True)],
    })


def test_matching_signal_hint_points_at_the_right_input():
    task = copy_b_task()
    c = task.blank_circuit()
    blank = evaluate(c, task).match[0]
    c.place((1, 0, 4), Block.DUST)
    c.place((2, 0, 4), Block.DUST)
    closer = evaluate(c, task).match[0]
    assert 0 < blank < closer < 1        # B's signal grows towards the lamp
    c2 = task.blank_circuit()
    for x in (1, 2, 3):
        c2.place((x, 0, 0), Block.DUST)  # A's signal is wrong in half the rows
    c3 = task.blank_circuit()
    for x in (1, 2, 3):
        c3.place((x, 0, 4), Block.DUST)  # B's signal, just as far along
    assert evaluate(c2, task).match[0] < evaluate(c3, task).match[0]


def test_opposite_signal_counts_as_one_torch_away():
    # For AND, a NAND signal next to the lamp is worth more than a lone A wire there.
    from redspark.designer import matching_reach
    import numpy as np
    task = TruthTableTask.from_dict({
        "size": [3, 1, 1], "inputs": [{"label": "A", "pos": [0, 0, 0]}],
        "outputs": [{"label": "OUT1", "pos": [2, 0, 0]}],
        "rows": [{"in": [a], "out": [a]} for a in (False, True)],
    })
    maps = [np.zeros((3, 1, 1), bool), np.zeros((3, 1, 1), bool)]
    maps[0][1, 0, 0] = True          # cell next to the lamp carries NOT A
    opposite = matching_reach(task, maps)[0]
    maps = [np.zeros((3, 1, 1), bool), np.zeros((3, 1, 1), bool)]
    maps[1][1, 0, 0] = True          # ...or exactly A
    exact = matching_reach(task, maps)[0]
    assert 0 < opposite < exact


def test_stuck_population_starts_over_but_keeps_the_best():
    trainer = Trainer(copy_b_task(), seed=1, patience=5)
    stats = [trainer.generation() for _ in range(40)]
    assert any(s.restarted for s in stats) and trainer.restarts >= 1
    assert trainer.best.result.rows_passed == max(s.best_rows for s in stats)


def test_faster_circuit_wins_when_blocks_and_size_tie():
    task = TruthTableTask.from_dict({
        "size": [6, 1, 3],
        "inputs": [{"label": "A", "pos": [0, 0, 1]}],
        "outputs": [{"label": "OUT1", "pos": [5, 0, 1]}],
        "rows": [{"in": [False], "out": [True]}, {"in": [True], "out": [False]}],
    })
    trainer = Trainer(task, seed=0, workers=0)

    def score(last):
        c = task.blank_circuit()
        for pos, kind, facing in [((1, 0, 1), Block.DUST, 0), ((2, 0, 1), Block.SOLID, 0),
                                  ((3, 0, 1), Block.TORCH, 1), last]:
            c.place(pos, kind, facing=facing)
        key, res, *_ = trainer.fitness(c)
        return key, res

    dust_key, dust = score(((4, 0, 1), Block.DUST, 0))
    rep_key, rep = score(((4, 0, 1), Block.REPEATER, 0))   # same blocks and box, +2 gt
    assert dust.rows_passed == rep.rows_passed == 2
    assert 0 < dust.delay < rep.delay
    assert dust_key > rep_key


def test_layout_trainer_solves_and_quickly():
    import itertools
    from redspark.designer import LayoutTrainer
    rows = [{"in": list(c), "out": [c[0] and c[1]]} for c in itertools.product([False, True], repeat=2)]
    task = TruthTableTask.from_dict({
        "size": [7, 1, 5], "inputs": [{"label": "A", "pos": [0, 0, 0]}, {"label": "B", "pos": [0, 0, 4]}],
        "outputs": [{"label": "OUT1", "pos": [6, 0, 2]}], "rows": rows})
    solved = False
    for seed in range(5):  # search is random: most seeds solve AND within a few hundred generations
        tr = LayoutTrainer(task, seed=seed, max_blocks=14, workers=0)
        try:
            for _ in range(400):
                if tr.generation().best_rows == 4:
                    break
            if tr.best.result.rows_passed == 4:
                assert tr.best.circuit.count_blocks() - len(task.fixed) <= 14
                solved = True
                break
        finally:
            tr.close()
    assert solved


def test_layout_trainer_places_comparators():
    from redspark.designer import LayoutTrainer
    tr = LayoutTrainer(copy_b_task(), seed=3, max_blocks=10, workers=0)
    seen = set()
    for _ in range(40):
        tr.generation()
        for ind in tr.pop:
            seen |= set(int(v) for v in np.unique(ind.lay.block))
    tr.close()
    assert int(Block.COMPARATOR) in seen
