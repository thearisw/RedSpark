import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from redspark.env import PLACEABLE, RedstoneGateEnv  # noqa: E402
from redspark.sim import Block  # noqa: E402


def action_for(env, pos, kind, facing=0):
    import numpy as np
    cell = int(np.ravel_multi_index(pos, env.size))
    return cell * len(PLACEABLE) + PLACEABLE.index((kind, facing))


def test_env_passes_gymnasium_checks():
    check_env(RedstoneGateEnv("NOT"), skip_render_check=True)
    check_env(RedstoneGateEnv("XOR"), skip_render_check=True)


def test_building_a_not_gate_gets_full_reward():
    env = RedstoneGateEnv("NOT")
    env.reset(seed=0)
    total = 0.0
    # lever -> dust -> block -> torch on the block -> lamp
    for pos, kind, facing in [((1, 0, 1), Block.DUST, 0), ((2, 0, 1), Block.SOLID, 0),
                              ((3, 0, 1), Block.TORCH, 1)]:
        _, r, terminated, _, info = env.step(action_for(env, pos, kind, facing))
        total += r
    assert terminated and info["solved"]
    # score goes 0.5 (empty grid: lamp always off) -> 1.0, plus solve bonus, minus 3 blocks
    assert total == pytest.approx((1.0 - 0.5) + 1.0 - 0.03)


def test_invalid_moves_are_penalised():
    env = RedstoneGateEnv("NOT")
    env.reset(seed=0)
    _, r, *_ = env.step(action_for(env, (2, 0, 1), Block.TORCH, 1))  # no block to attach to
    assert r < 0
    mask = env.action_masks()
    assert not mask[action_for(env, (2, 0, 1), Block.TORCH, 1)]
    assert mask[action_for(env, (1, 0, 1), Block.SOLID)]
    assert not mask[action_for(env, (1, 0, 1), Block.TORCH, 1)]  # would hang on the lever
    _, r, *_ = env.step(action_for(env, (0, 0, 1), Block.SOLID))  # overwrite the input lever
    assert r < 0


def test_comparators_are_opt_in():
    plain = RedstoneGateEnv("NOT")
    with_cmp = RedstoneGateEnv("NOT", allow_comparators=True)
    assert with_cmp.action_space.n == plain.action_space.n + plain.n_cells * 4
    check_env(with_cmp, skip_render_check=True)
    with_cmp.reset(seed=0)
    cell = 1 * 3 + 1  # (1, 0, 1)
    a = cell * len(with_cmp.placeable) + with_cmp.placeable.index((Block.COMPARATOR, 0))
    with_cmp.step(a)
    assert with_cmp.circuit.get((1, 0, 1)) == Block.COMPARATOR


def test_mask_needs_a_block_beneath_parts_on_upper_layers():
    env = RedstoneGateEnv("NOT", size=(5, 2, 3))
    env.reset(seed=0)
    for kind, facing in [(Block.DUST, 0), (Block.REPEATER, 0), (Block.TORCH, 5)]:
        assert not env.action_masks()[action_for(env, (2, 1, 1), kind, facing)]
    env.step(action_for(env, (2, 0, 1), Block.SOLID))
    for kind, facing in [(Block.DUST, 0), (Block.REPEATER, 0), (Block.TORCH, 5)]:
        assert env.action_masks()[action_for(env, (2, 1, 1), kind, facing)]
    env.step(action_for(env, (2, 1, 1), Block.DUST))
    env.step(action_for(env, (2, 0, 1), Block.AIR))          # clearing the block drops the dust
    assert env.circuit.get((2, 1, 1)) == Block.AIR


def test_curriculum_stages_share_spaces():
    """With n_inputs=2 every gate uses one grid, so a curriculum can reuse the network."""
    envs = [RedstoneGateEnv(g, n_inputs=2) for g in ("NOT", "BUFFER", "OR", "AND", "XOR")]
    for env in envs:
        assert env.observation_space == envs[0].observation_space
        assert env.action_space == envs[0].action_space
        assert set(env.inputs) == {"A", "B"}
    check_env(envs[0], skip_render_check=True)


def test_not_gate_on_two_lever_grid_ignores_b():
    env = RedstoneGateEnv("NOT", n_inputs=2)  # 5 x 1 x 5, A at z=0, B at z=4, OUT at z=2
    env.reset(seed=0)
    # lever A -> dust -> block -> torch on its far side (facing points back at the block) -> dust -> lamp
    steps = [((1, 0, 0), Block.DUST, 0), ((2, 0, 0), Block.SOLID, 0), ((3, 0, 0), Block.TORCH, 1),
             ((4, 0, 0), Block.DUST, 0), ((4, 0, 1), Block.DUST, 0)]
    info = {}
    for pos, kind, facing in steps:
        _, _, terminated, _, info = env.step(action_for(env, pos, kind, facing))
        if terminated:
            break
    assert info["solved"]


def test_n_inputs_must_cover_the_gate():
    with pytest.raises(ValueError):
        RedstoneGateEnv("AND", n_inputs=1)
