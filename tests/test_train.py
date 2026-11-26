import pytest

pytest.importorskip("sb3_contrib")

from redspark.train import DEFAULT_CURRICULUM, StopWhenSolved, train_curriculum  # noqa: E402


def feed(cb, solved_flags):
    """Pretend each flag is one finished episode and return what the callback said."""
    keep_going = True
    for s in solved_flags:
        cb.locals = {"dones": [True], "infos": [{"solved": s}]}
        keep_going = cb._on_step()
    return keep_going


def test_stop_when_solved_waits_for_a_full_window():
    cb = StopWhenSolved(rate=0.9, window=10)
    assert feed(cb, [True] * 9)          # only 9 episodes seen: keep training
    assert not feed(cb, [True])          # 10/10 solved: stop
    assert cb.solve_rate == 1.0


def test_stop_when_solved_uses_recent_episodes_only():
    cb = StopWhenSolved(rate=0.9, window=10)
    assert feed(cb, [False] * 10 + [True] * 8)   # last 10: 8 solved
    assert not feed(cb, [True])                   # last 10: 9 solved
    assert cb.episodes == 19


def test_default_curriculum_order():
    assert DEFAULT_CURRICULUM == ["NOT", "BUFFER", "OR", "AND", "XOR"]


def test_curriculum_runs_every_stage_and_moves_on_early(tmp_path):
    # advance_at=0 means "move on as soon as the 100-episode window is full",
    # which with 4-step episodes is far below the 5000-step budget.
    summary = train_curriculum(["NOT", "BUFFER"], timesteps=5000, envs=1, max_steps=4, seed=0,
                               comparators=False, advance_at=0.0, verbose=False, root=tmp_path)
    assert [s["gate"] for s in summary] == ["NOT", "BUFFER"]
    for s in summary:
        assert s["steps"] < 5000
        assert (tmp_path / "runs" / "curriculum" / f"{s['gate'].lower()}.zip").exists()
        assert (tmp_path / "circuits" / f"curriculum_{s['gate'].lower()}.json").exists()
        assert s["rows"] in (2, 4)
