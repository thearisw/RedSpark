"""Train a (maskable) PPO agent to build a logic gate.

    python -m redspark.train --gate NOT --timesteps 100000

Saves the model to runs/<gate>/ and the best circuit it built to
circuits/trained_<gate>.json, which the viewer can open.

Curriculum: train on a list of gates in turn, keeping the same network.

    python -m redspark.train --curriculum NOT,BUFFER,OR,AND,XOR --timesteps 60000

Every stage uses the same 5 x 1 x 5 grid with levers A and B (one-input gates
just ignore B), so the observation and action spaces match and the weights carry
over. ``--timesteps`` is the budget per stage; a stage ends early once the agent
solves ``--advance-at`` of its recent episodes. Each stage saves
runs/curriculum/<gate>.zip and circuits/curriculum_<gate>.json. ``--curriculum
default`` is the order above.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from .env import RedstoneGateEnv
from .tasks import GATES, evaluate_gate
from .sim import Circuit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRICULUM = ["NOT", "BUFFER", "OR", "AND", "XOR"]


class StopWhenSolved(BaseCallback):
    """Stop learning once at least ``rate`` of the last ``window`` episodes were solved."""

    def __init__(self, rate: float, window: int = 100):
        super().__init__()
        self.rate = rate
        self.recent: deque[bool] = deque(maxlen=window)
        self.episodes = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.recent.append(bool(info.get("solved")))
                self.episodes += 1
        return not (len(self.recent) == self.recent.maxlen and self.solve_rate >= self.rate)

    @property
    def solve_rate(self) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else 0.0


def rollout(model, env: RedstoneGateEnv, deterministic: bool = True) -> tuple[float, dict]:
    obs, _ = env.reset()
    total, done = 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        done = terminated or truncated
    return total, info


def best_circuit(model, env_kwargs: dict, prefix: str = "trained", tries: int = 20, root: Path = ROOT):
    """Keep the best of a greedy rollout and a few sampled ones; save and print it."""
    gate = env_kwargs["gate"]
    env = RedstoneGateEnv(**env_kwargs)
    reward, info = max((rollout(model, env, deterministic=(i == 0)) for i in range(tries)), key=lambda r: r[0])
    circuit = Circuit.from_dict(info["circuit"])
    result = evaluate_gate(circuit, gate)
    out = root / "circuits" / f"{prefix}_{gate.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info["circuit"], indent=1) + "\n")
    print(f"\n{gate}: best episode reward {reward:.3f}, truth table {result.correct}/{len(result.rows)}, "
          f"solved={result.solved}")
    env.circuit = circuit
    print(env.render())
    print(f"saved circuit to {out}")
    return result


def make_model(vec_env, seed: int) -> MaskablePPO:
    # MaskablePPO is PPO that only samples legal moves (see RedstoneGateEnv.action_masks).
    return MaskablePPO("MlpPolicy", vec_env, n_steps=256, batch_size=256, ent_coef=0.02,
                       learning_rate=3e-4, verbose=1, seed=seed)


def train_curriculum(gates: list[str], timesteps: int, envs: int, max_steps: int, seed: int,
                     comparators: bool, advance_at: float, verbose: bool = True,
                     root: Path = ROOT) -> list[dict]:
    """Train one network on each gate in turn. Returns one summary per stage.

    ``root`` is where runs/curriculum/ and circuits/ go (the repo by default).
    """
    run_dir = root / "runs" / "curriculum"
    run_dir.mkdir(parents=True, exist_ok=True)
    model = None
    summary = []
    for i, gate in enumerate(gates):
        env_kwargs = {"gate": gate, "max_steps": max_steps, "allow_comparators": comparators, "n_inputs": 2}
        vec_env = make_vec_env(RedstoneGateEnv, n_envs=envs, seed=seed + i, env_kwargs=env_kwargs)
        if model is None:
            model = make_model(vec_env, seed)
        else:
            model.set_env(vec_env)  # same spaces, so the trained weights carry over
        if not verbose:
            model.verbose = 0
        stop = StopWhenSolved(advance_at)
        before = model.num_timesteps
        print(f"\n=== stage {i + 1}/{len(gates)}: {gate} (up to {timesteps} steps) ===")
        model.learn(total_timesteps=timesteps, callback=stop, reset_num_timesteps=False)
        model.save(run_dir / gate.lower())
        result = best_circuit(model, env_kwargs, prefix="curriculum", root=root)
        used = model.num_timesteps - before
        summary.append({"gate": gate, "steps": used, "solve_rate": stop.solve_rate,
                        "correct": result.correct, "rows": len(result.rows), "solved": result.solved})
        print(f"stage {gate}: {used} steps, recent solve rate {stop.solve_rate:.0%}")
    print("\ncurriculum summary")
    for s in summary:
        print(f"  {s['gate']:<7} {s['steps']:>7} steps  solve rate {s['solve_rate']:>4.0%}  "
              f"best {s['correct']}/{s['rows']}{'  solved' if s['solved'] else ''}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", default="NOT", choices=sorted(GATES))
    parser.add_argument("--curriculum", help="comma-separated gates to train in turn, or 'default' "
                        f"({','.join(DEFAULT_CURRICULUM)}); overrides --gate")
    parser.add_argument("--timesteps", type=int, default=100_000, help="training steps (per stage with --curriculum)")
    parser.add_argument("--advance-at", type=float, default=0.9,
                        help="curriculum: move on once this share of the last 100 episodes is solved")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comparators", action="store_true", help="let the agent place comparators")
    args = parser.parse_args()

    if args.curriculum:
        gates = DEFAULT_CURRICULUM if args.curriculum == "default" else [g.strip().upper() for g in args.curriculum.split(",")]
        unknown = [g for g in gates if g not in GATES]
        if unknown:
            parser.error(f"unknown gate(s) {unknown}; choose from {sorted(GATES)}")
        train_curriculum(gates, args.timesteps, args.envs, args.max_steps, args.seed, args.comparators, args.advance_at)
        return

    env_kwargs = {"gate": args.gate, "max_steps": args.max_steps, "allow_comparators": args.comparators}
    vec_env = make_vec_env(RedstoneGateEnv, n_envs=args.envs, seed=args.seed, env_kwargs=env_kwargs)
    model = make_model(vec_env, args.seed)
    model.learn(total_timesteps=args.timesteps)

    run_dir = ROOT / "runs" / args.gate.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "maskable_ppo")
    best_circuit(model, env_kwargs)


if __name__ == "__main__":
    main()
