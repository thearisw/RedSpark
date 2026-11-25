"""Train a (maskable) PPO agent to build a logic gate.

    python -m redspark.train --gate NOT --timesteps 100000

Saves the model to runs/<gate>/ and the best circuit it built to
circuits/trained_<gate>.json, which the viewer can open.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from .env import RedstoneGateEnv
from .tasks import GATES, evaluate_gate
from .sim import Circuit

ROOT = Path(__file__).resolve().parents[1]


def rollout(model, env: RedstoneGateEnv, deterministic: bool = True) -> tuple[float, dict]:
    obs, _ = env.reset()
    total, done = 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        done = terminated or truncated
    return total, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gate", default="NOT", choices=sorted(GATES))
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comparators", action="store_true", help="let the agent place comparators")
    args = parser.parse_args()

    env_kwargs = {"gate": args.gate, "max_steps": args.max_steps, "allow_comparators": args.comparators}
    vec_env = make_vec_env(RedstoneGateEnv, n_envs=args.envs, seed=args.seed, env_kwargs=env_kwargs)
    # MaskablePPO is PPO that only samples legal moves (see RedstoneGateEnv.action_masks).
    model = MaskablePPO("MlpPolicy", vec_env, n_steps=256, batch_size=256, ent_coef=0.02,
                learning_rate=3e-4, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)

    run_dir = ROOT / "runs" / args.gate.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "maskable_ppo")

    # Keep the best of a greedy rollout and a few sampled ones.
    env = RedstoneGateEnv(**env_kwargs)
    best = max((rollout(model, env, deterministic=(i == 0)) for i in range(20)), key=lambda r: r[0])
    reward, info = best
    circuit = Circuit.from_dict(info["circuit"])
    result = evaluate_gate(circuit, args.gate)
    out = ROOT / "circuits" / f"trained_{args.gate.lower()}.json"
    out.write_text(json.dumps(info["circuit"], indent=1) + "\n")
    print(f"\nbest episode reward {reward:.3f}, truth table {result.correct}/{len(result.rows)}, "
          f"solved={result.solved}")
    env.circuit = circuit
    print(env.render())
    print(f"saved circuit to {out}")


if __name__ == "__main__":
    main()
