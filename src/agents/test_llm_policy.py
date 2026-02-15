import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from src.envs.kula_env import KulaWorldEnv
from src.agents.action_mask import kula_action_mask

from src.agents.llm_policy import LLMPolicyAgent, DeepSeekConfig


def run_eval(
    levels=range(8),
    episodes_per_level=50,
    render=False,
    model_name="deepseek-chat",
    out_csv="logs/eval_llm_policy_by_level.csv",
    seed_base=1234,
    debug=False
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    results = []

    for lvl in levels:
        # Create env fresh per level (same style as baseline)
        env = KulaWorldEnv(render_mode="human" if render else None)
        env = ActionMasker(env, kula_action_mask)

        agent = LLMPolicyAgent(
            config=DeepSeekConfig(
                model=model_name,
                debug=debug,
                debug_log_file="logs/llm_debug.txt" if debug else None,
            )
        )

        # Stats
        counts = defaultdict(int)
        lengths = []
        returns = []

        for ep in range(episodes_per_level):
            obs, info = env.reset(seed=seed_base + ep + 1000 * lvl, options={"difficulty": lvl})

            done = False
            ep_ret = 0.0
            ep_len = 0

            while not done:
                mask = get_action_masks(env)
                action = agent.act(obs, mask, difficulty=lvl)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)

                if render:
                    env.render()

            event = info.get("event", "none")
            if event not in {"success", "death", "timeout"}:
                # align with your baseline accounting: treat everything else as "other"
                counts["other"] += 1
            else:
                counts[event] += 1

            lengths.append(ep_len)
            returns.append(ep_ret)

        # summarize
        n = episodes_per_level
        row = {
            "level": lvl,
            "episodes": n,
            "success_rate": counts["success"] / n,
            "death_rate": counts["death"] / n,
            "timeout_rate": counts["timeout"] / n,
            "other_rate": counts["other"] / n,
            "ep_len_mean": float(np.mean(lengths)) if lengths else 0.0,
            "ep_len_std": float(np.std(lengths)) if lengths else 0.0,
            "ep_rew_mean": float(np.mean(returns)) if returns else 0.0,
            "ep_rew_std": float(np.std(returns)) if returns else 0.0,
        }
        results.append(row)

        print(
            f"[L{lvl}] success={row['success_rate']:.2f} "
            f"death={row['death_rate']:.2f} timeout={row['timeout_rate']:.2f} "
            f"len={row['ep_len_mean']:.1f}±{row['ep_len_std']:.1f} "
            f"rew={row['ep_rew_mean']:.1f}±{row['ep_rew_std']:.1f}"
        )

        env.close()

    # write csv
    fieldnames = list(results[0].keys()) if results else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nSaved: {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max_level", type=int, default=7)
    p.add_argument("--render", action="store_true")
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--out_csv", type=str, default="logs/eval_llm_policy_by_level.csv")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    run_eval(
        levels=range(args.max_level + 1),
        episodes_per_level=args.episodes,
        render=args.render,
        model_name=args.model,
        out_csv=args.out_csv,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
