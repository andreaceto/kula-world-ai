import os
import csv
import time
import argparse
from typing import Dict, Any, List

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from src.envs.kula_env import KulaWorldEnv
from src.agents.action_mask import kula_action_mask

# se il tuo file LLM policy ha un nome diverso, aggiorna import qui:
from src.agents.llm_policy import LLMPolicyAgent


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _seed_for(level: int, ep: int, seed_base: int) -> int:
    return int(seed_base + 1000 * level + ep)


def _init_env(render: bool):
    env = KulaWorldEnv(render_mode="human" if render else None)
    env = ActionMasker(env, kula_action_mask)
    return env


def eval_rl_agent(
    agent_name: str,
    model_path: str,
    levels: List[int],
    episodes_per_level: int,
    seed_base: int,
    render: bool,
    out_csv: str,
):
    _ensure_dir(out_csv)
    model = MaskablePPO.load(model_path)

    rows = []
    for lvl in levels:
        counts = {"success": 0, "death": 0, "timeout": 0, "other": 0}
        lens, rews = [], []

        for ep in range(episodes_per_level):
            env = _init_env(render)
            obs, info = env.reset(seed=_seed_for(lvl, ep, seed_base), options={"difficulty": lvl})

            done = False
            ep_len = 0
            ep_rew = 0.0
            while not done:
                mask = get_action_masks(env)
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_rew += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)
                if render:
                    env.render()

            event = str(info.get("event", "none"))
            if event in counts:
                counts[event] += 1
            else:
                counts["other"] += 1

            lens.append(ep_len)
            rews.append(ep_rew)
            env.close()

        n = episodes_per_level
        rows.append({
            "agent": agent_name,
            "level": lvl,
            "episodes": n,
            "success_rate": counts["success"] / n,
            "death_rate": counts["death"] / n,
            "timeout_rate": counts["timeout"] / n,
            "other_rate": counts["other"] / n,
            "ep_len_mean": float(np.mean(lens)) if lens else 0.0,
            "ep_len_std": float(np.std(lens)) if lens else 0.0,
            "ep_rew_mean": float(np.mean(rews)) if rews else 0.0,
            "ep_rew_std": float(np.std(rews)) if rews else 0.0,
        })

        print(f"[{agent_name}][L{lvl}] succ={rows[-1]['success_rate']:.2f} "
              f"death={rows[-1]['death_rate']:.2f} timeout={rows[-1]['timeout_rate']:.2f} "
              f"len={rows[-1]['ep_len_mean']:.1f} rew={rows[-1]['ep_rew_mean']:.1f}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out_csv}")
    return rows


def eval_llm_policy_agent(
    agent_name: str,
    levels: List[int],
    episodes_per_level: int,
    seed_base: int,
    render: bool,
    out_csv: str,
):
    _ensure_dir(out_csv)
    agent = LLMPolicyAgent()  # usa .env

    rows = []
    for lvl in levels:
        counts = {"success": 0, "death": 0, "timeout": 0, "other": 0}
        lens, rews = [], []
        call_lat_ms = []

        for ep in range(episodes_per_level):
            env = _init_env(render)
            obs, info = env.reset(seed=_seed_for(lvl, ep, seed_base), options={"difficulty": lvl})

            done = False
            ep_len = 0
            ep_rew = 0.0
            while not done:
                mask = get_action_masks(env)

                t0 = time.perf_counter()
                action = agent.act(obs, mask, difficulty=lvl)
                t1 = time.perf_counter()
                call_lat_ms.append((t1 - t0) * 1000.0)

                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_rew += float(reward)
                ep_len += 1
                done = bool(terminated or truncated)
                if render:
                    env.render()

            event = str(info.get("event", "none"))
            if event in counts:
                counts[event] += 1
            else:
                counts["other"] += 1

            lens.append(ep_len)
            rews.append(ep_rew)
            env.close()

        n = episodes_per_level
        rows.append({
            "agent": agent_name,
            "level": lvl,
            "episodes": n,
            "success_rate": counts["success"] / n,
            "death_rate": counts["death"] / n,
            "timeout_rate": counts["timeout"] / n,
            "other_rate": counts["other"] / n,
            "ep_len_mean": float(np.mean(lens)) if lens else 0.0,
            "ep_len_std": float(np.std(lens)) if lens else 0.0,
            "ep_rew_mean": float(np.mean(rews)) if rews else 0.0,
            "ep_rew_std": float(np.std(rews)) if rews else 0.0,
            "mean_step_latency_ms": float(np.mean(call_lat_ms)) if call_lat_ms else 0.0,
        })

        print(f"[{agent_name}][L{lvl}] succ={rows[-1]['success_rate']:.2f} "
              f"death={rows[-1]['death_rate']:.2f} timeout={rows[-1]['timeout_rate']:.2f} "
              f"len={rows[-1]['ep_len_mean']:.1f} rew={rows[-1]['ep_rew_mean']:.1f} "
              f"lat={rows[-1]['mean_step_latency_ms']:.1f}ms")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out_csv}")
    return rows


def write_merged(all_rows: List[Dict[str, Any]], out_csv: str):
    _ensure_dir(out_csv)
    # union of keys (llm policy has extra latency column)
    keys = sorted({k for r in all_rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"Saved: {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max_level", type=int, default=7)
    p.add_argument("--seed_base", type=int, default=1234)
    p.add_argument("--render", action="store_true")

    p.add_argument("--baseline_model", type=str, default="models/maskableppo_best_overall.zip")
    p.add_argument("--llm_reward_model", type=str, default="models/maskableppo_llm_reward.zip")

    args = p.parse_args()
    levels = list(range(args.max_level + 1))

    all_rows = []

    # 1) RL baseline
    all_rows += eval_rl_agent(
        agent_name="rl_baseline",
        model_path=args.baseline_model,
        levels=levels,
        episodes_per_level=args.episodes,
        seed_base=args.seed_base,
        render=args.render,
        out_csv="logs/eval_baseline.csv",
    )

    # 2) LLM-as-Policy
    all_rows += eval_llm_policy_agent(
        agent_name="llm_policy",
        levels=levels,
        episodes_per_level=args.episodes,
        seed_base=args.seed_base,
        render=args.render,
        out_csv="logs/eval_llm_policy.csv",
    )

    # 3) RL trained with LLM reward
    all_rows += eval_rl_agent(
        agent_name="rl_llm_reward",
        model_path=args.llm_reward_model,
        levels=levels,
        episodes_per_level=args.episodes,
        seed_base=args.seed_base,
        render=args.render,
        out_csv="logs/eval_llm_reward.csv",
    )

    write_merged(all_rows, "logs/eval_all_agents.csv")


if __name__ == "__main__":
    main()
