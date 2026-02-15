import os
import sys
import argparse
import time
from collections import defaultdict
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from action_mask import kula_action_mask
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.envs.kula_env import KulaWorldEnv

MODEL_PATH = "models\maskableppo_mixed_ckpt_272000000_steps.zip"
VISUALIZE = False

def run_visual_eval(
    model_path: str,
    episodes_by_level: dict[int, int],
    seed: int = 0,
    sleep_between_episodes: float = 0.25,
):
    """
    Visual evaluation (render_mode="human") across all levels.
    Uses action masking during eval (must match training).
    Prints per-episode outcomes and a final detailed summary by level.
    """

    # Load model (we don't need VecNormalize for obs since we normalized reward only)
    model = MaskablePPO.load(model_path)

    # Create ONE env window and reuse it (faster + cleaner)
    env = KulaWorldEnv(render_mode="human")
    env = ActionMasker(env, kula_action_mask)

    # Stats containers
    # stats[level] -> counts + lists
    stats = {
        lvl: {
            "success": 0,
            "death": 0,
            "timeout": 0,
            "other": 0,
            "lengths": [],
            "returns": [],
        }
        for lvl in sorted(episodes_by_level.keys())
    }

    global_ep = 0

    try:
        for lvl in sorted(episodes_by_level.keys()):
            n_eps = int(episodes_by_level[lvl])
            rng = np.random.default_rng(seed + 10_000 + lvl)
            ep_seeds = rng.integers(0, 1_000_000, size=n_eps)

            print(f"\n==============================")
            print(f"VISUAL EVAL — Level L{lvl} — Episodes: {n_eps}")
            print(f"==============================")

            for i in range(n_eps):
                global_ep += 1
                obs, info = env.reset(seed=int(ep_seeds[i]), options={"difficulty": int(lvl)})

                terminated = truncated = False
                ep_ret = 0.0
                ep_len = 0

                while not (terminated or truncated):
                    # IMPORTANT: apply action masks in evaluation
                    masks = get_action_masks(env)
                    action, _ = model.predict(obs, deterministic=True, action_masks=masks)

                    obs, reward, terminated, truncated, info = env.step(int(action))

                    if VISUALIZE:
                        env.render()

                    ep_ret += float(reward)
                    ep_len += 1

                    # Let pygame breathe a bit; renderer already caps FPS internally,
                    # but sleeping between episodes makes it easier to watch transitions.
                    # (Keep this small.)
                    # time.sleep(0.0)

                event = info.get("event", "other")
                if event == "success":
                    stats[lvl]["success"] += 1
                elif event == "death":
                    stats[lvl]["death"] += 1
                elif event == "timeout":
                    stats[lvl]["timeout"] += 1
                else:
                    stats[lvl]["other"] += 1

                stats[lvl]["lengths"].append(ep_len)
                stats[lvl]["returns"].append(ep_ret)

                print(
                    f"[L{lvl} | ep {i+1:03d}/{n_eps:03d} | global {global_ep:04d}] "
                    f"event={event:7s} len={ep_len:4d} return={ep_ret:8.2f}"
                )

                if sleep_between_episodes > 0:
                    time.sleep(sleep_between_episodes)

    finally:
        env.close()

    # ---- Detailed final summary ----
    print("\n\n#############################################")
    print("FINAL VISUAL EVAL SUMMARY (by difficulty)")
    print("#############################################")

    for lvl in sorted(stats.keys()):
        s = stats[lvl]
        n = len(s["lengths"])
        if n == 0:
            print(f"L{lvl}: no episodes run")
            continue

        succ = s["success"]
        death = s["death"]
        tout = s["timeout"]
        other = s["other"]

        lengths = np.array(s["lengths"], dtype=np.float32)
        rets = np.array(s["returns"], dtype=np.float32)

        def pct(x): return 100.0 * x / n

        print(f"\n--- Level L{lvl} (N={n}) ---")
        print(f"Success: {succ} ({pct(succ):.1f}%) | Death: {death} ({pct(death):.1f}%) | Timeout: {tout} ({pct(tout):.1f}%) | Other: {other} ({pct(other):.1f}%)")
        print(f"Episode length: mean={lengths.mean():.1f}, std={lengths.std():.1f}, min={lengths.min():.0f}, max={lengths.max():.0f}")
        print(f"Return:         mean={rets.mean():.2f}, std={rets.std():.2f}, min={rets.min():.2f}, max={rets.max():.2f}")
        print(f"Length percentiles: p25={np.percentile(lengths, 25):.0f}, p50={np.percentile(lengths, 50):.0f}, p75={np.percentile(lengths, 75):.0f}")
        print(f"Return percentiles: p25={np.percentile(rets, 25):.2f}, p50={np.percentile(rets, 50):.2f}, p75={np.percentile(rets, 75):.2f}")

    # Global summary
    all_lengths = []
    all_rets = []
    all_succ = all_death = all_tout = all_other = 0
    for lvl in stats:
        all_lengths += stats[lvl]["lengths"]
        all_rets += stats[lvl]["returns"]
        all_succ += stats[lvl]["success"]
        all_death += stats[lvl]["death"]
        all_tout += stats[lvl]["timeout"]
        all_other += stats[lvl]["other"]

    N = len(all_lengths)
    if N:
        all_lengths = np.array(all_lengths, dtype=np.float32)
        all_rets = np.array(all_rets, dtype=np.float32)
        print("\n=============================================")
        print(f"OVERALL (all levels) N={N}")
        print(f"Success={all_succ} ({100*all_succ/N:.1f}%) | Death={all_death} ({100*all_death/N:.1f}%) | Timeout={all_tout} ({100*all_tout/N:.1f}%) | Other={all_other} ({100*all_other/N:.1f}%)")
        print(f"Len mean={all_lengths.mean():.1f}, Ret mean={all_rets.mean():.2f}")
        print("=============================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.25)

    # if you want to override episodes easily
    ap.add_argument("--eps_easy", type=int, default=5, help="episodes for L0-L5")
    ap.add_argument("--eps_hard", type=int, default=10, help="episodes for L6-L7")
    args = ap.parse_args()

    episodes_by_level = {lvl: args.eps_easy for lvl in range(0, 6)}
    episodes_by_level[6] = args.eps_hard
    episodes_by_level[7] = args.eps_hard

    run_visual_eval(
        model_path=args.model,
        episodes_by_level=episodes_by_level,
        seed=args.seed,
        sleep_between_episodes=args.sleep,
    )


if __name__ == "__main__":
    main()
