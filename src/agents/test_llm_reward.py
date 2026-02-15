import sys
import os
import argparse
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from src.envs.kula_env import KulaWorldEnv
from src.agents.action_mask import kula_action_mask

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def run(levels=range(8), episodes=10, render=False, model_path="models/maskableppo_llm_reward.zip"):
    model = MaskablePPO.load(model_path)

    for lvl in levels:
        success = death = timeout = other = 0
        lens, rews = [], []

        for ep in range(episodes):
            env = KulaWorldEnv(render_mode="human" if render else None)
            env = ActionMasker(env, kula_action_mask)

            obs, info = env.reset(seed=1000 * lvl + ep, options={"difficulty": lvl})
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

            event = info.get("event", "none")
            if event == "success":
                success += 1
            elif event == "death":
                death += 1
            elif event == "timeout":
                timeout += 1
            else:
                other += 1

            lens.append(ep_len)
            rews.append(ep_rew)
            env.close()

        print(
            f"[L{lvl}] success={success/episodes:.2f} death={death/episodes:.2f} "
            f"timeout={timeout/episodes:.2f} len={np.mean(lens):.1f} rew={np.mean(rews):.1f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_level", type=int, default=7)
    p.add_argument("--render", action="store_true")
    p.add_argument("--model", type=str, default="models/maskableppo_llm_reward.zip")
    args = p.parse_args()

    run(levels=range(args.max_level + 1), episodes=args.episodes, render=args.render, model_path=args.model)


if __name__ == "__main__":
    main()
