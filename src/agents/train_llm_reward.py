import os
import argparse

import numpy as np
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.envs.kula_env import KulaWorldEnv
from src.agents.action_mask import kula_action_mask
from src.agents.llm_reward import LLMRewardModel, DeepSeekRewardConfig


class LLMRewardWrapper(gym.Wrapper):
    """
    Replaces env reward with LLM reward and forces a fixed difficulty at every reset.
    """

    def __init__(self, env: gym.Env, reward_model: LLMRewardModel, fixed_difficulty: int):
        super().__init__(env)
        self.rm = reward_model
        self.fixed_difficulty = int(fixed_difficulty)
        self._last_obs = None

    def reset(self, **kwargs):
        # Force difficulty every episode
        options = dict(kwargs.get("options") or {})
        options["difficulty"] = self.fixed_difficulty
        kwargs["options"] = options

        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs2, env_reward, terminated, truncated, info = self.env.step(action)
        llm_r = self.rm.score_transition(self._last_obs, int(action), obs2, info)
        self._last_obs = obs2
        return obs2, float(llm_r), terminated, truncated, info


def make_env(difficulty: int, seed: int, cfg: DeepSeekRewardConfig):
    def _thunk():
        base = KulaWorldEnv(render_mode=None)
        rm = LLMRewardModel(cfg)
        env = LLMRewardWrapper(base, rm, fixed_difficulty=difficulty)
        env = ActionMasker(env, kula_action_mask)
        return env
    return _thunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", type=int, default=2)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--out", type=str, default="models/maskableppo_llm_reward.zip")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = DeepSeekRewardConfig(model=args.model, debug=args.debug)

    env_fns = [make_env(args.difficulty, seed=1000 + i, cfg=cfg) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # keep it simple; you can reuse your baseline hyperparams later
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log="logs/tb_llm_reward",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)

    print(f"Saved model to: {args.out}")
    vec_env.close()


if __name__ == "__main__":
    main()
