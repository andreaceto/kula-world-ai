import os
import sys
import csv
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from action_mask import kula_action_mask
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.envs.kula_env import KulaWorldEnv

# -------------------------
# Difficulty sampler wrapper
# -------------------------
class DifficultySampler(gym.Wrapper):
    """
    Wrapper that overrides difficulty at every reset by sampling from [0..max_difficulty],
    biased toward current max.

    - If max_difficulty < mix_start_level: uses fixed difficulty = max_difficulty
    - Else: with prob p_current choose max_difficulty, else choose uniform among [0..max_difficulty-1]
    """
    def __init__(
        self,
        env: gym.Env,
        shared_state: Dict[str, Any],
        mix_start_level: int = 2,
        p_current: float = 0.70,
    ):
        super().__init__(env)
        self.shared_state = shared_state
        self.mix_start_level = int(mix_start_level)
        self.p_current = float(p_current)
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            # keep reproducibility per-episode
            self._rng = np.random.default_rng(int(seed) + 1234567)

        max_d = int(self.shared_state["max_difficulty"])

        if max_d < self.mix_start_level:
            d = max_d
        else:
            if max_d == 0:
                d = 0
            else:
                if self._rng.random() < self.p_current:
                    d = max_d
                else:
                    d = int(self._rng.integers(0, max_d))  # 0..max_d-1

        opts = dict(options or {})
        opts["difficulty"] = d
        return self.env.reset(seed=seed, options=opts)


# -------------------------
# Config
# -------------------------
@dataclass
class CurriculumCfg:
    start_difficulty: int = 0
    max_difficulty: int = 7

    # Mixed difficulty starts at this level (your request: 2)
    mix_start_level: int = 2
    p_current: float = 0.70  # bias toward current difficulty during mixed training

    # Parallel envs
    n_envs: int = 8

    # PPO rollout/update
    n_steps: int = 4096
    batch_size: int = 1024

    # Per-difficulty schedules
    train_steps_by_difficulty: Dict[int, int] = None
    eval_episodes_by_difficulty: Dict[int, int] = None
    success_threshold_by_difficulty: Dict[int, float] = None
    patience_by_difficulty: Dict[int, int] = None

    # Reproducibility
    seed: int = 0

    # Output
    log_dir: str = "logs"
    model_dir: str = "models"
    eval_csv: str = "logs/eval_by_level.csv"


def _defaults_if_none(cfg: CurriculumCfg) -> None:
    if cfg.train_steps_by_difficulty is None:
        cfg.train_steps_by_difficulty = {
            0: 1_259,
            1: 6_250,
            2: 6_250,
            3: 6_250,
            4: 12_500,
            5: 25_000,
            6: 62_500,
            7: 187_500,
        }

    if cfg.eval_episodes_by_difficulty is None:
        cfg.eval_episodes_by_difficulty = {
            0: 30,
            1: 40,
            2: 60,
            3: 80,
            4: 100,
            5: 120,
            6: 150,
            7: 200,
        }

    if cfg.success_threshold_by_difficulty is None:
        cfg.success_threshold_by_difficulty = {
            0: 0.90,
            1: 0.85,
            2: 0.80,
            3: 0.80,
            4: 0.75,
            5: 0.75,
            6: 0.65,
            7: 0.65,
        }

    if cfg.patience_by_difficulty is None:
        cfg.patience_by_difficulty = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 2,
            7: 2,
        }


# -------------------------
# Env factories
# -------------------------
def make_train_env(i: int, shared_state: Dict[str, Any], cfg: CurriculumCfg):
    def _init():
        env = KulaWorldEnv(render_mode=None)
        env = ActionMasker(env, kula_action_mask)
        env = DifficultySampler(
            env,
            shared_state=shared_state,
            mix_start_level=cfg.mix_start_level,
            p_current=cfg.p_current,
        )
        # seed each env differently
        env.reset(seed=cfg.seed + i, options={"difficulty": int(shared_state["max_difficulty"])})
        return env
    return _init


def make_eval_env(seed: int) -> gym.Env:
    # IMPORTANT: eval env is fixed-difficulty per reset (no sampling wrapper)
    env = KulaWorldEnv(render_mode=None)
    env = ActionMasker(env, kula_action_mask)
    env.reset(seed=seed, options={"difficulty": 0})
    return env


# -------------------------
# Evaluation
# -------------------------
def eval_on_levels(
    model: MaskablePPO,
    levels: List[int],
    episodes_per_level: Dict[int, int],
    seed_base: int,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate deterministically on each specific level.
    Uses action masks during evaluation (must match training constraints).
    Returns per-level stats.
    """
    env = make_eval_env(seed_base + 9999)

    stats: Dict[int, Dict[str, float]] = {}
    for d in levels:
        n_eps = int(episodes_per_level.get(d, 50))
        rng = np.random.default_rng(seed_base + 10_000 + d)
        ep_seeds = rng.integers(0, 1_000_000, size=n_eps)

        success = 0
        deaths = 0
        timeouts = 0
        ep_lens: List[int] = []
        ep_rets: List[float] = []

        for i in range(n_eps):
            obs, info = env.reset(seed=int(ep_seeds[i]), options={"difficulty": int(d)})
            terminated = truncated = False
            ret = 0.0
            length = 0

            while not (terminated or truncated):
                masks = get_action_masks(env)
                action, _ = model.predict(obs, deterministic=True, action_masks=masks)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ret += float(reward)
                length += 1

            event = info.get("event", "none")
            if event == "success":
                success += 1
            elif event == "death":
                deaths += 1
            elif event == "timeout":
                timeouts += 1

            ep_lens.append(length)
            ep_rets.append(ret)

        stats[d] = {
            "success_rate": success / n_eps,
            "death_rate": deaths / n_eps,
            "timeout_rate": timeouts / n_eps,
            "ep_len_mean": float(np.mean(ep_lens)) if ep_lens else 0.0,
            "ep_ret_mean": float(np.mean(ep_rets)) if ep_rets else 0.0,
        }

    env.close()
    return stats


def ensure_eval_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "global_iter",
                "trained_max_difficulty",
                "eval_level",
                "success_rate",
                "death_rate",
                "timeout_rate",
                "ep_len_mean",
                "ep_ret_mean",
            ])


def append_eval_csv(path: str, global_iter: int, trained_max: int, stats: Dict[int, Dict[str, float]]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for lvl, s in stats.items():
            w.writerow([
                global_iter,
                trained_max,
                lvl,
                s["success_rate"],
                s["death_rate"],
                s["timeout_rate"],
                s["ep_len_mean"],
                s["ep_ret_mean"],
            ])


# -------------------------
# Training
# -------------------------
def train_curriculum(cfg: CurriculumCfg):
    _defaults_if_none(cfg)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    ensure_eval_csv(cfg.eval_csv)

    # Shared state: controls current max difficulty used by sampler wrapper
    shared_state = {"max_difficulty": int(cfg.start_difficulty)}

    # VecEnv (parallel envs)
    vec_env = DummyVecEnv([make_train_env(i, shared_state, cfg) for i in range(cfg.n_envs)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(cfg.log_dir, "monitor.csv"))

    # Reward normalization only (keep one-hot obs untouched)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Slightly bigger heads (not a custom CNN, just more capacity in policy/value MLP)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=cfg.log_dir,
        seed=cfg.seed,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=200_000,
        save_path=cfg.model_dir,
        name_prefix="maskableppo_mixed_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=True,  # important when using VecNormalize
    )

    difficulty = int(cfg.start_difficulty)
    passes = 0
    global_iter = 0

    while True:
        shared_state["max_difficulty"] = difficulty

        train_steps = int(cfg.train_steps_by_difficulty.get(difficulty, 300_000))
        eval_eps = int(cfg.eval_episodes_by_difficulty.get(difficulty, 100))
        threshold = float(cfg.success_threshold_by_difficulty.get(difficulty, 0.80))
        patience = int(cfg.patience_by_difficulty.get(difficulty, 2))

        print(f"\n=== TRAIN (mixed from L{cfg.mix_start_level}): max=L{difficulty}, steps={train_steps:,}, n_envs={cfg.n_envs} ===")
        model.learn(total_timesteps=train_steps, reset_num_timesteps=False, callback=checkpoint_cb)

        # Evaluate on ALL levels up to current max (this is what you asked)
        levels_to_eval = list(range(0, difficulty + 1))
        # Use the same eval_eps for current, slightly fewer for earlier if you want.
        ep_map = {lvl: (eval_eps if lvl == difficulty else max(30, eval_eps // 3)) for lvl in levels_to_eval}

        stats = eval_on_levels(
            model=model,
            levels=levels_to_eval,
            episodes_per_level=ep_map,
            seed_base=cfg.seed + 999,
        )

        # Pretty print per level
        print("\n--- EVAL BY LEVEL ---")
        for lvl in levels_to_eval:
            s = stats[lvl]
            print(
                f"L{lvl}: SR={s['success_rate']:.2%} | "
                f"Death={s['death_rate']:.2%} | Timeout={s['timeout_rate']:.2%} | "
                f"Len={s['ep_len_mean']:.1f} | Ret={s['ep_ret_mean']:.1f}"
            )

        append_eval_csv(cfg.eval_csv, global_iter=global_iter, trained_max=difficulty, stats=stats)
        global_iter += 1

        # Promotion based only on CURRENT level success rate
        sr_curr = stats[difficulty]["success_rate"]
        print(f"\n=== PROMOTION CHECK: L{difficulty} SR={sr_curr:.2%} (threshold={threshold:.2%}, patience={patience}) ===")

        if sr_curr >= threshold:
            passes += 1
            print(f"Passes at L{difficulty}: {passes}/{patience}")
            if passes >= patience:
                # Save model + VecNormalize statistics together
                model.save(os.path.join(cfg.model_dir, f"maskableppo_best_through_L{difficulty}.zip"))
                vec_env.save(os.path.join(cfg.model_dir, f"vecnormalize_best_through_L{difficulty}.pkl"))

                if difficulty >= cfg.max_difficulty:
                    print("\n✅ Solved max difficulty. Stopping.")
                    break

                difficulty += 1
                passes = 0
                print(f"\n>>> PROMOTE to L{difficulty}")
        else:
            passes = 0

    # Final saves
    model.save(os.path.join(cfg.model_dir, "maskableppo_final_curriculum.zip"))
    vec_env.save(os.path.join(cfg.model_dir, "vecnormalize_final.pkl"))
    vec_env.close()


if __name__ == "__main__":
    cfg = CurriculumCfg(
        start_difficulty=0,
        max_difficulty=7,
        mix_start_level=2,   # your request
        p_current=0.70,
        n_envs=8,
        n_steps=2048,
        batch_size=512,
        seed=0,
        log_dir="logs",
        model_dir="models",
        eval_csv="logs/eval_by_level.csv",
    )
    train_curriculum(cfg)
