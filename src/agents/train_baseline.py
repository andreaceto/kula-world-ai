import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.envs.kula_env import KulaWorldEnv

# --- CONFIGURATION ---
VERBOSE_LEVEL = 0            # 0: No Output, 1: Basic, 2: Detailed
TOTAL_TIMESTEPS = 10_000_000       
EVAL_FREQ = 10_000              # How often to test (steps)
EVAL_EPISODES = 5               # How many times to test per eval
PROMOTION_THRESHOLD = 30.0      # Reward needed to level up
LOG_DIR = "logs/curriculum"
MODELS_DIR = "models/ppo_baseline"

class CurriculumManager(BaseCallback):
    """
    A unified callback that manages Evaluation, Saving, and Curriculum updates.
    Replaces EvalCallback to ensure we have full control over the environment.
    """
    def __init__(self, train_env, eval_env, verbose=1):
        super(CurriculumManager, self).__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.current_level = 0
        self.max_level = 7
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.n_calls % EVAL_FREQ == 0:
            self._run_evaluation()
        return True

    def _run_evaluation(self):
        print(f"\n--- EVALUATION AT STEP {self.num_timesteps} ---")
        print(f"Current Difficulty Level: {self.current_level}")
        
        # 1. Run Evaluation
        # We use the built-in evaluate_policy function
        mean_reward, std_reward = evaluate_policy(
            self.model, 
            self.eval_env, 
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True
        )
        
        print(f"Result: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # 2. Log to Tensorboard
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("curriculum/level", self.current_level)

        # 3. Save Best Model
        if mean_reward > self.best_mean_reward:
            print("New Best Model! Saving...")
            self.best_mean_reward = mean_reward
            self.model.save(f"{MODELS_DIR}/best_model")

        # 4. Check for Promotion
        if self.current_level < self.max_level:
            if mean_reward >= PROMOTION_THRESHOLD:
                self._promote_agent()
            else:
                print(f"Status: Failed to pass (Need {PROMOTION_THRESHOLD}). Retrying...")
        else:
            if mean_reward >= PROMOTION_THRESHOLD:
                print("Status: Mastered Final Level.")

    def _promote_agent(self):
        self.current_level += 1
        # print(f"\n{'!'*40}")
        # print(f" PROMOTION! Upgrading to Level {self.current_level}")
        # print(f"{'!'*40}\n")
        
        # A. Update Training Environment
        self.train_env.unwrapped.set_curriculum_level(self.current_level)
        
        # B. Update Evaluation Environment
        self.eval_env.unwrapped.set_curriculum_level(self.current_level)
        
        # Reset to ensure the next episode starts fresh with new logic
        self.eval_env.reset()

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Initialize Environments
    # We keep eval_env as a simple Monitor (not Vectorized) so we can control it easily
    train_env = Monitor(KulaWorldEnv(verbose=VERBOSE_LEVEL))
    eval_env = Monitor(KulaWorldEnv(verbose=VERBOSE_LEVEL)) # No rendering during automated eval

    # 2. Initialize Manager Callback
    manager = CurriculumManager(train_env, eval_env)

    # 3. Agent
    model = PPO(
        "MultiInputPolicy", 
        train_env, 
        verbose=VERBOSE_LEVEL,
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        n_steps=2048,
        ent_coef=0.05
    )

    print("--- STARTING TRAINING ---")
    print(f"Goal: {TOTAL_TIMESTEPS} steps. Check every {EVAL_FREQ} steps.")

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=manager)
    except KeyboardInterrupt:
        print("Training interrupted.")

    model.save(f"{MODELS_DIR}/final_model")
    print("Done.")

if __name__ == "__main__":
    main()