import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.envs.kula_env import KulaWorldEnv

# --- CONSTANTS ---
VERBOSE_LEVEL = 0                # 0=no output, 1=info, 2=debug
TOTAL_TIMESTEPS = 10_000_000     # Total steps to train
EVAL_FREQ = 100_000              # Evaluate the agent every X steps
EVAL_EPISODES = 5                # Average reward over X episodes during evaluation (MUST match the number of seeds below)
FIXED_SEEDS = [
    101, 102, 103, 104, 105
    ]                           # Fixed seeds for evaluation episodes
PATIENCE = 10                    # Stop training if no improvement after X evaluations
LEARNING_RATE = 0.0003

class FixedSeedWrapper(gym.Wrapper):
    """
    Forces the environment to cycle through a fixed list of seeds on every reset.
    This ensures evaluation happens on the EXACT same levels every time.
    """
    def __init__(self, env, seeds):
        super().__init__(env)
        self.seeds = seeds
        self.idx = 0

    def reset(self, **kwargs):
        # Pick the current seed
        seed_to_use = self.seeds[self.idx]
        
        # Inject it into kwargs
        kwargs['seed'] = seed_to_use
        
        # Advance index (loop back to 0 if we finish the list)
        self.idx = (self.idx + 1) % len(self.seeds)
        
        return self.env.reset(**kwargs)

def main():
    # 1. Setup Directories
    models_dir = "models/ppo_baseline"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Initialize Environments
    # Training Environment
    train_env = Monitor(KulaWorldEnv(verbose=VERBOSE_LEVEL, render_mode=None)) 
    
    # Evaluation Environment (Separate instance is crucial for accurate testing)
    base_eval_env = KulaWorldEnv(verbose=VERBOSE_LEVEL, render_mode=None)
    eval_env = Monitor(FixedSeedWrapper(base_eval_env, seeds=FIXED_SEEDS))
    
    # 3. Define Callbacks
    # A. Checkpoint (Backup every 50k steps regardless of performance)
    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=models_dir, 
        name_prefix="ppo_kula_backup"
    )

    # B. Early Stopping Mechanism
    # Stops training if there is no improvement after PATIENCE * EVAL_FREQ steps
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=PATIENCE, 
        min_evals=3, 
        verbose=VERBOSE_LEVEL
    )

    # C. Evaluation & Best Model Saver
    # This triggers the evaluation, saves the "best_model.zip", and checks for early stopping
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        callback_after_eval=stop_train_callback, # Link early stopping here
        verbose=VERBOSE_LEVEL
    )

    # 4. Agent Configuration
    model = PPO(
        "MultiInputPolicy", 
        train_env, 
        verbose=VERBOSE_LEVEL,
        tensorboard_log=log_dir,
        learning_rate=LEARNING_RATE,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.05,
        device="auto"
    )

    print(f"--- Starting Training ({TOTAL_TIMESTEPS} steps) ---")
    print(f"{'='*50}")
    print(f"TENSORBOARD COMMAND:")
    print(f"tensorboard --logdir {os.path.abspath(log_dir)}")
    print(f"{'='*50}\n")
    
    try:
        # Pass the list of callbacks (Checkpoint + Eval/EarlyStop)
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, eval_callback]
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current state...")

    # 6. Final Save
    model.save(f"{models_dir}/final_model")
    print("--- Training Completed ---")
    print(f"Best model saved to: {models_dir}/best_model.zip")
    print(f"Final model saved to: {models_dir}/final_model.zip")
    
    # Reminder at the end
    print(f"\nMonitor progress: tensorboard --logdir {log_dir}")

if __name__ == "__main__":
    main()