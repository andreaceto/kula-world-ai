import os
import sys
import time
import gymnasium as gym
from stable_baselines3 import PPO

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.envs.kula_env import KulaWorldEnv

def test_arcade_mode():
    model_path = "models/ppo_baseline/best_model"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}. Run training first.")
        return

    print(f"Loading Arcade Mode...")
    model = PPO.load(model_path)
    
    # Render mode is 'human' so we can watch
    env = KulaWorldEnv(render_mode="human")
    
    # ARCADE CONSTANTS
    MAX_LEVELS = 5
    current_level = 1
    
    # Initial Reset to set lives to 3
    obs, info = env.reset(seed=42)
    total_lives = env.lives  # Should be 3
    
    print(f"\n{'='*40}")
    print(f" ARCADE CAMPAIGN STARTED")
    print(f" Goal: Complete {MAX_LEVELS} Levels")
    print(f" Lives: {total_lives}")
    print(f"{'='*40}\n")

    # Campaign Loop
    while current_level <= MAX_LEVELS and env.lives > 0:
        
        # Calculate a unique seed. 
        # If we retry, we add a 'retry_offset' to generate a fresh map 
        # so the deterministic agent doesn't die in the exact same spot.
        current_seed = (current_level * 100) + (3 - env.lives)
        
        print(f"--- LEVEL {current_level} (Lives: {env.lives}) ---")
        
        # Reset Env but KEEP LIVES (Pass 'keep_lives': True)
        obs, info = env.reset(seed=current_seed, options={"keep_lives": True})
        
        terminated = False
        truncated = False
        level_won = False
        
        # Gameplay Loop
        while not (terminated or truncated):
            # Deterministic=True for best performance
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Speed control
            time.sleep(1)

        # CHECK OUTCOME
        # If we have the key and terminated with positive reward, we won.
        if env.has_key and terminated and reward > 0:
            print(f">>> VICTORY! Level {current_level} Complete.")
            current_level += 1
            time.sleep(1.0) # Victory pause
        else:
            # DEATH
            print(f">>> DIED on Level {current_level}.")
            # Note: env.lives is automatically decremented by the environment step()
            # We just loop back. If env.lives is 0, the main loop will break.
            time.sleep(0.1) # Death pause

    # FINAL RESULTS
    print(f"\n{'#'*40}")
    if env.lives > 0:
        print(f" CAMPAIGN VICTORY! All {MAX_LEVELS} levels completed.")
        print(f" Remaining Lives: {env.lives}")
    else:
        print(f" GAME OVER.")
        print(f" The agent was defeated on Level {current_level}.")
    print(f"{'#'*40}")

    env.close()

if __name__ == "__main__":
    test_arcade_mode()