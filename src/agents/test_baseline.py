import os
import sys
import time
import gymnasium as gym
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.envs.kula_env import KulaWorldEnv

def test_curriculum_showcase():
    model_path = "models/ppo_curriculum/best_model" # Or final_model
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}. Run training first.")
        return

    print(f"Loading Model from {model_path}...")
    model = PPO.load(model_path)
    
    # Render mode 'human' to watch
    env = KulaWorldEnv(render_mode="human")
    
    # Define the showcase levels
    levels = [
        (0, "Basic Hallway"),
        (1, "Jumping Gap"),
        (2, "Corner Turn"),
        (3, "Coin Maze"),
        (4, "Full Kula World")
    ]
    
    print("\n--- STARTING CURRICULUM SHOWCASE ---")
    
    for level_id, name in levels:
        print(f"\nSetting Environment to LEVEL {level_id}: {name}")
        env.set_curriculum_level(level_id)
        
        # Reset environment
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        print(f"Playing Level {level_id}...")
        
        while not (terminated or truncated):
            # Deterministic action
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            time.sleep(0.1) # Adjustable speed
            
        if env.has_key and reward > 0:
            print(f">>> SUCCESS on Level {level_id}!")
        else:
            print(f">>> FAILED on Level {level_id}.")
        
        time.sleep(1.0) # Pause between levels

    env.close()
    print("\nShowcase Finished.")

if __name__ == "__main__":
    test_curriculum_showcase()