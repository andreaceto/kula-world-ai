from src.envs.kula_env import KulaWorldEnv
import time

# Create environment in 'human' mode to visualize it
env = KulaWorldEnv(grid_size=8, render_mode="human")
obs, info = env.reset()

print("Environment started! Press Ctrl+C to stop.")
print("Legend: BLUE=Player, RED=Spike, GOLD=Key, GREEN=Goal")

try:
    for _ in range(100):
        # Sample a random action (0-7)
        action = env.action_space.sample()
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Action: {action}, Reward: {reward}")
        
        if terminated:
            print("Episode finished!")
            obs, info = env.reset()
            time.sleep(0.5) # Dramatic pause
            
except KeyboardInterrupt:
    print("Test interrupted.")
finally:
    env.close()