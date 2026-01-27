import pygame
import sys
import os

# Ensure project root is in path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.envs.kula_env import KulaWorldEnv

def main():
    # Initialize Environment
    env = KulaWorldEnv(render_mode="human")
    obs, info = env.reset()
    
    # --- CONTROLS ---
    # Movement
    action_map = {
        pygame.K_UP: 0,     # Turn Up
        pygame.K_DOWN: 1,   # Turn Down
        pygame.K_LEFT: 2,   # Turn Left
        pygame.K_RIGHT: 3,  # Turn Right
        pygame.K_f: 4,      # Move Forward
        pygame.K_SPACE: 5   # Jump
    }
    
    # Curriculum Switching
    level_map = {
        pygame.K_0: 0, # Hallway
        pygame.K_1: 1, # Turn
        pygame.K_2: 2, # Turn + Jump
        pygame.K_3: 3, # Turn + Jump + Spike
        pygame.K_4: 4, # Single Room
        pygame.K_5: 5, # Single Room + Coins
        pygame.K_6: 6, # Multi Room Small
        pygame.K_7: 7  # Full Kula World
    }
    
    running = True
    
    print("\n" + "="*40)
    print(" KULA WORLD AI - MANUAL TEST BENCH")
    print("="*40)
    print(" [MOVEMENT]")
    print("  ARROWS : Turn Agent")
    print("  F      : Move Forward")
    print("  SPACE  : Jump (2 Tiles)")
    print("-" * 40)
    print(" [CURRICULUM TOOLS]")
    print("  0 : Level 0 (Straight Line)")
    print("  1 : Level 1 (Jumping Gap)")
    print("  2 : Level 2 (L-Shape Turn)")
    print("  3 : Level 3 (Small Maze + Coins)")
    print("  4 : Level 4 (Full Random + Spikes)")
    print("="*40 + "\n")
    
    current_level = 0
    
    while running:
        # Process Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: 
                    running = False
                
                # --- CHANGE DIFFICULTY ---
                if event.key in level_map:
                    new_level = level_map[event.key]
                    print(f"\n>>> SWITCHING TO LEVEL {new_level} <<<")
                    
                    # 1. Update Env
                    env.set_curriculum_level(new_level)
                    current_level = new_level
                    
                    # 2. Reset to generate new layout immediately
                    obs, info = env.reset()
                
                # --- PERFORM ACTION ---
                elif event.key in action_map:
                    action = action_map[event.key]
                    
                    # Execute Step
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Logging
                    act_names = ["FACE UP", "FACE DOWN", "FACE LEFT", "FACE RIGHT", "MOVE FWD", "JUMP"]
                    print(f"[Lvl {current_level}] Act: {act_names[action]:10} | Rew: {reward:5.1f} | Lives: {info['lives']}")
                    
                    # Handle End of Episode
                    if terminated or truncated:
                        status = "VICTORY" if reward > 0 else "DIED/TIMEOUT"
                        print(f"--- {status} ---")
                        
                        # Respawn Logic
                        if info['lives'] > 0:
                            print("Respawning with remaining lives...")
                            env.reset(options={"keep_lives": True})
                        else:
                            print("GAME OVER - Resetting to 3 Lives")
                            env.reset(options={"keep_lives": False})

        # Render Frame
        env.render()

    env.close()
    print("Test Closed.")

if __name__ == "__main__":
    main()