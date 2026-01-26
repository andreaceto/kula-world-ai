import pygame
from src.envs.kula_env import KulaWorldEnv

def main():
    env = KulaWorldEnv(render_mode="human")
    obs, info = env.reset()
    
    # Control Scheme
    key_map = {
        pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3, # Turn
        pygame.K_f: 4,      # Forward
        pygame.K_SPACE: 5   # Jump
    }
    
    running = True
    print("--- MANUAL TEST (Updated) ---")
    print("Goal: Get KEY -> EXIT")
    print("Avoid: Spikes (Purple, never in corners)")
    print("Bonus: Coins (Yellow)")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                
                if event.key in key_map:
                    action = key_map[event.key]
                    obs, reward, term, trunc, info = env.step(action)
                    
                    act_name = ["Face UP", "Face DOWN", "Face LEFT", "Face RIGHT", "MOVE FWD", "JUMP"][action]
                    print(f"Act: {act_name:10} | Rew: {reward:.1f} | Lives: {info['lives']}")
                    
                    if term:
                        print("Episode End.")
                        if info['lives'] > 0:
                            env.reset(options={"keep_lives": True})
                        else:
                            print("GAME OVER")
                            env.reset(options={"keep_lives": False})

        env.render()

    env.close()

if __name__ == "__main__":
    main()