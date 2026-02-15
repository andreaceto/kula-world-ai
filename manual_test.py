from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym

from src.envs.kula_env import (
    KulaWorldEnv,
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
    JUMP_UP, JUMP_DOWN, JUMP_LEFT, JUMP_RIGHT,
)

# Pygame is used indirectly by the renderer, but we read keyboard modifiers via pygame.
import pygame


def action_from_keys() -> Optional[int]:
    """
    Returns an action if a movement key is pressed, else None.
    - Arrow keys / WASD: move
    - Hold SHIFT or press J: jump modifier
    """
    keys = pygame.key.get_pressed()
    mods = pygame.key.get_mods()
    jump_mod = bool(mods & pygame.KMOD_SHIFT) or keys[pygame.K_j]

    # Direction priority: Up/Down/Left/Right
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return JUMP_UP if jump_mod else MOVE_UP
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return JUMP_DOWN if jump_mod else MOVE_DOWN
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return JUMP_LEFT if jump_mod else MOVE_LEFT
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return JUMP_RIGHT if jump_mod else MOVE_RIGHT

    return None


def reset_env(env: gym.Env, difficulty: int, seed: Optional[int] = None):
    obs, info = env.reset(seed=seed, options={"difficulty": difficulty})
    return obs, info


def main():
    difficulty = 0
    seed = None  # change or set to None if you want fully random runs

    env = KulaWorldEnv(render_mode="human", current_difficulty=difficulty)
    obs, info = reset_env(env, difficulty=difficulty, seed=seed)

    # Initial render
    env.render()

    print("Manual Test Controls:")
    print("- Move: Arrow keys or WASD")
    print("- Jump: Hold SHIFT (or hold J) + direction")
    print("- Reset: R")
    print("- Difficulty: 0..7 (resets immediately)")
    print("- Next/Prev difficulty: N / P")
    print("- Quit: ESC")

    running = True
    # Control how often we apply repeated movement when holding keys
    repeat_delay_sec = 0.08
    last_step_time = 0.0

    while running:
        # Pump events so pygame updates key states (renderer also pumps, but we do it here too)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Reset current difficulty
                elif event.key == pygame.K_r:
                    obs, info = reset_env(env, difficulty=difficulty, seed=seed)

                # Difficulty hotkeys 0..7
                elif pygame.K_0 <= event.key <= pygame.K_7:
                    difficulty = event.key - pygame.K_0
                    print(f"Switched difficulty to L{difficulty}")
                    obs, info = reset_env(env, difficulty=difficulty, seed=seed)

                # Next/Prev difficulty
                elif event.key == pygame.K_n:
                    difficulty = min(7, difficulty + 1)
                    print(f"Switched difficulty to L{difficulty}")
                    obs, info = reset_env(env, difficulty=difficulty, seed=seed)

                elif event.key == pygame.K_p:
                    difficulty = max(0, difficulty - 1)
                    print(f"Switched difficulty to L{difficulty}")
                    obs, info = reset_env(env, difficulty=difficulty, seed=seed)

        # Apply movement if enough time elapsed (so holding a key doesn't step too fast)
        now = time.time()
        if now - last_step_time >= repeat_delay_sec:
            act = action_from_keys()
            if act is not None:
                obs, reward, terminated, truncated, step_info = env.step(act)
                last_step_time = now

                # Render after each applied action
                env.render()

                # Print key events (optional but useful)
                if step_info.get("event") not in (None, "none"):
                    print(f"Event: {step_info['event']}, reward={reward:.2f}")

                # Auto-reset on episode end
                if terminated or truncated:
                    end_event = step_info.get("event", "end")
                    print(f"Episode ended ({end_event}). Resetting on L{difficulty}...")
                    time.sleep(0.4)
                    obs, info = reset_env(env, difficulty=difficulty, seed=seed)

        # Render continuously at the renderer's FPS cap (optional)
        # If your renderer already ticks FPS, this can be light:
        env.render()

    env.close()


if __name__ == "__main__":
    main()
