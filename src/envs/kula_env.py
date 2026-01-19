import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class KulaWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode=None):
        super(KulaWorldEnv, self).__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.window_size = 512  # Pygame window size
        self.window = None
        self.clock = None
        
        # Grid value mapping
        self.EMPTY = 0
        self.WALL = 1
        self.PLAYER = 2
        self.KEY = 3
        self.SPIKE = 4
        self.GOAL = 5

        # ACTION SPACE: 0-3 (Move), 4-7 (Jump)
        # 0: Up, 1: Right, 2: Down, 3: Left
        # 4: Jump Up, 5: Jump Right, 6: Jump Down, 7: Jump Left
        self.action_space = spaces.Discrete(8)

        # OBSERVATION SPACE: The grid itself
        # 0=Empty, 1=Wall, 2=Player, 3=Key, 4=Spike, 5=Goal
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(grid_size, grid_size), dtype=np.int32
        )

        # Direction mapping (dy, dx)
        self._action_to_direction = {
            0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1),  # Move
            4: (-2, 0), 5: (0, 2), 6: (2, 0), 7: (0, -2)   # Jump (2 cells)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # 2. Place Player (fixed start or random)
        self._agent_location = np.array([0, 0])
        
        # 3. Place Goal
        self._target_location = self._get_random_empty_loc()
        self.grid[tuple(self._target_location)] = self.GOAL
        
        # 4. Place Key
        self.has_key = False
        self._key_location = self._get_random_empty_loc()
        self.grid[tuple(self._key_location)] = self.KEY
        
        # 5. Place Obstacles (Spikes)
        self.num_obstacles = int(self.grid_size * 1.5) # E.g., 15 obstacles on 10x10
        for _ in range(self.num_obstacles):
            loc = self._get_random_empty_loc()
            self.grid[tuple(loc)] = self.SPIKE

        # Initial render
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        # Map action to direction
        direction = self._action_to_direction[action]
        
        # Calculate target position
        current_pos = self._agent_location
        target_pos = current_pos + np.array(direction)
        
        # Boundary check (stay in place if out of bounds)
        if not (0 <= target_pos[0] < self.grid_size and 0 <= target_pos[1] < self.grid_size):
            target_pos = current_pos 

        # Collision Logic
        cell_value = self.grid[tuple(target_pos)]
        terminated = False
        reward = -0.1  # Base time penalty to encourage speed

        # Handle Death (Spikes)
        if cell_value == self.SPIKE:
            terminated = True
            reward = -10.0 # Death penalty
        
        # Handle Key
        elif cell_value == self.KEY:
            self.has_key = True
            self.grid[tuple(target_pos)] = self.EMPTY # Remove key from grid
            reward = 5.0 # Key bonus
            self._agent_location = target_pos
            
        # Handle Goal (Exit)
        elif cell_value == self.GOAL:
            if self.has_key:
                terminated = True
                reward = 20.0 # Victory!
            else:
                # Reached goal without key -> Stay there, no victory yet
                pass 
            self._agent_location = target_pos

        else:
            # Empty cell
            self._agent_location = target_pos

        # Get observation
        observation = self._get_obs()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, {}

    def _get_obs(self):
        # Return a copy of the grid with the player superimposed
        obs = self.grid.copy()
        obs[tuple(self._agent_location)] = self.PLAYER
        return obs

    def _get_random_empty_loc(self):
        # Find a random cell with value 0 (EMPTY)
        while True:
            loc = np.random.randint(0, self.grid_size, size=2)
            if np.array_equal(loc, self._agent_location): continue
            if self.grid[tuple(loc)] == self.EMPTY:
                return loc

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        pix_square_size = (self.window_size / self.grid_size)

        # Draw elements
        # Player: Blue, Goal: Green, Key: Gold, Spike: Red
        obs = self._get_obs()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    y * pix_square_size, x * pix_square_size, 
                    pix_square_size, pix_square_size
                )
                val = obs[x, y]
                
                if val == self.PLAYER:
                    pygame.draw.rect(canvas, (0, 0, 255), rect) # Blue
                elif val == self.GOAL:
                    pygame.draw.rect(canvas, (0, 255, 0), rect) # Green
                elif val == self.KEY:
                    pygame.draw.rect(canvas, (255, 215, 0), rect) # Gold
                elif val == self.SPIKE:
                    pygame.draw.rect(canvas, (255, 0, 0), rect) # Red
                
                # Grid lines
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()