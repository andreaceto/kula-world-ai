import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import yaml

from .kula_renderer import KulaRenderer

# Grid Constants
GRID_WIDTH = 10
GRID_HEIGHT = 10

# Element IDs
EMPTY = 0       # Void (Black - Death)
PATH = 1        # Walkable path (White)
START = 3       # Start Point
EXIT = 4        # Exit
SPIKE = 5       # Spike (Death)
KEY = 6         # Key
COIN = 7        # Coin (Bonus Points) - Replaces Fruit

MAX_TIME = 50  # Max time steps per episode

class KulaWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, reward_config=os.path.join(os.path.dirname(__file__), "config.yml"), verbose=1, render_mode=None):
        super(KulaWorldEnv, self).__init__()
        
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.render_mode = render_mode
        self.renderer = None
        self.verbose = verbose

        # Load Rewards
        self.reward_config_path = reward_config
        self.rewards = self._load_rewards()
        
        # Action Space: 
        # 0=Face Up, 1=Face Down, 2=Face Left, 3=Face Right (No Movement)
        # 4=Move Forward (1 step), 5=Jump Forward (2 steps)
        self.action_space = spaces.Discrete(6)
        
        # Observation Space
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=10, shape=(self.height, self.width), dtype=np.int32),
            "agent_pos": spaces.Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32),
            "agent_dir": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),
            "has_key": spaces.Discrete(2),
            "lives": spaces.Discrete(10)
        })

        self.reset_state()

    def _load_rewards(self):
        """Loads reward values from YAML file."""
        if not os.path.exists(self.reward_config_path):
            print(f"Warning: {self.reward_config_path} not found. Using defaults.")
            return {
                "exploration_reward": 0.5,
                "step_penalty": -0.1,
                # "jump_penalty": -0.5,
                "camping_penalty_factor": 0.2,
                "collect_key": 10.0,
                "collect_coin": 2.0,
                "level_complete": 50.0,
                "exit_without_key": -1.0,
                "death_void": -10.0,
                "death_spike": -10.0,
                "death_timeout": -10.0
            }
            
        with open(self.reward_config_path, "r") as f:
            return yaml.safe_load(f)

    def reset_state(self):
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.agent_dir = (0, 1) # Facing Right (dy, dx)
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.has_key = False
        self.lives = 3
        self.max_time = MAX_TIME
        self.current_time = 0

        # Anti-camping counter
        self.steps_on_same_tile = 0
        # Track visited positions for exploration reward
        self.visited_mask = np.zeros((self.height, self.width), dtype=bool)
        self.visited_mask[0, 0] = True # Mark start as visited

    def _get_obs(self):
        return {
            "grid": self.grid.copy(),
            "agent_pos": self.agent_pos.copy(),
            "agent_dir": np.array(self.agent_dir, dtype=np.int32),
            "has_key": 1 if self.has_key else 0,
            "lives": self.lives
        }

    def _get_info(self):
        time_left = max(0, self.max_time - self.current_time)
        return {"time": time_left, "lives": self.lives}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        keep_lives = options and options.get("keep_lives", False)
        current_lives = self.lives if keep_lives else 3
        
        self.reset_state()
        self.lives = current_lives

        self._generate_level()
        
        # Initialize Renderer if needed and not already done
        if self.render_mode == "human" and self.renderer is None:
            self.renderer = KulaRenderer(self.width, self.height)

        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), self._get_info()

    def _generate_level(self):
        self.grid.fill(EMPTY)
        rooms = [] 

        def check_collision(r1, r2, buffer=1):
            return not (r1[0] >= r2[0] + r2[2] + buffer or 
                        r1[0] + r1[2] + buffer <= r2[0] or 
                        r1[1] >= r2[1] + r2[3] + buffer or 
                        r1[1] + r1[3] + buffer <= r2[1])

        # 1. Generate Rooms
        w, h = random.randint(4, 7), random.randint(4, 7)
        x, y = self.width // 2 - w // 2, self.height // 2 - h // 2
        rooms.append((x, y, w, h))

        max_rooms = 5
        attempts = 0
        
        while len(rooms) < max_rooms and attempts < 200:
            attempts += 1
            target_idx = random.randint(0, len(rooms) - 1)
            tx, ty, tw, th = rooms[target_idx]
            
            nw, nh = random.randint(4, 6), random.randint(4, 6)
            side = random.choice([0, 1, 2, 3])
            gap = random.choice([0, 1])
            
            if side == 0: nx, ny = tx + random.randint(-nw + 2, tw - 2), ty - nh + 1 - gap 
            elif side == 1: nx, ny = tx + random.randint(-nw + 2, tw - 2), ty + th - 1 + gap
            elif side == 2: nx, ny = tx - nw + 1 - gap, ty + random.randint(-nh + 2, th - 2)
            elif side == 3: nx, ny = tx + tw - 1 + gap, ty + random.randint(-nh + 2, th - 2)
            
            if nx < 1 or ny < 1 or nx + nw >= self.width - 1 or ny + nh >= self.height - 1: continue

            new_rect = (nx, ny, nw, nh)
            has_overlap = False
            for i, r in enumerate(rooms):
                if i == target_idx and gap == 0: continue
                if check_collision(new_rect, r, buffer=0):
                    has_overlap = True
                    break
            
            if not has_overlap: rooms.append(new_rect)

        # 2. Paint Grid
        for (rx, ry, rw, rh) in rooms:
            for i in range(rx, rx + rw):
                for j in range(ry, ry + rh):
                    if i == rx or i == rx + rw - 1 or j == ry or j == ry + rh - 1:
                        self.grid[j, i] = PATH

        # 3. Place Objects
        path_coords = np.argwhere(self.grid == PATH)
        np.random.shuffle(path_coords)
        
        if len(path_coords) > 5:
            # Fixed Objects
            self.agent_pos = path_coords[0]
            self.grid[self.agent_pos[0], self.agent_pos[1]] = START
            
            self.grid[path_coords[1][0], path_coords[1][1]] = KEY
            self.grid[path_coords[2][0], path_coords[2][1]] = EXIT
            
            # Place Coins (Randomly)
            remaining_paths = path_coords[3:]
            for c in remaining_paths:
                if random.random() < 0.08: # 8% chance for coin
                     self.grid[c[0], c[1]] = COIN

            # Place Spikes (Smart Logic: No Corners)
            # Re-fetch paths because we might have placed coins
            current_paths = np.argwhere((self.grid == PATH)) 
            np.random.shuffle(current_paths)
            
            for c in current_paths:
                r, c_idx = c[0], c[1]
                # Probability check first
                if random.random() > 0.05: continue 
                
                # CORNER CHECK:
                # A spike is valid only if there is a straight line through it (Vertical or Horizontal)
                # We check neighbors. 
                has_vertical_path = (self.grid[r-1, c_idx] != EMPTY and self.grid[r+1, c_idx] != EMPTY)
                has_horizontal_path = (self.grid[r, c_idx-1] != EMPTY and self.grid[r, c_idx+1] != EMPTY)
                
                if has_vertical_path or has_horizontal_path:
                    self.grid[r, c_idx] = SPIKE

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        moves_dir = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        reward = self.rewards.get("step_penalty")
        terminated = False
        info = self._get_info()
        self.current_time += 1

        # Check Timeout immediately
        if self.current_time >= self.max_time:
            return self._handle_death("Timeout", self.rewards["death_timeout"])

        # Save old position to check if we moved
        old_pos = self.agent_pos.copy()
        target_pos = self.agent_pos.copy()

        if action < 4:
            self.agent_dir = moves_dir[action] # Turn
        elif action == 4:
            dy, dx = self.agent_dir
            target_pos = self.agent_pos + np.array([dy, dx]) # Move
        elif action == 5:
            dy, dx = self.agent_dir
            target_pos = self.agent_pos + np.array([dy * 2, dx * 2]) # Jump
            # reward += self.rewards["jump_penalty"]

        # Bounds
        if not (0 <= target_pos[0] < self.height and 0 <= target_pos[1] < self.width):
            return self._handle_death("Fell off world", self.rewards["death_void"])

        cell = self.grid[target_pos[0], target_pos[1]]

        if cell == EMPTY:
            return self._handle_death("Fell in void", self.rewards["death_void"])
        elif cell == SPIKE:
            return self._handle_death("Hit spike", self.rewards["death_spike"])
        
        # Update Position
        self.agent_pos = target_pos
        
        # If we are on a valid path and haven't been here before
        if self.grid[target_pos[0], target_pos[1]] != EMPTY and not self.visited_mask[target_pos[0], target_pos[1]]:
            reward += self.rewards.get("exploration_reward", 0.5)
            self.visited_mask[target_pos[0], target_pos[1]] = True

        # Check if we actually moved from the previous tile
        if np.array_equal(self.agent_pos, old_pos):
            self.steps_on_same_tile += 1
            # Penalty increases linearly: -0.1, -0.3, -0.5...
            # This discourages spinning in place or hitting walls repeatedly
            reward -= (self.rewards["camping_penalty_factor"] * self.steps_on_same_tile)
        else:
            # Reset counter if we moved to a new tile
            self.steps_on_same_tile = 0
        
        # Object Interaction
        if cell == KEY:
            self.has_key = True
            self.grid[target_pos[0], target_pos[1]] = PATH
            reward += self.rewards["collect_key"]
            if self.verbose:
                print("Key collected!")
        elif cell == COIN:
            reward += self.rewards["collect_coin"]
            self.grid[target_pos[0], target_pos[1]] = PATH
            if self.verbose:
                print("Coin collected!")
        elif cell == EXIT:
            if self.has_key:
                reward += self.rewards["level_complete"]
                terminated = True
                if self.verbose:
                    print("Level Complete!")
            else:
                reward += self.rewards["exit_without_key"]

        if self.current_time >= self.max_time:
            return self._handle_death("Timeout", self.rewards["death_timeout"])
            
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, info

    def _handle_death(self, reason, penalty_reward):
        self.lives -= 1
        if self.verbose:
            print(f"Dead: {reason} | Lives: {self.lives}")
        return self._get_obs(), penalty_reward, True, False, self._get_info()

    def render(self):
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(self)

    def close(self):
        if self.renderer:
            self.renderer.close()