from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ----------------------------
# Tile IDs (keep consistent everywhere)
# ----------------------------
VOID  = 0
FLOOR = 1
START = 2
# 3 reserved if you ever need
EXIT  = 4
SPIKE = 5
KEY   = 6
COIN  = 7


# ----------------------------
# Actions (baseline)
# Discrete(8): move/jump in 4 directions
# ----------------------------
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
JUMP_UP = 4
JUMP_DOWN = 5
JUMP_LEFT = 6
JUMP_RIGHT = 7


@dataclass
class RewardConfig:
    step: float = -0.1
    key: float = 10.0
    coin: float = 2.0
    complete: float = 50.0
    death: float = -50.0


class KulaWorldEnv(gym.Env):
    """
    KulaWorldEnv (baseline spec)

    - Observation: Dict with one-hot grid (C,H,W), agent_pos (2,), has_key (0/1)
    - Actions: Discrete(8) move/jump cardinal
    - Jump: only landing tile matters; intermediate ignored; no walls
    - Hazards: VOID/SPIKE on landing => terminated (death)
    - Objective: pick KEY then reach EXIT
    - Rewards:
        step -0.1, key +10, coin +2 (finite in episode), complete +50, death -50
      No exploration reward, no anti-camping
    - Episode ends:
        terminated=True for death/success
        truncated=True for timeout
      info["event"] in {"death","success","timeout","none"}
    """

    metadata = {
        "render_modes": [None, "human"],  # per your choice: renderer disabled during training
        "render_fps": 30,
    }

    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        max_steps: int = 250,
        reward_cfg: Optional[RewardConfig] = None,
        current_difficulty: int = 7,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.H, self.W = grid_size
        self.max_steps = int(max_steps)
        self.reward = reward_cfg or RewardConfig()
        self.current_difficulty = int(current_difficulty)
        self.render_mode = render_mode

        # One-hot channels: choose a stable ordering.
        # (You can include only tiles that appear, but keep it stable across runs.)
        self.tile_ids: List[int] = [VOID, FLOOR, START, EXIT, SPIKE, KEY, COIN]
        self.C = len(self.tile_ids)
        self._tile_to_channel = {tid: i for i, tid in enumerate(self.tile_ids)}

        # ---------
        # Spaces
        # ---------
        self.action_space = spaces.Discrete(8)

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.C, self.H, self.W),
                    dtype=np.uint8,
                ),
                "agent_pos": spaces.Box(
                    low=np.array([0, 0], dtype=np.int32),
                    high=np.array([self.H - 1, self.W - 1], dtype=np.int32),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "has_key": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            }
        )

        # ---------
        # State
        # ---------
        self.grid: np.ndarray = np.zeros((self.H, self.W), dtype=np.int32)
        self.agent_pos: np.ndarray = np.zeros((2,), dtype=np.int32)
        self.agent_dir = "right"   # only for rendering
        self.score: int = 0
        self.has_key: bool = False
        self.step_count: int = 0

        # Track coin count only if you want extra info/logging
        self.coins_collected: int = 0

        # RNG (Gymnasium-style)
        self.np_random: np.random.Generator = np.random.default_rng()

        # Renderer is kept separate (you already have KulaRenderer)
        self._renderer = None

    # ----------------------------
    # Gymnasium API
    # ----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self.score = 0
        self.has_key = False
        self.coins_collected = 0

        # Allow overriding difficulty per reset (useful for eval)
        if options and "difficulty" in options:
            self.current_difficulty = int(options["difficulty"])

        # Generate a fresh level
        self._generate_level(self.current_difficulty)

        obs = self._get_obs()
        info = {"event": "none", "difficulty": self.current_difficulty}
        return obs, info

    def step(self, action: int):
        self.step_count += 1

        # Default flags
        terminated = False
        truncated = False
        info: Dict[str, Any] = {"event": "none"}

        # Base step cost
        reward = float(self.reward.step)

        # Compute intended move (landing-based)
        dy, dx, is_jump = self._action_to_delta(int(action))
        step_len = 2 if is_jump else 1

        new_y = int(self.agent_pos[0]) + dy * step_len
        new_x = int(self.agent_pos[1]) + dx * step_len

        # Out of bounds behaves like landing in void => death (consistent + simple)
        if not (0 <= new_y < self.H and 0 <= new_x < self.W):
            reward += float(self.reward.death)
            terminated = True
            info["event"] = "death"
            return self._get_obs(), reward, terminated, truncated, info

        landing_tile = int(self.grid[new_y, new_x])

        # Landing on hazard => death (landing-only rule)
        if landing_tile in (VOID, SPIKE):
            # Move agent (optional; not strictly needed, but can help debugging)
            self.agent_pos[:] = (new_y, new_x)
            reward += float(self.reward.death)
            terminated = True
            info["event"] = "death"
            return self._get_obs(), reward, terminated, truncated, info

        # Safe landing: update position
        self.agent_pos[:] = (new_y, new_x)

        # Pickups / interactions
        if landing_tile == KEY and not self.has_key:
            self.has_key = True
            self.score += 100
            reward += float(self.reward.key)
            # Remove key from grid
            self.grid[new_y, new_x] = FLOOR
            info["event"] = "key"

        elif landing_tile == COIN:
            self.score += 10
            reward += float(self.reward.coin)
            self.coins_collected += 1
            # Remove coin from grid (coins are finite by construction)
            self.grid[new_y, new_x] = FLOOR
            info["event"] = "coin"

        elif landing_tile == EXIT:
            if self.has_key:
                self.score += 500
                reward += float(self.reward.complete)
                terminated = True
                info["event"] = "success"
            else:
                # Exit is "locked" if no key: treat as just a floor-like tile or keep EXIT.
                # We'll allow standing on it but not completing.
                info["event"] = "exit_locked"

        # Timeout handling (strict truncation)
        if (not terminated) and self.step_count >= self.max_steps:
            truncated = True
            info["event"] = "timeout"

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ----------------------------
    # Helpers
    # ----------------------------
    def _action_to_delta(self, action: int) -> Tuple[int, int, bool]:
        # Returns (dy, dx, is_jump)
        if action == MOVE_UP:
            self.agent_dir = "up"
            return -1, 0, False
        if action == MOVE_DOWN:
            self.agent_dir = "down"
            return 1, 0, False
        if action == MOVE_LEFT:
            self.agent_dir = "left"
            return 0, -1, False
        if action == MOVE_RIGHT:
            self.agent_dir = "right"
            return 0, 1, False
        if action == JUMP_UP:
            self.agent_dir = "up"
            return -1, 0, True
        if action == JUMP_DOWN:
            self.agent_dir = "down"
            return 1, 0, True
        if action == JUMP_LEFT:
            self.agent_dir = "left"
            return 0, -1, True
        if action == JUMP_RIGHT:
            self.agent_dir = "right"
            return 0, 1, True

        raise ValueError(f"Invalid action: {action}")

    def _get_obs(self) -> Dict[str, Any]:
        onehot = self._one_hot(self.grid)
        return {
            "grid": onehot,
            "agent_pos": self.agent_pos.astype(np.int32),
            "has_key": np.array([1 if self.has_key else 0], dtype=np.uint8),
        }

    def _one_hot(self, grid: np.ndarray) -> np.ndarray:
        # grid: (H,W) int32
        out = np.zeros((self.C, self.H, self.W), dtype=np.uint8)

        # If some tile IDs are not in tile_ids, they will be ignored (stays 0 everywhere).
        # Better to ensure your generator only uses known tiles.
        for tid, ch in self._tile_to_channel.items():
            out[ch] = (grid == tid).astype(np.uint8)

        return out

    
    # ----------------------------
    # Level generation (adapted from old implementation)
    # ----------------------------
    def _rng_int(self, low: int, high: int) -> int:
        """Inclusive randint using env RNG."""
        return int(self.np_random.integers(low, high + 1))

    def _rng_choice(self, seq):
        return seq[int(self.np_random.integers(0, len(seq)))]

    def _gen_L0_straight(self) -> None:
        """L0: Straight line. Move -> Key -> Exit"""
        row = self.H // 2
        start_col = 5
        length = 6

        # Keep within bounds for safety
        start_col = max(1, min(start_col, self.W - length - 1))

        for i in range(length):
            self.grid[row, start_col + i] = FLOOR

        self.agent_pos[:] = (row, start_col)
        self.grid[row, start_col] = START

        # Place key then exit
        self.grid[row, start_col + 3] = KEY
        self.grid[row, start_col + 5] = EXIT

    def _gen_procedural_path(self, feature: str = "turn") -> None:
        """L1-L3: Random corridor (1-tile wide) that guarantees a specific mechanic."""
        # 1) Random start (center-ish)
        start_r = self._rng_int(6, 14)
        start_c = self._rng_int(6, 14)

        cr, cc = start_r, start_c

        # Directions: 0=Up, 1=Down, 2=Left, 3=Right
        move_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        cdir = self._rng_choice([0, 1, 2, 3])

        path_tiles = [(cr, cc)]

        # Segment 1
        seg_len = self._rng_int(3, 5)
        for _ in range(seg_len):
            dr, dc = move_map[cdir]
            nr, nc = cr + dr, cc + dc
            if 1 <= nr < self.H - 1 and 1 <= nc < self.W - 1:
                cr, cc = nr, nc
                path_tiles.append((cr, cc))

        # Feature injection
        if feature == "turn":
            # turn 90 degrees
            if cdir in (0, 1):
                cdir = self._rng_choice([2, 3])
            else:
                cdir = self._rng_choice([0, 1])

            seg_len = self._rng_int(3, 5)
            for _ in range(seg_len):
                dr, dc = move_map[cdir]
                nr, nc = cr + dr, cc + dc
                if 1 <= nr < self.H - 1 and 1 <= nc < self.W - 1:
                    cr, cc = nr, nc
                    path_tiles.append((cr, cc))

        elif feature == "jump":
            # Add a gap (the middle tile stays VOID)
            dr, dc = move_map[cdir]
            # gap = (cr+dr, cc+dc) intentionally left VOID
            land_r, land_c = cr + (dr * 2), cc + (dc * 2)
            if 1 <= land_r < self.H - 1 and 1 <= land_c < self.W - 1:
                cr, cc = land_r, land_c
                path_tiles.append((cr, cc))

                # short segment after jump
                for _ in range(3):
                    dr, dc = move_map[cdir]
                    nr, nc = cr + dr, cc + dc
                    if 1 <= nr < self.H - 1 and 1 <= nc < self.W - 1:
                        cr, cc = nr, nc
                        path_tiles.append((cr, cc))
            else:
                # fallback: treat as turn instead (rare)
                if cdir in (0, 1):
                    cdir = self._rng_choice([2, 3])
                else:
                    cdir = self._rng_choice([0, 1])

        elif feature == "spike":
            seg_len = self._rng_int(4, 6)
            for _ in range(seg_len):
                dr, dc = move_map[cdir]
                nr, nc = cr + dr, cc + dc
                if 1 <= nr < self.H - 1 and 1 <= nc < self.W - 1:
                    cr, cc = nr, nc
                    path_tiles.append((cr, cc))

        # 2) Paint corridor tiles as FLOOR
        for (r, c) in path_tiles:
            self.grid[r, c] = FLOOR

        # 3) Place objects
        s = path_tiles[0]
        self.agent_pos[:] = s
        self.grid[s] = START

        # Infer agent_dir for sprite from first movement
        if len(path_tiles) > 1:
            (sr, sc) = s
            (tr, tc) = path_tiles[1]
            if tr < sr:
                self.agent_dir = "up"
            elif tr > sr:
                self.agent_dir = "down"
            elif tc < sc:
                self.agent_dir = "left"
            else:
                self.agent_dir = "right"

        # Exit at end
        self.grid[path_tiles[-1]] = EXIT

        # Key somewhere mid->end (avoid last cell)
        mid_idx = len(path_tiles) // 2
        key_idx = self._rng_int(mid_idx, max(mid_idx, len(path_tiles) - 2))
        key_idx = max(1, min(key_idx, len(path_tiles) - 2))
        self.grid[path_tiles[key_idx]] = KEY

        # Spike between start and key (on the corridor)
        if feature == "spike":
            valid = path_tiles[1:key_idx]
            if valid:
                sp = self._rng_choice(valid)
                self.grid[sp] = SPIKE

    def _generate_random_layout(self, max_rooms: int = 5, bounds: int = 20, spikes: bool = True, coins: bool = True) -> None:
        """L4-L7: Room-based procedural layout (adapted from old implementation)."""
        self.grid.fill(VOID)
        rooms = []

        def check_collision(r1, r2, buffer: int = 1) -> bool:
            # r = (x, y, w, h)
            return not (
                r1[0] >= r2[0] + r2[2] + buffer
                or r1[0] + r1[2] + buffer <= r2[0]
                or r1[1] >= r2[1] + r2[3] + buffer
                or r1[1] + r1[3] + buffer <= r2[1]
            )

        # Bounds define a centered sub-area where rooms may spawn
        min_b = (self.W - bounds) // 2
        max_b = min_b + bounds

        # 1) Initial room
        rw = self._rng_int(4, 7)
        rh = self._rng_int(4, 7)
        if bounds < 8:
            rw, rh = 4, 4

        x = self._rng_int(min_b, max_b - rw - 1)
        y = self._rng_int(min_b, max_b - rh - 1)
        rooms.append((x, y, rw, rh))

        # 2) Add more rooms
        attempts = 0
        while len(rooms) < max_rooms and attempts < 200:
            attempts += 1
            target_idx = self._rng_int(0, len(rooms) - 1)
            tx, ty, tw, th = rooms[target_idx]

            nw, nh = self._rng_int(4, 6), self._rng_int(4, 6)
            side = self._rng_choice([0, 1, 2, 3])
            gap = self._rng_choice([0, 1])  # 0=touching, 1=gap (jump)

            if side == 0:  # above
                nx = tx + self._rng_int(-nw + 2, tw - 2)
                ny = ty - nh + 1 - gap
            elif side == 1:  # below
                nx = tx + self._rng_int(-nw + 2, tw - 2)
                ny = ty + th - 1 + gap
            elif side == 2:  # left
                nx = tx - nw + 1 - gap
                ny = ty + self._rng_int(-nh + 2, th - 2)
            else:  # right
                nx = tx + tw - 1 + gap
                ny = ty + self._rng_int(-nh + 2, th - 2)

            # bounds check against the centered region
            if nx < min_b or ny < min_b or nx + nw >= max_b or ny + nh >= max_b:
                continue

            new_rect = (nx, ny, nw, nh)

            has_overlap = False
            for i, r in enumerate(rooms):
                # if attaching with no gap, allow overlap with the target room only
                if i == target_idx and gap == 0:
                    continue
                if check_collision(new_rect, r, buffer=0):
                    has_overlap = True
                    break

            if not has_overlap:
                rooms.append(new_rect)

        # 3) Paint room borders as FLOOR (interior remains VOID)
        for (rx, ry, rw, rh) in rooms:
            for ix in range(rx, rx + rw):
                for iy in range(ry, ry + rh):
                    if ix == rx or ix == rx + rw - 1 or iy == ry or iy == ry + rh - 1:
                        self.grid[iy, ix] = FLOOR

        # 4) Place objects on FLOOR border cells
        path_coords = np.argwhere(self.grid == FLOOR)
        if len(path_coords) < 6:
            # Fallback: if layout is too small, force a simple corridor
            self._gen_L0_straight()
            return

        perm = self.np_random.permutation(len(path_coords))
        path_coords = path_coords[perm]

        self.agent_pos[:] = path_coords[0]
        self.grid[tuple(path_coords[0])] = START

        self.grid[tuple(path_coords[1])] = KEY
        self.grid[tuple(path_coords[2])] = EXIT

        # Coins (finite by construction)
        if coins:
            for c in path_coords[3:]:
                if float(self.np_random.random()) < 0.08:
                    self.grid[tuple(c)] = COIN

        # Spikes (avoid corners; only straight segments)
        if spikes:
            current_paths = np.argwhere(self.grid == FLOOR)
            current_paths = current_paths[self.np_random.permutation(len(current_paths))]

            for cell in current_paths:
                r, c = int(cell[0]), int(cell[1])
                if float(self.np_random.random()) > 0.10:
                    continue

                # avoid border indexing issues
                if r <= 0 or r >= self.H - 1 or c <= 0 or c >= self.W - 1:
                    continue

                has_vert = (self.grid[r - 1, c] != VOID and self.grid[r + 1, c] != VOID)
                has_horz = (self.grid[r, c - 1] != VOID and self.grid[r, c + 1] != VOID)

                if has_vert or has_horz:
                    # don't overwrite key/exit/start
                    if self.grid[r, c] == FLOOR:
                        self.grid[r, c] = SPIKE

    def _generate_level(self, difficulty: int) -> None:
        """Dispatch curriculum level generation (L0-L7)."""
        self.grid.fill(VOID)

        if difficulty == 0:
            self._gen_L0_straight()
        elif difficulty == 1:
            self._gen_procedural_path(feature="turn")
        elif difficulty == 2:
            self._gen_procedural_path(feature="jump")
        elif difficulty == 3:
            self._gen_procedural_path(feature="spike")
        elif difficulty == 4:
            self._generate_random_layout(max_rooms=1, bounds=10, spikes=True, coins=False)
        elif difficulty == 5:
            self._generate_random_layout(max_rooms=1, bounds=12, spikes=True, coins=True)
        elif difficulty == 6:
            self._generate_random_layout(max_rooms=3, bounds=14, spikes=True, coins=True)
        else:
            self._generate_random_layout(max_rooms=5, bounds=20, spikes=True, coins=True)

    # ----------------------------
    # Render hook
    # ----------------------------
    def render(self):
        # Per baseline decision: renderer only in "human" mode (manual/eval).
        if self.render_mode != "human":
            return None

        # Lazy import/instantiate to avoid slowing training
        if self._renderer is None:
            from .kula_renderer import KulaRenderer  # adjust import to your structure
            self._renderer = KulaRenderer()

        # Renderer should accept a state snapshot rather than reading env internals.
        state = {
            "grid": self.grid,
            "agent_pos": tuple(int(v) for v in self.agent_pos),
            "agent_dir": self.agent_dir,
            "score": self.score,
            "has_key": self.has_key,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }
        self._renderer.render(state)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
