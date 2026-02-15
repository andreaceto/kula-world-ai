import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Action mapping (MUST match env)
# -----------------------------
ACTION_ID_TO_NAME = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "JUMP_UP",
    5: "JUMP_DOWN",
    6: "JUMP_LEFT",
    7: "JUMP_RIGHT",
}
ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}


CHANNEL_TO_CHAR = {
    0: "_",  # VOID (visible!)
    1: ".",  # FLOOR
    2: "S",  # START
    3: "E",  # EXIT
    4: "^",  # SPIKE
    5: "K",  # KEY
    6: "C",  # COIN
}


@dataclass
class DeepSeekConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_tokens: int = 16
    timeout_s: int = 30
    max_retries: int = 3
    retry_backoff_s: float = 0.75  # exponential-ish

    debug: bool = False
    debug_log_file: Optional[str] = None

class LLMCache:
    """
    Very small, low-friction cache.
    - In-memory for now (cheap + enough).
    - You can later replace with sqlite/shelve if you want persistence.
    """
    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self._store: Dict[str, int] = {}

    def get(self, key: str) -> Optional[int]:
        return self._store.get(key)

    def set(self, key: str, value: int) -> None:
        if len(self._store) >= self.max_size:
            # crude eviction: drop ~10% oldest by iteration order
            # (dict preserves insertion order in py3.7+)
            n_drop = max(1, self.max_size // 10)
            for k in list(self._store.keys())[:n_drop]:
                self._store.pop(k, None)
        self._store[key] = value


def _obs_to_tile_ids(obs_grid_onehot: np.ndarray) -> np.ndarray:
    """
    obs_grid_onehot: (C,H,W), one-hot channels.
    returns: (H,W) tile ids via argmax
    """
    if obs_grid_onehot.ndim != 3:
        raise ValueError(f"Expected grid shape (C,H,W), got {obs_grid_onehot.shape}")
    return np.argmax(obs_grid_onehot, axis=0).astype(np.int32)


def _local_patch(tile_ids: np.ndarray, agent_y: int, agent_x: int, radius: int = 2) -> List[str]:
    """
    Returns a small (2r+1)x(2r+1) text patch around agent.
    Marks agent as '@'.
    """
    h, w = tile_ids.shape
    lines: List[str] = []
    for dy in range(-radius, radius + 1):
        row_chars = []
        y = agent_y + dy
        for dx in range(-radius, radius + 1):
            x = agent_x + dx
            if y < 0 or y >= h or x < 0 or x >= w:
                ch = "#"
            else:
                tid = int(tile_ids[y, x])
                ch = CHANNEL_TO_CHAR.get(tid, "?")
            row_chars.append(ch)
        lines.append("".join(row_chars))

    # overlay agent
    center = radius
    line = list(lines[center])
    line[center] = "@"
    lines[center] = "".join(line)
    return lines


def _valid_actions_from_mask(mask: np.ndarray) -> List[int]:
    mask = np.asarray(mask).astype(bool)
    if mask.shape != (8,):
        raise ValueError(f"Expected mask shape (8,), got {mask.shape}")
    return [i for i, ok in enumerate(mask.tolist()) if ok]


def _parse_action_int(text: str) -> Optional[int]:
    if not text:
        return None

    t = text.strip()

    # STRICT: must match exactly "ACTION=<digit>"
    m = re.fullmatch(r"(?i)ACTION\s*=\s*([0-7])", t)
    if not m:
        return None

    return int(m.group(1))

def _choose_fallback(valid_actions: List[int], target_vec: Optional[Tuple[int,int,int]]) -> int:
    # Prefer actions that reduce distance to target using (dy, dx).
    # If no target, prefer move actions.
    if not valid_actions:
        return 0

    # Default order preference
    preferred_moves = [0, 3, 1, 2]  # UP, RIGHT, DOWN, LEFT (example)
    preferred_jumps = [4, 7, 5, 6]

    if target_vec is None:
        for a in preferred_moves:
            if a in valid_actions:
                return a
        return valid_actions[0]

    dy, dx, _ = target_vec

    # Choose axis with larger absolute distance (greedy)
    candidates = []
    if abs(dx) >= abs(dy):
        candidates += ([3] if dx > 0 else []) + ([2] if dx < 0 else [])
        candidates += ([1] if dy > 0 else []) + ([0] if dy < 0 else [])
    else:
        candidates += ([1] if dy > 0 else []) + ([0] if dy < 0 else [])
        candidates += ([3] if dx > 0 else []) + ([2] if dx < 0 else [])

    # prefer move candidates, then jumps in same direction
    for a in candidates:
        if a in valid_actions:
            return a

    # if moves not available, try directional jumps
    jump_map = {0:4, 1:5, 2:6, 3:7}
    for a in candidates:
        ja = jump_map.get(a)
        if ja is not None and ja in valid_actions:
            return ja

    # fallback to any move
    for a in preferred_moves:
        if a in valid_actions:
            return a
    return valid_actions[0]


def _stable_cache_key(
    has_key: bool,
    agent_pos: Tuple[int, int],
    patch_lines: List[str],
    valid_actions: List[int],
    key_vec: Optional[Tuple[int, int, int]],
    exit_vec: Optional[Tuple[int, int, int]],
) -> str:
    payload = {
        "k": bool(has_key),
        "p": [int(agent_pos[0]), int(agent_pos[1])],
        "patch": patch_lines,
        "va": valid_actions,
        "key": list(key_vec) if key_vec is not None else None,
        "exit": list(exit_vec) if exit_vec is not None else None,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _find_nearest(tile_ids: np.ndarray, src_y: int, src_x: int, targets: List[int]) -> Optional[Tuple[int, int, int]]:
    """
    Returns (dy, dx, manhattan_dist) to the nearest target tile id in `targets`.
    If none exists, returns None.
    """
    ys, xs = np.where(np.isin(tile_ids, targets))
    if len(ys) == 0:
        return None

    dys = ys.astype(int) - int(src_y)
    dxs = xs.astype(int) - int(src_x)
    dists = np.abs(dys) + np.abs(dxs)
    i = int(np.argmin(dists))
    dy = int(dys[i])
    dx = int(dxs[i])
    dist = int(dists[i])
    return dy, dx, dist

def _in_local_patch(dy: int, dx: int, radius: int = 2) -> bool:
    return (-radius <= dy <= radius) and (-radius <= dx <= radius)

def _debug_print(config: DeepSeekConfig, text: str):
    if not config.debug:
        return

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")

    if config.debug_log_file:
        os.makedirs(os.path.dirname(config.debug_log_file), exist_ok=True)
        with open(config.debug_log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(text + "\n")
            f.write("=" * 60 + "\n")


class LLMPolicyAgent:
    """
    LLM-as-Policy agent using DeepSeek via OpenAI SDK compatibility.
    It chooses an action among valid actions (from mask).
    """

    def __init__(self, config: Optional[DeepSeekConfig] = None, cache: Optional[LLMCache] = None):
        self.config = config or DeepSeekConfig()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key, base_url=self.config.base_url)
        self.cache = cache or LLMCache()

    def build_prompt(
        self,
        obs: Dict[str, Any],
        action_mask: np.ndarray,
        difficulty: int,
    ) -> Tuple[str, str, str, Any, Any]:
        """
        Returns: (cache_key, system_prompt, user_prompt, debug_patch_text)
        """

        grid = np.asarray(obs["grid"])
        agent_y, agent_x = map(int, obs["agent_pos"])
        has_key = bool(obs["has_key"])

        tile_ids = _obs_to_tile_ids(grid)
        patch = _local_patch(tile_ids, agent_y, agent_x, radius=2)

        key_vec = _find_nearest(tile_ids, agent_y, agent_x, targets=[5])   # KEY
        exit_vec = _find_nearest(tile_ids, agent_y, agent_x, targets=[3])  # EXIT

        # If present, also compute if it's inside the 5x5 patch
        key_in_patch = False
        exit_in_patch = False
        if key_vec is not None:
            key_in_patch = _in_local_patch(key_vec[0], key_vec[1], radius=2)
        if exit_vec is not None:
            exit_in_patch = _in_local_patch(exit_vec[0], exit_vec[1], radius=2)


        valid_actions = _valid_actions_from_mask(action_mask)
        cache_key = _stable_cache_key(
            has_key=has_key,
            agent_pos=(agent_y, agent_x),
            patch_lines=patch,
            valid_actions=valid_actions,
            key_vec=key_vec,
            exit_vec=exit_vec,
        )

        system_prompt = (
            "You are a policy for a grid game. Choose the best NEXT action.\n"
            "Orientation: in view_5x5, first row is UP (north), last row is DOWN (south). "
            "Left-to-right is LEFT-to-RIGHT (west-to-east). '@' is the agent at the center.\n"
            "Symbols: . floor, _ void (death), ^ spike (death), K key, E exit, # out-of-bounds.\n"
            "Exit finishes only if has_key=1; otherwise stepping on E does nothing special.\n"
            "Actions: 0=UP,1=DOWN,2=LEFT,3=RIGHT,4=JUMP_UP,5=JUMP_DOWN,6=JUMP_LEFT,7=JUMP_RIGHT.\n"
            "Jump moves exactly 2 tiles; ONLY the landing tile matters.\n"
            "Global hints: key_rel=(dy,dx) and exit_rel=(dy,dx) give the direction from agent to the target.\n"
            "Rule: If has_key=0, prioritize moving to the key. If has_key=1, prioritize moving to the exit.\n"
            "\nOUTPUT FORMAT (MANDATORY): ACTION=<integer>\n"
            "Example: ACTION=3\n"
            "No other text."
        )

        # local view + allowed actions only
        allowed = ", ".join(str(a) for a in valid_actions)
        patch_txt = "\n".join(patch)
        
        # format vectors
        def fmt_vec(v):
            if v is None:
                return "none"
            dy, dx, dist = v
            return f"(dy={dy}, dx={dx}, dist={dist})"

        user_prompt = (
            f"has_key={int(has_key)}\n"
            f"allowed_actions=[{allowed}]\n"
            f"key_rel={fmt_vec(key_vec)}\n"
            f"exit_rel={fmt_vec(exit_vec)}\n"
            f"key_in_view={int(key_in_patch)}\n"
            f"exit_in_view={int(exit_in_patch)}\n"
            "view_5x5:\n"
            f"{patch_txt}\n"
            "\nRemember: output exactly ACTION=<0-7> and nothing else."
        )

        return cache_key, system_prompt, user_prompt, key_vec, exit_vec

    def act(self, obs: Dict[str, Any], action_mask: np.ndarray, difficulty: int) -> int:
        valid_actions = _valid_actions_from_mask(action_mask)

        if not valid_actions:
            return 0  # should never happen, but safe guard

        # Build prompt and get state features
        cache_key, system_prompt, user_prompt, key_vec, exit_vec = self.build_prompt(
            obs, action_mask, difficulty
        )

        if self.config.debug:
            _debug_print(
                self.config,
                f"[PROMPT]\n"
                f"CACHE_KEY={cache_key}\n"
                f"VALID_ACTIONS={valid_actions}\n"
                f"HAS_KEY={int(bool(obs['has_key']))}\n"
                f"TARGET_VEC={exit_vec if bool(obs['has_key']) else key_vec}\n\n"
                f"[SYSTEM]\n{system_prompt}\n\n"
                f"[USER]\n{user_prompt}"
            )

        # Choose current target vector
        has_key = bool(obs["has_key"])
        target_vec = exit_vec if has_key else key_vec

        # ---- CACHE CHECK ----
        cached = self.cache.get(cache_key)
        if cached is not None and cached in valid_actions:
            if self.config.debug:
                _debug_print(self.config, f"[CACHE HIT] -> ACTION={cached}")
            return cached

        # ---- CALL MODEL ----
        last_err = None

        for attempt in range(self.config.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_s,
                )

                text = resp.choices[0].message.content or ""

                if self.config.debug:
                    _debug_print(self.config, f"[RAW LLM OUTPUT]\n{text}")

                action = _parse_action_int(text)

                # ---- VALID MODEL OUTPUT ----
                if action is not None and action in valid_actions:
                    # ✅ cache only real model outputs
                    self.cache.set(cache_key, action)

                    if self.config.debug:
                        _debug_print(self.config, f"[PARSED VALID ACTION] ACTION={action}")

                    return action

                # ---- INVALID FORMAT OR INVALID ACTION ----
                if self.config.debug:
                    _debug_print(self.config, "[INVALID FORMAT OR ACTION] -> using fallback")

                # ❌ DO NOT CACHE fallback
                return _choose_fallback(valid_actions, target_vec)

            except Exception as e:
                last_err = e
                if self.config.debug:
                    _debug_print(self.config, f"[API ERROR] {e}")
                time.sleep(self.config.retry_backoff_s * (attempt + 1))

        # ---- API FAILED AFTER RETRIES ----
        if self.config.debug and last_err:
            _debug_print(self.config, "[API FAILED] -> fallback")

        return _choose_fallback(valid_actions, target_vec)
