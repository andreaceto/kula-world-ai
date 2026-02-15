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


# -----------------------------
# Tile mapping (from your kula_env)
# -----------------------------
# VOID=0, FLOOR=1, START=2, EXIT=4, SPIKE=5, KEY=6, COIN=7
TILE_ID_TO_CHAR = {
    0: "_",   # VOID
    1: ".",   # FLOOR
    2: "S",   # START
    4: "E",   # EXIT
    5: "^",   # SPIKE
    6: "K",   # KEY
    7: "C",   # COIN
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
                ch = TILE_ID_TO_CHAR.get(tid, "?")
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
    """
    Accepts outputs like:
    - "3"
    - "Action: 3"
    - "RIGHT (3)"
    - "JUMP_RIGHT"
    """
    if not text:
        return None

    t = text.strip().upper()

    # direct name
    if t in ACTION_NAME_TO_ID:
        return ACTION_NAME_TO_ID[t]

    # name inside text
    for name, aid in ACTION_NAME_TO_ID.items():
        if name in t:
            return aid

    # integer anywhere
    m = re.search(r"\b([0-7])\b", t)
    if m:
        return int(m.group(1))

    return None


def _stable_cache_key(
    has_key: bool,
    agent_pos: Tuple[int, int],
    patch_lines: List[str],
    valid_actions: List[int],
) -> str:
    payload = {
        "k": bool(has_key),
        "p": [int(agent_pos[0]), int(agent_pos[1])],
        "patch": patch_lines,
        "va": valid_actions,
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    ) -> Tuple[str, str, str, str]:
        """
        Returns: (cache_key, system_prompt, user_prompt, debug_patch_text)
        """

        grid = np.asarray(obs["grid"])
        agent_y, agent_x = map(int, obs["agent_pos"])
        has_key = bool(obs["has_key"])

        tile_ids = _obs_to_tile_ids(grid)
        patch = _local_patch(tile_ids, agent_y, agent_x, radius=2)

        valid_actions = _valid_actions_from_mask(action_mask)
        cache_key = _stable_cache_key(
            has_key=has_key,
            agent_pos=(agent_y, agent_x),
            patch_lines=patch,
            valid_actions=valid_actions,
        )

        system_prompt = (
            "You are a policy for a 2D grid game. Choose the best NEXT action.\n"
            "The 5x5 view is centered on the agent '@'. Top row is UP (north). Left is LEFT (west).\n"
            "Symbols: . = floor, _ = void (death), ^ = spike (death), K = key, E = exit, # = out-of-bounds.\n"
            "Exit only finishes if has_key=1; otherwise E behaves like a normal tile.\n"
            "Actions:\n"
            "0=UP, 1=DOWN, 2=LEFT, 3=RIGHT,\n"
            "4=JUMP_UP, 5=JUMP_DOWN, 6=JUMP_LEFT, 7=JUMP_RIGHT.\n"
            "A jump moves exactly 2 tiles; ONLY the landing tile matters.\n"
            "You MUST output ONLY ONE integer that is in allowed_actions. No extra text.\n"
            "Tie-break: prefer moves (0-3) over jumps (4-7), then choose the smallest action id."
        )

        # local view + allowed actions only
        allowed = ", ".join(str(a) for a in valid_actions)
        patch_txt = "\n".join(patch)
        
        user_prompt = (
            f"has_key={int(has_key)}\n"
            f"allowed_actions=[{allowed}]\n"
            "view_5x5:\n"
            f"{patch_txt}\n"
            "Goal rule:\n"
            "- If has_key=0, move toward K if visible.\n"
            "- If has_key=1, move toward E if visible.\n"
            "- Otherwise, choose a safe action that increases open floor and avoids '_' and '#'.\n"
            "Answer: one integer."
        )

        return cache_key, system_prompt, user_prompt, patch_txt

    def act(self, obs: Dict[str, Any], action_mask: np.ndarray, difficulty: int) -> int:
        valid_actions = _valid_actions_from_mask(action_mask)

        # ultimate fallback: if something is weird, pick first valid
        if not valid_actions:
            return 0

        cache_key, system_prompt, user_prompt, _ = self.build_prompt(obs, action_mask, difficulty)
        
        if self.config.debug:
            _debug_print(
                self.config,
                f"[CACHE KEY]\n{cache_key}\n\n"
                f"[SYSTEM PROMPT]\n{system_prompt}\n\n"
                f"[USER PROMPT]\n{user_prompt}"
            )

        cached = self.cache.get(cache_key)
        if cached is not None and cached in valid_actions:
            if self.config.debug:
                _debug_print(
                    self.config,
                    f"[CACHE HIT] -> Action {cached}"
                )
            return cached

        # Call DeepSeek with retries
        last_err: Optional[Exception] = None
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
                    _debug_print(
                        self.config,
                        f"[RAW LLM OUTPUT]\n{text}"
                    )

                action = _parse_action_int(text)

                if self.config.debug:
                    _debug_print(
                        self.config,
                        f"[PARSED ACTION]\n{action}\nVALID ACTIONS: {valid_actions}"
                    )

                if action in valid_actions:
                    self.cache.set(cache_key, action)
                    return action

                # If invalid, pick a safe fallback among valid actions:
                # prefer moves 0-3, else any valid.
                for a in [0, 1, 2, 3]:
                    if a in valid_actions:
                        self.cache.set(cache_key, a)
                        return a
                self.cache.set(cache_key, valid_actions[0])
                return valid_actions[0]

            except Exception as e:
                last_err = e
                # backoff
                time.sleep(self.config.retry_backoff_s * (attempt + 1))

        # If API keeps failing, degrade gracefully
        if last_err:
            # prefer moves
            for a in [0, 1, 2, 3]:
                if a in valid_actions:
                    return a
        return valid_actions[0]
