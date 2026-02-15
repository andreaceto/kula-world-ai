import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# Channel mapping (IMPORTANT): these are one-hot CHANNEL indices, not tile ids.
# 0=VOID, 1=FLOOR, 2=START, 3=EXIT, 4=SPIKE, 5=KEY, 6=COIN
CH_EXIT = 3
CH_KEY = 5


@dataclass
class DeepSeekRewardConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_tokens: int = 16
    timeout_s: int = 30
    max_retries: int = 3
    retry_backoff_s: float = 0.75
    debug: bool = False


class _Cache:
    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self._d: Dict[str, float] = {}

    def get(self, k: str) -> Optional[float]:
        return self._d.get(k)

    def set(self, k: str, v: float) -> None:
        if len(self._d) >= self.max_size:
            # drop ~10%
            n_drop = max(1, self.max_size // 10)
            for key in list(self._d.keys())[:n_drop]:
                self._d.pop(key, None)
        self._d[k] = v


def _onehot_to_channels(grid_onehot: np.ndarray) -> np.ndarray:
    # grid_onehot: (C,H,W) -> (H,W) channel index
    return np.argmax(grid_onehot, axis=0).astype(np.int32)


def _find_nearest(ch_map: np.ndarray, target_ch: int, y: int, x: int) -> Optional[Tuple[int, int, int, int, int]]:
    ys, xs = np.where(ch_map == target_ch)
    if len(ys) == 0:
        return None
    dy = ys - y
    dx = xs - x
    dist = np.abs(dy) + np.abs(dx)
    i = int(np.argmin(dist))
    ty, tx = int(ys[i]), int(xs[i])
    return (ty, tx, int(dy[i]), int(dx[i]), int(dist[i]))


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _cache_key(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parse_reward(text: str) -> Optional[float]:
    """
    Accept ONLY a strict format: REWARD=<float>
    No extra text allowed (to avoid poisoning).
    """
    if not text:
        return None
    t = text.strip()
    m = re.fullmatch(r"REWARD\s*=\s*([-+]?\d+(\.\d+)?)", t, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


class LLMRewardModel:
    """
    Scores transitions. Reward range recommended: [-1, +1]
    If API fails or output invalid => returns 0.0 (neutral).
    """

    def __init__(self, cfg: Optional[DeepSeekRewardConfig] = None):
        self.cfg = cfg or DeepSeekRewardConfig()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY. Put it in .env and load via python-dotenv.")
        self.client = OpenAI(api_key=api_key, base_url=self.cfg.base_url)
        self.cache = _Cache()

        # simple counters for debug/summary
        self.api_calls = 0
        self.cache_hits = 0
        self.parse_fails = 0
        self.timeouts_or_errors = 0

    def score_transition(
        self,
        obs: Dict[str, Any],
        action: int,
        next_obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        # Extract features
        y0, x0 = map(int, obs["agent_pos"])
        y1, x1 = map(int, next_obs["agent_pos"])
        has_key0 = int(bool(obs["has_key"]))
        has_key1 = int(bool(next_obs["has_key"]))
        event = str(info.get("event", "none"))

        ch0 = _onehot_to_channels(np.asarray(obs["grid"]))
        ch1 = _onehot_to_channels(np.asarray(next_obs["grid"]))

        key0 = _find_nearest(ch0, CH_KEY, y0, x0)  # (ty,tx,dy,dx,dist)
        key1 = _find_nearest(ch1, CH_KEY, y1, x1)
        ex0 = _find_nearest(ch0, CH_EXIT, y0, x0)
        ex1 = _find_nearest(ch1, CH_EXIT, y1, x1)

        # Distances (use big number if not present)
        key_dist0 = key0[4] if key0 else 999
        key_dist1 = key1[4] if key1 else 999
        ex_dist0 = ex0[4] if ex0 else 999
        ex_dist1 = ex1[4] if ex1 else 999

        # Define "current target": key if we don't have it, else exit
        if has_key0 == 0:
            target = "KEY"
            d0, d1 = key_dist0, key_dist1
        else:
            target = "EXIT"
            d0, d1 = ex_dist0, ex_dist1

        delta = int(d0) - int(d1)  # >0 means we got closer

        # Cache (keep payload tiny)
        payload = {
            "hk0": has_key0,
            "hk1": has_key1,
            "a": int(action),
            "ev": event,
            "t": target,
            "d0": int(d0),
            "d1": int(d1),
            "del": int(delta),
        }
        ck = _cache_key(payload)
        cached = self.cache.get(ck)
        if cached is not None:
            self.cache_hits += 1
            return cached

        system = (
            "You are a reward model for a grid game.\n"
            "Return ONLY: REWARD=<number>\n"
            "Number must be between -1 and 1.\n"
            "No other text."
        )

        # Keep prompt extremely short and structured
        user = (
            f"has_key_before={has_key0}\n"
            f"has_key_after={has_key1}\n"
            f"event={event}\n"
            f"action={int(action)}\n"
            f"target={target}\n"
            f"dist_before={int(d0)}\n"
            f"dist_after={int(d1)}\n"
            f"delta={int(delta)}\n"
            "Guidelines:\n"
            "- Success event => +1\n"
            "- Death event => -1\n"
            "- If delta>0 (closer) => positive\n"
            "- If delta<0 (farther) => negative\n"
            "- Small penalty for no progress (delta=0)\n"
            "Output only REWARD=<number>."
        )

        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                self.api_calls += 1
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=self.cfg.timeout_s,
                )
                text = (resp.choices[0].message.content or "").strip()
                r = _parse_reward(text)
                if r is None:
                    self.parse_fails += 1
                    if self.cfg.debug:
                        print(f"[LLM REWARD PARSE FAIL] raw='{text}' payload={payload}")
                    return 0.0  # neutral
                r = _clip(float(r), -1.0, 1.0)
                self.cache.set(ck, r)
                return r
            except Exception as e:
                last_err = e
                self.timeouts_or_errors += 1
                time.sleep(self.cfg.retry_backoff_s * (attempt + 1))

        if self.cfg.debug:
            print(f"[LLM REWARD ERROR] {last_err} payload={payload}")
        return 0.0
