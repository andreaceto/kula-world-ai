import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Import action ids
from src.envs.kula_env import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
    JUMP_UP, JUMP_DOWN, JUMP_LEFT, JUMP_RIGHT,
    VOID, SPIKE,
)

_ACTIONS = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, JUMP_UP, JUMP_DOWN, JUMP_LEFT, JUMP_RIGHT]

# dy, dx, is_jump
_DELTAS = {
    MOVE_UP: (-1, 0, False),
    MOVE_DOWN: (1, 0, False),
    MOVE_LEFT: (0, -1, False),
    MOVE_RIGHT: (0, 1, False),
    JUMP_UP: (-1, 0, True),
    JUMP_DOWN: (1, 0, True),
    JUMP_LEFT: (0, -1, True),
    JUMP_RIGHT: (0, 1, True),
}

def kula_action_mask(env) -> np.ndarray:
    """
    Returns a boolean mask of valid actions for MaskablePPO.
    Valid = stays in bounds AND landing tile is not VOID/SPIKE.
    Optional curriculum action unlock: disable jumps until difficulty >= 2.
    """
    e = env.unwrapped  # get the real env
    H, W = e.H, e.W
    y, x = int(e.agent_pos[0]), int(e.agent_pos[1])

    mask = np.zeros((8,), dtype=np.bool_)

    # Optional: "unlock" jumps only from L2 onward
    allow_jumps = (int(e.current_difficulty) >= 2)

    for a in _ACTIONS:
        dy, dx, is_jump = _DELTAS[a]
        if is_jump and not allow_jumps:
            mask[a] = False
            continue

        step_len = 2 if is_jump else 1
        ny = y + dy * step_len
        nx = x + dx * step_len

        # Out of bounds => invalid
        if not (0 <= ny < H and 0 <= nx < W):
            mask[a] = False
            continue

        landing = int(e.grid[ny, nx])

        # Landing on hazard => invalid
        if landing in (VOID, SPIKE):
            mask[a] = False
            continue

        # Otherwise valid
        mask[a] = True

    # Safety: if everything is masked (should be rare), allow all moves (not jumps)
    if not mask.any():
        mask[MOVE_UP] = True
        mask[MOVE_DOWN] = True
        mask[MOVE_LEFT] = True
        mask[MOVE_RIGHT] = True

    return mask
