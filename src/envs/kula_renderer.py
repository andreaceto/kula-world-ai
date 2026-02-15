from __future__ import annotations

import os
from typing import Dict, Tuple, Optional, Any

import pygame
import numpy as np


# Keep tile IDs consistent with env
VOID  = 0
FLOOR = 1
START = 2
EXIT  = 4
SPIKE = 5
KEY   = 6
COIN  = 7


# Direction names for sprites
DIR_UP = "up"
DIR_DOWN = "down"
DIR_LEFT = "left"
DIR_RIGHT = "right"


class KulaRenderer:
    """
    Pygame renderer for KulaWorld.

    Expected render(state) input:
      state = {
        "grid": np.ndarray(H,W) ints,
        "agent_pos": (y,x),
        "agent_dir": "up"/"down"/"left"/"right"  (optional; default right),
        "has_key": bool,
        "step_count": int,
        "max_steps": int,
        "score": int|float|None
      }
    """

    def __init__(
        self,
        assets_dir: str = "assets",
        tile_px: int = 32,
        hud_h: int = 60,
        fps: int = 30,
        window_caption: str = "Kula World",
    ):
        pygame.init()
        pygame.font.init()

        self.assets_dir = assets_dir
        self.tile_px = int(tile_px)
        self.hud_h = int(hud_h)
        self.fps = int(fps)

        self.clock = pygame.time.Clock()
        self.screen: Optional[pygame.Surface] = None
        self.caption = window_caption

        # Fonts (keep simple & readable)
        self.font = pygame.font.SysFont("arial", 20)
        self.font_small = pygame.font.SysFont("arial", 16)

        # Cache for fallback colored surfaces by key/size
        self._fallback_cache: Dict[Tuple[str, int, int], pygame.Surface] = {}

        # Asset dicts
        self.tiles: Dict[str, pygame.Surface] = {}
        self.items: Dict[str, pygame.Surface] = {}
        self.player: Dict[str, pygame.Surface] = {}
        self.hud: Dict[str, pygame.Surface] = {}

        # Load all assets (with safe fallbacks)
        self._load_assets()

        self._last_grid_shape: Optional[Tuple[int, int]] = None

    # ----------------------------
    # Public API
    # ----------------------------
    def render(self, state: Dict[str, Any]) -> None:
        # Handle window events (so the OS doesn't think the app hung)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        grid: np.ndarray = state["grid"]
        H, W = grid.shape
        agent_y, agent_x = state["agent_pos"]
        agent_dir = state.get("agent_dir", DIR_RIGHT)
        has_key = bool(state.get("has_key", False))
        step_count = int(state.get("step_count", 0))
        max_steps = int(state.get("max_steps", 0))
        score = state.get("score", None)

        # Create/recreate window if size changed (dynamic resolution)
        if self.screen is None or self._last_grid_shape != (H, W):
            win_w = W * self.tile_px
            win_h = H * self.tile_px + self.hud_h
            self.screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption(self.caption)
            self._last_grid_shape = (H, W)

            # Scale assets to current tile size (reload scaled)
            self._load_assets(scale=True)

        assert self.screen is not None

        # --- Layer 1: background (void)
        self.screen.fill((10, 10, 10))

        # --- Layer 2: grid tiles + objects
        for y in range(H):
            for x in range(W):
                tile_id = int(grid[y, x])
                px = x * self.tile_px
                py = y * self.tile_px

                if tile_id == VOID:
                    continue  # void stays background
                
                self._blit(self.tiles.get("floor"), px, py, fallback_key="floor", fallback_color=(60, 60, 60))

                # Base floor for any non-void cell
                if tile_id == FLOOR:
                    self._blit(self.tiles.get("floor"), px, py, fallback_key="floor", fallback_color=(60, 60, 60))
                # Overlay objects
                elif tile_id == START:
                    self._blit(self.tiles.get("start"), px, py, fallback_key="start", fallback_color=(120, 120, 255))
                elif tile_id == EXIT:
                    if has_key:
                        self._blit(self.tiles.get("exit_unlocked"), px, py, fallback_key="exit_unlocked", fallback_color=(0, 120, 255))
                    else:
                        self._blit(self.tiles.get("exit_locked"), px, py, fallback_key="exit_locked", fallback_color=(255, 80, 80))
                elif tile_id == KEY:
                    self._blit(self.items.get("key"), px, py, fallback_key="key", fallback_color=(255, 215, 0))
                elif tile_id == COIN:
                    self._blit(self.items.get("coin"), px, py, fallback_key="coin", fallback_color=(255, 255, 0))
                elif tile_id == SPIKE:
                    self._blit(self.items.get("spike"), px, py, fallback_key="spike", fallback_color=(180, 0, 180))
                else:
                    # Unknown tile: show a neutral marker
                    self._blit(None, px, py, fallback_key="unknown", fallback_color=(0, 0, 0))

        # --- Layer 3: agent
        apx = agent_x * self.tile_px
        apy = agent_y * self.tile_px
        sprite = self.player.get(agent_dir)
        self._blit(sprite, apx, apy, fallback_key=f"player_{agent_dir}", fallback_color=(0, 200, 0))

        # --- Layer 4: HUD
        self._draw_hud(score=score, step_count=step_count, max_steps=max_steps, has_key=has_key)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
        self.screen = None

    # ----------------------------
    # Asset loading
    # ----------------------------
    def _load_assets(self, scale: bool = False) -> None:
        """
        Loads .png assets; missing ones are handled via fallback at blit time.
        If scale=True, scale loaded surfaces to tile_px.
        """
        def load_png(path: str) -> Optional[pygame.Surface]:
            if not os.path.exists(path):
                return None
            try:
                img = pygame.image.load(path).convert_alpha()
                if scale:
                    img = pygame.transform.smoothscale(img, (self.tile_px, self.tile_px))
                return img
            except Exception:
                return None

        # Tiles
        self.tiles["floor"] = load_png(os.path.join(self.assets_dir, "tiles", "floor.png"))
        self.tiles["start"] = load_png(os.path.join(self.assets_dir, "tiles", "start.png"))
        self.tiles["exit_locked"] = load_png(os.path.join(self.assets_dir, "tiles", "exit_locked.png"))
        self.tiles["exit_unlocked"] = load_png(os.path.join(self.assets_dir, "tiles", "exit_unlocked.png"))

        # Items
        self.items["coin"] = load_png(os.path.join(self.assets_dir, "items", "coin.png"))
        self.items["spike"] = load_png(os.path.join(self.assets_dir, "items", "spikes.png"))
        self.items["key"] = load_png(os.path.join(self.assets_dir, "items", "key.png"))

        # Player (directional)
        self.player[DIR_UP] = load_png(os.path.join(self.assets_dir, "player", "player_facing_up.png"))
        self.player[DIR_DOWN] = load_png(os.path.join(self.assets_dir, "player", "player_facing_down.png"))
        self.player[DIR_LEFT] = load_png(os.path.join(self.assets_dir, "player", "player_facing_left.png"))
        self.player[DIR_RIGHT] = load_png(os.path.join(self.assets_dir, "player", "player_facing_right.png"))

        # HUD icons (optional; renderer works without them)
        self.hud["player_icon"] = load_png(os.path.join(self.assets_dir, "hud", "hud_player.png"))
        self.hud["key_icon"] = load_png(os.path.join(self.assets_dir, "items", "key.png"))

        # If scaling changed, clear fallback cache because sizes changed
        if scale:
            self._fallback_cache.clear()

    # ----------------------------
    # Drawing helpers
    # ----------------------------
    def _fallback_surface(self, key: str, color: Tuple[int, int, int]) -> pygame.Surface:
        cache_key = (key, self.tile_px, self.tile_px)
        if cache_key in self._fallback_cache:
            return self._fallback_cache[cache_key]
        surf = pygame.Surface((self.tile_px, self.tile_px), pygame.SRCALPHA)
        surf.fill(color)
        # simple border for visibility
        pygame.draw.rect(surf, (0, 0, 0), surf.get_rect(), 2)
        self._fallback_cache[cache_key] = surf
        return surf

    def _blit(
        self,
        surf: Optional[pygame.Surface],
        x: int,
        y: int,
        fallback_key: str,
        fallback_color: Tuple[int, int, int],
    ) -> None:
        assert self.screen is not None
        if surf is None:
            surf = self._fallback_surface(fallback_key, fallback_color)
        self.screen.blit(surf, (x, y))

    def _draw_hud(self, score: Any, step_count: int, max_steps: int, has_key: bool) -> None:
        assert self.screen is not None

        W_px = self.screen.get_width()
        H_px = self.screen.get_height()

        hud_y = H_px - self.hud_h
        hud_rect = pygame.Rect(0, hud_y, W_px, self.hud_h)

        # HUD background
        pygame.draw.rect(self.screen, (25, 25, 25), hud_rect)
        pygame.draw.line(self.screen, (60, 60, 60), (0, hud_y), (W_px, hud_y), 2)

        # Left: player icon + score
        x_left = 10
        y_mid = hud_y + self.hud_h // 2

        icon = self.hud.get("player_icon")
        if icon is not None:
            icon_small = pygame.transform.smoothscale(icon, (32, 32))
            self.screen.blit(icon_small, (x_left, y_mid - 16))
            x_left += 40

        score_txt = "Score: -" if score is None else f"Score: {score}"
        text = self.font.render(score_txt, True, (230, 230, 230))
        self.screen.blit(text, (x_left, y_mid - text.get_height() // 2))

        # Center: timer (remaining steps)
        remaining = max(0, max_steps - step_count) if max_steps > 0 else 0
        timer_color = (230, 230, 230)
        if remaining < 30:
            timer_color = (255, 80, 80)

        timer_txt = f"Time: {remaining}"
        timer_surface = self.font.render(timer_txt, True, timer_color)
        self.screen.blit(
            timer_surface,
            (W_px // 2 - timer_surface.get_width() // 2, y_mid - timer_surface.get_height() // 2),
        )

        # Right: key status
        status_txt = "COLLECTED" if has_key else "NEED KEY"
        status_color = (255, 215, 0) if has_key else (160, 160, 160)

        # optional key icon
        x_right = W_px - 10
        key_icon = self.hud.get("key_icon")
        if key_icon is not None:
            key_small = pygame.transform.smoothscale(key_icon, (28, 28))
            x_right -= 28
            self.screen.blit(key_small, (x_right, y_mid - 14))
            x_right -= 10

        status_surface = self.font.render(status_txt, True, status_color)
        x_right -= status_surface.get_width()
        self.screen.blit(status_surface, (x_right, y_mid - status_surface.get_height() // 2))
