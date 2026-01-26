import pygame
import os
import numpy as np

# Configuration
TILE_SIZE = 40
HUD_HEIGHT = 60  # Extra space at the bottom for UI

class KulaRenderer:
    def __init__(self, width, height, caption="Kula World AI"):
        pygame.init()
        pygame.display.set_caption(caption)
        
        self.grid_width = width
        self.grid_height = height
        self.window_width = width * TILE_SIZE
        self.window_height = (height * TILE_SIZE) + HUD_HEIGHT
        
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        
        # --- Asset Loading ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.asset_dir = os.path.join(project_root, "assets")
        # --- Asset Loading ---
        self.assets = {}
        self._load_assets()

    def _load_assets(self):
        """Loads images using paths relative to the project root."""
        
        # Helper to join paths correctly
        def get_path(category, filename):
            return os.path.join(self.asset_dir, category, filename)

        # Mapping: logical_name -> (full_path, fallback_color)
        asset_map = {
            # Player
            "player_up":    (get_path("player", "player_facing_up.png"), (0, 191, 255)),
            "player_down":  (get_path("player", "player_facing_down.png"), (0, 191, 255)),
            "player_left":  (get_path("player", "player_facing_left.png"), (0, 191, 255)),
            "player_right": (get_path("player", "player_facing_right.png"), (0, 191, 255)),
            
            # Items
            "spike":        (get_path("items", "spikes.png"), (128, 0, 128)),
            "coin":         (get_path("items", "coin.png"), (255, 255, 0)),
            "key":          (get_path("items", "key.png"), (255, 215, 0)),
            
            # Tiles
            "floor":        (get_path("tiles", "floor.png"), (200, 200, 200)),
            "start":        (get_path("tiles", "start.png"), (0, 255, 0)),
            "exit_locked":  (get_path("tiles", "exit_locked.png"), (255, 0, 0)),
            "exit_open":    (get_path("tiles", "exit_unlocked.png"), (0, 0, 255)),
            
            # HUD
            "hud_heart":       (get_path("hud", "hud_heart.png"), (255, 50, 50)),
            "hud_heart_empty": (get_path("hud", "hud_heart_empty.png"), (50, 0, 0)),
            "hud_player":      (get_path("hud", "hud_player.png"), (0, 191, 255))
        }

        for name, (path, color) in asset_map.items():
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    # Scale logic
                    if "hud" in name:
                        scale = 32
                    else:
                        scale = TILE_SIZE
                    
                    img = pygame.transform.scale(img, (scale, scale))
                    self.assets[name] = img
                except Exception as e:
                    print(f"Error loading {path}: {e}. Using fallback.")
                    self.assets[name] = self._create_fallback(color)
            else:
                # Debug print to help you verify paths
                print(f"Warning: Asset missing at {path}")
                self.assets[name] = self._create_fallback(color)

    def _create_fallback(self, color, name):
        """Creates a colored square if image is missing."""
        surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
        surface.fill(color)
        # Add a border to distinguish tiles
        pygame.draw.rect(surface, (50, 50, 50), surface.get_rect(), 1)
        return surface

    def render(self, env):
        """
        Main render loop. 
        Args:
            env: The KulaWorldEnv instance to read state from.
        """
        self.window.fill((10, 10, 10)) # Dark background (Void)

        # 1. Draw Grid & Objects
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_id = env.grid[y, x]
                
                # Calculate pixel position
                pos_x = x * TILE_SIZE
                pos_y = y * TILE_SIZE
                
                # Determine what to draw
                if cell_id != 0: # If not EMPTY/VOID
                    # Always draw floor first for background of the tile
                    self.window.blit(self.assets["floor"], (pos_x, pos_y))

                    # Draw objects on top
                    if cell_id == 3: # START
                        self.window.blit(self.assets["start"], (pos_x, pos_y))
                    elif cell_id == 5: # SPIKE
                        self.window.blit(self.assets["spike"], (pos_x, pos_y))
                    elif cell_id == 6: # KEY
                        self.window.blit(self.assets["key"], (pos_x, pos_y))
                    elif cell_id == 7: # COIN
                        self.window.blit(self.assets["coin"], (pos_x, pos_y))
                    elif cell_id == 4: # EXIT
                        texture = "exit_open" if env.has_key else "exit_locked"
                        self.window.blit(self.assets[texture], (pos_x, pos_y))

        # 2. Draw Agent
        agent_y, agent_x = env.agent_pos
        dy, dx = env.agent_dir
        
        # Determine facing sprite
        if dy == -1: sprite_key = "player_up"
        elif dy == 1: sprite_key = "player_down"
        elif dx == -1: sprite_key = "player_left"
        else: sprite_key = "player_right"
        
        self.window.blit(self.assets[sprite_key], (agent_x * TILE_SIZE, agent_y * TILE_SIZE))

        # 3. Draw HUD
        self._render_hud(env)

        # Update Display
        pygame.display.flip()
        self.clock.tick(30) # Capped at 30 FPS for smooth rendering

    def _render_hud(self, env):
        hud_y = self.grid_height * TILE_SIZE
        
        # Draw HUD Background
        pygame.draw.rect(self.window, (30, 30, 40), (0, hud_y, self.window_width, HUD_HEIGHT))
        pygame.draw.line(self.window, (255, 255, 255), (0, hud_y), (self.window_width, hud_y), 2)

        # -- Left: Lives --
        start_x = 20
        # Draw 3 hearts (full or empty based on lives)
        max_lives = 3
        for i in range(max_lives):
            heart_type = "hud_heart" if i < env.lives else "hud_heart_empty"
            self.window.blit(self.assets[heart_type], (start_x + (i * 40), hud_y + 14))
        
        # -- Center: Timer --
        time_left = env.max_time - env.current_time
        time_color = (255, 255, 255)
        if time_left < 30: time_color = (255, 50, 50) # Red warning
        
        time_surf = self.font.render(f"TIME: {time_left}", True, time_color)
        time_rect = time_surf.get_rect(center=(self.window_width // 2, hud_y + 30))
        self.window.blit(time_surf, time_rect)

        # -- Right: Inventory (Key & Score) --
        # We assume score isn't tracked in Env yet, but we show Key status
        key_x = self.window_width - 150
        
        if env.has_key:
            self.window.blit(self.assets["key"], (key_x, hud_y + 10))
            key_text = "COLLECTED"
            key_col = (255, 215, 0)
        else:
            # Draw faded key or text
            key_text = "NEED KEY"
            key_col = (150, 150, 150)
            
        text_surf = self.font.render(key_text, True, key_col)
        self.window.blit(text_surf, (key_x + 45, hud_y + 20))

    def close(self):
        pygame.quit()