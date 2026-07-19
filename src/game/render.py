"""Slugline renderer: GameState -> np.uint8[112,160,3] IWBTG-style frame.

Role: pure-numpy procedural sprites (no assets, no SDL).  High-contrast pixel
blocks; static level layer is pre-rendered per (level, variant) and cached,
then dynamic entities + HUD are drawn on top.  Deterministic w.r.t. state.
"""

import numpy as np

from .engine import (FP, TILE, LEVEL_W, LEVEL_H, SCREEN_W, SCREEN_H, PLAYER_W,
                     PLAYER_H, PLAYER_H_CROUCH, ENEMY_SIZE, A_UP)

# ---------------------------------------------------------------- palette
BG_TOP = (24, 26, 40)
BG_BOT = (34, 38, 58)
TILE_FILL = (168, 168, 182)
TILE_EDGE = (92, 92, 108)
SPIKE_C = (224, 224, 232)
FLAG_POLE = (180, 180, 190)
FLAG_C = (70, 220, 90)
CP_OFF = (232, 210, 64)
CP_ON = (96, 196, 255)   # distinct from the green exit flag
PLAYER_BODY = (80, 140, 240)
PLAYER_HEAD = (240, 220, 190)
GUN_C = (235, 235, 240)
WALKER_C = (222, 60, 50)
WALKER_LEG = (150, 34, 30)
TURRET_C = (172, 80, 224)
TURRET_BARREL = (230, 200, 255)
FLYER_C = (242, 212, 64)
CRATE_C = (172, 122, 62)
CRATE_X = (110, 74, 34)
CAGE_BAR = (150, 150, 160)
CAPTIVE_C = (244, 152, 172)
VPLAT_C = (120, 200, 230)
TRAP_C = (204, 84, 84)
B_PLAYER = (255, 255, 120)
B_ROCKET = (255, 140, 40)
B_ENEMY = (255, 82, 224)
GRENADE_C = (128, 232, 84)
EXPL_CORE = (255, 250, 220)
EXPL_RING = (255, 150, 50)
HUD_BG = (10, 10, 14)
HEART_C = (235, 50, 60)
TEXT_C = (240, 240, 245)
PICKUP_C = {"mg": (80, 220, 235), "rocket": (255, 140, 40),
            "grenade": (128, 232, 84), "life": (235, 50, 60)}
WEAPON_ICON = {"pistol": (235, 235, 240), "mg": (80, 220, 235),
               "rocket": (255, 140, 40)}

# 3x5 bitmap font for digits + dash
FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "011", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "-": ("000", "000", "111", "000", "000"),
}

_BG_CACHE = {}


def _rect(img, x, y, w, h, color):
    """Clipped rectangle fill on the frame."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(SCREEN_W, x + w), min(SCREEN_H, y + h)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = color


def _text(img, x, y, s, color):
    """Draw digits/dashes with the 3x5 font (4 px advance)."""
    for ch in s:
        glyph = FONT.get(ch)
        if glyph:
            for r, row in enumerate(glyph):
                for c, bit in enumerate(row):
                    if bit == "1" and 0 <= y + r < SCREEN_H and 0 <= x + c < SCREEN_W:
                        img[y + r, x + c] = color
        x += 4


def _static_layer(state):
    """Pre-render background + solid tiles + spikes + exit flag for whole level."""
    key = (state.level_id, state.variant_seed, state.exit_x, hash(tuple(state.tiles)))
    layer = _BG_CACHE.get(key)
    if layer is not None:
        return layer
    w_px = LEVEL_W * TILE
    layer = np.zeros((SCREEN_H, w_px, 3), dtype=np.uint8)
    for y in range(SCREEN_H):                                   # vertical gradient
        f = y / (SCREEN_H - 1)
        layer[y, :] = [int(a + (b - a) * f) for a, b in zip(BG_TOP, BG_BOT)]
    for ty in range(LEVEL_H):
        for tx in range(LEVEL_W):
            ch = state.tiles[ty][tx]
            x, y = tx * TILE, ty * TILE
            if ch == "#":
                layer[y:y + TILE, x:x + TILE] = TILE_FILL
                layer[y, x:x + TILE] = TILE_EDGE
                layer[y + TILE - 1, x:x + TILE] = TILE_EDGE
                layer[y:y + TILE, x] = TILE_EDGE
                layer[y:y + TILE, x + TILE - 1] = TILE_EDGE
            elif ch == "^":                                     # spike triangle
                for r in range(TILE):
                    half = (TILE - 1 - r) * TILE // (2 * TILE)  # narrower at top
                    x0 = x + TILE // 2 - 1 - half
                    x1 = x + TILE // 2 + 1 + half
                    layer[y + r, max(0, x0):min(w_px, x1)] = SPIKE_C
    # exit flag
    fx = state.exit_x * TILE + 3
    fy = (LEVEL_H - 1) * TILE
    layer[fy - 20:fy, fx:fx + 1] = FLAG_POLE
    layer[fy - 20:fy - 14, fx + 1:fx + 7] = FLAG_C
    _BG_CACHE[key] = layer
    return layer


def render(state):
    """Render one frame: [112,160,3] uint8."""
    cam = state.camera_x
    img = _static_layer(state)[:, cam:cam + SCREEN_W].copy()
    t = state.tick

    # checkpoints (dynamic color)
    for cp in state.checkpoints:
        x = cp["x"] - cam + 3
        y = cp["y"] + TILE
        _rect(img, x, y - 14, 1, 14, FLAG_POLE)
        _rect(img, x + 1, y - 14, 5, 4, CP_ON if cp["taken"] else CP_OFF)

    # vanishing platforms (flicker while vanishing)
    for v in state.vplats:
        if v["state"] == "gone" or (v["state"] == "vanishing" and t % 2 == 0):
            continue
        _rect(img, v["tx"] * TILE - cam, v["ty"] * TILE, TILE, TILE, VPLAT_C)
        _rect(img, v["tx"] * TILE - cam + 1, v["ty"] * TILE + 1, TILE - 2, 1,
              (220, 245, 255))

    # ceiling traps
    for tr in state.traps:
        c = TRAP_C if tr["state"] != "done" else (120, 60, 60)
        _rect(img, tr["x"] - cam, tr["y"], tr["w"], tr["h"], c)
        for i in range(tr["x"], tr["x"] + tr["w"], 4):          # jagged bottom
            _rect(img, i - cam + 1, tr["y"] + tr["h"], 2, 2, c)

    # pickups
    for pk in state.pickups:
        if pk["taken"]:
            continue
        c = PICKUP_C.get(pk["kind"], TEXT_C)
        _rect(img, pk["x"] - cam + 1, pk["y"] + 2, 6, 6, c)
        _rect(img, pk["x"] - cam + 2, pk["y"] + 3, 2, 2, (255, 255, 255))

    # cages (grid bars over captive)
    for cg in state.cages:
        x, y = cg["x"] - cam, cg["y"]
        if cg["rescued"]:
            _rect(img, x, y + 10, 10, 2, CAGE_BAR)              # opened remains
            continue
        _rect(img, x + 3, y + 3, 4, 8, CAPTIVE_C)
        for i in range(0, 11, 3):
            _rect(img, x + i, y, 1, 12, CAGE_BAR)
        _rect(img, x, y, 10, 1, CAGE_BAR)
        _rect(img, x, y + 11, 10, 1, CAGE_BAR)

    # enemies
    for e in state.enemies:
        if e["hp"] <= 0:
            continue
        k = e["kind"]
        x, y = e["x"] // FP - cam, e["y"] // FP
        w, h = ENEMY_SIZE[k]
        if k == "walker":
            _rect(img, x, y, w, h - 3, WALKER_C)
            _rect(img, x + 1, y + 1, 2, 2, (255, 200, 200))     # eye
            ph = (e["x"] // (4 * FP)) % 2                       # leg animation
            _rect(img, x + (0 if ph else 3), y + h - 3, 3, 3, WALKER_LEG)
        elif k == "turret":
            _rect(img, x, y + 2, w, h - 2, TURRET_C)
            bdir = 1 if state.player["x"] > e["x"] else -1
            _rect(img, x + (w if bdir > 0 else -4), y + h // 2, 4, 2, TURRET_BARREL)
            _rect(img, x + 2, y, w - 4, 2, TURRET_BARREL)
        elif k == "flyer":
            _rect(img, x + 2, y, w - 4, h, FLYER_C)             # diamond-ish
            _rect(img, x, y + 2, w, h - 4, FLYER_C)
            _rect(img, x + w // 2 - 1, y + h // 2 - 1, 2, 2, (120, 90, 20))
        else:                                                   # crate
            _rect(img, x, y, w, h, CRATE_C)
            for i in range(w):
                yy0, yy1 = y + i * h // w, y + h - 1 - i * h // w
                if 0 <= x + i < SCREEN_W:
                    if 0 <= yy0 < SCREEN_H:
                        img[yy0, x + i] = CRATE_X
                    if 0 <= yy1 < SCREEN_H:
                        img[yy1, x + i] = CRATE_X

    # grenades
    for g in state.grenades:
        _rect(img, g["x"] // FP - cam - 1, g["y"] // FP - 1, 2, 2, GRENADE_C)

    # bullets (bright 2x2; rockets 3x3)
    for b in state.bullets:
        x, y = b["x"] // FP - cam, b["y"] // FP
        if b["kind"] == "rocket":
            _rect(img, x - 1, y - 1, 3, 3, B_ROCKET)
        elif b["kind"] == "enemy":
            _rect(img, x - 1, y - 1, 2, 2, B_ENEMY)
        else:
            _rect(img, x - 1, y - 1, 2, 2, B_PLAYER)

    # explosions (expanding ring + core)
    for ex in state.explosions:
        r = (7 - ex["ttl"]) * 2
        x, y = ex["x"] - cam, ex["y"]
        _rect(img, x - r, y - r, 2 * r, 2 * r, EXPL_RING)
        _rect(img, x - r // 2, y - r // 2, r, r, EXPL_CORE)

    # player (block guy with gun barrel; invuln flicker)
    p = state.player
    if p["alive"] and not (p["invuln"] > 0 and t % 2 == 0):
        x = p["x"] // FP - cam
        crouch = p["crouch"]
        h = PLAYER_H_CROUCH // FP if crouch else PLAYER_H // FP
        y = p["y"] // FP + (PLAYER_H - (PLAYER_H_CROUCH if crouch else PLAYER_H)) // FP
        w = PLAYER_W // FP
        _rect(img, x, y, w, h, PLAYER_BODY)
        if not crouch:
            _rect(img, x + 1, y, w - 2, 3, PLAYER_HEAD)         # head
        aim_up = state.prev_action[A_UP] and not crouch
        if aim_up:
            _rect(img, x + w // 2 - 1, y - 3, 2, 3, GUN_C)      # barrel up
        else:
            gy = y + (3 if not crouch else 2)
            _rect(img, x + (w if p["facing"] > 0 else -3), gy, 3, 2, GUN_C)

    # HUD top bar
    img[0:8, :] = HUD_BG
    for i in range(max(0, state.lives)):                        # hearts
        _rect(img, 2 + i * 7, 1, 5, 5, HEART_C)
        if 2 + i * 7 + 2 < SCREEN_W:
            img[1, 2 + i * 7 + 2] = HUD_BG                      # notch
    wx = 26                                                     # weapon icon
    _rect(img, wx, 1, 6, 6, WEAPON_ICON[p["weapon"]])
    _rect(img, wx + 1, 2, 4, 4, HUD_BG)
    _rect(img, wx + 2, 3, 2, 2, WEAPON_ICON[p["weapon"]])
    ammo = p["ammo"].get(p["weapon"], 0)
    ammo_s = "---" if p["weapon"] == "pistol" else str(min(ammo, 999)).rjust(3, "0")
    _text(img, wx + 8, 1, ammo_s, TEXT_C)
    _rect(img, 54, 2, 4, 4, GRENADE_C)                          # grenade count
    _text(img, 60, 1, str(min(p["grenades"], 99)).rjust(2, "0"), TEXT_C)
    _text(img, SCREEN_W - 4 - 6 * 4, 1, str(min(state.score, 999999)).rjust(6, "0"),
          TEXT_C)
    return img
