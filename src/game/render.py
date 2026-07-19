"""Slugline renderer v2: GameState -> np.uint8 [112,160,3], pixel-art edition.

Role: draws the v2 art (src/game/sprites.py palette-indexed sprites) over a
3-layer scene: far parallax (night gradient + stars + moon, scroll 1/4),
mid parallax (ridge/ruin silhouettes, scroll 1/2), and the tile layer
(embossed bricks, metal spikes, big exit flag).  Per-(level, camera) composites
are cached; dynamic entities and HUD are masked sprite blits.
engine / GameState / collect APIs are untouched; deterministic w.r.t. state.
"""

import numpy as np

from . import sprites as S
from .engine import (FP, TILE, LEVEL_W, LEVEL_H, SCREEN_W, SCREEN_H,
                     WEAPON_CD, MELEE_CD, SIN64, A_UP)

# ---------------------------------------------------------------- hud/font
HUD_BG = (10, 10, 14)
TEXT_C = (240, 240, 245)

# 3x5 bitmap font for digits + dash (shared with the racer renderer)
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


# ---------------------------------------------------------------- background
NIGHT_TOP = (16, 16, 38)
NIGHT_BOT = (44, 48, 84)
SIL_C = (20, 20, 40)
VPLAT_C = (108, 172, 224)
TRAP_C = (230, 102, 36)
TRAP_DARK = (120, 52, 22)
B_PLAYER = (255, 255, 140)
B_TAIL = (150, 130, 40)
B_ENEMY = (255, 82, 224)
B_ETAIL = (130, 40, 110)
PICKUP_CHIP = {"mg": (108, 172, 224), "rocket": (230, 102, 36),
               "grenade": (98, 138, 58), "life": (178, 52, 42)}
WEAPON_ICON = {"pistol": S.ICON_PISTOL, "mg": S.ICON_MG,
               "rocket": S.ICON_ROCKET}

FAR_W = SCREEN_W + (LEVEL_W * TILE - SCREEN_W) // 4 + 1     # scroll 1/4
MID_W = SCREEN_W + (LEVEL_W * TILE - SCREEN_W) // 2 + 1     # scroll 1/2
_CHK = (np.indices((TILE, TILE)).sum(0) % 2 == 0)           # dither checker

_LAYERS = {}          # level key -> dict of parallax + tile strips
_COMP = {}            # (level key, cam) -> composed background frame


def _hash2(a, b):
    return (a * 73856093 ^ b * 19349663) & 0x7FFFFFFF


def _build_layers(state, key):
    lid = state.level_id
    # far: night gradient + stars + crescent moon
    f = np.linspace(0.0, 1.0, SCREEN_H)[:, None]
    far = (np.asarray(NIGHT_TOP) * (1 - f) + np.asarray(NIGHT_BOT) * f) \
        .astype(np.uint8)
    far = np.repeat(far[:, None, :], FAR_W, axis=1)
    for i in range(46):                                      # deterministic stars
        h = _hash2(i * 31 + lid * 7, i * 17 + 3)
        x, y = h % FAR_W, (h >> 8) % 68
        far[y, x] = (200, 204, 220) if h & 4 else (150, 154, 180)
        if h & 8 and x + 1 < FAR_W:
            far[y, x + 1] = (120, 124, 150)
    S.MOON.draw(far, 24 + lid * 40, 8)
    # mid: ridge + ruin silhouettes (mask overlay)
    mid = np.zeros((SCREEN_H, MID_W, 3), dtype=np.uint8)
    mmask = np.zeros((SCREEN_H, MID_W), dtype=bool)
    xs = np.arange(MID_W)
    ridge = 64 - (np.take(np.asarray(SIN64), (xs // 3 + lid * 11) % 64) * 10
                  // 256) - (np.take(np.asarray(SIN64), (xs // 7 + 20) % 64)
                             * 6 // 256)
    for dy in range(SCREEN_H):
        mmask[dy, ridge <= dy] = True
    for i in range(6):                                       # ruin towers
        h = _hash2(lid * 13 + i * 71, i * 29 + 5)
        tx, tw, th = h % (MID_W - 20), 8 + h % 9, 22 + (h >> 6) % 18
        top = SCREEN_H - 8 - th
        mmask[top:, tx:tx + tw] = True
        if h & 16:                                           # broken-top notch
            mmask[top:top + 4, tx + tw // 2:tx + tw] = False
        wy = top + 4 + (h >> 3) % 6
        wx = tx + 2 + (h >> 9) % max(1, tw - 4)
        if wy + 2 < SCREEN_H and wx + 1 < MID_W:
            mmask[wy:wy + 2, wx:wx + 2] = False              # window hole
    mid[mmask] = SIL_C
    # tile layer over the full level width (mask overlay)
    w_px = LEVEL_W * TILE
    til = np.zeros((SCREEN_H, w_px, 3), dtype=np.uint8)
    tmask = np.zeros((SCREEN_H, w_px), dtype=bool)
    for ty in range(LEVEL_H):
        for tx in range(LEVEL_W):
            ch = state.tiles[ty][tx]
            x, y = tx * TILE, ty * TILE
            if ch == "#":
                til[y:y + TILE, x:x + TILE] = S.BRICK.rgb
                h = _hash2(tx * 7 + 1, ty * 13 + 1)
                for j in range(2):                           # per-brick noise
                    nx = x + 1 + ((h >> (3 * j)) % 6)
                    ny = y + 1 + ((h >> (3 * j + 7)) % 2) * 4 + j
                    til[ny, nx] = (110, 112, 126)
                tmask[y:y + TILE, x:x + TILE] = True
            elif ch == "^":
                til[y:y + TILE, x:x + TILE][S.SPIKE.mask] = \
                    S.SPIKE.rgb[S.SPIKE.mask]
                tmask[y:y + TILE, x:x + TILE] |= S.SPIKE.mask
    # big exit flag (anchored on the ground below its column)
    gty = LEVEL_H
    for ty in range(LEVEL_H):
        if state.tiles[ty][state.exit_x] == "#":
            gty = ty
            break
    fx, fy = state.exit_x * TILE - 2, gty * TILE - S.EXIT_FLAG.h
    x0, y0 = max(0, fx), max(0, fy)
    sub = S.EXIT_FLAG.mask[:SCREEN_H - y0, :w_px - x0]
    til[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]][sub] = \
        S.EXIT_FLAG.rgb[:sub.shape[0], :sub.shape[1]][sub]
    tmask[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] |= sub
    _LAYERS[key] = {"far": far, "mmask": mmask, "mid": mid,
                    "til": til, "tmask": tmask}
    return _LAYERS[key]


def _background(state):
    """Composite far/mid/tile layers for the current camera (cached)."""
    key = (state.level_id, state.exit_x, hash(tuple(state.tiles)))
    cam = state.camera_x
    comp = _COMP.get((key, cam))
    if comp is not None:
        return comp.copy()
    lay = _LAYERS.get(key) or _build_layers(state, key)
    img = lay["far"][:, cam // 4:cam // 4 + SCREEN_W].copy()
    mm = lay["mmask"][:, cam // 2:cam // 2 + SCREEN_W]
    img[mm] = lay["mid"][:, cam // 2:cam // 2 + SCREEN_W][mm]
    tm = lay["tmask"][:, cam:cam + SCREEN_W]
    img[tm] = lay["til"][:, cam:cam + SCREEN_W][tm]
    if len(_COMP) > 512:
        _COMP.clear()
    _COMP[(key, cam)] = img
    return img.copy()


# ---------------------------------------------------------------- render
def render(state):
    """Render one frame: [112,160,3] uint8."""
    cam = state.camera_x
    img = _background(state)
    t = state.tick
    p = state.player

    # checkpoints (waving flag; green once taken)
    for cp in state.checkpoints:
        frames = S.CP_FLAG_TAKEN if cp["taken"] else S.CP_FLAG
        frames[(t // 8) % 2].draw(img, cp["x"] - cam, cp["y"] - 6)

    # vanishing platforms: solid plank / dithered warning flicker
    for v in state.vplats:
        if v["state"] == "gone":
            continue
        x, y = v["tx"] * TILE - cam, v["ty"] * TILE
        if v["state"] == "solid":
            _rect(img, x, y, TILE, TILE, VPLAT_C)
            _rect(img, x, y, TILE, 1, (210, 240, 255))
            _rect(img, x, y + TILE - 1, TILE, 1, (60, 110, 160))
        elif 0 <= x <= SCREEN_W - TILE:                     # semi-transparent
            chk = _CHK if t % 2 == 0 else ~_CHK
            img[y:y + TILE, x:x + TILE][chk] = VPLAT_C

    # ceiling traps: riveted blood-orange block with jagged teeth
    for tr in state.traps:
        c = TRAP_C if tr["state"] != "done" else TRAP_DARK
        x, y = tr["x"] - cam, tr["y"]
        _rect(img, x, y, tr["w"], tr["h"], c)
        _rect(img, x, y, tr["w"], 1, (255, 170, 90))
        for i in range(0, tr["w"], 4):
            _rect(img, x + i + 1, y + tr["h"], 2, 2, c)
            _rect(img, x + i + 1, y + 2, 1, 1, TRAP_DARK)

    # pickups: supply box + colored chip
    for pk in state.pickups:
        if pk["taken"]:
            continue
        S.PICKUP_BOX.draw(img, pk["x"] - cam, pk["y"])
        _rect(img, pk["x"] - cam + 2, pk["y"] + 2, 4, 4,
              PICKUP_CHIP.get(pk["kind"], TEXT_C))

    # cages: wooden grid + waving captive; opened once rescued
    for cg in state.cages:
        x, y = cg["x"] - cam - 1, cg["y"] - 2
        if cg["rescued"]:
            S.CAGE_OPEN.draw(img, x, y)
        else:
            S.CAGE[(t // 8) % 2].draw(img, x, y)

    # enemies
    for e in state.enemies:
        if e["hp"] <= 0:
            continue
        k = e["kind"]
        x, y = e["x"] // FP - cam, e["y"] // FP
        if k == "walker":
            flip = e["dir"] < 0
            S.WALKER[(e["x"] // (4 * FP)) % 2].draw(img, x - 2, y - 6, flip)
            S.GUN_H.draw(img, x + (7 if not flip else -5), y + 1, flip)
        elif k == "turret":
            S.TURRET.draw(img, x - 2, y - 2)
            bdir = 1 if state.player["x"] > e["x"] else -1
            up = 1 if state.player["y"] + 8 * FP < e["y"] else 0
            _rect(img, x + (8 if bdir > 0 else -4), y - up, 4, 2, (196, 200, 210))
        elif k == "flyer":
            frames = S.FLYER_RED if (e.get("fstate") == "windup" and t % 2 == 0) \
                else S.FLYER
            frames[(t // 6) % 2].draw(img, x - 2, y - 2)
        else:                                               # crate
            S.PICKUP_BOX.draw(img, x, y)
            _rect(img, x + 2, y + 2, 4, 4, (138, 94, 52))

    # grenades (2-frame spin)
    for g in state.grenades:
        S.GRENADE[(t // 3) % 2].draw(img, g["x"] // FP - cam - 1, g["y"] // FP - 1)

    # bullets: bright head + fading 1px trail; rockets get sprite + flame
    for b in state.bullets:
        x, y = b["x"] // FP - cam, b["y"] // FP
        sx = 1 if b["vx"] > 0 else -1 if b["vx"] < 0 else 0
        sy = 1 if b["vy"] > 0 else -1 if b["vy"] < 0 else 0
        if b["kind"] == "rocket":
            S.ROCKET.draw(img, x - 2, y - 1, flip=b["vx"] < 0)
            S.ROCKET_FLAME.draw(img, x - 2 - 3 * sx, y - 1, flip=b["vx"] > 0)
        else:
            hc, tc = (B_ENEMY, B_ETAIL) if b["kind"] == "enemy" \
                else (B_PLAYER, B_TAIL)
            _rect(img, x - 1, y - 1, 2, 2, hc)
            _rect(img, x - 1 - 2 * sx, y - 1 - 2 * sy, 1, 1, tc)
            _rect(img, x - 1 - 3 * sx, y - 1 - 3 * sy, 1, 1, tc)

    # explosions: 4-frame anim (white core -> fireball -> smoke)
    for ex in state.explosions:
        fr = min(3, (6 - ex["ttl"]) * 4 // 6)
        S.EXPLOSION[fr].draw(img, ex["x"] - cam - 6, ex["y"] - 6)

    # player (12x18 soldier; hitbox 6x12 at sprite offset (3,6))
    px, py = p["x"] // FP - cam, p["y"] // FP
    if not p["alive"] and state.death_timer > 0:
        S.PLAYER_DEAD[0 if state.death_timer > 8 else 1].draw(
            img, px - 3, py - 6, flip=p["facing"] < 0)
    elif p["alive"] and not (p["invuln"] > 0 and t % 2 == 0):
        flip = p["facing"] < 0
        aim_up = bool(state.prev_action[A_UP]) and not p["crouch"]
        if p["crouch"]:
            S.PLAYER_CROUCH.draw(img, px - 3, py + 2, flip)
        elif not p["on_ground"]:
            S.PLAYER_JUMP.draw(img, px - 3, py - 6, flip)
        elif p["vx"] != 0:
            S.PLAYER_RUN[(p["x"] // (3 * FP)) % 3].draw(img, px - 3, py - 6, flip)
        else:
            S.PLAYER_STAND.draw(img, px - 3, py - 6, flip)
        # gun overlay: horizontal / diagonal (up-aim while moving) / vertical
        if aim_up and p["vx"] != 0:
            S.GUN_DIAG.draw(img, px + (5 if not flip else -4), py - 3, flip)
            tipx, tipy = px + (9 if not flip else -5), py - 4
        elif aim_up:
            S.GUN_UP.draw(img, px + (2 if not flip else 1), py - 10, flip)
            tipx, tipy = px + 2, py - 12
        else:
            gy = py + (4 if not p["crouch"] else 8)
            S.GUN_H.draw(img, px + (6 if not flip else -5), gy, flip)
            tipx, tipy = px + (11 if not flip else -7), gy
        just_fired = p["shoot_cd"] > 0 and p["shoot_cd"] in (
            WEAPON_CD.get(p["weapon"], 4), MELEE_CD)
        if just_fired:
            S.MUZZLE[t % 2].draw(img, tipx, tipy)

    # HUD
    img[0:8, :] = HUD_BG
    for i in range(max(0, state.lives)):
        S.HEART.draw(img, 2 + i * 8, 1)
    WEAPON_ICON[p["weapon"]].draw(img, 27, 1)
    ammo = p["ammo"].get(p["weapon"], 0)
    ammo_s = "---" if p["weapon"] == "pistol" else str(min(ammo, 999)).rjust(3, "0")
    _text(img, 37, 1, ammo_s, TEXT_C)
    S.GRENADE[0].draw(img, 54, 2)
    _text(img, 60, 1, str(min(p["grenades"], 99)).rjust(2, "0"), TEXT_C)
    _text(img, SCREEN_W - 4 - 6 * 4, 1, str(min(state.score, 999999)).rjust(6, "0"),
          TEXT_C)
    return img
