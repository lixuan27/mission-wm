"""Ridgeline renderer: GameState -> np.uint8[112,160,3] pseudo-3D scanline road.

Role: classic OutRun-style projection — below the horizon each screen row maps
to a ground depth z; road center bends by double-accumulated curvature, width
shrinks with depth, banded colors scroll with distance for the motion illusion.
Road region is painted fully vectorized (numpy mask/palette), sprites (deco,
traffic, player car) and HUD are small clipped rects.  Deterministic in state.
"""

import numpy as np

from game.render import FONT, _rect, _text                  # shared 3x5 font utils
from .engine import (SCREEN_W, SCREEN_H, SEG_LEN, ROAD_HALF, MAX_SPD,
                     TIME_INIT, NITRO_COUNT, seg_at)

# ---------------------------------------------------------------- palette
SKY_TOP = (66, 120, 210)
SKY_BOT = (150, 196, 240)
SUN_C = (255, 236, 150)
MOUNT_C = (58, 74, 92)
GRASS = ((34, 140, 54), (28, 118, 44))
ASPHALT = ((100, 100, 110), (88, 88, 98))
RUMBLE = ((214, 60, 50), (232, 232, 236))
LINE_C = (240, 240, 244)
CAR_SAME = ((202, 52, 40), (62, 112, 222), (232, 184, 52))
CAR_ONC = ((222, 222, 230), (182, 84, 202), (92, 202, 182))
CAR_GLASS = (30, 36, 48)
CAR_WHEEL = (14, 14, 18)
PLAYER_BODY = (235, 40, 40)
PLAYER_DARK = (150, 20, 24)
NITRO_FLAME = (255, 170, 40)
HUD_BG = (10, 10, 14)
TEXT_C = (240, 240, 245)
NITRO_ICON = (70, 170, 255)
PALM_TRUNK = (124, 84, 48)
PALM_LEAF = (24, 96, 40)
BOARD_C = ((238, 210, 60), (70, 210, 220), (226, 90, 200))
POLE_C = (150, 150, 158)

Z_SCALE = 3072            # z(d) = Z_SCALE // d  (track units of depth per row)
W_MAX = 72                # road half-width in px at the bottom row
HY_BASE = 44

_SEG_CACHE = {}           # (level_id, variant_seed) -> np arrays (curve, hill, deco)
_SKY = None               # (112,3) gradient cache
_SUN = None               # sun disk mask cache


def _seg_arrays(state):
    key = (state.level_id, state.variant_seed)
    hit = _SEG_CACHE.get(key)
    if hit is None:
        seg = np.asarray(state.segments, dtype=np.int64)    # (N,3)
        hit = (seg[:, 0].copy(), seg[:, 1].copy(), seg[:, 2].copy())
        _SEG_CACHE[key] = hit
    return hit


def _sky():
    global _SKY, _SUN
    if _SKY is None:
        f = np.linspace(0.0, 1.0, SCREEN_H)[:, None]
        _SKY = (np.asarray(SKY_TOP) * (1 - f) + np.asarray(SKY_BOT) * f) \
            .astype(np.uint8)
        r = 9
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        _SUN = (yy * yy + xx * xx) <= r * r
    return _SKY, _SUN


def render(state):
    """Render one frame: [112,160,3] uint8."""
    p = state.player
    curves, hills, decos = _seg_arrays(state)
    n_seg = len(curves)
    img = np.empty((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    # ---------------- horizon from hill profile (current + lookahead)
    i0 = min(max(p["dist"] // SEG_LEN, 0), n_seg - 1)
    i6 = min(i0 + 6, n_seg - 1)
    h_eff = (2 * int(hills[i0]) + int(hills[i6])) // 3
    hy = HY_BASE - 7 * h_eff                                # 30..58

    # ---------------- sky, sun, mountains
    sky, sun = _sky()
    img[:hy + 1] = sky[:hy + 1, None, :]
    sy, sx = hy - 22, 118
    ys0, ys1 = max(0, sy - 9), min(SCREEN_H, sy + 10)
    m = sun[ys0 - (sy - 9):19 - ((sy + 10) - ys1), :]
    img[ys0:ys1, sx - 9:sx + 10][m] = SUN_C
    shift = (p["dist"] >> 8) + (p["x"] >> 6)                # slow parallax
    from game.engine import SIN64
    idx = ((np.arange(SCREEN_W) + shift) // 5) % 64
    mh = 4 + (np.take(np.asarray(SIN64), idx) + 256) * 8 // 512    # 4..12
    for dy in range(12):
        row = hy - dy
        if 0 <= row < SCREEN_H:
            img[row, mh > dy] = MOUNT_C

    # ---------------- road region (vectorized scanlines)
    dmax = SCREEN_H - 1 - hy
    d = np.arange(1, dmax + 1)                              # top..bottom depth idx
    z = Z_SCALE // d                                        # track units ahead
    dist_row = p["dist"] + z
    seg_idx = np.clip(dist_row // SEG_LEN, 0, n_seg - 1)
    w = np.maximum(W_MAX * d // dmax, 2)                    # half width px
    base = 80 - (p["x"] * W_MAX // ROAD_HALF)
    c_row = np.take(curves, seg_idx)                        # curvature per row
    rev = c_row[::-1] * 2                                   # fp8 accumulation
    cx = base + (np.cumsum(np.cumsum(rev)) >> 8)[::-1]
    parity = ((dist_row // 256) & 1).astype(np.int64)

    X = np.arange(SCREEN_W)[None, :]
    dx = np.abs(X - cx[:, None])
    wcol = w[:, None]
    rw = np.maximum(2, w // 10)[:, None]
    cat = np.zeros((dmax, SCREEN_W), dtype=np.int64)        # 0 grass
    cat[dx <= wcol + rw] = 2                                # rumble
    cat[dx <= wcol] = 1                                     # asphalt
    line = (dx <= np.maximum(1, wcol // 24)) & (parity[:, None] == 0) & (wcol > 10)
    cat[line] = 3
    pal = np.empty((dmax, 4, 3), dtype=np.uint8)
    for ci, cols in ((0, GRASS), (1, ASPHALT), (2, RUMBLE)):
        pal[:, ci] = np.take(np.asarray(cols, dtype=np.uint8), parity, axis=0)
    pal[:, 3] = LINE_C
    img[hy + 1:] = pal[np.arange(dmax)[:, None], cat]

    def row_of(depth):
        """Screen row for a track depth (None if beyond horizon)."""
        if depth <= 0:
            return None
        di = Z_SCALE // max(depth, Z_SCALE // dmax)
        return hy + min(di, dmax)

    def geom(y):
        i = min(max(y - hy - 1, 0), dmax - 1)
        return int(cx[i]), int(w[i])

    # ---------------- roadside deco (far -> near)
    for si in range(min(i0 + 24, n_seg - 1), i0 + 1, -1):
        deco = int(decos[si])
        if deco == 0:
            continue
        y = row_of(si * SEG_LEN - p["dist"])
        if y is None or y <= hy + 1:
            continue
        ccx, cw = geom(y)
        s = max(2, 16 * (y - hy) // dmax)                   # sprite scale px
        side = 1 if deco > 0 else -1
        x = ccx + side * (cw + 6 + s // 2)
        if abs(deco) == 1:                                  # palm
            _rect(img, x - s // 8, y - s, max(1, s // 4), s, PALM_TRUNK)
            _rect(img, x - s // 2, y - s - s // 3, s, s // 3 + 1, PALM_LEAF)
            _rect(img, x - s // 3, y - s - s // 2, 2 * s // 3, s // 4 + 1, PALM_LEAF)
        else:                                               # billboard
            _rect(img, x - s // 8, y - s // 2, max(1, s // 4), s // 2, POLE_C)
            _rect(img, x - s // 2, y - s, s, s // 2, BOARD_C[si % 3])

    # ---------------- traffic (far -> near)
    for car in sorted(state.cars, key=lambda c: p["dist"] - c["dist"]):
        depth = car["dist"] - p["dist"]
        if depth < 20 or depth > Z_SCALE:
            continue
        y = row_of(depth)
        if y is None or y <= hy + 2:
            continue
        ccx, cw = geom(y)
        x = ccx + car["x"] * cw // ROAD_HALF
        s = max(4, 24 * (y - hy) // dmax)                   # car width px
        h = max(2, s * 5 // 12)
        col = (CAR_SAME if car["dir"] > 0 else CAR_ONC)[car["color"] % 3]
        _rect(img, x - s // 2, y - h, s, h, col)
        _rect(img, x - s // 3, y - h, 2 * s // 3, max(1, h // 3), CAR_GLASS)
        _rect(img, x - s // 2, y - 1, max(1, s // 5), 1, CAR_WHEEL)
        _rect(img, x + s // 2 - max(1, s // 5), y - 1, max(1, s // 5), 1, CAR_WHEEL)

    # ---------------- player car (fixed near bottom center)
    t = state.tick
    if not (p["invuln"] > 0 and p["spin"] == 0 and t % 2 == 0):     # flicker
        cx0, cy0 = 80, 105
        steer = state.prev_action[1] - state.prev_action[0]
        frame = (p["spin"] // 5) % 4 if p["spin"] > 0 else 0
        if frame == 2:                                      # sideways
            _rect(img, cx0 - 13, cy0 - 8, 26, 7, PLAYER_BODY)
            _rect(img, cx0 - 13, cy0 - 8, 26, 2, PLAYER_DARK)
        else:
            tilt = (-3 if frame == 3 else 3 if frame == 1 else steer * 2)
            _rect(img, cx0 - 12, cy0 - 5, 24, 5, PLAYER_BODY)
            _rect(img, cx0 - 9 + tilt, cy0 - 9, 18, 4, PLAYER_BODY)
            _rect(img, cx0 - 6 + tilt, cy0 - 9, 12, 2, CAR_GLASS)
            _rect(img, cx0 - 12, cy0 - 1, 5, 3, CAR_WHEEL)
            _rect(img, cx0 + 7, cy0 - 1, 5, 3, CAR_WHEEL)
            _rect(img, cx0 - 9, cy0 - 6, 3, 3, PLAYER_DARK)
            _rect(img, cx0 + 6, cy0 - 6, 3, 3, PLAYER_DARK)
        if p["nitro_t"] > 0 and t % 2 == 0:
            _rect(img, cx0 - 3, cy0 + 2, 6, 3, NITRO_FLAME)
            _rect(img, cx0 - 1, cy0 + 5, 2, 2, SUN_C)

    # ---------------- HUD
    img[0:8, :] = HUD_BG
    kph = p["spd"] * 300 // MAX_SPD
    _text(img, 2, 1, str(min(kph, 999)).rjust(3, "0"), TEXT_C)
    for i in range(NITRO_COUNT):                            # nitro stock
        c = NITRO_ICON if i < p["nitro"] else (50, 50, 60)
        _rect(img, 18 + i * 6, 2, 4, 4, c)
    bar = min(60, 60 * state.timer // TIME_INIT)            # time bar
    bcol = (80, 220, 90) if bar > 20 else (235, 200, 60) if bar > 10 \
        else (235, 60, 50)
    _rect(img, 40, 2, 60, 4, (40, 40, 48))
    _rect(img, 40, 2, bar, 4, bcol)
    _rect(img, 104, 1, 6, 6, (60, 60, 72))                  # stage badge
    _text(img, 105, 1, str(min(state.stage, 9)), TEXT_C)
    _text(img, SCREEN_W - 4 - 6 * 4, 1,
          str(min(state.score, 999999)).rjust(6, "0"), TEXT_C)
    return img
