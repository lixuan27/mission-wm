"""Ridgeline renderer v2: GameState -> np.uint8[112,160,3], polished pixel art.

Role: OutRun-style scanline pseudo-3D with per-stage moods (day / sunset /
night with lamp glows), 2-layer parallax clouds, jagged ridge silhouette,
sprite traffic (sedan/truck/beetle x 3 depth tiers), 3-view player car with
spin-out frames, palms/billboards/milestones, and a checkered finish banner.
Road region is fully vectorized; sky blocks and strips are cached per stage.
Deterministic w.r.t. state; engine API untouched.
"""

import numpy as np

from game.render import FONT, _rect                         # shared HUD helpers
from . import sprites as RS
from .engine import (SCREEN_W, SCREEN_H, SEG_LEN, ROAD_HALF, MAX_SPD,
                     TIME_INIT, NITRO_COUNT, TRACK_SEGS)

# ---------------------------------------------------------------- stage moods
STAGE_PAL = {
    1: {"sky": ((96, 156, 224), (176, 212, 242)),           # day
        "grass": ((34, 140, 54), (28, 118, 44)),
        "asphalt": ((100, 100, 110), (88, 88, 98)),
        "rumble": ((214, 60, 50), (232, 232, 236)),
        "mount": (56, 84, 104), "cloud": (250, 250, 252),
        "celestial": RS.SUN, "stars": False, "lamps": False},
    2: {"sky": ((122, 72, 148), (244, 158, 84)),            # sunset
        "grass": ((108, 118, 44), (90, 100, 36)),
        "asphalt": ((96, 88, 100), (84, 76, 88)),
        "rumble": ((200, 70, 50), (240, 214, 170)),
        "mount": (60, 42, 76), "cloud": (250, 196, 150),
        "celestial": RS.SUN, "stars": False, "lamps": False},
    3: {"sky": ((10, 12, 32), (40, 44, 80)),                # night
        "grass": ((24, 66, 34), (18, 52, 26)),
        "asphalt": ((66, 66, 78), (56, 56, 66)),
        "rumble": ((150, 50, 44), (180, 180, 196)),
        "mount": (14, 14, 30), "cloud": (60, 60, 88),
        "celestial": RS.MOON, "stars": True, "lamps": True},
}
LINE_C = (240, 240, 244)
HUD_BG = (10, 10, 14)
TEXT_C = (240, 240, 245)
NITRO_ICON = (70, 170, 255)
NITRO_FLAME = (255, 170, 40)

Z_SCALE = 3072
W_MAX = 72
HY_BASE = 44

# fast glyph blitting (FONT is 3x5)
_GLYPH = {ch: np.array([[b == "1" for b in row] for row in g], dtype=bool)
          for ch, g in FONT.items()}


def _textf(img, x, y, s, color):
    for ch in s:
        m = _GLYPH.get(ch)
        if m is not None and x >= 0 and x + 3 <= SCREEN_W and y + 5 <= SCREEN_H:
            img[y:y + 5, x:x + 3][m] = color
        x += 4


# ---------------------------------------------------------------- cached strips
_SEG_CACHE = {}
_SKY_CACHE = {}       # (stage, hy) -> (hy+1,160,3) gradient + stars + sun/moon
_CLOUDS = {}          # stage -> (strip1, strip2) bool masks
_RIDGE = None         # (18, 480) periodic jagged silhouette mask
_ROWG = {}            # hy -> (d, z, w) row geometry


def _seg_arrays(state):
    key = (state.level_id, state.variant_seed)
    hit = _SEG_CACHE.get(key)
    if hit is None:
        seg = np.asarray(state.segments, dtype=np.int64)
        hit = (seg[:, 0].copy(), seg[:, 1].copy(), seg[:, 2].copy())
        _SEG_CACHE[key] = hit
    return hit


def _sky_block(stage, hy):
    blk = _SKY_CACHE.get((stage, hy))
    if blk is None:
        pal = STAGE_PAL[stage]
        top, bot = np.asarray(pal["sky"][0]), np.asarray(pal["sky"][1])
        f = np.linspace(0.0, 1.0, max(hy + 1, 2))[:, None]
        col = (top * (1 - f) + bot * f).astype(np.uint8)
        blk = np.repeat(col[:, None, :], SCREEN_W, axis=1)
        if pal["stars"]:
            for i in range(40):
                h = (i * 2654435761 + 97) & 0x7FFFFFFF
                x, y = h % SCREEN_W, (h >> 8) % max(hy - 6, 1)
                blk[y, x] = (204, 208, 224) if h & 4 else (150, 154, 178)
        pal["celestial"].draw(blk, 118, max(2, hy - 34))
        _SKY_CACHE[(stage, hy)] = blk
    return blk


def _cloud_strips(stage):
    hit = _CLOUDS.get(stage)
    if hit is None:
        s1 = np.zeros((5, 480), dtype=bool)
        s2 = np.zeros((6, 480), dtype=bool)
        for i in range(5):
            h = (i * 40503 + stage * 977) & 0x7FFFFFFF
            x = h % (480 - 12)
            s1[1:5, x:x + 11] |= RS.CLOUD[0].mask
        for i in range(4):
            h = (i * 69069 + stage * 313 + 7) & 0x7FFFFFFF
            x = h % (480 - 16)
            s2[0:5, x:x + 15] |= RS.CLOUD[1].mask
        _CLOUDS[stage] = hit = (s1, s2)
    return hit


def _ridge_mask():
    global _RIDGE
    if _RIDGE is None:
        xs = np.arange(480)
        tri1 = np.abs((xs % 56) - 28)                      # jagged triangles
        tri2 = np.abs(((xs * 3 + 17) % 34) - 17)
        h = 3 + tri1 * 10 // 28 + tri2 * 4 // 17           # 3..17
        _RIDGE = np.arange(18)[::-1, None] < h[None, :]    # bottom-aligned
    return _RIDGE


_HALF_X = np.arange(SCREEN_W // 2, dtype=np.int16) * 2     # sampled even columns
_XCOL = np.arange(SCREEN_W)                                # reusable column index
_PALSTACK = {}        # stage -> (2,4,3) uint8: [grass, rumble, asphalt, line]
_HUD = {"key": None, "img": np.empty((8, SCREEN_W, 3), dtype=np.uint8)}


def _palstack(stage):
    ps = _PALSTACK.get(stage)
    if ps is None:
        pal = STAGE_PAL[stage]
        ps = np.empty((2, 4, 3), dtype=np.uint8)
        for par in (0, 1):
            ps[par, 0] = pal["grass"][par]
            ps[par, 1] = pal["rumble"][par]
            ps[par, 2] = pal["asphalt"][par]
            ps[par, 3] = LINE_C
        _PALSTACK[stage] = ps
    return ps


def _row_geom(hy):
    """Static per-horizon geometry + reusable scratch buffers."""
    g = _ROWG.get(hy)
    if g is None:
        dmax = SCREEN_H - 1 - hy
        d = np.arange(1, dmax + 1)
        z = Z_SCALE // d
        w = np.maximum(W_MAX * d // dmax, 2)
        hw = SCREEN_W // 2
        hh = (dmax + 1) // 2                               # half-res rows (even)
        wh = w[::2]
        g = {"dmax": dmax, "hh": hh, "z": z, "w": w, "zh": z[::2],
             "wcol": wh.astype(np.int16)[:, None],
             "wrcol": (wh + np.maximum(2, wh // 10)).astype(np.int16)[:, None],
             "lwcol": np.maximum(1, wh // 24).astype(np.int16)[:, None],
             "wide": (wh > 10)[:, None],
             "dx": np.empty((hh, hw), dtype=np.int16),
             "b1": np.empty((hh, hw), dtype=bool),
             "b2": np.empty((hh, hw), dtype=bool),
             "cat": np.empty((hh, hw), dtype=np.uint8),
             "out": np.empty((hh, hw, 3), dtype=np.uint8)}
        _ROWG[hy] = g
    return g


# ---------------------------------------------------------------- render
def render(state):
    """Render one frame: [112,160,3] uint8."""
    p = state.player
    stage = min(max(state.stage, 1), 3)
    pal = STAGE_PAL[stage]
    curves, hills, decos = _seg_arrays(state)
    n_seg = len(curves)
    img = np.empty((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    # horizon from hill profile
    i0 = min(max(p["dist"] // SEG_LEN, 0), n_seg - 1)
    i6 = min(i0 + 6, n_seg - 1)
    h_eff = (2 * int(hills[i0]) + int(hills[i6])) // 3
    hy = HY_BASE - 7 * h_eff

    # sky: cached gradient + celestial, then parallax clouds + jagged ridge
    img[:hy + 1] = _sky_block(stage, hy)
    if not pal["stars"]:
        s1, s2 = _cloud_strips(stage)
        idx1 = (_XCOL + (p["dist"] >> 7)) % 480
        idx2 = (_XCOL + (p["dist"] >> 6) + 160) % 480
        img[6:11][s1[:, idx1]] = pal["cloud"]
        img[14:20][s2[:, idx2]] = pal["cloud"]
    ridge = _ridge_mask()
    ridx = (_XCOL + (p["dist"] >> 8) + (p["x"] >> 6)) % 480
    y0 = hy - 17
    if y0 >= 0:
        img[y0:hy + 1][ridge[:, ridx]] = pal["mount"]

    # road region: vectorized scanlines at half resolution in both axes
    # (int16 + preallocated buffers + arithmetic category map, 2x2 doubling —
    # deliberate chunky retro scanlines)
    g = _row_geom(hy)
    dmax, hh, w = g["dmax"], g["hh"], g["w"]
    dist_row = p["dist"] + g["zh"]
    seg_idx = np.clip(dist_row >> 7, 0, n_seg - 1)         # SEG_LEN = 128
    base = 80 - (p["x"] * W_MAX // ROAD_HALF)
    c_row = np.take(curves, seg_idx)
    cx = base + (np.cumsum(np.cumsum(c_row[::-1] * 8)) >> 8)[::-1]
    parity = ((dist_row >> 8) & 1).astype(np.uint8)
    dx, b1, b2, cat, out = g["dx"], g["b1"], g["b2"], g["cat"], g["out"]
    np.subtract(_HALF_X[None, :], cx.astype(np.int16)[:, None], out=dx)
    np.abs(dx, out=dx)
    np.less_equal(dx, g["wrcol"], out=b1)                  # road + rumble
    np.less_equal(dx, g["wcol"], out=b2)                   # road proper
    np.add(b1.view(np.uint8), b2.view(np.uint8), out=cat)  # 0 grass 1 rumble 2 road
    np.less_equal(dx, g["lwcol"], out=b1)                  # reuse b1: center line
    b1 &= (parity[:, None] == 0) & g["wide"]
    cat[b1] = 3
    idx8 = (parity[:, None] << 2) | cat                    # flat palette index
    np.take(_palstack(stage).reshape(8, 3), idx8, axis=0, out=out)
    road = img[hy + 1:]
    n_even, n_odd = (dmax + 1) // 2, dmax // 2
    road[0::2, 0::2] = out[:n_even]
    road[0::2, 1::2] = out[:n_even]
    road[1::2, 0::2] = out[:n_odd]
    road[1::2, 1::2] = out[:n_odd]

    def row_of(depth):
        if depth <= 0:
            return None
        di = Z_SCALE // max(depth, Z_SCALE // dmax)
        return hy + min(di, dmax)

    def geom(y):
        i = min(max(y - hy - 1, 0), dmax - 1)
        return int(cx[min(i // 2, hh - 1)]), int(w[i])

    t = state.tick
    # roadside deco (far -> near): palms, billboards, milestones, night lamps
    for si in range(min(i0 + 20, n_seg - 1), i0 + 1, -1):
        deco = int(decos[si])
        lamp = pal["lamps"] and si % 8 == 4
        mile = si % 16 == 0
        if deco == 0 and not lamp and not mile:
            continue
        y = row_of(si * SEG_LEN - p["dist"])
        if y is None or y <= hy + 2:
            continue
        ccx, cw = geom(y)
        near = (y - hy) * 2 >= dmax
        if deco != 0:
            side = 1 if deco > 0 else -1
            if abs(deco) == 1:
                spr = RS.PALM[(t // 8) % 2] if near else RS.PALM_MID[(t // 8) % 2]
            else:
                spr = RS.BILLBOARDS[si % 3] if near else RS.BILLBOARDS_MID[si % 3]
            spr.draw(img, ccx + side * (cw + 8) - spr.w // 2, y - spr.h,
                     flip=side < 0)
        if lamp:
            RS.GLOW.draw(img, ccx + cw + 5, y - RS.LAMP.h - 2)
            RS.LAMP.draw(img, ccx + cw + 7, y - RS.LAMP.h)
        if mile and near:
            RS.MILESTONE.draw(img, ccx - cw - 9, y - RS.MILESTONE.h)

    # checkered finish banner at the end of stage 3
    fin_depth = TRACK_SEGS * SEG_LEN - p["dist"]
    if 0 < fin_depth <= Z_SCALE:
        y = row_of(fin_depth)
        if y is not None and y > hy + 6:
            ccx, cw = geom(y)
            bh = max(2, (y - hy) * 8 // dmax)
            x0, x1 = max(0, ccx - cw), min(SCREEN_W, ccx + cw)
            by = y - 10 - bh
            if x1 > x0 and by >= 0:
                bidx = np.arange(x0, x1) % 64
                img[by:by + bh, x0:x1] = RS.BANNER.rgb[:bh][:, bidx]
            _rect(img, ccx - cw - 2, by, 2, 10 + bh, (196, 200, 210))
            _rect(img, ccx + cw, by, 2, 10 + bh, (196, 200, 210))

    # traffic (far -> near), 3 shapes x 3 depth tiers
    vis = [(c["dist"] - p["dist"], c) for c in state.cars
           if 20 <= c["dist"] - p["dist"] <= Z_SCALE]
    vis.sort(key=lambda dc: -dc[0])
    for depth, car in vis:
        y = row_of(depth)
        if y is None or y <= hy + 2:
            continue
        ccx, cw = geom(y)
        x = ccx + car["x"] * cw // ROAD_HALF
        rel = y - hy
        tier = 0 if rel * 2 >= dmax else (1 if rel * 4 >= dmax else 2)
        spr = RS.TRAFFIC[(car["color"] % 3, tier, car["dir"] < 0)]
        spr.draw(img, x - spr.w // 2, y - spr.h)

    # player car (3 views + spin-out frames + nitro flame)
    if not (p["invuln"] > 0 and p["spin"] == 0 and t % 2 == 0):
        cx0, cy0 = 80, 106
        steer = state.prev_action[1] - state.prev_action[0]
        if p["spin"] > 0:
            fr = (p["spin"] // 5) % 4
            spr, flip = [(RS.CAR_STRAIGHT, False), (RS.CAR_LEAN, False),
                         (RS.CAR_SIDE, False), (RS.CAR_LEAN, True)][fr]
        elif steer != 0:
            spr, flip = RS.CAR_LEAN, steer > 0
        else:
            spr, flip = RS.CAR_STRAIGHT, False
        spr.draw(img, cx0 - spr.w // 2, cy0 - spr.h, flip)
        if p["nitro_t"] > 0 and t % 2 == 0:
            _rect(img, cx0 - 3, cy0 - 1, 6, 3, NITRO_FLAME)
            _rect(img, cx0 - 1, cy0 + 2, 2, 2, (255, 236, 150))

    # HUD (whole strip cached on its display key; cruise = high hit rate).
    # speed display quantized to 5 kph: steadier readout + better cache hits
    kph = (p["spd"] * 300 // MAX_SPD) // 5 * 5
    bar = min(60, 60 * state.timer // TIME_INIT)
    key = (kph, p["nitro"], bar, min(state.stage, 9), min(state.score, 999999))
    if _HUD["key"] != key:
        hud = _HUD["img"]
        hud[:] = HUD_BG
        _textf(hud, 2, 1, str(min(kph, 999)).rjust(3, "0"), TEXT_C)
        for i in range(NITRO_COUNT):
            c = NITRO_ICON if i < p["nitro"] else (50, 50, 60)
            hud[2:6, 18 + i * 6:22 + i * 6] = c
        bcol = (80, 220, 90) if bar > 20 else (235, 200, 60) if bar > 10 \
            else (235, 60, 50)
        hud[2:6, 40:100] = (40, 40, 48)
        if bar > 0:
            hud[2:6, 40:40 + bar] = bcol
        hud[1:7, 104:110] = (60, 60, 72)
        _textf(hud, 105, 1, str(key[3]), TEXT_C)
        _textf(hud, SCREEN_W - 4 - 6 * 4, 1, str(key[4]).rjust(6, "0"), TEXT_C)
        _HUD["key"] = key
    img[0:8, :] = _HUD["img"]
    return img
