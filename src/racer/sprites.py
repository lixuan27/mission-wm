"""Ridgeline pixel art: cars, palms, billboards, clouds, banner (v2 polish).

Role: ASCII-sketch sprites over the shared 16-color palette (game.sprites.sp),
pre-baked RGB+mask for masked blits.  Traffic = 3 silhouettes (sedan / truck /
beetle) x 3 depth sizes, hue-swapped per direction; player car = 3 views +
spin frames.  Deterministic at import.
"""

import numpy as np

from game.sprites import PALETTE, Sprite, sp, _swap

# ---------------------------------------------------------------- player car
# rear view 16x10 (red body, steel glass, orange tail lights)
CAR_STRAIGHT = sp("....kkkkkkkk....",
                  "...krrrrrrrrk...",
                  "..krrkddddkrrk..",
                  "..krkddddddkrk..",
                  ".kkrrkddddkrrkk.",
                  ".krrrrrrrrrrrrk.",
                  "kkorrrrrrrrrrokk",
                  "kkkrrrrrrrrrrkkk",
                  ".kkkkkkkkkkkkkk.",
                  ".kkk..kkkk..kkk.")
CAR_LEAN = sp("..kkkkkkkk......",           # leaning left; flip for right
              ".krrrrrrrrkk....",
              "krrkddddkrrrk...",
              "krkddddddkrrkk..",
              "kkrrkddddkrrrrk.",
              ".krrrrrrrrrrrrk.",
              "kkorrrrrrrrrokk.",
              ".kkrrrrrrrrrkkk.",
              ".kkkkkkkkkkkkk..",
              ".kkk..kkkk..kkk.")
CAR_SIDE = sp("................",           # spin-out side view
              "....kkkkkkkk....",
              "..kkrrrrrrrrkk..",
              ".krrrrddddrrrrk.",
              ".krrrrrrrrrrrrk.",
              "kkkkkkkkkkkkkkkk",
              ".kddk........kdk",
              ".kkkk......kkkk.",
              "................",
              "................")

# ---------------------------------------------------------------- traffic
# base shapes drawn in brick-red 'r'; hues swapped per type/direction below
_SEDAN_NEAR = sp("...kkkkkkkkk...",
                 "..krrkdddkrrk..",
                 ".krrkdddddkrrk.",
                 ".krrrrrrrrrrrk.",
                 "kkrrrrrrrrrrrkk",
                 "korrrrrrrrrrrok",
                 ".kkkkkkkkkkkkk.",
                 ".kk..kkkkk..kk.")
_TRUCK_NEAR = sp("..kkkkkkkkkkkk.",
                 ".krrrrrrrrrrrrk",
                 ".krrrrrrrrrrrrk",
                 ".krrrrrkkdddkrk",
                 "kkrrrrrkdddddkk",
                 "korrrrrkdddddok",
                 ".kkkkkkkkkkkkk.",
                 ".kkk..kkk..kkk.")
_BEETLE_NEAR = sp("....kkkkkkk....",
                  "...krrdddrrk...",
                  "..krrkdddkrrk..",
                  ".krrrrrrrrrrrk.",
                  ".korrrrrrrrrok.",
                  ".kkkkkkkkkkkkk.",
                  "..kk..kkk..kk..",
                  "...............")
_SEDAN_MID = sp("..kkkkkk..",
                ".krkdddkr.",
                "krrrrrrrrk",
                "korrrrrrok",
                ".kkkkkkkk.",
                ".k..kk..k.")
_TRUCK_MID = sp(".kkkkkkkk.",
                "krrrrrrrrk",
                "krrrkddkrk",
                "korrkddkok",
                ".kkkkkkkk.",
                ".kk.kk.kk.")
_BEETLE_MID = sp("..kkkkk...",
                 ".krdddrk..",
                 "krrrrrrrk.",
                 "korrrrrok.",
                 ".kkkkkkk..",
                 "..k.kk.k..")
_SEDAN_FAR = sp(".kkkk.",
                "krrrrk",
                "kkkkkk")
_TRUCK_FAR = sp("kkkkkk",
                "krrrrk",
                "krrrrk")
_BEETLE_FAR = sp(".kkk..",
                 "krrrk.",
                 "kkkkk.")

_SHAPES = [(_SEDAN_NEAR, _SEDAN_MID, _SEDAN_FAR),      # type 0
           (_TRUCK_NEAR, _TRUCK_MID, _TRUCK_FAR),      # type 1
           (_BEETLE_NEAR, _BEETLE_MID, _BEETLE_FAR)]   # type 2
# same-direction hues: sedan red / truck ice-blue / beetle gold
_SAME_HUE = [None, 14, 12]        # index swap from 10 (r)
# oncoming hues: light gray / purple / green
_ONC_HUE = [5, 15, 9]

TRAFFIC = {}                      # (type, tier, oncoming) -> Sprite
for ti, tiers in enumerate(_SHAPES):
    for zi, base in enumerate(tiers):
        TRAFFIC[(ti, zi, False)] = base if _SAME_HUE[ti] is None \
            else _swap(base, 10, _SAME_HUE[ti])
        TRAFFIC[(ti, zi, True)] = _swap(base, 10, _ONC_HUE[ti])

# ---------------------------------------------------------------- roadside
PALM = [
    sp(".eGGGe.eGGe.",
       "eGGGGGGGGGGe",
       "eGG.ebbe.GGe",
       ".e..kbbk..e.",
       "....kbbk....",
       "....kbbk....",
       "...kbbk.....",
       "...kbbk.....",
       "...kbbk.....",
       "..kbbbk.....",
       "..kbbbk.....",
       ".kbbbbbk....",
       "............",
       "............"),
    sp(".eGGe.eGGGe.",
       "eGGGGGGGGGGe",
       "eGG.ebbe.GGe",
       ".e...bbk..e.",
       "....kbbk....",
       "....kbbk....",
       "....kbbk....",
       "....kbbk....",
       "...kbbk.....",
       "..kbbbk.....",
       "..kbbbk.....",
       ".kbbbbbk....",
       "............",
       "............"),
]
MILESTONE = sp(".rr.",
               "krrk",
               "kwwk",
               "kwwk",
               "kwwk",
               "kwwk",
               "kkkk")
LAMP = sp(".yy.",
          "kyyk",
          ".kg.",
          ".kg.",
          ".kg.",
          ".kg.",
          ".kg.",
          ".kg.",
          ".kg.",
          "kkgk")
GLOW = sp(".y.y.y.",
          "y.yyy.y",
          ".yyyyy.",
          "y.yyy.y",
          ".y.y.y.")

# billboard: framed board + pole, 3x5 letter faces stamped at bake time
_LETTERS = {"W": ("101", "101", "111", "111", "101"),
            "M": ("111", "111", "101", "101", "101"),
            "G": ("111", "100", "101", "101", "111"),
            "O": ("111", "101", "101", "101", "111")}


def _billboard(text):
    w = 4 + len(text) * 4 + 1
    idx = np.zeros((12, w), dtype=np.uint8)
    idx[0, :] = 1
    idx[7, :] = 1
    idx[0:8, 0] = 1
    idx[0:8, w - 1] = 1
    idx[1:7, 1:w - 1] = 6                       # white face
    for i, ch in enumerate(text):
        glyph = _LETTERS[ch]
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1":
                    idx[1 + r, 3 + i * 4 + c] = 10
    idx[8:12, w // 2 - 1:w // 2 + 1] = 4        # pole
    return Sprite(idx)


BILLBOARDS = [_billboard("GO"), _billboard("WM"), _billboard("OM")]

# ---------------------------------------------------------------- sky props
CLOUD = [
    sp("...wwww....",
       ".wwwwwwww..",
       "wwwwwwwwwww",
       ".wwww.www.."),
    sp("....wwwww......",
       "..wwwwwwwwww...",
       ".wwwwwwwwwwwww.",
       "wwwwwwwwwwwwwww",
       "..wwww..wwww..."),
]
SUN = Sprite((lambda r: np.where(
    (np.add.outer(np.arange(-r, r + 1) ** 2,
                  np.arange(-r, r + 1) ** 2)) <= r * r, 12, 0)
    .astype(np.uint8))(7))
MOON = sp(".wwww...",
          "www.....",
          "ww......",
          "ww......",
          "ww......",
          "www.....",
          ".wwww...",
          "..ww....")

# checkered finish banner: (8, 64) strip + poles drawn by renderer
_chk = ((np.indices((8, 64)).sum(0) // 4) % 2 == 0)
_b = np.zeros((8, 64), dtype=np.uint8)
_b[_chk] = 6
_b[~_chk] = 1
BANNER = Sprite(_b)


def half(spr):
    """2x nearest-neighbor downscale of a baked sprite (for mid-depth deco)."""
    s = Sprite.__new__(Sprite)
    s.rgb = spr.rgb[::2, ::2].copy()
    s.mask = spr.mask[::2, ::2].copy()
    s.frgb = s.rgb[:, ::-1].copy()
    s.fmask = s.mask[:, ::-1].copy()
    s.m3 = np.ascontiguousarray(s.mask[:, :, None])
    s.fm3 = np.ascontiguousarray(s.fmask[:, :, None])
    s.h, s.w = s.mask.shape
    return s


PALM_MID = [half(p) for p in PALM]
BILLBOARDS_MID = [half(b) for b in BILLBOARDS]
