"""Slugline pixel art: palette-indexed sprite tables (NES/IWBTG-grade, v2 art).

Role: every sprite is an ASCII pixel sketch in-source (readable + editable),
parsed at import into uint8 index arrays over a global 16-color palette and
pre-baked into RGB + mask (and mirrored copies) for fast masked blits.
Zero external assets; import is deterministic.

Char map ('.' = transparent):
  k outline-black   n night-blue   d dark-steel   g gun-gray
  l light-gray      w white        s skin         e army-dark
  G army-green      r brick-red    o blood-orange y gold
  b wood-brown      c ice-blue     p purple
"""

import numpy as np

# ---------------------------------------------------------------- palette
PALETTE = np.array([
    (0, 0, 0),          # 0 transparent
    (24, 20, 28),       # 1 k outline black
    (34, 38, 66),       # 2 n night blue
    (72, 76, 92),       # 3 d dark steel
    (134, 138, 152),    # 4 g gun gray
    (196, 200, 210),    # 5 l light gray
    (244, 244, 248),    # 6 w white
    (232, 188, 146),    # 7 s skin
    (52, 84, 44),       # 8 e army dark green
    (98, 138, 58),      # 9 G army green
    (178, 52, 42),      # 10 r brick red
    (230, 102, 36),     # 11 o blood orange
    (236, 194, 64),     # 12 y gold
    (138, 94, 52),      # 13 b wood brown
    (108, 172, 224),    # 14 c ice blue
    (150, 84, 188),     # 15 p purple
], dtype=np.uint8)

_CH = {".": 0, "k": 1, "n": 2, "d": 3, "g": 4, "l": 5, "w": 6, "s": 7,
       "e": 8, "G": 9, "r": 10, "o": 11, "y": 12, "b": 13, "c": 14, "p": 15}


class Sprite:
    """Pre-baked RGB + mask (and horizontally mirrored copies) for blitting."""

    def __init__(self, idx):
        self.h, self.w = idx.shape
        self.rgb = PALETTE[idx]
        self.mask = idx > 0
        self.frgb = self.rgb[:, ::-1].copy()
        self.fmask = self.mask[:, ::-1].copy()
        self.m3 = np.ascontiguousarray(self.mask[:, :, None])    # for copyto
        self.fm3 = np.ascontiguousarray(self.fmask[:, :, None])

    def draw(self, img, x, y, flip=False):
        """Masked blit with clipping (np.copyto avoids fancy-index temps)."""
        H, W = img.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + self.w), min(H, y + self.h)
        if x0 >= x1 or y0 >= y1:
            return
        sx, sy = x0 - x, y0 - y
        m = (self.fm3 if flip else self.m3)[sy:sy + y1 - y0, sx:sx + x1 - x0]
        r = (self.frgb if flip else self.rgb)[sy:sy + y1 - y0, sx:sx + x1 - x0]
        np.copyto(img[y0:y1, x0:x1], r, where=m)


def sp(*rows):
    assert len({len(r) for r in rows}) == 1, "ragged sprite rows"
    return Sprite(np.array([[_CH[c] for c in r] for r in rows], dtype=np.uint8))


# ---------------------------------------------------------------- player 12x18
# soldier: helmet + goggles band, green fatigues, forward hands (gun = overlay).
# faces RIGHT; hitbox 6x12 sits at sprite offset (3,6).
_P_HEAD = [
    "....kkkkk...",
    "...kGGGGGk..",
    "..kGGGGGGGk.",
    "..kkcckcckk.",
    "..kssssssk..",
    "...kssssk...",
]
PLAYER_STAND = sp(*_P_HEAD,
    "..keeeeeek..",
    ".keGGGGGGks.",
    ".keGGGGGGkk.",
    ".keGGGGGGk..",
    "..kGGGGGGk..",
    "..keekkeek..",
    "..keek.keek.",
    "..keek.keek.",
    "..kek...kek.",
    "..kbk...kbk.",
    ".kbbk..kbbk.",
    ".kkkk..kkkk.")
PLAYER_RUN = [
    sp(*_P_HEAD,
       "..keeeeeek..",
       ".keGGGGGGks.",
       ".keGGGGGGkk.",
       ".keGGGGGGk..",
       "..kGGGGGGk..",
       "..keekeek...",
       "..kek.keek..",
       ".kek...keek.",
       ".kbk....kbk.",
       ".kbbk...kbbk",
       ".kkk....kkkk",
       "............"),
    sp(*_P_HEAD,
       "..keeeeeek..",
       ".keGGGGGGks.",
       ".keGGGGGGkk.",
       ".keGGGGGGk..",
       "..kGGGGGGk..",
       "..keekkeek..",
       "...keekek...",
       "...kekkek...",
       "...kbkkbk...",
       "...kbbkbbk..",
       "...kkkkkkk..",
       "............"),
    sp(*_P_HEAD,
       "..keeeeeek..",
       ".keGGGGGGks.",
       ".keGGGGGGkk.",
       ".keGGGGGGk..",
       "..kGGGGGGk..",
       "...keekeek..",
       "..keek.kek..",
       ".keek...kek.",
       ".kbk.....kbk",
       "kbbk.....kbk",
       "kkkk.....kkk",
       "............"),
]
PLAYER_JUMP = sp(*_P_HEAD,
    "..keeeeeek..",
    ".keGGGGGGks.",
    ".keGGGGGGkk.",
    ".keGGGGGGk..",
    "..kGGGGGGk..",
    "..keekkeek..",
    "..kbek.ebk..",
    "..kbbk.bbk..",
    "..kkkkkkkk..",
    "............",
    "............",
    "............")
PLAYER_CROUCH = sp(          # 12x10, feet-aligned
    "....kkkkk...",
    "...kGGGGGk..",
    "..kGGGGGGGk.",
    "..kkcckcckk.",
    "..kssssssks.",
    ".keGGGGGGkk.",
    ".keGGGGGGk..",
    "..keekkeek..",
    ".kbbk..kbbk.",
    ".kkkk..kkkk.")
PLAYER_DEAD = [
    sp("............",     # reeling back
       "..kkkkk.....",
       ".kGGGGGk....",
       ".kkcckck....",
       ".kssssk.....",
       "..keeeeek...",
       ".keGGGGGGk..",
       ".keGGGGGGk..",
       "..kGGGGGk...",
       "..keekeek...",
       "..kek.kek...",
       ".kbbk.kbbk..",
       ".kkkk.kkkk..",
       "............",
       "............",
       "............",
       "............",
       "............"),
    sp("............",     # down
       "............",
       "............",
       "............",
       "............",
       "............",
       "............",
       "............",
       "............",
       "............",
       "............",
       "............",
       ".kkkkk......",
       "kGGGGGkkkkk.",
       "kkcckGGGGGbk",
       "ksskGGGGGkbk",
       ".kkkeekeekk.",
       "..kkkkkkkkk."),
]
# gun overlays (anchor = hand position), facing right
GUN_H = sp("kkkkk",
           "gllgk",
           "..kk.")
GUN_UP = sp(".kgk",
            ".klk",
            ".klk",
            "kkgk",
            ".kk.")
GUN_DIAG = sp("...kk",
              "..klk",
              ".klk.",
              "kgk..",
              "kk...")

# ---------------------------------------------------------------- enemies
# walker 11x16: red-bandana rifleman, 2 walk frames (hitbox 7x10 offset (2,6))
_W_HEAD = [
    "...kkkkk...",
    "..krrrrrk..",
    "..krrrrrrk.",
    "..ksssssk..",
    "...kssk....",
]
WALKER = [
    sp(*_W_HEAD,
       "..kdddddk..",
       ".kdrdddddk.",
       ".kdddddddsk",
       ".kddddddkk.",
       "..kdddddk..",
       "..kdkkddk..",
       "..kdk.kdk..",
       ".kdk...kdk.",
       ".kbk...kbk.",
       "kbbk...kbbk",
       "kkkk...kkkk"),
    sp(*_W_HEAD,
       "..kdddddk..",
       ".kdrdddddk.",
       ".kdddddddsk",
       ".kddddddkk.",
       "..kdddddk..",
       "..kkdddkk..",
       "...kdkdk...",
       "...kdkdk...",
       "...kbkbk...",
       "..kbbkbbk..",
       "..kkkkkkk.."),
]
# turret 12x10: sandbag mound + slit (barrel drawn separately toward player)
TURRET = sp(
    "...kkkkkk...",
    "..kggggggk..",
    ".kgglggggdk.",
    ".kgggggggdk.",
    "kbybybybybyk",
    "kybybybybybk",
    "kbybybybybyk",
    "kybybybybybk",
    "kbybybybybyk",
    "kkkkkkkkkkkk")
# flyer 12x10 bat, 2 flap frames + red windup variants (hitbox 8x6 offset (2,2))
FLYER = [
    sp("kk........kk",
       "kpk......kpk",
       "kppk.kk.kppk",
       "kpppkppkpppk",
       ".kppppppppk.",
       "..kpwppwpk..",
       "...kppppk...",
       "....kppk....",
       "....k..k....",
       "............"),
    sp("............",
       "............",
       ".kk......kk.",
       "kppk.kk.kppk",
       "kpppkppkpppk",
       ".kpwppwppk..",
       "..kppppppk..",
       "...kppk.....",
       "...k..k.....",
       "............"),
]


# red-flash windup variants: swap purple -> blood orange
def _swap(spr, frm, to):
    idx = np.argmax(np.all(spr.rgb[..., None, :] ==
                           PALETTE[None, None, :, :], axis=-1), axis=-1)
    idx[~spr.mask] = 0
    idx[idx == frm] = to
    return Sprite(idx.astype(np.uint8))


FLYER_RED = [_swap(f, 15, 11) for f in FLYER]

# ---------------------------------------------------------------- FX
MUZZLE = [
    sp(".w.",
       "wyw",
       ".w."),
    sp("y.y",
       ".w.",
       "y.y"),
]
EXPLOSION = [
    sp("............",
       "............",
       "....ww......",
       "...wwww.....",
       "...wwwww....",
       "....www.....",
       "............",
       "............",
       "............",
       "............",
       "............",
       "............"),
    sp("............",
       "...oyo......",
       "..oywyyo....",
       ".oywwwyo....",
       ".oywwwwyo...",
       "..yywwyo....",
       "...oyyo.....",
       "....o.......",
       "............",
       "............",
       "............",
       "............"),
    sp("...ooo......",
       "..oyyyoo....",
       ".oyywwyyo...",
       ".oywwwwyyo..",
       "oyywwwwwyo..",
       ".oywwwwyyo..",
       ".ooyywwyo...",
       "..ooyyyo....",
       "....ooo.....",
       "............",
       "............",
       "............"),
    sp("..d..d..d...",
       ".d.oo..d....",
       "..o..o...d..",
       ".d.o..o.d...",
       "d...dd...d..",
       ".o.d..d.o...",
       "..o....o....",
       ".d.o..o..d..",
       "....dd......",
       "..d....d....",
       "............",
       "............"),
]
GRENADE = [
    sp(".ke",
       "kGk",
       "ekk"),
    sp("ek.",
       "kGk",
       ".ke"),
]
ROCKET = sp("kggw.",
            "glllw",
            "kggw.")
ROCKET_FLAME = sp("oy",
                  "yw",
                  "oy")

# ---------------------------------------------------------------- scene
# ground brick 8x8: 3-tone relief + mortar seams (noise dots added at bake)
BRICK = sp("llllllll",
           "gggkgggg",
           "gdgkggdg",
           "kkkkkkkk",
           "ggggkggg",
           "gdggkgdg",
           "kkkkkkkk",
           "dddddddd")
SPIKE = sp("...ww...",
           "...lw...",
           "..kll...",
           "..kllw..",
           ".kglllw.",
           ".kgllll.",
           "kkgllllw",
           "kggllllk")
CAGE = [                       # 12x14 wooden cage, captive waving (2 frames)
    sp("kbbbbbbbbbbk",
       "kbkkbkkbkkbk",
       "kb.kb.skk.bk",
       "kbkkbksskkbk",
       "kb.kbsssk.bk",
       "kbkkbkskkkbk",
       "kb.kbssbk.bk",
       "kbkkbssbkkbk",
       "kb.kbssbk.bk",
       "kbkkbkkbkkbk",
       "kb.kb.kbk.bk",
       "kbkkbkkbkkbk",
       "kbbbbbbbbbbk",
       "kkkkkkkkkkkk"),
    sp("kbbbbbbbbbbk",
       "kbkkbkkbkkbk",
       "kb.kbskkk.bk",
       "kbkkbsskkkbk",
       "kb.kbsssk.bk",
       "kbkkbkskkkbk",
       "kb.kbssbk.bk",
       "kbkkbssbkkbk",
       "kb.kbssbk.bk",
       "kbkkbkkbkkbk",
       "kb.kb.kbk.bk",
       "kbkkbkkbkkbk",
       "kbbbbbbbbbbk",
       "kkkkkkkkkkkk"),
]
CAGE_OPEN = sp("kbbbbbbbbbbk",
               "kbk........k",
               "kb.........k",
               "kbk........k",
               "kb.........k",
               "kbk........k",
               "kb.........k",
               "kbk........k",
               "kb.........k",
               "kbk........k",
               "kb.........k",
               "kbk........k",
               "kbbbbbbbbbbk",
               "kkkkkkkkkkkk")
CP_FLAG = [                    # 8x14 checkpoint flag, 2 wave frames (gold cloth)
    sp("kyyyy...",
       "kyyyyyy.",
       "kyyyyy..",
       "kyyy....",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "kk......"),
    sp("kyyy....",
       "kyyyyy..",
       "kyyyyyy.",
       "kyyyy...",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "k.......",
       "kk......"),
]
EXIT_FLAG = sp("kGGGGGGGGG..",             # 12x24 big exit flag, white star
               "kGGGwGGGGGG.",
               "kGGwwwGGGGGG",
               "kGwwwwwGGGG.",
               "kGGwwwGGGGG.",
               "kGGwGwGGGG..",
               "kGGGGGGGG...",
               "kGGGGGG.....",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "k...........",
               "kk..........",
               "kkk.........")
MOON = sp(".wwww...",
          "www.....",
          "ww......",
          "ww......",
          "ww......",
          "www.....",
          ".wwww...",
          "..ww....")
# HUD
HEART = sp(".rr.rr.",
           "rrrrrrr",
           "rrrrrrr",
           ".rrrrr.",
           "..rrr..",
           "...r...")
ICON_PISTOL = sp("kkkkkk..",
                 "klllgkk.",
                 "..kgk...",
                 "..kgkk..",
                 "..kkk...")
ICON_MG = sp("kkkkkkkk",
             "kccclgkk",
             "..kgkkk.",
             "..kggk..",
             "..kkkk..")
ICON_ROCKET = sp("..kkkkk.",
                 "koolllwk",
                 "koollllw",
                 "koolllwk",
                 "..kkkkk.")
PICKUP_BOX = sp("kkkkkkkk",
                "kllllllk",
                "klggggdk",
                "klggggdk",
                "klggggdk",
                "klggggdk",
                "kddddddk",
                "kkkkkkkk")

# activated checkpoint: gold cloth -> army green
CP_FLAG_TAKEN = [_swap(f, 12, 9) for f in CP_FLAG]
