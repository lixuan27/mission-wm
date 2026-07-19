"""Ridgeline: deterministic OutRun-style 2.5D racer data engine for Mission-WM.

Game #2 — shares every pipeline contract with Slugline (src/game): GameState
dict save-states, 20 ticks/s, XorShift128-in-state, render 160x112 uint8,
same collect h5 schema.  `level_clear` = finish line crossed.
"""

from .engine import (GameState, step, serialize, deserialize, NOOP,
                     A_LEFT, A_RIGHT, A_UP, A_DOWN, A_NITRO,
                     SCREEN_W, SCREEN_H, TICKS_PER_SEC)
from .render import render
from .levels import make_level, variant

__all__ = ["GameState", "step", "serialize", "deserialize", "render",
           "make_level", "variant", "NOOP", "A_LEFT", "A_RIGHT", "A_UP",
           "A_DOWN", "A_NITRO", "SCREEN_W", "SCREEN_H", "TICKS_PER_SEC"]
