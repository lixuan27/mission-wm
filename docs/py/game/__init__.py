"""Slugline: deterministic side-scrolling platform-shooter data engine for Mission-WM.

Public API: GameState, step, serialize/deserialize (engine); render; make_level /
variant (levels); policy classes (policies); collect CLI (collect).
"""

from .engine import (GameState, XorShift128, step, serialize, deserialize,
                     NOOP, A_LEFT, A_RIGHT, A_UP, A_DOWN, A_FIRE, A_JUMP,
                     A_GREN, A_SWITCH, SCREEN_W, SCREEN_H, TICKS_PER_SEC)
from .render import render
from .levels import make_level, variant, LEVELS

__all__ = ["GameState", "XorShift128", "step", "serialize", "deserialize",
           "render", "make_level", "variant", "LEVELS", "NOOP",
           "A_LEFT", "A_RIGHT", "A_UP", "A_DOWN", "A_FIRE", "A_JUMP",
           "A_GREN", "A_SWITCH", "SCREEN_W", "SCREEN_H", "TICKS_PER_SEC"]
