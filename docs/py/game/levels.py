"""Slugline levels: 3 hand-crafted maps + seed-driven variant generator.

Role: parse 40x14 ASCII tile maps into GameState.  Static tiles keep only
'#' (solid) / '^' (spike) / '.'; every other char spawns an entity.
Legend: P player, C checkpoint, F exit flag, w walker, t turret, f flyer,
b crate, g cage, D vanishing platform, T ceiling trap,
m mg-ammo, r rocket-ammo, n grenades, l extra life.
variant(level_id, seed): bounded jitter of enemy/trap/pickup positions
(clearability-safe: never onto spikes/walls/pits; small amplitudes).
"""

from .engine import (FP, TILE, LEVEL_W, LEVEL_H, PLAYER_H, ENEMY_HP, ENEMY_SIZE,
                     GameState, XorShift128, default_player)

# crate drop table per level (kind of pickup dropped when destroyed)
_CRATE_DROP = {1: "mg", 2: "grenade", 3: "rocket"}
# cage gift per (level, cage index)
_CAGE_GIFT = {(3, 0): "mg", (3, 1): "rocket"}

L1 = [  # teaching level: flat run, 2 walkers, crate, mg pickup
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "..............####......................",
    "........................................",
    "........................................",
    ".P....w....b......w...C......m.......F..",
    "########################################",
]

L2 = [  # spike + vanishing-platform gauntlet (dense hazards, no enemies)
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "................DD......................",
    ".P.m..^^...C..........C..^^.....^..F....",
    "##############......####################",
]

L3 = [  # mixed: spikes, cage x2, ceiling trap, pit, flyer, walker, crate, turret
    "........................................",
    "........................................",
    "......T.................................",
    "........................................",
    "........................................",
    "........................................",
    ".............f..........................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    "........................................",
    ".P..n...^^.g..C.......w..m.C.b....t.g.F.",
    "#################...####################",
]

LEVELS = {1: L1, 2: L2, 3: L3}


def _entity_y(row, kind):
    """Top y (px) so the entity's feet sit on the bottom of its char row."""
    h = ENEMY_SIZE.get(kind, (8, 8))[1]
    return (row + 1) * TILE - h


def make_level(level_id, variant_seed=0):
    """Build a fresh GameState for a level (variant_seed=0 -> base layout)."""
    rows = LEVELS[level_id]
    assert len(rows) == LEVEL_H and all(len(r) == LEVEL_W for r in rows), \
        f"level {level_id} must be {LEVEL_W}x{LEVEL_H}"
    st = GameState(level_id=level_id, variant_seed=variant_seed)
    st.rng = XorShift128(seed=level_id * 7919 + variant_seed * 104729 + 12345).state()
    tiles, eid, cage_i = [], 0, 0
    for r, row in enumerate(rows):
        static = []
        for c, ch in enumerate(row):
            if ch in "#^":
                static.append(ch)
                continue
            static.append(".")
            x = c * TILE
            if ch == "P":
                px, py = x * FP, ((r + 1) * TILE) * FP - PLAYER_H
                st.player = default_player(px, py)
                st.checkpoint = [px, py]
            elif ch == "C":
                st.checkpoints.append({"x": x, "y": r * TILE, "taken": False})
            elif ch == "F":
                st.exit_x = c
            elif ch in "wtb":
                kind = {"w": "walker", "t": "turret", "b": "crate"}[ch]
                e = {"id": eid, "kind": kind, "x": x * FP,
                     "y": _entity_y(r, kind) * FP, "hp": ENEMY_HP[kind]}
                if kind == "walker":
                    e["dir"] = -1
                elif kind == "turret":
                    e["timer"] = 40
                elif kind == "crate":
                    e["drop"] = _CRATE_DROP.get(level_id)
                st.enemies.append(e)
                eid += 1
            elif ch == "f":
                bx, by = x * FP, (r * TILE) * FP
                st.enemies.append({"id": eid, "kind": "flyer", "x": bx, "y": by,
                                   "base_x": bx, "base_y": by, "phase": 0,
                                   "fstate": "hover", "cool": 0, "tvx": 0,
                                   "tvy": 0, "tx": 0, "ty": 0, "timer": 0,
                                   "hp": ENEMY_HP["flyer"]})
                eid += 1
            elif ch == "g":
                gift = _CAGE_GIFT.get((level_id, cage_i), "mg")
                st.cages.append({"id": cage_i, "x": x, "y": (r + 1) * TILE - 12,
                                 "rescued": False, "gift": gift})
                cage_i += 1
            elif ch == "D":
                st.vplats.append({"tx": c, "ty": r, "state": "solid", "timer": 0})
            elif ch == "T":
                if st.traps and st.traps[-1]["x"] + st.traps[-1]["w"] == x:
                    st.traps[-1]["w"] += TILE          # merge adjacent T chars
                else:
                    st.traps.append({"x": x, "y": r * TILE, "w": TILE, "h": TILE,
                                     "state": "armed", "vy": 0})
            elif ch in "mrnl":
                kind = {"m": "mg", "r": "rocket", "n": "grenade", "l": "life"}[ch]
                st.pickups.append({"id": len(st.pickups), "kind": kind, "x": x,
                                   "y": (r + 1) * TILE - 8, "taken": False})
        tiles.append("".join(static))
    st.tiles = tiles
    if variant_seed != 0:
        _apply_variant(st, tiles, variant_seed)
    return st


def variant(level_id, seed):
    """Spec-name alias: jittered but clearability-safe variant of a level."""
    return make_level(level_id, variant_seed=seed)


def _ground_ok(tiles, col, row):
    """A ground entity may stand at (col,row): empty tile, solid below."""
    if not (0 <= col < LEVEL_W):
        return False
    below = tiles[row + 1][col] if row + 1 < LEVEL_H else "."
    return tiles[row][col] == "." and below == "#"


def _apply_variant(st, tiles, seed):
    """Jitter enemies/pickups/traps with an independent RNG (bounded, safe)."""
    rng = XorShift128(seed=seed * 2654435761 + st.level_id)
    for e in st.enemies:
        if e["kind"] == "flyer":
            e["base_x"] += (rng.randint(17) - 8) * FP   # +/-8 px
            e["x"] = e["base_x"]
            continue
        col = e["x"] // FP // TILE
        row = (e["y"] // FP + ENEMY_SIZE[e["kind"]][1] - 1) // TILE
        cand = [c for c in (col - 1, col, col + 1) if _ground_ok(tiles, c, row)]
        if cand:
            e["x"] = cand[rng.randint(len(cand))] * TILE * FP
    for pk in st.pickups:
        col, row = pk["x"] // TILE, (pk["y"] + 7) // TILE
        cand = [c for c in (col - 1, col, col + 1) if _ground_ok(tiles, c, row)]
        if cand:
            pk["x"] = cand[rng.randint(len(cand))] * TILE
    for tr in st.traps:
        tr["x"] = max(TILE, min(LEVEL_W * TILE - tr["w"] - TILE,
                                tr["x"] + rng.randint(9) - 4))  # +/-4 px
