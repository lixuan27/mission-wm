"""Ridgeline track: 3 hand-designed stages + seed-driven variant generator.

Role: build the 270-segment course (stage 1 gentle teach, stage 2 hills +
medium curves, stage 3 sharp chicanes) and deterministic traffic.  Segments
are [curve, hill, deco] (curve in -3..3; deco 0 none, +/-1 palm, +/-2
billboard, sign = roadside).  variant(level_id, seed): bounded curvature
jitter (|c| stays <= 3, straights stay straight) + re-rolled traffic — the
autopilot's brake-for-curvature law keeps every variant clearable.
level_id = starting stage (1 = full 3-stage run).
"""

from game.engine import XorShift128
from .engine import (GameState, SEG_LEN, STAGE_SEGS, N_STAGES, TIME_INIT,
                     default_player)

# stage designs: runs of (count, curve, hill)
STAGE_RUNS = {
    1: [(10, 0, 0), (12, 1, 0), (10, 0, 0), (12, -1, 0), (8, 0, 1),
        (10, 1, 1), (10, 0, -1), (12, -2, 0), (6, 0, 0)],
    2: [(8, 0, 2), (12, 2, 0), (8, 0, -2), (12, -2, 1), (10, 1, 0),
        (12, -1, -1), (12, 2, 2), (8, 0, 0), (8, -2, 0)],
    3: [(6, 0, 0), (10, 3, 0), (8, 0, 0), (10, -3, 0), (8, 2, 0),
        (10, -3, 0), (10, 3, 1), (8, 0, -1), (12, -2, 0), (8, 0, 0)],
}

SAME_LANES = (-600, -200, 200, 600)
ONC_LANES = (-700, -350)


def _build_segments(rng):
    """Expand runs to 270 [curve, hill, deco] records; rng jitters (None=base)."""
    segs = []
    for st in range(1, N_STAGES + 1):
        for count, c, h in STAGE_RUNS[st]:
            if rng is not None and c != 0:                  # jitter curves only
                c = max(-3, min(3, c + (rng.randint(3) - 1)))
                if c == 0:
                    c = 1 if rng.randint(2) else -1         # keep it a curve
            for k in range(count):
                deco = 0
                if k % 4 == 2:                              # sparse roadside deco
                    kind = 1 if (len(segs) // 8) % 2 == 0 else 2
                    side = 1 if (len(segs) % 8) < 4 else -1
                    deco = kind * side
                segs.append([c, h, deco])
    assert len(segs) == STAGE_SEGS * N_STAGES
    return segs


def _build_traffic(rng, density):
    """Deterministic traffic list across the whole course."""
    cars, cid = [], 0
    for i in range(8, STAGE_SEGS * N_STAGES):               # first 8 segs clear
        if rng.randint(100) >= density:
            continue
        if rng.randint(4) == 0:                             # oncoming
            cars.append({"id": cid, "dist": i * SEG_LEN + rng.randint(SEG_LEN),
                         "x": ONC_LANES[rng.randint(2)], "dir": -1,
                         "spd": 260 + rng.randint(60),
                         "color": rng.randint(3), "passed": True})
        else:                                               # same direction
            cars.append({"id": cid, "dist": i * SEG_LEN + rng.randint(SEG_LEN),
                         "x": SAME_LANES[rng.randint(4)], "dir": 1,
                         "spd": 200 + rng.randint(80),
                         "color": rng.randint(3), "passed": False})
        cid += 1
    return cars


def make_level(level_id=1, variant_seed=0):
    """Fresh course; level_id = starting stage (1..3), 0 seed = base layout."""
    assert 1 <= level_id <= N_STAGES
    st = GameState(level_id=level_id, variant_seed=variant_seed)
    st.rng = XorShift128(seed=level_id * 6151 + variant_seed * 104729 + 777).state()
    seg_rng = None if variant_seed == 0 else \
        XorShift128(seed=variant_seed * 2654435761 + 17)
    st.segments = _build_segments(seg_rng)
    traffic_rng = XorShift128(seed=variant_seed * 48271 + level_id * 9973 + 5)
    density = 22 if variant_seed == 0 else 16 + traffic_rng.randint(14)
    st.cars = _build_traffic(traffic_rng, density)
    start = (level_id - 1) * STAGE_SEGS * SEG_LEN
    st.player = default_player(start)
    st.stage = level_id
    st.timer = TIME_INIT
    # drop traffic parked on the grid right around the start position
    st.cars = [c for c in st.cars
               if not (0 <= c["dist"] - start < 6 * SEG_LEN and abs(c["x"]) < 400)]
    return st


def variant(level_id, seed):
    """Spec-name alias: jittered but clearable course variant."""
    return make_level(level_id, variant_seed=seed)
