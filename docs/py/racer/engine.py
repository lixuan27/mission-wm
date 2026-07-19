"""Ridgeline core engine: deterministic OutRun-style 2.5D arcade time-trial.

Role: GameState dataclass + step(state, action) -> (state, events) + serialize.
Shares every pipeline contract with Slugline (src/game): 20 ticks/s fixed step,
integer fixed-point state, XorShift128 RNG stored in the state, json-safe dicts,
8-bit multi-hot actions.  Action slots: [0]=left [1]=right [2]=throttle
[3]=brake [4]=nitro; [5..7] reserved (always ignored).
`level_clear` means "crossed the finish line" (pipeline-compatible flag name).
"""

from dataclasses import dataclass, field

from game.engine import XorShift128, _deep   # shared deterministic RNG + deep copy

# ---------------------------------------------------------------- constants
TICKS_PER_SEC = 20
SCREEN_W, SCREEN_H = 160, 112

A_LEFT, A_RIGHT, A_UP, A_DOWN, A_NITRO = 0, 1, 2, 3, 4
NOOP = (0, 0, 0, 0, 0, 0, 0, 0)

SEG_LEN = 128                 # track-distance units per segment
STAGE_SEGS = 90
N_STAGES = 3
TRACK_SEGS = STAGE_SEGS * N_STAGES

ROAD_HALF = 1024              # lateral units from center to road edge
LAT_CLAMP = 1700              # hard shoulder
MAX_SPD = 512                 # speed fixed-point; dist advance = spd >> 4
NITRO_SPD = 768
ACCEL, BRAKE, COAST = 4, 12, 2
NITRO_TICKS = 60
NITRO_COUNT = 3
STEER_K = 36                  # lateral px/tick at MAX_SPD (dx = K*spd//MAX_SPD)
CENTRI_SHIFT = 14             # push = curve * spd^2 >> shift (outward)
GRASS_DECEL = 14
GRASS_FLOOR = 120
CAR_LEN_D = 100               # collision half-window along track
CAR_HW = 280                  # collision half-window lateral
SPIN_TICKS = 40
CRASH_INVULN = 60
TIME_INIT = 1100              # ticks (55 s) to first checkpoint
CHECK_BONUS = 800             # ticks added per checkpoint
OVERTAKE_SCORE = 100
CHECK_SCORE = 500
FINISH_SCORE = 2000


# ---------------------------------------------------------------- state
@dataclass
class GameState:
    """Racer state; json-serializable via to_dict(), mirrors game.GameState API."""
    level_id: int = 1                              # starting stage (1..3)
    variant_seed: int = 0
    tick: int = 0
    rng: list = field(default_factory=lambda: XorShift128(seed=1).state())
    segments: list = field(default_factory=list)   # [curve, hill, deco] per segment
    player: dict = field(default_factory=dict)
    cars: list = field(default_factory=list)
    timer: int = TIME_INIT
    stage: int = 1
    score: int = 0
    game_over: bool = False
    level_clear: bool = False                      # finish-line crossed
    prev_action: list = field(default_factory=lambda: [0] * 8)

    def to_dict(self):
        return {
            "level_id": self.level_id, "variant_seed": self.variant_seed,
            "tick": self.tick, "rng": list(self.rng),
            "segments": _deep(self.segments), "player": _deep(self.player),
            "cars": _deep(self.cars), "timer": self.timer, "stage": self.stage,
            "score": self.score, "game_over": self.game_over,
            "level_clear": self.level_clear,
            "prev_action": list(self.prev_action),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: _deep(v) for k, v in d.items()})


def serialize(state):
    return state.to_dict()


def deserialize(d):
    return GameState.from_dict(d)


def default_player(start_dist):
    return {"x": 0, "spd": 0, "dist": start_dist, "nitro": NITRO_COUNT,
            "nitro_t": 0, "spin": 0, "invuln": 0, "offroad": False}


def seg_at(segments, dist):
    """Segment record for a track distance (clamped to track)."""
    i = dist // SEG_LEN
    if i < 0:
        i = 0
    elif i >= len(segments):
        i = len(segments) - 1
    return segments[i]


def _ev(events, etype, tick, p, **extra):
    d = {"type": etype, "tick": tick, "x": p["x"], "dist": p["dist"]}
    d.update(extra)
    events.append(d)


# ---------------------------------------------------------------- step
def step(state, action):
    """Advance one tick.  Mutates and returns (state, events).  Frozen after
    finish (level_clear) or time-out (game_over)."""
    if state.game_over or state.level_clear:
        return state, []

    act = tuple(int(bool(a)) for a in action)
    prev = state.prev_action
    rng = XorShift128(state=state.rng)
    state.tick += 1
    t = state.tick
    events = []
    p = state.player
    cur = seg_at(state.segments, p["dist"])
    curve = cur[0]

    # ---------------- longitudinal control
    if p["spin"] > 0:
        p["spin"] -= 1                              # spin-out: controls dead
        p["spd"] = 0
    else:
        if act[A_NITRO] and not prev[A_NITRO] and p["nitro"] > 0 \
                and p["nitro_t"] == 0:
            p["nitro"] -= 1
            p["nitro_t"] = NITRO_TICKS
            _ev(events, "nitro", t, p, left=p["nitro"])
        if act[A_DOWN]:
            p["spd"] -= BRAKE
        elif act[A_UP]:
            p["spd"] += ACCEL
        else:
            p["spd"] -= COAST
        cap = NITRO_SPD if p["nitro_t"] > 0 else MAX_SPD
        if p["spd"] > cap:
            p["spd"] = max(cap, p["spd"] - 8)       # boost decay back to cap
        p["spd"] = max(0, p["spd"])
    if p["nitro_t"] > 0:
        p["nitro_t"] -= 1

    # ---------------- lateral: steering + centrifugal + grass
    if p["spin"] == 0:
        p["x"] += (act[A_RIGHT] - act[A_LEFT]) * (STEER_K * p["spd"] // MAX_SPD)
    p["x"] -= (curve * p["spd"] * p["spd"]) >> CENTRI_SHIFT
    off = abs(p["x"]) > ROAD_HALF
    if off:
        if p["spd"] > GRASS_FLOOR:
            p["spd"] = max(GRASS_FLOOR, p["spd"] - GRASS_DECEL)
        p["x"] += rng.randint(9) - 4                # grass shake
        if not p["offroad"]:
            _ev(events, "offroad_enter", t, p)
    p["offroad"] = off
    p["x"] = max(-LAT_CLAMP, min(LAT_CLAMP, p["x"]))

    # ---------------- advance
    p["dist"] += p["spd"] >> 4

    # ---------------- traffic
    for c in state.cars:
        c["dist"] += (c["spd"] >> 4) * c["dir"]
        if c["dir"] > 0 and not c["passed"] and c["dist"] < p["dist"]:
            c["passed"] = True
            state.score += OVERTAKE_SCORE
            _ev(events, "overtake", t, p, car_id=c["id"], score=OVERTAKE_SCORE)
        if p["invuln"] == 0 and abs(c["dist"] - p["dist"]) < CAR_LEN_D \
                and abs(c["x"] - p["x"]) < CAR_HW:
            p["spd"] = 0
            p["spin"] = SPIN_TICKS
            p["invuln"] = CRASH_INVULN
            _ev(events, "crash", t, p, car_id=c["id"])
    if p["invuln"] > 0:
        p["invuln"] -= 1

    # ---------------- checkpoints / finish  (absolute gates at stage ends)
    while state.stage <= N_STAGES and \
            p["dist"] >= state.stage * STAGE_SEGS * SEG_LEN:
        if state.stage == N_STAGES:
            state.level_clear = True
            state.score += FINISH_SCORE + state.timer
            _ev(events, "finish", t, p, score=FINISH_SCORE, time_left=state.timer)
        else:
            state.timer += CHECK_BONUS
            state.score += CHECK_SCORE
            _ev(events, "checkpoint", t, p, stage=state.stage,
                time_added=CHECK_BONUS)
        state.stage += 1

    # ---------------- timer
    if not state.level_clear:
        state.timer -= 1
        if state.timer <= 0:
            state.timer = 0
            state.game_over = True
            _ev(events, "time_out", t, p)

    state.rng = rng.state()
    state.prev_action = list(act)
    return state, events
