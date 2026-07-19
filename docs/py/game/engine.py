"""Slugline core engine: deterministic fixed-timestep platform-shooter simulation.

Role: GameState dataclass + step(state, action) -> (state, events) + serialize helpers.
Design: 20 ticks/s logic, 1/16 fixed-point integer coordinates, explicit xorshift128
RNG stored inside the state.  Same state + same inputs => bit-identical states/frames.
No use of `random` / `np.random` anywhere; all stochasticity flows through XorShift128.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------- constants
FP = 16                       # fixed-point units per pixel
TILE = 8                      # tile size in px
TILE_FP = TILE * FP
LEVEL_W, LEVEL_H = 40, 14     # tiles
LEVEL_W_PX = LEVEL_W * TILE   # 320
LEVEL_H_PX = LEVEL_H * TILE   # 112
SCREEN_W, SCREEN_H = 160, 112
TICKS_PER_SEC = 20

# action indices (8-bit multi-hot)
A_LEFT, A_RIGHT, A_UP, A_DOWN, A_FIRE, A_JUMP, A_GREN, A_SWITCH = range(8)
NOOP = (0, 0, 0, 0, 0, 0, 0, 0)

# physics (fp units per tick / tick^2)
GRAV = 14
GRAV_HOLD = 6                 # reduced gravity while jump held and rising
JUMP_V = -76
MAX_FALL = 120
RUN = 24                      # 1.5 px/tick
CROUCH_RUN = 10

PLAYER_W = 6 * FP
PLAYER_H = 12 * FP
PLAYER_H_CROUCH = 6 * FP

BULLET_SPD = 64               # 4 px/tick
EBULLET_SPD = 48
BULLET_TTL = 60

WEAPONS = ("pistol", "mg", "rocket")
WEAPON_CD = {"pistol": 4, "mg": 1, "rocket": 8}
WEAPON_DMG = {"pistol": 1, "mg": 1, "rocket": 3}
GRENADE_CD = 20
MELEE_CD = 8
MELEE_DMG = 2
MELEE_RANGE = 8 * FP

ROCKET_SPLASH_R = 12          # px
ROCKET_SPLASH_DMG = 2
GRENADE_SPLASH_R = 14
GRENADE_SPLASH_DMG = 2

ENEMY_HP = {"walker": 2, "turret": 3, "flyer": 1, "crate": 1}
ENEMY_SCORE = {"walker": 100, "turret": 200, "flyer": 150, "crate": 50}
ENEMY_SIZE = {"walker": (7, 10), "turret": (8, 8), "flyer": (8, 6), "crate": (8, 8)}  # px

RESCUE_SCORE = 500
PICKUP_SCORE = 50
CLEAR_SCORE = 1000
START_LIVES = 3
DEATH_TIMER = 16
INVULN_TICKS = 40

VPLAT_VANISH = 24             # ticks solid after touch
VPLAT_GONE = 50               # ticks before respawn

# 64-entry sine table scaled by 256 (hardcoded for cross-platform determinism)
SIN64 = [0, 25, 50, 74, 98, 121, 142, 162, 181, 198, 213, 226, 237, 245, 251, 255,
         256, 255, 251, 245, 237, 226, 213, 198, 181, 162, 142, 121, 98, 74, 50, 25,
         0, -25, -50, -74, -98, -121, -142, -162, -181, -198, -213, -226, -237, -245,
         -251, -255, -256, -255, -251, -245, -237, -226, -213, -198, -181, -162,
         -142, -121, -98, -74, -50, -25]

MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------- RNG
class XorShift128:
    """Deterministic xorshift128 PRNG; 4x uint32 state lives inside GameState."""

    def __init__(self, seed=None, state=None):
        if state is not None:
            self.s = [int(v) & MASK32 for v in state]
        else:
            # splitmix32-style expansion of a scalar seed into 4 nonzero words
            x = (int(seed) & MASK32) or 0x9E3779B9
            s = []
            for _ in range(4):
                x = (x + 0x9E3779B9) & MASK32
                z = x
                z = ((z ^ (z >> 16)) * 0x85EBCA6B) & MASK32
                z = ((z ^ (z >> 13)) * 0xC2B2AE35) & MASK32
                z ^= z >> 16
                s.append(z or 1)
            self.s = s

    def next_u32(self):
        x, y, z, w = self.s
        t = (x ^ ((x << 11) & MASK32)) & MASK32
        nw = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
        self.s = [y, z, w, nw & MASK32]
        return self.s[3]

    def randint(self, n):
        """Uniform-ish integer in [0, n)."""
        return self.next_u32() % n if n > 0 else 0

    def state(self):
        return list(self.s)


# ---------------------------------------------------------------- state
def _deep(obj):
    """Deep-copy a json-like structure (dict/list/scalars only)."""
    if isinstance(obj, dict):
        return {k: _deep(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep(v) for v in obj]
    return obj


@dataclass
class GameState:
    """Whole game state; every field is json-serializable via to_dict()."""
    level_id: int = 1
    variant_seed: int = 0
    tick: int = 0
    tiles: list = field(default_factory=list)      # 14 strings x 40 chars ('.', '#', '^')
    rng: list = field(default_factory=lambda: XorShift128(seed=1).state())
    player: dict = field(default_factory=dict)
    enemies: list = field(default_factory=list)
    bullets: list = field(default_factory=list)
    grenades: list = field(default_factory=list)
    pickups: list = field(default_factory=list)
    cages: list = field(default_factory=list)
    checkpoints: list = field(default_factory=list)
    vplats: list = field(default_factory=list)
    traps: list = field(default_factory=list)
    explosions: list = field(default_factory=list)
    camera_x: int = 0                              # px
    score: int = 0
    lives: int = START_LIVES
    game_over: bool = False
    level_clear: bool = False
    checkpoint: list = field(default_factory=lambda: [0, 0])  # respawn pos (fp)
    exit_x: int = 0                                # flag tile x (tiles)
    death_timer: int = 0
    prev_action: list = field(default_factory=lambda: [0] * 8)

    def to_dict(self):
        """Serialize to plain-python json-safe dict with deep-copy semantics."""
        return {
            "level_id": self.level_id, "variant_seed": self.variant_seed,
            "tick": self.tick, "tiles": list(self.tiles), "rng": list(self.rng),
            "player": _deep(self.player), "enemies": _deep(self.enemies),
            "bullets": _deep(self.bullets), "grenades": _deep(self.grenades),
            "pickups": _deep(self.pickups), "cages": _deep(self.cages),
            "checkpoints": _deep(self.checkpoints), "vplats": _deep(self.vplats),
            "traps": _deep(self.traps), "explosions": _deep(self.explosions),
            "camera_x": self.camera_x, "score": self.score, "lives": self.lives,
            "game_over": self.game_over, "level_clear": self.level_clear,
            "checkpoint": list(self.checkpoint), "exit_x": self.exit_x,
            "death_timer": self.death_timer, "prev_action": list(self.prev_action),
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild from dict; deep-copies input so forks never alias."""
        return cls(**{k: _deep(v) for k, v in d.items()})


def serialize(state):
    return state.to_dict()


def deserialize(d):
    return GameState.from_dict(d)


def default_player(x_fp, y_fp):
    return {
        "x": x_fp, "y": y_fp, "vx": 0, "vy": 0, "facing": 1,
        "crouch": False, "on_ground": False, "jump_hold": False,
        "alive": True, "invuln": 0, "shoot_cd": 0, "grenade_cd": 0,
        "weapon": "pistol", "ammo": {"mg": 0, "rocket": 0}, "grenades": 10,
    }


# ---------------------------------------------------------------- tile helpers
def tile_at(tiles, tx, ty):
    if 0 <= tx < LEVEL_W and 0 <= ty < LEVEL_H:
        return tiles[ty][tx]
    if tx < 0 or tx >= LEVEL_W:
        return "#"          # side walls
    return "."              # open top/bottom

def _solid(tiles, vset, tx, ty):
    return tile_at(tiles, tx, ty) == "#" or (tx, ty) in vset


def _move_axis_x(tiles, vset, x, y, w, h, dx):
    """Move AABB horizontally with tile collision; returns (new_x, blocked)."""
    nx = x + dx
    if dx > 0:
        edge = (nx + w - 1) // TILE_FP
        for ty in range(y // TILE_FP, (y + h - 1) // TILE_FP + 1):
            if _solid(tiles, vset, edge, ty):
                return edge * TILE_FP - w, True
    elif dx < 0:
        edge = nx // TILE_FP
        for ty in range(y // TILE_FP, (y + h - 1) // TILE_FP + 1):
            if _solid(tiles, vset, edge, ty):
                return (edge + 1) * TILE_FP, True
    return nx, False


def _move_axis_y(tiles, vset, x, y, w, h, dy):
    """Move AABB vertically with tile collision; returns (new_y, hit_floor, hit_ceil)."""
    ny = y + dy
    if dy > 0:
        edge = (ny + h - 1) // TILE_FP
        for tx in range(x // TILE_FP, (x + w - 1) // TILE_FP + 1):
            if _solid(tiles, vset, tx, edge):
                return edge * TILE_FP - h, True, False
    elif dy < 0:
        edge = ny // TILE_FP
        for tx in range(x // TILE_FP, (x + w - 1) // TILE_FP + 1):
            if _solid(tiles, vset, tx, edge):
                return (edge + 1) * TILE_FP, False, True
    return ny, False, False


def _overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _player_box(p):
    h = PLAYER_H_CROUCH if p["crouch"] else PLAYER_H
    return p["x"], p["y"] + (PLAYER_H - h if p["crouch"] else 0), PLAYER_W, h


def _enemy_box(e):
    w, h = ENEMY_SIZE[e["kind"]]
    return e["x"], e["y"], w * FP, h * FP


def _ev(events, etype, tick, x_fp, y_fp, **extra):
    d = {"type": etype, "tick": tick, "x": x_fp // FP, "y": y_fp // FP}
    d.update(extra)
    events.append(d)


# ---------------------------------------------------------------- step
def step(state, action):
    """Advance one tick.  `action` = 8-bit multi-hot tuple.  Mutates and returns
    (state, events).  Frozen when game_over/level_clear (reset via levels.make_level)."""
    if state.game_over or state.level_clear:
        return state, []

    act = tuple(int(bool(a)) for a in action)
    prev = state.prev_action
    rng = XorShift128(state=state.rng)
    state.tick += 1
    t = state.tick
    events = []
    p = state.player
    tiles = state.tiles
    vset = {(v["tx"], v["ty"]) for v in state.vplats if v["state"] != "gone"}

    # ---------------- player input + physics
    if p["alive"]:
        # crouch (ground only); stand up only with headroom
        if act[A_DOWN] and p["on_ground"]:
            p["crouch"] = True
        elif p["crouch"]:
            htx0, htx1 = p["x"] // TILE_FP, (p["x"] + PLAYER_W - 1) // TILE_FP
            hty = (p["y"]) // TILE_FP
            if not any(_solid(tiles, vset, tx, hty) for tx in range(htx0, htx1 + 1)):
                p["crouch"] = False

        # horizontal velocity + facing
        spd = CROUCH_RUN if p["crouch"] else RUN
        if act[A_LEFT] and not act[A_RIGHT]:
            p["vx"] = -spd
            p["facing"] = -1
        elif act[A_RIGHT] and not act[A_LEFT]:
            p["vx"] = spd
            p["facing"] = 1
        else:
            p["vx"] = 0

        # jump: starts whenever B held on ground (auto-rehop keeps branches simple)
        if act[A_JUMP] and p["on_ground"] and not p["crouch"]:
            p["vy"] = JUMP_V
            p["on_ground"] = False
            p["jump_hold"] = True
        if not act[A_JUMP]:
            p["jump_hold"] = False
        g = GRAV_HOLD if (p["jump_hold"] and p["vy"] < 0) else GRAV
        p["vy"] = min(p["vy"] + g, MAX_FALL)

        # integrate + collide
        bx, by, bw, bh = _player_box(p)
        nx, _ = _move_axis_x(tiles, vset, bx, by, bw, bh, p["vx"])
        nx = max(nx, state.camera_x * FP)                      # camera left wall
        ny, floor, ceil = _move_axis_y(tiles, vset, nx, by, bw, bh, p["vy"])
        if floor:
            p["on_ground"] = True
            p["vy"] = 0
        else:
            p["on_ground"] = False
            if ceil:
                p["vy"] = 0
        p["x"] = nx
        p["y"] = ny - (PLAYER_H - bh)                          # back to standing top

        # weapon switch (edge-triggered)
        if act[A_SWITCH] and not prev[A_SWITCH]:
            avail = [w for w in WEAPONS if w == "pistol" or p["ammo"].get(w, 0) > 0]
            if len(avail) > 1:
                i = avail.index(p["weapon"]) if p["weapon"] in avail else 0
                p["weapon"] = avail[(i + 1) % len(avail)]
                _ev(events, "weapon_switch", t, p["x"], p["y"], weapon=p["weapon"])

        # fire (hold-to-fire, per-weapon cooldown; point-blank melee is automatic)
        if p["shoot_cd"] > 0:
            p["shoot_cd"] -= 1
        if act[A_FIRE] and p["shoot_cd"] == 0:
            _player_fire(state, p, act, rng, events, t)

        # grenade (hold = repeat with cooldown)
        if p["grenade_cd"] > 0:
            p["grenade_cd"] -= 1
        if act[A_GREN] and p["grenade_cd"] == 0 and p["grenades"] > 0:
            p["grenades"] -= 1
            p["grenade_cd"] = GRENADE_CD
            gx = p["x"] + PLAYER_W // 2
            gy = p["y"] + 2 * FP
            state.grenades.append({"x": gx, "y": gy, "vx": p["facing"] * 36,
                                   "vy": -70, "ttl": 50})

        # hazards: spikes (1px-forgiving box), pits
        sx, sy, sw, sh = _player_box(p)
        sx += FP; sy += FP; sw -= 2 * FP; sh -= 2 * FP
        hit_spike = False
        for ty in range(sy // TILE_FP, (sy + sh - 1) // TILE_FP + 1):
            for tx in range(sx // TILE_FP, (sx + sw - 1) // TILE_FP + 1):
                if tile_at(tiles, tx, ty) == "^":
                    hit_spike = True
        if hit_spike:
            _kill_player(state, "spike", events, t)
        elif p["y"] // FP > LEVEL_H_PX:
            _kill_player(state, "pit", events, t)

        if p["invuln"] > 0:
            p["invuln"] -= 1

    # ---------------- vanishing platforms
    for v in state.vplats:
        if v["state"] == "solid" and p["alive"] and p["on_ground"]:
            fx, fy, fw, fh = _player_box(p)
            feet_ty = (fy + fh) // TILE_FP
            if feet_ty == v["ty"] and fx // TILE_FP <= v["tx"] <= (fx + fw - 1) // TILE_FP:
                v["state"] = "vanishing"
                v["timer"] = VPLAT_VANISH
                _ev(events, "trap_trigger", t, v["tx"] * TILE_FP, v["ty"] * TILE_FP,
                    trap="vplat")
        elif v["state"] == "vanishing":
            v["timer"] -= 1
            if v["timer"] <= 0:
                v["state"] = "gone"
                v["timer"] = VPLAT_GONE
        elif v["state"] == "gone":
            v["timer"] -= 1
            if v["timer"] <= 0:
                v["state"] = "solid"

    # ---------------- ceiling traps
    for tr in state.traps:
        if tr["state"] == "armed" and p["alive"]:
            px = (p["x"] + PLAYER_W // 2) // FP
            if tr["x"] - 4 <= px <= tr["x"] + tr["w"] + 4 and p["y"] // FP > tr["y"]:
                tr["state"] = "falling"
                tr["vy"] = 0
                _ev(events, "trap_trigger", t, tr["x"] * FP, tr["y"] * FP, trap="ceiling")
        elif tr["state"] == "falling":
            tr["vy"] = min(tr["vy"] + GRAV, MAX_FALL)
            tr["yf"] = tr.get("yf", tr["y"] * FP) + tr["vy"]
            tr["y"] = tr["yf"] // FP
            if p["alive"]:
                bx, by, bw, bh = _player_box(p)
                if _overlap(bx, by, bw, bh, tr["x"] * FP, tr["y"] * FP,
                            tr["w"] * FP, tr["h"] * FP):
                    _kill_player(state, "trap", events, t)
            bot_ty = (tr["y"] + tr["h"]) // TILE
            if bot_ty >= LEVEL_H or any(
                    tile_at(tiles, tx, bot_ty) == "#"
                    for tx in range(tr["x"] // TILE, (tr["x"] + tr["w"] - 1) // TILE + 1)):
                tr["y"] = bot_ty * TILE - tr["h"]
                tr["state"] = "done"

    # ---------------- enemies
    for e in state.enemies:
        if e["hp"] <= 0:
            continue
        k = e["kind"]
        if k == "walker":
            _walker_ai(state, e, tiles, vset)
        elif k == "turret":
            _turret_ai(state, e, rng, t)
        elif k == "flyer":
            _flyer_ai(state, e)
        # contact damage (crates are harmless)
        if k != "crate" and p["alive"] and p["invuln"] == 0:
            bx, by, bw, bh = _player_box(p)
            ex, ey, ew, eh = _enemy_box(e)
            if _overlap(bx, by, bw, bh, ex, ey, ew, eh):
                _kill_player(state, k, events, t)

    # ---------------- bullets
    keep = []
    for b in state.bullets:
        b["x"] += b["vx"]
        b["y"] += b["vy"]
        b["ttl"] -= 1
        bx_px = b["x"] // FP
        dead = b["ttl"] <= 0 or bx_px < state.camera_x - 16 or \
            bx_px > state.camera_x + SCREEN_W + 16 or b["y"] < -32 * FP or \
            b["y"] > (LEVEL_H_PX + 32) * FP
        if not dead and _solid(tiles, vset, b["x"] // TILE_FP, b["y"] // TILE_FP):
            dead = True
            if b["kind"] == "rocket":
                _explode(state, b["x"], b["y"], ROCKET_SPLASH_R, ROCKET_SPLASH_DMG,
                         events, t)
        if not dead and b["kind"] != "enemy":
            for e in state.enemies:
                if e["hp"] <= 0:
                    continue
                ex, ey, ew, eh = _enemy_box(e)
                if _overlap(b["x"] - FP, b["y"] - FP, 2 * FP, 2 * FP, ex, ey, ew, eh):
                    _damage_enemy(state, e, WEAPON_DMG[b["kind"]], events, t)
                    if b["kind"] == "rocket":
                        _explode(state, b["x"], b["y"], ROCKET_SPLASH_R,
                                 ROCKET_SPLASH_DMG, events, t)
                    dead = True
                    break
        if not dead and b["kind"] == "enemy" and p["alive"] and p["invuln"] == 0:
            bx, by, bw, bh = _player_box(p)
            if _overlap(b["x"] - FP, b["y"] - FP, 2 * FP, 2 * FP, bx, by, bw, bh):
                _kill_player(state, "bullet", events, t)
                dead = True
        if not dead:
            keep.append(b)
    state.bullets = keep

    # ---------------- grenades
    keep = []
    for gnd in state.grenades:
        gnd["vy"] = min(gnd["vy"] + 10, MAX_FALL)
        gnd["x"] += gnd["vx"]
        gnd["y"] += gnd["vy"]
        gnd["ttl"] -= 1
        boom = gnd["ttl"] <= 0 or _solid(tiles, vset, gnd["x"] // TILE_FP,
                                         gnd["y"] // TILE_FP)
        if not boom:
            for e in state.enemies:
                if e["hp"] > 0 and _overlap(gnd["x"] - FP, gnd["y"] - FP, 2 * FP,
                                            2 * FP, *_enemy_box(e)):
                    boom = True
                    break
        if boom:
            _explode(state, gnd["x"], gnd["y"], GRENADE_SPLASH_R, GRENADE_SPLASH_DMG,
                     events, t)
        else:
            keep.append(gnd)
    state.grenades = keep

    # ---------------- pickups / cages / checkpoints / exit
    if p["alive"]:
        bx, by, bw, bh = _player_box(p)
        for pk in state.pickups:
            if not pk["taken"] and _overlap(bx, by, bw, bh, pk["x"] * FP, pk["y"] * FP,
                                            8 * FP, 8 * FP):
                pk["taken"] = True
                _apply_pickup(state, p, pk["kind"], events, t)
        for cg in state.cages:
            if not cg["rescued"] and _overlap(bx, by, bw, bh, cg["x"] * FP,
                                              cg["y"] * FP, 10 * FP, 12 * FP):
                cg["rescued"] = True
                state.score += RESCUE_SCORE
                if cg["gift"] == "mg":
                    p["ammo"]["mg"] += 50
                elif cg["gift"] == "rocket":
                    p["ammo"]["rocket"] += 5
                _ev(events, "rescue", t, cg["x"] * FP, cg["y"] * FP, cage_id=cg["id"],
                    score=RESCUE_SCORE)
        for cp in state.checkpoints:
            if not cp["taken"] and _overlap(bx, by, bw, bh, cp["x"] * FP,
                                            (cp["y"] - 8) * FP, 8 * FP, 16 * FP):
                cp["taken"] = True
                state.checkpoint = [cp["x"] * FP, cp["y"] * FP + 8 * FP - PLAYER_H]
                _ev(events, "checkpoint", t, cp["x"] * FP, cp["y"] * FP)
        fx = state.exit_x * TILE_FP
        if _overlap(bx, by, bw, bh, fx, 0, TILE_FP, LEVEL_H_PX * FP):
            state.level_clear = True
            state.score += CLEAR_SCORE
            _ev(events, "level_clear", t, fx, p["y"], score=CLEAR_SCORE)

    # ---------------- explosions (visual ttl)
    state.explosions = [dict(x=e["x"], y=e["y"], ttl=e["ttl"] - 1)
                        for e in state.explosions if e["ttl"] > 1]

    # ---------------- camera: forward-only, dead-zone follow
    target = p["x"] // FP - 72
    state.camera_x = max(state.camera_x, min(target, LEVEL_W_PX - SCREEN_W))
    state.camera_x = max(0, state.camera_x)

    # ---------------- death / respawn
    if not p["alive"] and not state.game_over:
        state.death_timer -= 1
        if state.death_timer <= 0:
            p["x"], p["y"] = state.checkpoint
            p["vx"] = p["vy"] = 0
            p["alive"] = True
            p["crouch"] = False
            p["invuln"] = INVULN_TICKS
            state.camera_x = max(0, min(p["x"] // FP - 64, LEVEL_W_PX - SCREEN_W))

    state.rng = rng.state()
    state.prev_action = list(act)
    return state, events


# ---------------------------------------------------------------- subroutines
def _player_fire(state, p, act, rng, events, t):
    """Handle A press: automatic melee at point-blank, else spawn a bullet."""
    # melee: enemy inside a small box in front of the player
    fx = p["x"] + (PLAYER_W if p["facing"] > 0 else -MELEE_RANGE)
    bx, by, bw, bh = _player_box(p)
    for e in state.enemies:
        if e["hp"] > 0 and e["kind"] != "crate" and \
                _overlap(fx, by, MELEE_RANGE, bh, *_enemy_box(e)):
            _damage_enemy(state, e, MELEE_DMG, events, t)
            p["shoot_cd"] = MELEE_CD
            return
    w = p["weapon"]
    if w != "pistol":
        if p["ammo"][w] <= 0:                       # safety: no ammo -> pistol
            _auto_pistol(state, p, events, t)
            w = "pistol"
        else:
            p["ammo"][w] -= 1
    gy = by + (4 * FP if not p["crouch"] else 3 * FP)   # low enough to hit 8px targets
    if act[A_UP] and not p["crouch"]:
        bvx, bvy = 0, -BULLET_SPD
        gx = p["x"] + PLAYER_W // 2
        gy = p["y"]
    else:
        bvx, bvy = p["facing"] * BULLET_SPD, 0
        gx = p["x"] + (PLAYER_W if p["facing"] > 0 else -2 * FP)
    state.bullets.append({"x": gx, "y": gy, "vx": bvx, "vy": bvy,
                          "kind": w, "ttl": BULLET_TTL})
    p["shoot_cd"] = WEAPON_CD[w]
    if w != "pistol" and p["ammo"][w] <= 0:          # depleted -> auto back to pistol
        _auto_pistol(state, p, events, t)


def _auto_pistol(state, p, events, t):
    if p["weapon"] != "pistol":
        p["weapon"] = "pistol"
        _ev(events, "weapon_switch", t, p["x"], p["y"], weapon="pistol", auto=True)


def _damage_enemy(state, e, dmg, events, t):
    e["hp"] -= dmg
    _ev(events, "hit", t, e["x"], e["y"], target=e["kind"], target_id=e["id"], dmg=dmg)
    if e["hp"] <= 0:
        sc = ENEMY_SCORE[e["kind"]]
        state.score += sc
        _ev(events, "kill", t, e["x"], e["y"], target=e["kind"], target_id=e["id"],
            score=sc)
        if e["kind"] == "crate" and e.get("drop"):
            state.pickups.append({"id": 100 + e["id"], "kind": e["drop"],
                                  "x": e["x"] // FP, "y": e["y"] // FP, "taken": False})


def _explode(state, x_fp, y_fp, radius_px, dmg, events, t):
    state.explosions.append({"x": x_fp // FP, "y": y_fp // FP, "ttl": 6})
    r = radius_px * FP
    for e in state.enemies:
        if e["hp"] <= 0:
            continue
        ex, ey, ew, eh = _enemy_box(e)
        cx, cy = ex + ew // 2, ey + eh // 2
        if abs(cx - x_fp) <= r + ew // 2 and abs(cy - y_fp) <= r + eh // 2:
            _damage_enemy(state, e, dmg, events, t)


def _apply_pickup(state, p, kind, events, t):
    state.score += PICKUP_SCORE
    if kind == "mg":
        p["ammo"]["mg"] += 100
    elif kind == "rocket":
        p["ammo"]["rocket"] += 10
    elif kind == "grenade":
        p["grenades"] += 5
    elif kind == "life":
        state.lives += 1
    _ev(events, "pickup", t, p["x"], p["y"], kind=kind, score=PICKUP_SCORE)


def _kill_player(state, cause, events, t):
    p = state.player
    if not p["alive"] or p["invuln"] > 0:
        return
    p["alive"] = False
    state.lives -= 1
    state.death_timer = DEATH_TIMER
    _ev(events, "player_death", t, p["x"], p["y"], cause=cause, lives=state.lives)
    if state.lives <= 0:
        state.game_over = True


def _walker_ai(state, e, tiles, vset):
    spd = 12
    nx = e["x"] + e["dir"] * spd
    w, h = ENEMY_SIZE["walker"]
    front = (nx + (w * FP if e["dir"] > 0 else 0)) // TILE_FP
    ty_body = (e["y"] + h * FP // 2) // TILE_FP
    ty_feet = (e["y"] + h * FP) // TILE_FP           # tile below feet
    wall = _solid(tiles, vset, front, ty_body)
    ledge = not _solid(tiles, vset, front, ty_feet)
    if wall or ledge:
        e["dir"] = -e["dir"]
    else:
        e["x"] = nx


def _turret_ai(state, e, rng, t):
    p = state.player
    if e["timer"] > 0:
        e["timer"] -= 1
        return
    if not p["alive"]:
        return
    ex, ey, ew, eh = _enemy_box(e)
    cx, cy = ex + ew // 2, ey + eh // 2
    px = p["x"] + PLAYER_W // 2
    py = p["y"] + PLAYER_H // 2
    dx, dy = px - cx, py - cy
    if abs(dx) // FP > 110:
        return
    n = max(abs(dx), abs(dy), 1)
    state.bullets.append({"x": cx, "y": cy, "vx": dx * EBULLET_SPD // n,
                          "vy": dy * EBULLET_SPD // n, "kind": "enemy",
                          "ttl": BULLET_TTL})
    e["timer"] = 45 + rng.randint(16)


def _flyer_ai(state, e):
    p = state.player
    st = e.get("fstate", "hover")
    if st == "hover":
        e["phase"] += 1
        e["y"] = e["base_y"] + (SIN64[(e["phase"] * 3) % 64] * 128) // 256
        e["x"] = e["base_x"]
        if e["cool"] > 0:
            e["cool"] -= 1
        elif p["alive"]:
            pcx = p["x"] + PLAYER_W // 2
            pcy = p["y"] + PLAYER_H // 2
            if abs(pcx - e["x"]) // FP < 40 and pcy > e["y"]:
                e["tx"], e["ty"] = pcx, pcy          # stale target: telegraphed dive
                e["fstate"] = "windup"
                e["timer"] = 10
    elif st == "windup":                             # freeze = visual telegraph
        e["timer"] -= 1
        if e["timer"] <= 0:
            dx, dy = e["tx"] - e["x"], e["ty"] - e["y"]
            n = max(abs(dx), abs(dy), 1)
            e["tvx"], e["tvy"] = dx * 64 // n, dy * 64 // n
            e["fstate"] = "dive"
            e["timer"] = 18
    elif st == "dive":
        e["x"] += e["tvx"]
        e["y"] += e["tvy"]
        e["timer"] -= 1
        if e["timer"] <= 0:
            e["fstate"] = "return"
    else:                                            # return
        dx = e["base_x"] - e["x"]
        dy = e["base_y"] - e["y"]
        if abs(dx) <= 32 and abs(dy) <= 32:
            e["x"], e["y"] = e["base_x"], e["base_y"]
            e["fstate"] = "hover"
            e["cool"] = 40
        else:
            n = max(abs(dx), abs(dy), 1)
            e["x"] += dx * 32 // n
            e["y"] += dy * 32 // n
