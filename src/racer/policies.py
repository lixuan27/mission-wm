"""Ridgeline data-collection policies (same four-family contract as Slugline).

Role:
- RandomLegal: throttle-biased random multi-hot (no left+right conflict)
- StickyMacro: full throttle + weak lane keeping (no curve anticipation)
- ScriptedClear: heuristic autopilot — speed-scaled curvature lookahead sets a
  brake-to speed (c: 0->768, 1->512, 2->440, 3->320), bang-bang steering =
  centrifugal feed-forward + lane feedback, traffic dodging, nitro on straights.
  Rollout-verified to finish all 3 stages (hard acceptance in tests).
- FailureMining: autopilot + recurring 40-tick corruption windows
  (no-steer / hard-left / full-stop) -> crash & time-out mining.
"""

from game.engine import XorShift128
from .engine import (MAX_SPD, NOOP, SEG_LEN, STEER_K, seg_at)

TARGET_SPD = {0: 768, 1: 512, 2: 440, 3: 320}


class Policy:
    """Base: reset(seed, level_id) then act(state, tick) -> 8-bit multi-hot."""
    name = "base"

    def reset(self, seed, level_id):
        self.rng = XorShift128(seed=seed)
        self.level_id = level_id

    def act(self, state, tick):
        return NOOP


class RandomLegal(Policy):
    name = "random_legal"

    def act(self, state, tick):
        r = self.rng
        d = r.randint(4)
        left, right = (1, 0) if d == 0 else (0, 1) if d == 1 else (0, 0)
        return (left, right, int(r.randint(4) < 3), int(r.randint(16) == 0),
                int(r.randint(64) == 0), 0, 0, 0)


class StickyMacro(Policy):
    """Pedal to the floor + weak centering; no braking, no curve anticipation."""
    name = "sticky_macro"

    def act(self, state, tick):
        x = state.player["x"]
        left, right = (0, 1) if x < -350 else (1, 0) if x > 350 else (0, 0)
        return (left, right, 1, 0, 0, 0, 0, 0)


def _autopilot(state):
    """Heuristic driving action for the current state."""
    p = state.player
    segs = state.segments
    n = len(segs)
    i = min(max(p["dist"] // SEG_LEN, 0), n - 1)
    look = 3 + p["spd"] // 64                              # speed-scaled horizon
    cmax = 0
    straight = True
    for k in range(look + 1):
        c = segs[min(i + k, n - 1)][0]
        cmax = max(cmax, abs(c))
        if c != 0:
            straight = False
    target_spd = TARGET_SPD[min(cmax, 3)]
    throttle = 1 if p["spd"] < target_spd else 0
    brake = 1 if p["spd"] > target_spd + 24 else 0

    # lateral target: corridor scoring over ALL windowed traffic (clusters!)
    # with a time-aware lane-crossing penalty; alongside cars stay in window
    target_x = 350
    steer_rate = max(8, STEER_K * p["spd"] // MAX_SPD)
    window = []
    for c in state.cars:
        dz = c["dist"] - p["dist"]
        if -150 < dz < 1500:
            rel = ((c["spd"] * c["dir"]) - p["spd"]) >> 4   # d(dz)/tick
            window.append((c["x"], dz, rel))
    if window:
        best_score = -(1 << 30)
        best_clear = 0
        for lane in (-700, -350, 0, 350, 700):
            clear = 1 << 30
            score = 0
            lo = min(p["x"], lane) - 260
            hi = max(p["x"], lane) + 260
            for cx_, dz_, rel_ in window:
                clear = min(clear, abs(lane - cx_))
                if lo <= cx_ <= hi:                        # path crosses its lane
                    t_reach = abs(cx_ - p["x"]) // steer_rate + 1
                    dz_then = dz_ + rel_ * t_reach
                    if -260 < dz_then < 260:
                        score -= 4000                      # would arrive alongside
            score += min(clear, 600) * 8 - abs(lane - p["x"]) // 32
            if score > best_score:
                best_score, best_clear, target_x = score, clear, lane
        if best_clear < 320 or best_score < 0:             # boxed in: tail traffic
            target_spd = min(target_spd, 240)
            throttle = 1 if p["spd"] < target_spd else 0
            brake = 1 if p["spd"] > target_spd + 24 else 0
    if p["offroad"]:
        target_x = 0
    push = (segs[i][0] * p["spd"] * p["spd"]) >> 14        # engine will do x -= push
    desired = (target_x - p["x"]) // 8 + push
    left, right = (0, 1) if desired > 2 else (1, 0) if desired < -2 else (0, 0)

    nitro = 0
    if straight and not window and p["nitro"] > 0 and p["nitro_t"] == 0 \
            and p["spd"] >= 500 and not p["offroad"]:
        nitro = 1                                          # engine edge-triggers
    return (left, right, throttle, brake, nitro, 0, 0, 0)


class ScriptedClear(Policy):
    name = "scripted_clear"

    def act(self, state, tick):
        return _autopilot(state)


class FailureMining(Policy):
    """Autopilot with recurring 40-tick corruption windows."""
    name = "failure_mining"

    def reset(self, seed, level_id):
        super().reset(seed, level_id)
        self.t0 = 20 + self.rng.randint(180)
        self.mode = self.rng.randint(3)

    def act(self, state, tick):
        phase = (tick - self.t0) % 260
        if tick >= self.t0 and 0 <= phase < 40:
            if self.mode == 0:                             # no steer, full gas
                return (0, 0, 1, 0, 0, 0, 0, 0)
            if self.mode == 1:                             # hard left
                return (1, 0, 1, 0, 0, 0, 0, 0)
            return (0, 0, 0, 1, 0, 0, 0, 0)                # full stop
        return _autopilot(state)


POLICIES = {p.name: p for p in (RandomLegal, StickyMacro, ScriptedClear,
                                FailureMining)}


def make_policy(name):
    return POLICIES[name]()
