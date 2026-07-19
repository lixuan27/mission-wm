"""Slugline data-collection policies (SPEC section 5 mix).

Role: four deterministic policy families, each seeded with its own XorShift128:
- RandomLegal: fresh multi-hot each tick (no left+right conflicts)
- StickyMacro: hold-right runner with periodic jump / fire-burst / grenade macros
- ScriptedClear: hardcoded per-level key sequences, rollout-verified to clear
  the base variant of L1/L2/L3 deathless (tuned 2026-07-19)
- FailureMining: scripted base + a randomized perturbation window -> likely death
"""

from .engine import XorShift128, NOOP

R_ = (0, 1, 0, 0, 0, 0, 0, 0)
RA = (0, 1, 0, 0, 1, 0, 0, 0)
RB = (0, 1, 0, 0, 0, 1, 0, 0)
RAB = (0, 1, 0, 0, 1, 1, 0, 0)
A_ = (0, 0, 0, 0, 1, 0, 0, 0)


def _seq(*segments):
    out = []
    for n, a in segments:
        out.extend([a] * n)
    return out


# rollout-verified clear sequences (base variants): L1 t=189, L2 t=178, L3 t=254
SCRIPTS = {
    1: (_seq((300, RA)), RA),
    2: (_seq((21, R_), (13, RB), (29, R_), (13, RB), (2, R_), (13, RB),
             (30, R_), (13, RB), (21, R_), (13, RB)), R_),
    3: (_seq((30, RA), (13, RAB), (37, RA), (13, RAB), (22, RA), (60, A_)), RA),
}


class Policy:
    """Base: reset(seed, level_id) then act(state, tick) -> 8-bit multi-hot."""
    name = "base"

    def reset(self, seed, level_id):
        self.rng = XorShift128(seed=seed)
        self.level_id = level_id

    def act(self, state, tick):
        return NOOP


class RandomLegal(Policy):
    """Independent multi-hot per tick; left/right mutually exclusive, right-biased."""
    name = "random_legal"

    def act(self, state, tick):
        r = self.rng
        d = r.randint(8)
        left, right = (1, 0) if d < 2 else (0, 1) if d < 6 else (0, 0)
        return (left, right, int(r.randint(8) == 0), int(r.randint(8) == 0),
                int(r.randint(4) == 0), int(r.randint(5) < 2),
                int(r.randint(24) == 0), int(r.randint(32) == 0))


class StickyMacro(Policy):
    """Persistent macros: run right + periodic jump + fire bursts + rare grenade."""
    name = "sticky_macro"

    def reset(self, seed, level_id):
        super().reset(seed, level_id)
        self.phase = self.rng.randint(40)

    def act(self, state, tick):
        c = (tick + self.phase) % 40
        jump = 1 if c < 10 else 0
        fire = 1 if 14 <= c < 30 else 0
        gren = 1 if c == 32 and self.rng.randint(4) == 0 else 0
        return (0, 1, 0, 0, fire, jump, gren, 0)


class ScriptedClear(Policy):
    """Open-loop hardcoded clear sequence for the level's base variant."""
    name = "scripted_clear"

    def act(self, state, tick):
        script, tail = SCRIPTS[self.level_id]
        return script[tick] if tick < len(script) else tail


class FailureMining(Policy):
    """Scripted base with a randomized 40-tick corruption window (death mining)."""
    name = "failure_mining"

    def reset(self, seed, level_id):
        super().reset(seed, level_id)
        script, _ = SCRIPTS[level_id]
        self.t0 = 10 + self.rng.randint(max(1, len(script) - 20))
        self.mode = self.rng.randint(3)   # 0: freeze, 1: no-jump, 2: random mash

    def act(self, state, tick):
        script, tail = SCRIPTS[self.level_id]
        base = script[tick] if tick < len(script) else tail
        if self.t0 <= tick < self.t0 + 40:
            if self.mode == 0:
                return NOOP                                   # stall under hazards
            if self.mode == 1:
                return (base[0], base[1], base[2], base[3], base[4], 0, 0, base[7])
            r = self.rng
            return (int(r.randint(2)), int(r.randint(2)), 0, int(r.randint(4) == 0),
                    int(r.randint(2)), int(r.randint(3) == 0),
                    int(r.randint(8) == 0), 0)
        return base


POLICIES = {p.name: p for p in (RandomLegal, StickyMacro, ScriptedClear,
                                FailureMining)}


def make_policy(name):
    return POLICIES[name]()
