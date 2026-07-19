"""Ridgeline racer test suite (unittest, CPU, seconds-scale; mirrors test_engine).

Covers: determinism golden test, serialize round-trip + fork isolation,
centrifugal cornering physics, crash spin-out + invulnerability, checkpoint
time bonus + timeout, scripted_clear 3-stage finish, render contract,
collect --game racer smoke.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from racer import GameState, NOOP, make_level, render, serialize, deserialize, step
from racer.engine import (CHECK_BONUS, ROAD_HALF, SEG_LEN, SPIN_TICKS,
                          STAGE_SEGS, N_STAGES)
from racer.policies import RandomLegal, make_policy
from game import collect as collect_mod

LEFT = (1, 0, 0, 0, 0, 0, 0, 0)
RIGHT = (0, 1, 0, 0, 0, 0, 0, 0)
THROTTLE = (0, 0, 1, 0, 0, 0, 0, 0)
RIGHT_THROTTLE = (0, 1, 1, 0, 0, 0, 0, 0)


def _run(variant_seed, actions, n):
    st = make_level(1, variant_seed=variant_seed)
    frames, events = [], []
    for t in range(n):
        frames.append(render(st))
        st, ev = step(st, actions[t])
        events.extend(ev)
    return frames, serialize(st), events


class TestDeterminism(unittest.TestCase):
    def test_golden_bitwise_repeat(self):
        """Same seed + same action sequence x2 -> identical frames/state/events."""
        pol = RandomLegal()
        pol.reset(seed=99, level_id=1)
        st = make_level(1, variant_seed=3)
        actions = []
        for t in range(150):
            a = pol.act(st, t)
            actions.append(a)
            st, _ = step(st, a)
        f1, d1, e1 = _run(3, actions, 150)
        f2, d2, e2 = _run(3, actions, 150)
        for a, b in zip(f1, f2):
            self.assertTrue(np.array_equal(a, b))
        self.assertEqual(d1, d2)
        self.assertEqual(e1, e2)

    def test_json_roundtrip_continuation(self):
        st = make_level(2)
        for _ in range(60):
            st, _ = step(st, THROTTLE)
        d = json.loads(json.dumps(serialize(st)))
        st_a, st_b = deserialize(d), deserialize(serialize(st))
        for _ in range(40):
            st_a, _ = step(st_a, RIGHT_THROTTLE)
            st_b, _ = step(st_b, RIGHT_THROTTLE)
        self.assertEqual(serialize(st_a), serialize(st_b))


class TestSerializeFork(unittest.TestCase):
    def test_roundtrip_and_fork_isolation(self):
        st = make_level(1)
        for _ in range(80):
            st, _ = step(st, THROTTLE)
        d = serialize(st)
        d_frozen = json.loads(json.dumps(d))
        fork_a, fork_b = deserialize(d), deserialize(d)
        self.assertEqual(serialize(fork_a), d_frozen)
        for _ in range(25):
            fork_a, _ = step(fork_a, LEFT)
        self.assertEqual(serialize(fork_b), d_frozen)      # A stepping left B alone
        for _ in range(25):
            fork_b, _ = step(fork_b, RIGHT_THROTTLE)
        self.assertEqual(d, d_frozen)                      # source dict untouched
        self.assertLess(fork_a.player["x"], fork_b.player["x"])
        self.assertLess(fork_a.player["dist"], fork_b.player["dist"])


class TestCentrifugal(unittest.TestCase):
    def _sharp_curve_state(self, spd):
        st = make_level(3)                                 # stage 3: c=3 at seg 186
        st.cars = []
        st.player["dist"] = 186 * SEG_LEN
        st.player["spd"] = spd
        return st

    def test_no_steer_flies_off(self):
        """Full speed into a c=3 bend without steering -> off the road."""
        st = self._sharp_curve_state(512)
        offroad_ev = []
        for _ in range(40):
            st, ev = step(st, THROTTLE)
            offroad_ev += [e for e in ev if e["type"] == "offroad_enter"]
        self.assertTrue(offroad_ev)
        self.assertGreater(abs(st.player["x"]), ROAD_HALF)

    def test_counter_steer_at_speed_limit_holds(self):
        """Counter-steering at the brake-to speed keeps the car on the road."""
        st = self._sharp_curve_state(320)
        for _ in range(40):
            st, ev = step(st, RIGHT)                       # c>0 pushes -x
            self.assertFalse([e for e in ev if e["type"] == "offroad_enter"])
        self.assertLessEqual(abs(st.player["x"]), ROAD_HALF)


class TestCrash(unittest.TestCase):
    def test_crash_spinout_and_invuln(self):
        st = make_level(1)
        st.cars = [{"id": 999, "dist": st.player["dist"] + 500, "x": 0,
                    "dir": 1, "spd": 0, "color": 0, "passed": False}]
        crash = None
        for _ in range(120):
            st, ev = step(st, THROTTLE)
            crash = next((e for e in ev if e["type"] == "crash"), None)
            if crash:
                break
        self.assertIsNotNone(crash)
        self.assertEqual(crash["car_id"], 999)
        self.assertEqual(st.player["spd"], 0)
        self.assertEqual(st.player["spin"], SPIN_TICKS)
        self.assertGreater(st.player["invuln"], 0)
        while st.player["invuln"] > 0:                     # no re-crash while invuln
            st, ev = step(st, NOOP)
            self.assertFalse([e for e in ev if e["type"] == "crash"])


class TestTiming(unittest.TestCase):
    def test_checkpoint_adds_time(self):
        st = make_level(1)
        st.cars = []
        st.player["dist"] = STAGE_SEGS * SEG_LEN - 60
        st.player["spd"] = 512
        t0 = st.timer
        cp = None
        for _ in range(10):
            st, ev = step(st, THROTTLE)
            cp = next((e for e in ev if e["type"] == "checkpoint"), None)
            if cp:
                break
        self.assertIsNotNone(cp)
        self.assertEqual(cp["time_added"], CHECK_BONUS)
        self.assertGreater(st.timer, t0)
        self.assertEqual(st.stage, 2)

    def test_timeout_game_over_freezes(self):
        st = make_level(1)
        st.timer = 5
        to = []
        for _ in range(8):
            st, ev = step(st, NOOP)
            to += [e for e in ev if e["type"] == "time_out"]
        self.assertEqual(len(to), 1)
        self.assertTrue(st.game_over)
        tick0 = st.tick
        st, ev = step(st, THROTTLE)                        # frozen after game over
        self.assertEqual((st.tick, ev), (tick0, []))


class TestScriptedClear(unittest.TestCase):
    def test_autopilot_finishes_three_stages(self):
        pol = make_policy("scripted_clear")
        pol.reset(seed=1, level_id=1)
        st = make_level(1)
        checkpoints, finish = 0, None
        for t in range(4000):
            st, ev = step(st, pol.act(st, t))
            checkpoints += sum(1 for e in ev if e["type"] == "checkpoint")
            finish = finish or next((e for e in ev if e["type"] == "finish"), None)
            if st.level_clear or st.game_over:
                break
        self.assertIsNotNone(finish, "autopilot did not finish the course")
        self.assertTrue(st.level_clear)
        self.assertEqual(checkpoints, N_STAGES - 1)
        self.assertGreater(st.timer, 0)


class TestRender(unittest.TestCase):
    def test_shape_dtype_and_difference(self):
        st = make_level(1)
        f0 = render(st)
        self.assertEqual(f0.shape, (112, 160, 3))
        self.assertEqual(f0.dtype, np.uint8)
        for _ in range(60):
            st, _ = step(st, RIGHT_THROTTLE)
        f1 = render(st)
        self.assertEqual(f1.shape, (112, 160, 3))
        self.assertTrue(np.any(f0 != f1))


class TestCollectRacer(unittest.TestCase):
    def test_small_racer_collection(self):
        import h5py
        tmp = tempfile.mkdtemp(prefix="ridge_collect_")
        try:
            out = os.path.join(tmp, "smoke.h5")
            n = collect_mod.collect(out, episodes=2, policy_mix="30,25,25,20",
                                    fork_every=60, fork_branches=9, seed=11,
                                    max_ticks=150, levels=(1, 2), game="racer")
            self.assertGreater(n, 0)
            with h5py.File(out, "r") as f:
                rows = f["frames"].shape[0]
                self.assertEqual(rows, n)
                self.assertEqual(f.attrs["game"], "racer")
                for name in ("actions", "episode_id", "branch_id", "tick"):
                    self.assertEqual(f[name].shape[0], rows)
                self.assertEqual(f["frames"].shape[1:], (112, 160, 3))
                branch_ids = set(int(b) for b in f["branch_id"][:])
                self.assertIn(0, branch_ids)
                self.assertGreater(max(branch_ids), 0)
            with open(os.path.join(tmp, "smoke.states.jsonl")) as fh:
                snaps = [json.loads(line) for line in fh]
            self.assertGreaterEqual(len(snaps), 2)
            st = deserialize(snaps[0]["state"])
            self.assertIsInstance(st, GameState)
            step(st, NOOP)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
