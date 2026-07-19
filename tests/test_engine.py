"""Slugline engine test suite (unittest, CPU, seconds-scale).

Covers: determinism golden test, serialize round-trip + fork isolation, jump
physics bounds, spike insta-death + checkpoint respawn, bullet kill + score,
ammo depletion auto-switch, scripted clears, render contract, collect smoke.
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

from game import (GameState, NOOP, make_level, render, serialize, deserialize,
                  step)
from game.engine import FP, TILE
from game.policies import RandomLegal, ScriptedClear
from game import collect as collect_mod

RIGHT = (0, 1, 0, 0, 0, 0, 0, 0)
LEFT = (1, 0, 0, 0, 0, 0, 0, 0)
RIGHT_FIRE = (0, 1, 0, 0, 1, 0, 0, 0)
JUMP = (0, 0, 0, 0, 0, 1, 0, 0)
FIRE = (0, 0, 0, 0, 1, 0, 0, 0)
RIGHT_JUMP = (0, 1, 0, 0, 0, 1, 0, 0)


def _run_policy(level_id, variant_seed, actions, n):
    """Roll a fresh level for n ticks; return (frames, final_dict, events)."""
    st = make_level(level_id, variant_seed=variant_seed)
    frames, events = [], []
    for t in range(n):
        frames.append(render(st))
        st, ev = step(st, actions[t])
        events.extend(ev)
    return frames, serialize(st), events


class TestDeterminism(unittest.TestCase):
    def test_golden_bitwise_repeat(self):
        """Same seed + same action sequence x2 -> identical frames and state."""
        pol = RandomLegal()
        pol.reset(seed=1234, level_id=3)
        st = make_level(3, variant_seed=5)
        actions = []
        for t in range(150):
            a = pol.act(st, t)
            actions.append(a)
            st, _ = step(st, a)
        f1, d1, e1 = _run_policy(3, 5, actions, 150)
        f2, d2, e2 = _run_policy(3, 5, actions, 150)
        for a, b in zip(f1, f2):
            self.assertTrue(np.array_equal(a, b))
        self.assertEqual(d1, d2)
        self.assertEqual(e1, e2)

    def test_json_roundtrip_continuation(self):
        """State surviving json dump/load continues bit-identically."""
        st = make_level(2)
        for _ in range(40):
            st, _ = step(st, RIGHT)
        d = json.loads(json.dumps(serialize(st)))
        st_a, st_b = deserialize(d), deserialize(serialize(st))
        for _ in range(30):
            st_a, _ = step(st_a, RIGHT_JUMP)
            st_b, _ = step(st_b, RIGHT_JUMP)
        self.assertEqual(serialize(st_a), serialize(st_b))


class TestSerializeFork(unittest.TestCase):
    def test_roundtrip_and_fork_isolation(self):
        st = make_level(1)
        for _ in range(40):
            st, _ = step(st, RIGHT)
        d = serialize(st)
        d_frozen = json.loads(json.dumps(d))
        fork_a, fork_b = deserialize(d), deserialize(d)
        self.assertEqual(serialize(fork_a), d_frozen)      # round trip
        for _ in range(25):
            fork_a, _ = step(fork_a, LEFT)
        mid_b = serialize(fork_b)
        self.assertEqual(mid_b, d_frozen)                  # A stepping left B alone
        for _ in range(25):
            fork_b, _ = step(fork_b, RIGHT)
        self.assertEqual(d, d_frozen)                      # source dict untouched
        self.assertLess(fork_a.player["x"], fork_b.player["x"])
        self.assertNotEqual(serialize(fork_a), serialize(fork_b))


class TestJumpPhysics(unittest.TestCase):
    def _peak(self, hold_ticks):
        st = make_level(1)
        y0 = st.player["y"]
        peak = y0
        for t in range(40):
            a = JUMP if t < hold_ticks else NOOP
            st, _ = step(st, a)
            peak = min(peak, st.player["y"])
        return (y0 - peak) / FP

    def test_variable_jump_height(self):
        full = self._peak(20)
        tap = self._peak(2)
        self.assertGreaterEqual(full, 3.0 * TILE)          # >= 3 tiles
        self.assertLessEqual(full, 4.5 * TILE)             # <= 4.5 tiles
        self.assertLess(tap, full)                         # variable jump works
        self.assertGreaterEqual(tap, 1.0 * TILE)


class TestSpikeDeath(unittest.TestCase):
    def test_spike_instadeath_respawn_costs_life(self):
        st = make_level(2)                                 # spikes at cols 6-7
        spawn = list(st.checkpoint)
        death = None
        for _ in range(40):                                # run right, no jump
            st, ev = step(st, RIGHT)
            death = next((e for e in ev if e["type"] == "player_death"), None)
            if death:
                break
        self.assertIsNotNone(death)
        self.assertEqual(death["cause"], "spike")
        self.assertFalse(st.player["alive"])
        self.assertEqual(st.lives, 2)
        for _ in range(20):                                # wait out death timer
            st, _ = step(st, NOOP)
        self.assertTrue(st.player["alive"])
        self.assertEqual([st.player["x"], st.player["y"]], spawn)


class TestCombat(unittest.TestCase):
    def test_bullet_kills_walker_scores(self):
        st = make_level(1)
        kills = []
        for _ in range(60):
            st, ev = step(st, RIGHT_FIRE)
            kills += [e for e in ev if e["type"] == "kill"]
            if kills:
                break
        self.assertTrue(kills, "no kill within 60 ticks")
        self.assertEqual(kills[0]["target"], "walker")
        self.assertEqual(kills[0]["score"], 100)
        self.assertGreaterEqual(st.score, 100)

    def test_ammo_depletion_auto_switch_to_pistol(self):
        st = make_level(1)
        st.player["ammo"]["mg"] = 2
        st.player["weapon"] = "mg"
        switches = []
        for _ in range(6):
            st, ev = step(st, FIRE)
            switches += [e for e in ev if e["type"] == "weapon_switch"]
        self.assertEqual(st.player["ammo"]["mg"], 0)
        self.assertEqual(st.player["weapon"], "pistol")
        self.assertTrue(any(e.get("auto") for e in switches))


class TestScriptedClear(unittest.TestCase):
    def _clear(self, level_id):
        pol = ScriptedClear()
        pol.reset(seed=0, level_id=level_id)
        st = make_level(level_id)
        for t in range(600):
            st, ev = step(st, pol.act(st, t))
            if any(e["type"] == "level_clear" for e in ev):
                return st
        self.fail("scripted_clear did not clear L%d" % level_id)

    def test_l1_clear(self):
        st = self._clear(1)
        self.assertTrue(st.level_clear)
        self.assertEqual(st.lives, 3)

    def test_l2_l3_clear(self):
        self.assertTrue(self._clear(2).level_clear)
        self.assertTrue(self._clear(3).level_clear)


class TestRender(unittest.TestCase):
    def test_shape_dtype_and_difference(self):
        st = make_level(1)
        f0 = render(st)
        self.assertEqual(f0.shape, (112, 160, 3))
        self.assertEqual(f0.dtype, np.uint8)
        for _ in range(30):
            st, _ = step(st, RIGHT_JUMP)
        f1 = render(st)
        self.assertEqual(f1.shape, (112, 160, 3))
        self.assertTrue(np.any(f0 != f1))


class TestCollectSmoke(unittest.TestCase):
    def test_small_collection(self):
        import h5py
        tmp = tempfile.mkdtemp(prefix="slug_collect_")
        try:
            out = os.path.join(tmp, "smoke.h5")
            n = collect_mod.collect(out, episodes=2, policy_mix="30,25,25,20",
                                    fork_every=50, fork_branches=9, seed=7,
                                    max_ticks=120, levels=(1, 2))
            self.assertGreater(n, 0)
            with h5py.File(out, "r") as f:
                rows = f["frames"].shape[0]
                self.assertEqual(rows, n)
                for name in ("actions", "episode_id", "branch_id", "tick"):
                    self.assertEqual(f[name].shape[0], rows)
                self.assertEqual(f["frames"].shape[1:], (112, 160, 3))
                self.assertEqual(f["actions"].shape[1], 8)
                self.assertEqual(f["frames"].dtype, np.uint8)
                branch_ids = set(int(b) for b in f["branch_id"][:])
                self.assertIn(0, branch_ids)
                self.assertGreater(max(branch_ids), 0)     # forks recorded
                _ = f["frames"][rows - 1]                  # readable back
            with open(os.path.join(tmp, "smoke.events.jsonl")) as fh:
                events = [json.loads(line) for line in fh]
            self.assertTrue(all("type" in e and "tick" in e for e in events))
            with open(os.path.join(tmp, "smoke.states.jsonl")) as fh:
                snaps = [json.loads(line) for line in fh]
            self.assertGreaterEqual(len(snaps), 2)         # >=1 fork point/episode
            st = deserialize(snaps[0]["state"])            # snapshot rebuilds
            self.assertIsInstance(st, GameState)
            step(st, NOOP)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
