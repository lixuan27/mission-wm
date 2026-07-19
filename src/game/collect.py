"""Mission-WM data collector CLI: episodes -> h5 (+ events/state jsonl).

Role: rolls out the four SPEC-section-5 policies over 3 levels (+variants),
records frames/actions/ids, and at every --fork-every ticks serializes the
state and rolls out 9 preset counterfactual action branches for 60 ticks each.
`--game slugline` (default): branches = noop/left/right/jump/shoot/grenade/
jump+shoot/crouch+shoot/up-aim-shoot.  `--game racer` (Ridgeline): branches =
noop/left/right/throttle/brake/left+throttle/right+throttle/brake+left/nitro.
Both games share the exact h5 schema (attrs record which game was collected).

Output:
  <out>.h5           frames(N,112,160,3 u8 lzf) actions(N,8 u8)
                     episode_id/branch_id/tick (i32)
  <out>.events.jsonl one json line per event (episode_id/branch_id merged in)
  <out>.states.jsonl one json line per fork point (full serialized GameState)

Usage:
  python -m game.collect --out data/slug.h5 --episodes 12 \
      --policy-mix 30,25,25,20 --fork-every 150 --fork-branches 9 --seed 1
  python -m game.collect --game racer --out data/ridge.h5 --episodes 6 --seed 1
"""

import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):                     # allow `python src/game/collect.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from game.engine import XorShift128, deserialize, serialize, step
    from game.levels import make_level
    from game.policies import make_policy
    from game.render import render
else:
    from .engine import XorShift128, deserialize, serialize, step
    from .levels import make_level
    from .policies import make_policy
    from .render import render

POLICY_ORDER = ("random_legal", "sticky_macro", "scripted_clear", "failure_mining")

# SPEC section 5 counterfactual branch action patterns (held for the whole branch)
BRANCH_ACTIONS = [
    ("noop",         (0, 0, 0, 0, 0, 0, 0, 0)),
    ("left",         (1, 0, 0, 0, 0, 0, 0, 0)),
    ("right",        (0, 1, 0, 0, 0, 0, 0, 0)),
    ("jump",         (0, 0, 0, 0, 0, 1, 0, 0)),
    ("shoot",        (0, 0, 0, 0, 1, 0, 0, 0)),
    ("grenade",      (0, 0, 0, 0, 0, 0, 1, 0)),
    ("jump_shoot",   (0, 0, 0, 0, 1, 1, 0, 0)),
    ("crouch_shoot", (0, 0, 0, 1, 1, 0, 0, 0)),
    ("up_shoot",     (0, 0, 1, 0, 1, 0, 0, 0)),
]
BRANCH_ACTIONS_RACER = [
    ("noop",           (0, 0, 0, 0, 0, 0, 0, 0)),
    ("left",           (1, 0, 0, 0, 0, 0, 0, 0)),
    ("right",          (0, 1, 0, 0, 0, 0, 0, 0)),
    ("throttle",       (0, 0, 1, 0, 0, 0, 0, 0)),
    ("brake",          (0, 0, 0, 1, 0, 0, 0, 0)),
    ("left_throttle",  (1, 0, 1, 0, 0, 0, 0, 0)),
    ("right_throttle", (0, 1, 1, 0, 0, 0, 0, 0)),
    ("brake_left",     (1, 0, 0, 1, 0, 0, 0, 0)),
    ("nitro",          (0, 0, 0, 0, 1, 0, 0, 0)),
]
BRANCH_TICKS = 60
GAME_DEFAULT_MAX_TICKS = {"slugline": 900, "racer": 2200}


def _game_api(game):
    """Per-game module bundle; both games expose the identical API surface."""
    if game == "racer":
        from racer.engine import step as r_step, serialize as r_ser, \
            deserialize as r_deser
        from racer.levels import make_level as r_make_level
        from racer.policies import make_policy as r_make_policy
        from racer.render import render as r_render
        return {"step": r_step, "serialize": r_ser, "deserialize": r_deser,
                "make_level": r_make_level, "make_policy": r_make_policy,
                "render": r_render, "branches": BRANCH_ACTIONS_RACER}
    return {"step": step, "serialize": serialize, "deserialize": deserialize,
            "make_level": make_level, "make_policy": make_policy,
            "render": render, "branches": BRANCH_ACTIONS}


def _append(ds, arr):
    n0 = ds.shape[0]
    ds.resize(n0 + len(arr), axis=0)
    ds[n0:] = arr


class _Writer:
    """Incremental h5 + jsonl writer."""

    def __init__(self, out_path, meta):
        import h5py
        self.h5 = h5py.File(out_path, "w")
        self.h5.create_dataset("frames", shape=(0, 112, 160, 3), dtype=np.uint8,
                               maxshape=(None, 112, 160, 3),
                               chunks=(64, 112, 160, 3), compression="lzf")
        self.h5.create_dataset("actions", shape=(0, 8), dtype=np.uint8,
                               maxshape=(None, 8), chunks=(1024, 8))
        for name in ("episode_id", "branch_id", "tick"):
            self.h5.create_dataset(name, shape=(0,), dtype=np.int32,
                                   maxshape=(None,), chunks=(4096,))
        for k, v in meta.items():
            self.h5.attrs[k] = v
        base = out_path[:-3] if out_path.endswith(".h5") else out_path
        self.events_f = open(base + ".events.jsonl", "w")
        self.states_f = open(base + ".states.jsonl", "w")

    def add_rows(self, frames, actions, ep, branch, ticks):
        _append(self.h5["frames"], np.asarray(frames, dtype=np.uint8))
        _append(self.h5["actions"], np.asarray(actions, dtype=np.uint8))
        _append(self.h5["episode_id"], np.full(len(frames), ep, dtype=np.int32))
        _append(self.h5["branch_id"], np.full(len(frames), branch, dtype=np.int32))
        _append(self.h5["tick"], np.asarray(ticks, dtype=np.int32))

    def add_events(self, events, ep, branch):
        for e in events:
            rec = {"episode_id": ep, "branch_id": branch}
            rec.update(e)
            self.events_f.write(json.dumps(rec) + "\n")

    def add_state(self, ep, fork_idx, state_dict):
        self.states_f.write(json.dumps(
            {"episode_id": ep, "fork_idx": fork_idx, "tick": state_dict["tick"],
             "state": state_dict}) + "\n")

    def close(self):
        n = self.h5["frames"].shape[0]
        self.h5.close()
        self.events_f.close()
        self.states_f.close()
        return n


def _rollout_branch(api, writer, snap, ep, branch_id, action):
    """Roll a counterfactual branch from a serialized fork point."""
    st = api["deserialize"](snap)
    frames, actions, ticks, events = [], [], [], []
    for _ in range(BRANCH_TICKS):
        if st.game_over or st.level_clear:
            break
        frames.append(api["render"](st))
        actions.append(action)
        ticks.append(st.tick)
        st, ev = api["step"](st, action)
        events.extend(ev)
    if frames:
        writer.add_rows(frames, actions, ep, branch_id, ticks)
        writer.add_events(events, ep, branch_id)


def collect(out, episodes, policy_mix, fork_every, fork_branches, seed,
            max_ticks=900, levels=(1, 2, 3), game="slugline"):
    """Run the collection; returns total rows written."""
    api = _game_api(game)
    rng = XorShift128(seed=seed)
    mix = [int(x) for x in policy_mix.split(",")]
    assert len(mix) == 4 and sum(mix) > 0, "policy-mix must be 4 comma ints"
    cum = np.cumsum(mix)
    writer = _Writer(out, {"seed": seed, "policy_mix": policy_mix,
                           "fork_every": fork_every,
                           "fork_branches": fork_branches,
                           "episodes": episodes, "max_ticks": max_ticks,
                           "game": game})
    branch_set = api["branches"][:fork_branches]
    for ep in range(episodes):
        lid = levels[ep % len(levels)]
        r = rng.randint(int(cum[-1]))
        pname = POLICY_ORDER[int(np.searchsorted(cum, r, side="right"))]
        # scripted runs on the base layout (sequences are verified there);
        # other policies also explore jittered variants
        vseed = 0 if pname == "scripted_clear" else rng.randint(5) * (
            1 + rng.randint(999))
        pol = api["make_policy"](pname)
        pol.reset(seed=rng.next_u32(), level_id=lid)
        st = api["make_level"](lid, variant_seed=vseed)
        frames, actions, ticks, events = [], [], [], []
        fork_snaps = []
        for t in range(max_ticks):
            if st.game_over or st.level_clear:
                break
            if fork_every > 0 and t > 0 and t % fork_every == 0:
                snap = api["serialize"](st)
                writer.add_state(ep, len(fork_snaps), snap)
                fork_snaps.append(snap)
            a = pol.act(st, t)
            frames.append(api["render"](st))
            actions.append(a)
            ticks.append(st.tick)
            st, ev = api["step"](st, a)
            events.extend(ev)
        writer.add_rows(frames, actions, ep, 0, ticks)
        writer.add_events(events, ep, 0)
        for fi, snap in enumerate(fork_snaps):
            for bi, (_bname, ba) in enumerate(branch_set):
                _rollout_branch(api, writer, snap, ep,
                                fi * len(branch_set) + bi + 1, ba)
        print("episode %d: level=%d policy=%s variant=%d main_rows=%d forks=%d"
              % (ep, lid, pname, vseed, len(frames), len(fork_snaps)))
    n = writer.close()
    print("done: %d rows -> %s" % (n, out))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mission-WM data collector")
    ap.add_argument("--game", choices=("slugline", "racer"), default="slugline")
    ap.add_argument("--out", required=True, help="output .h5 path")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--policy-mix", default="30,25,25,20",
                    help="percent for random,sticky,scripted,failure")
    ap.add_argument("--fork-every", type=int, default=150,
                    help="fork counterfactual branches every K main ticks (0=off)")
    ap.add_argument("--fork-branches", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-ticks", type=int, default=None,
                    help="per-episode tick cap (default 900 slugline / 2200 racer)")
    ap.add_argument("--levels", default="1,2,3")
    args = ap.parse_args(argv)
    max_ticks = args.max_ticks if args.max_ticks is not None \
        else GAME_DEFAULT_MAX_TICKS[args.game]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    return collect(args.out, args.episodes, args.policy_mix, args.fork_every,
                   min(args.fork_branches, 9), args.seed, max_ticks,
                   tuple(int(x) for x in args.levels.split(",")),
                   game=args.game)


if __name__ == "__main__":
    main()
