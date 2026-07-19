# 🕹️ MISSION-WM — two tiny games built to be learned

**Play them in your browser: https://lixuan27.github.io/mission-wm/** — the exact
deterministic simulators our neural world models train on, running as real
Python + NumPy in your tab (Pyodide). No server, no GPU.

- **SLUGLINE** — an IWBTG-flavored run-and-gun: 3 weapons, grenades, spikes,
  hostages, checkpoints, 3 stages.
- **RIDGELINE** — an OutRun-style pseudo-3D racer: curvature/hill scanline
  projection, traffic, nitro, checkpoint time-attack.

Both engines are ~1000 lines of pure Python+NumPy each, bit-deterministic
(fixed-point physics, in-state xorshift RNG), with microsecond save-states
(`GameState` is a serializable dict) and free ground-truth labels for every
action, event (hit/kill/pickup/rescue/crash/overtake…) and state variable
(lives/ammo/score/speed). One collector produces action-labeled frames plus
**9-way counterfactual forks** from any moment — the same past, a different key.

## Why

Mission-WM builds *model-only* game engines — neural networks that generate a
playable game frame-by-frame from pixels, inputs and their own memory, with no
simulator at inference time. The research agenda on top of these engines:

1. **Branch-consistency training** — paired ground-truth counterfactual
   branches teach the model what a button *causes* (and what it must not touch);
2. **Event-memory ledger** — discrete game state supervised by the engine in
   training, purely self-predicted at inference;
3. **BreakLength** — a legal-action red-team metric: the shortest input
   sequence that makes the neural game break a rule.

The neural twins of both games are training now; they will be playable here,
side by side with these originals.

## Run locally

```bash
pip install numpy h5py
PYTHONPATH=src python - <<'PY'
from game.levels import make_level
from game.engine import step
st = make_level(1)
st, events = step(st, (0,1,0,0,1,0,0,0))   # hold right + fire
PY
# collect action-labeled data with counterfactual forks:
PYTHONPATH=src python src/game/collect.py --out data.h5 --episodes 10 --game slugline
```

MIT © 2026 [@lixuan27](https://github.com/lixuan27) · sister project:
[butterfly](https://github.com/lixuan27/butterfly) (save/rewind/fork for
pretrained neural games)
