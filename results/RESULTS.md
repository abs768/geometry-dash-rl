# Comparison results: DQN vs PPO vs Genetic Algorithm

Setup: 3 algorithms × 3 seeds (0, 1, 2) × 2 levels, identical observation,
action space and reward. Trained in the headless sim; evaluated with a single
**greedy** (deterministic, argmax) rollout per run — the sim is deterministic so
one rollout is an exact measurement. Reproduce with:

```bash
python sweep.py --seeds 0 1 2 --levels spikes_easy blocks_and_spikes
python evaluate.py && python plot.py
```

![comparison](comparison.png)

## Final greedy performance (% of level completed, mean over 3 seeds)

| Algorithm | spikes_easy | blocks_and_spikes | Seeds solved (of 6) |
|-----------|:-----------:|:-----------------:|:-------------------:|
| **DQN** (double)        | **100%** (3/3) | **100%** (3/3) | **6/6** |
| **Genetic Algorithm**   | **100%** (3/3) | **100%** (3/3) | **6/6** |
| **PPO**                 | **100%** (3/3) | 48% (1/3)      | 4/6 |

## Findings

**1. Value-based (DQN) and evolutionary (GA) methods are robust; PPO is not.**
DQN and the GA solve every seed on both levels. PPO solves the easy level every
time but clears `blocks_and_spikes` on only 1 of 3 seeds — the other two get
trapped at exactly 21.4% progress, the first block-over-spike section, where a
mistimed jump means death.

**2. PPO's failure is a local optimum, and compute does not fix it.** The first
~21 blocks are free forward progress; the first risky jump carries a −10 death
penalty. On-policy PPO converges to "collect the free progress, don't risk the
jump." Raising the entropy coefficient from 0.01 → 0.05 rescued seed 0, but
seeds 1 and 2 stayed trapped even when trained to **6M steps** (2× the reported
budget). This is the on-policy local-optimum trap, not undertraining.

**3. The GA is the most sample-efficient by a wide margin.** It reaches 100% on
`spikes_easy` in under a second of wall-clock and a few thousand environment
steps (hence the near-vertical green sliver in the learning-curve panels),
because a deterministic level makes a single rollout an exact fitness signal.
On `blocks_and_spikes` it converges in ~85 generations.

**4. Structured observations + the fractional-x fix make DQN stable.** With the
block-aligned grid alone, the exact jump frame was aliased across ~6 frames and
DQN was erratic; adding the sub-block x-offset to the observation took DQN to
6/6 solved. (This is why the DQN *training* curve looks noisy — it is logged
under ε-greedy exploration — while its *greedy* evaluation is 100%.)

## Why this is a stronger result than the prior single-algorithm work

[geometry-dash-ai](https://github.com/ThePickleGawd/geometry-dash-ai) reports a
DQN (and MoE-DQN) that plays the game, but with no controlled comparison across
algorithm families, no seed statistics, and training capped at real time. Here,
identical conditions across three algorithm families and three seeds surface a
concrete, reproducible finding — PPO's seed-dependent local-optimum trap on
precision timing — that a single-algorithm project cannot show. All 18 runs
finished in well under an hour on one CPU core thanks to the headless sim.

## Caveats

- Levels are two hand-built sim levels, not yet official GD levels (the
  level-string importer and the Geode bridge are the next milestones). Absolute
  numbers will change on real levels; the *relative* algorithm behavior is the
  transferable result.
- PPO was given its own tuned hyperparameters (entropy 0.05, 3M steps); DQN and
  GA use their defaults. Each method is allowed its own budget/hyperparameters,
  as is standard, but PPO is the only one that needed tuning to get anywhere on
  the hard level — itself part of the finding.
