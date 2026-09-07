# Simulator throughput

- **Machine:** macOS-26.5.2-arm64-arm-64bit / arm64
- **Python:** 3.11.14
- **Steps timed per level:** 1,000,000 (plus 10,000 warm-up)
- **Sim timestep:** DT = 1/60 s, so x real time = steps/sec / 60

Single process, one core, fixed action, no policy forward pass.

| Level | Steps/sec | x real time | Episodes |
|---|---:|---:|---:|
| `spikes_easy` | 162,947 | 2,716x | 15,151 |
| `blocks_and_spikes` | 147,371 | 2,456x | 15,151 |

Range across levels: **147,371-162,947 steps/sec** (**2,456x-2,716x** real time).

Reproduce: `python bench_sim.py`
