# I taught an AI to play the real Geometry Dash — here's everything that went wrong first

> The clean version of this story is "I built an RL agent that beats Geometry Dash." The honest version is more interesting: my agent died at 11%, got stuck at 35%, and I spent three hours fighting a bug that was silently sabotaging every run. This is the honest version.

*(Cover image: the `agent_clears_stereo_madness.gif` clip — the green ship weaving through the level.)*

If you've never played Geometry Dash, here's all you need to know: you control a little square that moves right on its own, and you press one button to jump. Touch anything and you instantly die. The whole game is timing. It looks trivial and it is absolutely not.

I wanted to see if I could get a reinforcement-learning agent to play it — and not some Python re-creation of the game, the *actual* game you buy on Steam. That "actual game" part is where things got interesting.

## The obvious approach is a trap

Every hobby project I found does the same thing: they wire an RL agent directly into the running game and let it learn by playing. It works, kind of. But there's a wall you hit immediately — the game runs at 60 frames per second, and it will not go faster. Reinforcement learning is hungry; it wants millions of frames. At 60 frames a second, "millions of frames" means days of your computer playing Geometry Dash in real time.

That felt like the wrong problem to be solving. So I flipped it.

Instead of training *in* the game, I wrote a small physics simulator that mimics Geometry Dash's movement — just the parts that matter, no graphics, no sound. Because it's stripped down, it runs at over **two thousand times real speed**. An agent that would take days to train in the game trains in *seconds* in the sim. The real game stops being the training ground and becomes the exam: I train in the fast fake world, then see if what the agent learned survives contact with the real one.

That "does it survive contact" question — the sim-to-real gap — turned out to be the actual heart of the project.

## A three-way fight, and a genuinely surprising loser

Before touching the real game, I used the sim to run an experiment I'd been curious about: throw three different families of algorithms at the same levels, under identical conditions, and see who wins. Value-based (DQN), policy-gradient (PPO), and a genetic algorithm — the evolutionary, "breed the best networks" approach that everyone treats as the unfashionable option.

*(Image: the `comparison.png` chart.)*

I expected PPO to win. It's the modern default, the one everyone reaches for. It lost.

On the harder level, PPO reliably got **stuck** — not failing randomly, but converging to the same wrong answer every time. Here's why, and it's a neat little lesson: the level gives you free progress up to the first risky jump, and dying costs a penalty. PPO, being cautious and on-policy, learned "collect the free progress, don't take the scary jump." It parked itself at a local optimum and refused to leave. I threw six million training steps at it. Didn't matter.

Meanwhile the "unfashionable" genetic algorithm and the old-reliable DQN both solved every level, every random seed. The GA was the fastest of all — which makes sense once you realize Geometry Dash levels are *deterministic*. The same inputs always produce the same run. That's the exact regime where an algorithm that just memorizes a good input sequence shines.

The takeaway I keep coming back to: "state of the art" is a property of the *problem*, not the algorithm. You only learn things like this by actually running the comparison instead of trusting the leaderboard.

## Meeting the real game (and a gut punch)

To connect my Python agents to the retail game, I had to write a mod in C++ using Geode, the Geometry Dash modding framework. The mod hooks into the game's update loop, reads out the cube's real position and velocity every frame, and injects the jump input — all streamed over a local socket to my Python code.

Getting this working on the actual game was its own saga (more on the bugs later), but the moment it connected and I watched a policy — trained *entirely in my fake sim, never having seen the real game* — start driving the real cube, jumping real spikes... that was the best moment of the project.

It also died at 11%.

And honestly? That death is the whole point. My sim's physics are an *approximation* of the real game's. Close, but not frame-perfect. The agent learned to jump with millisecond timing tuned to my sim's slightly-wrong gravity, and over enough obstacles that small error compounds into a death. This is the sim-to-real gap in its purest form, and no amount of clever ML makes it disappear — you have to actually measure the difference and close it.

## Getting clever with determinism

11% wasn't going to cut it. So I leaned into that "deterministic" property.

If the same inputs always produce the same result, then beating a level is really a search problem: find the input sequence that survives. And Geometry Dash has a feature that makes this tractable — **practice mode checkpoints**. Clear a chunk, drop a checkpoint, and you respawn there instead of the start.

So I built a search that runs *on the real game*: from the last checkpoint, try jump patterns; when one survives a stretch, lock it in with a checkpoint and search the next stretch. Each attempt only replays a few seconds, not the whole level. Nothing has to transfer from the sim — it's optimizing against the real game's own physics.

It worked. The frontier marched forward — 12%, 30%, and then it blew through a long stretch in one lucky run to **35%**.

*(Image: a frame of the checkpoint search, or the `real_stereo_clip.gif`.)*

Then it hit the ship section and stopped dead.

The ship is a different beast. Instead of discrete jumps, you're holding the button to fly, threading a corridor of spikes on the floor and ceiling. It's *continuous control*, and my "try some jump timings" search had no idea what to do with it. It could brute-force discrete jumps forever and never fly a ship.

## The pivot: just show it how

This is where I stopped trying to be clever and asked a better question. If the hard part is the ship section, and *I* can fly the ship section... why not let the agent learn from me?

This is learning from demonstration, and it sidesteps everything I was stuck on. I built a "record mode" into the mod — a passive one, where the game runs at full native speed and the mod just *watches* my inputs each frame instead of taking control (getting that lag-free was its own fix; the first version frame-locked the game and made it unplayable). I'd play the level myself, once, all the way through. The mod records my exact frame-by-frame inputs. Then, because the game is deterministic, I replay those inputs and the agent reproduces my run perfectly.

Simple idea. It did not work. It kept dying at 62%.

## The bug that ate three hours

Here's the part I'd normally leave out of a portfolio, which is exactly why I'm putting it in.

The replay reproduced my run flawlessly through the entire first half — cube section, ship section, all of it — and then died at 62%. Every time. I tried three different ways of feeding the inputs back. I rewrote how the mod applies button presses. I forced a fixed physics timestep so the replay couldn't drift. Each fix required rebuilding the mod and relaunching the game, and each attempt died in roughly the same place. I was convinced it was a deep, unfixable frame-timing problem in the game's continuous-control physics. I was about ready to write it up as "reproduces 62%, full clear is future work."

Then I noticed something in the logs: the game was **crashing** on shutdown. My mod had a background thread that wasn't being torn down safely, and it was corrupting the game's state — which meant every "clean" test I'd run afterward was starting from a subtly broken state. The 62% wasn't a physics limit. It was my own crash poisoning the well.

I fixed the crash. And here's the human part: I was still skeptical it would matter, and I almost didn't bother trying again. The person I was working through this with pushed me to just run it one more time on a genuinely clean launch.

`REPLAY CLEARED THE LEVEL (100%)`.

*(Image: the full-clear `agent_clears_stereo_madness.mp4`.)*

The fix had been right the whole time. The crash was the thing lying to me. I'd been debugging the wrong layer for hours — a very ordinary, very humbling engineering experience that no clean writeup ever admits to.

## Making it "learned," honestly

Now, I want to be precise about what that clear *is*, because this is where people oversell and get caught. Deterministic replay of a recorded run is a systems achievement — it proves the mod, the bridge, and the determinism all work — but it is *not* "an RL agent autonomously discovered how to beat the level." It's playback.

So I took the demonstration data — the game state and my input, every frame — and trained a **behavior-cloning** policy: a neural network that learns to predict "would a human jump right now?" from the game state. It hit **90.7% accuracy** on held-out frames, with a solid F1 on the jump decisions specifically (jumps are rare, so raw accuracy would be misleading — always worth checking).

That's a genuinely *learned* policy that imitates my play. It also drifts if you let it drive the whole level solo, which is the honest, well-known failure mode of behavior cloning on a single frame-perfect task — one wrong frame and the errors snowball. So the guaranteed clear still uses replay; the learned policy is the "it actually learned something" companion, not a magic solo act. Being clear about that distinction is, I think, more impressive than pretending otherwise.

## What I'd tell myself at the start

- **Decouple the expensive thing.** The single best decision was building the fast sim so I wasn't held hostage to 60 fps. Most of the ML only worked *because* iteration was cheap.
- **Run the experiment; don't trust the reputation.** The PPO-gets-stuck result is my favorite finding, and it came from seeds and controls, not a bigger model.
- **The bug is usually one layer below where you're looking.** I spent hours "fixing" a physics problem that was actually a memory-management crash. When results are inexplicably consistent-but-wrong, suspect your harness before your theory.
- **Say what your result actually is.** "Completed the level via learning from demonstration" is true and defensible. "My RL agent beat Geometry Dash" is a sentence someone will take apart in an interview.

The code, all the footage, the algorithm comparison, and a blunt list of what still doesn't work are on GitHub: **[github.com/abs768/geometry-dash-rl](https://github.com/abs768/geometry-dash-rl)**.

It plays the real game, start to finish. Getting there taught me more about debugging and honesty than about reinforcement learning — which is probably the most honest thing I can say about it.
