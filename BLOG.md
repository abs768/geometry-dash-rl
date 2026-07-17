# I tried to get an AI to play the real Geometry Dash

I spent a while getting a reinforcement learning agent to play Geometry Dash — the actual game on Steam, not a Python clone of it. It works now, but there were a lot of dead ends and one genuinely dumb bug along the way, so I figured I'd just write down how it went.

*(Image: agent_clears_stereo_madness.gif)*

If you haven't played Geometry Dash: you control a little square that runs to the right on its own, and you press one button to jump. Touch anything and you die instantly. That's basically the whole game. Easy to describe, not easy to do.

## Why I didn't just train in the game

Most projects I found do the same thing — they hook an RL agent straight into the running game and let it learn by playing. That works, but the game runs at 60 frames per second and won't go faster, and RL needs a *lot* of frames to learn anything (like millions). At 60 fps that's days of your computer just playing Geometry Dash over and over.

That felt like the wrong thing to be waiting on, so I wrote a small physics simulator instead — just the movement, no graphics or sound. Because it's so stripped down it runs a couple thousand times faster than the real game, so training takes seconds instead of days. The plan was to train in the fast fake version and then check whether it still works on the real one.

That last part — whether it survives on the real game — turned out to be the actual hard problem.

## Comparing a few algorithms first

Before touching the real game I used the sim to compare three approaches on the same levels: DQN, PPO, and a genetic algorithm. I assumed PPO would win because it's the one everyone reaches for. It didn't.

*(Image: comparison.png)*

On the harder level PPO kept getting stuck in the exact same spot every time. The reason is kind of interesting: you get free progress up to the first risky jump, and dying costs you points. So PPO learned "just take the free progress and don't risk the jump" and refused to try anything else. I threw 6 million training steps at it and it still wouldn't budge. Meanwhile DQN and the genetic algorithm both cleared every level on every random seed, and the genetic algorithm was actually the fastest of the three. That makes sense once you notice the levels are deterministic — same inputs always give the same run — so an approach that basically memorizes a good input sequence does really well here.

Not a groundbreaking result, but it was a good reminder that the "best" algorithm depends entirely on the problem, not on what's trendy.

## Getting it onto the real game

To connect my Python agents to the actual game I had to write a mod in C++ using Geode (the Geometry Dash modding framework). The mod reads the cube's position and velocity every frame and injects the jump input, and sends all of that back and forth to Python over a local socket.

The first time it connected and I watched a policy that had only ever seen my fake sim start driving the real cube and jumping real spikes — that was easily the best moment of the whole thing.

It also died at 11%.

*(Image: real_stereo_clip.gif)*

Which, honestly, is the whole point. My sim's physics are only an approximation of the real game's — close, but not exactly right. The agent had learned jump timings tuned to my slightly-wrong gravity, and over enough obstacles that small error adds up until it dies. This is the "sim-to-real gap" everyone talks about, and there's no clever trick that makes it go away. You just have to measure the difference and fix your sim.

## Trying to be clever about it

11% wasn't great, so I tried a different angle. Since the levels are deterministic, beating one is really just finding an input sequence that survives. And Geometry Dash has practice mode, where you can drop checkpoints and respawn at them instead of the start.

So I wrote a search that runs on the real game: from the last checkpoint, try some jump patterns, and when one survives a stretch, drop a checkpoint there and search the next stretch. Each attempt only replays a few seconds instead of the whole level. This worked pretty well — it slowly pushed forward and eventually got to about 35%.

Then it hit the ship section and stopped cold. In the ship parts you hold the button to fly instead of jumping, threading through spikes above and below. My "try different jump timings" search had no idea what to do with continuous flying, so it just got stuck.

## Just showing it how

At this point I stopped trying to be clever. If the hard part was the ship section and *I* can fly the ship section, why not just let it learn from me?

So I added a record mode to the mod. This one's passive — the game runs at full normal speed and the mod just watches my inputs each frame instead of taking over. (Getting that right took a couple tries; my first version paused the game to wait for my program every frame, which made it laggy and impossible to play.) I'd play the level myself, once, and it records exactly what I pressed on every frame. Then, because the game is deterministic, I can feed those inputs back and the agent reproduces my run.

Nice and simple. It kept dying at 62%.

## The dumb bug

This is the part I'd usually leave out, which is exactly why I'm including it.

The replay would go perfectly through the whole first half — cube, ship, everything — and then die at 62% every single time. I tried three different ways of feeding the inputs back in. I rewrote how the mod presses the button. I forced a fixed physics timestep so it couldn't drift. Every fix meant rebuilding the mod and relaunching the game, and it kept dying in the same place. I was pretty sure it was some deep timing problem in the game's physics and was about ready to give up and call it "reproduces 62%."

Then I actually read the logs and noticed the game was *crashing on shutdown*. My mod had a background thread I wasn't cleaning up properly, and it was leaving the game in a slightly broken state — so every test I ran afterward was starting from corrupted state. The 62% wasn't a physics limit. My own crash was quietly messing up the results.

I fixed the crash. And here's the embarrassing part: I still didn't think it would matter and almost didn't retry. Someone I was working through it with told me to just run it once more on a clean launch. It cleared the whole level, 100%. The fix had been right the entire time — the crash was the thing lying to me. I'd spent hours debugging the wrong layer.

*(Image / video: the full clear — agent_clears_stereo_madness.mp4 on YouTube, or the gif.)*

## Making it actually "learned"

I want to be clear about what that clear is, because it's easy to oversell. Replaying a recorded run is a systems thing — it proves the mod and the determinism work — but it's *not* an RL agent figuring out the level on its own. It's playback.

So I took the recording (game state + what I pressed, every frame) and trained a small network to predict "would a human jump here?" from the game state. That's behavior cloning, and it got to about 90.7% accuracy on frames it hadn't seen. It's a genuinely learned policy that imitates how I play. It also drifts if you let it drive the whole level by itself, which is the known weakness of behavior cloning on a single frame-perfect task — one wrong frame and the mistakes pile up. So the guaranteed clear still uses the replay; the learned policy is more of an "okay, it did learn something" companion. I'd rather be straight about that than pretend it solved the level on its own.

## Stuff I'd tell myself starting over

- Building the fast sim was the best decision by far. Almost none of the ML would've been practical if every experiment took days instead of seconds.
- Actually run the comparison instead of assuming. The PPO-getting-stuck thing is my favorite part and it only showed up because I ran it with proper seeds.
- When something is consistently wrong in a way that makes no sense, suspect your setup before your theory. I spent hours "fixing" physics when the real problem was a crash.
- Say what your result actually is. "Completed the level via learning from demonstration" is honest. "My RL agent beat Geometry Dash" is the kind of sentence that falls apart the moment someone asks a follow-up question.

The code, all the clips, the algorithm comparison, and a blunt list of what still doesn't work are on GitHub: **[github.com/abs768/geometry-dash-rl](https://github.com/abs768/geometry-dash-rl)**.

There's still a bunch left to do — the sim's physics for the non-cube modes are rough, and I never got the autonomous agent past the ship section. I did go back and calibrate the cube jump against the real trajectories from my recording, and it turned out my sim was undershooting the real jump height by about 8%, which is exactly the kind of thing that quietly wrecks sim-to-real transfer — so that's fixed now, though closing the transfer gap fully needs more than just the jump arc. But it does play the real game start to finish, and I learned more about debugging and being honest about results than I did about RL, which is probably the most useful thing I can say about it.

If you have questions or want to poke at the code, feel free.
