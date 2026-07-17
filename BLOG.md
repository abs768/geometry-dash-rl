# I spent a month teaching an AI to play the real Geometry Dash

I don't fully remember why I decided to do this. I think I'd seen a couple of projects where people trained an AI to play Geometry Dash, and every single one of them was playing a Python remake of the game — a little clone someone wrote to be easy to plug an agent into. Which is fine, but it always felt like a bit of a cop-out. The interesting version, to me, was the one where the AI plays the *actual* game. The one on Steam. The one you and I would open and rage-quit.

So that's what I set out to do. Here's roughly how it went, dead ends included, because the dead ends are where I actually learned anything.

If you've never played it: you control a little square that runs to the right on its own, and you have exactly one button — jump. Touch a spike or a wall and you die instantly, back to the start. That's the whole game. It's easy to explain and miserable to be good at, which turns out to be a great combination for this kind of project.

## The first decision was the most important one

My initial instinct was the obvious one: hook an agent into the running game and let it learn by playing, over and over, until it gets good. This is how most of the projects I'd seen did it.

The problem is that reinforcement learning is *stupendously* data-hungry. Agents need to see millions of frames before they learn anything useful, and the game runs at 60 frames per second and won't go any faster. Do the math and you're looking at your computer playing Geometry Dash for days, in real time, just to run a single experiment. And I knew I'd want to run a lot of experiments.

So before I trained anything, I sat down and wrote my own physics simulator. Not the whole game — no graphics, no music, no menus. Just the part that matters: a square, gravity, a jump, and some spikes to die on. Because it's so stripped down, it runs *thousands* of times faster than the real game. An experiment that would've taken a day now took a few seconds.

That one decision — build the fast fake version first — quietly made the entire project possible. The plan from there was simple to say and hard to do: train the agent in the fast fake world, then see if it still works in the real one.

That last part, "does it still work in the real one," turned out to be the whole ballgame.

## I was sure PPO would win. It didn't.

Before touching the real game at all, I used my simulator to do something I'd wanted to do for a while: actually compare a few different learning algorithms on the same problem, properly, with multiple random seeds so I wasn't fooling myself.

I tried three. DQN, which learns the value of actions. PPO, which is the algorithm everyone reaches for these days and the one I quietly assumed would come out on top. And a genetic algorithm, which is almost embarrassingly simple — it basically breeds a population of button-mashing sequences and keeps the ones that die the latest.

PPO lost. Badly, and in a really interesting way.

On the harder test level, PPO would get to the exact same spot every time and then just... stop trying. It'd sit there collecting the easy progress up to the first genuinely risky jump, and then refuse to attempt the jump. And once I thought about the reward it was optimizing, it made total sense: it got points for moving forward, and it lost points for dying. So the safest strategy — the one that reliably scores okay — is "walk up to the scary part and stand there." Dying to try the jump was, from PPO's point of view, a bad trade. I threw six million training steps at it. It did not care.

Meanwhile the genetic algorithm and DQN cleared every level on every seed, and the genetic algorithm was actually the fastest of the three to get there. That flips the usual intuition, but it makes sense here: these levels are completely deterministic. The same inputs always produce the same run. So an approach that essentially memorizes a good sequence of button presses is playing to the game's biggest weakness.

It's not a groundbreaking result. But it was a good, concrete reminder that the "best" algorithm is entirely a function of your problem, not of what's fashionable — and I only saw it because I bothered to run the comparison with proper seeds instead of assuming.

## First contact with the real game

Now the fun part. To connect my Python agents to the actual game, I had to write a mod in C++ using Geode, the Geometry Dash modding framework. The mod's job is to read the cube's position and speed on every single frame, hand that to my Python code, and then take the jump decision my code sends back and press the button. All of it flying back and forth over a local socket.

Getting that pipe working was its own saga — I'll spare you most of it — but I vividly remember the first time it connected. A policy that had *only ever seen my fake simulator*, that had never touched the real game once, suddenly started driving the real cube. Jumping real spikes. In the real game. That was easily the best moment of the whole project.

It died at 11%.

And honestly, that's the entire point of the exercise, so I wasn't even disappointed. My simulator's physics are an approximation of the real game's — close, but not exactly right. My gravity was slightly off, my jump was slightly off, and the agent had learned its timing against *my* slightly-wrong physics. Over a handful of obstacles that tiny error is invisible. Over enough of them, it compounds until the cube is a few pixels off and clips a spike. This is the famous "sim-to-real gap," and there's no clever trick that makes it disappear. You just have to measure where your fake world and the real world disagree, and close it.

## Trying to be clever instead

11% was not a great look, so I went hunting for a shortcut. Here was my reasoning: the levels are deterministic, so "beating" one is really just the problem of *finding* a sequence of inputs that survives to the end. And Geometry Dash has a practice mode where you can drop checkpoints and respawn at them instead of at the start.

So I wrote a search that runs directly on the real game. From the most recent checkpoint, it tries a bunch of jump timings; whenever one survives a stretch of the level, it drops a fresh checkpoint there and starts searching the next stretch. Each attempt only has to replay a few seconds instead of the whole level, so it can grind through a surprising amount.

It worked. It slowly, stubbornly clawed its way forward and got to about 35% — the entire cube section of the level, cleared autonomously by search.

Then it hit the ship.

In the ship sections of Geometry Dash you don't jump — you *hold* the button to fly upward and release to fall, threading a little rocket through gaps with spikes above and below. My whole search was built around the idea of "try different discrete jump timings," and that concept is just meaningless for continuous flight. The search had no idea what to do and got stuck cold.

## Fine — I'll just show it how

At this point I stopped trying to outsmart the problem. If the hard part was the ship section, and *I* can personally fly the ship section, why was I making this so complicated? Let the thing learn from me.

So I added a record mode to the mod. This one is passive: the game runs at completely normal speed and the mod just quietly watches which frames I press the button on, without ever pausing or taking control. (My first version of this got it backwards — it froze the game every frame waiting for my program to respond, which made it lag so badly the level was unplayable. Took a couple of tries to get right.) The idea was that I'd play the level myself, once, cleanly, and it would record exactly what I pressed on every frame. Then, because the game is deterministic, I could feed those inputs back and watch the agent reproduce my run perfectly.

Simple, clean, foolproof.

It kept dying at 62%.

## The dumb bug (this is the good part)

This is the section I'd normally quietly leave out of a write-up, which is exactly why I'm putting it in.

The replay would sail through the entire first half of the level — cube, ship, the works — and then die at 62%. Every single time. Same spot. So I did what you do: I assumed my input-replaying code was subtly wrong. I rewrote the way the mod presses the button. I tried three different ways of feeding the inputs back in. I forced the physics to run on a fixed timestep so nothing could drift out of sync. Every attempt meant rebuilding the mod and relaunching the game, and every attempt died at 62%. I became convinced it was some deep, unfixable timing quirk in the game's physics engine, and I was about ready to write it up as "reproduces 62%, close enough" and move on.

Then I actually read the logs. And I noticed the game was *crashing on shutdown*.

I had a background thread in my mod that I wasn't cleaning up properly. When the game closed, that thread was tearing down messily and leaving things in a slightly corrupted state — which meant every test I ran *after* that point was starting from a subtly broken baseline. The 62% wall wasn't a physics limit at all. My own crash had been quietly poisoning every experiment for hours, and I'd spent that entire time debugging the wrong layer of the stack.

I fixed the crash. And here's the genuinely embarrassing part: I *still* didn't believe it would matter. I almost didn't bother re-testing. Someone I was talking through it with basically had to tell me to stop theorizing and just run it one more time on a clean launch.

It cleared the whole level. 100%.

The fix had been correct the entire time. The crash was the thing lying to me, and I'd spent hours arguing with the lie instead of reading the logs that were sitting right there. I think about that one a lot.

## Making sure it actually "learned" something

I want to be really clear about what that 100% clear is, because it's the kind of result that's easy to oversell and I'd rather not.

Replaying a recorded human run is a systems achievement. It proves the mod works, it proves the determinism holds, it proves the whole pipeline is solid. But it is emphatically *not* an AI figuring out the level on its own. It's playback. Calling it "my AI beat Geometry Dash" would be the kind of sentence that falls apart the second anyone asks a follow-up question.

So I did one more thing. I took that same recording — the game state plus what I pressed on every frame — and trained a small neural network to answer one question: "given what the screen looks like right now, would a human jump here?" That's behavior cloning, and it got to about 90.7% accuracy on frames it had never seen. It's a genuinely learned policy that imitates the way I play.

It also drifts if you let it fly solo for the whole level, which is the well-known weakness of behavior cloning on a single frame-perfect task — one slightly-wrong frame nudges you into a state you never saw in training, and the mistakes snowball. So the *guaranteed* clear still uses the deterministic replay, and the learned policy is more of an "okay, it did genuinely learn the mapping" companion piece. I'd much rather explain that honestly than pretend it solved something it didn't.

## Going back to fix the lie in my simulator

There was one loose thread that bugged me. Remember the sim-to-real gap — the reason the sim-trained agent died at 11%? I'd waved my hands at "the physics are a bit off," but I'd never actually gone and measured *how* off.

So I did. I had a perfect recording of a real human run, which meant I had a perfect record of the real cube's actual trajectory — how high it jumps, how long it stays in the air. I pulled out 18 clean flat-ground jumps and measured them: the real cube peaks at about 2.13 blocks, and stays airborne for 25 frames.

Then I checked what *my* simulator did with the same jump. It peaked at 1.96 blocks.

That's an 8% shortfall in jump height, sitting quietly in my physics the whole time. Not enough to notice on any single jump, but exactly the kind of small, consistent error that compounds over a level until your agent is mistiming everything. I'd been blaming a vague "gap"; the gap had a number, and now I could see it.

The fix was fiddly in a way I found kind of satisfying. The obvious move is to plug the real measurements into the textbook projectile equations and solve for gravity and jump strength — but when I did that, the sim was *still* wrong, because a discrete step-by-step simulation doesn't behave exactly like the smooth continuous math. So instead I fit the constants against my actual simulation loop until its jump landed precisely on the real 2.13 blocks over 25 frames. Now the sim jumps like the real game does, at least for the cube. Closing the *full* transfer gap needs more than just the jump arc, but it's a real, measured improvement instead of a shrug — and finding that hidden 8% felt like the most honest bit of debugging in the whole project.

## What I'd tell myself if I started over

A few things stuck with me more than the ML did.

**Building the fast simulator first was the best decision by a mile.** Almost none of this would have been practical if every experiment took a day. If there's a version of your problem that runs a thousand times faster, build that version first, even if it's fake.

**Actually run the comparison instead of assuming.** The PPO-getting-stuck result is my favorite thing that came out of this, and it only showed up because I ran the experiment properly with multiple seeds instead of trusting my gut about which algorithm "should" win.

**When something is consistently wrong in a way that makes no sense, suspect your setup before your theory.** I spent hours "fixing" physics when the actual problem was a crash corrupting my tests. Read the logs first. The logs were right there.

**Say what your result actually is.** "Completed the level via learning from demonstration" is a true, defensible sentence. "My RL agent conquered Geometry Dash" is a sentence that dissolves the moment someone smart asks you a question about it. The honest version is less impressive for about five seconds and more impressive forever after.

Does it play the real Geometry Dash, start to finish? Yes. Is it a triumphant autonomous AI that taught itself to master the game? No, and I'll be the first to tell you exactly where it stops. Somewhere in the gap between those two sentences is a project I learned a genuinely stupid amount from — mostly about debugging, honesty, and the specific feeling of arguing for three hours with a bug that was your own crash the whole time.

The code, all the clips, the algorithm comparison, and a blunt list of what still doesn't work are on GitHub: **[github.com/abs768/geometry-dash-rl](https://github.com/abs768/geometry-dash-rl)**. If you want to poke at it or tell me what I did wrong, please do.
