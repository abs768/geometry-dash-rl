"""Cube-mode physics constants, in block units (1 block = 30 GD units).

Calibrated against real trajectories logged from the game. 18 flat-ground cube
jumps from a recorded playthrough give a jump apex of 2.128 blocks over an
airtime of 25 frames (0.417 s) and a horizontal speed of 10.39 blocks/s.
JUMP_VELOCITY and GRAVITY are then fit *to the sim's discrete integration* (not
the textbook v^2/2g, which the semi-implicit Euler step undershoots) so the sim
jump reproduces the real apex and airtime exactly.

Note: the previous community values (20.0 / 94.0) actually undershot the real
apex by ~8% in the discrete sim (2.128 -> 1.96 blocks) — a real miscalibration
that would mistime sim-to-real transfer.
"""

# Simulation tick. GD's engine runs physics at 240 Hz internally; 60 Hz is
# enough for cube mode and keeps episodes short for RL.
DT = 1.0 / 60.0

# Horizontal speed at 1x ("normal") speed, blocks/s (measured: 10.389).
SPEED_1X = 10.389

# Vertical physics, blocks/s and blocks/s^2. Fit so the discrete sim jump hits
# apex 2.128 blocks in 25 frames, matching the logged real cube jump.
JUMP_VELOCITY = 21.80
GRAVITY = 103.0
TERMINAL_VELOCITY = -26.0

# Player hitbox is one block. Death-on-side-collision uses the full box;
# spike hitboxes are much smaller than their visual triangle (see level.py).
PLAYER_SIZE = 1.0

# Tolerance when deciding "landed on top of a block" vs "hit its side":
# if the player's previous bottom edge was at least this close to above the
# block's top, the contact counts as a landing.
LANDING_TOLERANCE = 0.20

# --- other gamemodes --------------------------------------------------------
# These constants are approximate: they give controllable, plausible motion in
# block units, not frame-perfect GD fidelity. Calibrate against real-game
# trajectories (Geode mod) before trusting sim-to-real for these modes.

# Ship: gentle gravity, stronger thrust; velocity capped so flight is smooth.
SHIP_GRAVITY = 34.0
SHIP_THRUST = 68.0
SHIP_MAX_VY = 12.0

# UFO: each tap is a fixed upward impulse against a cube-like gravity.
UFO_GRAVITY = 60.0
UFO_IMPULSE = 11.0

# Ball: rolls on a surface, each tap flips gravity. Uses a cube-like gravity.
BALL_GRAVITY = 85.0

# Wave moves at 45 degrees, so its |vertical speed| equals the horizontal speed
# (SPEED_1X * speed_multiplier); no separate constant needed.

# Robot: variable-height jump. A press gives an initial pop; holding adds thrust
# for up to ROBOT_BOOST_FRAMES frames, so hold duration controls jump height.
ROBOT_JUMP_VELOCITY = 15.0
ROBOT_BOOST_ACCEL = 130.0   # while holding during the boost window (net up vs GRAVITY)
ROBOT_BOOST_FRAMES = 12

# Spider: each tap teleports instantly to the opposite surface and flips gravity
# (no travel time). Between taps it clings to its surface, falling only if it
# runs off an edge, under a cube-like gravity.
SPIDER_GRAVITY = 94.0
