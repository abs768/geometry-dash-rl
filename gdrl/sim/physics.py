"""Cube-mode physics constants, in block units (1 block = 30 GD units).

These are community-derived approximations chosen to reproduce the two
observable invariants of the cube at 1x speed: a jump apex of ~2.1 blocks
reached in ~0.21 s, and horizontal travel of ~10.38 blocks/s. They are the
calibration surface for sim-to-real transfer — once the Geode mod can log
real trajectories, fit these constants to match before trusting transfer.
"""

# Simulation tick. GD's engine runs physics at 240 Hz internally; 60 Hz is
# enough for cube mode and keeps episodes short for RL.
DT = 1.0 / 60.0

# Horizontal speed at 1x ("normal") speed, blocks/s.
SPEED_1X = 10.3761

# Vertical physics, blocks/s and blocks/s^2.
JUMP_VELOCITY = 20.0
GRAVITY = 94.0
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
