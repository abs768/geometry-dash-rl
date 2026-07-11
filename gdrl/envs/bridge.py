"""Bridge to the official game via the Geode mod — NOT IMPLEMENTED YET.

This module will expose the same interface as GDSim/GDEnv but backed by the
real game: the mod streams structured state each frame and accepts hold/release
actions plus reset commands. Protocol spec lives in gd-mod/PROTOCOL.md.

Design constraints (learned from prior art's pain points):
  * length-prefixed binary messages over a local socket, not ad-hoc text;
  * the mod drives the loop (it sends state, waits for the action) so the
    game and the agent stay frame-locked;
  * reset must be instant (practice-mode restart), not menu navigation.
"""


class RealGameBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 51717):
        raise NotImplementedError(
            "Real-game bridge pending Geode mod implementation; see gd-mod/PROTOCOL.md"
        )
