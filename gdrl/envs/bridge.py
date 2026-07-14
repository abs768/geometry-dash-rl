"""Bridge to the official game via the Geode mod (gd-mod/).

RealGameBridge is the socket client that speaks the binary protocol in
protocol.py. GDRealEnv wraps it in the Gymnasium API, producing observations
byte-identical to the sim env (via gdrl.envs.observation) so a policy trained
in the sim drives the real game with no changes.

The mod drives the loop: it sends STATE and blocks for our ACTION each frame,
so from Python the interaction is a simple recv-state / send-action exchange.
"""

from __future__ import annotations

import socket
import struct

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gdrl.envs import protocol as proto
from gdrl.envs.observation import OBS_LEN, build_observation
from gdrl.sim.level import BLOCK, Level, LevelObject, SLOPE_DOWN, SLOPE_UP, SPIKE

# GeometryRecord kind byte -> sim object type.
_KIND_TO_TYPE = {0: BLOCK, 1: SPIKE, 2: SLOPE_UP, 3: SLOPE_DOWN}


class RealGameBridge:
    """Length-prefixed socket client for the mod. Not thread-safe."""

    def __init__(self, host: str = "127.0.0.1", port: int = proto.DEFAULT_PORT, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        # Most recent geometry dump, cached whenever a GEOMETRY message arrives.
        self.last_geometry: list[tuple[int, float, float]] | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # First message must be HELLO with a matching protocol version.
        payload = self._recv_msg()
        if payload[0] != proto.MSG_HELLO:
            raise ConnectionError(f"expected HELLO, got {payload[0]:#x}")
        (version,) = struct.unpack("<H", payload[1:3])
        if version != proto.PROTOCOL_VERSION:
            raise ConnectionError(
                f"protocol mismatch: mod v{version}, client v{proto.PROTOCOL_VERSION}")

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    # -- framing -----------------------------------------------------------

    def _recv_exact(self, n: int) -> bytes:
        chunks = []
        got = 0
        while got < n:
            chunk = self.sock.recv(n - got)
            if not chunk:
                raise ConnectionError("mod closed the connection")
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def _recv_msg(self) -> bytes:
        (size,) = struct.unpack("<I", self._recv_exact(4))
        return self._recv_exact(size)

    def _send_msg(self, payload: bytes) -> None:
        self.sock.sendall(struct.pack("<I", len(payload)) + payload)

    # -- protocol ----------------------------------------------------------

    def recv_state(self) -> proto.State:
        """Receive the response to an action: a STATE, optionally preceded by a
        GEOMETRY message (cached in last_geometry) when geometry was requested."""
        while True:
            payload = self._recv_msg()
            tag = payload[0]
            if tag == proto.MSG_STATE:
                return proto.State.unpack(payload)
            if tag == proto.MSG_GEOMETRY:
                self.last_geometry = proto.unpack_geometry(payload)
                continue
            raise ValueError(f"unexpected message {tag:#x} while awaiting STATE")

    def send_action(self, hold: bool, request_reset: bool = False,
                    request_geom: bool = False) -> None:
        self._send_msg(proto.pack_action(hold, request_reset, request_geom))


def geometry_to_level(records: list[tuple[int, float, float]], length: float) -> Level:
    """Build a Level (for the shared observation grid) from a geometry dump."""
    objs = [LevelObject(_KIND_TO_TYPE.get(kind, BLOCK), x, y) for kind, x, y in records]
    return Level("real", length, objs)


class GDRealEnv(gym.Env):
    """Gymnasium env backed by the real game over the bridge.

    Observations match GDEnv exactly. Reward mirrors the sim env so returns are
    comparable, but this env is intended for EVALUATION of sim-trained policies,
    not training (the game runs near real time even under a speedhack).
    """

    metadata = {"render_modes": []}

    def __init__(self, bridge: RealGameBridge, max_steps: int = 10_000):
        self.bridge = bridge
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_LEN,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.level: Level | None = None
        self._steps = 0
        self._last_x = 0.0

    def _obs(self, s: proto.State) -> np.ndarray:
        gravity = -1 if (s.flags & proto.FLAG_UPSIDE) else 1
        return build_observation(s.x, s.y, s.vy, s.grounded, self.level,
                                 mode=s.gamemode, gravity=gravity)

    @staticmethod
    def _info(s: proto.State) -> dict:
        return {"x": s.x, "progress": s.percent, "won": s.complete, "dead": s.dead}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # One action requests both a reset and the level geometry; the response
        # is a GEOMETRY message (cached) followed by the spawn STATE.
        self.bridge.send_action(hold=False, request_reset=True, request_geom=True)
        s = self.bridge.recv_state()
        if self.bridge.last_geometry is None:
            raise ConnectionError("mod did not send geometry after reset request")
        self.level = geometry_to_level(self.bridge.last_geometry, s.length)
        self._steps = 0
        self._last_x = s.x
        return self._obs(s), self._info(s)

    def step(self, action):
        self.bridge.send_action(hold=bool(action))
        s = self.bridge.recv_state()

        reward = s.x - self._last_x
        self._last_x = s.x
        terminated = False
        if s.dead:
            reward += -10.0
            terminated = True
        elif s.complete:
            reward += 10.0
            terminated = True
        self._steps += 1
        truncated = not terminated and self._steps >= self.max_steps
        return self._obs(s), reward, terminated, truncated, self._info(s)

    def close(self):
        self.bridge.close()
