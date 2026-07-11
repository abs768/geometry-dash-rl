"""Protocol-faithful mock of the Geode mod, backed by the headless sim.

This lets the entire Python bridge path (socket framing, binary protocol,
GDRealEnv) be exercised end-to-end without Geometry Dash installed. It speaks
exactly the wire protocol in protocol.py, so if a policy runs correctly against
this mock, the only untested surface left is the C++ mod's game-state reads.

Run standalone:  python -m gdrl.envs.mock_mod --level spikes_easy
Or use MockModServer(...) as a context manager in tests.
"""

from __future__ import annotations

import socket
import struct
import threading

from gdrl.envs import protocol as proto
from gdrl.sim import GDSim, Level
from gdrl.sim.level import BLOCK
from gdrl.envs.gd_env import LEVELS_DIR


def _level_geometry(level: Level) -> list[tuple[int, float, float]]:
    return [(0 if o.type == BLOCK else 1, o.x, o.y) for o in level.objects]


class MockModServer:
    """Serves one client at a time, mirroring the mod's request/response loop."""

    def __init__(self, level: str | Level, host: str = "127.0.0.1", port: int = 0):
        if not isinstance(level, Level):
            path = LEVELS_DIR / f"{level}.json"
            level = Level.from_file(path)
        self.level = level
        self.sim = GDSim(level)
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "MockModServer":
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- framing -----------------------------------------------------------

    @staticmethod
    def _send(conn: socket.socket, payload: bytes) -> None:
        conn.sendall(struct.pack("<I", len(payload)) + payload)

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            chunk = conn.recv(n - got)
            if not chunk:
                raise ConnectionError("client disconnected")
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def _recv_msg(self, conn: socket.socket) -> bytes:
        (size,) = struct.unpack("<I", self._recv_exact(conn, 4))
        return self._recv_exact(conn, size)

    # -- server loop -------------------------------------------------------

    def _state_payload(self, state, frame: int) -> proto.State:
        flags = 0
        if state.dead:
            flags |= proto.FLAG_DEAD
        if state.won:
            flags |= proto.FLAG_COMPLETE
        return proto.State(
            x=state.x, y=state.y, vy=state.vy, grounded=state.grounded,
            gamemode=0, flags=flags, length=self.level.length,
            frame=frame, percent=self.sim.progress(state), speed_mult=1.0,
        )

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # HELLO handshake.
            self._send(conn, struct.pack("<BH", proto.MSG_HELLO, proto.PROTOCOL_VERSION))

            state = self.sim.reset()
            frame = 0
            geometry = _level_geometry(self.level)

            while not self._stop.is_set():
                try:
                    msg = self._recv_msg(conn)
                except (ConnectionError, OSError):
                    return
                if not msg or msg[0] != proto.MSG_ACTION:
                    continue
                flags = proto.unpack_action(msg)

                if flags & proto.ACT_REQUEST_RESET:
                    state = self.sim.reset()
                    frame = 0
                else:
                    state = self.sim.step(state, hold=bool(flags & proto.ACT_HOLD))
                    frame += 1

                if flags & proto.ACT_REQUEST_GEOM:
                    self._send(conn, proto.pack_geometry(geometry))
                self._send(conn, self._state_payload(state, frame).pack())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="spikes_easy")
    parser.add_argument("--port", type=int, default=proto.DEFAULT_PORT)
    args = parser.parse_args()
    server = MockModServer(args.level, port=args.port)
    print(f"mock mod serving level {args.level!r} on 127.0.0.1:{server.port} — Ctrl-C to stop")
    server.start()
    try:
        server._thread.join()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
