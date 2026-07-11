// Minimal blocking TCP server for the agent bridge. One client at a time.
// Framing: every message is u32 little-endian length prefix + payload.
#pragma once

#include <cstdint>
#include <vector>

namespace gdbridge {

class SocketServer {
public:
    explicit SocketServer(uint16_t port);
    ~SocketServer();

    // Open the listening socket. Returns false on failure (logged by caller).
    bool listen();

    // Non-blocking: accept a pending client if one is waiting. Safe to call
    // every frame; a no-op once connected.
    void pollAccept();

    bool connected() const { return m_client >= 0; }

    // Send a length-prefixed message. Drops the client on error.
    bool send(const void* data, uint32_t size);

    // Receive one length-prefixed message into `out`. Blocks up to timeout_ms
    // (0 = block forever). Returns false on timeout/error/disconnect.
    bool recv(std::vector<uint8_t>& out, int timeout_ms);

    void dropClient();

private:
    uint16_t m_port;
    int m_listen = -1;
    int m_client = -1;

    bool recvExact(void* buf, uint32_t n, int timeout_ms);
};

}  // namespace gdbridge
