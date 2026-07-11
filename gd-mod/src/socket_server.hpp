// Minimal TCP server for the agent bridge. One client at a time.
// Framing: every message is u32 little-endian length prefix + payload.
//
// Accepting runs on a dedicated background thread so connections and the
// handshake work regardless of game state (menu or in a level). The game's
// update loop only does the per-frame state/action exchange. This decouples
// "is a client connected" from "is the game running a level", which is what
// makes connecting reliable.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace gdbridge {

class SocketServer {
public:
    explicit SocketServer(uint16_t port);
    ~SocketServer();

    // Open the listening socket and start the background accept thread.
    bool listen();

    // Bytes sent to every client immediately on accept (the HELLO handshake).
    void setOnConnectMessage(std::vector<uint8_t> msg);

    bool connected();

    // Send a length-prefixed message. Drops the client on error.
    bool send(const void* data, uint32_t size);

    // Receive one length-prefixed message into `out`. Blocks up to timeout_ms
    // (0 = block forever). Returns false on timeout/error/disconnect.
    bool recv(std::vector<uint8_t>& out, int timeout_ms);

    void dropClient();

private:
    uint16_t m_port;
    int m_listen = -1;
    std::atomic<int> m_client{-1};
    std::atomic<bool> m_running{false};
    std::mutex m_ioMutex;              // serializes send/recv/replace on m_client
    std::vector<uint8_t> m_helloMsg;
    std::thread m_acceptThread;

    void acceptLoop();
    bool sendRaw(int fd, const void* data, uint32_t size);
    bool recvExact(int fd, void* buf, uint32_t n, int timeout_ms);
};

}  // namespace gdbridge
