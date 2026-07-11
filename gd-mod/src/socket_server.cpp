#include "socket_server.hpp"

#include <cstring>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

namespace gdbridge {

SocketServer::SocketServer(uint16_t port) : m_port(port) {}

SocketServer::~SocketServer() {
    dropClient();
    if (m_listen >= 0) ::close(m_listen);
}

bool SocketServer::listen() {
    m_listen = ::socket(AF_INET, SOCK_STREAM, 0);
    if (m_listen < 0) return false;

    int yes = 1;
    ::setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(m_port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // 127.0.0.1 only
    if (::bind(m_listen, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(m_listen);
        m_listen = -1;
        return false;
    }
    if (::listen(m_listen, 1) < 0) {
        ::close(m_listen);
        m_listen = -1;
        return false;
    }
    // Non-blocking accept so pollAccept() never stalls the game loop.
    ::fcntl(m_listen, F_SETFL, O_NONBLOCK);
    return true;
}

void SocketServer::pollAccept() {
    if (m_listen < 0 || m_client >= 0) return;
    int c = ::accept(m_listen, nullptr, nullptr);
    if (c >= 0) {
        int yes = 1;
        ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));  // low latency
        m_client = c;
    }
}

void SocketServer::dropClient() {
    if (m_client >= 0) {
        ::close(m_client);
        m_client = -1;
    }
}

bool SocketServer::send(const void* data, uint32_t size) {
    if (m_client < 0) return false;
    uint32_t prefix = size;  // little-endian on all supported targets (x86_64/arm64)
    if (::send(m_client, &prefix, sizeof(prefix), 0) != (ssize_t)sizeof(prefix)) {
        dropClient();
        return false;
    }
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint32_t sent = 0;
    while (sent < size) {
        ssize_t n = ::send(m_client, p + sent, size - sent, 0);
        if (n <= 0) {
            dropClient();
            return false;
        }
        sent += static_cast<uint32_t>(n);
    }
    return true;
}

bool SocketServer::recvExact(void* buf, uint32_t n, int timeout_ms) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    uint32_t got = 0;
    while (got < n) {
        if (timeout_ms > 0) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(m_client, &fds);
            timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
            int r = ::select(m_client + 1, &fds, nullptr, nullptr, &tv);
            if (r <= 0) return false;  // timeout or error
        }
        ssize_t r = ::recv(m_client, p + got, n - got, 0);
        if (r <= 0) {
            dropClient();
            return false;
        }
        got += static_cast<uint32_t>(r);
    }
    return true;
}

bool SocketServer::recv(std::vector<uint8_t>& out, int timeout_ms) {
    if (m_client < 0) return false;
    uint32_t size = 0;
    if (!recvExact(&size, sizeof(size), timeout_ms)) return false;
    out.resize(size);
    return size == 0 ? true : recvExact(out.data(), size, timeout_ms);
}

}  // namespace gdbridge
