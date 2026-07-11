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
    m_running = false;
    if (m_listen >= 0) ::shutdown(m_listen, SHUT_RDWR);
    if (m_acceptThread.joinable()) m_acceptThread.join();
    dropClient();
    if (m_listen >= 0) ::close(m_listen);
}

void SocketServer::setOnConnectMessage(std::vector<uint8_t> msg) {
    std::lock_guard<std::mutex> lock(m_ioMutex);
    m_helloMsg = std::move(msg);
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
    if (::listen(m_listen, 8) < 0) {  // deeper backlog so a stuck client can't brick it
        ::close(m_listen);
        m_listen = -1;
        return false;
    }
    m_running = true;
    m_acceptThread = std::thread(&SocketServer::acceptLoop, this);
    return true;
}

void SocketServer::acceptLoop() {
    while (m_running) {
        int c = ::accept(m_listen, nullptr, nullptr);  // blocking
        if (c < 0) {
            if (!m_running) return;
            continue;
        }
        int yes = 1;
        ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

        std::lock_guard<std::mutex> lock(m_ioMutex);
        // A new client replaces any previous one (latest agent wins).
        int old = m_client.exchange(c);
        if (old >= 0) ::close(old);
        if (!m_helloMsg.empty()) sendRaw(c, m_helloMsg.data(),
                                         static_cast<uint32_t>(m_helloMsg.size()));
    }
}

bool SocketServer::connected() {
    return m_client.load() >= 0;
}

void SocketServer::dropClient() {
    int c = m_client.exchange(-1);
    if (c >= 0) ::close(c);
}

bool SocketServer::sendRaw(int fd, const void* data, uint32_t size) {
    uint32_t prefix = size;  // little-endian on all supported targets
    if (::send(fd, &prefix, sizeof(prefix), 0) != (ssize_t)sizeof(prefix)) return false;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint32_t sent = 0;
    while (sent < size) {
        ssize_t n = ::send(fd, p + sent, size - sent, 0);
        if (n <= 0) return false;
        sent += static_cast<uint32_t>(n);
    }
    return true;
}

bool SocketServer::send(const void* data, uint32_t size) {
    std::lock_guard<std::mutex> lock(m_ioMutex);
    int fd = m_client.load();
    if (fd < 0) return false;
    if (!sendRaw(fd, data, size)) {
        dropClient();
        return false;
    }
    return true;
}

bool SocketServer::recvExact(int fd, void* buf, uint32_t n, int timeout_ms) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    uint32_t got = 0;
    while (got < n) {
        if (timeout_ms > 0) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd, &fds);
            timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
            int r = ::select(fd + 1, &fds, nullptr, nullptr, &tv);
            if (r <= 0) return false;  // timeout or error
        }
        ssize_t r = ::recv(fd, p + got, n - got, 0);
        if (r <= 0) return false;
        got += static_cast<uint32_t>(r);
    }
    return true;
}

bool SocketServer::recv(std::vector<uint8_t>& out, int timeout_ms) {
    std::lock_guard<std::mutex> lock(m_ioMutex);
    int fd = m_client.load();
    if (fd < 0) return false;
    uint32_t size = 0;
    if (!recvExact(fd, &size, sizeof(size), timeout_ms)) return false;
    out.resize(size);
    if (size != 0 && !recvExact(fd, out.data(), size, timeout_ms)) {
        dropClient();
        return false;
    }
    return true;
}

}  // namespace gdbridge
