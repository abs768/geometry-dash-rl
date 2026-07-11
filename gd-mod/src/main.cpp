// Geometry Dash RL bridge mod.
//
// Exposes the running game to the Python agent over a local socket using the
// protocol in protocol.hpp / PROTOCOL.md. The mod DRIVES the loop: each frame
// it exports state, blocks briefly for the agent's action, applies it, and
// steps the game. This keeps the agent frame-locked to the game so evaluation
// is deterministic and can run faster than real time under a speedhack.
//
// Binding names marked `VERIFY` are GD-version / Geode-bindings sensitive and
// must be checked against the installed headers before the first build.

#include <memory>
#include <vector>

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include "protocol.hpp"
#include "socket_server.hpp"

using namespace geode::prelude;
using namespace gdbridge;

namespace {

// Forward declarations (definitions live lower in this file).
bool isSpikeId(int id);
bool isSolidId(int id);

// One bridge for the whole process; hooks talk to it.
class Bridge {
public:
    static Bridge& get() {
        static Bridge instance;
        return instance;
    }

    void ensureListening() {
        if (m_started) return;
        m_started = true;
        m_server = std::make_unique<SocketServer>(DEFAULT_PORT);
        if (m_server->listen()) {
            log::info("gd-bridge listening on 127.0.0.1:{}", DEFAULT_PORT);
        } else {
            log::error("gd-bridge failed to bind port {}", DEFAULT_PORT);
        }
    }

    // Send the Hello handshake to a freshly connected client.
    void sendHello() {
        uint8_t buf[3] = {static_cast<uint8_t>(MsgType::Hello),
                          static_cast<uint8_t>(PROTOCOL_VERSION & 0xff),
                          static_cast<uint8_t>((PROTOCOL_VERSION >> 8) & 0xff)};
        m_server->send(buf, sizeof(buf));
    }

    SocketServer* server() { return m_server.get(); }

private:
    bool m_started = false;
    std::unique_ptr<SocketServer> m_server;
};

// Read the current cube state out of PlayLayer into a wire payload.
StatePayload readState(PlayLayer* pl) {
    StatePayload s{};
    s.type = static_cast<uint8_t>(MsgType::State);

    PlayerObject* p = pl->m_player1;  // VERIFY: field name m_player1
    const CCPoint pos = p->getPosition();
    s.x = pos.x / UNITS_PER_BLOCK;
    s.y = pos.y / UNITS_PER_BLOCK;
    s.vy = static_cast<float>(p->m_yVelocity) / UNITS_PER_BLOCK;  // VERIFY: m_yVelocity units
    s.grounded = p->m_isOnGround ? 1 : 0;                          // VERIFY: m_isOnGround

    // Gamemode. v0 only trains cube, but we report it so Python can gate.
    GameMode mode = GameMode::Cube;
    if (p->m_isShip) mode = GameMode::Ship;                        // VERIFY: m_isShip et al.
    else if (p->m_isBall) mode = GameMode::Ball;
    else if (p->m_isBird) mode = GameMode::Ufo;
    else if (p->m_isDart) mode = GameMode::Wave;
    else if (p->m_isRobot) mode = GameMode::Robot;
    else if (p->m_isSpider) mode = GameMode::Spider;
    s.gamemode = static_cast<uint8_t>(mode);

    uint8_t flags = 0;
    if (p->m_isDead) flags |= FLAG_DEAD;                          // VERIFY: m_isDead
    if (pl->m_hasCompletedLevel) flags |= FLAG_COMPLETE;         // VERIFY: m_hasCompletedLevel
    if (p->m_isUpsideDown) flags |= FLAG_UPSIDE;                 // VERIFY: m_isUpsideDown
    s.flags = flags;

    // Level length in blocks. m_levelLength is in units. VERIFY.
    s.length = pl->m_levelLength / UNITS_PER_BLOCK;
    s.frame = Bridge::get().server() ? 0 : 0;  // frame counter maintained by hook below
    s.percent = static_cast<float>(pl->getCurrentPercent()) / 100.0f;  // VERIFY: getCurrentPercent
    s.speed_mult = 1.0f;  // TODO: derive from active speed portal
    s.reserved = 0;
    return s;
}

// Dump the level's obstacle list so Python can build the same look-ahead grid.
std::vector<GeometryRecord> readGeometry(PlayLayer* pl) {
    std::vector<GeometryRecord> out;
    CCArray* objects = pl->m_objects;  // VERIFY: field name m_objects
    if (!objects) return out;
    for (unsigned i = 0; i < objects->count(); ++i) {
        auto* obj = static_cast<GameObject*>(objects->objectAtIndex(i));
        if (!obj) continue;
        // Classify by object type id. GD spike IDs are a known set (8, 39, 103,
        // ...); everything solid-and-collidable is treated as a block. This
        // mapping is the main thing to expand for full level coverage. VERIFY.
        int id = obj->m_objectID;  // VERIFY: m_objectID
        uint8_t kind;
        if (isSpikeId(id)) kind = 1;
        else if (isSolidId(id)) kind = 0;
        else continue;  // decoration / non-colliding: skip
        const CCPoint p = obj->getPosition();
        out.push_back({kind, p.x / UNITS_PER_BLOCK, p.y / UNITS_PER_BLOCK});
    }
    return out;
}

// Placeholder classifiers — replace with the real GD object-ID tables.
bool isSpikeId(int id) {
    switch (id) { case 8: case 39: case 103: case 392: return true; default: return false; }
}
bool isSolidId(int /*id*/) { return true; }  // TODO: restrict to collidable blocks

}  // namespace

// ---- hooks ----------------------------------------------------------------

class $modify(BridgePlayLayer, PlayLayer) {
    struct Fields {
        uint32_t frame = 0;
        bool holding = false;
    };

    // Called when a level starts. Open the socket and reset per-attempt state.
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
        Bridge::get().ensureListening();
        m_fields->frame = 0;
        return true;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        m_fields->frame = 0;  // frame counter resets on each attempt (per protocol)
    }

    // Send the current level geometry as one GEOMETRY message.
    void sendGeometry(SocketServer* srv) {
        auto geom = readGeometry(this);
        std::vector<uint8_t> buf;
        buf.push_back(static_cast<uint8_t>(MsgType::Geometry));
        uint32_t count = static_cast<uint32_t>(geom.size());
        buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&count),
                   reinterpret_cast<uint8_t*>(&count) + sizeof(count));
        buf.insert(buf.end(), reinterpret_cast<uint8_t*>(geom.data()),
                   reinterpret_cast<uint8_t*>(geom.data()) + geom.size() * sizeof(GeometryRecord));
        srv->send(buf.data(), static_cast<uint32_t>(buf.size()));
    }

    // Per-frame driver, request/response with the agent (see PROTOCOL.md):
    // block for one ACTION, apply it, run the frame, reply with [GEOMETRY]+STATE.
    // Blocking here is the frame-lock: the game waits for the agent each frame.
    void update(float dt) {
        Bridge& bridge = Bridge::get();
        SocketServer* srv = bridge.server();
        if (!srv) { PlayLayer::update(dt); return; }

        srv->pollAccept();
        if (!srv->connected()) { PlayLayer::update(dt); return; }
        if (m_fields->frame == 0) bridge.sendHello();

        // Block for the agent's action (short timeout so a dead agent can't
        // hang the game forever; on timeout, repeat the previous input).
        std::vector<uint8_t> msg;
        bool resetting = false, wants_geom = false;
        if (srv->recv(msg, /*timeout_ms=*/1000) && msg.size() >= 2 &&
            msg[0] == static_cast<uint8_t>(MsgType::Action)) {
            uint8_t act = msg[1];
            m_fields->holding = act & ACT_HOLD;
            resetting = act & ACT_REQUEST_RESET;
            wants_geom = act & ACT_REQUEST_GEOM;
        }

        if (resetting) {
            this->resetLevel();  // repositions to spawn; reply reflects spawn state
        } else {
            // Apply input before the frame's physics, then step.
            if (m_fields->holding) this->m_player1->pushButton(PlayerButton::Jump);   // VERIFY
            else this->m_player1->releaseButton(PlayerButton::Jump);                  // VERIFY
            PlayLayer::update(dt);
        }

        if (wants_geom) sendGeometry(srv);

        StatePayload s = readState(this);
        s.frame = m_fields->frame++;
        srv->send(&s, sizeof(s));
    }
};
