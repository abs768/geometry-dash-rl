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
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include "protocol.hpp"
#include "socket_server.hpp"

using namespace geode::prelude;
using namespace gdbridge;

namespace {

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
        // The server sends this HELLO to every client the moment it connects,
        // from its background accept thread — so the handshake works whether or
        // not the game is currently running a level.
        m_server->setOnConnectMessage({static_cast<uint8_t>(MsgType::Hello),
                                       static_cast<uint8_t>(PROTOCOL_VERSION & 0xff),
                                       static_cast<uint8_t>((PROTOCOL_VERSION >> 8) & 0xff)});
        if (m_server->listen()) {
            log::info("gd-bridge listening on 127.0.0.1:{}", DEFAULT_PORT);
        } else {
            log::error("gd-bridge failed to bind port {}", DEFAULT_PORT);
        }
    }

    SocketServer* server() { return m_server.get(); }

    // Per-attempt state, shared between the PlayLayer hooks (init/reset) and the
    // GJBaseGameLayer::update hook that does the exchange.
    uint32_t frame = 0;
    bool holding = false;
    bool human_holding = false;  // live human jump input, tracked via handleButton
    bool recording = false;      // passive record: stream state, DON'T frame-lock

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
    s.frame = 0;  // set by the update hook
    s.percent = static_cast<float>(pl->getCurrentPercent()) / 100.0f;  // VERIFY: getCurrentPercent
    s.speed_mult = 1.0f;  // TODO: derive from active speed portal
    s.reserved = 0;
    return s;
}

// Dump the level's COLLIDABLE obstacles so Python gets a clean collision map.
// We classify by the game's own GameObjectType, not by guessing IDs, so
// decorations / triggers / pads / rings / coins are dropped and only things the
// cube actually collides with are sent.
std::vector<GeometryRecord> readGeometry(PlayLayer* pl) {
    std::vector<GeometryRecord> out;
    CCArray* objects = pl->m_objects;  // GJBaseGameLayer field
    if (!objects) return out;
    for (unsigned i = 0; i < objects->count(); ++i) {
        auto* obj = static_cast<GameObject*>(objects->objectAtIndex(i));
        if (!obj) continue;
        uint8_t kind;
        switch (obj->m_objectType) {
            case GameObjectType::Solid:
            case GameObjectType::Breakable:
                kind = 0;  // collidable block
                break;
            case GameObjectType::Hazard:
                kind = 1;  // instant death
                break;
            case GameObjectType::Slope:
                // 45-deg floor ramp. kind 2 = rises rightward, 3 = falls
                // rightward (via flipX). Ceiling slopes (flipY) are skipped —
                // the sim only rides floor ramps. HEURISTIC: verify orientation
                // against the game on re-capture and flip if reversed.
                if (obj->isFlipY()) continue;
                kind = obj->isFlipX() ? 3 : 2;
                break;
            default:
                continue;  // decoration, portals, pads, rings, coins, triggers: skip
        }
        const CCPoint p = obj->getPosition();
        out.push_back({kind, p.x / UNITS_PER_BLOCK, p.y / UNITS_PER_BLOCK});
    }
    return out;
}

// Send the current level geometry as one GEOMETRY message.
void sendGeometry(SocketServer* srv, PlayLayer* pl) {
    auto geom = readGeometry(pl);
    std::vector<uint8_t> buf;
    buf.push_back(static_cast<uint8_t>(MsgType::Geometry));
    uint32_t count = static_cast<uint32_t>(geom.size());
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(&count),
               reinterpret_cast<uint8_t*>(&count) + sizeof(count));
    buf.insert(buf.end(), reinterpret_cast<uint8_t*>(geom.data()),
               reinterpret_cast<uint8_t*>(geom.data()) + geom.size() * sizeof(GeometryRecord));
    srv->send(buf.data(), static_cast<uint32_t>(buf.size()));
}

}  // namespace

// ---- hooks ----------------------------------------------------------------

// PlayLayer owns the level lifecycle: open the socket, reset the frame counter
// on level start and on each attempt.
class $modify(BridgePlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
        Bridge::get().ensureListening();
        Bridge::get().frame = 0;
        return true;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        Bridge::get().frame = 0;  // frame counter resets on each attempt (per protocol)
    }
};

// The per-frame exchange lives on GJBaseGameLayer::update — that's the function
// actually driving the game loop in GD 2.2 (PlayLayer does NOT get its own
// update hook there). We only act when `this` is really a PlayLayer (gameplay,
// not the editor) and an agent is connected. Blocking for the action is the
// frame-lock: the game waits for the agent each frame.
class $modify(BridgeBaseLayer, GJBaseGameLayer) {
    // Track the human's live jump input (button 1 = jump for player 1) so record
    // mode can capture their play frame-by-frame.
    void handleButton(bool down, int button, bool isPlayer1) {
        GJBaseGameLayer::handleButton(down, button, isPlayer1);
        if (isPlayer1 && button == 1) Bridge::get().human_holding = down;
    }

    // Build a state payload for the current frame (with the human's live input).
    StatePayload frameState(PlayLayer* pl) {
        StatePayload s = readState(pl);
        if (Bridge::get().human_holding) s.flags |= FLAG_INPUT_HELD;
        s.frame = Bridge::get().frame++;
        return s;
    }

    void update(float dt) {
        Bridge& bridge = Bridge::get();
        SocketServer* srv = bridge.server();
        auto* pl = geode::cast::typeinfo_cast<PlayLayer*>(this);
        if (!srv || !srv->connected() || !pl) {
            bridge.recording = false;  // client gone -> leave record mode
            GJBaseGameLayer::update(dt);
            return;
        }

        // PASSIVE RECORD MODE: run the game at native speed (NO frame-lock) and
        // just stream state + the human's input each frame. This is what makes
        // human play lag-free — nothing waits on the socket.
        if (bridge.recording) {
            GJBaseGameLayer::update(dt);           // native step; human drives
            StatePayload s = frameState(pl);
            if (!srv->send(&s, sizeof(s))) bridge.recording = false;
            return;
        }

        // FRAME-LOCKED AGENT MODE: block for the action, apply it, step, reply.
        std::vector<uint8_t> msg;
        uint8_t act = 0;
        if (srv->recv(msg, /*timeout_ms=*/1000) && msg.size() >= 2 &&
            msg[0] == static_cast<uint8_t>(MsgType::Action)) {
            act = msg[1];
            bridge.holding = act & ACT_HOLD;
        }

        if (act & ACT_RECORD) {
            // Enter passive record mode: normal (non-practice) so death restarts
            // the whole level, optionally reset to start, then stream from here.
            bridge.recording = true;
            pl->togglePracticeMode(false);
            if (act & ACT_REQUEST_RESET) { pl->removeAllCheckpoints(); pl->resetLevelFromStart(); }
            bridge.frame = 0;
            GJBaseGameLayer::update(dt);
            if (act & ACT_REQUEST_GEOM) sendGeometry(srv, pl);
            StatePayload s = frameState(pl);
            srv->send(&s, sizeof(s));
            return;
        }

        if (act & ACT_PRACTICE_ON) pl->togglePracticeMode(true);
        if (act & ACT_PLACE_CHECKPOINT) pl->markCheckpoint();

        if (act & ACT_REQUEST_RESET) {
            pl->removeAllCheckpoints();
            pl->resetLevelFromStart();
            bridge.frame = 0;
        } else if (act & ACT_LOAD_CHECKPOINT) {
            pl->resetLevel();  // practice mode -> respawn at last checkpoint
            bridge.frame = 0;
        } else {
            if (bridge.holding) this->m_player1->pushButton(PlayerButton::Jump);
            else this->m_player1->releaseButton(PlayerButton::Jump);
            GJBaseGameLayer::update(dt);
        }

        if (act & ACT_REQUEST_GEOM) sendGeometry(srv, pl);
        StatePayload s = frameState(pl);
        srv->send(&s, sizeof(s));
    }
};

// Start listening as soon as the mod loads, so the socket is up at the menu —
// the agent can connect any time and just waits for the player to enter a level.
$on_mod(Loaded) {
    Bridge::get().ensureListening();
}
