// Wire protocol shared with the Python bridge (gdrl/envs/bridge.py).
// Keep this file and PROTOCOL.md in lockstep — the Python side hard-codes the
// same struct layout and byte offsets.
#pragma once

#include <cstdint>

namespace gdbridge {

constexpr uint16_t PROTOCOL_VERSION = 0;
constexpr uint16_t DEFAULT_PORT = 51717;

// One GD block is 30 game units; state is exported in blocks to match the sim.
constexpr float UNITS_PER_BLOCK = 30.0f;

enum class MsgType : uint8_t {
    Hello    = 0x00,  // mod -> py, first message: {u8 type, u16 version}
    State    = 0x01,  // mod -> py, per frame
    Action   = 0x02,  // py -> mod, per frame
    Geometry = 0x03,  // mod -> py, on request
};

// State flags (bitfield in State::flags).
enum StateFlag : uint8_t {
    FLAG_DEAD       = 1 << 0,
    FLAG_COMPLETE   = 1 << 1,
    FLAG_UPSIDE     = 1 << 2,  // gravity inverted
    FLAG_INPUT_HELD = 1 << 3,  // the game's jump input is held this frame (human in record mode)
};

// Action flags (bitfield in the action byte).
enum ActionFlag : uint8_t {
    ACT_HOLD             = 1 << 0,
    ACT_REQUEST_RESET    = 1 << 1,  // full restart from start, clear checkpoints
    ACT_REQUEST_GEOM     = 1 << 2,
    ACT_PRACTICE_ON      = 1 << 3,  // togglePracticeMode(true)
    ACT_PLACE_CHECKPOINT = 1 << 4,  // markCheckpoint() at the current position
    ACT_LOAD_CHECKPOINT  = 1 << 5,  // respawn at the last checkpoint (segment search)
    ACT_RECORD           = 1 << 6,  // don't inject input; let the human play, report their input
};

// Gamemodes, matching PlayerObject state. Only Cube is handled in v0.
enum class GameMode : uint8_t {
    Cube = 0, Ship = 1, Ball = 2, Ufo = 3, Wave = 4, Robot = 5, Spider = 6,
};

// Fixed 33-byte State payload. #pragma pack so the layout is exactly the byte
// offsets documented in PROTOCOL.md, with no compiler padding.
#pragma pack(push, 1)
struct StatePayload {
    uint8_t  type;        // = MsgType::State
    float    x;           // blocks
    float    y;           // blocks, bottom of hitbox
    float    vy;          // blocks/s
    uint8_t  grounded;    // 0/1
    uint8_t  gamemode;    // GameMode
    uint8_t  flags;       // StateFlag bitfield
    float    length;      // blocks
    uint32_t frame;       // resets on attempt restart
    float    percent;     // [0,1]
    float    speed_mult;  // 0.5..4.0
    uint8_t  reserved;
};
static_assert(sizeof(StatePayload) == 33, "StatePayload must be 33 bytes");

struct GeometryRecord {
    uint8_t kind;  // 0 block, 1 spike (mirrors gdrl.sim.level)
    float   x;     // blocks, bottom-left
    float   y;     // blocks
};
static_assert(sizeof(GeometryRecord) == 9, "GeometryRecord must be 9 bytes");
#pragma pack(pop)

}  // namespace gdbridge
