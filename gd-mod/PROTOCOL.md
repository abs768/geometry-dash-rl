# Bridge protocol v0 (mod <-> Python)

Transport: TCP on 127.0.0.1:51717, mod is the server. All messages are
length-prefixed: `u32 little-endian payload size` + payload. Payloads are
fixed-layout little-endian binary — no JSON in the hot loop.

The **mod drives the loop**: it sends STATE, then blocks the game's update
until it receives ACTION. One STATE/ACTION pair per physics frame.

## Messages

### mod -> Python: STATE (33 bytes)
| offset | type | field |
|--------|------|-------|
| 0  | u8  | msg type = 0x01 |
| 1  | f32 | x (blocks) |
| 5  | f32 | y (blocks, bottom of hitbox) |
| 9  | f32 | y-velocity (blocks/s) |
| 13 | u8  | grounded (0/1) |
| 14 | u8  | gamemode (0 cube, 1 ship, 2 ball, 3 ufo, 4 wave, 5 robot, 6 spider) |
| 15 | u8  | flags: bit0 dead, bit1 level complete, bit2 upside-down gravity |
| 16 | f32 | level length (blocks) |
| 20 | u32 | frame counter (resets on attempt restart) |
| 24 | f32 | percent progress [0,1] |
| 28 | f32 | speed multiplier currently active (0.5x..4x) |
| 32 | u8  | reserved |

### Python -> mod: ACTION (2 bytes)
| offset | type | field |
|--------|------|-------|
| 0 | u8 | msg type = 0x02 |
| 1 | u8 | bit0 hold; bit1 request reset; bit2 request level geometry dump |

### mod -> Python: GEOMETRY (variable, on request)
`u8 0x03` + `u32 count` + count records of `{u8 kind, f32 x, f32 y}` — the
level's obstacle list in block units, letting Python build the same
look-ahead grid observation used in the sim.

## Versioning

First message after connect: mod sends `u8 0x00` + `u16 protocol_version`.
Python closes the connection on mismatch.
