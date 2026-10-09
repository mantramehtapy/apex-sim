# The 2025 UDP telemetry layout, as verified

`m_packetFormat = 2025`, UDP, all fields **little-endian**. This is the wire
layout several racing titles publish their live telemetry in; Apex Sim emits
it so standard telemetry tooling can consume the sim's output.

This file records what was actually verified, and from where. It exists so a
future change to `src/telemetry.rs` can be checked against a source rather
than against a memory of one. Apex Sim is an independent project and is not
affiliated with any game studio or vendor; the layout is implemented purely
for interoperability.

## How this was verified

Every field name, type and offset in this document was cross-checked against
**several independent public community decoders** of the layout, and against
the **published packet sizes**, which constrain it arithmetically:

| Source | What it fixed |
|---|---|
| [`LarsLaskowski/F1-Telemetry`](https://raw.githubusercontent.com/LarsLaskowski/F1-Telemetry/main/docs/F1%202025%20Telemetry.md) | The surrounding packet documentation, including the `÷ 32767.0f` normalisation note. |
| [`nemo-MV33/f1-telemetry-21-26`](https://github.com/nemo-MV33/f1-telemetry-21-26) `python/f1_telemetry/{layouts,decoder}.py` | Per-year stride table; header field offsets |
| [`Si6gma/RaceLabs`](https://github.com/Si6gma/RaceLabs) `backend/app/decoder.py` | Independent field offsets for the 2023/2024 layouts |
| [`racehub-io/f1-telemetry-client`](https://github.com/racehub-io/f1-telemetry-client) `src/parsers/packets/*` + `src/constants/packetSizes.ts` | Independent field offsets **and** exact published packet sizes per year |

These decoders are what this repository's own bytes are actually tested
against, as real receivers, in `tools/interop_check*.{py,js}`.

### The size cross-check

This is the strongest constraint, because packet sizes are integers that only
come out right if every stride is right:

```
Motion (id 0), 2025:   29-byte header + 22 cars x 60-byte block          = 1349  ✓ published 1349
Car Telemetry (id 6),  2025: 29-byte header + 22 cars x 60-byte block + 3 = 1352  ✓ published 1352
Car Telemetry (id 6),  2026: 29-byte header + 24 cars x 59-byte block + 3 = 1448  ✓ published 1448
Motion (id 0), 2026:   29-byte header + 24 cars x 54-byte block          = 1325  ✓ published 1325
```

Every stride in this repository is load-bearing: change the 60 and the packet
no longer parses. `tests/phase9_gate.rs` asserts all four of these totals.

## Packet header — 29 bytes

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 2 | u16 | `m_packetFormat` = 2025 |
| 2 | 1 | u8 | `m_gameYear` = 25 |
| 3 | 1 | u8 | `m_gameMajorVersion` |
| 4 | 1 | u8 | `m_gameMinorVersion` |
| 5 | 1 | u8 | `m_packetVersion` |
| 6 | 1 | u8 | `m_packetId` |
| 7 | 8 | u64 | `m_sessionUID` |
| 15 | 4 | f32 | `m_sessionTime` (s) |
| 19 | 4 | u32 | `m_frameIdentifier` |
| 23 | 4 | u32 | `m_overallFrameIdentifier` |
| 27 | 1 | u8 | `m_playerCarIndex` |
| 28 | 1 | u8 | `m_secondaryPlayerCarIndex` |

The 2021/2022 layouts use a 24-byte header; from 2023 the extra
`m_gameYear` and `m_overallFrameIdentifier` push everything from
`m_sessionUID` onward five bytes later. **2025 is 29.**

## Packet ids used here

| id | Name |
|---|---|
| 0 | Motion |
| 6 | Car Telemetry |

## Motion packet (id 0) — 60-byte block per car

| Offset | Type | Field | Unit |
|---|---|---|---|
| 0 | f32 | `m_worldPositionX` | m |
| 4 | f32 | `m_worldPositionY` | m |
| 8 | f32 | `m_worldPositionZ` | m |
| 12 | f32 | `m_velocityX` | m/s |
| 16 | f32 | `m_velocityY` | m/s |
| 20 | f32 | `m_velocityZ` | m/s |
| 24 | **i16** | `m_worldForwardDirX` | normalised, ÷32767 |
| 26 | **i16** | `m_worldForwardDirY` | normalised, ÷32767 |
| 28 | **i16** | `m_worldForwardDirZ` | normalised, ÷32767 |
| 30 | **i16** | `m_worldRightDirX` | normalised, ÷32767 |
| 32 | **i16** | `m_worldRightDirY` | normalised, ÷32767 |
| 34 | **i16** | `m_worldRightDirZ` | normalised, ÷32767 |
| 36 | f32 | `m_gForceLateral` | g |
| 40 | f32 | `m_gForceLongitudinal` | g |
| 44 | f32 | `m_gForceVertical` | g |
| 48 | f32 | `m_yaw` | rad |
| 52 | f32 | `m_pitch` | rad |
| 56 | f32 | `m_roll` | rad |

### The int16 direction channels — a bug this repository shipped and then caught

The first version of `encode_motion` wrote the six direction channels as
`float32`. They are **`int16`**, and the format's own note says how to read
them:

> *N.B. For the normalised vectors below, to convert to float values **divide
> by 32767.0f** — 16-bit signed values are used to pack the data and on the
> assumption that direction values are always between −1.0f and 1.0f.*

Getting this wrong is silent: a float and an int16 occupy the same four
bytes, so every decoder still *parses* the packet, reports no error, and
simply shows a direction of up to ±32767. It also shifts nothing — the field
*count* is the same either way — which is why the packet-size arithmetic in
`tests/phase9_gate.rs` could not detect it.

It was found by `tools/interop_check_racehub.js`, the one decoder of the
three that reads these channels at all: it declared them `int16le`, read the
low two bytes of this engine's floats, and disagreed on 6 920 of 8 250
decodes. The fix is `dir_i16()` in `src/telemetry.rs`.

Two of the other three decoders *skipped* the direction block entirely
(RaceLabs does `offset += 12` and then jumps to g-force at +24, landing on
the same absolute bytes; `f1-telemetry-21-26` reads only position and
g-force). They were correct but blind to it, which is worth recording: three
"independent confirmations" that agree are not three confirmations if two of
them never look at the field in question.

The divisor is 32767, not 32768. The two differ only at exactly ±1.0, and the
spec names 32767.

## Car Telemetry packet (id 6) — 60-byte block per car, then 3 bytes

| Offset | Type | Field | Unit |
|---|---|---|---|
| 0 | u16 | `m_speed` | **km/h** |
| 2 | f32 | `m_throttle` | 0–1 |
| 6 | f32 | `m_steer` | −1–1 |
| 10 | f32 | `m_brake` | 0–1 |
| 14 | u8 | `m_clutch` | 0–100 |
| 15 | i8 | `m_gear` | −1 R, 0 N, 1–8 |
| 16 | u16 | `m_engineRPM` | rev/min |
| 18 | u8 | `m_drs` | 0/1 |
| 19 | u8 | `m_revLightsPercent` | 0–100 |
| 20 | u16 | `m_revLightsBitValue` | bitmask |
| 22 | u16 ×4 | `m_brakesTemperature` | — |
| 30 | u8 ×4 | `m_tyresSurfaceTemperature` | — |
| 34 | u8 ×4 | `m_tyresInnerTemperature` | — |
| 38 | u16 | `m_engineTemperature` | — |
| 40 | f32 ×4 | `m_tyresPressure` | — |
| 56 | u8 ×4 | `m_surfaceType` | — |

Then, after all 22 blocks: u8 `m_mfdPanelIndex`, u8 `m_mfdPanelIndexSecondaryPlayer`,
i8 `m_suggestedGear`.

Tyre temperature and pressure ordering is **rear-left, rear-right, front-left,
front-right** — not front-first. This is the single most common decoder bug.

## Car indexing

Cars are a **fixed 22-entry array**, indexed by position. There is no car id
in the block itself; slot `i` is car `i`. `m_playerCarIndex` does not select
which car a block describes — it tells a receiver which of the 22 slots the
local player is driving, and which to show by default.

So for this engine: sim car `id` → slot `id`, and slots `cars..21` are
zero-filled. A zero block is unambiguous (`m_gear` 0 = neutral, `m_speed` 0),
and it is what a real receiver expects for an empty slot.

## Units, and the one thing this engine has to invent

The physics model is planar and has **no engine** — no crank, no torque curve,
no gearbox state. `m_engineRPM` and `m_gear` are therefore *presentation
channels* derived from wheel speed by `src/telemetry.rs`'s gearbox model. They
are documented as synthetic, they never feed back into physics, and the gate
`telemetry_never_feeds_back_into_physics` proves the race hashes identically
with and without them.

Everything else maps to a quantity the physics actually integrates.

## What is NOT claimed

This repository claims **no** vendor compatibility: no racing game, MoTeC,
or SimHub build exists on the development machine, so none was tested
against. A telemetry display that opens and shows moving cars is not the same
claim as "MoTeC i2 accepts this", and only the first one is made here.

What *is* verified, on a real loopback capture of a real race:

- the bytes this engine emits are decoded by **three independent third-party
  decoders** — in Python, in Python again (a different codebase), and in
  TypeScript — with **zero disagreement** on every field compared, across
  ~4 100 car-slot decodes per decoder, of which ~1 600 carried a moving car
  rather than a stationary grid;
- packet sizes match the published ones exactly (1349 / 1352);
- and the comparison was shown to be live, by shifting one field's offset
  four bytes and confirming the check fails.

Two limits on that evidence, stated plainly:

1. **It is a layout check, not a value check.** Both decoders read the same
   bytes, so corrupting a packet leaves them in agreement. What the check
   catches is disagreement about what a byte *means* — which is precisely the
   failure mode that bit this encoder once already.
2. **No vendor application was involved**, for the reason above.

Tools: `tools/telemetry_recv.py` (independent receiver, live table, hex dump,
CSV, raw `.bin` for feeding other decoders), `tools/interop_check.py`,
`tools/interop_check_racelabs.py`, `tools/interop_check_racehub.js`, and
`tests/phase9_gate.rs`.