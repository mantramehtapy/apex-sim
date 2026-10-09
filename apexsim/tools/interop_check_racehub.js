#!/usr/bin/env node
// Third cross-check: the engine's telemetry against the racehub TypeScript
// decoder.
//
// A third codebase, in a third language, with its own `binary-parser` layout
// declarations and its own published packet-size table. tools/interop_check.py
// and tools/interop_check_racelabs.py cover two of those axes; this one
// covers the third, and it is the decoder whose `packetSizes.ts` supplied the
// integer sizes the whole layout was validated against in the first place.
//
// The decoder is not vendored. Build it first:
//
//   git clone --depth 1 https://github.com/racehub-io/f1-telemetry-client.git
//   cd f1-telemetry-client && npm install && npx tsc -p tsconfig.build.json
//
// Usage:
//   node interop_check_racehub.js --packets DIR --decoder RACE/repo/build/main
//
// Exit status 0 when every packet agreed on every compared field.

const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 2) {
    out[argv[i].replace(/^--/, "")] = argv[i + 1];
  }
  return out;
}

// This repo's decoder, in JS. Deliberately written from the published layout
// with its own struct strings rather than translated from telemetry_recv.py,
// so a mistake in one is not a mistake in all three.
const HEADER_FMT = "<HBBBBBQfIIBB";
const HEADER_SIZE = 29;
const MOTION_STRIDE = 60;
const CAR_STRIDE = 60;
const CAR_FMT = "<H3fBbHBBH4H4B4BH4f4B";
// The six direction channels are int16 normalised, not float32. Read as
// floats they silently produce garbage while still "parsing".
const DIR_SCALE = 32767.0;
// Byte offsets, not an index into one flat float array: the six direction
// channels sit between the velocities and the g-force, and indexing them as
// floats silently shifts every field after them by three words.
const MOTION_FIELDS = [
  "pos_x", "pos_y", "pos_z",
  "vel_x", "vel_y", "vel_z",
  "g_lat", "g_lon", "g_vert",
  "yaw", "pitch", "roll",
];
const MOTION_F32_OFFSET = {
  pos_x: 0, pos_y: 4, pos_z: 8,
  vel_x: 12, vel_y: 16, vel_z: 20,
  g_lat: 36, g_lon: 40, g_vert: 44,
  yaw: 48, pitch: 52, roll: 56,
};
const DIR_FIELDS = ["fwd_x", "fwd_y", "fwd_z", "right_x", "right_y", "right_z"];
const DIR_I16_OFFSET = { fwd_x: 24, fwd_y: 26, fwd_z: 28, right_x: 30, right_y: 32, right_z: 34 };

function decodeHeader(buf) {
  const v = Buffer.from(buf).readUInt16LE(0);
  return {
    packet_format: v,
    game_year: buf[2],
    packet_id: buf[6],
    session_uid: Number(buf.readBigUInt64LE(7)),
    session_time: buf.readFloatLE(15),
    frame_id: buf.readUInt32LE(19),
    overall_frame_id: buf.readUInt32LE(23),
    player_car_index: buf[27],
    secondary_player_index: buf[28],
  };
}

function slotOffset(header, car, stride) {
  return header.header_size + car * stride;
}

function main() {
  const args = parseArgs(process.argv);
  if (!args.packets || !args.decoder) {
    console.error("usage: node interop_check_racehub.js --packets DIR --decoder BUILD/MAIN");
    return 2;
  }
  let client;
  try {
    client = require(path.resolve(args.decoder, "index.js"));
  } catch (e) {
    console.error("error: cannot load the racehub decoder at %s: %s", args.decoder, e.message);
    return 2;
  }

  const files = fs.readdirSync(args.packets)
    .filter((f) => f.endsWith(".bin"))
    .sort()
    .map((f) => path.join(args.packets, f));
  if (files.length === 0) {
    console.error("error: no .bin packets in %s", args.packets);
    return 2;
  }

  const TOL = {
    pos_x: 1e-3, pos_y: 1e-3, pos_z: 1e-3,
    vel_x: 1e-3, vel_y: 1e-3, vel_z: 1e-3,
    g_lat: 1e-3, g_lon: 1e-3, g_vert: 1e-3,
    yaw: 1e-3, pitch: 1e-3, roll: 1e-3,
    speed: 0, throttle: 1e-6, brake: 1e-6, steer: 1e-6,
    m_gear: 0, m_engineRPM: 0, m_drs: 0, m_revLightsPercent: 0,
    m_engineTemperature: 0,
  };

  let motionN = 0, motionOk = 0, carN = 0, carOk = 0, moving = 0;
  const maxdiff = {};
  const bad = [];

  for (const file of files) {
    const raw = fs.readFileSync(file);
    const name = path.basename(file);
    const hdr = decodeHeader(raw);
    hdr.header_size = HEADER_SIZE;
    let parsed;
    try {
      parsed = client.F1TelemetryClient.parseBufferMessage(raw, false);
    } catch (e) {
      bad.push([name, "decoder threw: " + e.message]);
      continue;
    }
    if (!parsed) {
      bad.push([name, "decoder refused the packet"]);
      continue;
    }
    // binary-parser puts the decoded values on `.data`; the parser object
    // itself carries the chain of field definitions.
    const pd = parsed.packetData.data;
    if (pd.m_header.m_packetId !== hdr.packet_id) {
      bad.push([name, "packet id " + pd.m_header.m_packetId + " vs " + hdr.packet_id]);
      continue;
    }
    if (Math.abs(pd.m_header.m_sessionTime - hdr.session_time) > 1e-4) {
      bad.push([name, "session time " + pd.m_header.m_sessionTime + " vs " + hdr.session_time]);
      continue;
    }
    if (pd.m_header.m_frameIdentifier !== hdr.frame_id) {
      bad.push([name, "frame id " + pd.m_header.m_frameIdentifier + " vs " + hdr.frame_id]);
      continue;
    }
    if (pd.m_header.m_playerCarIndex !== hdr.player_car_index) {
      bad.push([name, "player index " + pd.m_header.m_playerCarIndex
        + " vs " + hdr.player_car_index]);
      continue;
    }

    if (hdr.packet_id === 0) {
      const mineAll = [];
      for (let i = 0; i < 22; i++) {
        const off = slotOffset(hdr, i, MOTION_STRIDE);
        const vals = {};
        for (const f of MOTION_FIELDS) {
          vals[f] = Buffer.from(raw).readFloatLE(off + MOTION_F32_OFFSET[f]);
        }
        for (const f of DIR_FIELDS) {
          vals[f] = Buffer.from(raw).readInt16LE(off + DIR_I16_OFFSET[f]) / DIR_SCALE;
        }
        vals.present = MOTION_FIELDS.concat(DIR_FIELDS).some((f) => vals[f] !== 0);
        mineAll.push(vals);
      }
      for (let i = 0; i < 22; i++) {
        const t = pd.m_carMotionData[i];
        const m = mineAll[i];
        let ok = true;
        for (const [k, tv] of [["pos_x", t.m_worldPositionX], ["pos_y", t.m_worldPositionY],
                               ["pos_z", t.m_worldPositionZ], ["vel_x", t.m_worldVelocityX],
                               ["vel_y", t.m_worldVelocityY], ["vel_z", t.m_worldVelocityZ],
                               ["g_lat", t.m_gForceLateral], ["g_lon", t.m_gForceLongitudinal],
                               ["g_vert", t.m_gForceVertical], ["yaw", t.m_yaw],
                               ["pitch", t.m_pitch], ["roll", t.m_roll],
                               ["fwd_x", t.m_worldForwardDirX / DIR_SCALE],
                               ["fwd_y", t.m_worldForwardDirY / DIR_SCALE],
                               ["fwd_z", t.m_worldForwardDirZ / DIR_SCALE],
                               ["right_x", t.m_worldRightDirX / DIR_SCALE],
                               ["right_y", t.m_worldRightDirY / DIR_SCALE],
                               ["right_z", t.m_worldRightDirZ / DIR_SCALE]]) {
          if (tv === undefined || tv === null) {
            ok = false;
            bad.push([name, "motion slot " + i + " " + k + ": reference supplied no value -- the comparison would be vacuous"]);
            continue;
          }
          const d = Math.abs(m[k] - tv);
          maxdiff[k] = Math.max(maxdiff[k] || 0, d);
          if (d > TOL[k]) { ok = false; bad.push([name, "motion slot " + i + " " + k + ": " + m[k] + " vs " + tv]); }
        }
        if (Math.abs(m.vel_x) > 1.0) moving++;
        motionN++;
        if (ok) motionOk++;
      }
    } else if (hdr.packet_id === 6) {
      for (let i = 0; i < 22; i++) {
        const t = pd.m_carTelemetryData[i];
        const off = slotOffset(hdr, i, CAR_STRIDE);
        const b = Buffer.from(raw);
        const mine = {
          speed: b.readUInt16LE(off + 0),
          throttle: b.readFloatLE(off + 2),
          steer: b.readFloatLE(off + 6),
          brake: b.readFloatLE(off + 10),
          m_gear: b.readInt8(off + 15),
          m_engineRPM: b.readUInt16LE(off + 16),
          m_drs: b.readUInt8(off + 18),
          m_revLightsPercent: b.readUInt8(off + 19),
          m_clutch: b.readUInt8(off + 14),
          m_revLightsBitValue: b.readUInt16LE(off + 20),
          m_engineTemperature: b.readUInt16LE(off + 38),
          brakes: [0, 1, 2, 3].map((k) => b.readUInt16LE(off + 22 + k * 2)),
          surf: [0, 1, 2, 3].map((k) => b.readUInt8(off + 30 + k)),
          inner: [0, 1, 2, 3].map((k) => b.readUInt8(off + 34 + k)),
          press: [0, 1, 2, 3].map((k) => b.readFloatLE(off + 40 + k * 4)),
          surfaceType: [0, 1, 2, 3].map((k) => b.readUInt8(off + 56 + k)),
        };
        let ok = true;
        for (const [k, tv] of [["speed", t.m_speed], ["throttle", t.m_throttle],
                               ["steer", t.m_steer], ["brake", t.m_brake],
                               ["m_gear", t.m_gear], ["m_engineRPM", t.m_engineRPM],
                               ["m_drs", t.m_drs], ["m_revLightsPercent", t.m_revLightsPercent],
                               ["m_engineTemperature", t.m_engineTemperature]]) {
          if (tv === undefined || tv === null) {
            ok = false;
            bad.push([name, "car slot " + i + " " + k + ": reference supplied no value -- the comparison would be vacuous"]);
            continue;
          }
          const d = Math.abs(Number(mine[k]) - Number(tv));
          maxdiff[k] = Math.max(maxdiff[k] || 0, d);
          if (d > TOL[k]) { ok = false; bad.push([name, "car slot " + i + " " + k + ": " + mine[k] + " vs " + tv]); }
        }
        // The 16 tyre/brake channels are the part most often transposed
        // between decoders, and they are zero here, so agreement proves both
        // decoders put the block boundary in the same place.
        const groups = [["brakes", t.m_brakesTemperature], ["surf", t.m_tyresSurfaceTemperature],
                        ["inner", t.m_tyresInnerTemperature], ["press", t.m_tyresPressure],
                        ["surfaceType", t.m_surfaceType]];
        for (const [key, arr] of groups) {
          if (!Array.isArray(arr)) { ok = false; bad.push([name, "car slot " + i + " " + key + " is not an array"]); continue; }
          if (arr.length !== 4) { ok = false; bad.push([name, "car slot " + i + " " + key + " has " + arr.length + " entries"]); continue; }
          for (let k = 0; k < 4; k++) {
            const d = Math.abs(Number(mine[key][k]) - Number(arr[k]));
            maxdiff[key + k] = Math.max(maxdiff[key + k] || 0, d);
            if (d > 0) { ok = false; bad.push([name, "car slot " + i + " " + key + "[" + k + "]: " + mine[key][k] + " vs " + arr[k]]); }
          }
        }
        // Trailing bytes after all 22 blocks.
        const tail = HEADER_SIZE + 22 * CAR_STRIDE;
        const mfd = b.readUInt8(tail);
        const mfd2 = b.readUInt8(tail + 1);
        const suggested = b.readInt8(tail + 2);
        if (mfd !== pd.m_mfdPanelIndex || mfd2 !== pd.m_mfdPanelIndexSecondaryPlayer
            || suggested !== pd.m_suggestedGear) {
          ok = false;
          bad.push([name, "trailing bytes " + mfd + "," + mfd2 + "," + suggested
            + " vs " + pd.m_mfdPanelIndex + "," + pd.m_mfdPanelIndexSecondaryPlayer
            + "," + pd.m_suggestedGear]);
        }
        if (mine.speed > 10) moving++;
        carN++;
        if (ok) carOk++;
      }
    }
  }

  console.log("racehub decoder  : %s", path.resolve(args.decoder));
  console.log("packets read     : %d", files.length);
  console.log("motion decodes   : %d, agreed on all 18 fields: %d", motionN, motionOk);
  console.log("car telemetry    : %d, agreed on 9 fields + 20 tyre/brake + 3 trailing: %d", carN, carOk);
  console.log("decodes w/ MOVING data: %d", moving);
  console.log("largest absolute differences:");
  for (const k of Object.keys(maxdiff).sort()) {
    console.log("   " + k.padEnd(22) + " " + maxdiff[k]);
  }
  if (bad.length > 0) {
    console.log("\nDISAGREEMENTS: %d", bad.length);
    for (const [n, m] of bad.slice(0, 10)) console.log("   %s: %s", n, m);
    return 1;
  }
  console.log("\nno disagreements: the racehub decoder read every packet identically");
  const ok = motionOk === motionN && carOk === carN && motionN > 0 && moving > 0;
  if (!ok) console.log("\nrefusing to call this a pass: too few packets, or nothing moved");
  return ok ? 0 : 1;
}

process.exit(main());