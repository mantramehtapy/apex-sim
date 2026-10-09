#!/usr/bin/env python3
"""An independent receiver for the 2025 UDP telemetry layout.

This is deliberately a *second implementation* of the packet layout. It shares
no code with src/telemetry.rs -- it decodes straight from the published
layout, using struct.unpack with explicit little-endian format strings. If
both the Rust encoder and this decoder agree on the bytes, the bytes are
right; if they disagreed, one of them is wrong and the disagreement is
visible. A receiver that re-used the encoder's own offsets could only ever
prove the encoder is self-consistent.

It is also the "one receiver" the plan asks for: it listens on a real UDP
socket and prints a real-time dashboard of what a race is doing.

Usage
-----
  telemetry_recv.py --port 20777 --seconds 10
  telemetry_recv.py --port 20777 --seconds 10 --cap out.cap --csv out.csv

Outputs
-------
  stdout        a live table, refreshed in place
  --cap PATH    a hex dump of the first packets received, raw bytes
  --csv PATH    one decoded row per Motion packet, per car

Exit status
-----------
  0  at least one packet of each expected id decoded cleanly
  1  no packets, or a packet that did not match the expected size
"""

import argparse
import csv
import os
import socket
import struct
import sys
import time

# --- The published 2025 layout. All little-endian. ---------------------------

HEADER_SIZE = 29
HEADER_FMT = "<HBBBBBQfIIBB"
MAX_CARS = 22
MOTION_ID = 0
CAR_TELEMETRY_ID = 6
MOTION_STRIDE = 60
CAR_TELEMETRY_STRIDE = 60
MOTION_SIZE = HEADER_SIZE + MAX_CARS * MOTION_STRIDE              # 1349
CAR_TELEMETRY_SIZE = HEADER_SIZE + MAX_CARS * CAR_TELEMETRY_STRIDE + 3  # 1352

HEADER_FIELDS = (
    "packet_format", "game_year", "major", "minor", "packet_version",
    "packet_id", "session_uid", "session_time", "frame_id", "overall_frame_id",
    "player_car_index", "secondary_player_index",
)

# Motion block: three parts, NOT fifteen floats. The six direction channels
# are int16 normalised (divide by 32767.0), and reading them as floats is a
# silent corruption -- a receiver would show a direction of up to 32767.
# Note also that gForce is at +36 and yaw at +48.
MOTION_FMT_A = "<6f"                 # pos xyz, vel xyz            (0..24)
MOTION_FMT_B = "<6h"                 # forward xyz, right xyz      (24..36)
MOTION_FMT_C = "<6f"                 # gForce xyz, yaw pitch roll  (36..60)
DIRECTION_SCALE = 32767.0
MOTION_FIELDS = (
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "g_lat", "g_lon", "g_vert",
    "yaw", "pitch", "roll",
)
DIR_FIELDS = ("fwd_x", "fwd_y", "fwd_z", "right_x", "right_y", "right_z")
assert struct.calcsize(MOTION_FMT_A) + struct.calcsize(MOTION_FMT_B)     + struct.calcsize(MOTION_FMT_C) == MOTION_STRIDE

# Car telemetry block. Two structs, because the fields are not uniform: u16
# speed, four f32, three u8/i8, u16 RPM, three u8, four u16 brake temps,
# four u8 surface temps, four u8 inner temps, u16 engine temp, four f32
# pressures, four u8 surface type.
CAR_FMT = "<H3fBbHBBH4H4B4BH4f4B"
CAR_FIELDS = (
    "speed_kph", "throttle", "steer", "brake",
    "clutch", "gear", "rpm", "drs", "rev_lights_pct", "rev_lights_bits",
    "brake_t0", "brake_t1", "brake_t2", "brake_t3",
    "surf_t0", "surf_t1", "surf_t2", "surf_t3",
    "inner_t0", "inner_t1", "inner_t2", "inner_t3",
    "engine_temp",
    "press0", "press1", "press2", "press3",
    "surface_type0", "surface_type1", "surface_type2", "surface_type3",
)
assert struct.calcsize(CAR_FMT) == CAR_TELEMETRY_STRIDE, struct.calcsize(CAR_FMT)


class BadPacket(Exception):
    pass


def decode_header(buf):
    """Return a dict for the 29-byte header, or raise if the size is wrong."""
    if len(buf) < HEADER_SIZE:
        raise BadPacket("short header: %d bytes" % len(buf))
    vals = struct.unpack_from(HEADER_FMT, buf, 0)
    return dict(zip(HEADER_FIELDS, vals))


def check_size(packet_id, size):
    if packet_id == MOTION_ID and size != MOTION_SIZE:
        raise BadPacket("motion packet is %d bytes, expected %d" % (size, MOTION_SIZE))
    if packet_id == CAR_TELEMETRY_ID and size != CAR_TELEMETRY_SIZE:
        raise BadPacket(
            "car telemetry packet is %d bytes, expected %d" % (size, CAR_TELEMETRY_SIZE)
        )


def decode_motion(buf):
    """Decode a Motion packet into {header fields..., 'cars': [ ... ]}."""
    hdr = decode_header(buf)
    check_size(hdr["packet_id"], len(buf))
    cars = []
    for i in range(MAX_CARS):
        off = HEADER_SIZE + i * MOTION_STRIDE
        d = dict(zip(MOTION_FIELDS,
                     struct.unpack_from(MOTION_FMT_A, buf, off)
                     + struct.unpack_from(MOTION_FMT_C, buf, off + 36)))
        raw = struct.unpack_from(MOTION_FMT_B, buf, off + 24)
        d.update({k: v / DIRECTION_SCALE for k, v in zip(DIR_FIELDS, raw)})
        # An empty slot is all-zero: a real receiver must be able to tell
        # "car 20 does not exist" from "car 20 is stationary".
        d["present"] = any(v != 0.0 for v in d.values())
        cars.append(d)
    hdr["cars"] = cars
    return hdr


def decode_car_telemetry(buf):
    hdr = decode_header(buf)
    check_size(hdr["packet_id"], len(buf))
    cars = []
    for i in range(MAX_CARS):
        off = HEADER_SIZE + i * CAR_TELEMETRY_STRIDE
        vals = struct.unpack_from(CAR_FMT, buf, off)
        d = dict(zip(CAR_FIELDS, vals))
        d["present"] = d["speed_kph"] != 0 or d["rpm"] != 0 or d["gear"] != 0
        cars.append(d)
    tail = struct.unpack_from("<BBb", buf, HEADER_SIZE + MAX_CARS * CAR_TELEMETRY_STRIDE)
    hdr["mfd_panel_index"] = tail[0]
    hdr["mfd_panel_index_secondary"] = tail[1]
    hdr["suggested_gear"] = tail[2]
    hdr["cars"] = cars
    return hdr


def check_car_telemetry(pkt):
    """Range checks on a Car Telemetry packet.

    These are invariants of the format and of driving, not of this engine:
    a value that fails one of them means the *bytes* are wrong somewhere,
    not that a car did something odd.
    """
    problems = []
    for i, c in enumerate(pkt["cars"]):
        if not c["present"]:
            continue
        if not (-1.0001 <= c["steer"] <= 1.0001):
            problems.append("car %d steer %.3f out of range" % (i, c["steer"]))
        for k in ("throttle", "brake"):
            if not (-0.0001 <= c[k] <= 1.0001):
                problems.append("car %d %s %.3f out of range" % (i, k, c[k]))
        if not (-1 <= c["gear"] <= 8):
            problems.append("car %d gear %d out of range" % (i, c["gear"]))
        if not (0 <= c["rev_lights_pct"] <= 100):
            problems.append("car %d rev lights %d out of range" % (i, c["rev_lights_pct"]))
        if not (0 <= c["rpm"] <= 20000):
            problems.append("car %d rpm %d out of range" % (i, c["rpm"]))
    return problems


def check_motion(pkt, last):
    """Continuity of the Motion stream: time must not reverse, and no car may
    teleport between two consecutive packets.

    At the fastest this engine goes, one 4 ms tick moves under 2 m. A jump of
    200 m in one frame is a decode error -- a misaligned stride, or a decoder
    reading the wrong block -- and not a car.
    """
    problems = []
    if last is None:
        return problems
    dt = pkt["session_time"] - last["session_time"]
    if dt < 0:
        problems.append("session time went backwards (%.4f < %.4f)"
                        % (pkt["session_time"], last["session_time"]))
        return problems
    if dt <= 0 or dt >= 1.0:
        # Same instant, or a gap too large for continuity to mean anything.
        return problems
    for i in range(MAX_CARS):
        a, b = last["cars"][i], pkt["cars"][i]
        if not (a["present"] and b["present"]):
            continue
        jump = ((a["pos_x"] - b["pos_x"]) ** 2 + (a["pos_y"] - b["pos_y"]) ** 2) ** 0.5
        if jump > 200.0 * dt + 5.0:
            problems.append("car %d jumped %.1f m in %.3f s" % (i, jump, dt))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=20777, help="UDP port (default 20777)")
    ap.add_argument("--seconds", type=float, default=5.0, help="listen duration")
    ap.add_argument("--cap", help="write a hex dump of the first N packets here")
    ap.add_argument("--raw-dir",
                    help="write raw .bin packets here, for feeding a third-party "
                         "decoder such as f1-telemetry-21-26")
    ap.add_argument("--cap-packets", type=int, default=4, help="packets to hex-dump")
    ap.add_argument("--csv", help="write decoded Motion rows here")
    ap.add_argument("--quiet", action="store_true", help="no live table")
    ap.add_argument("--expect-player", type=int, default=None,
                    help="fail unless this playerCarIndex is seen")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(0.25)

    print("listening on %s:%d for %.1f s" % (args.host, args.port, args.seconds),
          file=sys.stderr)

    counts = {MOTION_ID: 0, CAR_TELEMETRY_ID: 0}
    other_ids = {}
    bad = []
    problems = []
    cap_f = open(args.cap, "w") if args.cap else None
    raw_dir = None
    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)
        raw_dir = args.raw_dir
    csv_f = None
    csv_w = None
    if args.csv:
        csv_f = open(args.csv, "w", newline="")
        cols = ["session_uid", "session_time", "frame_id", "player_car_index"]
        cols += ["car", "pos_x", "pos_y", "pos_z", "g_lat", "g_lon", "yaw"]
        csv_w = csv.DictWriter(csv_f, fieldnames=cols)
        csv_w.writeheader()

    started = time.time()
    deadline = started + args.seconds
    last_motion = None
    seen_player = set()
    capped = 0
    raw_n = 0
    last_print = 0.0
    latest = None

    while time.time() < deadline:
        try:
            data, _src = sock.recvfrom(65535)
        except socket.timeout:
            continue
        try:
            hdr = decode_header(data)
            pid = hdr["packet_id"]
            if pid not in counts:
                other_ids[pid] = other_ids.get(pid, 0) + 1
                continue
            if pid == MOTION_ID:
                pkt = decode_motion(data)
                problems.extend(check_motion(pkt, last_motion))
                last_motion = pkt
                latest = pkt
                if csv_w is not None:
                    for i, c in enumerate(pkt["cars"]):
                        if not c["present"]:
                            continue
                        row = {
                            "session_uid": pkt["session_uid"],
                            "session_time": round(pkt["session_time"], 4),
                            "frame_id": pkt["frame_id"],
                            "player_car_index": pkt["player_car_index"],
                            "car": i,
                        }
                        for k in ("pos_x", "pos_y", "pos_z", "g_lat", "g_lon", "yaw"):
                            row[k] = round(c[k], 4)
                        csv_w.writerow(row)
            else:
                pkt = decode_car_telemetry(data)
                problems.extend(check_car_telemetry(pkt))
                if latest is None:
                    latest = pkt
            counts[pid] += 1
            seen_player.add(hdr["player_car_index"])
        except (BadPacket, struct.error) as e:
            if len(bad) < 20:
                bad.append("%s: %s" % (type(e).__name__, e))

        if cap_f is not None and capped < args.cap_packets:
            cap_f.write("# packet %d: %d bytes\n" % (capped, len(data)))
            for row in range(0, len(data), 16):
                chunk = data[row:row + 16]
                cap_f.write("%04x  %s\n" % (
                    row,
                    " ".join("%02x" % b for b in chunk),
                ))
            cap_f.write("\n")
            capped += 1

        if raw_dir is not None:
            # Raw bytes, named by arrival order. This is what a *third-party*
            # decoder consumes, and it is the strongest evidence available
            # that the encoder matches the published layout: a decoder this
            # project did not write, reading the engine's real packets.
            with open(os.path.join(raw_dir, "p%06d.bin" % raw_n), "wb") as f:
                f.write(data)
            raw_n += 1

        now = time.time()
        if not args.quiet and latest is not None and now - last_print >= 0.1:
            last_print = now
            _render(latest)

    sock.close()
    if cap_f:
        cap_f.close()
    if csv_f:
        csv_f.close()

    print("", file=sys.stderr)
    print("=" * 68)
    print("motion packets          : %d" % counts[MOTION_ID])
    print("car telemetry packets   : %d" % counts[CAR_TELEMETRY_ID])
    if other_ids:
        print("other packet ids        : %s" %
              ", ".join("%d x%d" % (k, v) for k, v in sorted(other_ids.items())))
    print("player car indices seen : %s" % (sorted(seen_player) or "none"))
    print("expected packet sizes   : motion %d, car telemetry %d"
          % (MOTION_SIZE, CAR_TELEMETRY_SIZE))
    if bad:
        print("MALFORMED               : %d (first: %s)" % (len(bad), bad[0]))
    if problems:
        print("PLAUSIBILITY FAILURES   : %d (first: %s)" % (len(problems), problems[0]))
    else:
        print("PLAUSIBILITY            : all checks passed")
    if args.cap:
        print("hex capture             : %s (%d packets)" % (args.cap, capped))
    if args.csv:
        print("decoded csv             : %s" % args.csv)

    ok = counts[MOTION_ID] > 0 and counts[CAR_TELEMETRY_ID] > 0
    ok = ok and not bad and not problems
    if args.expect_player is not None and args.expect_player not in seen_player:
        print("expected playerCarIndex %d was never seen" % args.expect_player)
        ok = False
    return 0 if ok else 1


def _render(pkt):
    """A live table: one row per car present in the last Motion packet.

    Only Motion carries position and velocity, so a Car Telemetry packet has
    nothing to draw; the table simply keeps the previous Motion frame.
    """
    if pkt.get("packet_id") != MOTION_ID:
        return
    cars = [c for c in pkt["cars"] if c["present"]]
    lines = ["", "t=%8.3f s  frame %-8d uid %d  player slot %d  (%d cars)"
             % (pkt["session_time"], pkt["frame_id"], pkt["session_uid"],
                pkt["player_car_index"], len(cars))]
    lines.append("  car      x        y        speed      g_lat    g_lon      yaw")
    for i, c in enumerate(pkt["cars"]):
        if not c["present"]:
            continue
        lines.append("  %3d %8.1f %8.1f %9.1f %+8.2f %+8.2f %8.2f"
                     % (i, c["pos_x"], c["pos_y"],
                        (c["vel_x"] ** 2 + c["vel_y"] ** 2) ** 0.5,
                        c["g_lat"], c["g_lon"], c["yaw"]))
    sys.stderr.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())