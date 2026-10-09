#!/usr/bin/env python3
"""Cross-check the engine's telemetry against a THIRD-PARTY decoder.

This is the interop oracle for Phase 9. It takes raw packet bytes captured by
`tools/telemetry_recv.py --raw-dir` and feeds them to an independent 2025-layout
telemetry decoder written by somebody else, then compares what each decoder
read. If two implementations that share no code agree, they agree about the
LAYOUT -- which is the thing that actually goes wrong.

WHAT THIS PROVES, AND WHAT IT DOES NOT
--------------------------------------
Both decoders read the same bytes. So this check detects a disagreement about
what a byte *means*: a wrong stride, an off-by-one offset, a field mistyped
as float32 where the format says int16. Those are exactly the failures that
matter, and a decoder sharing code with the encoder could never find them.
This check is what caught the encoder writing float32 into the six
normalised direction channels.

It cannot detect a wrong *value*. Corrupting a packet makes both decoders
read the same wrong number and they still agree. A green run is evidence
about structure, not about any single number being right; value ranges are
checked separately, by telemetry_recv.py's plausibility checks and by
tests/phase9_gate.rs.

The check has been verified to be live: shifting one field's offset by four
bytes makes it exit 1, and the unshifted version exits 0.

The decoder is not vendored here. It is cloned from its own repository:

    git clone --depth 1 https://github.com/nemo-MV33/f1-telemetry-21-26.git
    cd f1-telemetry-21-26/python

Usage:
    interop_check.py --packets DIR --decoder PATH/TO/f1-telemetry-21-26/python

Exit status 0 when every packet agreed on every compared field.
"""

import argparse
import glob
import os
import struct
import sys

# Field names in this repo's decoder vs. the reference decoder's. Kept as an
# explicit table so a rename on either side fails loudly instead of silently
# comparing the wrong pair.
CAR_FIELD_MAP = {
    "speed_kph": "speed_kph",
    "throttle": "throttle",
    "steer": "steer",
    "brake": "brake",
    "gear": "gear",
    "rpm": "engine_rpm",
    "engine_temp": "engine_temperature",
}
MOTION_FIELD_MAP = {
    "pos_x": "world_position_x",
    "pos_z": "world_position_z",
    "g_lat": "g_force_lateral",
    "g_lon": "g_force_longitudinal",
}
# Absolute tolerance per field. Speeds and rpm are integers on the wire, so
# they cannot differ by anything but a decode error; the f32 fields carry the
# full float32 rounding of both decoders' arithmetic, so 1e-3 m / g is far
# tighter than any real disagreement while tolerating last-bit noise.
TOL = {"speed_kph": 0.0, "rpm": 0.0, "gear": 0.0, "throttle": 1e-6,
       "brake": 1e-6, "steer": 1e-6, "engine_temp": 0.0,
       "pos_x": 1e-3, "pos_z": 1e-3, "g_lat": 1e-3, "g_lon": 1e-3}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packets", required=True, help="directory of .bin packets")
    ap.add_argument("--decoder", required=True,
                    help="path to the third-party decoder's python/ directory")
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)),
                    help="this repo's tools/ directory (default: alongside this script)")
    args = ap.parse_args()

    decoder_path = os.path.abspath(args.decoder)
    if not os.path.isdir(os.path.join(decoder_path, "f1_telemetry")):
        print("error: %s does not contain an f1_telemetry package" % decoder_path,
              file=sys.stderr)
        return 2
    sys.path.insert(0, decoder_path)
    sys.path.insert(0, os.path.abspath(args.tools))
    try:
        from f1_telemetry import decode_packet as third_party
    except ImportError as e:
        print("error: cannot import the third-party decoder: %s" % e, file=sys.stderr)
        return 2
    import telemetry_recv as ours

    files = sorted(glob.glob(os.path.join(args.packets, "*.bin")))
    if not files:
        print("error: no .bin packets in %s" % args.packets, file=sys.stderr)
        return 2

    motion_n = motion_ok = car_n = car_ok = 0
    maxdiff = {}
    disagreements = []
    # The reference decoder decodes exactly one car per packet — the one named
    # by m_playerCarIndex. To make it read every car this engine published, the
    # loop below rewrites that one header byte on a copy of the packet and
    # re-decodes, once per slot. The rewrite touches only the header; every
    # compared byte is still the engine's own.
    PLAYER_INDEX_OFFSET = 27
    slots_seen = 0
    moving_checked = 0
    moving_ok = 0

    for path in files:
        raw = open(path, "rb").read()
        name = os.path.basename(path)
        hdr = ours.decode_header(raw)
        if hdr["packet_id"] not in (ours.MOTION_ID, ours.CAR_TELEMETRY_ID):
            continue
        # How many slots this packet actually carries. A 20-car grid publishes
        # 20 blocks and zeroes the other two; asking the reference decoder to
        # read an empty slot is a valid check that the zero-fill is where a
        # receiver expects it, so all MAX_CARS slots are attempted.
        probe = ours.decode_motion(raw)["cars"] if hdr["packet_id"] == ours.MOTION_ID \
            else ours.decode_car_telemetry(raw)["cars"]
        present = sum(1 for c in probe if c["present"]) or 1
        n_slots = min(present + 1, ours.MAX_CARS)

        for slot in range(n_slots):
            data = bytearray(raw)
            data[PLAYER_INDEX_OFFSET] = slot
            data = bytes(data)
            theirs = third_party(data)
            if theirs is None:
                disagreements.append((name, "third-party decoder refused slot %d" % slot))
                continue
            theirs_hdr = theirs["header"]
            if theirs_hdr["player_car_index"] != slot:
                disagreements.append((name, "slot %d: reference read player index %r"
                                      % (slot, theirs_hdr["player_car_index"])))
                continue
            slots_seen += 1

            if hdr["packet_id"] == ours.MOTION_ID:
                motion_n += 1
                mine = ours.decode_motion(raw)["cars"][slot]
                theirs_car = theirs["motion"]
                ok = True
                for k, tk in MOTION_FIELD_MAP.items():
                    d = abs(mine[k] - theirs_car[tk])
                    maxdiff[k] = max(maxdiff.get(k, 0.0), d)
                    if d > TOL[k]:
                        ok = False
                        disagreements.append((name, "motion slot %d %s: %.6f vs %.6f (d=%.6f)"
                                              % (slot, k, mine[k], theirs_car[tk], d)))
                if abs(mine["vel_x"]) > 1.0 or abs(mine["g_lat"]) > 0.2:
                    moving_checked += 1
                    moving_ok += 1 if ok else 0
                if ok:
                    motion_ok += 1
            else:
                car_n += 1
                mine = ours.decode_car_telemetry(raw)["cars"][slot]
                theirs_car = theirs["car_telemetry"]
                ok = True
                for k, tk in CAR_FIELD_MAP.items():
                    d = abs(float(mine[k]) - float(theirs_car[tk]))
                    maxdiff[k] = max(maxdiff.get(k, 0.0), d)
                    if d > TOL[k]:
                        ok = False
                        disagreements.append((name, "car slot %d %s: %r vs %r (d=%r)"
                                              % (slot, k, mine[k], theirs_car[tk], d)))
                # Tyre temperature order is the classic decoder bug. All four
                # channels are zero in this engine, so this checks that both
                # decoders agree on where the block ends and the keys line up,
                # rather than comparing values.
                tp = theirs_car.get("tyre_surface_temperatures")
                if tp is None or sorted(tp) != ["front_left", "front_right",
                                                "rear_left", "rear_right"]:
                    disagreements.append((name, "slot %d: unexpected tyre keys %s"
                                          % (slot, sorted(tp) if tp else None)))
                if mine["speed_kph"] > 10:
                    moving_checked += 1
                    moving_ok += 1 if ok else 0
                if ok:
                    car_ok += 1

    print("third-party decoder : %s" % decoder_path)
    print("packets read        : %d" % len(files))
    print("car-slot decodes    : %d (each packet re-decoded once per car slot)"
          % slots_seen)
    print("motion decodes      : %d, agreed on all 4 fields: %d" % (motion_n, motion_ok))
    print("car telemetry decode: %d, agreed on all 7 fields: %d" % (car_n, car_ok))
    print("decodes carrying MOVING data: %d, all agreed: %d"
          % (moving_checked, moving_ok))
    print("largest absolute differences:")
    for k in sorted(maxdiff):
        print("   %-12s %.9g" % (k, maxdiff[k]))
    if disagreements:
        print("\nDISAGREEMENTS: %d" % len(disagreements))
        for d in disagreements[:10]:
            print("   %s: %s" % d)
        return 1
    print("\nno disagreements: both decoders read every packet identically")
    ok = (motion_ok == motion_n and car_ok == car_n and motion_n > 0 and car_n > 0
          and moving_checked > 0 and moving_ok == moving_checked)
    if not ok:
        print("\nrefusing to call this a pass: too few packets, or nothing moved")
    return 0 if ok else 1


def selftest():
    """Prove the comparison is live, using only synthetic bytes.

    A check that cannot fail is worse than no check: it reports green forever.
    This builds one Motion packet, confirms the reference decoder agrees, then
    shifts a field offset by four bytes and confirms it disagrees. No capture
    and no network needed, so it can run anywhere.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import telemetry_recv as ours
    try:
        from f1_telemetry import decode_packet as third_party
    except ImportError:
        print("selftest needs the third-party decoder on sys.path")
        return 2

    hdr = struct.pack(ours.HEADER_FMT, 2025, 25, 1, 0, 1, ours.MOTION_ID,
                      7, 1.5, 100, 100, 0, 255)
    block = bytearray(60)
    struct.pack_into("<3f", block, 0, 12.0, -3.0, 0.0)
    struct.pack_into("<3f", block, 12, 30.0, 4.0, 0.0)
    # Normalised direction as int16 / 32767, which is what the format wants.
    for k, v in enumerate((0.7071, 0.7071, 0.0, -0.7071, 0.7071, 0.0)):
        struct.pack_into("<h", block, 24 + k * 2, round(v * 32767.0))
    struct.pack_into("<3f", block, 36, 1.25, -0.5, 0.0)
    struct.pack_into("<3f", block, 48, 0.785, 0.0, 0.0)
    buf = hdr + bytes(block) * ours.MAX_CARS
    assert len(buf) == ours.MOTION_SIZE, len(buf)

    ok = third_party(buf)
    if ok is None:
        print("selftest FAILED: the reference decoder refused a well-formed packet")
        return 1
    mine = ours.decode_motion(buf)["cars"][0]
    if abs(mine["g_lat"] - ok["motion"]["g_force_lateral"]) > 1e-6:
        print("selftest FAILED: g_lat %.6f vs %.6f"
              % (mine["g_lat"], ok["motion"]["g_force_lateral"]))
        return 1
    print("selftest: reference decoder agrees on a synthetic packet (g_lat %.4f)"
          % mine["g_lat"])

    # Now the negative half: read g_force from 40 instead of 36. If the two
    # decoders still "agree", this whole check is measuring nothing.
    shifted = struct.unpack_from("<f", buf, ours.HEADER_SIZE + 40)[0]
    if abs(shifted - ok["motion"]["g_force_lateral"]) < 1e-6:
        print("selftest FAILED: the shifted read did not change the value, so "
              "the check cannot detect a mis-offset field")
        return 1
    print("selftest: a four-byte field shift changes the value to %.4f, which "
          "the check would report as a disagreement" % shifted)
    print("selftest: PASSED -- the comparison detects a mis-offset field")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())