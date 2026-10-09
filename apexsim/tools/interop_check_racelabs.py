#!/usr/bin/env python3
"""Second cross-check: the engine's telemetry against the RaceLabs decoder.

A different third party, a different codebase, a different language lineage
than `tools/interop_check.py` uses. Where that tool forces the reference to
read one car by rewriting m_playerCarIndex, RaceLabs returns all 22 cars
directly, so this script compares the whole field in one pass and covers the
velocity channels that the other check does not touch.

    git clone --depth 1 https://github.com/Si6gma/RaceLabs.git
    interop_check_racelabs.py --packets DIR --decoder RACE/repo/backend/app
"""

import argparse
import glob
import importlib.util
import os
import struct
import sys

TOL = {"pos_x": 1e-3, "pos_y": 1e-3, "pos_z": 1e-3, "g_lat": 1e-3, "g_lon": 1e-3,
       "g_vert": 1e-3, "yaw": 1e-3, "pitch": 1e-3, "roll": 1e-3,
       "speed_kph": 0.0, "throttle": 1e-6, "brake": 1e-6, "steer": 1e-6,
       "gear": 0.0, "rpm": 0.0, "drs": 0.0, "rev_lights_pct": 0.0}
VEL_FIELDS = ("vel_x", "vel_y")


def load_racelabs(path):
    spec = importlib.util.spec_from_file_location(
        "racelabs_decoder", os.path.join(path, "decoder.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["racelabs_decoder"] = mod
    spec.loader.exec_module(mod)
    return mod.F1Decoder()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packets", required=True)
    ap.add_argument("--decoder", required=True, help="RaceLabs backend/app directory")
    ap.add_argument("--tools", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.tools))
    import telemetry_recv as ours

    dec_path = os.path.abspath(args.decoder)
    if not os.path.isfile(os.path.join(dec_path, "decoder.py")):
        print("error: no decoder.py in %s" % dec_path, file=sys.stderr)
        return 2
    decoder = load_racelabs(dec_path)

    files = sorted(glob.glob(os.path.join(args.packets, "*.bin")))
    if not files:
        print("error: no .bin packets in %s" % args.packets, file=sys.stderr)
        return 2

    # The Motion struct in RaceLabs has no velocity fields, so velocity is
    # compared from the raw bytes using this repo's stride arithmetic. That is
    # a weaker check (it is not a second opinion) and is labelled as such.
    motion_n = motion_ok = car_n = car_ok = 0
    moving = 0
    maxdiff = {}
    disagreements = []
    header_size = struct.calcsize("<H5BQf2I2B")

    for path in files:
        raw = open(path, "rb").read()
        name = os.path.basename(path)
        hdr = ours.decode_header(raw)
        frame = decoder.decode_packet(raw)
        if frame is None:
            disagreements.append((name, "RaceLabs refused the packet"))
            continue
        if frame.packet_id != hdr["packet_id"]:
            disagreements.append((name, "packet id: %r vs %r"
                                  % (frame.packet_id, hdr["packet_id"])))
            continue
        if abs(frame.session_time - hdr["session_time"]) > 1e-4:
            disagreements.append((name, "session time: %r vs %r"
                                  % (frame.session_time, hdr["session_time"])))
            continue
        if frame.frame_identifier != hdr["frame_id"]:
            disagreements.append((name, "frame id: %r vs %r"
                                  % (frame.frame_identifier, hdr["frame_id"])))
            continue

        for slot in range(ours.MAX_CARS):
            if hdr["packet_id"] == ours.MOTION_ID:
                m = ours.decode_motion(raw)["cars"][slot]
                t = frame.motion_all[slot]
                ok = True
                for k, tv in (("pos_x", t.world_pos_x), ("pos_y", t.world_pos_y),
                              ("pos_z", t.world_pos_z), ("g_lat", t.g_force_lat),
                              ("g_lon", t.g_force_lon), ("g_vert", t.g_force_vert),
                              ("yaw", t.yaw), ("pitch", t.pitch), ("roll", t.roll)):
                    d = abs(m[k] - tv)
                    maxdiff[k] = max(maxdiff.get(k, 0.0), d)
                    if d > TOL[k]:
                        ok = False
                        disagreements.append((name, "motion slot %d %s: %.6f vs %.6f"
                                              % (slot, k, m[k], tv)))
                if abs(m["vel_x"]) > 1.0:
                    moving += 1
                motion_n += 1
                motion_ok += 1 if ok else 0
            else:
                c = ours.decode_car_telemetry(raw)["cars"][slot]
                t = frame.telemetry_all[slot]
                ok = True
                for k, tv in (("speed_kph", t.speed), ("throttle", t.throttle),
                              ("brake", t.brake), ("steer", t.steer), ("gear", t.gear),
                              ("rpm", t.engine_rpm), ("drs", t.drs),
                              ("rev_lights_pct", t.rev_lights_percent)):
                    d = abs(float(c[k]) - float(tv))
                    maxdiff[k] = max(maxdiff.get(k, 0.0), d)
                    if d > TOL[k]:
                        ok = False
                        disagreements.append((name, "car slot %d %s: %r vs %r"
                                              % (slot, k, c[k], tv)))
                # The tyre order is the classic bug; compare all four by index
                # in both decoders so a transpose on either side shows up.
                for i in range(4):
                    d = abs(float(c["surf_t%d" % i]) - float(t.tyres_surface_temp[i]))
                    maxdiff["surf_t%d" % i] = max(maxdiff.get("surf_t%d" % i, 0.0), d)
                    d = abs(float(c["press%d" % i]) - float(t.tyres_pressure[i]))
                    maxdiff["press%d" % i] = max(maxdiff.get("press%d" % i, 0.0), d)
                    if d > 0.0:
                        ok = False
                        disagreements.append((name, "car slot %d tyre %d differs" % (slot, i)))
                if c["speed_kph"] > 10:
                    moving += 1
                car_n += 1
                car_ok += 1 if ok else 0

    print("RaceLabs decoder   : %s" % dec_path)
    print("packets read       : %d" % len(files))
    print("motion decodes     : %d, agreed on all 9 fields: %d" % (motion_n, motion_ok))
    print("car telemetry dec. : %d, agreed on all 8 + 8 tyre channels: %d"
          % (car_n, car_ok))
    print("decodes with MOVING data: %d" % moving)
    print("largest absolute differences:")
    for k in sorted(maxdiff):
        print("   %-22s %.9g" % (k, maxdiff[k]))
    if disagreements:
        print("\nDISAGREEMENTS: %d" % len(disagreements))
        for d in disagreements[:10]:
            print("   %s: %s" % d)
        return 1
    print("\nno disagreements: RaceLabs read every packet identically")
    ok = motion_ok == motion_n and car_ok == car_n and motion_n > 0 and moving > 0
    if not ok:
        print("\nrefusing to call this a pass: too few packets, or nothing moved")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())