#!/usr/bin/env python3
"""Generate the 13 Phase 9 sound cues as validated mono 48 kHz 16-bit WAVs.

The plan says: "Prefer a small validated sound vocabulary before procedural
synthesis." The vocabulary is the 13 cues in src/sound.rs; these are the
assets behind them. Each one is synthesised from a written recipe -- a short
noise burst through a shaped envelope, or a damped tone -- so the set is
reproducible byte-for-byte from this script and auditable line by line.

Every file must pass Rust's `sound::validate_wav`:
  mono, 48 000 Hz, 16-bit PCM, 20-1500 ms, peak 0.05-0.98.
This script checks the same rules itself, so a failure is caught here rather
than in a test.

Usage:  make_sounds.py [--out ../static/sounds]
"""

import argparse
import math
import os
import struct

RATE = 48000
BITS = 16

# ---------------------------------------------------------------------------
# Deterministic noise. A fixed LCG, so re-running gives identical bytes -- a
# cue that changed every generation could not be diffed or cached.
# ---------------------------------------------------------------------------


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 0x7FFFFFFF * 2.0 - 1.0


def noise_burst(ms, seed, low_hz, high_hz, curve, out=None):
    """A band-limited noise burst with a rising/falling envelope.

    `curve` shapes the amplitude ramp: 1.0 is linear, 2.0 quadratic attack.
    The band is a one-pole high-pass followed by a one-pole low-pass, which
    is crude but deterministic and audibly separates the cues.
    """
    n = int(RATE * ms / 1000.0)
    rng = Lcg(seed)
    # One-pole coefficients for the requested corner frequencies.
    hp_a = math.exp(-2.0 * math.pi * low_hz / RATE)
    lp_a = math.exp(-2.0 * math.pi * high_hz / RATE)
    hp_prev = 0.0
    hp_y = 0.0
    lp_y = 0.0
    buf = out if out is not None else [0.0] * n
    for i in range(n):
        x = rng.next()
        hp_y = hp_a * (hp_y + x - hp_prev)
        hp_prev = x
        lp_y = lp_a * lp_y + (1.0 - lp_a) * hp_y
        # Symmetric envelope: fast attack, exponential decay.
        p = i / max(1, n - 1)
        env = math.sin(math.pi * p) ** curve
        buf[i] += lp_y * env
    return buf


def tone(ms, freq, decay_hz, harmonics, seed_offset=0.0):
    """A damped tone: sum of harmonics under an exponential decay envelope."""
    n = int(RATE * ms / 1000.0)
    buf = [0.0] * n
    for i in range(n):
        t = i / RATE
        env = math.exp(-2.0 * math.pi * decay_hz * t)
        s = 0.0
        for k, amp in harmonics:
            s += amp * math.sin(2.0 * math.pi * freq * k * t + seed_offset)
        buf[i] = s * env
    return buf


def silence(ms):
    return [0.0] * int(RATE * ms / 1000.0)


def mix(a, b):
    n = max(len(a), len(b))
    out = [0.0] * n
    for i, v in enumerate(a):
        out[i] += v
    for i, v in enumerate(b):
        out[i] += v
    return out


def seq(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


# ---------------------------------------------------------------------------
# The vocabulary. Order and names match SoundCue in src/sound.rs exactly.
# Each recipe is written for what the cue has to *communicate*:
#   shift      -- a mechanical click, not an engine note
#   limiter    -- a hard, bright buzz
#   tyre slip  -- a mid-band scrub that rises with slip
#   wind       -- a broadband rush
#   race event -- a short identifiable signature, never mush
# ---------------------------------------------------------------------------

def build():
    cues = {}

    # -- continuous cues --------------------------------------------------
    cues["shift_up"] = seq(
        noise_burst(12, 11, 1200, 9000, 1.4),
        noise_burst(45, 12, 300, 3000, 3.0),
    )
    cues["shift_down"] = seq(
        noise_burst(12, 13, 900, 7000, 1.4),
        tone(70, 260.0, 22.0, [(1.0, 1.0), (2.0, 0.25)]),
    )
    cues["rev_limiter"] = seq(
        tone(60, 1800.0, 14.0, [(1.0, 1.0), (2.0, 0.6), (3.0, 0.3)]),
        tone(60, 2400.0, 14.0, [(1.0, 0.5), (2.0, 0.4)]),
    )
    cues["tyre_slip"] = noise_burst(220, 21, 700, 6000, 0.7)
    cues["wind"] = noise_burst(400, 22, 200, 14000, 0.5)

    # -- race events ------------------------------------------------------
    cues["lap_complete"] = seq(
        tone(90, 880.0, 9.0, [(1.0, 1.0), (2.0, 0.3)]),
        tone(140, 1320.0, 8.0, [(1.0, 0.9), (2.0, 0.25)]),
    )
    cues["start_light"] = seq(
        tone(110, 520.0, 11.0, [(1.0, 1.0), (2.0, 0.2)]),
        silence(40),
        tone(140, 520.0, 8.0, [(1.0, 1.0), (2.0, 0.2)]),
    )
    cues["green_flag"] = seq(
        tone(80, 660.0, 12.0, [(1.0, 1.0)]),
        tone(80, 990.0, 12.0, [(1.0, 0.9)]),
        tone(300, 1320.0, 5.0, [(1.0, 1.0), (2.0, 0.3), (3.0, 0.1)]),
    )
    cues["pit_entered"] = seq(
        tone(70, 440.0, 13.0, [(1.0, 1.0), (2.0, 0.2)]),
        silence(30),
        tone(90, 392.0, 12.0, [(1.0, 1.0), (2.0, 0.2)]),
    )
    cues["contact"] = seq(
        noise_burst(35, 31, 80, 4000, 1.0),
        noise_burst(140, 32, 150, 5500, 2.2),
    )
    cues["weather_change"] = seq(
        noise_burst(120, 41, 400, 9000, 0.8),
        noise_burst(160, 42, 250, 7000, 0.8),
    )
    cues["session_finish"] = seq(
        tone(160, 523.0, 5.0, [(1.0, 1.0), (2.0, 0.3)]),
        tone(160, 659.0, 5.0, [(1.0, 1.0), (2.0, 0.3)]),
        tone(160, 784.0, 5.0, [(1.0, 1.0), (2.0, 0.3)]),
        tone(700, 1046.0, 2.2, [(1.0, 1.0), (2.0, 0.35), (3.0, 0.12)]),
    )
    cues["formation_tick"] = seq(
        noise_burst(18, 51, 2000, 12000, 1.2),
        tone(50, 1500.0, 30.0, [(1.0, 0.7), (2.0, 0.2)]),
    )

    return cues


# ---------------------------------------------------------------------------
# WAV writing, and the same validation Rust performs.
# ---------------------------------------------------------------------------


def to_wav(samples):
    """Normalise to a safe peak and write a mono 16-bit PCM WAV."""
    peak = max(abs(v) for v in samples) or 1.0
    # 0.89 is inside Rust's 0.05..=0.98 peak window with headroom on both
    # sides, and leaves the file valid for any other consumer too.
    scale = 0.89 / peak
    frames = b"".join(
        struct.pack("<h", max(-32768, min(32767, int(round(v * scale * 32767)))))
        for v in samples
    )
    data_size = len(frames)
    # Canonical 44-byte RIFF/WAVE header: 8 RIFF + 4 + 8 fmt + 16 fmt +
    # 8 data + 4. "fmt " is four bytes, so this is "WAVE" then "fmt ".
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, BITS)
        + b"data"
        + struct.pack("<I", data_size)
        + frames
    )


def validate(wav, samples):
    """Re-check the exact rules sound::validate_wav enforces."""
    assert wav[0:4] == b"RIFF" and wav[8:12] == b"WAVE", "not a RIFF/WAVE file"
    assert wav[12:16] == b"fmt ", "fmt chunk is not at offset 12"
    # fmt body starts at 16: size, format, channels, rate, byte rate,
    # block align, bits per sample.
    fmt_size, audio_fmt, channels, rate, _br, _ba, bits = struct.unpack_from(
        "<IHHIIHH", wav, 16
    )
    problems = []
    if fmt_size != 16:
        problems.append("fmt chunk is %d bytes, expected 16" % fmt_size)
    if audio_fmt != 1:
        problems.append("format %d is not uncompressed PCM" % audio_fmt)
    if channels != 1:
        problems.append("%d channels, expected mono" % channels)
    if rate != 48000:
        problems.append("%d Hz, expected 48000" % rate)
    if bits != 16:
        problems.append("%d-bit, expected 16" % bits)
    data_size = struct.unpack_from("<I", wav, 40)[0]
    if data_size != len(wav) - 44:
        problems.append("data chunk claims %d bytes, file has %d"
                        % (data_size, len(wav) - 44))
    ms = data_size / 2 / RATE * 1000.0
    if not (20.0 <= ms <= 1500.0):
        problems.append("%.0f ms, outside 20..=1500" % ms)
    # Measure the peak off the encoded int16 samples, not off the source
    # floats: that is the number a player would actually hear, and it is the
    # number Rust reads back out of the file.
    peak = measured_peak(wav)
    if not (0.05 <= peak <= 0.98):
        problems.append("peak %.3f, outside 0.05..=0.98" % peak)
    return problems


def measured_peak(wav):
    data_size = struct.unpack_from("<I", wav, 40)[0]
    frames = struct.unpack_from("<%dh" % (data_size // 2), wav, 44)
    return max(abs(v) for v in frames) / 32768.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "static", "sounds"))
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    cues = build()
    failures = 0
    print("%-16s %8s %8s  %s" % ("cue", "bytes", "ms", "peak"))
    for name in sorted(cues):
        samples = cues[name]
        wav = to_wav(samples)
        problems = validate(wav, samples)
        path = os.path.join(out_dir, name + ".wav")
        with open(path, "wb") as f:
            f.write(wav)
        peak = measured_peak(wav)
        ms = (len(wav) - 44) / 2 / RATE * 1000.0
        status = "OK" if not problems else "FAIL: " + "; ".join(problems)
        if problems:
            failures += 1
        print("%-16s %8d %8.0f  %.3f  %s"
              % (name, len(wav), ms, peak, status))

    print("\n%d cues written to %s" % (len(cues), out_dir))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())