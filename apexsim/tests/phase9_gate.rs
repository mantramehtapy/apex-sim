//! Phase 9 completion gates, mapped one-to-one to the plan's wording.
//!
//! The plan's gate is three bullets:
//!
//!   a. "Packet capture matches the selected protocol and is consumed by the
//!      target."
//!   b. "A disconnected receiver has no effect on the race."
//!   c. "Audio events remain synchronized during pause, replay, and
//!      accelerated mode."
//!
//! and its build items are: choose one exact protocol and version; map Rust
//! state into it with units, rates, packet headers and car indexing confirmed;
//! begin with one receiver and test the real thing; add audio for RPM, tyre
//! slip and wind from a validated vocabulary rather than by synthesis.
//!
//! Two of these need care, because the obvious assertion is a proxy that
//! passes for the wrong reason:
//!
//! * **(a)** Decoding a packet with the encoder's own helpers proves only that
//!   the encoder agrees with itself. The decoder below is written against the
//!   field table in `docs/f1_2025_udp_spec.md` and shares no code with
//!   `telemetry.rs`, and the test additionally pins the packet *sizes* and the
//!   *representation* of every field type — a float written where the format
//!   wants an int16 changes no size and no field count, which is exactly how
//!   this encoder shipped a real bug. Interop against three third-party
//!   decoders is done out-of-band by `tools/interop_check*.py|js`; that
//!   evidence is recorded in the spec doc, and this file is what keeps the
//!   bytes honest between those runs.
//! * **(b)** The strong form is not "no crash". It is that a race with a dead
//!   receiver hashes *identically* to the same race with no telemetry at all,
//!   which is only true because nothing on `Race` is reachable from
//!   `mix_state`. The test also asserts the failure is *visible*
//!   (`send_errors`), so it cannot pass by having silently published nothing.
//! * **(c)** "Synchronized" is three different claims, and each needs its own
//!   test: paused produces no cues and no lost ordering; a replay reproduces
//!   the cue log tick for tick; and an accelerated run still delivers cues in
//!   order while shedding the least important ones instead of queueing them.

use apexsim::race::{Race, RaceConfig, RacePhase};
use apexsim::sound::{
    self, AudioConsumer, SoundConfig, SoundCue, SoundRuntime, ALL_CUES, BACKLOG_LIMIT, CUE_COUNT,
};
use apexsim::telemetry::{
    self, CarSample, Gearbox, LinkStats, PacketHeader, TelemetryConfig, TelemetryLink,
    CAR_TELEMETRY_PACKET_SIZE, HEADER_SIZE, MAX_CARS, MOTION_PACKET_SIZE, PACKET_ID_CAR_TELEMETRY,
    PACKET_ID_MOTION,
};
use std::collections::HashSet;
use std::net::UdpSocket;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn tmp_log(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("f1_p9_{}_{}.log", name, std::process::id()));
    p
}

// ---------------------------------------------------------------------------
// An independent decoder.
//
// Deliberately written against the field table in docs/f1_2025_udp_spec.md
// with its own byte reads, so a mistake in the encoder cannot hide behind a
// matching mistake in the decoder. It shares no function with telemetry.rs.
// ---------------------------------------------------------------------------

fn i16le(b: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([b[off], b[off + 1]])
}
fn u16le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}
fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn u64le(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}
fn f32le(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// The common 29-byte header.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Header {
    m_packet_format: u16,
    m_game_year: u8,
    m_packet_version: u8,
    m_packet_id: u8,
    m_session_uid: u64,
    m_session_time: f32,
    m_frame_id: u32,
    m_overall_frame_id: u32,
    m_player_car_index: u8,
    m_secondary_player_car_index: u8,
}

impl Header {
    fn parse(b: &[u8]) -> Header {
        assert!(b.len() >= HEADER_SIZE, "packet shorter than the header");
        Header {
            m_packet_format: u16le(b, 0),
            m_game_year: b[2],
            m_packet_version: b[5],
            m_packet_id: b[6],
            m_session_uid: u64le(b, 7),
            m_session_time: f32le(b, 15),
            m_frame_id: u32le(b, 19),
            m_overall_frame_id: u32le(b, 23),
            m_player_car_index: b[27],
            m_secondary_player_car_index: b[28],
        }
    }
}

/// One 60-byte CarMotionData block, decoded.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Motion {
    pos: [f32; 3],
    vel: [f32; 3],
    /// Normalised int16 directions, as sent.
    fwd_i16: [i16; 3],
    right_i16: [i16; 3],
    /// The same channels as a receiver reads them: int16 / 32767.0.
    fwd: [f32; 3],
    right: [f32; 3],
    g: [f32; 3],
    angles: [f32; 3],
}

impl Motion {
    fn parse(b: &[u8], slot: usize) -> Motion {
        let base = HEADER_SIZE + slot * 60;
        let mut fwd_i16 = [0i16; 3];
        let mut right_i16 = [0i16; 3];
        for k in 0..3 {
            fwd_i16[k] = i16le(b, base + 24 + k * 2);
            right_i16[k] = i16le(b, base + 30 + k * 2);
        }
        let scale = |v: i16| v as f32 / 32767.0;
        Motion {
            pos: [f32le(b, base), f32le(b, base + 4), f32le(b, base + 8)],
            vel: [f32le(b, base + 12), f32le(b, base + 16), f32le(b, base + 20)],
            fwd_i16,
            right_i16,
            fwd: fwd_i16.map(scale),
            right: right_i16.map(scale),
            g: [f32le(b, base + 36), f32le(b, base + 40), f32le(b, base + 44)],
            angles: [f32le(b, base + 48), f32le(b, base + 52), f32le(b, base + 56)],
        }
    }
}

/// One 60-byte CarTelemetryData block, decoded.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CarBlock {
    speed_kph: u16,
    throttle: f32,
    steer: f32,
    brake: f32,
    gear: i8,
    rpm: u16,
    drs: u8,
    rev_lights_pct: u8,
}

impl CarBlock {
    fn parse(b: &[u8], slot: usize) -> CarBlock {
        let base = HEADER_SIZE + slot * 60;
        CarBlock {
            speed_kph: u16le(b, base),
            throttle: f32le(b, base + 2),
            steer: f32le(b, base + 6),
            brake: f32le(b, base + 10),
            gear: b[base + 15] as i8,
            rpm: u16le(b, base + 16),
            drs: b[base + 18],
            rev_lights_pct: b[base + 19],
        }
    }
}

fn samples(n: usize) -> Vec<CarSample> {
    (0..n)
        .map(|i| {
            let f = i as f64;
            CarSample {
                pos_x: 10.0 + f,
                pos_y: -4.0 + 0.5 * f,
                heading: 0.3 * f,
                vel_x: 40.0 + f,
                vel_y: -2.0,
                vel_long: 70.0 + f,
                g_lat: 1.2,
                g_lon: 0.4,
                throttle: 0.8,
                brake: 0.05,
                steer: -0.25,
                gear: 5,
                engine_rpm: 11_000,
                rev_lights_percent: 60,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Build item 1 & 2 — one exact protocol and version, mapped with units,
// rates, headers and car indexing confirmed.
// ---------------------------------------------------------------------------

#[test]
fn build1_the_protocol_is_one_exact_named_format_and_version() {
    // "Choose one exact target telemetry protocol and version" — a constant
    // the CLI, the log and the doc all read, so there is no second opinion
    // about what is being spoken.
    assert_eq!(telemetry::PACKET_FORMAT, 2025, "the format version is part of the contract");
    assert_eq!(telemetry::GAME_YEAR, 25);
    // Published packet sizes fall out of the header and the strides only if
    // every stride is right, which is why they are asserted as literals
    // rather than recomputed from the same constants that built them.
    assert_eq!(MOTION_PACKET_SIZE, 1349);
    assert_eq!(CAR_TELEMETRY_PACKET_SIZE, 1352);
    assert_eq!(HEADER_SIZE + MAX_CARS * 60, 1349);
}

#[test]
fn build2_units_rates_headers_and_car_indexing_are_what_the_format_says() {
    let cars = samples(4);
    let hdr = PacketHeader {
        session_uid: 0x1122_3344_5566_7788,
        session_time: 12.5,
        frame_id: 77,
        overall_frame_id: 78,
        player_car_index: 2,
    };

    // ---- Motion -------------------------------------------------------
    let mut buf = vec![0u8; MOTION_PACKET_SIZE];
    let n = telemetry::encode_motion(&mut buf, &hdr, &cars);
    assert_eq!(n, 4);
    assert_eq!(buf.len(), 1349);

    let h = Header::parse(&buf);
    assert_eq!(h.m_packet_format, 2025);
    assert_eq!(h.m_game_year, 25);
    assert_eq!(h.m_packet_version, 1);
    assert_eq!(h.m_packet_id, PACKET_ID_MOTION);
    assert_eq!(h.m_session_uid, 0x1122_3344_5566_7788, "uid is a u64 little-endian");
    assert_eq!(h.m_session_time, 12.5, "session time is f32 seconds");
    assert_eq!(h.m_frame_id, 77);
    assert_eq!(h.m_overall_frame_id, 78);
    assert_eq!(h.m_player_car_index, 2);
    assert_eq!(
        h.m_secondary_player_car_index, 255,
        "a receiver with no second player expects 255"
    );

    // Car indexing: slot i holds car i, regardless of which slot the header
    // calls the player. A receiver that indexed by player_car_index would put
    // car 2 in slot 0.
    for (slot, &c) in cars.iter().take(4).enumerate() {
        let m = Motion::parse(&buf, slot);
        assert!((m.pos[0] - c.pos_x as f32).abs() < 0.01, "slot {slot} x");
        assert!((m.pos[1] - c.pos_y as f32).abs() < 0.01, "slot {slot} y");
        assert!((m.vel[0] - c.vel_x as f32).abs() < 0.01, "slot {slot} vel x");
        assert_eq!(m.pos[2], 0.0, "this engine is planar: z is zero");
        assert!((m.g[0] - c.g_lat as f32).abs() < 0.01, "lateral g is in g, not m/s^2");
        assert!((m.angles[0] - c.heading as f32).abs() < 0.001);
    }
    // Unoccupied slots are all zero, which reads unambiguously as "no car".
    for slot in 4..MAX_CARS {
        let base = HEADER_SIZE + slot * 60;
        assert!(
            buf[base..base + 60].iter().all(|b| *b == 0),
            "empty slot {slot} must be a zero block"
        );
    }

    // The six direction channels are **int16 normalised**, not float32. This
    // is the bug this repository shipped once: a float here leaves the packet
    // exactly the right size with exactly the right field count, so nothing
    // structural catches it, and the receiver reads the low two bytes of a
    // float as an integer that was never a direction.
    let (sin_h, cos_h) = cars[1].heading.sin_cos();
    let m1 = Motion::parse(&buf, 1);
    assert_eq!(
        m1.fwd_i16[0],
        (cos_h * 32767.0).round() as i16,
        "forward x must be int16, divided by 32767 by the receiver"
    );
    assert_eq!(m1.fwd_i16[1], (sin_h * 32767.0).round() as i16);
    assert_eq!(m1.fwd_i16[2], 0);
    assert_eq!(m1.right_i16[0], (-sin_h * 32767.0).round() as i16);
    assert_eq!(m1.right_i16[1], (cos_h * 32767.0).round() as i16);
    // And the values must be directions, not integers: |v| <= 1 after scaling.
    for v in m1.fwd.iter().chain(m1.right.iter()) {
        assert!(v.abs() <= 1.0 + 1e-6, "scaled direction out of range: {v}");
    }
    // Negative control for the three equalities above. They are only worth
    // anything if the *offset* is what makes them pass: one slot over is the
    // world-right direction, which must NOT satisfy the forward equations. If
    // this ever passed, the forward check could be satisfied by accident and
    // would be reporting nothing.
    //
    // (Reading these bytes as a float32 instead would not discriminate: a
    // normalised int16 pair almost always decodes to a denormal near zero,
    // which looks just as plausible as a direction.)
    let at_offset_plus_2: Vec<i16> = (0..3)
        .map(|k| i16le(&buf, HEADER_SIZE + 60 + 24 + k * 2 + 2))
        .collect();
    assert_ne!(
        at_offset_plus_2,
        vec![
            (cos_h * 32767.0).round() as i16,
            (sin_h * 32767.0).round() as i16,
            0
        ],
        "the world-right channels also satisfy the forward equations — the \
         forward assertion above has no teeth"
    );

    // ---- Car Telemetry -------------------------------------------------
    let mut cbuf = vec![0u8; CAR_TELEMETRY_PACKET_SIZE];
    let n = telemetry::encode_car_telemetry(&mut cbuf, &hdr, &cars);
    assert_eq!(n, 4);
    assert_eq!(cbuf.len(), 1352);
    let ch = Header::parse(&cbuf);
    assert_eq!(ch.m_packet_id, PACKET_ID_CAR_TELEMETRY);
    assert_eq!(ch.m_session_uid, h.m_session_uid, "both packets share one header");

    for (slot, &c) in cars.iter().take(4).enumerate() {
        let b = CarBlock::parse(&cbuf, slot);
        // The one unit in this format that is not SI: km/h, not m/s.
        let expect_kph = (c.vel_long * 3.6) as u16;
        assert_eq!(b.speed_kph, expect_kph, "slot {slot}: m_speed must be km/h");
        assert!((b.throttle - c.throttle as f32).abs() < 0.001);
        assert!((b.steer - c.steer as f32).abs() < 0.001, "steer is normalised -1..1");
        assert!((b.brake - c.brake as f32).abs() < 0.001);
        assert_eq!(b.gear, 5);
        assert_eq!(b.rpm, 11_000);
        assert_eq!(b.drs, 0);
        assert_eq!(b.rev_lights_pct, 60);
    }

    // Trailing mfd block: two 255 panel indices and a suggested gear.
    let t = HEADER_SIZE + MAX_CARS * 60;
    assert_eq!(cbuf[t], 255);
    assert_eq!(cbuf[t + 1], 255);
    assert_eq!(cbuf[t + 2] as i8, 5);

    // Rates: the publish interval is derived from the configured Hz against
    // the fixed step, and never degenerates to every tick or never.
    for hz in [1u32, 30, 60, 250] {
        let cfg = TelemetryConfig { hz, ..TelemetryConfig::default() };
        let n = cfg.interval_ticks(apexsim::race::DT);
        assert!(n >= 1, "{hz} Hz must publish at least once per tick interval");
        let effective = 1.0 / (n as f64 * apexsim::race::DT);
        assert!(
            (effective - hz as f64).abs() <= hz as f64 * 0.15 + 0.5,
            "{hz} Hz requested, {effective:.1} Hz delivered"
        );
    }
    assert_eq!(
        TelemetryConfig { hz: 0, ..TelemetryConfig::default() }.interval_ticks(apexsim::race::DT),
        250,
        "a 0 Hz request must clamp to the slowest rate, not divide by zero or publish every tick"
    );
    assert_eq!(telemetry::speed_kph(100.0), 360);
    assert_eq!(telemetry::speed_kph(-5.0), 0, "reverse is not negative km/h");
}

// ---------------------------------------------------------------------------
// Gate (a) — a real capture off the loopback wire, consumed by an independent
// decoder, from a race that actually ran.
// ---------------------------------------------------------------------------

/// Bind the capture socket with a large receive buffer.
///
/// This is the difference between a test that passes and one that passes only
/// on an idle machine. A max-speed race publishes thousands of datagrams in
/// milliseconds; the default Windows receive buffer holds roughly forty, and
/// everything past it is dropped by the *kernel* — silently, and before this
/// code ever runs. Under `cargo test` the machine is also running eleven other
/// test binaries, so the reading thread may not be scheduled at all during the
/// burst. Neither is a property of the packet format, and both used to decide
/// whether this gate passed.
fn bind_wide(port: u16) -> UdpSocket {
    let plain = UdpSocket::bind(("127.0.0.1", port)).expect("bind capture socket");
    let sock = socket2::Socket::from(plain);
    let want = 8 * 1024 * 1024;
    if let Err(e) = sock.set_recv_buffer_size(want) {
        panic!("could not raise SO_RCVBUF to {want}: {e}");
    }
    let got = sock.recv_buffer_size().expect("read back SO_RCVBUF");
    // Windows silently caps this and reports success, so asking is not proof.
    // A buffer below ~1 MB cannot hold the burst this test needs to capture.
    assert!(
        got >= 1024 * 1024,
        "SO_RCVBUF was capped at {got} bytes; the capture would drop the burst \
         before the test could read it"
    );
    let out: UdpSocket = sock.into();
    out
}

/// Receive packets until `want_moving` Motion packets contain a moving car, or
/// until `deadline`.
///
/// The field is stationary through the formation lap, so a capture that
/// stopped on a packet count alone would end before the race starts and prove
/// nothing. Every packet received is kept, not just the last window: at max
/// speed the loopback socket buffer overflows and the receiver — like any real
/// one — misses datagrams. That is a property of the capture, not of the
/// publisher, and the assertions below account for it.
fn capture(port: u16, want_moving: usize, deadline: Duration) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let sock = bind_wide(port);
    sock.set_read_timeout(Some(Duration::from_millis(200)))
        .unwrap();
    let (mut motion, mut car) = (Vec::new(), Vec::new());
    let (mut moving, mut total) = (0usize, 0usize);
    let end = Instant::now() + deadline;
    // Motion and Car Telemetry leave on two independent sender threads, so the
    // two kinds interleave arbitrarily and one consistently trails the other.
    // Stopping the instant the moving-packet target is hit strands the trailing
    // kind — the capture then holds many motion frames and none of their car
    // partners, and the cross-packet check below compares nothing. So once the
    // target is met, keep reading until the wire goes quiet.
    const QUIET: Duration = Duration::from_millis(300);
    let mut last_rx = Instant::now();
    loop {
        if Instant::now() >= end || total >= 4000 {
            break;
        }
        if moving >= want_moving && last_rx.elapsed() >= QUIET {
            break;
        }
        let mut buf = vec![0u8; 2048];
        let Ok((n, _)) = sock.recv_from(&mut buf) else {
            continue;
        };
        last_rx = Instant::now();
        buf.truncate(n);
        total += 1;
        match buf.get(6).copied() {
            Some(PACKET_ID_MOTION) => {
                for slot in 0..6 {
                    let m = Motion::parse(&buf, slot);
                    if (m.vel[0] * m.vel[0] + m.vel[1] * m.vel[1]).sqrt() > 0.5 {
                        moving += 1;
                        break;
                    }
                }
                motion.push(buf);
            }
            Some(PACKET_ID_CAR_TELEMETRY) => car.push(buf),
            Some(other) => panic!("unexpected packet id {other}"),
            None => panic!("empty datagram"),
        }
    }
    (motion, car)
}

#[test]
fn gatea_a_real_capture_off_the_wire_decodes_as_the_documented_format() {
    let probe = UdpSocket::bind(("127.0.0.1", 0)).expect("probe");
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let cfg = TelemetryConfig {
        target: format!("127.0.0.1:{port}"),
        hz: 60,
        player_car: 0,
        session_uid: 0xDEAD_BEEF_CAFE_0001,
        gearbox: Gearbox::default(),
    };
    let race_cfg = RaceConfig {
        cars: 6,
        laps: 2,
        seed: 5,
        telemetry: Some(cfg),
        sound: Some(SoundConfig::default()),
        ..RaceConfig::default()
    };

    let rx = std::thread::spawn(move || capture(port, 150, Duration::from_secs(30)));
    let (outcome, _inputs) = apexsim::race::run_race(race_cfg, false);
    let (motion, car) = rx.join().expect("capture thread");

    // A capture that never arrived, or that only saw the stationary formation
    // lap, would make every assertion below vacuous — so both are checked
    // before anything is believed.
    println!("captured {} motion and {} car packets", motion.len(), car.len());
    assert!(
        motion.len() >= 60 && car.len() >= 60,
        "capture too small: {} motion, {} car packets",
        motion.len(),
        car.len()
    );

    // Every packet is exactly the published size for its id.
    for p in motion.iter().chain(car.iter()) {
        let want = match p[6] {
            PACKET_ID_MOTION => MOTION_PACKET_SIZE,
            PACKET_ID_CAR_TELEMETRY => CAR_TELEMETRY_PACKET_SIZE,
            other => panic!("unknown packet id {other}"),
        };
        assert_eq!(p.len(), want, "packet id {} wrong length", p[6]);
    }

    // One session, one header shape, monotonic frames and time. This is what
    // a real display checks before it draws anything.
    let uid = u64le(&motion[0], 7);
    assert_eq!(uid, 0xDEAD_BEEF_CAFE_0001);
    let mut prev_frame = 0u32;
    let mut prev_time = -1.0f32;
    let mut moving = 0usize;
    for p in &motion {
        let h = Header::parse(p);
        assert_eq!(h.m_session_uid, uid);
        assert_eq!(h.m_packet_format, 2025);
        assert_eq!(h.m_player_car_index, 0);
        assert!(h.m_frame_id > prev_frame, "frame ids must advance: {} after {prev_frame}", h.m_frame_id);
        assert!(h.m_session_time >= prev_time, "session time must not go backwards");
        prev_frame = h.m_frame_id;
        prev_time = h.m_session_time;
        for slot in 0..6 {
            let m = Motion::parse(p, slot);
            if m.vel[0].abs() > 0.5 {
                moving += 1;
                // Directions are unit-ish while the car is actually moving —
                // the check that a float-in-an-int16-field would fail.
                let len = (m.fwd[0] * m.fwd[0] + m.fwd[1] * m.fwd[1]).sqrt();
                assert!(
                    (len - 1.0).abs() < 0.01,
                    "forward vector is not a unit direction (len {len})"
                );
                assert!(m.g[0].abs() < 6.0, "lateral g implausible: {}", m.g[0]);
            }
        }
    }
    assert!(
        moving > 100,
        "only {moving} moving-car slots in the capture — the gate would prove little"
    );

    // Car telemetry agrees with motion for the same frame: two packets, one
    // frame, one truth. This is a cross-packet check, so it cannot pass if
    // both packets are self-consistently wrong in the same way. Indexed by
    // frame id rather than by arrival order, because the two packet kinds come
    // off two independent sender threads and interleave arbitrarily.
    let car_by_frame: std::collections::HashMap<u32, &Vec<u8>> =
        car.iter().map(|c| (Header::parse(c).m_frame_id, c)).collect();
    let mut matched = 0usize;
    for m in &motion {
        let Some(c) = car_by_frame.get(&Header::parse(m).m_frame_id) else {
            continue;
        };
        for slot in 0..6 {
            let mo = Motion::parse(m, slot);
            let cb = CarBlock::parse(c, slot);
            let from_vel = (mo.vel[0] * mo.vel[0] + mo.vel[1] * mo.vel[1]).sqrt();
            let expect = (from_vel * 3.6) as u16;
            if cb.speed_kph == 0 && expect == 0 {
                continue;
            }
            assert!(
                (cb.speed_kph as i32 - expect as i32).abs() <= 2,
                "slot {slot}: car packet says {} km/h, motion packet implies {expect}",
                cb.speed_kph
            );
            matched += 1;
        }
    }
    // Loopback drops datagrams the receiver did not read in time — at max
    // speed the publisher outruns any consumer — so how many pairs survive is
    // a property of the capture, not of the format. The threshold is a
    // vacuity guard only: the point of the check is that every pair that *was*
    // captured agrees, which the asserts inside the loop enforce.
    let pairs = motion
        .iter()
        .filter(|m| car_by_frame.contains_key(&Header::parse(m).m_frame_id))
        .count();
    println!("cross-checked {matched} car slots across {pairs} frame pairs");
    assert!(pairs >= 10, "only {pairs} frame pairs survived the capture");
    assert!(matched >= pairs, "each pair contributes at least one slot");

    // And the publisher's own counters agree that the wire was used.
    let st = outcome.telemetry.expect("telemetry was configured");
    assert!(
        st.sent + st.dropped <= st.offered,
        "sent {} + dropped {} exceeds offered {}",
        st.sent,
        st.dropped,
        st.offered
    );
    // The outcome snapshot is taken while the sender threads may still be
    // flushing, so `sent` and `bytes` legitimately lag the wire by the
    // in-flight tail. The allowance is the publisher's own bound on that
    // tail: one bounded queue per packet kind, sized to a whole packet.
    let captured = (motion.len() + car.len()) as u64;
    let tail_bytes = 2 * telemetry::POOL_DEPTH as u64 * CAR_TELEMETRY_PACKET_SIZE as u64;
    let tail_packets = 2 * telemetry::POOL_DEPTH as u64 * 6;
    assert!(
        captured <= st.sent + tail_packets,
        "captured {captured} packets but the publisher counted only {} sent; \
         the gap exceeds the {tail_packets} datagrams a flushing sender can hold",
        st.sent
    );
    // The wire was genuinely used — not a publisher that silently sent nothing.
    assert!(
        captured >= 100 && st.sent >= 100,
        "only {captured} packets on the wire ({} sent): the gate would prove little",
        st.sent
    );
    // Bytes and packets must agree: every datagram this format sends is one of
    // two fixed sizes, so the captured bytes must match what the capture
    // holds, allowing for the same in-flight tail.
    let bytes_captured = motion.iter().map(|m| m.len() as u64).sum::<u64>()
        + car.iter().map(|c| c.len() as u64).sum::<u64>();
    assert!(
        st.bytes + tail_bytes >= bytes_captured,
        "{} bytes counted but {bytes_captured} bytes were captured, beyond the \
         {tail_bytes}-byte tail a flushing sender can hold",
        st.bytes
    );
    // And the counter must still be in the right neighbourhood: a publisher
    // that counted one byte per datagram would sail through the check above.
    assert!(
        st.bytes >= st.sent * MOTION_PACKET_SIZE as u64 / 2,
        "byte counter implausible: {} bytes for {} datagrams",
        st.bytes,
        st.sent
    );
}

// ---------------------------------------------------------------------------
// Gate (b) — a disconnected receiver has no effect on the race.
// ---------------------------------------------------------------------------

fn base_cfg(seed: u64) -> RaceConfig {
    RaceConfig { cars: 6, laps: 2, seed, ..RaceConfig::default() }
}

#[test]
fn gateb_a_disconnected_receiver_cannot_change_the_race() {
    // Find a UDP port with nothing on it.
    let probe = UdpSocket::bind(("127.0.0.1", 0)).expect("probe");
    let dead_port = probe.local_addr().unwrap().port();
    drop(probe);

    let with_dead_receiver = RaceConfig {
        telemetry: Some(TelemetryConfig {
            target: format!("127.0.0.1:{dead_port}"),
            hz: 60,
            ..TelemetryConfig::default()
        }),
        sound: Some(SoundConfig::default()),
        ..base_cfg(11)
    };
    let without = RaceConfig { sound: Some(SoundConfig::default()), ..base_cfg(11) };

    let t0 = Instant::now();
    let (dead, _) = apexsim::race::run_race(with_dead_receiver, false);
    let dead_elapsed = t0.elapsed();
    let (none, _) = apexsim::race::run_race(without, false);

    // The strong claim: not "it did not crash", but "it is the same race".
    assert_eq!(
        dead.state_hash, none.state_hash,
        "a receiver that is not there changed the trajectory"
    );
    assert_eq!(dead.ticks, none.ticks);
    // LeaderRow has no PartialEq, so the classification is compared the way a
    // results sheet is read: id, position and laps done, in order.
    let order = |o: &apexsim::race::RaceOutcome| -> Vec<(u32, u32, u32)> {
        o.classification.iter().map(|r| (r.id, r.pos, r.laps_done)).collect()
    };
    assert_eq!(
        order(&dead),
        order(&none),
        "the finishing order changed with no receiver attached"
    );
    let (dl, nl) = (&dead.leader_lap_times, &none.leader_lap_times);
    assert_eq!(dl.len(), nl.len());
    for (i, (a, b)) in dl.iter().zip(nl.iter()).enumerate() {
        assert_eq!(a, b, "leader lap {i} differs with a dead receiver attached");
    }
    // Sound is an observer too, and must not leak into the race either.
    assert_eq!(
        dead.sound.as_ref().map(|s| s.log_hash),
        none.sound.as_ref().map(|s| s.log_hash),
        "the cue log changed because telemetry was also on"
    );

    // The failure is *visible*. Without this the test could pass by having
    // silently published nothing at all, which is a different bug.
    let st = dead.telemetry.expect("a link was opened");
    assert!(st.offered > 0, "nothing was ever offered — the test proves nothing");
    assert!(st.sent > 0, "the socket accepted nothing at all");
    // The sender threads outlive the race by a few milliseconds, so the
    // snapshot can be taken with datagrams still in the channel: not yet sent,
    // not yet dropped. The invariant that holds at any instant is that nothing
    // was invented — sent and dropped together never exceed what was offered.
    assert!(
        st.sent + st.dropped <= st.offered,
        "sent {} + dropped {} exceeds offered {}",
        st.sent,
        st.dropped,
        st.offered
    );
    // A max-speed race generates far more than 60 Hz of wall-clock traffic, so
    // drops here are the bounded queue doing its job, not a fault. What must
    // hold is that they are counted and that they never cost the race time.
    if st.dropped > 0 {
        println!("{} of {} datagrams dropped under back-pressure (race unaffected)", st.dropped, st.offered);
    }

    // And the race still finished at max speed. A publisher that blocked on a
    // dead peer would show up here as an enormous wall-clock cost; the whole
    // race is a couple of seconds of simulation at thousands of ticks.
    let sim_s = dead.ticks as f64 * apexsim::race::DT;
    assert!(
        sim_s > 30.0,
        "sanity: the race should be long enough for this to mean something ({sim_s:.0} s)"
    );
    assert!(
        dead_elapsed.as_secs_f64() < sim_s * 0.5,
        "the race took {dead_elapsed:?} for {sim_s:.0} s of simulation — something waited on the socket"
    );
    assert!(dead.phase_finished(), "the race did not run to the flag: {:?}", dead.classification);
}

#[test]
fn gateb_a_destination_this_process_cannot_use_is_reported_not_silently_ignored() {
    // A target that is not host:port, and a port the OS will not hand out,
    // are configuration errors: they must surface, and the race must still
    // run. A silently-ignored bad target is how a Phase 9 feature rots.
    for bad in ["not-an-address", "127.0.0.1:0", "127.0.0.1:99999", "127.0.0.1:"] {
        let r = TelemetryConfig::parse_target(bad);
        assert!(r.is_err(), "'{bad}' must be rejected, got {r:?}");
    }
    assert!(TelemetryConfig::parse_target("127.0.0.1:20777").is_ok());
    assert!(TelemetryConfig::parse_target(" 192.168.1.5:20777 ").is_ok(), "surrounding space is trimmed");

    // An unresolvable host fails at spawn, and the race proceeds anyway.
    let cfg = RaceConfig {
        telemetry: Some(TelemetryConfig {
            target: "no-such-host.invalid:20777".into(),
            ..TelemetryConfig::default()
        }),
        ..base_cfg(3)
    };
    let (outcome, _) = apexsim::race::run_race(cfg, false);
    assert!(outcome.telemetry_error.is_some(), "an unusable target must be reported");
    assert!(outcome.phase_finished(), "the race still has to run");
    let (clean, _) = apexsim::race::run_race(base_cfg(3), false);
    assert_eq!(outcome.state_hash, clean.state_hash, "and still has to be the same race");
}

// ---------------------------------------------------------------------------
// Gate (c) — audio events remain synchronized during pause, replay, and
// accelerated mode.
// ---------------------------------------------------------------------------

#[test]
fn gatec_pausing_the_race_produces_no_cues_and_loses_no_ordering() {
    let mut race = Race::new(RaceConfig { sound: Some(SoundConfig::default()), ..base_cfg(9) });
    race.start();

    // Run into the race proper, past the lights.
    while race.state.phase == RacePhase::Formation {
        race.tick();
    }
    let green_tick = race.state.tick;
    for _ in 0..1500 {
        race.tick();
    }
    let rt = race.sound_runtime().expect("sound enabled");
    assert!(!rt.events.is_empty(), "the pause test needs cues to be at stake");
    let seen_before_pause = rt.events.len();

    // Paused: the server stops calling tick(), which is the whole of pause.
    // Nothing is extracted, nothing drifts, and the log does not advance.
    let hash_at_pause = rt.log_hash();
    let last_tick = rt.events.last().expect("non-empty").tick;
    let tick_at_pause = race.state.tick;
    std::thread::sleep(Duration::from_millis(400));
    let rt = race.sound_runtime().expect("sound enabled");
    assert_eq!(rt.events.len(), seen_before_pause, "a paused race emitted a cue");
    assert_eq!(rt.log_hash(), hash_at_pause, "a paused race changed its cue log");
    assert_eq!(race.state.tick, tick_at_pause, "a paused race advanced");
    assert!(rt.events.last().unwrap().tick <= last_tick, "a cue was dated into the future");

    // Resumed: cues continue from where they stopped, in tick order, and the
    // final log is exactly the log of an identical run that was never paused.
    for _ in 0..1500 {
        race.tick();
    }
    let resumed = race.sound_runtime().expect("sound enabled");
    assert!(resumed.events.len() > seen_before_pause, "resuming produced no new cues");
    assert_monotonic(&resumed.events);

    // The comparison run must cover exactly the same ticks — a race stopped at
    // a different tick has a different log for uninteresting reasons.
    let mut unpaused = Race::new(RaceConfig {
        sound: Some(SoundConfig::default()),
        ..base_cfg(9)
    });
    unpaused.start();
    while unpaused.state.phase == RacePhase::Formation {
        unpaused.tick();
    }
    while unpaused.state.tick < race.state.tick {
        unpaused.tick();
    }
    assert_eq!(unpaused.state.tick, race.state.tick);
    let straight = unpaused.sound_runtime().expect("sound enabled");
    assert_eq!(
        resumed.log_hash(),
        straight.log_hash(),
        "a pause changed the cue log: pausing must not be observable in the sound"
    );
    assert!(green_tick < race.state.tick);
}

fn assert_monotonic(events: &[sound::CueEvent]) {
    for w in events.windows(2) {
        assert!(
            w[1].tick >= w[0].tick,
            "cue log out of order: tick {} then {}",
            w[0].tick,
            w[1].tick
        );
    }
}

#[test]
fn gatec_a_replay_reproduces_the_cue_log_tick_for_tick() {
    let path = tmp_log("replay");
    let cfg = RaceConfig { sound: Some(SoundConfig::default()), ..base_cfg(21) };
    let (live, inputs) = apexsim::race::run_race(cfg.clone(), false);

    let live_events = live.sound_events.as_ref().expect("a cue log");
    assert!(!live_events.is_empty(), "the replay gate needs cues to compare");
    let live_sum = live.sound.as_ref().expect("sound summary");
    assert_eq!(live_sum.events, live_events.len());

    apexsim::race::write_race_log(&path, &cfg, &inputs, &live).expect("write log");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\nT "), "the log carries no telemetry/sound line");
    assert_eq!(
        text.lines().filter(|l| l.starts_with("E ")).count(),
        live_events.len(),
        "the log must record every cue, not a summary"
    );

    let (ok, replayed) = apexsim::race::replay_race_log(&path).expect("replay");
    assert!(ok, "replay diverged: {} vs {}", live.state_hash, replayed.state_hash);
    assert_eq!(replayed.state_hash, live.state_hash);

    // Every cue, same tick, same car, same cue, same value — not a hash match
    // alone, because a hash can be compared without understanding it.
    let rep = replayed.sound_events.as_ref().expect("replay kept the cue log");
    assert_eq!(rep.len(), live_events.len(), "cue count changed under replay");
    for (i, (a, b)) in live_events.iter().zip(rep.iter()).enumerate() {
        assert_eq!((a.tick, a.car, a.cue, a.value), (b.tick, b.car, b.cue, b.value),
            "cue {i} differs under replay");
    }
    assert_eq!(
        replayed.sound.as_ref().unwrap().log_hash,
        live_sum.log_hash
    );
    // And the replay read the expectation out of the log, rather than having
    // nothing to compare against.
    assert_eq!(replayed.sound_expected_hash, Some(live_sum.log_hash));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gatec_accelerated_mode_keeps_order_and_sheds_the_least_important_cue() {
    // A full race at max speed, then every cue it produced handed to a real
    // consumer at the default 40 cue/s ceiling. An accelerated race generates
    // cues far faster than that, so shedding is guaranteed — the test is that
    // it is bounded, ordered, and biased the right way.
    let (outcome, _inputs) = apexsim::race::run_race(
        RaceConfig { sound: Some(SoundConfig::default()), ..base_cfg(33) },
        false,
    );
    let events = outcome.sound_events.as_ref().expect("a cue log");
    assert!(events.len() > 200, "only {} cues — too few to stress the ceiling", events.len());

    // The ceiling is per **wall-clock** second, so what matters is the rate
    // the cues arrive at, which is the simulation length divided by how long
    // the run actually took. A max-speed run is thousands of times real time,
    // so the same cue log that would trickle for two minutes of racing lands
    // in about a second of listening — which is the whole reason the consumer
    // sheds instead of queueing.
    let wall_s = outcome.wall_time_s;
    assert!(wall_s > 0.0, "wall time was not measured");
    let per_s = events.len() as f64 / wall_s;
    assert!(
        per_s > 40.0,
        "the run produced only {per_s:.0} cues per wall second, which the ceiling would not have to shed"
    );
    println!(
        "{} cues in {wall_s:.2} s of wall clock = {per_s:.0} cue/s against a 40 cue/s ceiling",
        events.len()
    );

    // Replay the run the way a listener would meet it: cues are offered as
    // fast as the race produced them, one wall-clock second per window.
    let mut c = AudioConsumer::new(40);
    let mut started: Vec<sound::CueEvent> = Vec::new();
    let mut max_pending = 0usize;
    let mut windows = 0u32;
    let mut cursor = 0usize;
    while cursor < events.len() {
        // The share of the log that "arrived" during this wall-clock second.
        // A real listener is bounded by the backlog the consumer allows, so
        // the cursor only advances by what could physically have been queued.
        let due = (cursor + BACKLOG_LIMIT).min(events.len());
        for e in &events[cursor..due] {
            c.offer(*e);
        }
        cursor = due;
        max_pending = max_pending.max(c.pending());
        c.begin_second(40);
        // Drained at the start of the window: the consumer polls, plays what
        // it can, and the window closes on it.
        started.extend(c.drain(0.0));
        c.expire();
        windows += 1;
    }
    let st = c.stats();
    assert_eq!(st.started + st.shed, events.len() as u64, "cues went missing");
    assert!(
        st.started <= windows as u64 * 40,
        "{} cues started over {windows} windows at a ceiling of 40/s",
        st.started
    );
    assert!(
        st.shed > 0,
        "nothing was shed, so the ceiling never bound — this test is not testing shedding"
    );
    assert!(
        st.started as f64 / windows as f64 <= 40.0 + 1e-9,
        "{} cues/s delivered against a 40 cue/s ceiling",
        st.started as f64 / windows as f64
    );
    assert!(
        max_pending <= BACKLOG_LIMIT,
        "the backlog grew to {max_pending}, past the {BACKLOG_LIMIT} bound"
    );

    // Order: what was started is a subsequence of the race's own order, so a
    // listener hears the race as it unfolded.
    assert_monotonic(&started);
    let mut i = 0usize;
    for e in &started {
        while i < events.len() && events[i] != *e {
            i += 1;
        }
        assert!(i < events.len(), "started a cue that never happened in order: {e:?}");
        i += 1;
    }

    // Bias: the shed set is dominated by the low-priority continuous cues,
    // and the never-shed ones include the race events. Losing "green flag"
    // makes the race unfollowable; losing a wind gust costs nothing.
    // Bias: priority decides what is thrown away. Two claims, both of which
    // the design actually makes and neither of which is automatic.
    //
    // 1. The session-defining cues are never shed. They are rare (six start
    //    lights, one green flag in a whole race), so they are always the most
    //    important thing queued when pressure comes.
    // 2. The losses land on the continuous cues. They are the bulk of the
    //    log, so priority can always find something cheaper to drop than a
    //    race event.
    //
    // It is *not* claimed that every priority-2 cue survives: in a window
    // whose backlog happens to hold few low-priority cues, priority 2 loses to
    // nothing and is shed in turn. Shedding by priority is a preference
    // between available cues, not a reservation.
    let total_of = |c: SoundCue| events.iter().filter(|e| e.cue == c).count();
    let started_of = |c: SoundCue| started.iter().filter(|e| e.cue == c).count();

    for c in ALL_CUES.iter().filter(|c| c.priority() == 3) {
        assert_eq!(
            started_of(*c),
            total_of(*c),
            "{} (priority 3) was shed",
            c.name()
        );
    }
    assert!(
        started.iter().any(|e| e.cue == SoundCue::GreenFlag),
        "the green flag never reached a listener"
    );

    let shed_total = events.len() - started.len();
    let shed_low_priority = events
        .iter()
        .filter(|e| e.cue.priority() == 0)
        .count()
        - ALL_CUES
            .iter()
            .filter(|c| c.priority() == 0)
            .map(|c| started_of(*c))
            .sum::<usize>();
    assert!(
        shed_total > 0,
        "nothing was shed, so the ceiling never bound — this test is not testing shedding"
    );
    assert!(
        shed_low_priority as f64 >= 0.95 * shed_total as f64,
        "only {shed_low_priority} of {shed_total} shed cues were the least important kind;                  shedding is not concentrated where it should be"
    );

    // Continuous cues dominate the log, so they dominate the shed set.
    let continuous: usize = events
        .iter()
        .filter(|e| matches!(e.cue, SoundCue::ShiftUp | SoundCue::ShiftDown | SoundCue::Wind | SoundCue::TyreSlip))
        .count();
    assert!(
        continuous * 2 > events.len(),
        "the log is not continuous-cue dominated, so this test is not exercising what it claims"
    );
}

#[test]
fn gatec_a_consumer_never_starts_more_than_the_ceiling_in_one_second() {
    // A direct check on the rate itself, because "the backlog is bounded" and
    // "the rate is bounded" are different properties and only one of them
    // keeps a listener from being machine-gunned.
    let mut c = AudioConsumer::new(10);
    c.begin_second(10);
    for i in 0..100u32 {
        c.offer(sound::CueEvent {
            tick: i as u64,
            car: 0,
            cue: SoundCue::Wind,
            value: 30.0,
        });
    }
    let started = c.drain(0.0);
    assert_eq!(started.len(), 10, "10 cue/s must start exactly 10 in a fresh window");
    // Draining again in the same window starts nothing more: the budget is a
    // rate, not a per-call allowance.
    assert_eq!(c.drain(0.0).len(), 0, "the ceiling was re-charged mid-second");
    c.begin_second(10);
    assert_eq!(c.drain(0.0).len(), 10, "a new window re-charges the ceiling");

    // A late poll cannot burst the whole allowance either: the caller says it
    // is already 0.95 s into the window, so only what is left may start.
    let mut c = AudioConsumer::new(10);
    c.begin_second(10);
    for i in 0..100u32 {
        c.offer(sound::CueEvent { tick: i as u64, car: 0, cue: SoundCue::Wind, value: 30.0 });
    }
    let late = c.drain(0.95e9);
    assert!(late.len() <= 1, "a late poll started {} cues at 0.95 s", late.len());

    // Zero means "no ceiling", which is only sensible for a real-time race.
    let mut c = AudioConsumer::new(0);
    c.begin_second(0);
    for i in 0..100u32 {
        c.offer(sound::CueEvent { tick: i as u64, car: 0, cue: SoundCue::Wind, value: 30.0 });
    }
    assert_eq!(c.drain(0.0).len(), 100, "max_cues_per_s = 0 must mean unlimited");
}

#[test]
fn gatec_shedding_prefers_to_lose_the_newest_lowest_priority_cue() {
    let mk = |cue: SoundCue, tick: u64| sound::CueEvent { tick, car: 0, cue, value: 1.0 };
    let mut c = AudioConsumer::new(1000);
    c.begin_second(1000);
    // Old and important first, then a flood of the least important cue.
    c.offer(mk(SoundCue::GreenFlag, 1));
    c.offer(mk(SoundCue::LapComplete, 2));
    for t in 3..(3 + BACKLOG_LIMIT + 50) {
        c.offer(mk(SoundCue::Wind, t as u64));
    }
    let started = c.drain(0.0);
    // Everything started, or everything shed, but the two important cues must
    // have survived: they were queued first and nothing outranks them.
    let kept: Vec<&sound::CueEvent> = started.iter().collect();
    assert!(kept.iter().any(|e| e.cue == SoundCue::GreenFlag), "the green flag was shed");
    assert!(kept.iter().any(|e| e.cue == SoundCue::LapComplete), "a lap chime was shed");
    // Among the flood, what went was the newest, not the oldest.
    assert_monotonic(&started);
    let oldest_wind = started.iter().find(|e| e.cue == SoundCue::Wind).expect("some wind kept");
    assert_eq!(oldest_wind.tick, 3, "the oldest wind cue was shed instead of the newest");
}

// ---------------------------------------------------------------------------
// Build item 4 — a small validated sound vocabulary, assets on disk.
// ---------------------------------------------------------------------------

#[test]
fn build4_every_cue_has_a_validated_asset_and_the_vocabulary_is_small() {
    assert_eq!(CUE_COUNT, 13, "the vocabulary is meant to stay small and fixed");
    assert_eq!(ALL_CUES.len(), CUE_COUNT);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static/sounds");
    let mut names: HashSet<String> = HashSet::new();
    for cue in ALL_CUES {
        assert!(names.insert(cue.name().to_string()), "duplicate cue name {}", cue.name());
        assert_eq!(cue.code(), ALL_CUES.iter().position(|c| *c == cue).unwrap() as u8,
            "cue codes must equal their index — the log stores the code");
        let path = root.join(cue.asset());
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("cue {} has no asset at {}: {e}", cue.name(), path.display()));
        let info = sound::validate_wav(&bytes)
            .unwrap_or_else(|e| panic!("cue {} failed validation: {e}", cue.name()));
        assert!(
            info.duration_s >= 0.02 && info.duration_s <= 1.5,
            "cue {} is {} ms",
            cue.name(),
            info.duration_s * 1000.0
        );
        assert!(
            info.peak > 0.05 && info.peak < 0.99,
            "cue {} peaks at {:.3} — inaudible or clipping",
            cue.name(),
            info.peak
        );
    }
    // The plan names three cues specifically; they must be in the vocabulary.
    for required in ["shift_up", "tyre_slip", "wind"] {
        assert!(names.contains(required), "the plan asks for {required}");
    }
}

// ---------------------------------------------------------------------------
// The observers stay observers: enabling Phase 9 cannot reach the hash.
// ---------------------------------------------------------------------------

#[test]
fn phase9_is_off_by_default_and_enabling_it_costs_nothing_when_receiving() {
    let plain = base_cfg(17);
    assert!(plain.telemetry.is_none() && plain.sound.is_none());

    let (a, _) = apexsim::race::run_race(plain.clone(), false);
    assert!(a.telemetry.is_none() && a.sound.is_none(), "no cue log without sound");

    // Sound on, nothing listening: same race, plus a cue log.
    let probe = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
    let dead = probe.local_addr().unwrap().port();
    drop(probe);
    let observed = RaceConfig {
        telemetry: Some(TelemetryConfig { target: format!("127.0.0.1:{dead}"), ..Default::default() }),
        sound: Some(SoundConfig::default()),
        ..plain.clone()
    };
    let (b, _) = apexsim::race::run_race(observed.clone(), false);
    assert_eq!(a.state_hash, b.state_hash, "observers reached mix_state");
    assert_eq!(a.ticks, b.ticks);
    assert!(!b.sound_events.as_ref().unwrap().is_empty(), "sound produced no cues at all");

    // Sound off but telemetry on: still the same race, and no cue log.
    let tel_only = RaceConfig { sound: None, ..observed.clone() };
    let (c, _) = apexsim::race::run_race(tel_only, false);
    assert_eq!(a.state_hash, c.state_hash);
    assert!(c.sound.is_none(), "a cue log appeared with sound disabled");

    // Disabled sound config is inert even when present.
    let inert = RaceConfig {
        sound: Some(SoundConfig { enabled: false, ..SoundConfig::default() }),
        ..plain.clone()
    };
    let (d, _) = apexsim::race::run_race(inert, false);
    assert_eq!(a.state_hash, d.state_hash);
    assert!(d.sound.is_none(), "a disabled vocabulary still produced a runtime");
    assert!(d.sound_events.is_none(), "a disabled vocabulary still logged cues");
}

// ---------------------------------------------------------------------------
// Sound extraction is a pure function of simulation state.
// ---------------------------------------------------------------------------

#[test]
fn extraction_is_deterministic_and_depends_only_on_tick_and_state() {
    let gb = Gearbox::default();
    let run = || {
        let mut rt = SoundRuntime::new(SoundConfig::default(), 2, gb);
        for t in 0..4000u64 {
            let s = vec![
                CarSample {
                    vel_long: 20.0 + (t as f64) * 0.02,
                    gear: if (t / 250) % 2 == 0 { 3 } else { 4 },
                    engine_rpm: 9000,
                    g_lat: if (t / 100) % 3 == 0 { 1.5 } else { 0.2 },
                    throttle: 0.9,
                    ..Default::default()
                },
                CarSample { vel_long: 30.0, gear: 5, engine_rpm: 11_000, ..Default::default() },
            ];
            rt.extract(t, t as f64 * apexsim::race::DT, &s);
        }
        rt.events.clone()
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "the same tick sequence must give the same cue log");
    assert!(!a.is_empty(), "the synthetic input produced no cues");
    assert_eq!(sound::hash_log(&a), sound::hash_log(&b));
    assert_monotonic(&a);
    // Every tick used is inside the run, and values are the triggering quantity.
    for e in &a {
        assert!(e.tick < 4000);
        assert!(e.value.is_finite());
    }
}

// ---------------------------------------------------------------------------
// The link never blocks, even against a peer that will never answer.
// ---------------------------------------------------------------------------

#[test]
fn publishing_to_a_dead_peer_costs_microseconds_and_never_blocks() {
    let probe = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
    let dead = probe.local_addr().unwrap().port();
    drop(probe);
    let cfg = TelemetryConfig {
        target: format!("127.0.0.1:{dead}"),
        hz: 250,
        ..TelemetryConfig::default()
    };
    let mut link = TelemetryLink::spawn(&cfg).expect("link to a dead port is not a failure");
    let cars = samples(6);
    let t0 = Instant::now();
    let frames = 500;
    for i in 0..frames {
        link.publish(i as f64 * apexsim::race::DT, &cars);
    }
    let elapsed = t0.elapsed();
    let st: LinkStats = link.stats();
    // Offered is counted per datagram, so two per frame, and drops can never
    // exceed it. A burst this tight *should* drop — the queue is bounded on
    // purpose — but it must never wait.
    assert_eq!(st.offered as usize, frames * 2, "two datagrams per frame");
    // Let the sender threads finish so the counters are complete.
    std::thread::sleep(Duration::from_millis(200));
    let st = link.stats();
    assert_eq!(st.offered, st.sent + st.dropped, "counters do not add up");
    assert!(st.sent > 0, "nothing reached the wire");
    // 500 frames is 2 s of race time. If each one waited on the socket the
    // publish loop alone would exceed the wall clock it is simulating.
    assert!(
        elapsed.as_secs_f64() < 0.05,
        "500 publishes took {elapsed:?}, which is not non-blocking"
    );
}

// ---------------------------------------------------------------------------
// Build item 4, the "separate sound consumer" half: the browser has to be
// able to fetch the assets it is asked to play.
//
// Validating the WAVs on disk proves the files are sound, not that the
// browser can reach them. The consumer is only real if a GET of
// /sounds/<cue>.wav returns those bytes over HTTP, so that is what is
// checked here — including a 404 for a name that is not in the vocabulary,
// which is what stops a typo in the dashboard from failing silently as
// "audio just did not play".
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_dashboard_can_fetch_every_cue_asset_over_http() {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpStream;

    let cfg = apexsim::sim::SimConfig {
        laps: 999,
        seed: 5,
        ..apexsim::sim::SimConfig::default()
    };
    let (addr, _h) = apexsim::server::spawn_server(cfg, 0, true).expect("spawn server");
    assert!(
        apexsim::server::wait_until_ready(addr, Duration::from_secs(5)).await,
        "server did not report ready"
    );

    async fn get(addr: std::net::SocketAddr, path: &str) -> (u16, Vec<u8>, String) {
        let mut s = TcpStream::connect(addr).await.expect("connect");
        let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
        s.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).await.unwrap();
        let split = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("no header terminator");
        let head = String::from_utf8_lossy(&buf[..split]).to_string();
        let status: u16 = head.split(' ').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        (status, buf[split + 4..].to_vec(), head)
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static/sounds");
    for cue in ALL_CUES {
        let (status, body, head) = get(addr, &format!("/sounds/{}", cue.asset())).await;
        assert_eq!(status, 200, "GET /sounds/{} -> {status}", cue.asset());
        assert!(
            head.to_ascii_lowercase().contains("audio/wav"),
            "cue {} was not served as audio/wav, so the browser will refuse it:\n{head}",
            cue.name()
        );
        let on_disk = std::fs::read(root.join(cue.asset())).expect("asset on disk");
        assert_eq!(
            body.len(),
            on_disk.len(),
            "served bytes for {} differ in length from the file",
            cue.name()
        );
        // The served body must be the *validated* asset, not merely the right
        // size: a stale embedded copy would still pass a length check.
        assert_eq!(body, on_disk, "served bytes for {} are not the file", cue.name());
        sound::validate_wav(&body).expect("served bytes are a valid wav");
    }

    let (status, _, _) = get(addr, "/sounds/not_a_cue.wav").await;
    assert_eq!(status, 404, "an unknown cue name must 404, not serve something");
}
