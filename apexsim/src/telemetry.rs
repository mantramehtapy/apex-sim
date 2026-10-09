//! Phase 9 — hardware telemetry.
//!
//! The engine's state is published to external devices using the **2025 UDP
//! telemetry packet layout** (`m_packetFormat = 2025`) — the wire format
//! popularized by several racing titles, on UDP port 20777.
//!
//! `docs/f1_2025_udp_spec.md` records the byte layout and how each field was
//! verified. Read that before changing an offset: the layout is pinned by
//! published packet totals, so a wrong stride produces a packet no receiver
//! can parse.
//!
//! Two properties are load-bearing, both required by the phase:
//!
//! * **It never blocks physics.** The sim thread encodes into a buffer taken
//!   from a small pool and hands it to a sender thread over a *bounded*
//!   channel. If the channel is full the frame is **dropped and counted** —
//!   telemetry is a lossy observer of the race, never a participant in it.
//! * **It never feeds back into physics.** Nothing here is reachable from
//!   `mix_state`, and `m_engineRPM` / `m_gear` are presentation channels
//!   derived from wheel speed by [`Gearbox`] — this physics model has no
//!   engine. The gate `telemetry_never_reaches_the_physics` proves a race
//!   hashes identically with and without a listener attached.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Wire constants. See docs/f1_2025_udp_spec.md.
// ---------------------------------------------------------------------------

/// Packet format / game year, as written into every header.
pub const PACKET_FORMAT: u16 = 2025;
pub const GAME_YEAR: u8 = 25;
/// The protocol's fixed field array. Car `i` occupies slot `i`.
pub const MAX_CARS: usize = 22;
/// The layout's 2023+ header. (2021/22 used 24 bytes; we emit only 2025.)
pub const HEADER_SIZE: usize = 29;
/// Per-car block in both the Motion and Car Telemetry packets.
pub const MOTION_STRIDE: usize = 60;
pub const CAR_TELEMETRY_STRIDE: usize = 60;

pub const PACKET_ID_MOTION: u8 = 0;
pub const PACKET_ID_CAR_TELEMETRY: u8 = 6;

/// Published packet totals. These are not free parameters — they are the
/// arithmetic identity that pins every stride in this file, and they match the
/// sizes a real receiver of this layout expects. See docs/udp_telemetry_spec.md.
pub const MOTION_PACKET_SIZE: usize = HEADER_SIZE + MAX_CARS * MOTION_STRIDE; // 1349
/// Car Telemetry carries three trailing bytes after the 22 blocks.
pub const CAR_TELEMETRY_PACKET_SIZE: usize =
    HEADER_SIZE + MAX_CARS * CAR_TELEMETRY_STRIDE + 3; // 1352

/// `g`, for the g-force channels.
pub const G: f64 = 9.81;

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// The 29 common header fields. Every packet this engine sends repeats them,
/// and a receiver reads them before it trusts anything else in the datagram.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PacketHeader {
    pub session_uid: u64,
    pub session_time: f64,
    pub frame_id: u32,
    pub overall_frame_id: u32,
    /// Which of the 22 slots the local player drives. It does *not* select
    /// which car a block describes — see `docs/f1_2025_udp_spec.md`.
    pub player_car_index: u8,
}

impl Default for PacketHeader {
    fn default() -> Self {
        PacketHeader {
            session_uid: 1,
            session_time: 0.0,
            frame_id: 0,
            overall_frame_id: 0,
            player_car_index: 0,
        }
    }
}

impl PacketHeader {
    /// Write the header at byte 0 of `buf`. `session_time` is f32 seconds and
    /// `session_uid` u64, so a long session loses precision the same way the
    /// real format does — that is intended, not a shortcut.
    pub fn write(&self, buf: &mut [u8], packet_id: u8) {
        buf[0..2].copy_from_slice(&PACKET_FORMAT.to_le_bytes());
        buf[2] = GAME_YEAR;
        buf[3] = 25; // m_gameMajorVersion — carries no information for a receiver
        buf[4] = 0; // m_gameMinorVersion
        buf[5] = 1; // m_packetVersion
        buf[6] = packet_id;
        buf[7..15].copy_from_slice(&self.session_uid.to_le_bytes());
        buf[15..19].copy_from_slice(&(self.session_time as f32).to_le_bytes());
        buf[19..23].copy_from_slice(&self.frame_id.to_le_bytes());
        buf[23..27].copy_from_slice(&self.overall_frame_id.to_le_bytes());
        buf[27] = self.player_car_index;
        buf[28] = self.secondary_player_car_index();
    }

    /// A receiver with no second player expects 255 here.
    fn secondary_player_car_index(&self) -> u8 {
        255
    }
}

// ---------------------------------------------------------------------------
// Gearbox — the presentation model
// ---------------------------------------------------------------------------

/// A fixed-ratio gearbox used only to produce a plausible `m_gear` /
/// `m_engineRPM` pair for the telemetry packet and the audio cues.
///
/// This is **not** an engine. The physics model has no crank, no torque curve
/// and no gear state; this derives a display value from wheel speed. Nothing
/// here is readable from the vehicle state or the trajectory hash.
///
/// The ratios are an ordinary road-car-like 8-speed set chosen so the shift
/// points land in a plausible 3 500–13 000 rev/min window across the top
/// speed this engine reaches (~90 m/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gearbox {
    /// Final drive (gear ratio × final-drive ratio) per gear, 1st first.
    pub ratios: [f64; 8],
    /// Wheel radius, m. 0.33 m is a plausible 18-inch wheel.
    pub wheel_radius_m: f64,
    pub idle_rpm: f64,
    pub limiter_rpm: f64,
    /// Rev/min at which the next gear is selected.
    pub shift_up_rpm: f64,
}

impl Default for Gearbox {
    fn default() -> Self {
        Gearbox {
            ratios: [3.13, 2.19, 1.61, 1.28, 1.06, 0.90, 0.79, 0.71],
            wheel_radius_m: 0.33,
            idle_rpm: 3_500.0,
            limiter_rpm: 13_000.0,
            shift_up_rpm: 11_500.0,
        }
    }
}

impl Gearbox {
    /// Engine speed implied by a road speed, in rev/min. Clamped to
    /// [idle, limiter] because the format has no out-of-range value: an
    /// engine at 15 000 or 900 would be a bug in the sender, not information.
    pub fn rpm_at(&self, gear: usize, vel_long: f64) -> f64 {
        let ratio = self.ratios[gear.min(self.ratios.len() - 1)];
        let wheel_circumference = 2.0 * std::f64::consts::PI * self.wheel_radius_m;
        let wheel_revs_per_s = vel_long.max(0.0) / wheel_circumference;
        let engine_rev_per_s = wheel_revs_per_s * ratio;
        (engine_rev_per_s * 60.0).clamp(self.idle_rpm, self.limiter_rpm)
    }

    /// The gear a car in `vel_long` would be in: neutral below walking pace,
    /// otherwise the lowest gear that has not yet reached its shift point.
    pub fn gear_for(&self, vel_long: f64) -> i8 {
        if vel_long < 1.0 {
            return 0; // stationary -> neutral, as a real box would report
        }
        let mut chosen = self.ratios.len() - 1;
        for (i, _) in self.ratios.iter().enumerate() {
            if self.rpm_at(i, vel_long) < self.shift_up_rpm {
                chosen = i;
                break;
            }
        }
        (chosen + 1) as i8
    }

    /// Gear and rev/min together, which is what the packet actually carries.
    pub fn readout(&self, vel_long: f64) -> (i8, u16) {
        let g = self.gear_for(vel_long);
        if g <= 0 {
            return (0, self.idle_rpm as u16);
        }
        (g, self.rpm_at(g as usize - 1, vel_long) as u16)
    }

    /// Shift-light bar, 0–100. Derived from the same rev/min, so a receiver
    /// and the audio limiter cue can never disagree.
    pub fn rev_lights_percent(&self, gear: i8, vel_long: f64) -> u8 {
        if gear <= 0 {
            return 0;
        }
        let rpm = self.rpm_at(gear as usize - 1, vel_long);
        let span = (self.limiter_rpm - self.idle_rpm).max(1.0);
        (((rpm - self.idle_rpm) / span) * 100.0).clamp(0.0, 100.0) as u8
    }
}

// ---------------------------------------------------------------------------
// Per-car sample
// ---------------------------------------------------------------------------

/// One car, in the units the protocol wants. Built by the race, consumed by
/// the encoders. Keeping it a plain data struct is what lets the encoder be
/// tested without standing up a simulation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarSample {
    pub pos_x: f64,
    pub pos_y: f64,
    pub heading: f64,
    /// World-frame velocity, m/s.
    pub vel_x: f64,
    pub vel_y: f64,
    /// Body-frame longitudinal speed, m/s. Also the source of `m_speed`.
    pub vel_long: f64,
    pub g_lat: f64,
    pub g_lon: f64,
    /// Normalised 0–1.
    pub throttle: f64,
    pub brake: f64,
    /// Normalised −1–1 (left positive).
    pub steer: f64,
    pub gear: i8,
    pub engine_rpm: u16,
    pub rev_lights_percent: u8,
}

impl CarSample {
    /// Build a sample from raw vehicle state. `accel_long` is the
    /// longitudinal acceleration in m/s², which the caller measures because
    /// only the caller knows the previous tick's speed.
    pub fn from_car(
        c: &crate::physics::CarState,
        ctrl: &crate::physics::Controls,
        max_steer: f64,
        accel_long: f64,
        gearbox: &Gearbox,
    ) -> CarSample {
        // Body-frame velocity -> world frame, so the motion packet's velocity
        // and forward-direction channels agree with its position channel.
        let (sin_h, cos_h) = c.heading.sin_cos();
        let (gear, rpm) = gearbox.readout(c.vel_long);
        CarSample {
            pos_x: c.pos_x,
            pos_y: c.pos_y,
            heading: c.heading,
            vel_x: c.vel_long * cos_h - c.vel_lat * sin_h,
            vel_y: c.vel_long * sin_h + c.vel_lat * cos_h,
            vel_long: c.vel_long,
            // a_lat = v·r for a steady turn; signed with the engine's
            // positive-left convention rather than guessed from the format.
            g_lat: (c.yaw_rate * c.vel_long) / G,
            g_lon: accel_long / G,
            throttle: ctrl.throttle.clamp(0.0, 1.0),
            brake: ctrl.brake.clamp(0.0, 1.0),
            steer: if max_steer > 0.0 {
                (ctrl.steer_cmd / max_steer).clamp(-1.0, 1.0)
            } else {
                0.0
            },
            gear,
            engine_rpm: rpm,
            rev_lights_percent: gearbox.rev_lights_percent(gear, c.vel_long),
        }
    }
}

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------

fn f32_at(buf: &mut [u8], off: usize, v: f64) {
    buf[off..off + 4].copy_from_slice(&(v as f32).to_le_bytes());
}
fn u16_at(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
fn u8_at(buf: &mut [u8], off: usize, v: u8) {
    buf[off] = v;
}
fn i16_at(buf: &mut [u8], off: usize, v: i16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

/// Pack a normalised direction component into the format's int16.
///
/// The divisor is 32767.0, not 32768: the spec's own note says so, and the
/// difference matters only for exactly +1.0 and −1.0 — but a receiver
/// dividing by 32768 and one dividing by 32767 must agree on the bytes, so
/// the documented divisor is the one used.
fn dir_i16(v: f64) -> i16 {
    if !v.is_finite() {
        return 0;
    }
    (v.clamp(-1.0, 1.0) * 32767.0).round() as i16
}
fn i8_at(buf: &mut [u8], off: usize, v: i8) {
    buf[off] = v as u8;
}

/// Encode a Motion packet into `buf`, which must be exactly
/// [`MOTION_PACKET_SIZE`] bytes. Returns the number of cars actually written.
// `+ 0` is kept deliberately: each write names its byte offset inside the car
// block, so `base + 0`, `base + 4`, `base + 8` read as the packet layout they
// are. Dropping the `+ 0` would hide which channel sits at offset zero.
#[allow(clippy::identity_op)]
pub fn encode_motion(buf: &mut [u8], hdr: &PacketHeader, cars: &[CarSample]) -> usize {
    assert_eq!(buf.len(), MOTION_PACKET_SIZE, "motion packet has a fixed size");
    hdr.write(buf, PACKET_ID_MOTION);
    for slot in 0..MAX_CARS {
        let base = HEADER_SIZE + slot * MOTION_STRIDE;
        let Some(c) = cars.get(slot) else {
            // Empty slots stay all-zero. A zero block reads unambiguously:
            // speed 0, gear neutral, no motion.
            for b in buf[base..base + MOTION_STRIDE].iter_mut() {
                *b = 0;
            }
            continue;
        };
        let (sin_h, cos_h) = c.heading.sin_cos();
        f32_at(buf, base + 0, c.pos_x);
        f32_at(buf, base + 4, c.pos_y);
        f32_at(buf, base + 8, 0.0); // Z: this engine is planar
        f32_at(buf, base + 12, c.vel_x);
        f32_at(buf, base + 16, c.vel_y);
        f32_at(buf, base + 20, 0.0);
        // The six direction channels are **int16 normalised**, not floats.
        // The spec is explicit: "For the normalised vectors below, to convert
        // to float values divide by 32767.0f". Writing a float here is a
        // silent corruption — the receiver reads the low two bytes of a float
        // and sees an integer of up to ±32767 that was never a direction.
        // Caught by cross-checking against three third-party decoders; see
        // docs/f1_2025_udp_spec.md.
        i16_at(buf, base + 24, dir_i16(cos_h));
        i16_at(buf, base + 26, dir_i16(sin_h));
        i16_at(buf, base + 28, 0);
        // Right = forward rotated +90° (clockwise in this engine's frame).
        i16_at(buf, base + 30, dir_i16(-sin_h));
        i16_at(buf, base + 32, dir_i16(cos_h));
        i16_at(buf, base + 34, 0);
        f32_at(buf, base + 36, c.g_lat);
        f32_at(buf, base + 40, c.g_lon);
        f32_at(buf, base + 44, 0.0);
        f32_at(buf, base + 48, c.heading);
        f32_at(buf, base + 52, 0.0); // pitch
        f32_at(buf, base + 56, 0.0); // roll
    }
    cars.len().min(MAX_CARS)
}

/// Encode a Car Telemetry packet into `buf`, which must be exactly
/// [`CAR_TELEMETRY_PACKET_SIZE`] bytes.
// `+ 0` is kept deliberately: each write names its byte offset inside the car
// block, so `base + 0`, `base + 4`, `base + 8` read as the packet layout they
// are. Dropping the `+ 0` would hide which channel sits at offset zero.
#[allow(clippy::identity_op)]
pub fn encode_car_telemetry(buf: &mut [u8], hdr: &PacketHeader, cars: &[CarSample]) -> usize {
    assert_eq!(
        buf.len(),
        CAR_TELEMETRY_PACKET_SIZE,
        "car telemetry packet has a fixed size"
    );
    hdr.write(buf, PACKET_ID_CAR_TELEMETRY);
    for slot in 0..MAX_CARS {
        let base = HEADER_SIZE + slot * CAR_TELEMETRY_STRIDE;
        let Some(c) = cars.get(slot) else {
            for b in buf[base..base + CAR_TELEMETRY_STRIDE].iter_mut() {
                *b = 0;
            }
            continue;
        };
        u16_at(buf, base + 0, speed_kph(c.vel_long));
        f32_at(buf, base + 2, c.throttle);
        f32_at(buf, base + 6, c.steer);
        f32_at(buf, base + 10, c.brake);
        u8_at(buf, base + 14, 0); // clutch — no analogue input in this model
        i8_at(buf, base + 15, c.gear);
        u16_at(buf, base + 16, c.engine_rpm);
        u8_at(buf, base + 18, 0); // DRS — not fitted
        u8_at(buf, base + 19, c.rev_lights_percent);
        u16_at(buf, base + 20, 0); // rev-lights bitmask
        // Brake, tyre and pressure channels stay zero: this model has no
        // thermal or pressure state, and a plausible-looking invented number
        // would be worse than an honest zero. See docs/f1_2025_udp_spec.md.
    }
    // Trailing bytes: mfd panel indices (unused) and a suggested gear.
    u8_at(buf, HEADER_SIZE + MAX_CARS * CAR_TELEMETRY_STRIDE + 0, 255);
    u8_at(buf, HEADER_SIZE + MAX_CARS * CAR_TELEMETRY_STRIDE + 1, 255);
    i8_at(buf, HEADER_SIZE + MAX_CARS * CAR_TELEMETRY_STRIDE + 2, cars.first().map_or(0, |c| c.gear));
    cars.len().min(MAX_CARS)
}

/// `m_speed` is a u16 in **km/h**, the one unit in this packet that is not
/// SI. A receiver that assumed m/s would show 3.6× the real speed, so the
/// conversion lives here in one place.
pub fn speed_kph(vel_long: f64) -> u16 {
    let kph = vel_long * 3.6;
    if !kph.is_finite() || kph <= 0.0 {
        0
    } else {
        kph.min(u16::MAX as f64) as u16
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Where and how fast to publish. Constructed from CLI flags; the default
/// target is the loopback address on the format's conventional port.
#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryConfig {
    pub target: String,
    /// Publish rate, independent of the 250 Hz physics step.
    pub hz: u32,
    pub player_car: u32,
    pub session_uid: u64,
    pub gearbox: Gearbox,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        TelemetryConfig {
            target: "127.0.0.1:20777".to_string(),
            hz: 30,
            player_car: 0,
            session_uid: 1,
            gearbox: Gearbox::default(),
        }
    }
}

impl TelemetryConfig {
    /// Ticks between publishes at the fixed physics step. Clamped to at least
    /// one tick: a 0 Hz request must not turn into a divide-by-zero or a
    /// publish on every single tick.
    pub fn interval_ticks(&self, dt: f64) -> u64 {
        let hz = self.hz.clamp(1, 250);
        let n = (1.0 / (hz as f64 * dt)).round() as u64;
        n.max(1)
    }

    /// Parse a `host:port` target.
    pub fn parse_target(s: &str) -> Result<String, String> {
        let t = s.trim();
        if t.is_empty() {
            return Err("telemetry target cannot be empty".to_string());
        }
        let Some((h, p)) = t.rsplit_once(':') else {
            // No port at all. It would fail later at `spawn`, but "telemetry
            // target 'localhost' is not host:port" at the command line is a
            // better error than the same message from inside the publisher.
            return Err(format!("telemetry target '{t}' has no port (expected host:port)"));
        };
        let port: u16 = p
            .parse()
            .map_err(|_| format!("telemetry target port '{p}' is not a number"))?;
        if port == 0 {
            return Err("telemetry target port must be 1..=65535, got 0".to_string());
        }
        if h.is_empty() {
            return Err(format!("telemetry target '{t}' has no host"));
        }
        Ok(t.to_string())
    }

    /// True when this config would not publish anything a receiver could see.
    pub fn is_inert(&self) -> bool {
        self.target.trim().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Link statistics
// ---------------------------------------------------------------------------

/// Observable link counters. A silent failure — the most likely way a UDP
/// sender breaks — has to be visible, so every drop and error is counted and
/// surfaced on /api/metrics and on the dashboard.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LinkStats {
    /// Datagrams the socket accepted.
    pub sent: u64,
    /// Frames dropped because the sender thread was behind. This is the
    /// intended behaviour under load, and the reason the race never waits.
    pub dropped: u64,
    /// `send_to` errors — on loopback, almost always ICMP "connection
    /// refused" from a port with nothing listening.
    pub send_errors: u64,
    pub bytes: u64,
    /// Datagrams the sim thread handed to a sender channel. `dropped` can
    /// never exceed it — the two are counted in the same unit, deliberately:
    /// one frame offers two datagrams, and a counter that mixed frames with
    /// datagrams would report more drops than there were chances to send.
    pub offered: u64,
}

#[derive(Debug, Default)]
struct Counters {
    sent: AtomicU64,
    dropped: AtomicU64,
    send_errors: AtomicU64,
    bytes: AtomicU64,
    offered: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> LinkStats {
        LinkStats {
            sent: self.sent.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            send_errors: self.send_errors.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            offered: self.offered.load(Ordering::Relaxed),
        }
    }
}

/// Buffers in flight. Enough to absorb a scheduling hiccup, small enough that
/// a stalled sender thread costs a few frames before the race starts dropping
/// telemetry rather than queueing it.
///
/// Public because it is also the *bound on the tail*: when a race returns, the
/// sender threads may still hold this many datagrams per packet kind, so a
/// stats snapshot can legitimately undercount what the wire already carried.
pub const POOL_DEPTH: usize = 4;

/// Where the sim thread and the sender thread meet.
struct Channel {
    tx: SyncSender<Vec<u8>>,
    rx: Arc<Mutex<Option<Receiver<Vec<u8>>>>>,
    pool: Arc<Mutex<VecDeque<Vec<u8>>>>,
    counters: Arc<Counters>,
}

impl Channel {
    fn new(size: usize, counters: Arc<Counters>) -> Channel {
        let (tx, rx) = sync_channel::<Vec<u8>>(POOL_DEPTH);
        let pool = Arc::new(Mutex::new((0..POOL_DEPTH).map(|_| vec![0u8; size]).collect()));
        Channel {
            tx,
            rx: Arc::new(Mutex::new(Some(rx))),
            pool,
            counters,
        }
    }

    /// A buffer to encode into: recycled if one is free, otherwise a fresh
    /// one. Never blocks and never fails.
    fn take(&self) -> Vec<u8> {
        match self.pool.lock() {
            Ok(mut p) => p.pop_front().unwrap_or_default(),
            // A poisoned pool means a sender thread panicked mid-push.
            // Losing buffer reuse is harmless; blocking is not.
            Err(_) => Vec::new(),
        }
    }

    fn put(&self, mut buf: Vec<u8>) {
        if buf.is_empty() {
            return;
        }
        if let Ok(mut p) = self.pool.lock() {
            if p.len() < POOL_DEPTH {
                buf.iter_mut().for_each(|b| *b = 0);
                p.push_back(buf);
            }
        }
    }

    /// Hand a finished datagram to the sender thread. Drops and counts rather
    /// than waiting when the sender is behind.
    fn offer(&self, buf: Vec<u8>) {
        self.counters.offered.fetch_add(1, Ordering::Relaxed);
        match self.tx.try_send(buf) {
            Ok(()) => {}
            Err(TrySendError::Full(b)) | Err(TrySendError::Disconnected(b)) => {
                self.counters.dropped.fetch_add(1, Ordering::Relaxed);
                self.put(b);
            }
        }
    }
}

/// One telemetry destination: a socket thread plus the two packet channels.
pub struct TelemetryLink {
    motion: Channel,
    car: Channel,
    counters: Arc<Counters>,
    header: PacketHeader,
    /// Previous longitudinal speed per car, for the g-force channel. Held
    /// here rather than in `RaceState` so it cannot reach the trajectory hash.
    prev_vel: Vec<f64>,
    frame_id: u32,
    sock: Arc<std::net::UdpSocket>,
}

impl TelemetryLink {
    /// Connect to `cfg.target` on a dedicated thread.
    ///
    /// Fails only on an address this process cannot use. A target that is
    /// simply *not listening* is not a failure here — that is the
    /// disconnected-receiver case the phase requires to be harmless, and it
    /// surfaces as `send_errors` at run time instead.
    pub fn spawn(cfg: &TelemetryConfig) -> Result<TelemetryLink, String> {
        let addr: std::net::SocketAddr = cfg
            .target
            .parse()
            .map_err(|_| format!("telemetry target '{}' is not host:port", cfg.target))?;
        let sock = std::net::UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| format!("telemetry socket bind: {e}"))?;
        sock.connect(addr)
            .map_err(|e| format!("telemetry connect {addr}: {e}"))?;
        // Connected UDP still reports ICMP errors from a closed port. An
        // effectively-zero write timeout makes those surface immediately
        // instead of stalling a send — the whole point is that the race
        // never waits on the network. A literal zero is rejected by Windows
        // ("cannot set a 0 duration timeout"), so this asks for the smallest
        // value the platform accepts; measured on the build machine, 100
        // sends to a dead port complete in about 1 ms total.
        sock.set_write_timeout(Some(Duration::from_micros(1)))
            .map_err(|e| format!("telemetry socket timeout: {e}"))?;

        let counters = Arc::new(Counters::default());
        let motion = Channel::new(MOTION_PACKET_SIZE, counters.clone());
        let car = Channel::new(CAR_TELEMETRY_PACKET_SIZE, counters.clone());
        let sock = Arc::new(sock);

        for ch in [&motion, &car] {
            let rx = Arc::clone(&ch.rx);
            let pool = Arc::clone(&ch.pool);
            let c = counters.clone();
            let s = Arc::clone(&sock);
            std::thread::Builder::new()
                .name("telemetry".into())
                .spawn(move || sender_thread(s, rx, pool, c))
                .map_err(|e| format!("telemetry thread: {e}"))?;
        }

        Ok(TelemetryLink {
            motion,
            car,
            counters,
            header: PacketHeader {
                session_uid: cfg.session_uid,
                session_time: 0.0,
                frame_id: 0,
                overall_frame_id: 0,
                player_car_index: cfg.player_car.min(255) as u8,
            },
            prev_vel: Vec::new(),
            frame_id: 0,
            sock,
        })
    }

    pub fn stats(&self) -> LinkStats {
        self.counters.snapshot()
    }

    /// True while the receiver end still looks attached. A UDP sender cannot
    /// truly know this — there is no handshake — so this reports whether the
    /// socket has taken an ICMP error back, which on loopback is the closest
    /// available proxy for "nothing is listening".
    pub fn receiver_reachable(&self) -> bool {
        match self.sock.take_error() {
            Ok(None) => true,
            Ok(Some(_)) | Err(_) => false,
        }
    }

    /// Longitudinal acceleration in m/s² per car, measured against the
    /// previous published frame. Needs memory, so it lives here rather than in
    /// the stateless encoders.
    pub fn measure_accel(&mut self, vel: &[f64], dt: f64) -> Vec<f64> {
        if self.prev_vel.len() != vel.len() {
            // First frame: seed with the current speed so acceleration starts
            // from zero rather than reporting a phantom step from rest.
            self.prev_vel = vel.to_vec();
            return vec![0.0; vel.len()];
        }
        let out: Vec<f64> = if dt > 0.0 {
            vel.iter()
                .zip(self.prev_vel.iter())
                .map(|(v, p)| (v - p) / dt)
                .collect()
        } else {
            vec![0.0; vel.len()]
        };
        self.prev_vel = vel.to_vec();
        out
    }

    /// Encode and hand off one frame at `sim_time_s`. Never blocks.
    pub fn publish(&mut self, sim_time_s: f64, cars: &[CarSample]) {
        self.frame_id = self.frame_id.wrapping_add(1);
        self.header.session_time = sim_time_s;
        self.header.frame_id = self.frame_id;
        self.header.overall_frame_id = self.frame_id;

        let mut buf = self.motion.take();
        if buf.len() != MOTION_PACKET_SIZE {
            buf.resize(MOTION_PACKET_SIZE, 0);
        }
        encode_motion(&mut buf, &self.header, cars);
        self.motion.offer(buf);

        let mut buf = self.car.take();
        if buf.len() != CAR_TELEMETRY_PACKET_SIZE {
            buf.resize(CAR_TELEMETRY_PACKET_SIZE, 0);
        }
        encode_car_telemetry(&mut buf, &self.header, cars);
        self.car.offer(buf);
    }
}

/// Drain one channel and send whatever is waiting, then recycle the buffers.
fn sender_thread(
    sock: Arc<std::net::UdpSocket>,
    rx: Arc<Mutex<Option<Receiver<Vec<u8>>>>>,
    pool: Arc<Mutex<VecDeque<Vec<u8>>>>,
    counters: Arc<Counters>,
) {
    let Some(rx) = rx.lock().ok().and_then(|mut g| g.take()) else {
        return;
    };
    loop {
        match rx.try_recv() {
            Ok(buf) => {
                match sock.send(&buf) {
                    Ok(n) => {
                        counters.sent.fetch_add(1, Ordering::Relaxed);
                        counters.bytes.fetch_add(n as u64, Ordering::Relaxed);
                    }
                    Err(_) => {
                        counters.send_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                if let Ok(mut p) = pool.lock() {
                    if p.len() < POOL_DEPTH {
                        let mut b = buf;
                        b.iter_mut().for_each(|x| *x = 0);
                        p.push_back(b);
                    }
                }
            }
            // Every sender handle is gone: this thread has nothing left to do.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Nothing queued. Sleep a fraction of the publish interval
                // rather than spinning a core beside the physics thread.
                std::thread::sleep(Duration::from_micros(500));
            }
        }
    }
}