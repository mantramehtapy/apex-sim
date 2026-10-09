//! Versioned wire protocol between the Rust server and the browser
//! dashboard (Phase 2). Every message carries the protocol version `v`,
//! the session id `sid`, and where relevant the simulation `tick` and a
//! monotonic `seq` so clients can detect gaps (dropped/replaced updates).

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u16 = 1;

/// Replaceable full-state snapshot, published at ~25 Hz via a watch channel
/// (latest value wins — old snapshots are dropped, physics never blocks).
#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub v: u16,
    pub sid: String,
    pub seq: u64,
    pub tick: u64,
    pub sim_time_s: f64,
    pub phase: String, // ready | running | paused | finished
    pub car: CarSnap,
    pub s: f64,
    pub laps_done: u32,
    pub lap_times: Vec<f64>,
    pub offtrack_ticks: u64,
    pub last_applied: Option<AppliedCmd>,
    pub server_now_ms: u64,
    /// Phase 4 race mode: the whole field. Empty in single-car mode.
    pub cars: Vec<CarBrief>,
    /// Phase 4 race mode: classification order.
    pub leaderboard: Vec<LbRow>,
    /// Phase 4 race mode: total car-to-car contacts so far this session.
    pub contacts_total: u64,
    /// Phase 5: session mode label (race / qualifying / practice).
    pub session: String,
    /// Phase 5: start-light count during the formation phase (0-5).
    pub lights: u8,
    /// Phase 8: live director-panel state at `tick`. `None` when the session
    /// has no weather at all, which is how the panel knows to hide itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weather: Option<WeatherBrief>,
    /// Phase 9: UDP link state at `tick`. `None` means no telemetry was
    /// requested, which is how the panel knows to hide itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<TelemetryBrief>,
    /// Phase 9: sound observer state at `tick`. `None` means sound is off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sound: Option<SoundBrief>,
    /// Realism pack race-control state (SC, yellow sectors).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rc: Option<RaceControlBrief>,
    /// Weekend qualifying state, on the snapshot while qualifying runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quali: Option<QualiBrief>,
}

/// Phase 9: what the UDP link has actually done. The pair that matters is
/// `offered` vs `sent` — a gap is the drop counter doing its job, and the
/// panel shows both so it cannot be mistaken for the race slowing down.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct TelemetryBrief {
    pub sent: u64,
    pub dropped: u64,
    pub send_errors: u64,
    pub bytes: u64,
    pub offered: u64,
    /// Whether the last send got no ICMP rejection. `false` means nothing is
    /// listening, which is a normal state and never affects the race.
    pub reachable: bool,
    pub hz: u32,
}

/// Phase 9: the sound vocabulary's live tally. `enabled` is what the panel
/// uses to know whether the mute button means anything.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SoundBrief {
    pub events: usize,
    pub max_cues_per_s: u32,
    /// `name: count` for each cue that has fired at least once, ordered by
    /// priority so the panel's list is stable.
    pub counts: Vec<(String, u32)>,
    pub log_hash: u64,
}

/// Phase 8: the race-director panel's live view of the sliders. Sent with
/// every snapshot rather than on a timer, so the panel can never show a value
/// the physics has already moved past.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct WeatherBrief {
    /// Preset name, or "dry" / "damp" / "wet" / "mixed" / "custom".
    pub label: String,
    pub water: f64,
    pub track_temp_c: f64,
    pub grip_scale: f64,
    /// Overrides queued but not yet reached — non-zero only while paused.
    pub queued: usize,
    /// Overrides that have taken effect so far this session.
    pub applied: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CarSnap {
    pub pos_x: f64,
    pub pos_y: f64,
    pub heading: f64,
    pub vel_long: f64,
    pub vel_lat: f64,
    pub yaw_rate: f64,
    pub steer: f64,
}

/// One car of the race field, for dashboard rendering.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CarBrief {
    pub id: u32,
    pub pos_x: f64,
    pub pos_y: f64,
    pub heading: f64,
    pub v: f64, // vel_long, m/s
    pub s: f64, // station within the lap, m
    pub lap: u32,
    pub retired: bool,
    pub in_pit: bool, // Phase 5: currently in the pit lane
    /// Realism pack: about to be lapped — must concede.
    pub blue_flag: bool,
    /// Realism pack: DRS active (in zone, within a second).
    pub drs: bool,
}


/// One row of the race classification.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct LbRow {
    pub id: u32,
    pub pos: u32,
    pub laps_done: u32,
    pub gap_s: Option<f64>,
    pub retired: bool,
    pub finished: bool,
    pub v: f64,
    // Phase 5: best lap, tire compound letter, stop count, in-pit flag.
    pub best_lap_s: Option<f64>,
    pub compound: &'static str,
    pub stops: u32,
    pub in_pit: bool,
    /// Realism pack: about to be lapped — must concede.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub blue_flag: bool,
    /// Realism pack: DRS active (in zone, within a second).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub drs: bool,
    /// Realism pack W3: fuel fraction (percent of capacity) and tyre temp
    /// (percent of warmed). None when the pack is off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuel: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tyre_temp: Option<u8>,
}


/// Round-trip record for a command: lets the client see when its command
/// was received, at which tick it was applied, and how stale it was.
#[derive(Clone, Debug, Serialize)]
pub struct AppliedCmd {
    pub cmd: String,
    pub nonce: u64,
    pub queued_ms: u64,
    pub applied_ms: u64,
    pub applied_tick: u64,
}

/// Weekend qualifying state: the car on track, the run counter, and the
/// order sheet so far (best lap ascending).
#[derive(Clone, Debug, Serialize)]
pub struct QualiRowBrief {
    pub car: u32,
    pub best_lap_s: Option<f64>,
    pub top_mps: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct QualiBrief {
    /// The car on track right now.
    pub car: u32,
    /// 1-based run index and the field size.
    pub run: u32,
    pub total: u32,
    /// Seconds of the window that remain for this run.
    pub remaining_s: f64,
    /// Rows so far, best lap ascending (the eventual grid).
    pub rows: Vec<QualiRowBrief>,
}

/// Realism pack race-control state, on the snapshot while anything is
/// active. `sc` is Some while the safety car is out; the yellow sectors are
/// (from_s, to_s) arcs around the lap.
#[derive(Clone, Debug, Serialize)]
pub struct RaceControlBrief {
    pub sc: Option<String>,
    pub yellows: Vec<[f64; 2]>,    /// Realism pack: the safety car itself, while deployed — (x, y, heading).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sc_car: Option<[f64; 3]>,
}


/// Track geometry for drawing, sent once in the welcome message.
#[derive(Clone, Debug, Serialize)]
pub struct TrackDef {
    pub straight_len: f64,
    pub radius: f64,
    pub half_width: f64,
    pub length: f64,
    /// Centreline polyline (closed loop), world metres.
    pub points: Vec<[f64; 2]>,
    /// Phase 5: pit lane world points (entry, box, exit), race mode only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pit: Option<PitLaneDef>,    #[serde(skip_serializing_if = "Option::is_none")]
    pub dressing: Option<TrackDressing>,
}


/// Pit lane polyline for the dashboard, in world metres.
#[derive(Clone, Debug, Serialize)]
pub struct PitLaneDef {
    pub points: Vec<[f64; 2]>, // entry, box, exit (at the lane offset)
    pub box_index: usize,      // which point is the service box
}

/// Realism pack circuit dressing: world-space geometry the dashboard paints
/// around the track — pit buildings, grandstands, gantry, gravel, kerbs.
/// Everything derives from the circuit's own centreline, so every circuit
/// (presets and editor tracks) dresses itself.
#[derive(Clone, Debug, Serialize)]
pub struct TrackDressing {
    /// Garage/pit building boxes along the pit lane: (x, y, heading, w, l).
    pub garages: Vec<[f64; 5]>,
    /// The pit wall: a polyline separating lane from track.
    pub pit_wall: Vec<[f64; 2]>,
    /// Grandstand blocks: (x, y, heading, w, l).
    pub stands: Vec<[f64; 5]>,
    /// The start/finish gantry: (x, y, heading, span).
    pub gantry: [f64; 4],
    /// Gravel traps: (x, y, heading, w, l) at the exit of the sharpest corners.
    pub gravel: Vec<[f64; 5]>,
    /// Run-off width painted around the whole track (metres beyond the edge).
    pub runoff_m: f64,
}

/// Important, non-replaceable events (never dropped silently; a gap notice
/// is emitted if the client falls behind).
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    Started,
    Paused,
    Resumed,
    Reset { seed: u64 },
    LapCompleted { n: u32, time_s: f64 },
    Finished,
    /// Phase 4: car-to-car contact (`retired` = car knocked out, if any).
    Contact { a: u32, b: u32, closing_speed: f64, retired: Option<u32> },
    /// Phase 5 race control: start lights and green.
    Lights { count: u8 },
    GreenFlag,
    /// Phase 5 race control: pit lane and fastest lap.
    PitEntered { car: u32, lap: u32 },
    PitServiceDone { car: u32, held_s: f64 },
    PitExited { car: u32, lane_s: f64 },
    FastestLap { car: u32, time_s: f64, lap: u32 },
    /// Phase 8: a race-director slider was queued for a future tick. The
    /// `tick` is the simulation tick it will take effect on — sent on receipt
    /// so the panel can show "at 12.0s", ahead of the physics actually moving.
    WeatherQueued { slider: &'static str, value: f64, tick: u64 },
    /// Phase 8: a queued slider command took effect. The enclosing message's
    /// `tick` is the effective simulation tick, which is the one the log
    /// records and a replay re-derives from.
    WeatherApplied { slider: &'static str, value: f64 },
    /// Realism pack: a car is out (spun / beached / mechanical).
    CarOut {
        car: u32,
        cause: Box<String>,
        detail: Box<String>,
    },
    /// Realism pack: safety car deployed or cleared.
    SafetyCar { phase: Box<String> },
    /// Weekend: a quali run ended (or the whole quali / race leg changed).
    WeekendPhase {
        phase: String,
        car: u32,
        run: u32,
        total: u32,
        best_lap_s: Option<f64>,
    },
    /// Phase 9: a sound cue was extracted. Forwarded to the browser so the
    /// dashboard can drive Web Audio from the same ordered log the replay
    /// reproduces — the browser is a sound consumer, not a second extractor.
    SoundCueFired { car: u32, cue: String, value: f64 },
    /// The command reached the simulation thread and could not be carried out.
    /// Distinct from a `rejected_*` ack, which means it never got this far.
    CommandRejected { cmd: String, nonce: u64, reason: String },
    CommandApplied { cmd: String, nonce: u64, applied_tick: u64 },
}

#[derive(Clone, Debug, Serialize)]
pub struct EventMsg {
    pub v: u16,
    pub sid: String,
    pub event: EventKind,
    pub tick: u64,
    pub server_now_ms: u64,
}

/// Server → client messages.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)] // Welcome carries the track once
pub enum ToClient {
    Welcome {
        v: u16,
        sid: String,
        track: TrackDef,
        snapshot: Snapshot,
        server_now_ms: u64,
    },
    Snapshot(Snapshot),
    Event(EventMsg),
    Ack {
        v: u16,
        sid: String,
        nonce: u64,
        received_ms: u64,
        status: &'static str, // "queued" | "rejected_stale_session" | "rejected_unknown_command" | "rejected_bad_version"
        detail: String,
    },
}

/// Client → server command envelope. `sid` must match the current session;
/// commands from an older session are rejected as stale.
#[derive(Clone, Debug, Deserialize)]
pub struct InMsg {
    pub v: u16,
    pub sid: String,
    pub nonce: u64,
    /// `start` | `pause` | `resume` | `reset` | `slider:<name>:<value>`.
    pub cmd: String,
}

/// Why a command string was refused. Split from the reason text because the
/// ack status is part of the Phase 2 contract: an unrecognised verb stays
/// `rejected_unknown_command`, and a real command with an unusable argument
/// gets its own status instead of being filed under "unknown".
#[derive(Clone, Debug, PartialEq)]
pub enum CommandError {
    /// Not a command at all.
    Unknown(String),
    /// A real command, with a bad argument.
    Invalid(String),
}

impl CommandError {
    pub fn status(&self) -> &'static str {
        match self {
            CommandError::Unknown(_) => "rejected_unknown_command",
            CommandError::Invalid(_) => "rejected_bad_command",
        }
    }
    pub fn detail(&self) -> &str {
        match self {
            CommandError::Unknown(d) | CommandError::Invalid(d) => d,
        }
    }
}

/// Check a command string before it is queued, so a malformed one is answered
/// with a reason instead of being dropped at the tick boundary.
///
/// Slider values are range-checked here rather than clamped: a director who
/// asks for 3.0 units of water should be told the scale stops at 1.0, not
/// silently given 1.0.
pub fn validate_command(cmd: &str) -> Result<(), CommandError> {
    let c = cmd.trim();
    if matches!(c, "start" | "pause" | "resume" | "reset") {
        return Ok(());
    }
    if c.starts_with("slider:") {
        return crate::weather::parse_slider_cmd(c)
            .map(|_| ())
            .map_err(CommandError::Invalid);
    }
    Err(CommandError::Unknown(format!(
        "unknown command '{c}' (expected start | pause | resume | reset | slider:<name>:<value>)"
    )))
}

/// Validated dashboard settings (HTTP POST /api/config). Applied at the
/// next reset — never mid-race.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConfigUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub laps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Phase 8: the surface layout the next session starts with. Mid-race
    /// changes go through the `slider:` command, which is tick-quantised; a
    /// preset needs no such care because it takes effect at the reset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weather: Option<String>,
    /// Phase 8: slider values the next session starts at. Clamped to the
    /// slider's own range rather than rejected, so the panel can show a
    /// session that was set up from a URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub water: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_temp_c: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grip: Option<f64>,
    /// Track editor (Phase D): the full circuit — including control points —
    /// the NEXT session runs on. Validated at POST time (an undriveable
    /// track is refused where the dashboard can show the error), applied at
    /// the reset like every other staged setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track: Option<crate::track::TrackSpec>,
}

pub fn validate_update(u: &ConfigUpdate) -> Result<(), String> {
    if let Some(laps) = u.laps {
        if !(1..=1000).contains(&laps) {
            return Err(format!("laps must be 1..=1000, got {laps}"));
        }
    }
    if let Some(noise) = u.noise {
        if !(0.0..=0.1).contains(&noise) {
            return Err(format!("noise must be 0.0..=0.1, got {noise}"));
        }
    }
    if let Some(mode) = &u.mode {
        if crate::race::SessionMode::parse(mode).is_none() {
            return Err(format!(
                "mode must be race | qualifying | practice, got '{mode}'"
            ));
        }
    }
    if let Some(w) = &u.weather {
        if crate::weather::preset_zones(w).is_none() {
            return Err(format!(
                "weather must be dry | damp | wet | mixed, got '{w}'"
            ));
        }
    }
    for (name, v) in [
        ("water", u.water),
        ("track_temp_c", u.track_temp_c),
        ("grip", u.grip),
    ] {
        if let Some(v) = v {
            if !v.is_finite() {
                return Err(format!("{name} must be a finite number"));
            }
        }
    }
    Ok(())
}

/// Server-side observability counters (GET /api/metrics): snapshot age and
/// dropped/replaced updates must be observable per the Phase 2 gate.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Metrics {
    pub ws_clients: u64,
    pub snapshots_published: u64,
    pub ws_snapshots_sent: u64,
    pub ws_snapshots_dropped: u64,
    pub events_published: u64,
    pub events_lagged: u64,
    pub commands_received: u64,
    pub commands_rejected: u64,
    pub uptime_ms: u64,
}
