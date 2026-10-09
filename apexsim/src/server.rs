//! Phase 2 server: HTTP for initial configuration, WebSocket for live
//! snapshots (~25 Hz) and commands.
//!
//! Separation rules from the plan:
//! - Physics runs on a dedicated std thread; the web layer only *reads*
//!   copies of snapshots. Slow or dead clients cannot stall the sim.
//! - Snapshots are replaceable (tokio watch = bounded latest-value queue);
//!   important events ride a separate broadcast channel and are never
//!   silently replaced — a gap notice is emitted instead.
//! - Every message is versioned and carries session id / tick / seq.

use crate::logging::{Level, Logger};
use crate::protocol::{
    AppliedCmd, CarBrief, CarSnap, ConfigUpdate, EventKind, EventMsg, InMsg, LbRow, Metrics,
    PitLaneDef, PROTOCOL_VERSION, RaceControlBrief, Snapshot, SoundBrief, TelemetryBrief,
    ToClient, TrackDef, TrackDressing,
    WeatherBrief, validate_command,
    validate_update,
};
use crate::race::{PitConfig, Race, RaceConfig, RacePhase, RcEvent, SessionMode};
use crate::sim::{Phase, Sim, SimConfig, DT};
use crate::track::TrackSpec;
use crate::weather::{self, WeatherConfig};
use axum::{
    extract::{State, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use axum::extract::ws::{Message, WebSocket};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, watch};

/// Command handed from the web layer to the sim thread. The sim applies
/// commands at tick boundaries and records them in the next snapshot.
#[derive(Clone, Debug)]
pub struct SimCommand {
    /// `start` | `pause` | `resume` | `reset` | `slider:<name>:<value>`.
    pub cmd: String,
    pub nonce: u64,
    pub queued_ms: u64,
}

/// How far ahead of the current tick a director slider command lands.
///
/// The web layer is asynchronous, so "now" would mean "at whatever tick the
/// sim thread happened to reach when it drained the queue" — a value that
/// differs run to run and cannot be replayed. One tick is the smallest
/// lookahead that makes the effective tick knowable at the moment of queuing:
/// the command is acknowledged with the exact tick it will take effect on,
/// and the log records that tick. At 250 Hz this is 4 ms — below the delay a
/// driver would ever perceive, and far below the point where the queue could
/// be starved by a burst of slider moves.
pub const SLIDER_LEAD_TICKS: u64 = 1;

/// Outcome of a director slider command reaching the simulation thread.
#[derive(Clone, Debug, PartialEq)]
pub enum SliderOutcome {
    /// Queued. The tick is the one it will take effect on.
    Queued {
        slider: weather::Slider,
        value: f64,
        tick: u64,
    },
    /// Well-formed but not carried out — the session has no live panel.
    Rejected(String),
}

/// Queue a `slider:<name>:<value>` command for a known tick.
///
/// Called from the sim thread's command loop, at a tick boundary, and public so
/// the gate tests drive the same code the dashboard does rather than a copy of
/// it. The value is queued, never applied inline: [`Race::queue_override`] puts
/// it on the same schedule the command line uses.
pub fn apply_slider_command(race: &mut Race, cmd: &str) -> SliderOutcome {
    let (slider, value) = match weather::parse_slider_cmd(cmd) {
        Ok(v) => v,
        Err(e) => return SliderOutcome::Rejected(e),
    };
    let at = race.state.tick + SLIDER_LEAD_TICKS;
    match race.queue_override(slider, value, at) {
        Ok(tick) => SliderOutcome::Queued { slider, value, tick },
        Err(e) => SliderOutcome::Rejected(e),
    }
}

pub struct WebState {
    pub snap_rx: watch::Receiver<Snapshot>,
    pub event_tx: broadcast::Sender<EventMsg>,
    pub cmd_tx: std_mpsc::Sender<SimCommand>,
    pub pending_config: Arc<Mutex<Option<ConfigUpdate>>>,
    pub current_config: Arc<Mutex<PublicConfig>>,
    pub metrics: Arc<Mutex<Metrics>>,
    pub started: Instant,
    pub track: Arc<std::sync::RwLock<TrackDef>>,
    /// Phase 10: the tournament running behind this dashboard, if any. The
    /// server is the batch's window — without it a long batch is a silent
    /// process, and the plan requires it to stay observable and stoppable.
    pub batch: Arc<BatchProgress>,
}

// ---------------------------------------------------------------------------
// Tournament progress (Phase 10)
// ---------------------------------------------------------------------------

/// Live progress of a headless tournament, shared with the HTTP layer.
///
/// Deliberately lock-light: the batch writes plain atomics on its own thread
/// and the dashboard reads them, so observing a batch can never slow the
/// races down. A `Mutex` around a struct here would be correct too, but the
/// whole point is that a long batch stays cheap to watch.
#[derive(Debug, Default)]
pub struct BatchProgress {
    name: Mutex<String>,
    done: AtomicUsize,
    total: AtomicUsize,
    current: Mutex<String>,
    /// Set by `POST /api/batch/stop`; read by the batch between races.
    stop: Arc<AtomicBool>,
    /// Trajectory hash of the most recently finished race, so a watcher can
    /// see results arriving rather than only progress.
    last_hash: AtomicU64,
    finished: AtomicBool,
    started: AtomicBool,
}

impl BatchProgress {
    pub fn new() -> BatchProgress {
        BatchProgress::default()
    }
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }
    pub fn begin(&self, name: &str, total: usize) {
        *self.name.lock().unwrap() = name.to_string();
        self.total.store(total, Ordering::SeqCst);
        self.done.store(0, Ordering::SeqCst);
        self.finished.store(false, Ordering::SeqCst);
        self.started.store(true, Ordering::SeqCst);
    }
    pub fn progress(&self, done: usize, total: usize, label: &str) {
        self.done.store(done, Ordering::SeqCst);
        self.total.store(total, Ordering::SeqCst);
        *self.current.lock().unwrap() = label.to_string();
    }
    /// Record the trajectory hash of a finished race. Named `record_hash`
    /// rather than `hash` so it cannot be mistaken for `Hash::hash`.
    pub fn record_hash(&self, h: u64) {
        self.last_hash.store(h, Ordering::SeqCst);
    }
    pub fn end(&self) {
        self.finished.store(true, Ordering::SeqCst);
        *self.current.lock().unwrap() = String::new();
    }
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
    pub fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "active": self.started.load(Ordering::SeqCst) && !self.finished.load(Ordering::SeqCst),
            "finished": self.finished.load(Ordering::SeqCst),
            "name": self.name.lock().unwrap().clone(),
            "done": self.done.load(Ordering::SeqCst),
            "total": self.total.load(Ordering::SeqCst),
            "current": self.current.lock().unwrap().clone(),
            "stop_requested": self.stop_requested(),
            "last_state_hash": format!("{:016x}", self.last_hash.load(Ordering::SeqCst)),
        })
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PublicConfig {
    pub seed: u64,
    pub laps: u32,
    pub noise: f64,
    pub mode: String,
    /// Phase 8: the scenario a reset will start from, and the slider values it
    /// will start at. Live values travel in `Snapshot::weather`; these are the
    /// session's starting point, which is what the settings form edits.
    pub weather: String,
    pub water: f64,
    pub track_temp_c: f64,
    pub grip: f64,
}

/// The settings form's view of a session, from the sim config that built it.
fn public_config(c: &SimConfig) -> PublicConfig {
    let w = c.weather.clone().unwrap_or_else(WeatherConfig::director);
    PublicConfig {
        seed: c.seed,
        laps: c.laps,
        noise: c.driver.noise_amplitude,
        mode: c.mode.clone(),
        weather: w.preset.clone().unwrap_or_else(|| w.label().to_string()),
        water: w.water,
        track_temp_c: w.track_temp_c,
        grip: w.grip_scale,
    }
}

fn _assert_webstate_send_sync() {
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<WebState>();
    is_send_sync::<Arc<WebState>>();
    is_send_sync::<Snapshot>();
    is_send_sync::<EventMsg>();
    is_send_sync::<SimCommand>();
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn make_sid(seed: u64) -> String {
    format!("s{seed}-{}", unix_ms())
}

/// Weekend orchestrator (Phase W1): quali runs car-by-car live on the
/// dashboard, then the race leg with the grid from the quali order. Lives
/// inside the race loop; each leg is a full `Race` rebuild through the same
/// checklist the reset command uses (new sid, cursors zeroed), so clients
/// resync on the sid change for free.
struct WeekendOrch {
    window_s: f64,
    total: usize,
    /// 0-based index of the car currently on track (quali) or past quali.
    run: usize,
    rows: Vec<crate::weekend::QualiRow>,
    stage: WeekendStage,
    leg_start_tick: u64,
}

#[derive(Clone, Copy, PartialEq)]
enum WeekendStage {
    Quali,
    Race,
}

fn boot_snapshot() -> Snapshot {
    Snapshot {
        v: PROTOCOL_VERSION,
        sid: "boot".into(),
        seq: 0,
        tick: 0,
        sim_time_s: 0.0,
        phase: "ready".into(),
        car: CarSnap {
            pos_x: 0.0,
            pos_y: 0.0,
            heading: 0.0,
            vel_long: 0.0,
            vel_lat: 0.0,
            yaw_rate: 0.0,
            steer: 0.0,
        },
        s: 0.0,
        laps_done: 0,
        lap_times: vec![],
        offtrack_ticks: 0,
        last_applied: None,
        server_now_ms: 0,
        cars: vec![],
        leaderboard: vec![],
        contacts_total: 0,
        session: "race".into(),
        lights: 0,
        weather: None,
        telemetry: None,
        sound: None,
        rc: None,
        quali: None,
    }
}

// ---------------------------------------------------------------------------
// Simulation thread
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn sim_server_loop(
    mut config: SimConfig,
    autostart: bool,
    snap_tx: watch::Sender<Snapshot>,
    event_tx: broadcast::Sender<EventMsg>,
    cmd_rx: std_mpsc::Receiver<SimCommand>,
    pending: Arc<Mutex<Option<ConfigUpdate>>>,
    current: Arc<Mutex<PublicConfig>>,
    metrics: Arc<Mutex<Metrics>>,
    logger: Option<Arc<Logger>>,
) {
    let _timer = crate::sim::TimerPeriod::acquire();
    let mut sim = Sim::new(config.clone());
    let mut sid = make_sid(config.seed);
    let mut seq: u64 = 0;
    let mut last_applied: Option<AppliedCmd> = None;
    if let Some(l) = &logger {
        l.log(Level::Info, None, None, &format!(
            "sim loop started (cars 1, laps {}, mode single-car)", config.laps
        ));
    }
    sim.state.phase = if autostart { Phase::Running } else { Phase::Ready };
    if autostart {
        if let Some(l) = &logger {
            l.log(Level::Info, Some(&sid), Some(0), "session started (autostart)");
        }
        let _ = event_tx.send(EventMsg { v: PROTOCOL_VERSION, sid: sid.clone(), event: EventKind::Started, tick: 0, server_now_ms: unix_ms() });
    }

    let publish = |sim: &Sim, sid: &str, seq: &mut u64, last_applied: &Option<AppliedCmd>| {
        *seq += 1;
        let c = &sim.state.car;
        let snap = Snapshot {
            v: PROTOCOL_VERSION,
            sid: sid.to_string(),
            seq: *seq,
            tick: sim.state.tick,
            sim_time_s: sim.state.sim_time,
            phase: phase_str(sim.state.phase).to_string(),
            car: CarSnap {
                pos_x: c.pos_x,
                pos_y: c.pos_y,
                heading: c.heading,
                vel_long: c.vel_long,
                vel_lat: c.vel_lat,
                yaw_rate: c.yaw_rate,
                steer: c.steer,
            },
            s: sim.state.s_progress,
            laps_done: sim.state.laps_done,
            lap_times: sim.state.lap_times.clone(),
            offtrack_ticks: sim.state.offtrack_ticks,
            last_applied: last_applied.clone(),
            server_now_ms: unix_ms(),
            cars: vec![],
            leaderboard: vec![],
            contacts_total: 0,
            session: "race".into(),
            lights: 0,
            weather: None,
            telemetry: None,
            sound: None,
            rc: None,
            quali: None,
        };
        let _ = snap_tx.send(snap); // watch: overwrites, never blocks
        if let Ok(mut m) = metrics.lock() {
            m.snapshots_published += 1;
        }
    };

    publish(&sim, &sid, &mut seq, &last_applied);

    let mut last_pub = Instant::now();
    let mut deadline = Instant::now();
    let mut prev_laps = 0u32;
    let mut prev_phase = sim.state.phase;

    loop {
        // ---- Apply queued commands at the tick boundary ----
        while let Ok(cmd) = cmd_rx.try_recv() {
            let applied_ms = unix_ms();
            let applied_tick = sim.state.tick;
            let mut kind: Option<EventKind> = None;
            match cmd.cmd.as_str() {
                // Single-car mode is the Phase 1-3 engine: there are no sliders
                // to move, and saying so beats dropping the command silently.
                s if s.starts_with("slider:") => {
                    let _ = event_tx.send(EventMsg {
                        v: PROTOCOL_VERSION,
                        sid: sid.clone(),
                        event: EventKind::CommandRejected {
                            cmd: cmd.cmd.clone(),
                            nonce: cmd.nonce,
                            reason: "this session is single-car; run with --cars 2 or more for the race-director panel".into(),
                        },
                        tick: applied_tick,
                        server_now_ms: applied_ms,
                    });
                    continue;
                }
                "start" => {
                    if sim.state.phase == Phase::Ready {
                        sim.state.phase = Phase::Running;
                        kind = Some(EventKind::Started);
                    }
                }
                "pause" => {
                    if sim.state.phase == Phase::Running {
                        sim.pause();
                        kind = Some(EventKind::Paused);
                    }
                }
                "resume" => {
                    if sim.state.phase == Phase::Paused {
                        sim.resume();
                        kind = Some(EventKind::Resumed);
                    }
                }
                "reset" => {
                    if let Some(up) = pending.lock().ok().and_then(|mut p| p.take()) {
                        if let Some(seed) = up.seed {
                            config.seed = seed;
                        }
                        if let Some(laps) = up.laps {
                            config.laps = laps;
                        }
                        if let Some(noise) = up.noise {
                            config.driver.noise_amplitude = noise;
                        }
                        if let Some(mode) = up.mode {
                            config.mode = mode;
                        }
                        // A preset or slider change from the settings form
                        // is a *starting* value, applied here at the reset.
                        if let Some(w) = up.weather.as_deref() {
                            if let Some(mut base) = WeatherConfig::preset(w) {
                                base.live = true;
                                config.weather = Some(base);
                            }
                        }
                        if let Some(w) = config.weather.as_mut() {
                            if let Some(v) = up.water {
                                w.water = weather::Slider::Water.clamp(v);
                            }
                            if let Some(v) = up.track_temp_c {
                                w.track_temp_c = weather::Slider::TrackTemp.clamp(v);
                            }
                            if let Some(v) = up.grip {
                                w.grip_scale = weather::Slider::Grip.clamp(v);
                            }
                        }
                        if let Ok(mut c) = current.lock() {
                            *c = public_config(&config);
                        }
                    }
                    sim.reset(config.seed);
                    sim.state.phase = Phase::Ready;
                    sid = make_sid(config.seed);
                    seq = 0;
                    prev_laps = 0;
                    prev_phase = Phase::Ready;
                    kind = Some(EventKind::Reset { seed: config.seed });
                }
                _ => {}
            }
            if let Some(k) = kind.clone() {
                let _ = event_tx.send(EventMsg {
                    v: PROTOCOL_VERSION,
                    sid: sid.clone(),
                    event: EventKind::CommandApplied { cmd: cmd.cmd.clone(), nonce: cmd.nonce, applied_tick },
                    tick: applied_tick,
                    server_now_ms: applied_ms,
                });
                let _ = event_tx.send(EventMsg { v: PROTOCOL_VERSION, sid: sid.clone(), event: k, tick: applied_tick, server_now_ms: applied_ms });
            }
            last_applied = Some(AppliedCmd {
                cmd: cmd.cmd,
                nonce: cmd.nonce,
                queued_ms: cmd.queued_ms,
                applied_ms,
                applied_tick,
            });
            publish(&sim, &sid, &mut seq, &last_applied);
            last_pub = Instant::now();
        }

        // ---- Step physics at the fixed rate while running ----
        if sim.state.phase == Phase::Running {
            let now = Instant::now();
            if now < deadline {
                crate::sim::wait_until(deadline);
            }
            deadline += Duration::from_secs_f64(DT);
            sim.tick_with_driver();
        } else {
            std::thread::sleep(Duration::from_millis(40));
            deadline = Instant::now();
        }

        // ---- Race events (separate channel, never replaced) ----
        if sim.state.laps_done > prev_laps {
            prev_laps = sim.state.laps_done;
            let _ = event_tx.send(EventMsg {
                v: PROTOCOL_VERSION,
                sid: sid.clone(),
                event: EventKind::LapCompleted { n: sim.state.laps_done, time_s: *sim.state.lap_times.last().unwrap_or(&0.0) },
                tick: sim.state.tick,
                server_now_ms: unix_ms(),
            });
        }
        if sim.state.phase != prev_phase {
            prev_phase = sim.state.phase;
            if sim.state.phase == Phase::Finished {
                let _ = event_tx.send(EventMsg {
                    v: PROTOCOL_VERSION,
                    sid: sid.clone(),
                    event: EventKind::Finished,
                    tick: sim.state.tick,
                    server_now_ms: unix_ms(),
                });
            }
        }

        // ---- Publish snapshots at ~25 Hz (every 10 ticks) ----
        let due = sim.state.phase == Phase::Running
            && sim.state.tick.is_multiple_of(10)
            && last_pub.elapsed() >= Duration::from_millis(30);
        let due_idle = sim.state.phase != Phase::Running && last_pub.elapsed() >= Duration::from_millis(250);
        if due || due_idle {
            publish(&sim, &sid, &mut seq, &last_applied);
            last_pub = Instant::now();
        }
    }
}

fn phase_str(p: Phase) -> &'static str {
    match p {
        Phase::Ready => "ready",
        Phase::Running => "running",
        Phase::Paused => "paused",
        Phase::Finished => "finished",
    }
}

// ---------------------------------------------------------------------------
// Race-mode sim thread (Phase 4): drives race::Race — the whole field with
// traffic awareness, contacts and classification — with the same realtime
// pacing, command handling and snapshot/event plumbing as the single-car
// loop. One authoritative physics owner; the web layer only reads copies.
// ---------------------------------------------------------------------------

fn race_phase_str(p: RacePhase, paused: bool) -> &'static str {
    match p {
        RacePhase::Ready => "ready",
        RacePhase::FormationLap => "formation-lap",
        RacePhase::Formation => "formation",
        RacePhase::Running => {
            if paused {
                "paused"
            } else {
                "running"
            }
        }
        RacePhase::Finished => "finished",
    }
}

#[allow(clippy::too_many_arguments)]
fn race_server_loop(
    mut config: SimConfig,
    autostart: bool,
    snap_tx: watch::Sender<Snapshot>,
    event_tx: broadcast::Sender<EventMsg>,
    cmd_rx: std_mpsc::Receiver<SimCommand>,
    pending: Arc<Mutex<Option<ConfigUpdate>>>,
    current: Arc<Mutex<PublicConfig>>,
    metrics: Arc<Mutex<Metrics>>,
    logger: Option<Arc<Logger>>,
    track_slot: Arc<std::sync::RwLock<TrackDef>>,
) {
    let _timer = crate::sim::TimerPeriod::acquire();

    let session_mode = |c: &SimConfig| {
        SessionMode::parse(&c.mode).unwrap_or(SessionMode::Race)
    };

    let race_cfg = |c: &SimConfig| RaceConfig {
        cars: c.cars.max(1),
        laps: c.laps,
        seed: c.seed,
        noise: c.driver.noise_amplitude,
        mode: session_mode(c),
        field: c.field,
        realism: c.realism,
        weather: c.weather.clone(),
        telemetry: c.telemetry.clone(),
        sound: c.sound,
        // Phase 10: the served race runs on the circuit the dashboard draws.
        // Before this it silently used `Track::default()` while `/api/track`
        // reported the configured one, so a non-default `--track-straight`
        // drew a different track from the one the cars were driving.
        track: TrackSpec {
            version: crate::track::TRACK_VERSION,
            straight_len: c.track_straight,
            radius: c.track_radius,
            half_width: c.track_half_width,
            custom: c.track_custom.clone(),
        },
        ..RaceConfig::default()
    };

    let mut cfg = race_cfg(&config);
    let mut race = Race::new(cfg.clone());
    let mut sid = make_sid(cfg.seed);
    let mut seq: u64 = 0;
    let mut last_applied: Option<AppliedCmd> = None;
    let mut paused = false;
    let mut rc_sent = 0usize;
    let mut sound_sent = 0usize;
    if let Some(l) = &logger {
        l.log(Level::Info, None, None, &format!(
            "race loop started (cars {}, laps {}, mode {}, seed {})",
            cfg.cars, cfg.laps, cfg.mode.label(), cfg.seed
        ));
    }
    // Weekend orchestrator: the served session opens with qualifying
    // (car-by-car solo windows) and hands off to the race on the quali grid.
    let mut weekend = config.weekend_quali.then(|| WeekendOrch {
        window_s: config.quali_window_s,
        total: config.cars as usize,
        run: 0,
        rows: Vec::new(),
        stage: WeekendStage::Quali,
        leg_start_tick: 0,
    });
    let mut leg_top_mps = 0.0f64;

    if autostart {
        race.start();
        if let Some(l) = &logger {
            l.log(Level::Info, Some(&sid), Some(0), &format!(
                "session started (autostart, {} cars, {} laps, {})",
                cfg.cars, cfg.laps, cfg.mode.label()
            ));
        }
        let _ = event_tx.send(EventMsg {
            v: PROTOCOL_VERSION,
            sid: sid.clone(),
            event: EventKind::Started,
            tick: 0,
            server_now_ms: unix_ms(),
        });
        // Weekend: the boot race becomes quali leg 0 immediately — the full
        // field never races before the grid is earned.
        if let Some(plan) = &mut weekend {
            let mut solo = race_cfg(&config);
            solo.solo_car = Some(0);
            solo.mode = crate::race::SessionMode::Qualifying;
            solo.formation = false;
            solo.grid_order.clear();
            solo.strategy = crate::race::PitStrategy::None;
            cfg = solo;
            race = Race::new(cfg.clone());
            race.start();
            plan.leg_start_tick = race.state.tick;
            leg_top_mps = 0.0;
            sid = make_sid(cfg.seed);
            seq = 0;
            *current.lock().unwrap() = public_config(&config);
            if let Some(l) = &logger {
                l.log(Level::Info, Some(&sid), Some(0), "weekend: qualifying leg 0");
            }
            let _ = event_tx.send(EventMsg {
                v: PROTOCOL_VERSION,
                sid: sid.clone(),
                event: EventKind::WeekendPhase {
                    phase: "quali-start".into(),
                    car: 0,
                    run: 1,
                    total: plan.total as u32,
                    best_lap_s: None,
                },
                tick: 0,
                server_now_ms: unix_ms(),
            });
        }
    } else {
        race.state.phase = RacePhase::Ready;
    }

    // Weekend quali brief slot: the weekend orchestrator writes it as runs
    // advance; publish reads it for the snapshot.
    let quali_now: Arc<std::sync::Mutex<Option<crate::protocol::QualiBrief>>> =
        Arc::new(std::sync::Mutex::new(None));
    let publish = |race: &Race,
                   paused: bool,
                   sid: &str,
                   seq: &mut u64,
                   last_applied: &Option<AppliedCmd>| {
        *seq += 1;
        let lead = race.state.leaderboard.first().map(|r| r.id as usize);
        let lead_entry = lead.map(|i| &race.state.cars[i]);
        let lead_car = lead_entry.map(|e| e.car);
        // Race-control brief (SC phase + yellow sectors) and the blue-flag
        // set, computed once per publish from the race itself.
        let (sc_phase, yellow_zones) = race.race_control_brief();
        let sc_pose = race.sc_car_pose();
        let rc = if sc_phase.is_some() || !yellow_zones.is_empty() {
            Some(RaceControlBrief {
                sc: sc_phase.map(|p| p.to_string()),
                yellows: yellow_zones.iter().map(|(a, b)| [*a, *b]).collect(),
                sc_car: sc_pose.map(|(x, y, h)| [x, y, h]),
            })
        } else {
            None
        };
        let blue_ids: std::collections::BTreeSet<u32> =
            race.blue_flagged_ids().into_iter().collect();
        let drs_ids: std::collections::BTreeSet<u32> =
            race.drs_active_ids().into_iter().collect();
        let snap = Snapshot {
            v: PROTOCOL_VERSION,
            sid: sid.to_string(),
            seq: *seq,
            tick: race.state.tick,
            sim_time_s: race.state.sim_time,
            phase: race_phase_str(race.state.phase, paused).to_string(),
            car: match lead_car {
                Some(c) => CarSnap {
                    pos_x: c.pos_x,
                    pos_y: c.pos_y,
                    heading: c.heading,
                    vel_long: c.vel_long,
                    vel_lat: c.vel_lat,
                    yaw_rate: c.yaw_rate,
                    steer: c.steer,
                },
                None => CarSnap {
                    pos_x: 0.0,
                    pos_y: 0.0,
                    heading: 0.0,
                    vel_long: 0.0,
                    vel_lat: 0.0,
                    yaw_rate: 0.0,
                    steer: 0.0,
                },
            },
            s: lead_entry.map(|e| e.s).unwrap_or(0.0),
            laps_done: race.state.leaderboard.first().map(|r| r.laps_done).unwrap_or(0),
            lap_times: lead_entry.map(|e| e.lap_times.clone()).unwrap_or_default(),
            offtrack_ticks: 0,
            last_applied: last_applied.clone(),
            server_now_ms: unix_ms(),
            cars: race
                .state
                .cars
                .iter()
                .map(|e| CarBrief {
                    id: e.id,
                    pos_x: e.car.pos_x,
                    pos_y: e.car.pos_y,
                    heading: e.car.heading,
                    v: e.car.vel_long,
                    s: e.s,
                    lap: e.laps_crossings,
                    retired: e.retired,
                    in_pit: e.pit_phase.in_pit(),
                    blue_flag: blue_ids.contains(&e.id),
                    drs: drs_ids.contains(&e.id),
                })
                .collect(),
            leaderboard: race
                .state
                .leaderboard
                .iter()
                .map(|r| LbRow {
                    id: r.id,
                    pos: r.pos,
                    laps_done: r.laps_done,
                    gap_s: r.gap_s,
                    retired: r.retired,
                    finished: r.finished,
                    v: r.v,
                    best_lap_s: r.best_lap_s,
                    compound: r.compound,
                    stops: r.stops,
                    in_pit: r.in_pit,
                    blue_flag: blue_ids.contains(&r.id),
                    drs: drs_ids.contains(&r.id),
                    fuel: r.fuel,
                    tyre_temp: r.tyre_temp,
                })
                .collect(),
            contacts_total: race.state.contacts.len() as u64,
            session: race.config.mode.label().to_string(),
            lights: race.state.lights,
            weather: race.weather_state().map(|w| WeatherBrief {
                label: w.label.to_string(),
                water: w.water,
                track_temp_c: w.track_temp_c,
                grip_scale: w.grip_scale,
                queued: w.queued,
                applied: w.applied,
            }),
            telemetry: race.telemetry_stats().map(|t| TelemetryBrief {
                sent: t.sent,
                dropped: t.dropped,
                send_errors: t.send_errors,
                bytes: t.bytes,
                offered: t.offered,
                reachable: race.telemetry_reachable(),
                hz: race.telemetry_hz(),
            }),
            sound: race.sound_runtime().map(|rt| {
                let sum = rt.summary();
                SoundBrief {
                    events: sum.events,
                    max_cues_per_s: rt.cfg.max_cues_per_s,
                    counts: sum
                        .lines()
                        .iter()
                        .filter_map(|l| l.split_once(' '))
                        .map(|(n, c)| (n.to_string(), c.parse().unwrap_or(0)))
                        .collect(),
                    log_hash: sum.log_hash,
                }
            }),
            rc,
            quali: quali_now.lock().ok().and_then(|q| q.clone()),
        };
        let _ = snap_tx.send(snap); // watch: overwrites, never blocks
        if let Ok(mut m) = metrics.lock() {
            m.snapshots_published += 1;
        }
    };

    publish(&race, paused, &sid, &mut seq, &last_applied);

    let mut last_pub = Instant::now();
    let mut deadline = Instant::now();
    let mut prev_leader_laps = 0u32;
    let mut contacts_sent = 0usize;
    let mut finished_sent = false;

    loop {
        // ---- Apply queued commands at the tick boundary ----
        while let Ok(cmd) = cmd_rx.try_recv() {
            let applied_ms = unix_ms();
            let applied_tick = race.state.tick;
            let mut kind: Option<EventKind> = None;
            match cmd.cmd.as_str() {
                s if s.starts_with("slider:") => {
                    match apply_slider_command(&mut race, s) {
                        SliderOutcome::Queued { slider, value, tick } => {
                            kind = Some(EventKind::WeatherQueued {
                                slider: slider.label(),
                                value,
                                tick,
                            });
                            if let Some(l) = &logger {
                                l.log(
                                    Level::Info,
                                    Some(&sid),
                                    Some(tick),
                                    &format!(
                                        "race director: {} -> {:.3} effective tick {}",
                                        slider.label(),
                                        value,
                                        tick
                                    ),
                                );
                            }
                        }
                        SliderOutcome::Rejected(reason) => {
                            let _ = event_tx.send(EventMsg {
                                v: PROTOCOL_VERSION,
                                sid: sid.clone(),
                                event: EventKind::CommandRejected {
                                    cmd: cmd.cmd.clone(),
                                    nonce: cmd.nonce,
                                    reason,
                                },
                                tick: applied_tick,
                                server_now_ms: applied_ms,
                            });
                            continue;
                        }
                    }
                }
                "start" => {
                    if race.state.phase == RacePhase::Ready {
                        race.start();
                        paused = false;
                        if let Some(l) = &logger {
                            l.log(Level::Info, Some(&sid), Some(applied_tick), "session started");
                        }
                        kind = Some(EventKind::Started);
                    }
                }
                "pause" => {
                    if race.state.phase != RacePhase::Ready
                        && race.state.phase != RacePhase::Finished
                        && !paused
                    {
                        paused = true;
                        kind = Some(EventKind::Paused);
                    }
                }
                "resume" => {
                    if paused {
                        paused = false;
                        kind = Some(EventKind::Resumed);
                    }
                }
                "reset" => {
                    if let Some(up) = pending.lock().ok().and_then(|mut p| p.take()) {
                        if let Some(seed) = up.seed {
                            config.seed = seed;
                        }
                        if let Some(laps) = up.laps {
                            config.laps = laps;
                        }
                        if let Some(noise) = up.noise {
                            config.driver.noise_amplitude = noise;
                        }
                        if let Some(mode) = up.mode {
                            config.mode = mode;
                        }
                        if let Some(t) = up.track {
                            // The editor's circuit becomes the session's: the
                            // fields race_cfg reads and the slot the dashboard
                            // draws from change together, at the reset.
                            config.track_custom = t.custom.clone();
                            config.track_half_width = t.half_width;
                            if t.custom.is_none() {
                                config.track_straight = t.straight_len;
                                config.track_radius = t.radius;
                            }
                            *track_slot.write().unwrap() = state_track_def(&t);
                        }
                        if let Ok(mut c) = current.lock() {
                            *c = public_config(&config);
                        }
                    }
                    cfg = race_cfg(&config);
                    race = Race::new(cfg.clone());
                    race.state.phase = RacePhase::Ready;
                    // Weekend mode: a staged track (editor "save & race")
                    // re-arms the CURRENT quali leg on the new circuit —
                    // otherwise the leg would sit at Ready waiting for a
                    // Start press nobody in the editor flow makes.
                    if let Some(plan) = &mut weekend {
                        if plan.stage == WeekendStage::Quali {
                            let mut solo = race_cfg(&config);
                            solo.solo_car = Some(plan.run as u32);
                            solo.mode = crate::race::SessionMode::Qualifying;
                            solo.formation = false;
                            solo.grid_order.clear();
                            solo.strategy = crate::race::PitStrategy::None;
                            cfg = solo;
                            race = Race::new(cfg.clone());
                            race.start();
                            plan.leg_start_tick = race.state.tick;
                            leg_top_mps = 0.0;
                        }
                    }
                    sid = make_sid(cfg.seed);
                    seq = 0;
                    prev_leader_laps = 0;
                    contacts_sent = 0;
                    rc_sent = 0;
                    sound_sent = 0;
                    finished_sent = false;
                    paused = false;
                    if let Some(l) = &logger {
                        l.log(Level::Info, Some(&sid), Some(0), &format!(
                            "session reset (seed {}, cars {}, laps {}, mode {})",
                            cfg.seed, cfg.cars, cfg.laps, cfg.mode.label()
                        ));
                    }
                    kind = Some(EventKind::Reset { seed: cfg.seed });
                }
                _ => {}
            }
            if let Some(k) = kind.clone() {
                let _ = event_tx.send(EventMsg {
                    v: PROTOCOL_VERSION,
                    sid: sid.clone(),
                    event: EventKind::CommandApplied {
                        cmd: cmd.cmd.clone(),
                        nonce: cmd.nonce,
                        applied_tick,
                    },
                    tick: applied_tick,
                    server_now_ms: applied_ms,
                });
                let _ = event_tx.send(EventMsg {
                    v: PROTOCOL_VERSION,
                    sid: sid.clone(),
                    event: k,
                    tick: applied_tick,
                    server_now_ms: applied_ms,
                });
            }
            last_applied = Some(AppliedCmd {
                cmd: cmd.cmd,
                nonce: cmd.nonce,
                queued_ms: cmd.queued_ms,
                applied_ms,
                applied_tick,
            });
            publish(&race, paused, &sid, &mut seq, &last_applied);
            last_pub = Instant::now();
        }

        // ---- Step the field at the fixed rate while the session is live ----
        // (Formation counts: the lights sequence ticks at the same rate.)
        let live = matches!(
            race.state.phase,
            RacePhase::Running | RacePhase::Formation | RacePhase::FormationLap
        ) && !paused;
        if live {
            let now = Instant::now();
            if now < deadline {
                crate::sim::wait_until(deadline);
            }
            deadline += Duration::from_secs_f64(DT);
            race.tick();
        } else {
            std::thread::sleep(Duration::from_millis(40));
            deadline = Instant::now();
        }

        // ---- Weekend quali brief refresh (cheap; the rows vec is <= field) ----
        if let Some(plan) = &weekend {
            if plan.stage == WeekendStage::Quali {
                let remaining = plan.window_s
                    - (race.state.tick.saturating_sub(plan.leg_start_tick)) as f64
                        * crate::sim::DT;
                let mut rows: Vec<crate::protocol::QualiRowBrief> = plan.rows.iter()
                    .map(|r| crate::protocol::QualiRowBrief {
                        car: r.car, best_lap_s: r.best_lap_s, top_mps: r.top_speed_mps,
                    })
                    .collect();
                // The live run's provisional lap sits on top of the sheet.
                let live_best = race.state.cars[0].best_lap;
                rows.push(crate::protocol::QualiRowBrief {
                    car: race.state.cars[0].id,
                    best_lap_s: live_best,
                    top_mps: leg_top_mps,
                });
                *quali_now.lock().unwrap() = Some(crate::protocol::QualiBrief {
                    car: race.state.cars[0].id,
                    run: plan.run as u32 + 1,
                    total: plan.total as u32,
                    remaining_s: remaining.max(0.0),
                    rows,
                });
            }
        }

        // ---- Race events (separate channel, never replaced) ----
        let leader_laps = race.state.leaderboard.first().map(|r| r.laps_done).unwrap_or(0);
        if leader_laps > prev_leader_laps {
            prev_leader_laps = leader_laps;
            let lead_id = race.state.leaderboard.first().map(|r| r.id as usize);
            let time_s = lead_id
                .and_then(|i| race.state.cars.get(i))
                .and_then(|e| e.lap_times.last().copied())
                .unwrap_or(0.0);
            let _ = event_tx.send(EventMsg {
                v: PROTOCOL_VERSION,
                sid: sid.clone(),
                event: EventKind::LapCompleted { n: leader_laps, time_s },
                tick: race.state.tick,
                server_now_ms: unix_ms(),
            });
        }
        // Race-control events (lights, green, pits, fastest lap) — appended
        // by the sim in order; forward new ones.
        while rc_sent < race.state.rc_events.len() {
            let kind = match race.state.rc_events[rc_sent] {
                RcEvent::Lights(n) => EventKind::Lights { count: n },
                RcEvent::Green => EventKind::GreenFlag,
                RcEvent::PitEntered { car, lap } => EventKind::PitEntered { car, lap },
                RcEvent::PitServiceDone { car, held_s } => {
                    EventKind::PitServiceDone { car, held_s }
                }
                RcEvent::PitExited { car, lane_s } => EventKind::PitExited { car, lane_s },
                RcEvent::FastestLap { car, time_s, lap } => {
                    EventKind::FastestLap { car, time_s, lap }
                }
                RcEvent::WeatherApplied { tick: _, slider, value } => {
                    EventKind::WeatherApplied { slider, value }
                }
                RcEvent::SafetyCar { phase } => {
                    EventKind::SafetyCar { phase: Box::new(phase.to_string()) }
                }
                RcEvent::CarOut { car, cause, detail } => {
                    EventKind::CarOut {
                        car,
                        cause: Box::new(cause.to_string()),
                        detail: Box::new(detail.to_string()),
                    }
                }
            };
            let tick = race.state.tick;
            let _ = event_tx.send(EventMsg {
                v: PROTOCOL_VERSION,
                sid: sid.clone(),
                event: kind,
                tick,
                server_now_ms: unix_ms(),
            });
            rc_sent += 1;
        }
        // Phase 9: forward newly extracted sound cues. The extraction is a
        // pure function of (tick, car state) and lives in the race, so the
        // browser is a *consumer* of the log, not a second extractor —
        // which is what keeps a paused/replayed/accelerated session showing
        // the same cues in the same order. Bounded per pass for the same
        // reason contacts are: a 20-car grid can produce a burst.
        let mut cues_this_pass = 0;
        while sound_sent < race.sound_runtime().map_or(0, |rt| rt.events.len()) && cues_this_pass < 32 {
            let ev = race.sound_runtime().expect("checked by loop condition").events[sound_sent];
            let _ = event_tx.send(EventMsg {
                v: PROTOCOL_VERSION,
                sid: sid.clone(),
                event: EventKind::SoundCueFired {
                    car: ev.car,
                    cue: ev.cue.name().to_string(),
                    value: ev.value,
                },
                tick: race.state.tick,
                server_now_ms: unix_ms(),
            });
            sound_sent += 1;
            cues_this_pass += 1;
        }
        // Forward new contacts (bounded per pass so a pile-up can't flood
        // the broadcast channel in one tick; the rest go out next pass).
        // Only meaningful hits ride the event channel — gentle grazes are
        // still counted in contacts_total, but would spam the log.
        let mut sent_this_pass = 0;
        while contacts_sent < race.state.contacts.len() && sent_this_pass < 8 {
            let c = &race.state.contacts[contacts_sent];
            if c.closing_speed >= 1.0 || c.retired.is_some() {
                let _ = event_tx.send(EventMsg {
                    v: PROTOCOL_VERSION,
                    sid: sid.clone(),
                    event: EventKind::Contact {
                        a: c.a,
                        b: c.b,
                        closing_speed: c.closing_speed,
                        retired: c.retired,
                    },
                    tick: c.tick,
                    server_now_ms: unix_ms(),
                });
            }
            contacts_sent += 1;
            sent_this_pass += 1;
        }
        // Weekend quali runs end on their window (or an early finish); the
        // race leg ends when the race ends. Both share the handoff below.
        let weekend_quali_running = weekend
            .as_ref()
            .map(|p| p.stage == WeekendStage::Quali)
            .unwrap_or(false);
        let window_expired = weekend_quali_running && {
            let plan = weekend.as_ref().unwrap();
            let window_ticks = (plan.window_s / crate::sim::DT) as u64;
            race.state.tick.saturating_sub(plan.leg_start_tick) >= window_ticks
        };
        let race_finished = race.state.phase == RacePhase::Finished && !finished_sent;
        if race_finished || window_expired {
            if race_finished {
            finished_sent = true;
            if let Some(l) = &logger {
                l.log(Level::Info, Some(&sid), Some(race.state.tick), &format!(
                    "session finished (t = {:.1} s, {} contacts)",
                    race.state.sim_time,
                    race.state.contacts.len()
                ));
            }
            let _ = event_tx.send(EventMsg {
                v: PROTOCOL_VERSION,
                sid: sid.clone(),
                event: EventKind::Finished,
                tick: race.state.tick,
                server_now_ms: unix_ms(),
            });
            }

            // ---- Weekend handoff: quali runs chain, then the race leg. ----
            // The rebuild mirrors the reset branch (new cfg, race, sid,
            // cursors zeroed) so clients resync on the sid change.
            if let Some(plan) = &mut weekend {
                if plan.stage == WeekendStage::Quali {
                    // Harvest the finished solo run.
                    let solo = &race.state.cars[0];
                    let row = crate::weekend::QualiRow::from_session(
                        solo.id,
                        solo.best_lap,
                        leg_top_mps,
                        solo.laps_crossings,
                    );
                    plan.rows.push(row.clone());
                    let (car_id, best, run1, total) = (
                        row.car,
                        row.best_lap_s,
                        plan.run as u32 + 1,
                        plan.total as u32,
                    );
                    *quali_now.lock().unwrap() = Some(crate::protocol::QualiBrief {
                        car: car_id,
                        run: run1,
                        total,
                        remaining_s: 0.0,
                        rows: plan.rows.iter().map(|r| crate::protocol::QualiRowBrief {
                            car: r.car, best_lap_s: r.best_lap_s, top_mps: r.top_speed_mps,
                        }).collect(),
                    });
                    plan.run += 1;
                    let next = if plan.run < plan.total {
                        Some(plan.run as u32)
                    } else {
                        None
                    };
                    let _ = event_tx.send(EventMsg {
                        v: PROTOCOL_VERSION,
                        sid: sid.clone(),
                        event: EventKind::WeekendPhase {
                            phase: "quali-run-done".into(),
                            car: car_id,
                            run: run1,
                            total,
                            best_lap_s: best,
                        },
                        tick: race.state.tick,
                        server_now_ms: unix_ms(),
                    });
                    if let Some(next_car) = next {
                        // Next solo leg. Built from the CURRENT config so a
                        // staged track (editor) or any staged setting rides
                        // into the next leg instead of being silently
                        // reverted to the boot circuit.
                        let mut solo = race_cfg(&config);
                        solo.solo_car = Some(next_car);
                        solo.mode = crate::race::SessionMode::Qualifying;
                        solo.formation = false;
                        solo.grid_order.clear();
                        solo.strategy = crate::race::PitStrategy::None;
                        cfg = solo;
                        race = Race::new(cfg.clone());
                        race.start();
                        plan.leg_start_tick = race.state.tick;
                        leg_top_mps = 0.0;
                        sid = make_sid(cfg.seed + plan.run as u64);
                        seq = 0;
                        prev_leader_laps = 0;
                        contacts_sent = 0;
                        rc_sent = 0;
                        sound_sent = 0;
                        finished_sent = false;
                        paused = false;
                        *current.lock().unwrap() = public_config(&config);
                        if let Some(l) = &logger {
                            l.log(Level::Info, Some(&sid), Some(0), &format!(
                                "weekend: qualifying leg {} (car {})",
                                plan.run + 1, next_car
                            ));
                        }
                        let _ = event_tx.send(EventMsg {
                            v: PROTOCOL_VERSION,
                            sid: sid.clone(),
                            event: EventKind::WeekendPhase {
                                phase: "quali-run".into(),
                                car: next_car,
                                run: plan.run as u32 + 1,
                                total,
                                best_lap_s: None,
                            },
                            tick: 0,
                            server_now_ms: unix_ms(),
                        });
                    } else {
                        // Quali complete: the race leg on the quali grid.
                        let mut race_leg = race_cfg(&config);
                        race_leg.grid_order = crate::weekend::grid_from_rows(&plan.rows);
                        race_leg.solo_car = None;
                        cfg = race_leg;
                        race = Race::new(cfg.clone());
                        race.start();
                        plan.stage = WeekendStage::Race;
                        sid = make_sid(cfg.seed + 900_000);
                        seq = 0;
                        prev_leader_laps = 0;
                        contacts_sent = 0;
                        rc_sent = 0;
                        sound_sent = 0;
                        finished_sent = false;
                        paused = false;
                        *current.lock().unwrap() = public_config(&config);
                        *quali_now.lock().unwrap() = None;
                        if let Some(l) = &logger {
                            l.log(Level::Info, Some(&sid), Some(0),
                                "weekend: race leg (grid by qualifying)");
                        }
                        let _ = event_tx.send(EventMsg {
                            v: PROTOCOL_VERSION,
                            sid: sid.clone(),
                            event: EventKind::WeekendPhase {
                                phase: "race".into(),
                                car: 0,
                                run: total,
                                total,
                                best_lap_s: None,
                            },
                            tick: 0,
                            server_now_ms: unix_ms(),
                        });
                    }
                }
            }
        }

        // ---- Publish snapshots at ~25 Hz (every 10 ticks) ----
        let due = live
            && race.state.tick.is_multiple_of(10)
            && last_pub.elapsed() >= Duration::from_millis(30);
        let due_idle = !live && last_pub.elapsed() >= Duration::from_millis(250);
        if due || due_idle {
            publish(&race, paused, &sid, &mut seq, &last_applied);
            last_pub = Instant::now();
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP + WebSocket layer
// ---------------------------------------------------------------------------

fn build_router(state: Arc<WebState>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/ws", get(ws_handler))
        .route("/api/health", get(health))
        .route("/api/state", get(api_state))
        .route("/api/track", get(api_track))
        .route("/api/tracks", post(api_tracks_save))
        .route("/api/config", get(get_config).post(post_config))
        .route("/api/metrics", get(get_metrics))
        .route("/api/batch", get(api_batch))
        .route("/api/batch/stop", post(api_batch_stop))
        .route("/cars/{name}", get(car_sprite))
        .route("/sounds/{name}", get(sound_asset))
        .with_state(state)
}

/// The user's 2026-season liveries, downsampled by tools/make_sprites.py
/// and embedded so the dashboard stays a single-binary deployment. Looked
/// up by file name; the id→livery grid order lives in the dashboard JS
/// (a 20-car default field runs the first 20 grid slots — Polaris
/// appears only in 22-car fields).
static CAR_SPRITES: &[(&str, &[u8])] = &[
    ("aurora_34", include_bytes!("../static/cars/aurora_34.png")),
    ("aurora_66", include_bytes!("../static/cars/aurora_66.png")),
    ("cobalt_72", include_bytes!("../static/cars/cobalt_72.png")),
    ("cobalt_92", include_bytes!("../static/cars/cobalt_92.png")),
    ("everest_36", include_bytes!("../static/cars/everest_36.png")),
    ("everest_83", include_bytes!("../static/cars/everest_83.png")),
    ("falcon_51", include_bytes!("../static/cars/falcon_51.png")),
    ("falcon_52", include_bytes!("../static/cars/falcon_52.png")),
    ("juniper_58", include_bytes!("../static/cars/juniper_58.png")),
    ("juniper_96", include_bytes!("../static/cars/juniper_96.png")),
    ("meridian_15", include_bytes!("../static/cars/meridian_15.png")),
    ("meridian_45", include_bytes!("../static/cars/meridian_45.png")),
    ("polaris_19", include_bytes!("../static/cars/polaris_19.png")),
    ("polaris_95", include_bytes!("../static/cars/polaris_95.png")),
    ("sable_41", include_bytes!("../static/cars/sable_41.png")),
    ("sable_93", include_bytes!("../static/cars/sable_93.png")),
    ("titan_64", include_bytes!("../static/cars/titan_64.png")),
    ("titan_65", include_bytes!("../static/cars/titan_65.png")),
    ("vortex_78", include_bytes!("../static/cars/vortex_78.png")),
    ("vortex_8", include_bytes!("../static/cars/vortex_8.png")),
    ("zenith_26", include_bytes!("../static/cars/zenith_26.png")),
    ("zenith_84", include_bytes!("../static/cars/zenith_84.png")),
];

async fn car_sprite(axum::extract::Path(name): axum::extract::Path<String>) -> Response {
    // URLs carry the extension (/cars/vortex_08.png); the table keys do not.
    let base = name.strip_suffix(".png").unwrap_or(&name);
    match CAR_SPRITES.iter().find(|(n, _)| *n == base) {
        Some((_, bytes)) => (
            [(axum::http::header::CONTENT_TYPE, "image/png")],
            *bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// The validated cue assets, generated by tools/make_sounds.py and embedded
/// for the same single-binary reason as the liveries. The browser is the
/// sound *consumer* — it plays these, and the simulation's extraction
/// decides which ones fire and in what order. Nothing is synthesised here.
static SOUND_ASSETS: &[(&str, &[u8])] = &[
    ("contact", include_bytes!("../static/sounds/contact.wav")),
    ("formation_tick", include_bytes!("../static/sounds/formation_tick.wav")),
    ("green_flag", include_bytes!("../static/sounds/green_flag.wav")),
    ("lap_complete", include_bytes!("../static/sounds/lap_complete.wav")),
    ("pit_entered", include_bytes!("../static/sounds/pit_entered.wav")),
    ("rev_limiter", include_bytes!("../static/sounds/rev_limiter.wav")),
    ("session_finish", include_bytes!("../static/sounds/session_finish.wav")),
    ("shift_down", include_bytes!("../static/sounds/shift_down.wav")),
    ("shift_up", include_bytes!("../static/sounds/shift_up.wav")),
    ("start_light", include_bytes!("../static/sounds/start_light.wav")),
    ("tyre_slip", include_bytes!("../static/sounds/tyre_slip.wav")),
    ("weather_change", include_bytes!("../static/sounds/weather_change.wav")),
    ("wind", include_bytes!("../static/sounds/wind.wav")),
];

async fn sound_asset(axum::extract::Path(name): axum::extract::Path<String>) -> Response {
    let base = name.strip_suffix(".wav").unwrap_or(&name);
    match SOUND_ASSETS.iter().find(|(n, _)| *n == base) {
        Some((_, bytes)) => (
            [(axum::http::header::CONTENT_TYPE, "audio/wav")],
            *bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn index() -> Response {
    ([(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")], INDEX_HTML).into_response()
}

async fn health(State(st): State<Arc<WebState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "uptime_ms": st.started.elapsed().as_millis() as u64 }))
}

/// Phase 10: how far the tournament behind this dashboard has got, and
/// whether anyone has asked it to stop. Polling this is how a long batch stays
/// observable without a second process watching it.
async fn api_batch(State(st): State<Arc<WebState>>) -> Json<serde_json::Value> {
    Json(st.batch.snapshot())
}

/// Ask the batch to stop after the race in flight. Deliberately not a kill:
/// the race in flight finishes and is still scored, because a half-run race
/// has no honest score.
async fn api_batch_stop(State(st): State<Arc<WebState>>) -> Json<serde_json::Value> {
    st.batch.request_stop();
    Json(serde_json::json!({ "ok": true, "stop_requested": true }))
}

async fn api_state(State(st): State<Arc<WebState>>) -> Json<Snapshot> {
    Json(st.snap_rx.borrow().clone())
}

async fn api_track(State(st): State<Arc<WebState>>) -> Json<TrackDef> {
    Json(st.track.read().unwrap().clone())
}

/// Build the dashboard's track definition from a spec. Shared by startup and
/// by the race loop's reset, so a track switched from the editor is drawn
/// with exactly the shape the first session drove.
pub fn state_track_def(spec: &TrackSpec) -> TrackDef {
    state_track_def_pub(spec)
}

/// Public wrapper for the gates: the dashboard's track definition, dressing
/// included.
pub fn state_track_def_pub(spec: &TrackSpec) -> TrackDef {
    let t = spec.build().expect("track spec must build (validated upstream)");
    let len = t.length();
    // Race mode sends the pit lane polyline so the dashboard can draw the
    // box lane. The resolved s values follow the same law Race::new applies:
    // the config's own values, or lap fractions on a custom circuit.
    // Resolved pit placement (config values, or lap fractions on a custom
    // circuit) — shared by the polyline and the dressing.
    let pc = {
        let mut pc = RaceConfig::default().pit;
        if let Some(c) = &spec.custom {
            let d = RaceConfig::default().pit;
            if pc.entry_s == d.entry_s && pc.box_s == d.box_s && pc.exit_s == d.exit_s {
                let f = c.pit.unwrap_or_default();
                pc.entry_s = f.entry * len;
                pc.box_s = f.box_frac * len;
                pc.exit_s = f.exit * len;
            }
        }
        pc
    };
    let pit = {
        let pc = &pc;
        if pc.usable(len) {
            let at = |s: f64| {
                let (x, y, h) = t.point_at(s);
                let (lx, ly) = (-h.sin(), h.cos());
                [x + lx * pc.offset_m, y + ly * pc.offset_m]
            };
            Some(PitLaneDef {
                points: vec![at(pc.entry_s), at(pc.box_s), at(pc.exit_s)],
                box_index: 1,
            })
        } else {
            None
        }
    };
    TrackDef {
        straight_len: spec.straight_len,
        radius: spec.radius,
        half_width: spec.half_width,
        length: len,
        points: (0..256)
            .map(|i| {
                let (x, y, _) = t.point_at(i as f64 * len / 256.0);
                [x, y]
            })
            .collect(),
        pit,
        dressing: Some(dress_track(&t, &pc)),
    }
}

/// Derive the circuit dressing from the geometry: garages along the pit
/// lane, the pit wall, a start/finish gantry, grandstands on the longest
/// straights, and gravel at the exit of the sharpest corners. Pure from the
/// track, so every circuit — presets and editor tracks — dresses itself.
fn dress_track(t: &crate::track::Track, pc: &PitConfig) -> TrackDressing {
    let len = t.length();
    let hw = t.half_width;
    let (entry_s, box_s, exit_s) = (pc.entry_s, pc.box_s, pc.exit_s);
    let at = |s: f64, off: f64| {
        let (x, y, h) = t.point_at(s);
        let (lx, ly) = (-h.sin(), h.cos());
        [x + lx * off, y + ly * off, h]
    };

    // Garages: 6 boxes along the entry->exit stretch, on the far side of
    // the lane (offset -7.5 lane => buildings at -14), facing the track.
    let mut garages = Vec::new();
    for k in 0..6u32 {
        // Spread around the BOX: the garage cluster belongs to the service
        // point, not the whole lane.
        let s = box_s + (k as f64 - 2.5) * 14.0;
        let [x, y, h] = at(s.rem_euclid(len), hw + 7.0);
        garages.push([x, y, h, 12.0, 10.0]);
    }
    // Pit wall: a polyline along the near side of the lane, spanning the
    // entry->exit stretch.
    let pit_span = (exit_s - entry_s).rem_euclid(len).max(1.0);
    let mut pit_wall = Vec::new();
    for k in 0..12u32 {
        let s = entry_s + pit_span * (k as f64 / 11.0);
        let [x, y, _h] = at(s.rem_euclid(len), -(hw + 1.0));
        pit_wall.push([x, y]);
    }

    // Grandstands: along the two longest straights, on the OUTSIDE (left of
    // travel on the stadium = +lateral; use the direction with room).
    let mut stands = Vec::new();
    for k in 0..3u32 {
        let s = len * (0.10 + 0.28 * k as f64);
        let [x, y, h] = at(s.rem_euclid(len), hw + 10.0);
        stands.push([x, y, h, 40.0, 12.0]);
    }
    let [x, y, h] = at(len * 0.60, -(hw + 10.0));
    stands.push([x, y, h, 50.0, 12.0]);

    // Gantry over the start/finish line.
    let [gx, gy, gh] = at(0.0, 0.0);
    let gantry = [gx, gy, gh, hw * 2.0 + 6.0];

    // Gravel: at the exit of the sharpest corners — sample curvature, pick
    // the top local maxima (separated by >= 15% of the lap).
    let mut peaks: Vec<(f64, f64)> = Vec::new(); // (s, |k|)
    let step = len / 240.0;
    for i in 0..240u32 {
        let s = i as f64 * step;
        let k = t.curvature_at(s).abs();
        let k_prev = t.curvature_at((s - step).rem_euclid(len)).abs();
        let k_next = t.curvature_at((s + step).rem_euclid(len)).abs();
        if k > 1.0 / 120.0 && k >= k_prev && k >= k_next {
            peaks.push((s, k));
        }
    }
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut chosen: Vec<f64> = Vec::new();
    for (s, _k) in peaks {
        if chosen.iter().all(|c: &f64| {
            let d = (*c - s).rem_euclid(len);
            d.min(len - d) > len * 0.15
        }) {
            chosen.push(s);
        }
        if chosen.len() == 4 {
            break;
        }
    }
    // The trap sits a little PAST the corner apex (exit side), outside.
    let gravel = chosen
        .iter()
        .map(|s| {
            let exit_s = (s + 25.0).rem_euclid(len);
            let [x, y, h] = at(exit_s, hw + 5.0);
            [x, y, h, 30.0, 16.0]
        })
        .collect();

    TrackDressing { garages, pit_wall, stands, gantry, gravel, runoff_m: hw + 12.0 }
}

/// POST /api/tracks — save a track file (what the dashboard editor writes).
/// The file lands in `tracks/<name>.json` under the working directory and is
/// validated with the same law as `--track`, so what is saved is drivable.
async fn api_tracks_save(
    State(_st): State<Arc<WebState>>,
    req: axum::extract::Request,
) -> axum::response::Response {
    let body = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": format!("unreadable body: {e}") })),
            )
                .into_response()
        }
    };
    let text = match std::str::from_utf8(&body) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": format!("not UTF-8: {e}") })),
            )
                .into_response()
        }
    };
    let spec = match TrackSpec::from_custom_json(text) {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": e })),
            )
                .into_response()
        }
    };
    if let Err(e) = crate::track::validate_track_spec(&spec) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": e })),
        )
            .into_response();
    }
    let name = &spec.custom.as_ref().expect("from_custom_json yields custom").name;
    // Filesystem-safe name: the file IS the track's handle.
    let safe: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let dir = std::path::Path::new("tracks");
    if let Err(e) = std::fs::create_dir_all(dir) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": format!("cannot create tracks/: {e}") })),
        )
            .into_response();
    }
    let path = dir.join(format!("{safe}.json"));
    if let Err(e) = std::fs::write(&path, text) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": format!("cannot write {}: {e}", path.display()) })),
        )
            .into_response();
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "file": path.display().to_string(), "name": name })),
    )
        .into_response()
}

async fn get_config(State(st): State<Arc<WebState>>) -> Json<serde_json::Value> {
    let cur = st.current_config.lock().unwrap().clone();
    let pending = st.pending_config.lock().unwrap().clone();
    Json(serde_json::json!({ "current": cur, "pending": pending }))
}

#[allow(clippy::type_complexity)]
async fn post_config(
    State(st): State<Arc<WebState>>,
    req: axum::extract::Request,
) -> Response {
    let body = match axum::body::to_bytes(req.into_body(), 64 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": format!("unreadable body: {e}") })),
            ).into_response();
        }
    };
    let up: ConfigUpdate = match serde_json::from_slice(&body) {
        Ok(u) => u,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": format!("invalid config body: {e}") })),
            ).into_response();
        }
    };
    if let Err(err) = validate_update(&up) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": err })),
        ).into_response();
    }
    // Track validation MUST run before the merge block below: that block
    // takes the pending config to merge into, and a 400 taken after the
    // take would silently discard whatever the user had already staged.
    if let Some(t) = &up.track {
        // An undriveable circuit is refused here, where the dashboard can
        // show the editor's error — never at the reset.
        if let Err(e) = crate::track::validate_track_spec(t) {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": format!("track: {e}") })),
            )
                .into_response();
        }
    }
    // Merge into pending; applied at the next reset, never mid-race.
    let mut p = st.pending_config.lock().unwrap();
    let merged = p.take().unwrap_or_default();
    *p = Some(ConfigUpdate {
        seed: up.seed.or(merged.seed),
        laps: up.laps.or(merged.laps),
        noise: up.noise.or(merged.noise),
        mode: up.mode.or(merged.mode),
        weather: up.weather.or(merged.weather),
        water: up.water.or(merged.water),
        track_temp_c: up.track_temp_c.or(merged.track_temp_c),
        grip: up.grip.or(merged.grip),
        track: up.track.or(merged.track),
    });
    drop(p);
    let cur = st.current_config.lock().unwrap().clone();
    let pending = st.pending_config.lock().unwrap().clone();
    Json(serde_json::json!({ "ok": true, "current": cur, "pending": pending })).into_response()
}

async fn get_metrics(State(st): State<Arc<WebState>>) -> Json<Metrics> {
    let mut m = st.metrics.lock().unwrap().clone();
    m.uptime_ms = st.started.elapsed().as_millis() as u64;
    Json(m)
}

async fn ws_handler(State(st): State<Arc<WebState>>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| client_connection(st, socket))
}

/// Per-client connection: writer task + forwarder task + reader loop.
/// The outbound queue is bounded (8); a slow client's snapshots are dropped
/// and counted — never allowed to back-pressure the sim or other clients.
async fn client_connection(st: Arc<WebState>, socket: WebSocket) {
    {
        let mut m = st.metrics.lock().unwrap();
        m.ws_clients += 1;
    }
    let (mut sender, mut receiver) = socket.split();
    let (out_tx, mut out_rx) = mpsc::channel::<String>(8);

    // Welcome: current session + track + latest snapshot (reconnect catch-up).
    let snap = st.snap_rx.borrow().clone();
    let welcome = ToClient::Welcome {
        v: PROTOCOL_VERSION,
        sid: snap.sid.clone(),
        track: st.track.read().unwrap().clone(),
        snapshot: snap,
        server_now_ms: unix_ms(),
    };
    let _ = out_tx.try_send(serde_json::to_string(&welcome).unwrap_or_default());

    // Writer: sole owner of the socket send half; drains the bounded queue.
    let writer = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
        let _ = sender.close().await;
    });

    // Forwarder: snapshots (replaceable) + events (never silently replaced).
    let fwd_state = st.clone();
    let fwd_out = out_tx.clone();
    let forwarder = tokio::spawn(async move {
        let mut snap_rx = fwd_state.snap_rx.clone();
        let mut event_rx = fwd_state.event_tx.subscribe();
        loop {
            tokio::select! {
                changed = snap_rx.changed() => {
                    if changed.is_err() { break; }
                    let snap = snap_rx.borrow().clone();
                    let payload = serde_json::to_string(&ToClient::Snapshot(snap)).unwrap_or_default();
                    let mut m = fwd_state.metrics.lock().unwrap();
                    match fwd_out.try_send(payload) {
                        Ok(()) => m.ws_snapshots_sent += 1,
                        Err(_) => m.ws_snapshots_dropped += 1,
                    }
                }
                ev = event_rx.recv() => {
                    match ev {
                        Ok(ev) => {
                            let payload = serde_json::to_string(&ToClient::Event(ev)).unwrap_or_default();
                            let mut m = fwd_state.metrics.lock().unwrap();
                            if fwd_out.try_send(payload).is_ok() {
                                m.events_published += 1;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            fwd_state.metrics.lock().unwrap().events_lagged += n;
                            let notice = format!("{{\"type\":\"event_gap\",\"missed\":{n}}}");
                            let _ = fwd_out.try_send(notice);
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
        }
    });

    // Reader: parse commands, validate session (stale rejection), enqueue.
    let current_sid = || st.snap_rx.borrow().sid.clone();
    while let Some(Ok(msg)) = receiver.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };
        let received_ms = unix_ms();
        let reply: ToClient = match serde_json::from_str::<InMsg>(&text) {
            Ok(m) if m.v != PROTOCOL_VERSION => ToClient::Ack {
                v: PROTOCOL_VERSION,
                sid: m.sid,
                nonce: m.nonce,
                received_ms,
                status: "rejected_bad_version",
                detail: format!("server speaks v{PROTOCOL_VERSION}"),
            },
            Ok(m) if m.sid != current_sid() => ToClient::Ack {
                v: PROTOCOL_VERSION,
                sid: m.sid,
                nonce: m.nonce,
                received_ms,
                status: "rejected_stale_session",
                detail: "session id no longer current (race was reset)".into(),
            },
            Ok(m) if let Err(why) = validate_command(&m.cmd) => {
                st.metrics.lock().unwrap().commands_rejected += 1;
                ToClient::Ack {
                    v: PROTOCOL_VERSION,
                    sid: m.sid,
                    nonce: m.nonce,
                    received_ms,
                    status: why.status(),
                    detail: why.detail().to_string(),
                }
            }
            Ok(m) => {
                st.metrics.lock().unwrap().commands_received += 1;
                let _ = st.cmd_tx.send(SimCommand { cmd: m.cmd.clone(), nonce: m.nonce, queued_ms: received_ms });
                ToClient::Ack {
                    v: PROTOCOL_VERSION,
                    sid: m.sid,
                    nonce: m.nonce,
                    received_ms,
                    status: "queued",
                    // A slider is queued for a named tick, so say which: the
                    // panel shows the effective time before the physics moves.
                    detail: match weather::parse_slider_cmd(&m.cmd) {
                        Ok(_) => format!("'{}' queued to take effect at tick +{}", m.cmd, SLIDER_LEAD_TICKS),
                        Err(_) => format!("'{}' queued for next tick boundary", m.cmd),
                    },
                }
            }
            Err(e) => ToClient::Ack {
                v: PROTOCOL_VERSION,
                sid: String::new(),
                nonce: 0,
                received_ms,
                status: "rejected_parse",
                detail: format!("unparseable message: {e}"),
            },
        };
        let _ = out_tx.try_send(serde_json::to_string(&reply).unwrap_or_default());
    }

    // Connection gone: stop forwarding, drop our sender so the writer drains
    // what is left and closes the socket, then count the exit.
    forwarder.abort();
    drop(out_tx);
    let _ = writer.await;
    st.metrics.lock().unwrap().ws_clients -= 1;
}

// ---------------------------------------------------------------------------
// Server bootstrap
// ---------------------------------------------------------------------------

/// Options for the serving entry point (Phase 6): structured logging and a
/// graceful, bounded lifetime. Plain `serve` without options behaves exactly
/// like the Phase 2–5 dashboard.
#[derive(Clone, Debug, Default)]
pub struct ServeOpts {
    /// Emit structured lifecycle logs (NDJSON) instead of human lines only.
    pub structured: bool,
    /// Stop the server gracefully after this long (used by the dev
    /// launcher's child and by deterministic smoke runs).
    pub duration: Option<std::time::Duration>,
    /// Run id shared with the launcher, so one `apexsim dev` run's lines all
    /// carry the same `run_id`.
    pub run_id: Option<String>,
    pub log_level: Option<Level>,
    pub log_file: Option<std::path::PathBuf>,
}

/// Poll `GET /api/health` until the service answers `ok: true` — the plan's
/// "readiness checks instead of fixed startup sleeps".
pub async fn wait_until_ready(
    addr: std::net::SocketAddr,
    timeout: std::time::Duration,
) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        if health_ok(addr).await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn health_ok(addr: std::net::SocketAddr) -> bool {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    tokio::time::timeout(Duration::from_millis(300), async move {
        let mut s = tokio::net::TcpStream::connect(addr).await.ok()?;
        let req = format!("GET /api/health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
        s.write_all(req.as_bytes()).await.ok()?;
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).await.ok()?;
        let text = String::from_utf8_lossy(&buf);
        (text.contains("\"ok\":true")).then_some(())
    })
    .await
    .ok()
    .is_some()
}

/// Spawns the sim thread (std) and the axum server (tokio task) on the
/// current runtime. Returns the bound address and the server task handle.
pub fn spawn_server(
    config: SimConfig,
    port: u16,
    autostart: bool,
) -> std::io::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    spawn_server_inner(config, port, autostart, None)
}

/// As [`spawn_server`], but also hands back the shared tournament progress so
/// a batch running alongside the dashboard can publish into it. Phase 10's
/// `engine+dashboard` profile: the server is the window onto the batch, and
/// this is how the batch reaches the window.
pub fn spawn_server_with_batch(
    config: SimConfig,
    port: u16,
    autostart: bool,
) -> std::io::Result<(
    std::net::SocketAddr,
    tokio::task::JoinHandle<()>,
    Arc<BatchProgress>,
)> {
    spawn_server_full(config, port, autostart, None, Some(Arc::new(BatchProgress::new())))
}

fn spawn_server_inner(
    config: SimConfig,
    port: u16,
    autostart: bool,
    logger: Option<Arc<Logger>>,
) -> std::io::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    let (addr, handle, _) = spawn_server_full(config, port, autostart, logger, None)?;
    Ok((addr, handle))
}

fn spawn_server_full(
    mut config: SimConfig,
    port: u16,
    autostart: bool,
    logger: Option<Arc<Logger>>,
    batch: Option<Arc<BatchProgress>>,
) -> std::io::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>, Arc<BatchProgress>)> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    let addr = listener.local_addr()?;
    listener.set_nonblocking(true)?;
    let listener = tokio::net::TcpListener::from_std(listener)?;

    // The dashboard is the race director's console, so the sliders are live in
    // every served session: at their neutral values the physics is exactly the
    // Phase 7 engine, and the first command that moves one is queued for a
    // known tick like any other override.
    if config.weather.is_none() {
        config.weather = Some(WeatherConfig::director());
    }

    let (snap_tx, snap_rx) = watch::channel(boot_snapshot());
    let (event_tx, _) = broadcast::channel::<EventMsg>(256);
    let (cmd_tx, cmd_rx) = std_mpsc::channel::<SimCommand>();
    let pending = Arc::new(Mutex::new(None));
    let current = Arc::new(Mutex::new(public_config(&config)));
    let metrics = Arc::new(Mutex::new(Metrics::default()));

    let track_spec = TrackSpec {
        version: crate::track::TRACK_VERSION,
        straight_len: config.track_straight,
        radius: config.track_radius,
        half_width: config.track_half_width,
        custom: config.track_custom.clone(),
    };
    // The circuit the dashboard draws must be the one the race drives;
    // --track was validated by the CLI parser, so expect is honest here.
    // The slot is shared: the race loop swaps it when a reset applies a
    // track change from the editor, so /api/track and the WS welcome always
    // describe what the current session drives.
    let track_slot = Arc::new(std::sync::RwLock::new(state_track_def(&track_spec)));

    let sim_cfg = config.clone();
    let sim_pending = pending.clone();
    let sim_current = current.clone();
    let sim_metrics = metrics.clone();
    let sim_events = event_tx.clone();
    let sim_logger = logger.clone();
    let sim_track_slot = track_slot.clone();
    let thread_name = if config.cars > 1 { "race" } else { "sim" };
    let is_race = config.cars > 1;
    std::thread::Builder::new()
        .name(thread_name.into())
        .spawn(move || {
            if is_race {
                race_server_loop(
                    sim_cfg,
                    autostart,
                    snap_tx,
                    sim_events,
                    cmd_rx,
                    sim_pending,
                    sim_current,
                    sim_metrics,
                    sim_logger,
                    sim_track_slot,
                )
            } else {
                sim_server_loop(sim_cfg, autostart, snap_tx, sim_events, cmd_rx, sim_pending, sim_current, sim_metrics, sim_logger)
            }
        })?;

    let state = Arc::new(WebState {
        snap_rx,
        event_tx,
        cmd_tx,
        pending_config: pending,
        current_config: current,
        metrics,
        started: Instant::now(),
        track: track_slot,
        batch: batch.unwrap_or_else(|| Arc::new(BatchProgress::new())),
    });
    let progress = state.batch.clone();

    let handle = tokio::spawn(async move {
        let app = build_router(state);
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("server error: {e}");
        }
    });
    Ok((addr, handle, progress))
}

/// Blocking entry point for the CLI (`apexsim serve`). Graceful by Phase 6
/// rules: readiness is observable (`wait_until_ready` / `/api/health`),
/// Ctrl-C or `opts.duration` ends the run with an explicit shutdown log and
/// a success status, and bind failures surface as errors.
pub async fn serve(
    config: SimConfig,
    port: u16,
    autostart: bool,
    opts: ServeOpts,
) -> std::io::Result<()> {
    let level = opts.log_level.unwrap_or(Level::Info);
    let run_id = opts.run_id.unwrap_or_else(|| {
        format!("r{:x}-{}", unix_ms(), std::process::id())
    });
    let logger = if opts.structured {
        Some(Arc::new(
            Logger::new(run_id.clone(), level, opts.log_file.as_deref())
                .map_err(|e| std::io::Error::other(format!("open log file: {e}")))?,
        ))
    } else {
        None
    };

    let (addr, handle) = match spawn_server_inner(config, port, autostart, logger.clone()) {
        Ok(v) => v,
        Err(e) => {
            let hint = if e.kind() == std::io::ErrorKind::AddrInUse {
                format!("port {port} is already in use — is another apexsim server running?")
            } else {
                String::new()
            };
            let msg = format!("bind 127.0.0.1:{port}: {e}. {hint}");
            if let Some(l) = &logger {
                l.log(Level::Error, None, None, &msg);
            }
            eprintln!("error: {msg}");
            return Err(e);
        }
    };
    let ready_msg = format!("apexsim dashboard: http://{addr}/");
    if let Some(l) = &logger {
        l.log(Level::Info, None, None, &format!("server ready at http://{addr}/"));
    } else {
        println!("{ready_msg}  (Ctrl-C to stop)");
    }

    // Graceful end: server task end, Ctrl-C, or an optional duration cap.
    let duration = opts.duration;
    let reason = tokio::select! {
        _ = handle => "server task ended",
        _ = tokio::signal::ctrl_c() => "ctrl-c",
        _ = async {
            match duration {
                Some(d) => tokio::time::sleep(d).await,
                None => std::future::pending::<()>().await,
            }
        } => "duration elapsed",
    };
    if let Some(l) = &logger {
        l.log(Level::Info, None, None, &format!("shutdown ({reason})"));
    } else {
        println!("shutdown ({reason}) — run {run_id}");
    }
    // The sim thread is an intentional daemon inside this process; ending
    // the run ends it. Exit explicitly so the launcher sees success.
    std::process::exit(0);
}

static INDEX_HTML: &str = include_str!("../static/index.html");
