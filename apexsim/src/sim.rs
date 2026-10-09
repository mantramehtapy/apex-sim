//! Authoritative simulation: one in-memory race state driven on a dedicated
//! thread by the runner, fixed 4 ms ticks, seeded reset, pause, per-tick
//! cost measurement, and input recording for replay.
//!
//! Real-time pacing uses absolute deadlines (never a bare sleep-per-tick,
//! never a dt change): every tick measures its own compute cost, and lag is
//! reported when a tick starts after its deadline.

use crate::driver::{Driver, DriverInput, DriverParams};
use crate::physics::{self, CarParams, CarState, Controls};
use crate::replay::RecordedInput;
use crate::stats::{summarize_us, Summary};
use crate::track::Track;
use std::thread;
use std::time::{Duration, Instant};

pub const DT: f64 = 0.004; // 4 ms fixed timestep (provisional 250 Hz)
pub const DRIVER_EVERY: u32 = 5; // local driving at 50 Hz

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub seed: u64,
    pub laps: u32,
    /// Phase 4: field size for race mode. 1 = single-car mode (Phases 1-3).
    pub cars: u32,
    /// Phase 5: session mode for the race field ("race"|"qualifying"|"practice").
    pub mode: String,
    /// Which field the race grid represents. The single-car `Sim` never
    /// reads it; the race loop does. Library default is classic so every
    /// existing session is untouched — user-facing commands default their
    /// new sessions to the 2026 teams.
    pub field: crate::race::FieldMode,
    /// The realism pack for the race loop (`None` = off). Same story as
    /// `field`: the single-car sim ignores it.
    pub realism: Option<crate::race::RealismConfig>,
    /// Weekend mode: the served session opens with QUALIFYING — each car
    /// gets a solo timed window, the grid forms from the order, then the
    /// race runs. Orchestration lives in the race loop; the single-car sim
    /// ignores it.
    pub weekend_quali: bool,
    /// The per-car qualifying window in sim seconds (weekend mode).
    pub quali_window_s: f64,
    pub realtime: bool,
    pub car: CarParams,
    pub driver: DriverParams,
    pub track_straight: f64,
    pub track_radius: f64,
    pub track_half_width: f64,
    /// A user-defined circuit (preset or track file). `None` = the built-in
    /// stadium described by the three numbers above.
    pub track_custom: Option<crate::track::CustomTrack>,
    /// Phase 8: the race-director panel's session state. `None` for a plain
    /// session — no sliders, and the Phase 7 engine bit for bit. `Some` keeps
    /// the sliders live even while every value sits at neutral.
    pub weather: Option<crate::weather::WeatherConfig>,
    /// Phase 9: where to publish 2025-layout UDP telemetry. `None` opens no
    /// socket at all. Note this is a *server* session config: the single-car
    /// `Sim` does not observe it.
    pub telemetry: Option<crate::telemetry::TelemetryConfig>,
    /// Phase 9: the sound cue vocabulary. `None` extracts nothing.
    pub sound: Option<crate::sound::SoundConfig>,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            seed: 42,
            laps: 5,
            cars: 1,
            mode: "race".into(),
            field: crate::race::FieldMode::Classic,
            realism: None,
            weekend_quali: false,
            quali_window_s: 240.0,
            realtime: false,
            car: CarParams::default(),
            driver: DriverParams::default(),
            track_straight: 300.0,
            track_radius: 80.0,
            track_half_width: 7.0,
            track_custom: None,
            weather: None,
            telemetry: None,
            sound: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Ready,
    Running,
    Paused,
    Finished,
}

/// The single authoritative race state. Everything the outside world sees is
/// a copy of this; nothing outside the sim thread mutates it.
#[derive(Clone, Debug, Default)]
pub struct RaceState {
    pub tick: u64,
    pub sim_time: f64,
    pub car: CarState,
    pub phase: Phase,
    pub laps_done: u32,
    pub lap_times: Vec<f64>,
    pub lap_start_time: f64,
    pub s_progress: f64,
    pub s_prev: f64,
    pub offtrack_ticks: u64,
}

pub struct Sim {
    pub config: SimConfig,
    pub state: RaceState,
    track: Track,
    driver: Driver,
    controls: Controls,
    state_hash: u64,
    tick_costs_us: Vec<f64>,
    max_lag_us: f64,
    deadline_misses: u64,
    recorded: Vec<RecordedInput>,
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv_mix(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

impl Sim {
    pub fn new(config: SimConfig) -> Self {
        let track = Track::new(
            config.track_straight,
            config.track_radius,
            config.track_half_width,
        );
        let mut sim = Sim {
            driver: Driver::new(config.driver, config.seed),
            track,
            config,
            state: RaceState::default(),
            controls: Controls::default(),
            state_hash: FNV_OFFSET,
            tick_costs_us: Vec::new(),
            max_lag_us: 0.0,
            deadline_misses: 0,
            recorded: Vec::new(),
        };
        sim.reset(sim.config.seed);
        sim
    }

    /// Seeded reset: same seed always restores the identical initial state.
    pub fn reset(&mut self, seed: u64) {
        self.config.seed = seed;
        self.driver = Driver::new(self.config.driver, seed);
        self.state = RaceState {
            tick: 0,
            sim_time: 0.0,
            car: CarState::default(), // on the start line, at rest
            phase: Phase::Ready,
            laps_done: 0,
            lap_times: Vec::new(),
            lap_start_time: 0.0,
            s_progress: 0.0,
            s_prev: 0.0,
            offtrack_ticks: 0,
        };
        self.controls = Controls::default();
        self.state_hash = FNV_OFFSET;
        self.tick_costs_us.clear();
        self.max_lag_us = 0.0;
        self.deadline_misses = 0;
        self.recorded.clear();
    }

    pub fn track(&self) -> &Track {
        &self.track
    }

    pub fn pause(&mut self) {
        if self.state.phase == Phase::Running {
            self.state.phase = Phase::Paused;
        }
    }

    pub fn resume(&mut self) {
        if self.state.phase == Phase::Paused {
            self.state.phase = Phase::Running;
        }
    }

    /// Run-time bookkeeping executed at the driver rate (50 Hz): project the
    /// car onto the track, count laps and off-track ticks. Lap timing
    /// granularity is therefore 20 ms in Phase 1 (refined in Phase 5).
    fn driver_rate_bookkeeping(&mut self) {
        let (s, lateral) = self.track.project(self.state.car.pos_x, self.state.car.pos_y);
        let len = self.track.length();
        if self.track.is_off_track(lateral) {
            self.state.offtrack_ticks += DRIVER_EVERY as u64;
        }
        // Forward crossing of the start/finish line.
        if self.state.s_prev > 0.8 * len && s < 0.2 * len {
            self.state.laps_done += 1;
            let lap_time = self.state.sim_time - self.state.lap_start_time;
            self.state.lap_times.push(lap_time);
            self.state.lap_start_time = self.state.sim_time;
            if self.state.laps_done >= self.config.laps {
                self.state.phase = Phase::Finished;
            }
        }
        self.state.s_prev = s;
        self.state.s_progress = s;
    }

    /// One physics tick with externally supplied controls (replay path).
    pub fn tick_with_controls(&mut self, ctrl: Controls) {
        if self.state.phase != Phase::Running {
            return;
        }
        if self.state.tick.is_multiple_of(DRIVER_EVERY as u64) {
            self.driver_rate_bookkeeping();
        }
        self.apply_tick(ctrl);
    }

    /// One physics tick with the local line-following driver (live path).
    /// Records the controls used at every 50 Hz boundary.
    pub fn tick_with_driver(&mut self) {
        if self.state.phase != Phase::Running {
            return;
        }
        if self.state.tick.is_multiple_of(DRIVER_EVERY as u64) {
            self.driver_rate_bookkeeping();
            let Sim { driver, state, track, recorded, .. } = self;
            let ctrl = driver.drive(
                &state.car,
                track,
                DriverInput { s: state.s_progress, ..Default::default() },
            );
            recorded.push(RecordedInput {
                tick: state.tick,
                throttle: ctrl.throttle,
                brake: ctrl.brake,
                steer_cmd: ctrl.steer_cmd,
            });
            self.controls = ctrl;
        }
        self.apply_tick(self.controls);
    }

    fn apply_tick(&mut self, ctrl: Controls) {
        let t0 = Instant::now();
        let new_car = physics::step(&self.state.car, &ctrl, &self.config.car, DT);
        self.state.tick += 1;
        self.state.sim_time = self.state.tick as f64 * DT;
        self.state.car = new_car;

        // Trajectory hash: any divergence from an identical run shows up here.
        let c = &self.state.car;
        let mut h = fnv_mix(self.state_hash, &self.state.tick.to_le_bytes());
        for bits in [
            c.pos_x.to_bits(),
            c.pos_y.to_bits(),
            c.heading.to_bits(),
            c.vel_long.to_bits(),
            c.vel_lat.to_bits(),
            c.yaw_rate.to_bits(),
            c.steer.to_bits(),
        ] {
            h = fnv_mix(h, &bits.to_le_bytes());
        }
        self.state_hash = h;

        let cost_us = t0.elapsed().as_secs_f64() * 1e6;
        if cost_us > DT * 1e6 {
            self.deadline_misses += 1;
        }
        self.tick_costs_us.push(cost_us);
    }

    pub fn state_hash(&self) -> u64 {
        self.state_hash
    }

    pub fn take_recorded_inputs(&mut self) -> Vec<RecordedInput> {
        std::mem::take(&mut self.recorded)
    }

    pub fn outcome(&mut self, wall_time_s: f64) -> SimOutcome {
        SimOutcome {
            laps_done: self.state.laps_done,
            lap_times: self.state.lap_times.clone(),
            ticks: self.state.tick,
            sim_time: self.state.sim_time,
            state_hash: self.state_hash,
            tick_stats: summarize_us(std::mem::take(&mut self.tick_costs_us)),
            max_lag_us: self.max_lag_us,
            deadline_misses: self.deadline_misses,
            offtrack_ticks: self.state.offtrack_ticks,
            final_state: self.state.car,
            wall_time_s,
        }
    }

    fn note_lag_us(&mut self, lag_us: f64) {
        self.max_lag_us = self.max_lag_us.max(lag_us);
    }
}

#[derive(Clone, Debug)]
pub struct SimOutcome {
    pub laps_done: u32,
    pub lap_times: Vec<f64>,
    pub ticks: u64,
    pub sim_time: f64,
    pub state_hash: u64,
    pub tick_stats: Summary,
    pub max_lag_us: f64,
    pub deadline_misses: u64,
    pub offtrack_ticks: u64,
    pub final_state: CarState,
    pub wall_time_s: f64,
}

/// Progress events sent from the simulation thread to the caller.
pub enum Event {
    Started { seed: u64, laps: u32, realtime: bool },
    Lap { n: u32, time_s: f64 },
    Progress { sim_time: f64, u: f64, s: f64 },
    Finished(Box<SimOutcome>),
}

/// Run a live session (local driver). Returns the outcome and the recorded
/// 50 Hz input log. This function is what a dedicated simulation thread calls.
pub fn run_session(
    config: SimConfig,
    tx: Option<&std::sync::mpsc::Sender<Event>>,
) -> (SimOutcome, Vec<RecordedInput>) {
    let realtime = config.realtime;
    let mut sim = Sim::new(config);
    sim.state.phase = Phase::Running;

    let _timer = if realtime { Some(TimerPeriod::acquire()) } else { None };
    if let Some(tx) = tx {
        let _ = tx.send(Event::Started { seed: sim.config.seed, laps: sim.config.laps, realtime });
    }

    let start = Instant::now();
    let mut next_progress = 0.0f64;
    let mut prev_laps = 0u32;
    while sim.state.phase == Phase::Running {
        let tick_idx = sim.state.tick;
        if realtime {
            let deadline = start + Duration::from_secs_f64(tick_idx as f64 * DT);
            let now = Instant::now();
            if now < deadline {
                wait_until(deadline);
            } else {
                sim.note_lag_us((now - deadline).as_secs_f64() * 1e6);
            }
        }
        sim.tick_with_driver();

        if let Some(tx) = tx {
            if sim.state.laps_done > prev_laps {
                prev_laps = sim.state.laps_done;
                let _ = tx.send(Event::Lap {
                    n: sim.state.laps_done,
                    time_s: *sim.state.lap_times.last().unwrap_or(&0.0),
                });
            }
            if sim.state.sim_time >= next_progress {
                next_progress = sim.state.sim_time + 2.0;
                let _ = tx.send(Event::Progress {
                    sim_time: sim.state.sim_time,
                    u: sim.state.car.vel_long,
                    s: sim.state.s_progress,
                });
            }
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let inputs = sim.take_recorded_inputs();
    let outcome = sim.outcome(wall);
    if let Some(tx) = tx {
        let _ = tx.send(Event::Finished(Box::new(outcome.clone())));
    }
    (outcome, inputs)
}

/// Re-run a session from recorded inputs; returns the recomputed outcome so
/// the caller can compare trajectory hashes.
pub fn replay_session(config: SimConfig, inputs: &[RecordedInput]) -> SimOutcome {
    let mut sim = Sim::new(config);
    sim.state.phase = Phase::Running;
    let start = Instant::now();
    let mut idx = 0usize;
    let mut held = Controls::default();
    while sim.state.phase == Phase::Running && idx < inputs.len() {
        let tick = sim.state.tick;
        if tick.is_multiple_of(DRIVER_EVERY as u64) {
            let rec = &inputs[idx];
            assert_eq!(rec.tick, tick, "input log misaligned at tick {tick}");
            held = Controls {
                throttle: rec.throttle,
                brake: rec.brake,
                steer_cmd: rec.steer_cmd,
            };
            idx += 1;
        }
        sim.tick_with_controls(held);
    }
    let wall = start.elapsed().as_secs_f64();
    sim.outcome(wall)
}

// ---- Real-time pacing helpers ----

/// Sleep until just before the deadline, then spin/yield the remainder.
/// Windows' default timer granularity (~15 ms) makes a bare sleep unusable
/// at 4 ms ticks, so the last slice is spun out on the clock.
pub fn wait_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let remaining = deadline - now;
        if remaining > Duration::from_millis(2) {
            thread::sleep(remaining - Duration::from_millis(2));
        } else if remaining > Duration::from_micros(150) {
            thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
}

#[cfg(windows)]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(ms: u32) -> u32;
    fn timeEndPeriod(ms: u32) -> u32;
}

/// Requests 1 ms timer resolution while alive (Windows only, best effort).
pub struct TimerPeriod {
    #[cfg(windows)]
    acquired: bool,
}

impl TimerPeriod {
    pub fn acquire() -> TimerPeriod {
        #[cfg(windows)]
        {
            // SAFETY: plain Win32 timer-resolution request; no state involved.
            let ok = unsafe { timeBeginPeriod(1) } == 0;
            TimerPeriod { acquired: ok }
        }
        #[cfg(not(windows))]
        TimerPeriod {}
    }
}

impl Drop for TimerPeriod {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            if self.acquired {
                // SAFETY: matches the timeBeginPeriod(1) above.
                unsafe { timeEndPeriod(1) };
            }
        }
    }
}
