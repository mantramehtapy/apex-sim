//! Phase 4: traffic, collisions, and grid scaling.
//!
//! One authoritative physics owner (this module's loop, on the sim thread)
//! steps every car sequentially at the fixed 4 ms tick — the plan
//! explicitly defers parallel physics until logging/network are separated
//! and measured. Cars get: nearby-car awareness with follow/brake caps,
//! lateral overtake/defend offsets, an impulse contact model with
//! retirement above a closing-speed threshold, lap wrap / gaps / position /
//! retirement / classification, and deterministic per-car seeds plus a
//! fixed per-index pace spread so the field strings out realistically.
//!
//! Grid size is a configuration choice, not an assumption: the default is
//! 20 (2024 season), and the plan's rule "do not mix 20-car and 22-car
//! assumptions" is honoured by making it explicit everywhere.
//!
//! Phase 5 turns laps into a complete, classifiable race weekend:
//! - Session modes: Race (standing start, finish-order classification),
//!   Qualifying/Practice (rolling start, best-lap classification).
//! - Start sequence: a formation phase with the five-light start and a
//!   seeded hold, then green.
//! - Tires: Soft/Medium/Hard compounds with distance-based wear; grip loss
//!   feeds both the physics friction coefficient and the driver's speed
//!   envelope, so degradation costs measurable lap time.
//! - Pits: a pit lane parallel to the bottom straight (brake zone, entry,
//!   speed limit, box service, exit), deterministic strategies, stops
//!   logged per car. Pit cars are on the same s-coordinate at a fixed
//!   lateral offset — outside every awareness window and the contact
//!   circle — so lane traffic and track traffic stay separate roads.
//!
//! Flags, safety-car procedures and penalties are later substeps per the
//! revised plan; rules values live in configuration, not code.

use crate::driver::{Driver, DriverInput, DriverParams};
use crate::physics::{self, CarParams, CarState, Controls};
use crate::sound::{self, SoundCue, SoundConfig, SoundRuntime, SoundSummary};
use crate::stats::{summarize_us, Summary};
use crate::steward;
use crate::telemetry::{self, CarSample, TelemetryConfig, TelemetryLink};
use crate::strategy;
use crate::teams;
use crate::track::{CustomTrack, PitFractions, Track, TrackSpec};
use serde::{Deserialize, Serialize};
use crate::weather::{self, AppliedOverride, Override, WeatherConfig, WeatherRuntime};
use std::time::Instant;

pub const DT: f64 = crate::sim::DT; // 4 ms, engine rate retained per Phase 3
pub const DRIVER_EVERY: u32 = crate::sim::DRIVER_EVERY;

/// The intended-season grid size (2024: 10 teams × 2). Never mixed with 22.
pub const DEFAULT_GRID: u32 = 20;

// ---------------------------------------------------------------------------
// Session / tire / pit configuration (Phase 5)
// ---------------------------------------------------------------------------

/// Session modes. Race is the full standing-start race; qualifying and
/// practice are rolling-start timed sessions classified by best lap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SessionMode {
    #[default]
    Race,
    Qualifying,
    Practice,
}

impl SessionMode {
    pub fn label(self) -> &'static str {
        match self {
            SessionMode::Race => "race",
            SessionMode::Qualifying => "qualifying",
            SessionMode::Practice => "practice",
        }
    }
    pub fn parse(s: &str) -> Option<SessionMode> {
        match s {
            "race" => Some(SessionMode::Race),
            "qualifying" | "quali" => Some(SessionMode::Qualifying),
            "practice" | "practise" => Some(SessionMode::Practice),
            _ => None,
        }
    }
}

/// Tire compound: grip multiplier and distance life. Season-specific
/// values deliberately live here (configuration), not in the physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TireCompound {
    #[default]
    Soft,
    Medium,
    Hard,
}

impl TireCompound {
    pub fn grip(self) -> f64 {
        match self {
            TireCompound::Soft => 1.03,
            TireCompound::Medium => 1.0,
            TireCompound::Hard => 0.975,
        }
    }
    /// Race-distance life in laps at this circuit's length.
    pub fn life_laps(self) -> f64 {
        match self {
            TireCompound::Soft => 8.0,
            TireCompound::Medium => 14.0,
            TireCompound::Hard => 22.0,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            TireCompound::Soft => "S",
            TireCompound::Medium => "M",
            TireCompound::Hard => "H",
        }
    }
    pub fn parse(s: &str) -> Option<TireCompound> {
        // Case-insensitive: the advisor lower-cases model output, and a CLI
        // user typing `--compound s` should get the same answer.
        match s.trim().to_ascii_lowercase().as_str() {
            "soft" | "s" => Some(TireCompound::Soft),
            "medium" | "m" => Some(TireCompound::Medium),
            "hard" | "h" => Some(TireCompound::Hard),
            _ => None,
        }
    }
}

/// Fraction of grip lost to tire wear at 100% wear (linear to the mark,
/// floored there — degraded, not punctured).
const WEAR_GRIP_LOSS: f64 = 0.06;
/// Realism pack: grip bled at FULL contact damage, and at a fully
/// flat-spotted tyre. Gentler than a puncture: a dented car is a slow
/// car, not a written-off one.
const DAMAGE_GRIP_LOSS: f64 = 0.10;
/// Race control: SC pace for the whole field, the minimum time the car stays
/// out, the cooldown between deployments, and the local-yellow pace.
const SC_SPEED: f64 = 22.0;
const SC_MIN_TICKS: u64 = 15_000; // 60 s: at least ~1.5 laps at SC pace
const SC_COOLDOWN_TICKS: u64 = 7_500; // 30 s between deployments
const YELLOW_TICKS: u64 = 11_250; // 45 s
const YELLOW_SPEED: f64 = 25.0;
const YELLOW_ZONE_M: f64 = 30.0; // the sector extends this far either side
const BLUE_WINDOW_M: f64 = 80.0; // leader this close behind = blue flag
const FLATSPOT_GRIP_LOSS: f64 = 0.08;
/// Realism pack W3: race-start fuel load (kg), burn rate (kg per metre),
/// and the grip bled by a fully COLD tyre (tyres warm to full grip over
/// roughly two laps of driving).
const FUEL_CAPACITY_KG: f64 = 60.0;
const FUEL_BURN_PER_M: f64 = 0.0030;
const COLD_TYRE_GRIP_LOSS: f64 = 0.08;
/// W3 refuelling: seconds of stop time per kg of fuel added (60 kg = +4.8 s
/// on top of the base service time — the short-fill-vs-long-fill trade-off).
const REFUEL_SEC_PER_KG: f64 = 0.08;
/// W3 tyre warm-up: rubber reaches full operating temperature after roughly
/// this much driving time (braking and sliding add extra, see realism_step).
const TYRE_WARM_RATE_PER_S: f64 = 1.0 / 120.0;
/// Braking hard warms the tyres meaningfully faster than cruising.
const TYRE_WARM_BRAKE_BONUS: f64 = 2.5;

/// Deterministic pit strategy (the plan starts with deterministic
/// strategies measured against replay scenarios).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PitStrategy {
    /// No stops.
    #[default]
    None,
    /// One stop at the race midpoint onto the Medium compound, with the
    /// field split into waves (car i stops at base + i % waves) so the
    /// single box queue stays short.
    Auto,
    /// One stop for every car at a fixed lap, onto a fixed compound.
    Fixed { lap: u32, switch_to: TireCompound },
}

/// Pit lane geometry and rules. All on the bottom straight (s 0..300):
/// the brake zone leads into the entry, the box sits mid-lane, the exit
/// releases back onto the racing surface before the first corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitConfig {
    pub enabled: bool,
    pub entry_s: f64,
    pub box_s: f64,
    pub exit_s: f64,
    /// Lateral offset of the pit lane centreline (m, + = left). Negative
    /// puts it outside the bottom straight, clear of the 7 m track edge
    /// and of every overtake target (±4 m) — the two roads never touch.
    pub offset_m: f64,
    /// Pit speed limit, m/s (24 ≈ 86 km/h).
    pub speed_limit: f64,
    /// Stationary service time at the box (tire change), s.
    pub service_time_s: f64,
    /// Braking room before the entry in which armed cars are held to the
    /// pit limit (they must arrive at the entry already slow).
    pub brake_zone_m: f64,
}

impl Default for PitConfig {
    fn default() -> Self {
        PitConfig {
            enabled: true,
            entry_s: 150.0,
            box_s: 210.0,
            exit_s: 290.0,
            offset_m: -7.5,
            speed_limit: 24.0,
            service_time_s: 2.4,
            brake_zone_m: 120.0,
        }
    }
}

impl PitConfig {
    /// Geometric sanity; a malformed config disables the lane rather than
    /// letting cars service inside a corner.
    pub fn usable(&self, track_len: f64) -> bool {
        self.enabled
            && self.entry_s > self.brake_zone_m
            && self.entry_s < self.box_s
            && self.box_s < self.exit_s
            && self.exit_s < track_len
            && self.speed_limit > 1.0
            && self.service_time_s >= 0.0
    }
}

/// Per-car pit phase. `Servicing` is the stationary box stop; the countdown
/// runs in the physics loop so both execution paths share it bit-exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PitPhase {
    OnTrack,
    InLane,
    Servicing { remaining_s: f64 },
}

impl PitPhase {
    pub fn in_pit(self) -> bool {
        !matches!(self, PitPhase::OnTrack)
    }
}

/// One completed pit stop, for the results sheet.
#[derive(Clone, Copy, Debug)]
pub struct PitStopRecord {
    /// Compound fitted during this stop (Phase 7: makes the stint history
    /// exact instead of reconstructed from the strategy).
    pub fitted: TireCompound,
    pub car: u32,
    pub lap: u32,
    pub entered_t: f64,
    pub exited_t: f64,
    pub held_s: f64,
}

/// Race-control events: deterministic, appended identically by the live
/// and replay paths, forwarded to clients by the server.
#[derive(Clone, Copy, Debug)]
pub enum RcEvent {
    Lights(u8),
    Green,
    PitEntered { car: u32, lap: u32 },
    PitServiceDone { car: u32, held_s: f64 },
    PitExited { car: u32, lane_s: f64 },
    FastestLap { car: u32, time_s: f64, lap: u32 },
    /// Realism pack: a car is out for a non-contact reason — spun, beached,
    /// or a mechanical failure. `cause`/`detail` are static labels.
    CarOut { car: u32, cause: &'static str, detail: &'static str },
    /// Realism pack: race control deployed ("deployed") or cleared ("clear")
    /// the safety car.
    SafetyCar { phase: &'static str },
    /// Phase 8: a race-director slider command took effect. `tick` is the
    /// simulation tick it landed on — the plan requires every override to be
    /// logged with its effective tick, because the command was queued rather
    /// than applied on arrival.
    WeatherApplied { tick: u64, slider: &'static str, value: f64 },
}

#[derive(Clone, Debug)]
pub struct RaceConfig {
    pub cars: u32,
    pub laps: u32,
    pub seed: u64,
    /// metres between grid rows (two-column staggered grid).
    pub grid_row_gap_m: f64,
    /// closing speed (m/s) above which a contact retires the chasing car.
    pub contact_retire_cs: f64,
    /// per-car pace spread: car i scales its speed limits by
    /// 1 - pace_spread * i (deterministic, seed-independent).
    pub pace_spread: f64,
    /// driver-input noise amplitude (per-car seeded; makes the seed matter).
    pub noise: f64,
    /// Phase 5: session mode (race / qualifying / practice).
    pub mode: SessionMode,
    /// Starting tire compound for the whole field (strategy may change it).
    pub start_compound: TireCompound,
    /// Deterministic pit strategy.
    pub strategy: PitStrategy,
    /// Pit lane geometry and rules.
    pub pit: PitConfig,
    /// Standing-start formation with the light sequence (race mode only).
    pub formation: bool,
    /// Legacy v2 log replay: reproduce the Phase 4 engine exactly — no
    /// formation, no stops, no tire scaling or wear, and the Phase 4
    /// trajectory-hash mix (which carried no tire/pit bytes). Set by the
    /// log reader when the header has no session line; never written to
    /// v3 logs.
    pub legacy_v2: bool,
    /// Phase 7: optional LLM strategy advisor for car 0. None = the
    /// deterministic Phase 5 race, bit-for-bit. Some(_) = car 0 also takes
    /// typed advice from a local model — the run is then NOT reproducible
    /// (model latency feeds back into decisions), which is why replay
    /// logging refuses AI runs and the hash stays untouched by advice.
    pub ai: Option<strategy::AdvisorConfig>,
    /// Phase 8: weather, race-director sliders, and seeded driver errors.
    /// `None` = the Phase 7 engine bit-for-bit. `Some(_)` enables surface
    /// grip zones, queued slider overrides, and (once tested) driver errors.
    pub weather: Option<WeatherConfig>,
    /// Phase 9: where to publish 2025-layout UDP telemetry. `None` = nothing is
    /// sent, no socket is opened and no thread is spawned, which is how a
    /// Phase 8 race stays bit-for-bit unchanged. Enabling it changes what
    /// the *world* can observe and nothing the race can feel — the gate
    /// `a_disconnected_receiver_does_not_change_the_race` proves that.
    pub telemetry: Option<TelemetryConfig>,
    /// Phase 9: sound-cue extraction. `None` = no cue log at all. `Some`
    /// records every cue the race produces into a deterministic log; playing
    /// them is a separate, lossy step outside the simulation.
    pub sound: Option<SoundConfig>,
    /// Phase 10: which circuit, and which version of that circuit. Recorded
    /// in the run log (`G` line) so a tournament result replays on the same
    /// geometry rather than on whatever the default happens to be that day.
    pub track: TrackSpec,
    /// Phase 11: the in-simulation stewarding aid. `None` = no incident
    /// record at all, and the race is bit-for-bit the Phase 10 race. The aid
    /// is an observer: it cannot change the trajectory, which the gate
    /// `a_stewarded_race_drives_exactly_the_same_race` proves by comparing
    /// state hashes rather than asserting it.
    pub steward: Option<steward::StewardConfig>,
    /// Phase E weekend: which car occupies each grid slot. Empty = identity —
    /// slot `i` holds car `i`, which is every race ever run before qualifying
    /// existed. A full permutation of `0..cars` (pole first) positions the
    /// field by, e.g., a qualifying result. Machinery, driver form and noise
    /// seed follow the CAR: the fastest qualifier takes pole in their own car,
    /// they do not inherit slot 0's.
    pub grid_order: Vec<u32>,
    /// Phase E qualifying: this session fields exactly one car, and it is car
    /// `N`. Machinery, form and noise seed all resolve for N — a solo run must
    /// measure the same car the race will field, which a plain `cars = 1`
    /// session cannot express (that would always be car 0).
    pub solo_car: Option<u32>,
    /// The realism pack. `None` = the pre-pack engine bit-for-bit; `Some`
    /// enables spins, mechanicals, damage and lockups per the config. The
    /// inert variant normalises to `None` in `Race::new`, exactly like
    /// weather and stewarding.
    pub realism: Option<RealismConfig>,
    /// Which field the grid represents. `Classic` is the Phase 4 engine:
    /// physically identical cars separated only by the driver `pace_spread`.
    /// `Teams` gives each team its own engine, top speed, brakes and tyre
    /// grip from `teams.json`, plus a tiny seeded driver-form delta —
    /// so qualifying separates cars because the machinery differs, not only
    /// the driver envelope. The library default stays `Classic` so every
    /// existing test, replay and validation fixture is untouched; the CLI
    /// and server default new sessions to `Teams`.
    pub field: FieldMode,
}

/// Which performance model the field uses. See [`RaceConfig::field`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FieldMode {
    /// Identical cars, `pace_spread` driver envelope only (all pre-2026
    /// behaviour, and the only mode v1–v3 logs know about).
    #[default]
    Classic,
    /// fictional 2026-style team machinery from `teams.json` (see [`crate::teams`]).
    Teams,
}

impl FieldMode {
    pub fn label(self) -> &'static str {
        match self {
            FieldMode::Classic => "classic",
            FieldMode::Teams => "teams",
        }
    }

    pub fn parse(s: &str) -> Option<FieldMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "classic" | "identical" | "none" => Some(FieldMode::Classic),
            "teams" | "2026" => Some(FieldMode::Teams),
            _ => None,
        }
    }
}

/// The realism pack: what can put a car out, and how. `None` on
/// `RaceConfig::realism` is the pre-pack engine bit-for-bit — the same
/// inert-config contract as weather, sound and stewarding.
///
/// The knobs are deliberately few. The user asked for cars that can crash:
/// mistakes get teeth (a spin), machinery fails (a mechanical), touches cost
/// grip (damage) and locked brakes flat-spot tyres. Every rate is a plain
/// number a human can reason about per race, not a physics constant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealismConfig {
    /// A mistake that lands while the car is near its grip limit can break
    /// the rear loose; a spun or beached car is out.
    pub spin_crashes: bool,
    /// Mistakes per driver per lap. 0 = clean-handed field (the error path
    /// never opens).
    pub error_rate: f64,
    /// Expected engine/gearbox failures per car per race.
    pub mechanical_rate: f64,
    /// Low-speed touches accumulate damage that bleeds grip until a stop.
    pub contact_damage: bool,
    /// A mistake while braking hard flat-spots the tyre (same bleed family
    /// as damage; a stop resets it).
    pub lockups: bool,
    /// A car stopped on the track deploys the safety car: the field slows,
    /// overtaking is banned, and the race restarts green when clear.
    pub safety_car: bool,
    /// Retirements open a local yellow sector (slow, no passing) for a while.
    pub yellow_zones: bool,
    /// Cars about to be lapped concede: they lift and move off the line.
    pub blue_flags: bool,
    /// A rolling formation lap before the standing start (race mode).
    pub formation_lap: bool,
    /// DRS: free drag reduction on the circuit's longest straights when
    /// within a second of the car ahead.
    pub drs_zones: bool,
    /// Fuel load: mass affects acceleration, burn per metre, refuel in stops.
    pub fuel: bool,
    /// Tyre temperature: cold tyres until warmed by driving; a stop fits
    /// cold rubber again.
    pub tyre_temp: bool,
}

impl Default for RealismConfig {
    fn default() -> Self {
        RealismConfig {
            spin_crashes: true,
            error_rate: 0.06,
            mechanical_rate: 0.06,
            contact_damage: true,
            lockups: true,
            safety_car: true,
            yellow_zones: true,
            blue_flags: true,
            formation_lap: true,
            drs_zones: true,
            fuel: true,
            tyre_temp: true,
        }
    }
}

impl RealismConfig {
    pub fn is_inert(&self) -> bool {
        !self.spin_crashes
            && self.error_rate <= 0.0
            && self.mechanical_rate <= 0.0
            && !self.contact_damage
            && !self.lockups
            && !self.safety_car
            && !self.yellow_zones
            && !self.blue_flags
            && !self.formation_lap
            && !self.drs_zones
            && !self.fuel
            && !self.tyre_temp
    }
}

/// Why a car is out. Recorded on the classification row and in the outcome;
/// the contact cause predates the pack (it is the only pre-realism way out).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum DnfCause {
    Contact,
    Spun,
    Beached,
    Mechanical,
}

impl DnfCause {
    pub fn label(self) -> &'static str {
        match self {
            DnfCause::Contact => "contact",
            DnfCause::Spun => "spun",
            DnfCause::Beached => "beached",
            DnfCause::Mechanical => "mechanical",
        }
    }
}

/// The resolved per-car performance of one grid slot, as multipliers on
/// `CarParams::default()`. 1.0 everywhere = the classic identical car.
#[derive(Clone, Copy, Debug)]
pub struct CarPerf {
    /// Engine force and power multiplier.
    pub engine: f64,
    /// Top-speed multiplier (physics: drag divided by its cube).
    pub top_speed: f64,
    /// Brake force multiplier.
    pub brakes: f64,
    /// Tyre grip multiplier — applied to `params_for` and `grip_scale`
    /// together, or the driver aims at grip the tyres do not have.
    pub tyres: f64,
    /// Driver form: seeded ±0.3%, applied ONLY to the driver envelope
    /// (target speed and brake anticipation), never to the physics. Both
    /// cars of a team share the machinery; this is what separates
    /// teammates, and it must never show up in `params_of` — a car's brakes
    /// do not change because the other seat is occupied by a better driver.
    pub form: f64,
}

impl CarPerf {
    pub const IDENTICAL: CarPerf =
        CarPerf { engine: 1.0, top_speed: 1.0, brakes: 1.0, tyres: 1.0, form: 1.0 };

    /// The per-car perf of grid slot `i` under `mode`.
    ///
    /// Team machinery comes from the committed table (both cars of a team
    /// identical); the driver-form delta is seeded, so the same seed always
    /// fields the same grid. `pace_spread` is *replaced*, not stacked: it was
    /// the stand-in for car differences, and teams mode has the real thing.
    fn resolve(mode: FieldMode, seed: u64, i: u32) -> CarPerf {
        match mode {
            FieldMode::Classic => CarPerf::IDENTICAL,
            FieldMode::Teams => {
                let t = teams::for_car(i);
                CarPerf {
                    engine: t.engine,
                    top_speed: t.top_speed,
                    brakes: t.brakes,
                    tyres: t.tyres,
                    form: teams::driver_form(seed, i),
                }
            }
        }
    }
}

impl Default for RaceConfig {
    fn default() -> Self {
        RaceConfig {
            cars: DEFAULT_GRID,
            laps: 5,
            seed: 42,
            grid_row_gap_m: 9.0,
            contact_retire_cs: 20.0,
            pace_spread: 0.002,
            noise: 0.002,
            mode: SessionMode::Race,
            start_compound: TireCompound::Soft,
            strategy: PitStrategy::Auto,
            pit: PitConfig::default(),
            formation: true,
            legacy_v2: false,
            ai: None,
            weather: None,
            telemetry: None,
            sound: None,
            track: TrackSpec::default(),
            steward: None,
            field: FieldMode::Classic,
            grid_order: Vec::new(),
            solo_car: None,
            realism: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CarEntry {
    pub id: u32,
    pub car: CarState,
    pub s: f64,
    pub lateral: f64, // actual lateral offset (m, + = left of centreline)
    pub line_offset: f64, // current racing-line target offset
    pub laps_crossings: u32,
    pub lap_times: Vec<f64>,
    pub lap_start_t: f64,
    pub total_progress: f64,
    pub retired: bool,
    pub finish_time_s: Option<f64>,
    pub last_contact_tick: u64,
    /// Realism pack: accumulated contact damage (0 = clean). Bleeds tyre
    /// grip through `tire_condition`; a box stop resets it.
    pub damage: f64,
    /// Realism pack W3: fuel on board, kg. Mass rides params_for (slower
    /// acceleration when heavy), burns per metre, refuelled at a stop.
    pub fuel_mass: f64,
    /// Realism pack W3: tyre temperature, 0 = stone cold, 1 = fully warmed.
    /// Cold rubber bleeds grip through `tire_condition`; a stop fits COLD
    /// rubber (fresh tyres are fast once warmed, not instantly).
    pub tyre_temp: f64,
    /// Realism pack: flat-spotted tyre from a braking mistake (0 = round).
    /// Same bleed family as damage; a box stop resets it.
    pub flatspot: f64,
    /// Why the car is out, when the pack (or a contact) retired it. An
    /// annotation: set at the retirement site, reproduced there on replay,
    /// never hashed.
    pub dnf: Option<&'static str>,
    /// Phase 5: tire state.
    pub compound: TireCompound,
    /// Accumulated wear fraction (0 = fresh, 1 = dead; floored at 1.25).
    pub wear: f64,
    pub best_lap: Option<f64>,
    /// Phase 5: pit state.
    pub pit_phase: PitPhase,
    /// sim_time when the car entered the pit lane (for the stop log).
    pub pit_entered_t: f64,
    /// Strategy-resolved stop lap for this car (None = will not stop).
    pub stop_lap: Option<u32>,
    /// Compound fitted at the stop.
    pub switch_to: Option<TireCompound>,
    pub stops_done: u32,
    pub pit_stops: Vec<PitStopRecord>,
}

#[derive(Clone, Debug)]
pub struct ContactEvent {
    pub tick: u64,
    pub a: u32,
    pub b: u32,
    pub closing_speed: f64,
    /// Component of car `a`'s velocity directed into car `b` (m/s). Negative
    /// when `a` was pulling away. Phase 11: this is the fault evidence — the
    /// difference between this and `closing_into_b` is what separates the two
    /// cars, and a small difference means the evidence does not. Recorded here
    /// rather than recomputed by the stewarding aid so the evidence and the
    /// event it came from can never disagree.
    pub closing_into_b: f64,
    /// Component of car `b`'s velocity directed into car `a` (m/s).
    pub closing_into_a: f64,
    pub s: f64,
    pub retired: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RacePhase {
    Ready,
    /// Rolling formation lap (realism pack): the field pulls away in grid
    /// order at reduced pace, no overtaking, and re-forms on the grid for
    /// the standing start. Timed sessions and pack-off races skip it.
    FormationLap,
    /// Standing-start formation: grid locked under brakes, lights counting
    /// up, then green. Timed sessions skip this phase entirely.
    Formation,
    Running,
    Finished,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct LeaderRow {
    pub id: u32,
    pub pos: u32,
    pub laps_done: u32,
    pub gap_s: Option<f64>,
    pub retired: bool,
    pub finished: bool,
    pub v: f64,
    /// Phase 5: best lap, compound, stop count, in-pit flag.
    pub best_lap_s: Option<f64>,
    pub compound: &'static str,
    pub stops: u32,
    pub in_pit: bool,
    /// Realism pack W3: fuel fraction (percent of capacity) and tyre temp
    /// (percent warmed). None when the pack is off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuel: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tyre_temp: Option<u8>,
    /// Realism pack: why the car is out, for the classification sheet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dnf: Option<&'static str>,
}

pub struct RaceState {
    pub tick: u64,
    pub sim_time: f64,
    pub phase: RacePhase,
    pub cars: Vec<CarEntry>,
    pub contacts: Vec<ContactEvent>,
    /// Leaderboard computed at the driver rate.
    pub leaderboard: Vec<LeaderRow>,
    /// Race-control log (lights, green, pits, fastest lap) — deterministic,
    /// reproduced identically on replay.
    pub rc_events: Vec<RcEvent>,
    /// Current start-light count during the formation phase (0-5).
    pub lights: u8,
    /// Tick at which lights go out (0 = rolling start, no formation).
    pub green_tick: u64,
    /// Session-fastest lap so far: (car id, time, lap number).
    pub fastest: Option<(u32, f64, u32)>,
}

pub struct Race {
    pub config: RaceConfig,
    pub track: Track,
    pub state: RaceState,
    car_params: CarParams,
    /// Per-car performance multipliers, precomputed in `Race::new`. In
    /// `Classic` mode every entry is 1.0 and this table is inert; in `Teams`
    /// mode it carries the team machinery plus the seeded driver-form delta.
    /// Precomputed (rather than consulted from the team file each tick) so a
    /// tick reads a plain array and the field cannot drift mid-race.
    car_perf: Vec<CarPerf>,
    drivers: Vec<Driver>,
    controls: Vec<Controls>,
    /// Traffic speed caps from the last awareness pass (INF = no cap).
    caps_hint: Vec<f64>,
    state_hash: u64,
    tick_costs_us: Vec<f64>,
    /// Per-pair impulse cooldown (flat n×n, last-impulse tick). One impulse
    /// per contact episode: without it, a pair resting in overlap re-fires
    /// the crunch every tick and the pair's speed craters, trapping
    /// followers into retirement-level closing speeds (cascade engine).
    pair_cooldown: Vec<u64>,
    deadline_misses: u64,
    recorded: Vec<(u32, u64, Controls)>, // (car, tick, controls) at 50 Hz
    /// Phase 12: controls for car 0 supplied from outside the race, by a
    /// learned policy. `None` — the default, and the only value every earlier
    /// phase ever sees — keeps the race bit-for-bit identical, so no observer
    /// rule is at risk here: this is not an observer, it is a replacement
    /// controller, and a race driven this way is deliberately a *different*
    /// race with a different trajectory hash.
    pub learned_controls: Option<Controls>,
    /// Phase 7 advisor runtime. `None` keeps the Phase 5 race bit-for-bit.
    ai: Option<AiRuntime>,
    /// True when the sim may pause at a lap boundary to wait for the model.
    /// Set from the run mode: a max-speed run has no real-time budget to
    /// protect, so waiting once per `every_laps` laps costs nothing real. A
    /// realtime run must never stall the 250 Hz loop, so it stays
    /// fire-and-forget there and stale-rejects answers that arrive late.
    blocking_advisor: bool,
    /// Model wait consumed by the tick currently running, subtracted from
    /// that tick's cost so an advisor stall never looks like physics
    /// missing its deadline.
    ai_block_this_tick_s: f64,
    /// Every advisor exchange of this run, in order (plan requirement: the
    /// prompt summary, model, response, latency, application tick, and
    /// outcome must all be on the record).
    pub ai_log: Vec<strategy::AiExchange>,
    /// Phase 8 weather runtime. `None` keeps the Phase 7 race bit-for-bit —
    /// every weather code path below is guarded on this being `Some`.
    pub weather: Option<WeatherRuntime>,
    /// Immutable half of the weather config (zones, error schedule); the
    /// runtime holds the mutable slider values.
    weather_cfg: Option<WeatherConfig>,
    /// Overrides that have actually fired this run, with the tick they landed
    /// on. Mirrors `RcEvent::WeatherApplied` for the results sheet.
    weather_applied: Vec<AppliedOverride>,
    /// Phase 8 telemetry: mistakes delivered per car, and a running grip sum
    /// for car 0 so the results can say how wet the race actually was.
    weather_errors_by_car: Vec<u32>,
    weather_grip_sum: f64,
    weather_grip_samples: u64,
    /// Lap whose mistake set was last tallied, so the count happens once.
    weather_error_lap: u32,
    /// Realism pack, resolved (inert normalised to None).
    realism: Option<RealismConfig>,
    /// The mistake generator the drivers actually answer to: the weather
    /// block's config when it set one, else the realism pack's own. Both
    /// paths read it through `error_cfg`, never through `weather_cfg`.
    error_cfg: Option<crate::weather::DriverErrorConfig>,
    /// Realism scratch: the last lap each car was failure-checked (a guard,
    /// not state — updated identically in both paths).
    mech_checked: Vec<u32>,
    /// The grid slot each car starts from (resolved grid order). Used again
    /// by the rolling formation lap's re-form snap.
    grid_slots: Vec<u32>,
    /// Max field progress when the formation lap began — the one-lap target.
    formation_start_progress: f64,
    /// Race control: the safety car's deploy tick, `None` when not out.
    sc_deployed: Option<u64>,
    /// The leader's crossing count at deploy — the SC comes in at the end of
    /// the first lap the leader COMPLETES after the minimum SC time.
    sc_leader_laps: u32,
    /// Tick the last SC period ended (cooldown between deployments).
    sc_last_end: u64,
    /// Race-control hazards: (retirement tick, track-s position) of every
    /// car that went out on the track. Drives SC deployment and yellow
    /// zones; entries expire.
    hazards: Vec<(u64, f64)>,
    /// How many hazard entries race control has already responded to.
    hazards_seen: usize,
    /// Active yellow zones: (tick opened, s of the incident).
    yellows: Vec<(u64, f64)>,
    /// The safety car itself, while deployed: absolute progress along the
    /// race order (comparable to CarEntry.total_progress) and the lap count
    /// it started on. A NON-car entity — no CarEntry, so it never touches
    /// classification, telemetry, contact pairs or the state hash; the field
    /// sees it only through the awareness snapshot.
    sc_car: Option<(f64, u32)>,
    /// Ids currently under a blue flag (scratch, rebuilt each driver tick).
    blue_flags: Vec<u32>,
    /// DRS zones as (from_s, length) arcs, derived once from the circuit.
    drs_zones: Vec<(f64, f64)>,
    /// Per car: DRS active this driver tick (scratch, rebuilt each pass).
    drs_active: Vec<bool>,
    // ---------------------------------------------------------------------
    // Phase 9. Everything below is a *observer* of the race. It is reachable
    // from nowhere in `mix_state`, which is what keeps a telemetry-enabled
    // race hashing identically to one without a receiver attached.
    // ---------------------------------------------------------------------
    /// The deterministic cue log. `None` keeps the cue vocabulary inert.
    pub sound: Option<SoundRuntime>,
    /// The UDP link, opened only when the run asked for telemetry.
    telemetry: Option<TelemetryLink>,
    /// Why the link could not be opened, if it was asked for and failed.
    telemetry_error: Option<String>,
    telemetry_cfg: Option<TelemetryConfig>,
    /// Ticks between telemetry publishes, from the configured rate.
    telemetry_interval: u64,
    /// How many `rc_events` have already been turned into cues.
    sound_event_cursor: usize,
    /// Lap crossings last seen, to detect a lap completing this tick.
    sound_last_laps: Vec<u32>,
    /// Contacts already turned into cues.
    sound_contacts_seen: usize,
    // ---------------------------------------------------------------------
    // Phase 11. Also an observer, under the same rule: the incident record is
    // reachable from nowhere in `mix_state`, so attaching the stewarding aid
    // cannot change what the cars did.
    // ---------------------------------------------------------------------
    /// The incident recorder. `None` keeps every earlier gate and artifact
    /// valid, because a race without this field is the Phase 10 race exactly.
    pub steward: Option<steward::Recorder>,
    /// Reused per-tick sample buffer, so the recorder does not allocate one
    /// per car per tick in the hot loop.
    steward_samples: Vec<steward::CarSample>,
}

/// Live advisor state for the advised car. Only ever touched on the sim
/// thread — the model itself runs on its own worker thread, so no physics
/// tick ever waits on it.
struct AiRuntime {
    cfg: strategy::AdvisorConfig,
    handle: strategy::AdvisorHandle,
    /// The single outstanding question, with the summary that asked it.
    inflight: Option<Inflight>,
    next_id: u64,
    /// Lap we last asked on (one question per lap, not one per tick).
    last_ask_lap: Option<u32>,
    /// Pace multiplier currently in force for the advised car; 1.0 = the
    /// deterministic Phase 5 driver, i.e. the fallback state.
    pace_scale: f64,
    /// Model failures seen this run (endpoint down, HTTP error, ...).
    transport_errors: u64,
}

#[derive(Clone)]
struct Inflight {
    id: u64,
    tick: u64,
    lap: u32,
    summary: String,
}

/// Aggregate advisor statistics — the "did it help, tie, or hurt" evidence.
#[derive(Clone, Debug, Default)]
pub struct AiSummary {
    pub exchanges: usize,
    pub applied: usize,
    pub rejected: usize,
    pub stale: usize,
    pub low_confidence: usize,
    pub bad_answer: usize,
    pub timeout: usize,
    pub error: usize,
    pub latency_ms_p50: f64,
    pub latency_ms_p99: f64,
    pub transport_errors: u64,
    pub pace_scale_final: f64,
    pub asks_suppressed: usize,
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Retired-car control: steer OFF the racing line while braking gently,
/// then halt at the edge — a wreck parked mid-line turns one incident
/// into a pile-up (a wall of dead cars cannot be passed). 0.12 rad is
/// about the tire-limited max drift turn at speed, so the car slides
/// sideways-by-several-metres while scrubbing speed, parking clear of
/// the line. A pure function of reconstructible state so the replay
/// path reproduces it bit-identically.
fn retired_controls(e: &CarEntry) -> Controls {
    let off_line = e.lateral.abs() < 6.0;
    Controls {
        throttle: 0.0,
        brake: if off_line { 0.25 } else { 1.0 },
        steer_cmd: if off_line {
            if e.lateral >= 0.0 { 0.12 } else { -0.12 }
        } else {
            0.0
        },
    }
}

/// Box service lock: the car stands on the brakes while the team works.
/// Constant, and applied per-tick in BOTH execution paths (like the
/// retired law), so replay never needs these controls in the log.
fn locked_controls() -> Controls {
    Controls { throttle: 0.0, brake: 1.0, steer_cmd: 0.0 }
}

fn fnv_mix(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Forward crossing of a mark on the lap coordinate between two successive
/// projections. The distance guard rejects the wrap-around jump (a car can
/// only travel a fraction of a lap in one 4 ms tick).
fn crossed(prev_s: f64, mark: f64, s: f64, lap_len: f64) -> bool {
    prev_s < mark && s >= mark && (s - prev_s) < lap_len * 0.5
}

/// Car-to-car contact envelope: two cars are in contact when their
/// centres are closer than this (bounding circles, car ≈ 5.6 × 2.0 m).
const CONTACT_DIST: f64 = 3.0;
/// Minimum ticks between impulses for the same pair (100 ms at 250 Hz):
/// one crunch per contact episode; resting overlap only separates.
const IMPULSE_COOLDOWN_TICKS: u64 = 25;

impl Race {
    pub fn new(config: RaceConfig) -> Race {
        // The spec was validated where the user supplied it (--track / the
        // batch config); reaching here with an undriveable circuit is an
        // engine bug, so the panic is the honest behaviour.
        let track =
            config.track.build().expect("track spec must build (validate at load)");
        // A solo qualifying run fields one car with a foreign identity: n
        // collapses to 1, and every per-car lookup below resolves for that
        // car's id, not for 0.
        let (car_count, solo_id): (u32, Option<u32>) = match config.solo_car {
            Some(id) => (1, Some(id)),
            None => (config.cars, None),
        };
        let n = car_count as usize;
        // Car id occupying each grid slot. Identity unless a qualifying
        // result says otherwise; a malformed order falls back to identity
        // rather than panicking — callers validate at load, this only has to
        // be deterministic.
        let slot_car: Vec<u32> = {
            let identity: Vec<u32> = (0..car_count).collect();
            let supplied = match solo_id {
                Some(id) => vec![id],
                None => config.grid_order.clone(),
            };
            let is_permutation = supplied.len() == n
                && {
                    let mut sorted = supplied.clone();
                    sorted.sort_unstable();
                    sorted == identity
                };
            if is_permutation { supplied } else { identity }
        };
        // Inverse map: car id -> grid slot. The cars vec stays id-indexed
        // (every observer assumes `cars[i].id == i`); only the SPAWN
        // position follows the slot.
        let mut slot_of = vec![0u32; n];
        // Kept on the race: the rolling formation lap re-forms the grid from
        // the same placement (a pure function, so the snap replays).
        for (slot, car) in slot_car.iter().enumerate() {
            slot_of[*car as usize] = slot as u32;
        }
        let len = track.length();
        // DRS zones: the longest straights (|curvature| below the sweep
        // threshold for a sustained run); a zone is the run's middle 70%,
        // keeping the ends honest braking and turn-in zones. Pure from the
        // geometry, so both paths derive identical zones.
        let drs_zones = if config.realism.map(|r| r.drs_zones).unwrap_or(false) {
            derive_drs_zones(&track)
        } else {
            Vec::new()
        };
        // Pit placement: on a custom circuit the stadium's absolute s values
        // (150/210/290) mean nothing — fractions of the lap place the lane,
        // overridable in the track file. A replay re-logs the resolved
        // absolute values, so it lands here with non-defaults and keeps them.
        let mut pit_cfg = config.pit;
        if let Some(c) = &config.track.custom {
            let d = PitConfig::default();
            if pit_cfg.entry_s == d.entry_s
                && pit_cfg.box_s == d.box_s
                && pit_cfg.exit_s == d.exit_s
            {
                let f = c.pit.unwrap_or_default();
                pit_cfg.entry_s = f.entry * len;
                pit_cfg.box_s = f.box_frac * len;
                pit_cfg.exit_s = f.exit * len;
            }
        }
        // Pit config only runs a race-mode session with sound geometry;
        // timed sessions and single-lap sprints don't stop.
        let pit = if config.mode == SessionMode::Race && pit_cfg.usable(len) {
            pit_cfg
        } else {
            PitConfig { enabled: false, ..pit_cfg }
        };
        // Resolve the strategy once, per car (all cars share it here; the
        // Auto wave staggers stop laps so the single box never queues the
        // whole field into one lap).
        let (base_lap, waves, switch_default) = match config.strategy {
            PitStrategy::None => (0, 0, TireCompound::Medium),
            PitStrategy::Auto => {
                let lap = (config.laps / 2).max(1);
                let waves = (config.laps - lap).clamp(1, 4);
                (lap, waves, TireCompound::Medium)
            }
            PitStrategy::Fixed { lap, switch_to } => (lap, 1, switch_to),
        };
        let stops_planned = config.mode == SessionMode::Race
            && pit.enabled
            && config.strategy != PitStrategy::None
            && base_lap >= 1
            && base_lap < config.laps;
        let mut cars = Vec::with_capacity(n);
        for (i, &slot) in slot_of.iter().enumerate() {
            // Staggered two-column grid behind the start line: the first
            // crossing of the line starts lap 1 timing (formation lap 0).
            // `i` is the CAR; the slot it starts from is whatever the grid
            // order (default: identity) assigned it — qualifying's whole
            // product is this line.
            let row = (slot / 2) as f64;
            let col = slot % 2;
            let s = len - 15.0 - row * config.grid_row_gap_m;
            let lateral = if col == 0 { -2.2 } else { 2.2 };
            let (x, y, heading) = track.point_at(s);
            // offset the spawn sideways from the centreline
            let (lx, ly) = (-heading.sin(), heading.cos());
            let (stop_lap, switch_to) = if stops_planned {
                (
                    Some((base_lap + (i as u32 % waves)).min(config.laps - 1)),
                    Some(switch_default),
                )
            } else {
                (None, None)
            };
            let entry = CarEntry {
                id: i as u32,
                car: CarState {
                    pos_x: x + lx * lateral,
                    pos_y: y + ly * lateral,
                    heading,
                    ..Default::default()
                },
                s,
                lateral,
                line_offset: 0.0,
                laps_crossings: 0,
                lap_times: Vec::new(),
                lap_start_t: 0.0,
                total_progress: -(len - s), // negative: behind the line
                retired: false,
                finish_time_s: None,
                last_contact_tick: 0,
                damage: 0.0,
                flatspot: 0.0,
                dnf: None,
                fuel_mass: if config.realism.map(|r| r.fuel).unwrap_or(false) {
                    FUEL_CAPACITY_KG
                } else {
                    0.0
                },
                tyre_temp: 0.0,
                compound: config.start_compound,
                wear: 0.0,
                best_lap: None,
                pit_phase: PitPhase::OnTrack,
                pit_entered_t: 0.0,
                stop_lap,
                switch_to,
                stops_done: 0,
                pit_stops: Vec::new(),
            };
            cars.push(entry);
        }
        // Per-car driver with deterministic seed and a pace envelope.
        // Classic mode: the Phase 4 stand-in spread (faster cars lower in the
        // grid). Teams mode: the envelope comes from the team's machinery —
        // top speed, cornering grip and brake anticipation — with the seeded
        // driver-form delta, so a team's second car is not a photocopy.
        // Resolved per CAR id: a solo run measures car `solo_id`'s own
        // machinery, and a qualifying-ordered race keeps each car in the car
        // it qualified in.
        let car_perf: Vec<CarPerf> = (0..n as u32)
            .map(|i| CarPerf::resolve(config.field, config.seed, solo_id.unwrap_or(i)))
            .collect();
        let base = DriverParams::default();
        let mut drivers = Vec::with_capacity(n);
        for (i, perf) in car_perf.iter().enumerate() {
            let mut dp = base;
            if config.field == FieldMode::Teams {
                // Team machinery sets the envelope's ceilings; the seeded
                // form delta nudges how close the driver works to them.
                dp.v_straight_max *= perf.top_speed * perf.form;
                dp.a_lat_max *= perf.tyres;
                dp.a_brake_anticipate *= perf.brakes * perf.form;
            } else {
                let ident = solo_id.unwrap_or(i as u32);
                let scale = 1.0 - config.pace_spread * ident as f64;
                dp.v_straight_max *= scale;
                dp.a_lat_max *= scale;
                dp.a_brake_anticipate *= scale;
            }
            dp.noise_amplitude = config.noise;
            let ident = solo_id.unwrap_or(i as u32);
            drivers.push(Driver::new(
                dp,
                config.seed ^ (0x9E37_u64.wrapping_mul(ident as u64 + 1)),
            ));
        }
        // Standing start: five lights at 1 s intervals, then a seeded hold
        // (1.0–3.2 s) before green — every tick of it deterministic. The
        // hold goes through the LCG scatter so adjacent seeds can't land on
        // the same hold (raw low bits of a seed XOR are too correlated).
        let formation = config.mode == SessionMode::Race && config.formation;
        let green_tick = if formation {
            let hold_s =
                1.0 + (crate::rng::Lcg::new(config.seed).next_u64() % 2201) as f64 / 1000.0;
            ((5.0 + hold_s) / DT).ceil() as u64
        } else {
            0
        };
        let (ai_cfg, legacy) = (config.ai.clone(), config.legacy_v2);
        // An inert weather config is normalised down to None so a caller who
        // passes `Some(WeatherConfig::default())` still gets the exact Phase 7
        // trajectory — "dry, no overrides, no errors" must be a no-op, not a
        // different-but-equal race.
        let mut wcfg = config.weather.clone().filter(|w| !w.is_inert());
        if let Some(w) = wcfg.as_mut() {
            w.normalise();
        }
        // The realism pack: an inert config is the absent config (the same
        // normalisation weather and stewarding undergo).
        let realism = config.realism.filter(|r| !r.is_inert());
        // The mistake generator: an explicit weather error block wins (it
        // carries the user's own seed and magnitudes); otherwise realism
        // supplies one seeded from the race. Timing stays pure per
        // (car, lap) either way, so replay reproduces every mistake.
        let error_cfg = match wcfg.as_ref().map(|w| &w.driver_errors) {
            Some(de) if !de.is_off() => Some(*de),
            _ => realism
                .filter(|r| r.error_rate > 0.0)
                .map(|r| crate::weather::DriverErrorConfig {
                    rate_per_lap: r.error_rate,
                    seed: config.seed,
                    ..crate::weather::DriverErrorConfig::default()
                }),
        };
        let runtime = wcfg.as_ref().map(WeatherRuntime::new);

        // Phase 9 setup. Both halves are Option-gated and both are observers:
        // `sound` decides whether a cue log exists at all, and `link` opens a
        // socket thread only when a target was actually configured. A failure
        // to open the link is NOT fatal — an unusable address is a
        // configuration error the caller is told about, but the race is the
        // product and still runs.
        let scfg = config.sound.filter(|c| !c.is_inert());
        let gearbox = config
            .telemetry
            .as_ref()
            .map(|t| t.gearbox)
            .unwrap_or_default();
        let mut sound = scfg.map(|c| SoundRuntime::new(c, n, gearbox));
        let tcfg = config.telemetry.clone().filter(|t| !t.is_inert());
        let telemetry_interval = tcfg.as_ref().map(|t| t.interval_ticks(DT)).unwrap_or(1);
        // A link that will not open is reported, not swallowed: the run
        // continues on a good deal that telemetry was requested and did not
        // start, and the reason travels in the result.
        let (link, telemetry_error) = match tcfg.as_ref() {
            Some(t) => match TelemetryLink::spawn(t) {
                Ok(l) => (Some(l), None),
                Err(e) => (None, Some(e)),
            },
            None => (None, None),
        };
        if let Some(rt) = sound.as_mut() {
            rt.prime(&[]);
        }

        // Phase 11 setup. An inert config is normalised to None on the same
        // terms as weather and sound, so `--steward` off, absent, or explicitly
        // disabled all mean the identical engine — and an aid that is switched
        // on cannot change a single tick of the race it watches.
        let steward_cfg = config.steward.filter(|s| !s.is_inert());
        let steward = steward_cfg.map(|c| steward::Recorder::new(c, track.half_width));

        Race {
            state: RaceState {
                tick: 0,
                sim_time: 0.0,
                phase: RacePhase::Ready,
                cars,
                contacts: Vec::new(),
                leaderboard: Vec::new(),
                rc_events: Vec::new(),
                lights: 0,
                green_tick,
                fastest: None,
            },
            config,
            track,
            car_params: CarParams::default(),
            car_perf,
            drivers,
            controls: vec![Controls::default(); n],
            caps_hint: vec![f64::INFINITY; n],
            state_hash: FNV_OFFSET,
            tick_costs_us: Vec::new(),
            pair_cooldown: vec![0; n * n],
            deadline_misses: 0,
            recorded: Vec::new(),
            learned_controls: None,
            ai: match &ai_cfg {
                // Legacy replays never ask the model, even if a config
                // carries the field: a replay must reproduce the logged
                // trajectory, and model latency is not reconstructible.
                Some(cfg) if !legacy && n > 0 => Some(AiRuntime {
                    cfg: cfg.clone(),
                    handle: strategy::spawn_advisor(cfg),
                    inflight: None,
                    next_id: 1,
                    last_ask_lap: None,
                    pace_scale: 1.0,
                    transport_errors: 0,
                }),
                _ => None,
            },
            ai_log: Vec::new(),
            blocking_advisor: false,
            ai_block_this_tick_s: 0.0,
            weather: runtime,
            weather_cfg: wcfg,
            weather_applied: Vec::new(),
            weather_errors_by_car: vec![0; n],
            weather_grip_sum: 0.0,
            weather_grip_samples: 0,
            weather_error_lap: u32::MAX,
            grid_slots: slot_of.clone(),
            formation_start_progress: 0.0,
            realism,
            error_cfg,
            mech_checked: vec![0; n],
            sc_deployed: None,
            sc_leader_laps: 0,
            sc_last_end: 0,
            hazards: Vec::new(),
            hazards_seen: 0,
            yellows: Vec::new(),
            blue_flags: Vec::new(),
            sc_car: None,
            drs_zones,
            drs_active: vec![false; n],
            sound,
            telemetry: link,
            telemetry_error,
            telemetry_cfg: tcfg,
            telemetry_interval,
            sound_event_cursor: 0,
            sound_last_laps: vec![0; n],
            sound_contacts_seen: 0,
            steward,
            steward_samples: Vec::with_capacity(n),
        }
    }

    /// Choose whether the sim may wait for the model at a lap boundary.
    /// Called by the run loops; a replay never calls it and so never waits.
    pub fn set_blocking_advisor(&mut self, blocking: bool) {
        self.blocking_advisor = blocking && self.ai.is_some();
    }

    /// Whether this race is currently allowed to wait on the model.
    pub fn ai_blocks(&self) -> bool {
        self.blocking_advisor
    }

    /// Begin the session: formation + lights for a standing-start race,
    /// immediate green for rolling timed sessions.
    /// Car 0's action from the deterministic local driver, computed for the
    /// current state but **not** applied.
    ///
    /// This exists so a training loop can label observations with the engine's
    /// own expert rather than a reimplementation of it. The plan warns against
    /// assuming self-play produces a good teacher; the teacher here is the
    /// Phase 4/5 driver, and asking the engine is the only way the labels
    /// cannot drift away from the behaviour they are supposed to imitate.
    ///
    /// It advances car 0's driver state (steering rate limit, noise LCG), which
    /// is harmless precisely because car 0 is under the learned policy and its
    /// local driver is never consulted again for that race.
    pub fn baseline_control_for_car0(&mut self) -> Controls {
        // During formation the race applies the locked standing start to every
        // car and never consults a driver. Asking one anyway advances its
        // steering rate limiter and its noise LCG a whole standing start ahead
        // of the race, so the first control it produced after green would be
        // one the real driver would never have given. Answering with what the
        // race actually applies keeps the teacher honest AND leaves the
        // driver's state where the race left it.
        if self.state.phase == RacePhase::Formation {
            return locked_controls();
        }
        let scale = self.grip_scale(0);
        self.drivers[0].set_grip_scale(scale);
        let input = DriverInput {
            s: self.state.cars[0].s,
            lateral: self.state.cars[0].line_offset,
            v_cap: self.caps_hint[0] * self.ai_pace_scale(0),
        };
        let car = self.state.cars[0].car;
        self.drivers[0].drive(&car, &self.track, input)
    }

    pub fn start(&mut self) {
        // The rolling formation lap is a realism-pack feature for race
        // sessions: one lap in grid order, then the standing lights.
        let rolling = self
            .realism
            .map(|r| r.formation_lap)
            .unwrap_or(false)
            && self.config.mode == SessionMode::Race
            && self.state.green_tick > 0;
        if rolling {
            self.formation_start_progress = self
                .state
                .cars
                .iter()
                .map(|e| e.total_progress)
                .fold(f64::NEG_INFINITY, f64::max);
        }
        self.state.phase = if rolling {
            RacePhase::FormationLap
        } else if self.state.green_tick > 0 {
            RacePhase::Formation
        } else {
            RacePhase::Running
        };
    }

    /// Total progress in metres along the race order (handles the pre-line
    /// start as negative progress so the grid order is correct from tick 0).
    /// Also advances lap accounting, best/fastest laps, and the pit-lane
    /// crossings — everything that follows from the refreshed position.
    fn recompute_progress(&mut self) {
        let len = self.track.length();
        let pit = self.config.pit;
        let sim_t = self.state.sim_time;
        let mut fastest_candidate: Option<(u32, f64, u32)> = None;
        for e in &mut self.state.cars {
            let prev_s = e.s;
            let (s, lateral) = self.track.project(e.car.pos_x, e.car.pos_y);
            e.lateral = lateral;
            // forward line crossing? Finished cars keep circulating through
            // the classification grace window — their crossings still
            // advance progress, but they must not keep timing laps: the
            // cool-down lap is not a race lap and must never set a
            // fastest-lap time.
            let crossings_before = e.laps_crossings;
            let timing_open = e.finish_time_s.is_none();
            if e.s > 0.8 * len && s < 0.2 * len {
                e.laps_crossings += 1;
                if e.laps_crossings == 1 {
                    e.lap_start_t = sim_t; // lap 1 timing starts
                } else if timing_open {
                    let t = sim_t - e.lap_start_t;
                    let lap_no = e.laps_crossings - 1; // the lap just completed
                    e.lap_times.push(t);
                    e.lap_start_t = sim_t;
                    if e.best_lap.is_none_or(|b| t < b) {
                        e.best_lap = Some(t);
                        fastest_candidate = Some((e.id, t, lap_no));
                    }
                } else {
                    e.lap_start_t = sim_t;
                }
            }
            // Pit-lane crossings, in road order: entry, box, exit. A car is
            // armed on its planned stop lap (crossings == stop lap number).
            // Retired cars never arm: a wreck belongs on the track, not
            // gliding into the box.
            if pit.enabled {
                let armed = !e.retired && e.stop_lap == Some(e.laps_crossings);
                match e.pit_phase {
                    PitPhase::OnTrack => {
                        if armed && crossed(prev_s, pit.entry_s, s, len) {
                            e.pit_phase = PitPhase::InLane;
                            e.pit_entered_t = sim_t;
                            let (car, lap) = (e.id, e.laps_crossings);
                            self.state.rc_events.push(RcEvent::PitEntered { car, lap });
                        }
                    }
                    PitPhase::InLane => {
                        if crossed(prev_s, pit.box_s, s, len) {
                            e.pit_phase = PitPhase::Servicing {
                                // W3: refuelling takes time proportional to
                                // what the tank is missing — short-fill and
                                // stop again, or long-fill and trundle.
                                remaining_s: pit.service_time_s
                                    + if self.realism.map(|r| r.fuel).unwrap_or(false) {
                                        (FUEL_CAPACITY_KG - e.fuel_mass).max(0.0)
                                            * REFUEL_SEC_PER_KG
                                    } else {
                                        0.0
                                    },
                            };
                        } else if crossed(prev_s, pit.exit_s, s, len) {
                            e.pit_phase = PitPhase::OnTrack;
                            let rec = PitStopRecord {
                                fitted: e.compound,
                                car: e.id,
                                lap: e.stop_lap.unwrap_or(0),
                                entered_t: e.pit_entered_t,
                                exited_t: sim_t,
                                held_s: pit.service_time_s,
                            };
                            e.pit_stops.push(rec);
                            let (car, lane_s) = (e.id, sim_t - e.pit_entered_t);
                            self.state.rc_events.push(RcEvent::PitExited { car, lane_s });
                        }
                    }
                    PitPhase::Servicing { .. } => {} // stationary; countdown runs per tick
                }
            }
            e.s = s;
            e.total_progress = e.laps_crossings as f64 * len + s - len; // start ≈ -15 m
            if crossings_before != e.laps_crossings
                && e.laps_crossings == self.config.laps + 1
                && e.finish_time_s.is_none()
            {
                e.finish_time_s = Some(sim_t);
            }
            let _ = crossings_before;
        }
        // Session fastest lap: strictly better than the standing best.
        if let Some(c @ (_, t, _)) = fastest_candidate {
            if self.state.fastest.is_none_or(|(_, bt, _)| t < bt) {
                self.state.fastest = Some(c);
                let (car, time_s, lap) = c;
                self.state.rc_events.push(RcEvent::FastestLap { car, time_s, lap });
            }
        }
    }

    /// Race control, at the driver rate, BEFORE awareness reads the caps:
    /// deploys and clears the safety car, prunes expired yellow sectors.
    /// Every transition derives from recorded race state (retirements, the
    /// leader's crossing count), so both execution paths agree with no new
    /// logging.
    fn race_control(&mut self) {
        let Some(rc) = self.realism else { return };
        let now = self.state.tick;
        // Expire yellow sectors.
        self.yellows.retain(|(t0, _)| now.saturating_sub(*t0) < YELLOW_TICKS);

        // Deployment: the first race-control hazard nobody has answered yet,
        // silenced during the cooldown and never during timed sessions.
        if rc.safety_car
            && self.config.mode == SessionMode::Race
            && self.sc_deployed.is_none()
            && now.saturating_sub(self.sc_last_end) >= SC_COOLDOWN_TICKS
        {
            while self.hazards_seen < self.hazards.len()
                && now.saturating_sub(self.hazards[self.hazards_seen].0) > YELLOW_TICKS + 5_000
            {
                // A hazard older than the yellow window was already handled by
                // its sector; skip it without an SC.
                self.hazards_seen += 1;
            }
            if self.hazards_seen < self.hazards.len() {
                let leader = self
                    .state
                    .cars
                    .iter()
                    .filter(|e| !e.retired)
                    .map(|e| e.total_progress)
                    .fold(f64::NEG_INFINITY, f64::max);
                let leader_laps = self
                    .state
                    .cars
                    .iter()
                    .filter(|e| !e.retired)
                    .map(|e| e.laps_crossings)
                    .max()
                    .unwrap_or(0);
                self.sc_deployed = Some(now);
                self.sc_leader_laps = leader_laps;
                self.hazards_seen = self.hazards.len();
                // The SC takes to the track 60 m AHEAD of the race leader:
                // the field's follow law queues everyone behind it, and the
                // 22 m/s cap means nobody runs it down.
                self.sc_car = Some((leader + 60.0, leader_laps));
                self.state
                    .rc_events
                    .push(RcEvent::SafetyCar { phase: "deployed" });
            }
        }
        // Clearance: the minimum time has elapsed AND the leader has
        // completed a crossing since deployment — "safety car in this lap",
        // green the moment it is announced.
        if let Some(t0) = self.sc_deployed {
            let leader_laps = self
                .state
                .cars
                .iter()
                .filter(|e| !e.retired)
                .map(|e| e.laps_crossings)
                .max()
                .unwrap_or(0);
            if now.saturating_sub(t0) >= SC_MIN_TICKS && leader_laps > self.sc_leader_laps {
                self.sc_deployed = None;
                self.sc_last_end = now;
                self.sc_car = None; // the SC peels into the pit lane
                self.hazards.clear();
                self.hazards_seen = 0;
                self.state
                    .rc_events
                    .push(RcEvent::SafetyCar { phase: "clear" });
            }
        }

        // Advance the SC entity (driver rate): it circulates at SC pace on
        // the racing line. Pure bookkeeping — its position is derived, not
        // hashed, and the field never touches it physically.
        if let Some((progress, spawn_laps)) = &mut self.sc_car {
            *progress += SC_SPEED * DT * DRIVER_EVERY as f64;
            let _ = spawn_laps; // laps implied by progress vs the field
        }
    }

    /// The SC entity's absolute progress and spawn-lap, while deployed.
    pub fn sc_entity_state(&self) -> Option<(f64, u32)> {
        self.sc_car
    }

    /// The safety car's pose on the track while deployed: (x, y, heading).
    pub fn sc_car_pose(&self) -> Option<(f64, f64, f64)> {
        let (progress, _laps) = *self.sc_car.as_ref()?;
        let len = self.track.length();
        let s = progress.rem_euclid(len);
        let (x, y, h) = self.track.point_at(s);
        Some((x, y, h))
    }

    /// The race-control state the dashboard sees: SC phase label and the
    /// active yellow sectors as (from_s, to_s) pairs around the lap.
    pub fn race_control_brief(&self) -> (Option<&'static str>, Vec<(f64, f64)>) {
        let len = self.track.length();
        let sc = self.sc_deployed.map(|_| "deployed");
        let yellows = self
            .yellows
            .iter()
            .map(|(_t, ys)| {
                let from = (*ys - YELLOW_ZONE_M).rem_euclid(len);
                let to = (*ys + YELLOW_ZONE_M).rem_euclid(len);
                (from, to)
            })
            .collect();
        (sc, yellows)
    }

    /// The safety car's deploy tick while it is out.
    pub fn sc_deployed_tick(&self) -> Option<u64> {
        self.sc_deployed
    }

    /// Open yellow sectors as (from_s, to_s) pairs.
    pub fn yellows_active(&self) -> Vec<(f64, f64)> {
        let len = self.track.length();
        self.yellows
            .iter()
            .map(|(_t, ys)| {
                (
                    (*ys - YELLOW_ZONE_M).rem_euclid(len),
                    (*ys + YELLOW_ZONE_M).rem_euclid(len),
                )
            })
            .collect()
    }

    /// The ids with DRS active this driver tick.
    pub fn drs_active_ids(&self) -> Vec<u32> {
        (0..self.state.cars.len())
            .filter(|i| self.drs_active.get(*i).copied().unwrap_or(false))
            .map(|i| self.state.cars[i].id)
            .collect()
    }

    /// The ids currently under a blue flag (about to be lapped).
    pub fn blue_flagged_ids(&self) -> Vec<u32> {
        if let Some(rc) = &self.realism {
            if rc.blue_flags {
                return self.blue_flags.clone();
            }
        }
        Vec::new()
    }

    /// Traffic awareness at the driver rate: follow caps, overtake and
    /// defend offsets, pit-lane speed enforcement. Deterministic: decisions
    /// read the previous tick's positions, cars processed in id order.
    fn traffic_awareness(&mut self) {
        self.race_control();
        let n = self.state.cars.len();
        let mut snapshot: Vec<(f64, f64, f64, bool, bool)> = self
            .state
            .cars
            .iter()
            .map(|e| (e.total_progress, e.lateral, e.car.vel_long, e.retired, e.pit_phase.in_pit()))
            .collect();
        // The safety car joins the awareness snapshot as a moving obstacle:
        // the existing safe-follow law brakes whoever approaches it, so the
        // field queues behind the SC with no new mechanics. (Snapshot index
        // n = the SC; every car loop below still iterates 0..n.)
        if let Some((progress, _laps)) = self.sc_car {
            snapshot.push((progress, 0.0, SC_SPEED, false, false));
        }
        let mut caps = vec![f64::INFINITY; n];
        let mut offsets: Vec<Option<f64>> = vec![None; n];
        let pit = self.config.pit;

        for i in 0..n {
            if snapshot[i].3 {
                continue; // retired cars hold position
            }
            // Nearest car ahead ON MY PATH: progress ahead AND laterally
            // close (< 3.2 m). The window MUST cover the contact circle
            // (3.0 m radius): an obstacle touching me but outside my
            // awareness window is one I never brake for and never swerve
            // around — I just re-hit it every cooldown expiry. Retired
            // (stopped) cars count as hazards — otherwise the field
            // plows into them at full speed. A car on a different line
            // doesn't cap us (that's what lets a pass complete once
            // alongside). Pit cars sit at their own lateral offset outside
            // this window, so track and lane traffic never see each other.
            let my_lat = snapshot[i].1;
            let mut ahead: Option<usize> = None;
            let mut best_gap = f64::MAX;
            // j scans the snapshot INCLUDING the SC entry (index n) — the
            // field must see the safety car as the car ahead and queue.
            for j in 0..snapshot.len() {
                if j == i {
                    continue;
                }
                // A STOPPED (retired) car is a hazard to the whole track, not
                // to one line: it may sit at the edge while I arrive on the
                // racing line, and the old 3.2 m window would let me plow
                // into it at full closing speed. Stopped hazards are seen
                // from anywhere on the tarmac; moving cars keep the 3.2 m
                // line window (that narrowness is what lets a pass complete
                // once alongside).
                let lat_window = if snapshot[j].3 { 12.0 } else { 3.2 };
                if (snapshot[j].1 - my_lat).abs() > lat_window {
                    continue;
                }
                let gap = snapshot[j].0 - snapshot[i].0;
                if gap > 0.0 && gap < best_gap {
                    best_gap = gap;
                    ahead = Some(j);
                }
            }
            let my_v = snapshot[i].2;
            let follow_zone = 0.55 * my_v + 10.0;
            if let Some(j) = ahead {
                let v_ahead = snapshot[j].2;
                if best_gap < follow_zone {
                    // Safe-follow speed over the CLEAR gap: contact happens
                    // when centers are CONTACT_DIST apart, so the first
                    // 3 m of progress gap isn't braking room. Braking to
                    // the bumper (gap < 3) yields cap ≈ 0 — no grinding.
                    // Fast enough to keep pressure but always with room to
                    // brake; a bare "v_ahead - 0.5" cap deadlocks a
                    // standing start (each grid row pins to the row ahead,
                    // cascading to zero). 6 m/s² (not the ~11 available)
                    // because the driver's proportional brake ramp needs
                    // real margin — at 8, fast tail-enders overshot into
                    // contact every time a queue compressed.
                    let clear = (best_gap - CONTACT_DIST).max(0.0);
                    let v_safe =
                        (v_ahead * v_ahead + 2.0 * 6.0 * clear).sqrt() - 0.5;
                    caps[i] = caps[i].min(v_safe.max(0.0));
                    // Overtake attempt on a low-curvature stretch with
                    // runway to finish the move: never launch a pass that
                    // ends in a braking zone, and never from a bumper.
                    let s_me = self.state.cars[i].s;
                    let on_straight = self.track.curvature_at(s_me + 20.0) < 1e-6
                        && self.track.curvature_at(s_me + 60.0) < 1e-6
                        && self.track.curvature_at(s_me + 100.0) < 1e-6;
                    let pass_straight = on_straight
                        && best_gap > 6.0
                        && best_gap < 25.0
                        && my_v > v_ahead - 1.0;
                    // Crawl past a RETIRED car — a permanent obstacle.
                    // This must be situational, not state-based: the
                    // safe-follow law pins a follower to 0 m/s behind a
                    // stopped car (clear gap ≈ 0), so a rule that requires
                    // the passer to already be moving deadlocks — the car
                    // waits forever for a speed the cap forbids. Live slow
                    // cars are NOT swerved around: they recover on their
                    // own, and swerving at a standing start would cross
                    // the grid columns.
                    let pass_stopped = snapshot[j].3 && best_gap < 20.0;
                    if pass_straight || pass_stopped {
                        let lat_ahead = snapshot[j].1;
                        let side = if lat_ahead > 0.0 { -1.0 } else { 1.0 };
                        if pass_stopped {
                            // Aim for the track EDGE on the free side: a
                            // target adjacent to the wreck pulls every
                            // arriving car onto the same line through the
                            // middle — the pack converges and piles up.
                            offsets[i] = Some(side * 4.0);
                            // Release enough cap to execute the crawl —
                            // the follow law alone pins to 0 here.
                            caps[i] = caps[i].max(4.0);
                        } else {
                            // A racing pass only needs to be alongside.
                            offsets[i] =
                                Some((lat_ahead + side * 3.4).clamp(-4.0, 4.0));
                        }
                    }
                } else if best_gap > 40.0 {
                    offsets[i] = Some(0.0); // clear air: back to the centreline
                }
            } else if self.state.cars[i].line_offset.abs() > 0.05 {
                // Clear air after a completed pass or pit exit: release the
                // line back to the centreline.
                offsets[i] = Some(0.0);
            }
            // Basic defending: attacker close behind, faster, and actually
            // on my line → nudge toward the attacker's side to close the
            // door. Pack neighbours in the other column are ignored.
            let mut behind: Option<usize> = None;
            let mut bgap = f64::MAX;
            for j in 0..n {
                if j == i || snapshot[j].3 || snapshot[j].4 {
                    continue;
                }
                if (snapshot[j].1 - my_lat).abs() > 2.0 {
                    continue;
                }
                let gap = snapshot[i].0 - snapshot[j].0;
                if gap > 0.0 && gap < bgap {
                    bgap = gap;
                    behind = Some(j);
                }
            }
            if let Some(j) = behind {
                if bgap < 18.0 && snapshot[j].2 > my_v + 0.5 {
                    let attacker_lat = snapshot[j].1;
                    let side = if attacker_lat >= 0.0 { 1.0 } else { -1.0 };
                    offsets[i] = Some((attacker_lat + side * 1.2).clamp(-4.0, 4.0));
                }
            }
        }
        // Pit-lane enforcement, after the racing rules: armed cars are held
        // to the limit through the brake zone BEFORE the entry (they must
        // arrive already slow — the lane itself is far too short to brake
        // in), lane cars are pinned to the lane offset, and every pit-speed
        // section min-clamps the traffic cap.
        if pit.enabled {
            for (i, e) in self.state.cars.iter_mut().enumerate() {
                if e.retired {
                    continue;
                }
                if e.pit_phase.in_pit() {
                    let in_lane = matches!(e.pit_phase, PitPhase::InLane)
                        && e.s < pit.box_s
                        && pit.box_s - e.s < 15.0;
                    e.line_offset = pit.offset_m;
                    caps[i] = caps[i].min(pit.speed_limit);
                    // Creep to the box: the crew expects a slow approach, so
                    // the car arrives near-stopped instead of sliding 18 m
                    // past its mark under full braking.
                    if in_lane {
                        caps[i] = caps[i].min(4.0);
                    }
                } else if e.stop_lap == Some(e.laps_crossings)
                    && e.s >= pit.entry_s - pit.brake_zone_m
                    && e.s <= pit.exit_s
                {
                    caps[i] = caps[i].min(pit.speed_limit);
                }
            }
        }
        // ---- Race control enforcement (after the per-car law, before the
        // offsets move) ----
        let (rc_on, rc_blue, rc_drs) = match &self.realism {
            Some(r) => (true, r.blue_flags, r.drs_zones),
            None => (false, false, false),
        };
        let sc_out = self.sc_deployed.is_some();
        let len = self.track.length().max(1.0);
        self.blue_flags.clear();

        if rc_on {
            // SAFETY CAR: everyone on track holds SC pace, and overtaking is
            // banned -- pass/defend offsets are cleared (the crawl-past-a-
            // wreck CAP stays; only the swerve-around goes).
            if sc_out {
                for i in 0..n {
                    if !snapshot[i].4 {
                        caps[i] = caps[i].min(SC_SPEED);
                    }
                }
                for (i, off) in offsets.iter_mut().enumerate() {
                    // Pass/defend offsets are race-craft and are what the ban
                    // removes; caps (follow, crawl-past, pit) all stay.
                    if !snapshot[i].4 {
                        *off = None;
                    }
                }
            }

            // LOCAL YELLOW SECTORS: while the SC is out the whole track is
            // already slow; when it is not, each open sector caps the cars
            // inside it and cancels their overtakes.
            if !sc_out && !self.yellows.is_empty() {
                for i in 0..n {
                    if snapshot[i].4 {
                        continue;
                    }
                    let in_zone = {
                        let s_i = self.state.cars[i].s;
                        self.yellows.iter().any(|(_t, ys)| {
                            // A sector can wrap the start/finish: distance
                            // to it is min(forward, backward) around the lap.
                            let d = (s_i - *ys).rem_euclid(len);
                            d.min(len - d) < YELLOW_ZONE_M
                        })
                    };
                    if in_zone {
                        caps[i] = caps[i].min(YELLOW_SPEED);
                        offsets[i] = None;
                    }
                }
            }

            // BLUE FLAGS: about to be lapped by the leader -- lift and move
            // off the leader's line until the pass is done.
            if !sc_out && rc_blue {
                let leader = (0..n)
                    .filter(|i| !snapshot[*i].3 && !snapshot[*i].4)
                    .max_by(|a, b| {
                        snapshot[*a]
                            .0
                            .partial_cmp(&snapshot[*b].0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                if let Some(l) = leader {
                    let leader_laps = {
                        let e = &self.state.cars[l];
                        e.laps_crossings
                    };
                    for i in 0..n {
                        if i == l || snapshot[i].3 || snapshot[i].4 {
                            continue;
                        }
                        let e = &self.state.cars[i];
                        let lapped = e.laps_crossings < leader_laps;
                        let close = snapshot[l].0 - snapshot[i].0;
                        if lapped && close > 0.0 && close < BLUE_WINDOW_M {
                            caps[i] = caps[i].min((snapshot[l].2 * 0.88).max(8.0));
                            let away = if snapshot[i].1 >= snapshot[l].1 { 3.0 } else { -3.0 };
                            offsets[i] = Some(away);
                            self.blue_flags.push(self.state.cars[i].id);
                        }
                    }
                }
            }
        }

        // DRS: within a zone, on a lap (not lap 0), within a second of the
        // car ahead, SC not out. Pure from the snapshot, so both paths agree.
        self.drs_active = vec![false; n];
        if rc_on
            && rc_drs
            && !sc_out
            && !self.drs_zones.is_empty()
        {
            for i in 0..n {
                if snapshot[i].3 || snapshot[i].4 {
                    continue;
                }
                let e = &self.state.cars[i];
                if e.laps_crossings == 0 {
                    continue; // formation/first lap: no racing yet
                }
                let in_zone = self.drs_zones.iter().any(|(from, zlen)| {
                    let d = (e.s - from).rem_euclid(len);
                    d < *zlen
                });
                if !in_zone || snapshot[i].2 < 15.0 {
                    continue;
                }
                // Nearest car ahead within the line window and 40 m: gap time.
                let mut gap_s = f64::INFINITY;
                for j in 0..n {
                    if j == i || snapshot[j].4 {
                        continue;
                    }
                    if (snapshot[j].1 - snapshot[i].1).abs() > 3.2 {
                        continue;
                    }
                    let gap = snapshot[j].0 - snapshot[i].0;
                    if gap > 0.0 && gap < 40.0 {
                        gap_s = gap.min(gap_s);
                    }
                }
                if gap_s.is_finite() && gap_s / snapshot[i].2.max(5.0) <= 1.0 {
                    self.drs_active[i] = true;
                }
            }
        }

        // Apply caps/offsets smoothly (first-order move toward targets).
        for (i, off) in offsets.iter().enumerate() {
            if let Some(off) = *off {
                let cur = self.state.cars[i].line_offset;
                let step = 2.0 * 0.02; // 2 m/s lateral target shift at 50 Hz
                self.state.cars[i].line_offset = cur + (off - cur).clamp(-step, step);
            }
        }
        self.caps_hint = caps;
    }

    /// Impulse contact pass over all pairs (one authoritative owner).
    fn contact_pass(&mut self) {
        let n = self.state.cars.len();
        let mut events: Vec<ContactEvent> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                // Pit-lane cars are on a parallel road: a pair with exactly
                // one car in the lane never touches. Pairs BOTH in the lane
                // do resolve — the box queue must not interpenetrate.
                if self.state.cars[i].pit_phase.in_pit()
                    != self.state.cars[j].pit_phase.in_pit()
                {
                    continue;
                }
                let (xi, yi, ui, vi, hi) = {
                    let c = &self.state.cars[i].car;
                    (c.pos_x, c.pos_y, c.vel_long, c.vel_lat, c.heading)
                };
                let (xj, yj, uj, vj, hj) = {
                    let c = &self.state.cars[j].car;
                    (c.pos_x, c.pos_y, c.vel_long, c.vel_lat, c.heading)
                };
                let dx = xj - xi;
                let dy = yj - yi;
                let d2 = dx * dx + dy * dy;
                if d2 >= CONTACT_DIST * CONTACT_DIST {
                    continue;
                }
                let d = d2.sqrt().max(1e-6);
                let nx = dx / d;
                let ny = dy / d;
                // World-frame velocities from body components.
                let (ihx, ihy) = (hi.cos(), hi.sin());
                let (jhx, jhy) = (hj.cos(), hj.sin());
                let vix = ui * ihx - vi * ihy;
                let viy = ui * ihy + vi * ihx;
                let vjx = uj * jhx - vj * jhy;
                // `vj * jhx`, not `uj * jhx`. Car j's lateral component was
                // dropped and its longitudinal component used twice, which
                // inflates this y component by up to a factor of sqrt(2) and
                // can report a "closing speed" larger than the car's own
                // speed. Phase 11 reads it as fault evidence, where a wrong
                // sign or magnitude silently converts an ambiguous contact
                // into a confident penalty, so it is fixed here rather than
                // worked around downstream. Phase 4's contact calibration and
                // the hashes that pinned it move with it; see PROGRESS.md.
                let vjy = uj * jhy + vj * jhx;
                let closing = (vjx - vix) * nx + (vjy - viy) * ny; // < 0 when closing
                if closing < 0.0 {
                    let cs = -closing;
                    // One impulse per contact episode: a real hit (cs above
                    // 1 m/s) is resolved once, then the pair cools down —
                    // resting overlap afterwards only separates positionally.
                    let key = i * n + j;
                    let episode = cs > 1.0
                        && self.state.tick.saturating_sub(self.pair_cooldown[key])
                            >= IMPULSE_COOLDOWN_TICKS;
                    // Per-car closing components, from the PRE-impulse
                    // velocities — the impulse rewrites them a few lines below,
                    // and the stewarding aid (Phase 11) reads these as its fault
                    // evidence, so they are computed once here and recorded with
                    // the event whether or not the contact is heavy enough to
                    // retire anyone.
                    // j sits at +n from i, so j's velocity *into* i is the
                    // -n component. Getting this sign wrong silently swaps
                    // which car is blamed, which is exactly the kind of error
                    // a stewarding aid must not be able to make.
                    let a_into_b = vix * nx + viy * ny;
                    let b_into_a = -(vjx * nx + vjy * ny);
                    if episode {
                        // Inelastic split of the closing speed along the
                        // normal: SLOW the chaser (i, behind along -n) and
                        // PUSH the leader (j) forward. Signs matter — adding
                        // +n·cs/2 to i instead would speed the chaser into
                        // the leader and pump energy every overlapping tick.
                        let dvx = nx * cs * 0.5;
                        let dvy = ny * cs * 0.5;
                        let new_vix = vix - dvx;
                        let viy2 = viy - dvy;
                        let new_vjx = vjx + dvx;
                        let vjy2 = vjy + dvy;
                        let set_vel =
                            |e: &mut CarEntry, hvx: f64, hvy: f64, wx: f64, wy: f64| {
                                // project world velocity back into the body frame
                                let f = wx * hvx + wy * hvy;
                                let l = -wx * hvy + wy * hvx;
                                e.car.vel_long = f.max(0.0);
                                e.car.vel_lat = l;
                            };
                        // A retired car is a stopped hazard: it stays put and
                        // deflects whoever runs into it, but it does not take an
                        // impulse. Writing `vel_long` on one launched it —
                        // `set_vel` projects the pair's world velocity into the
                        // car's body frame, so a stationary car sitting in
                        // somebody's path picked up that car's forward speed the
                        // moment the geometry lined up. Phase 5 caught it: a car
                        // retired at t≈8.5 s sat interpenetrating the next car
                        // for 17,968 consecutive ticks, and on the final tick
                        // the forward projection flipped sign and handed it
                        // 11.5 m/s — a retired car accelerating from a dead stop,
                        // 48 s after the race was decided. The `f.max(0.0)`
                        // clamp hid it for 99.9% of that run by pinning the car
                        // at zero, which is exactly why it looked like the
                        // hazard-stop logic had held. The live car takes the
                        // full split, so it still gets slowed and pushed.
                        if !self.state.cars[i].retired {
                            set_vel(&mut self.state.cars[i], ihx, ihy, new_vix, viy2);
                            self.state.cars[i].car.vel_long *= 0.96;
                        }
                        if !self.state.cars[j].retired {
                            set_vel(&mut self.state.cars[j], jhx, jhy, new_vjx, vjy2);
                            self.state.cars[j].car.vel_long *= 0.96;
                        }
                        self.pair_cooldown[key] = self.state.tick;
                    }
                    // Positional separation (always — resting pairs must not
                    // interpenetrate).
                    let push = (CONTACT_DIST - d) / 2.0 + 0.01;
                    self.state.cars[i].car.pos_x -= nx * push;
                    self.state.cars[i].car.pos_y -= ny * push;
                    self.state.cars[j].car.pos_x += nx * push;
                    self.state.cars[j].car.pos_y += ny * push;
                    if episode {
                        // Heavy contact retires the chasing car: the one
                        // whose velocity points INTO the other more strongly.
                        let mut retired = None;
                        if cs > self.config.contact_retire_cs {
                            let victim = if a_into_b > b_into_a { i } else { j } as u32;
                            self.state.cars[victim as usize].retired = true;
                            self.state.cars[victim as usize].dnf = Some("contact");
                            retired = Some(victim);
                            let (tick, s) = (self.state.tick, self.state.cars[victim as usize].s);
                            self.hazards.push((tick, s));
                            self.yellows.push((tick, s));
                        }
                        // Realism: lighter touches still hurt. Damage is
                        // once-per-episode (the same gate as the impulse),
                        // proportional to how hard the touch was, capped —
                        // and it bleeds grip through tire_condition until a
                        // stop resets it.
                        if let Some(rc) = self.realism {
                            if rc.contact_damage {
                                let add =
                                    (cs / self.config.contact_retire_cs.max(1.0)).clamp(0.05, 0.6);
                                self.state.cars[i].damage =
                                    (self.state.cars[i].damage + add).min(1.0);
                                self.state.cars[j].damage =
                                    (self.state.cars[j].damage + add).min(1.0);
                            }
                        }
                        self.state.cars[i].last_contact_tick = self.state.tick;
                        self.state.cars[j].last_contact_tick = self.state.tick;
                        let s_mid = self.state.cars[i].s;
                        events.push(ContactEvent {
                            tick: self.state.tick,
                            a: i as u32,
                            b: j as u32,
                            closing_speed: cs,
                            closing_into_b: a_into_b,
                            closing_into_a: b_into_a,
                            s: s_mid,
                            retired,
                        });
                    }
                }
            }
        }
        self.state.contacts.extend(events);
    }

    // ---- Per-car per-tick helpers shared by the live and replay paths ----
    // (The replay gate demands bit-identity: any control override or state
    // advance here must be a pure function of reconstructible state, applied
    // on the same cadence in both loops.)

    /// Realism pack, per car per physics tick: mechanical failures, lockup
    /// flatspots, loss-of-control crashes. Everything here is a pure
    /// function of reconstructible state (the seeded rolls included), so
    /// both execution paths retire the same cars on the same ticks with no
    /// new logging.
    fn realism_step(&mut self, i: usize) {
        let Some(rc) = self.realism else { return };
        let mode_ok = self.config.mode == SessionMode::Race;
        // Copy the state this pass needs up front: the burn/warm/flatspot
        // accruals want a mutable borrow, so nothing may hold the immutable
        // one past this block.
        let (retired, lap, vel_long, vel_lat) = {
            let e = &self.state.cars[i];
            (
                e.retired,
                e.laps_crossings,
                e.car.vel_long,
                e.car.vel_lat,
            )
        };
        if retired {
            return;
        }
        // Mechanical: checked once per car per lap, a seeded pure roll.
        if mode_ok
            && !self.config.legacy_v2
            && rc.mechanical_rate > 0.0
            && lap > 0
            && self.mech_checked[i] != lap
        {
            self.mech_checked[i] = lap;
            let p = (rc.mechanical_rate / self.config.laps as f64).clamp(0.0, 1.0);
            let mut roll = crate::rng::Lcg::new(
                self.config.seed ^ 0xF411u64.wrapping_mul(i as u64 + 1) ^ (lap as u64) << 20,
            );
            if roll.next_f64() < p {
                let part = if roll.next_f64() < 0.7 { "engine" } else { "gearbox" };
                self.retire(i, DnfCause::Mechanical, part);
                return;
            }
        }
        // A car retired above (or an earlier tick) takes no further part in
        // this pass.
        if self.state.cars[i].retired {
            return;
        }
        // Flat-spot: a mistake while braking hard locks a front wheel.
        if rc.lockups && self.controls[i].brake > 0.8 && vel_long > 25.0 && self.error_active(i)
        {
            let e = &mut self.state.cars[i];
            e.flatspot = (e.flatspot + DT * 0.12).min(1.0);
        }
        // W3 fuel burn + tyre warm-up: pure functions of reconstructible
        // state, mirroring accrue_wear's per-tick integration.
        let warm_bonus =
            (self.controls[i].brake * TYRE_WARM_BRAKE_BONUS + vel_lat.abs() * 0.8) * DT;
        {
            let e = &mut self.state.cars[i];
            if rc.fuel && e.fuel_mass > 0.0 {
                e.fuel_mass = (e.fuel_mass - vel_long * DT * FUEL_BURN_PER_M).max(0.0);
            }
            if rc.tyre_temp {
                // Cold rubber warms with distance; braking and sliding add
                // extra heat. Retired cars cool toward ambient.
                e.tyre_temp = (e.tyre_temp + DT * TYRE_WARM_RATE_PER_S + warm_bonus).min(1.0);
            }
        }
        // Order matters: judge the state as it is, THEN kick — a violent kick
        // retires on a later tick, not the same one.
        self.realism_crash_check(i);
        self.realism_kick(i);
    }

    /// Retire car i, recording why. Idempotent; announces on race control.
    fn retire(&mut self, i: usize, cause: DnfCause, label: &'static str) {
        let e = &mut self.state.cars[i];
        if e.retired {
            return;
        }
        e.retired = true;
        e.dnf = Some(cause.label());
        let car = e.id;
        // A retirement on the track is a race-control event: it can deploy
        // the safety car and opens a local yellow sector at this position.
        let (tick, s) = (self.state.tick, e.s);
        self.hazards.push((tick, s));
        self.yellows.push((tick, s));
        self.state.rc_events.push(RcEvent::CarOut { car, cause: cause.label(), detail: label });
    }

    /// Is a driver mistake active on car i right now?
    fn error_active(&self, i: usize) -> bool {
        let (ds, _dt) = self.error_offsets(i);
        ds != 0.0
    }

    /// Loss-of-control retirement: spun (facing or sliding at speed) or
    /// beached (well past the edge, in the gravel or the wall).
    fn realism_crash_check(&mut self, i: usize) {
        let Some(rc) = self.realism else { return };
        if !rc.spin_crashes || self.config.mode != SessionMode::Race {
            return;
        }
        let e = &self.state.cars[i];
        if e.retired {
            return;
        }
        let speed = (e.car.vel_long * e.car.vel_long + e.car.vel_lat * e.car.vel_lat).sqrt();
        if speed < 10.0 {
            return;
        }
        let (_x, _y, h_t) = self.track.point_at(e.s);
        let mut dh = e.car.heading - h_t;
        while dh > std::f64::consts::PI {
            dh -= 2.0 * std::f64::consts::PI;
        }
        while dh < -std::f64::consts::PI {
            dh += 2.0 * std::f64::consts::PI;
        }
        let sideways = e.car.vel_lat.abs() > 0.55 * e.car.vel_long.max(1.0);
        let spun = dh.abs() > 1.9 || (sideways && speed > 12.0);
        let beached = e.lateral.abs() > self.track.half_width + 6.0;
        if spun {
            self.retire(i, DnfCause::Spun, "crashed");
        } else if beached {
            self.retire(i, DnfCause::Beached, "beached");
        }
    }

    /// The spin kick: a mistake that lands while the car is already using
    /// most of its grip steps the rear out. The physics does the rest — some
    /// kicks are caught, big ones spin. Fires once per mistake, at onset.
    fn realism_kick(&mut self, i: usize) {
        let Some(rc) = self.realism else { return };
        if !rc.spin_crashes || self.config.mode != SessionMode::Race {
            return;
        }
        let Some(cfg) = &self.error_cfg else { return };
        let e = &self.state.cars[i];
        if e.retired || e.car.vel_long < 15.0 {
            return;
        }
        let (ds, _dt) = self.error_offsets(i);
        if ds == 0.0 {
            return; // no mistake active
        }
        // Onset: the mistake was NOT active one driver tick ago. The window
        // ramps in a triangle, so the first non-zero offset is a few ticks
        // into it; comparing against the phase one driver tick back makes
        // the kick fire exactly once without carrying state.
        let len = self.track.length().max(1.0);
        let prev_s = e.s - e.car.vel_long * DRIVER_EVERY as f64 * DT;
        let prev_phase = (prev_s.rem_euclid(len) / len).clamp(0.0, 1.0);
        let window = (cfg.duration_s * e.car.vel_long / len).clamp(0.005, 0.25);
        let mut prev = 0.0;
        for k in 0..weather::DriverErrorConfig::MAX_PER_LAP {
            let (pds, _) = cfg.offset(e.id, e.laps_crossings, k, prev_phase, window);
            prev += pds;
        }
        if prev != 0.0 {
            return; // an ongoing mistake: the driver already knows about it
        }
        // Near the limit? The drivers CRUISE at ~94% of the grip circle
        // (speed_margin 0.97 => lateral 0.94), so the gate must sit ABOVE
        // normal operation: below 0.90 every mistake anywhere would fire,
        // and half the grid would spin per race. At 0.90 a dry-track mistake
        // wobbles the car (mostly caught), while anything that RAISES the
        // ratio past cruise — a wet track, damage, wear — makes kicks bigger
        // and spins common exactly when they should be.
        let a_lat_eff = 11.0 * self.grip_scale(i);
        let a_lat_now = (e.car.vel_long * e.car.yaw_rate).abs();
        let ratio = a_lat_now / a_lat_eff.max(1.0);
        if ratio < 0.90 {
            return;
        }
        let severity = ((ratio - 0.90) / 0.10).clamp(0.2, 1.0);
        let sign = ds.signum();
        let e = &mut self.state.cars[i];
        let speed_f = (e.car.vel_long / 45.0).clamp(0.4, 1.0);
        e.car.yaw_rate += sign * 0.6 * severity * speed_f;
        e.car.vel_lat -= sign * 1.5 * severity;
    }

    /// Controls for car i this tick: retired law, box lock, else the held
    /// driver controls (live: from the driver, replay: from the log).
    fn control_for(&self, i: usize) -> Controls {
        let e = &self.state.cars[i];
        if e.retired {
            retired_controls(e)
        } else if matches!(e.pit_phase, PitPhase::Servicing { .. }) {
            locked_controls() // stationary box service
        } else {
            self.controls[i]
        }
    }

    /// Physics params for car i: the tire's effective grip scales the
    /// friction coefficient (compound × wear), and in Teams mode the team's
    /// machinery multiplies engine, top speed, brakes and tyre grip. Legacy
    /// v2 replays run the Phase 4 tire exactly (base grip, no scaling).
    /// How much of a fresh tyre's grip car `i` still has: compound, wear,
    /// contact damage and flat-spotting, in ONE place. `params_for` and
    /// `grip_scale` both multiply by this — the driver must aim at exactly
    /// the grip the physics grants, and a dented or flat-spotted car is
    /// slower for its driver, not just on paper.
    fn tire_condition(&self, i: usize) -> f64 {
        let e = &self.state.cars[i];
        if self.config.legacy_v2 {
            return 1.0;
        }
        let mut c = e.compound.grip() * (1.0 - WEAR_GRIP_LOSS * e.wear.min(1.0));
        c *= 1.0 - DAMAGE_GRIP_LOSS * e.damage.min(1.0);
        c *= 1.0 - FLATSPOT_GRIP_LOSS * e.flatspot.min(1.0);
        // W3: cold rubber. temp 0 = stone cold (full cold penalty), 1 =
        // warmed in. Accrues with driving distance; slip and braking warm
        // faster (see realism_step).
        if self.realism.map(|r| r.tyre_temp).unwrap_or(false) {
            c *= 1.0 - COLD_TYRE_GRIP_LOSS * (1.0 - e.tyre_temp.min(1.0));
        }
        c
    }

    fn params_for(&self, i: usize) -> CarParams {
        let mut p = self.car_params;
        p.tire_grip *= self.tire_condition(i);
        // W3 fuel: the car weighs what's in the tank. physics::step divides
        // acceleration by mass, so a heavy car is slower out of corners and
        // under braking; inertia scales too (the one place mass doesn't
        // reach), keeping turn-in response honest.
        if self.realism.map(|r| r.fuel).unwrap_or(false) {
            let extra = self.state.cars[i].fuel_mass;
            if extra > 0.0 {
                p.mass += extra;
                let ratio = p.mass / self.car_params.mass;
                p.inertia_z *= ratio;
            }
        }
        // Phase 8: surface grip is a separate multiplier ON TOP of the tire,
        // not folded into it — so "wet" and "worn" stay independently
        // readable in a report, and a re-dry leaves the wear untouched.
        p.tire_grip *= self.surface_grip(i);
        // Team machinery (Teams mode; all 1.0 and free in Classic). Tyres go
        // through the SAME multiplier `grip_scale` shows the driver, so the
        // envelope never aims at grip the car does not have. Top speed maps
        // to drag over its cube: power-limited vmax goes as (P/cdA)^(1/3),
        // so cdA/s³ raises vmax by exactly s.
        let perf = &self.car_perf[i];
        if perf.engine != 1.0 {
            p.engine_force_max *= perf.engine;
            p.engine_power_max *= perf.engine;
        }
        if perf.top_speed != 1.0 {
            p.drag_area /= perf.top_speed * perf.top_speed * perf.top_speed;
        }
        if perf.brakes != 1.0 {
            p.brake_force_max *= perf.brakes;
        }
        if perf.tyres != 1.0 {
            p.tire_grip *= perf.tyres;
        }
        // DRS: the flap opens, drag drops, top speed rises ~4% (vmax ~
        // (P/cdA)^(1/3): 0.8x cdA => 1.16x vmax if power-limited — the
        // driver's envelope raise is the binding effect most of the time).
        if self.drs_active[i] {
            p.drag_area *= 0.8;
        }
        p
    }

    /// Driver envelope scale for car i (same tire law the physics uses, so
    /// the driver aims at what the tires can actually hold).
    /// Public accessor for the grip a car currently has, so a learned policy
    /// can be told the same number the local driver scales itself by. Without
    /// it the policy is being asked to reproduce a control law whose own input
    /// it cannot see.
    pub fn grip_scale_of(&self, i: usize) -> f64 {
        self.grip_scale(i)
    }

    /// The full physics parameter set car `i` races with right now — the
    /// public read-only view of `params_for`. Team machinery, tyre compound,
    /// wear and surface all fold in here; gates use it to prove the physics
    /// and the driver envelope agree.
    pub fn params_of(&self, i: usize) -> CarParams {
        self.params_for(i)
    }

    /// The speed car `i`'s local driver is currently aiming for, m/s.
    ///
    /// Read-only on purpose: it borrows a clone of the driver, applies the grip
    /// scale to that, and reads the target off it. Building an observation
    /// must not be able to perturb the very driver it is describing — calling
    /// `set_grip_scale` on the real one would, and calling `drive` would also
    /// advance its steering rate limiter and noise generator.
    pub fn v_target_of(&self, i: usize) -> f64 {
        let mut d = self.drivers[i].clone();
        d.set_grip_scale(self.grip_scale(i));
        let input = DriverInput {
            s: self.state.cars[i].s,
            lateral: self.state.cars[i].line_offset,
            v_cap: self.caps_hint[i] * self.ai_pace_scale(i),
        };
        d.v_target(&self.state.cars[i].car, &self.track, input)
    }

    fn grip_scale(&self, i: usize) -> f64 {
        // The driver must be slowed by exactly what the surface (and now the
        // panel damage and the flat spot) takes away, or a wet car arrives at
        // every corner at dry pace and simply slides — the failure would look
        // like a physics bug instead of a condition. Teams mode: the team's
        // tyre quality rides the same channel, so a softer car and its driver
        // agree on what the tyres can hold.
        self.tire_condition(i) * self.surface_grip(i) * self.car_perf[i].tyres
    }

    // -----------------------------------------------------------------------
    // Phase 8: weather, surface zones, and queued overrides
    // -----------------------------------------------------------------------

    /// Surface grip multiplier for car i right now: the zone it is standing
    /// on, the live water/temperature sliders, the race-director grip trim,
    /// and its own speed (hydroplaning bites above a threshold, not below).
    /// Pure, so replay derives the identical value from logged controls.
    fn surface_grip(&self, i: usize) -> f64 {
        match (&self.weather, &self.weather_cfg) {
            (Some(rt), Some(cfg)) => {
                let e = &self.state.cars[i];
                let zone = weather::surface_at(&cfg.zones, e.s, self.track.length());
                weather::surface_grip(
                    zone,
                    rt.water,
                    rt.track_temp_c,
                    rt.grip_scale,
                    e.car.vel_long,
                )
            }
            _ => 1.0,
        }
    }

    /// The zone car i is currently on (dry when weather is off).
    pub fn surface_of(&self, i: usize) -> weather::Surface {
        match &self.weather_cfg {
            Some(cfg) => weather::surface_at(&cfg.zones, self.state.cars[i].s, self.track.length()),
            None => weather::Surface::Dry,
        }
    }

    /// Queue a race-director slider change for a future tick and return the
    /// tick it will take effect on.
    ///
    /// This is the whole of the "do not inject UI values directly into
    /// unsynchronized physics state" rule: the value joins the same queue the
    /// command-line `--overrides` schedule uses, so it lands on a known tick
    /// with a recorded effective tick, and a replay from that record
    /// reproduces the race. It is also added to the immutable half of the
    /// config, which is what the run log and results sheet read — a slider
    /// moved live is a logged override, not an invisible one.
    pub fn queue_override(&mut self, slider: weather::Slider, value: f64, at_tick: u64) -> Result<u64, String> {
        let rt = self
            .weather
            .as_mut()
            .ok_or_else(|| "this session has no weather; the director panel is not live".to_string())?;
        // A command for a tick already gone would land "now", which is the
        // unquantised injection the plan rules out. Put it just ahead instead.
        let tick = at_tick.max(self.state.tick + 1);
        rt.push(Override::new(tick, slider, value));
        if let Some(cfg) = self.weather_cfg.as_mut() {
            cfg.schedule.push(Override::new(tick, slider, value));
            cfg.schedule.sort_by_key(|o| (o.tick, o.slider.label()));
        }
        Ok(tick)
    }

    /// The weather config as it now stands, including any override queued since
    /// the race was built. This is what a run log must be written from: a
    /// slider moved live is only replayable if the log records the tick it
    /// landed on, and the caller's copy of the config predates the move.
    pub fn weather_config(&self) -> Option<&WeatherConfig> {
        self.weather_cfg.as_ref()
    }

    /// The director panel's current view of the sliders: live values at this
    /// tick, plus what is still queued. `None` when the session has no weather.
    pub fn weather_state(&self) -> Option<WeatherState> {
        let rt = self.weather.as_ref()?;
        let cfg = self.weather_cfg.as_ref()?;
        Some(WeatherState {
            // The live water level, not the one the session was built with.
            label: cfg.label_at(rt.water),
            water: rt.water,
            track_temp_c: rt.track_temp_c,
            grip_scale: rt.grip_scale,
            queued: rt.remaining(),
            applied: self.weather_applied.len() as u64,
        })
    }

    /// Apply every override whose tick has arrived. Called at the top of the
    /// driver-rate block so a queued command lands on a known simulation
    /// tick — never mid-physics-step, and never from the UI thread.
    ///
    /// The tick it advances to is `tick + 1`, the number the counter will read
    /// once this call returns: the increment happens at the end of `tick()`.
    /// Reading the counter as it stands would advertise a tick on which the new
    /// value is not yet in effect — and a director reading a snapshot at that
    /// tick would conclude the water was 0.9 when the physics still had 0.0.
    fn advance_weather(&mut self) {
        let tick = self.state.tick + 1;
        let fired = match self.weather.as_mut() {
            Some(rt) => rt.advance_to(tick),
            None => return,
        };
        for a in fired {
            self.state.rc_events.push(RcEvent::WeatherApplied {
                tick: a.tick,
                slider: a.slider.label(),
                value: a.value,
            });
            self.weather_applied.push(a);
        }
    }

    /// Driver-error control offset for car i this tick: `(steer, throttle)`.
    /// Sits in the control layer only — physics never sees a mistake, it just
    /// sees slightly wrong inputs, which is what keeps a scenario explainable.
    /// Pure in (car, lap, position), so a replay reproduces every mistake.
    fn error_offsets(&self, i: usize) -> (f64, f64) {
        self.driver_error_offset(i)
    }

    fn driver_error_offset(&self, i: usize) -> (f64, f64) {
        // The mistake generator is the error config: the weather block's when
        // it set one, else the realism pack's own (R1). Same timing law
        // either way, so a replay reproduces every mistake from the controls.
        let cfg = match &self.error_cfg {
            Some(de) if !de.is_off() => de,
            _ => return (0.0, 0.0),
        };
        let e = &self.state.cars[i];
        if e.retired {
            return (0.0, 0.0); // the retired law already owns the car
        }
        let lap = e.laps_crossings;
        // Where the car is through the lap, and how much of the lap one
        // mistake occupies (its duration expressed in track distance at this
        // car's current speed — the mistake must be felt for a fixed *time*,
        // not a fixed fraction of the lap).
        let phase = (e.s / self.track.length().max(1.0)).clamp(0.0, 1.0);
        let window = (cfg.duration_s * e.car.vel_long / self.track.length().max(1.0)).clamp(0.005, 0.25);
        let mut steer = 0.0;
        let mut throttle = 0.0;
        for k in 0..weather::DriverErrorConfig::MAX_PER_LAP {
            let (ds, dt) = cfg.offset(e.id, lap, k, phase, window);
            steer += ds;
            throttle += dt;
        }
        (steer, throttle)
    }

    /// Distance-based tire wear for car i after one physics step.
    fn accrue_wear(&mut self, i: usize) {
        if self.config.legacy_v2 {
            return; // Phase 4 had no tire state
        }
        let e = &mut self.state.cars[i];
        if e.retired {
            return;
        }
        let per_m = 1.0 / (e.compound.life_laps() * self.track.length());
        e.wear = (e.wear + e.car.vel_long * DT * per_m).min(1.25);
    }

    /// Box service countdown for car i (runs per tick; completion fits the
    /// fresh compound and resets wear).
    fn service_countdown(&mut self, i: usize) {
        let mut done = false;
        {
            let e = &mut self.state.cars[i];
            if let PitPhase::Servicing { remaining_s } = &mut e.pit_phase {
                *remaining_s -= DT;
                if *remaining_s <= 0.0 {
                    e.pit_phase = PitPhase::InLane;
                    if let Some(c) = e.switch_to {
                        e.compound = c;
                    }
                    e.wear = 0.0;
                    // The realism channels are crew-fixable too: fresh tyres
                    // are round, the bent suspension is rebuilt, the tank is
                    // full again — and the NEW rubber is stone cold.
                    e.damage = 0.0;
                    e.flatspot = 0.0;
                    if self.realism.map(|r| r.fuel).unwrap_or(false) {
                        e.fuel_mass = FUEL_CAPACITY_KG;
                    }
                    e.tyre_temp = 0.0;
                    e.stops_done += 1;
                    done = true;
                }
            }
        }
        if done {
            let car = self.state.cars[i].id;
            let held_s = self.config.pit.service_time_s;
            self.state.rc_events.push(RcEvent::PitServiceDone { car, held_s });
        }
    }

    // -----------------------------------------------------------------------
    // Phase 7: strategy advisor (car 0 only)
    //
    // Deliberately NOT part of the replay path. Every function below is
    // called from `tick()` only; `tick_replay()` never asks, never reads a
    // reply, and never touches `ai`. That is what keeps an AI run honestly
    // non-reproducible instead of silently diverging from its own log.
    // -----------------------------------------------------------------------

    /// Pace multiplier applied to the advised car's driver speed cap.
    /// 1.0 is exactly the Phase 5 fallback.
    fn ai_pace_scale(&self, i: usize) -> f64 {
        match &self.ai {
            Some(ai) if i == 0 => ai.pace_scale,
            _ => 1.0,
        }
    }

    /// Compact race summary for the advised car — the prompt payload. Only
    /// what a strategist would actually look at, and small enough that a
    /// 0.8B model is not swamped by it.
    fn ai_summary_json(&self) -> String {
        let e = &self.state.cars[0];
        let leader_progress = self
            .state
            .cars
            .iter()
            .map(|c| c.total_progress)
            .fold(f64::NEG_INFINITY, f64::max);
        let gap_m = leader_progress - e.total_progress;
        // Convert the progress deficit to seconds at the car's own speed,
        // with a floor so a car crawling in traffic doesn't report a
        // thousand-second gap to the leader.
        let gap_s = if e.total_progress > 0.0 {
            Some(gap_m / e.car.vel_long.max(15.0))
        } else {
            None
        };
        let pos = self
            .state
            .leaderboard
            .iter()
            .position(|r| r.id == e.id)
            .map(|p| p + 1)
            .unwrap_or(0);
        let mut j = serde_json::Map::new();
        j.insert("lap".into(), serde_json::json!(e.laps_crossings));
        j.insert("laps_total".into(), serde_json::json!(self.config.laps));
        j.insert("laps_left".into(), serde_json::json!(self.config.laps.saturating_sub(e.laps_crossings)));
        j.insert("position".into(), serde_json::json!(pos));
        j.insert("tire".into(), serde_json::json!(e.compound.label()));
        j.insert("wear".into(), serde_json::json!((e.wear * 100.0).round() / 100.0));
        j.insert("stops_done".into(), serde_json::json!(e.stops_done));
        j.insert("last_lap_s".into(), serde_json::json!(e.lap_times.last().map(|t| (t * 100.0).round() / 100.0)));
        j.insert("best_lap_s".into(), serde_json::json!(e.best_lap.map(|t| (t * 100.0).round() / 100.0)));
        j.insert("gap_to_leader_s".into(), serde_json::json!(gap_s.map(|g| (g * 10.0).round() / 10.0)));
        serde_json::Value::Object(j).to_string()
    }

    /// Ask the advisor, at most once per `every_laps` laps of the advised
    /// car and only when nothing is already in flight. Fire-and-forget:
    /// the sim thread hands the request over and continues driving.
    fn ai_maybe_ask(&mut self) {
        let Some(ai) = self.ai.as_ref() else { return };
        let cfg_every = ai.cfg.every_laps;
        if cfg_every == 0 || ai.inflight.is_some() {
            return;
        }
        let e = &self.state.cars[0];
        let lap = e.laps_crossings;
        let askable = !e.retired
            && e.finish_time_s.is_none()
            && !e.pit_phase.in_pit()
            && lap >= 1
            && lap < self.config.laps // a pit on the final lap is never legal
            && lap.is_multiple_of(cfg_every)
            && ai.last_ask_lap != Some(lap);
        if !askable {
            return;
        }
        let summary = self.ai_summary_json();
        let tick = self.state.tick;
        let ai = self.ai.as_mut().unwrap();
        let id = ai.next_id;
        ai.next_id += 1;
        ai.last_ask_lap = Some(lap);
        ai.inflight = Some(Inflight { id, tick, lap, summary: summary.clone() });
        let _ = ai
            .handle
            .req_tx
            .send(strategy::AdviceRequest { id, tick, summary });
        // A max-speed run has no wall-clock budget to protect, so it waits
        // here — once per `every_laps` laps, never per tick — instead of
        // racing a model that needs seconds while the sim covers 250 s of
        // race time in under a second. The wait is charged to the advisor,
        // not to the physics deadline (see `ai_block_this_tick_s`).
        if self.blocking_advisor {
            let wait = std::time::Duration::from_millis(ai.cfg.timeout_ms + 500);
            let started = Instant::now();
            let got = ai.handle.rep_rx.recv_timeout(wait).ok();
            let elapsed = started.elapsed().as_secs_f64();
            self.ai_block_this_tick_s += elapsed;
            match got {
                Some(reply) => self.ai_handle_reply(reply),
                // No answer even after the wait: record the miss so a run
                // that silently asked and silently gave up is visible.
                None => {
                    let f = self.ai.as_mut().and_then(|ai| ai.inflight.take());
                    if let Some(f) = f {
                        self.ai_log.push(strategy::AiExchange {
                            request_tick: f.tick,
                            applied_tick: None,
                            prompt_summary: f.summary,
                            model: self.ai.as_ref().map(|a| a.cfg.model.clone()).unwrap_or_default(),
                            response: "<no reply>".into(),
                            latency_ms: elapsed * 1e3,
                            outcome: "timeout",
                            detail: format!(
                            "no reply within {} ms",
                            self.ai.as_ref().map(|a| a.cfg.timeout_ms).unwrap_or(0)
                        ),
                        });
                    }
                }
            }
        }
    }


    /// Record one reply and apply whatever the legality gate allows.
    fn ai_handle_reply(&mut self, reply: strategy::AdviceReply) {
        // Copy out what the borrow checker needs, then drop the borrow
        // before deciding: `ai_decide` mutates the car entries.
        let Some((f, cfg)) = self.ai.as_ref().and_then(|ai| {
            let f = ai.inflight.clone()?;
            if f.id != reply.id {
                return None; // superseded answer
            }
            Some((f, ai.cfg.clone()))
        }) else {
            return;
        };
        let (outcome, detail, applied_tick, pace) =
            self.ai_decide(f.tick, f.lap, &reply.result, &cfg);
        let ai = self.ai.as_mut().unwrap();
        ai.inflight = None;
        if let Some(p) = pace {
            ai.pace_scale = p.clamp(0.95, 1.04);
        }
        if matches!(
            reply.result,
            Err(strategy::AiError::Unavailable(_)) | Err(strategy::AiError::Http(_, _))
        ) {
            ai.transport_errors += 1;
        }
        self.ai_log.push(strategy::AiExchange {
            request_tick: f.tick,
            applied_tick,
            prompt_summary: f.summary,
            model: cfg.model.clone(),
            response: match &reply.result {
                Ok(s) => s.clone(),
                Err(e) => format!("<{e:?}>"),
            },
            latency_ms: reply.latency_ms,
            outcome,
            detail,
        });
    }

    /// Collect every reply that has arrived and decide what, if anything, to
    /// do with it. Called at the driver rate; never blocks (try_recv only).
    fn ai_drain(&mut self) {
        if self.ai.is_none() {
            return;
        }
        while let Some(reply) = self
            .ai
            .as_ref()
            .and_then(|ai| ai.handle.rep_rx.try_recv().ok())
        {
            self.ai_handle_reply(reply);
        }
    }

    /// The legality gate. Returns (outcome, detail, applied_tick, pace_scale).
    ///
    /// Nothing here trusts the model: an answer only ever *narrow* what the
    /// deterministic strategy may do, and every rejection is recorded with
    /// the reason so the benchmark can explain a lost time.
    fn ai_decide(
        &mut self,
        _req_tick: u64,
        ask_lap: u32,
        result: &Result<String, strategy::AiError>,
        cfg: &strategy::AdvisorConfig,
    ) -> (&'static str, String, Option<u64>, Option<f64>) {
        let content = match result {
            Err(strategy::AiError::Timeout) => return ("timeout", "no answer in time".into(), None, None),
            Err(strategy::AiError::Unavailable(e)) => return ("error", format!("endpoint unavailable: {e}"), None, None),
            Err(strategy::AiError::Http(code, body)) => {
                return ("error", format!("HTTP {code}: {body}"), None, None)
            }
            Err(strategy::AiError::BadResponse(e)) => return ("bad_answer", e.clone(), None, None),
            Ok(s) => s,
        };
        let advice = match strategy::parse_advice(content) {
            Ok(a) => a,
            Err(e) => return ("bad_answer", e, None, None),
        };
        if advice.pace.is_none() && advice.pit.is_none() && advice.tire.is_none() {
            return ("bad_answer", "no usable field in answer".into(), None, None);
        }

        // Stale: the question's premise is gone. The car must still be
        // running, still be on track, and must not have run so far past the
        // lap it was asked on that the answer describes a different race.
        let e = &self.state.cars[0];
        let laps_since = e.laps_crossings.saturating_sub(ask_lap);
        if e.retired {
            return ("stale", "car retired before the answer arrived".into(), None, None);
        }
        if e.finish_time_s.is_some() || self.state.phase == RacePhase::Finished {
            return ("stale", "race finished before the answer arrived".into(), None, None);
        }
        if e.pit_phase.in_pit() {
            return ("stale", "car already in the pit lane".into(), None, None);
        }
        if laps_since >= cfg.every_laps {
            return (
                "stale",
                format!("answer is {laps_since} laps behind the question"),
                None,
                None,
            );
        }

        // Confidence gate. A field below the floor falls back to the
        // deterministic plan; if every field is below it, nothing is applied.
        let ok = |c: f64| c >= cfg.min_confidence;
        let mut considered = 0usize;
        let mut applied = 0usize;
        let mut notes: Vec<String> = Vec::new();
        let mut pace_out = None;
        let now_lap = e.laps_crossings;
        let on_final_or_past = now_lap >= self.config.laps;

        // --- pace (always legal in a running race session) ---
        if let Some((choice, conf)) = advice.pace {
            if ok(conf) {
                considered += 1;
                pace_out = Some(choice.scale());
                applied += 1;
                notes.push(format!("pace={}", choice.label()));
            } else {
                notes.push(format!("pace={} below floor", choice.label()));
            }
        }

        // --- pit now: only before the final lap, only once, only if the
        //     deterministic plan did not fix the stop lap itself ---
        let mut armed_stop = false;
        if let Some((want, conf)) = advice.pit {
            if !ok(conf) {
                notes.push("pit below floor".into());
            } else {
                considered += 1;
                if !want {
                    // "Stay" agrees with the deterministic plan: nothing to
                    // do, and nothing claimed for it.
                    notes.push("pit: stay".into());
                } else if on_final_or_past {
                    notes.push("pit rejected: final lap".into());
                } else if !self.config.pit.enabled {
                    notes.push("pit rejected: pits disabled".into());
                } else if self.state.cars[0].stops_done > 0 {
                    notes.push("pit rejected: stop already taken".into());
                } else if matches!(self.config.strategy, PitStrategy::Fixed { .. }) {
                    notes.push("pit rejected: fixed strategy owns the stop lap".into());
                } else if self.state.cars[0].stop_lap.is_some_and(|l| l <= now_lap) {
                    notes.push("pit rejected: already armed for this lap".into());
                } else {
                    self.state.cars[0].stop_lap = Some(now_lap);
                    armed_stop = true;
                    applied += 1;
                    notes.push(format!("pit armed for lap {now_lap}"));
                }
            }
        }

        // --- tire for the next stop: only a compound that differs and that
        //     will actually be fitted ---
        if let Some((compound, conf)) = advice.tire {
            if ok(conf) {
                considered += 1;
                let cur = self.state.cars[0].compound;
                if self.state.cars[0].stops_done > 0 {
                    notes.push("tire rejected: stop already taken".into());
                } else if matches!(self.config.strategy, PitStrategy::Fixed { .. }) {
                    // A Fixed strategy owns the compound as well as the lap:
                    // it is one explicit human decision, not two knobs.
                    notes.push("tire rejected: fixed strategy owns the compound".into());
                } else if compound == Some(cur) {
                    notes.push(format!("tire rejected: already on {}", cur.label()));
                } else if !armed_stop && self.state.cars[0].stop_lap.is_none() {
                    notes.push("tire rejected: no stop planned to fit it".into());
                } else {
                    self.state.cars[0].switch_to = compound;
                    applied += 1;
                    notes.push(match compound {
                        Some(c) => format!("tire={}", c.label()),
                        None => "tire=keep".into(),
                    });
                }
            } else {
                notes.push("tire below floor".into());
            }
        }

        if considered == 0 {
            return ("low_confidence", notes.join("; "), None, None);
        }
        let outcome = if applied > 0 { "applied" } else { "rejected" };
        let tick = if applied > 0 { Some(self.state.tick) } else { None };
        (outcome, notes.join("; "), tick, pace_out)
    }

    /// Trajectory hash over every car, every tick — includes tire and pit
    /// state so replay covers the whole Phase 5 state machine. Legacy v2
    /// replays reproduce the exact Phase 4 mix instead (those logs recorded
    /// hashes without tire/pit bytes).
    fn mix_state(&mut self) {
        let legacy = self.config.legacy_v2;
        let mut h = self.state_hash;
        h = fnv_mix(h, &self.state.tick.to_le_bytes());
        for e in &self.state.cars {
            let c = &e.car;
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
            h = fnv_mix(h, &[e.retired as u8]);
            if !legacy {
                h = fnv_mix(h, &[e.compound as u8]);
                h = fnv_mix(h, &e.wear.to_bits().to_le_bytes());
                h = fnv_mix(h, &[match e.pit_phase {
                    PitPhase::OnTrack => 0u8,
                    PitPhase::InLane => 1,
                    PitPhase::Servicing { .. } => 2,
                }]);
                // Realism pack: dents and flat spots change the physics, so
                // they are hash bytes — but only when the pack is on, so a
                // pack-off race hashes exactly like the pre-pack engine.
                if self.realism.is_some() {
                    h = fnv_mix(h, &e.damage.to_bits().to_le_bytes());
                    h = fnv_mix(h, &e.flatspot.to_bits().to_le_bytes());
                    h = fnv_mix(h, &e.fuel_mass.to_bits().to_le_bytes());
                    h = fnv_mix(h, &e.tyre_temp.to_bits().to_le_bytes());
                }
            }
        }
        self.state_hash = h;
    }

    /// One physics tick for the whole field (live drivers).
    pub fn tick(&mut self) {
        if self.state.phase == RacePhase::Ready {
            return;
        }
        // Overrides advance during formation too: a director moving a slider
        // on the grid must see it take effect on the tick they queued it, not
        // silently deferred to green.
        self.advance_weather();
        if self.state.phase == RacePhase::FormationLap {
            self.tick_formation_lap(false);
            return;
        }
        if self.state.phase == RacePhase::Formation {
            self.tick_formation();
            return;
        }
        let t0 = Instant::now();
        self.ai_block_this_tick_s = 0.0;
        // Overrides land on the tick they were queued for, before anything
        // reads a slider this tick.
        self.advance_weather();
        if self.state.tick.is_multiple_of(DRIVER_EVERY as u64) {
            self.recompute_progress();
            // Advisor sits between progress and traffic: a reply may arm a
            // stop, and the awareness pass must see that on the same tick.
            self.ai_drain();
            self.ai_maybe_ask();
            self.traffic_awareness();
            self.update_leaderboard();
            // Driver decisions in id order (deterministic); borrows resolved
            // by collecting into locals first.
            let mut new_controls = Vec::with_capacity(self.state.cars.len());
            let mut new_records = Vec::new();
            for i in 0..self.state.cars.len() {
                if self.state.cars[i].retired {
                    // Retired car brakes to a stop, steering off the line.
                    new_controls.push(retired_controls(&self.state.cars[i]));
                } else if i == 0 && self.learned_controls.is_some() {
                    // Phase 12: car 0 under an external (learned) policy. The
                    // local driver's grip scaling and rate limiter are
                    // deliberately NOT applied: the policy is the controller,
                    // and running its state through a second controller would
                    // quietly change what the policy asked for. The controls
                    // are recorded exactly as used, so a learned-driver race
                    // still replays bit-identically from its log.
                    let ctrl = self.learned_controls.unwrap();
                    new_records.push((0u32, self.state.tick, ctrl));
                    new_controls.push(ctrl);
                } else {
                    // The driver aims with the tires it has (compound+wear).
                    let scale = self.grip_scale(i);
                    self.drivers[i].set_grip_scale(scale);
                    let input = DriverInput {
                        s: self.state.cars[i].s,
                        lateral: self.state.cars[i].line_offset,
                        // Traffic cap from awareness, then the advisor's pace
                        // call on top (1.0 = no change, so car 0 without
                        // advice and every other car are untouched).
                        v_cap: self.caps_hint[i] * self.ai_pace_scale(i),
                    };
                    let car = self.state.cars[i].car;
                    let mut ctrl = self.drivers[i].drive(&car, &self.track, input);
                    // Phase 8: a seeded mistake is a control-layer perturbation
                    // only. It is baked into the recorded controls here, so a
                    // replay reproduces it from the log without needing the
                    // weather block to be re-derived at all.
                    if self.weather_cfg.is_some() {
                        let (ds, dt) = self.driver_error_offset(i);
                        if ds != 0.0 || dt != 0.0 {
                            ctrl.steer_cmd += ds;
                            ctrl.throttle = (ctrl.throttle + dt).clamp(0.0, 1.0);
                        }
                    }
                    new_records.push((i as u32, self.state.tick, ctrl));
                    new_controls.push(ctrl);
                }
            }
            self.controls = new_controls;
            self.recorded.extend(new_records);
            // Tally this lap's mistake set exactly once, on the first driver
            // tick after the lap number advances. The errors themselves are a
            // pure function of (seed, car, lap), so this is a count, not a
            // sampling artefact.
            if let Some(de) = &self.error_cfg {
                if !de.is_off() {
                    let lap = self.state.cars[0].laps_crossings;
                    if self.weather_error_lap != lap {
                        self.weather_error_lap = lap;
                        let per_car: Vec<u32> = (0..self.state.cars.len())
                            .map(|i| de.errors_on_lap(self.state.cars[i].id, lap))
                            .collect();
                        for (i, n) in per_car.iter().enumerate() {
                            self.weather_errors_by_car[i] += n;
                        }
                    }
                }
            }
        }
        for i in 0..self.state.cars.len() {
            let ctrl = self.control_for(i);
            let params = self.params_for(i);
            // Sample car 0's surface grip at the driver rate (the same cadence
            // the physics loop reads it on) — a per-tick sum would weight fast
            // parts of the lap 250× a slow corner and report a mean that never
            // occurred at any real point on track.
            if self.weather.is_some() && self.state.tick.is_multiple_of(DRIVER_EVERY as u64) {
                self.weather_grip_sum += self.surface_grip(0);
                self.weather_grip_samples += 1;
            }
            self.state.cars[i].car =
                physics::step(&self.state.cars[i].car, &ctrl, &params, DT);
            self.accrue_wear(i);
            self.service_countdown(i);
            self.realism_step(i);
        }
        self.contact_pass();
        self.state.tick += 1;
        self.state.sim_time = self.state.tick as f64 * DT;
        self.mix_state();
        // Phase 9 observers run after the hash has folded the tick, so they
        // provably cannot be part of what it recorded.
        self.observe();

        // Waiting on the model is not physics. Subtract it before judging
        // the deadline, otherwise a 5 s advisor stall would be reported as
        // one catastrophic tick rather than as advisor latency.
        let blocked_us = self.ai_block_this_tick_s * 1e6;
        self.ai_block_this_tick_s = 0.0;
        let cost = (t0.elapsed().as_secs_f64() * 1e6 - blocked_us).max(0.0);
        if cost > DT * 1e6 {
            self.deadline_misses += 1;
        }
        self.tick_costs_us.push(cost);

        // Race over when the leader completes the configured laps AND every
        // other active car has either finished or the leader crossed +30 s
        // grace elapsed — simplified: race finishes when the LAST active car
        // finishes, capped when the leader's finish + 60 s passes.
        let all_finished_or_retired = self.state.cars.iter().all(|e| e.retired || e.finish_time_s.is_some());
        let leader_done = self
            .state
            .cars
            .iter()
            .filter(|e| !e.retired)
            .map(|e| e.finish_time_s.unwrap_or(f64::INFINITY))
            .fold(f64::INFINITY, f64::min);
        if all_finished_or_retired
            || leader_done < self.state.sim_time - 60.0
            || self.state.sim_time > self.config.laps as f64 * 60.0 + 120.0
        {
            // Third clause is the red-flag cap: a blocked corner can stall
            // the leader forever; classify on progress instead of hanging.
            self.state.phase = RacePhase::Finished;
            self.update_leaderboard();
        }
    }

    /// Phase 9 — everything the world outside the simulation can observe,
    /// derived from state that already exists.
    ///
    /// Called once per tick after `mix_state`, so it can neither delay nor
    /// influence the physics step it follows. Two hard rules live here:
    ///
    /// * **It never blocks.** Telemetry hands buffers to a sender thread over
    ///   a bounded channel and drops on back-pressure; cue extraction is a
    ///   bounded scan of the field with no allocation once warmed up.
    /// * **It never feeds back.** Nothing written here is reachable from
    ///   `mix_state`, so the trajectory hash of a telemetry-enabled race is
    ///   the trajectory hash of the same race with no receiver attached.
    fn observe(&mut self) {
        // Phase 11 runs first and on its own sample buffer: an incident review
        // needs position, lateral offset and speed, and none of the telemetry
        // sample machinery. Keeping it separate means a stewarded race that
        // publishes nothing does not pay for a telemetry sample set per tick.
        if self.steward.is_some() {
            self.observe_steward();
        }
        if self.sound.is_none() && self.telemetry.is_none() {
            return; // the common case: two None checks, no work at all
        }
        let tick = self.state.tick;
        let sim_t = self.state.sim_time;

        // Longitudinal acceleration needs the previous tick's speed. Kept
        // here, outside `RaceState`, precisely so it cannot reach the hash.
        let vel: Vec<f64> = self.state.cars.iter().map(|e| e.car.vel_long).collect();
        let accel: Vec<f64> = match self.telemetry.as_mut() {
            Some(link) => link.measure_accel(&vel, DT),
            None => vec![0.0; vel.len()],
        };
        let max_steer = self.car_params.max_steer;

        let samples: Vec<CarSample> = self
            .state
            .cars
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let ctrl = self.control_for(i);
                CarSample::from_car(&e.car, &ctrl, max_steer, accel[i], &self.gearbox())
            })
            .collect();

        // The runtime is briefly taken out of `self` so the race-event scan,
        // which reads `self.state`, can run against it. Moving an
        // `Option<SoundRuntime>` moves three Vecs' pointers — no copy of the
        // cue log, and no reallocation.
        if let Some(mut rt) = self.sound.take() {
            if rt.cue_car_count() != samples.len() {
                rt.prime(&samples);
            }
            // Race events first: a lap chime that lands on the same tick as a
            // shift should not be reordered behind it.
            self.drain_race_events_into_sound(&mut rt, tick);
            rt.extract(tick, sim_t, &samples);
            self.sound = Some(rt);
        }

        // Publishing runs on its own schedule, not the physics rate: at 30 Hz
        // that is every 9th tick at 250 Hz, and the count is taken from the
        // configured rate rather than assumed.
        if tick.is_multiple_of(self.telemetry_interval) {
            if let Some(link) = self.telemetry.as_mut() {
                link.publish(sim_t, &samples);
            }
        }
    }

    fn gearbox(&self) -> telemetry::Gearbox {
        self.telemetry_cfg
            .as_ref()
            .map(|t| t.gearbox)
            .unwrap_or_default()
    }

    /// Phase 11 — one tick of incident recording. Reads the field into the
    /// recorder's own buffer and hands it over with the contact list. The
    /// recorder decides what is an incident; this function only samples.
    ///
    /// `under_steward` is `Running` and not yet finished. The formation lap is a
    /// locked grid, not a race: a stationary start would otherwise be scored as
    /// twenty track-limit excursions on every single race.
    fn observe_steward(&mut self) {
        self.steward_samples.clear();
        for e in &self.state.cars {
            self.steward_samples.push(steward::CarSample {
                id: e.id,
                x: e.car.pos_x,
                y: e.car.pos_y,
                s: e.s,
                lateral: e.lateral,
                speed_mps: e.car.vel_long,
                retired: e.retired,
                in_pit: e.pit_phase.in_pit(),
            });
        }
        let under_steward =
            self.state.phase == RacePhase::Running && self.state.phase != RacePhase::Finished;
        let tick = self.state.tick;
        let sim_t = self.state.sim_time;
        if let Some(rec) = self.steward.as_mut() {
            rec.observe(tick, sim_t, under_steward, &self.steward_samples, &self.state.contacts);
        }
    }

    /// The incident record for this race, closing any case still inside its
    /// review window. Called once, from `outcome`.
    pub fn steward_incidents(&mut self) -> Vec<steward::Incident> {
        match self.steward.as_mut() {
            Some(rec) => rec.finish(),
            None => Vec::new(),
        }
    }

    pub fn steward_config(&self) -> Option<&steward::StewardConfig> {
        self.steward.as_ref().map(|r| r.config())
    }

    /// Turn race-control events, laps and contacts that happened since the
    /// last tick into cues. Reads only what the race already recorded.
    fn drain_race_events_into_sound(&mut self, rt: &mut SoundRuntime, tick: u64) {
        while self.sound_event_cursor < self.state.rc_events.len() {
            let ev = self.state.rc_events[self.sound_event_cursor];
            let cue: Option<(SoundCue, f64, u32)> = match &ev {
                RcEvent::CarOut { .. } | RcEvent::SafetyCar { .. } => None,
                RcEvent::Lights(n) => Some((SoundCue::StartLight, f64::from(*n), 0)),
                RcEvent::Green => Some((SoundCue::GreenFlag, 0.0, 0)),
                RcEvent::PitEntered { car, lap } => {
                    Some((SoundCue::PitEntered, f64::from(*lap), *car))
                }
                RcEvent::WeatherApplied { value, .. } => {
                    Some((SoundCue::WeatherChange, *value, 0))
                }
                // Fastest lap is already covered by the per-car lap cue below,
                // which fires for every car rather than only the session best.
                // Pit service and exit are silent: the entry chime has already
                // told the story, and a second one mid-stop is noise.
                RcEvent::PitServiceDone { .. }
                | RcEvent::PitExited { .. }
                | RcEvent::FastestLap { .. } => None,
            };
            if let Some((c, v, car)) = cue {
                rt.race_event(tick, car, c, v);
            }
            self.sound_event_cursor += 1;
        }

        for i in 0..self.state.cars.len() {
            let laps = self.state.cars[i].laps_crossings;
            if laps != self.sound_last_laps[i] {
                self.sound_last_laps[i] = laps;
                // The first crossing starts lap 1's timing; a cue belongs to
                // the lap that has just been *completed*, so laps >= 2.
                if laps >= 2 && self.state.cars[i].finish_time_s.is_none() {
                    let t = self.state.cars[i].lap_times.last().copied().unwrap_or(0.0);
                    rt.lap_completed(tick, self.state.cars[i].id, laps - 1, t);
                }
            }
        }

        while self.sound_contacts_seen < self.state.contacts.len() {
            let c = &self.state.contacts[self.sound_contacts_seen];
            rt.race_event(tick, c.a, SoundCue::Contact, c.closing_speed);
            self.sound_contacts_seen += 1;
        }
    }
    /// Phase 9 accessors for the server, the CLI and the gates.
    pub fn sound_runtime(&self) -> Option<&SoundRuntime> {
        self.sound.as_ref()
    }
    pub fn telemetry_stats(&self) -> Option<telemetry::LinkStats> {
        self.telemetry.as_ref().map(|l| l.stats())
    }
    pub fn telemetry_reachable(&self) -> bool {
        self.telemetry.as_ref().is_some_and(|l| l.receiver_reachable())
    }
    pub fn telemetry_error(&self) -> Option<&str> {
        self.telemetry_error.as_deref()
    }
    pub fn telemetry_target(&self) -> Option<&str> {
        self.telemetry_cfg.as_ref().map(|t| t.target.as_str())
    }
    pub fn telemetry_hz(&self) -> u32 {
        self.telemetry_cfg.as_ref().map_or(0, |t| t.hz)
    }

    /// One formation tick: the field stands on the brakes while the start
    /// lights count up (one per second to five, plus the seeded hold), then
    /// green. No drivers, no awareness, no contacts — a locked grid.
    /// Rolling formation lap: a real running tick (drivers active, mistakes
    /// live, telemetry live) with the field held at formation pace and
    /// overtaking suppressed. When the leader has covered one lap distance,
    /// the field snaps back onto its grid boxes and the standing lights
    /// sequence runs. The snap is a pure function of the transition state,
    /// so both execution paths re-form identically.
    fn tick_formation_lap(&mut self, replay: bool) {
        let t0 = Instant::now();
        // Driver-rate block (same shape as the live tick, no AI: the model
        // has nothing to add to a parade lap).
        if self.state.tick.is_multiple_of(DRIVER_EVERY as u64) {
            self.recompute_progress();
            self.traffic_awareness();
            self.update_leaderboard();
            // Driver controls: recorded (the replay consumes them). Under
            // replay the held controls were copied in by the caller, so only
            // the RECORDING is skipped — physics, contact and the tick
            // advance below run identically in both paths.
            if !replay {
                // The same per-car control law as the live loop (retired law,
                // learned policy for car 0, else the driver) — mistakes
                // included, all recorded for the replay.
                let mut new_records = Vec::with_capacity(self.state.cars.len());
                let mut new_controls = Vec::with_capacity(self.state.cars.len());
                for i in 0..self.state.cars.len() {
                    if self.state.cars[i].retired {
                        new_controls.push(retired_controls(&self.state.cars[i]));
                    } else if i == 0 && self.learned_controls.is_some() {
                        let ctrl = self.learned_controls.unwrap();
                        new_records.push((0u32, self.state.tick, ctrl));
                        new_controls.push(ctrl);
                    } else {
                        let scale = self.grip_scale(i);
                        self.drivers[i].set_grip_scale(scale);
                        let input = DriverInput {
                            s: self.state.cars[i].s,
                            lateral: self.state.cars[i].line_offset,
                            v_cap: self.caps_hint[i],
                        };
                        let car = self.state.cars[i].car;
                        let mut ctrl = self.drivers[i].drive(&car, &self.track, input);
                        if self.error_cfg.is_some() {
                            let (ds, dt) = self.driver_error_offset(i);
                            if ds != 0.0 || dt != 0.0 {
                                ctrl.steer_cmd += ds;
                                ctrl.throttle = (ctrl.throttle + dt).clamp(0.0, 1.0);
                            }
                        }
                        new_records.push((i as u32, self.state.tick, ctrl));
                        new_controls.push(ctrl);
                    }
                }
                self.controls = new_controls;
                self.recorded.extend(new_records);
            }
        }
        // Physics loop (identical cadence to the live tick).
        for i in 0..self.state.cars.len() {
            let ctrl = self.control_for(i);
            let params = self.params_for(i);
            self.state.cars[i].car = physics::step(&self.state.cars[i].car, &ctrl, &params, DT);
            self.accrue_wear(i);
            self.service_countdown(i);
            self.realism_step(i);
        }
        self.contact_pass();
        self.state.tick += 1;
        self.state.sim_time = self.state.tick as f64 * DT;
        // Transition: leader has covered a lap's distance (or twice, if the
        // field got held up) — snap onto the grid and hold for the lights.
        let leader_progress = self
            .state
            .cars
            .iter()
            .map(|e| e.total_progress)
            .fold(f64::NEG_INFINITY, f64::max);
        let covered = leader_progress - self.formation_start_progress;
        let len = self.track.length();
        if covered >= len || covered >= 2.0 * len {
            self.snap_to_grid();
            self.state.phase = RacePhase::Formation;
        }
        self.mix_state();
        self.observe();
        let cost = t0.elapsed().as_secs_f64() * 1e6;
        self.tick_costs_us.push(cost);
    }

    /// Pure grid re-form: every live car back to its box, stopped; timing
    /// state reset so the standing start's first crossing opens lap 1.
    fn snap_to_grid(&mut self) {
        let len = self.track.length();
        for (i, e) in self.state.cars.iter_mut().enumerate() {
            if e.retired {
                continue; // wrecks stay where they stopped
            }
            let slot = self.grid_slots[i];
            let row = (slot / 2) as f64;
            let col = slot % 2;
            let s = len - 15.0 - row * self.config.grid_row_gap_m;
            let lateral = if col == 0 { -2.2 } else { 2.2 };
            let (x, y, heading) = self.track.point_at(s);
            let (lx, ly) = (-heading.sin(), heading.cos());
            e.car = CarState {
                pos_x: x + lx * lateral,
                pos_y: y + ly * lateral,
                heading,
                ..Default::default()
            };
            e.s = s;
            e.lateral = lateral;
            e.line_offset = 0.0;
            e.laps_crossings = 0;
            e.lap_times.clear();
            e.lap_start_t = 0.0;
            e.total_progress = -(len - s);
            e.finish_time_s = None;
        }
    }

    fn tick_formation(&mut self) {
        let t0 = Instant::now();
        let locked = locked_controls();
        let params = self.car_params;
        for e in &mut self.state.cars {
            e.car = physics::step(&e.car, &locked, &params, DT);
        }
        self.state.tick += 1;
        self.state.sim_time = self.state.tick as f64 * DT;
        let n = (self.state.sim_time.floor() as u64).min(5) as u8;
        if n != self.state.lights {
            self.state.lights = n;
            self.state.rc_events.push(RcEvent::Lights(n));
        }
        if self.state.tick >= self.state.green_tick {
            self.state.phase = RacePhase::Running;
            self.state.lights = 0; // the sequence is over; the HUD goes green
            self.state.rc_events.push(RcEvent::Green);
        }
        self.mix_state();
        self.observe();
        let cost = t0.elapsed().as_secs_f64() * 1e6;
        self.tick_costs_us.push(cost);
    }

    fn update_leaderboard(&mut self) {
        // Race: finished cars by finish time — INCLUDING cars retired after
        // taking the flag (a cool-down crash doesn't demote a finisher);
        // only retirement BEFORE the flag is a DNF, classified last. Timed
        // sessions (qualifying / practice): best lap ascending — a retired
        // car with a time keeps its place, no-time cars sink naturally.
        let timed = self.config.mode != SessionMode::Race;
        let mut order: Vec<usize> = (0..self.state.cars.len()).collect();
        order.sort_by(|&a, &b| {
            let ea = &self.state.cars[a];
            let eb = &self.state.cars[b];
            let sa = if timed {
                (false, ea.best_lap.unwrap_or(f64::INFINITY), -ea.total_progress)
            } else {
                (
                    ea.retired && ea.finish_time_s.is_none(),
                    ea.finish_time_s.unwrap_or(f64::INFINITY),
                    -ea.total_progress,
                )
            };
            let sb = if timed {
                (false, eb.best_lap.unwrap_or(f64::INFINITY), -eb.total_progress)
            } else {
                (
                    eb.retired && eb.finish_time_s.is_none(),
                    eb.finish_time_s.unwrap_or(f64::INFINITY),
                    -eb.total_progress,
                )
            };
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let leader_progress = order
            .first()
            .map(|&i| self.state.cars[i].total_progress)
            .unwrap_or(0.0);
        let leader_v = order
            .first()
            .map(|&i| self.state.cars[i].car.vel_long)
            .unwrap_or(0.0);
        let leader_best = order
            .first()
            .and_then(|&i| self.state.cars[i].best_lap);
        self.state.leaderboard = order
            .iter()
            .enumerate()
            .map(|(pos, &i)| {
                let e = &self.state.cars[i];
                let dnf = e.retired && e.finish_time_s.is_none();
                let gap_s = if pos == 0 || dnf {
                    None
                } else if timed {
                    // Time gap on the best-lap classification.
                    e.best_lap.map(|b| b - leader_best.unwrap_or(b))
                } else {
                    let dl = leader_progress - e.total_progress;
                    Some(dl / leader_v.max(5.0))
                };
                LeaderRow {
                    id: e.id,
                    pos: pos as u32 + 1,
                    // Capped at the race distance: finished cars circulate
                    // through the grace window without counting extra laps.
                    laps_done: e.laps_crossings.saturating_sub(1).min(self.config.laps),
                    gap_s,
                    retired: e.retired,
                    finished: e.finish_time_s.is_some(),
                    v: e.car.vel_long,
                    best_lap_s: e.best_lap,
                    compound: e.compound.label(),
                    stops: e.stops_done,
                    in_pit: e.pit_phase.in_pit(),
                    // W3: fuel and tyre temp as percentages, only with the
                    // pack on.
                    fuel: self.realism.filter(|r| r.fuel).map(|_| {
                        ((e.fuel_mass / FUEL_CAPACITY_KG).clamp(0.0, 1.0) * 100.0)
                            .round() as u8
                    }),
                    tyre_temp: self.realism.filter(|r| r.tyre_temp).map(|_| {
                        (e.tyre_temp.clamp(0.0, 1.0) * 100.0).round() as u8
                    }),
                    dnf: e.dnf,
                }
            })
            .collect();
    }

    pub fn state_hash(&self) -> u64 {
        self.state_hash
    }

    pub fn deadline_misses(&self) -> u64 {
        self.deadline_misses
    }

    pub fn tick_summary(&mut self) -> Summary {
        summarize_us(std::mem::take(&mut self.tick_costs_us))
    }

    pub fn take_recorded(&mut self) -> Vec<(u32, u64, Controls)> {
        std::mem::take(&mut self.recorded)
    }
}

// ---------------------------------------------------------------------------
// Runner, outcome, v2 log, replay
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct RaceOutcome {
    pub cars: u32,
    pub laps: u32,
    pub ticks: u64,
    pub sim_time_s: f64,
    pub wall_time_s: f64,
    pub state_hash: u64,
    pub tick_stats: Summary,
    pub deadline_misses: u64,
    pub classification: Vec<LeaderRow>,
    pub contacts: Vec<ContactEvent>,
    pub leader_lap_times: Vec<f64>,
    pub retired_ids: Vec<u32>,
    /// Realism pack: why each retired car is out, when known.
    pub dnf_causes: Vec<(u32, &'static str)>,
    /// Phase 5 results: session mode, fastest lap, and the full stop log.
    pub mode: &'static str,
    pub fastest: Option<(u32, f64, u32)>,
    pub pit_stops: Vec<PitStopRecord>,
    /// Phase 7: `Some` only when the advisor ran. A baseline result has
    /// `None` here, which is how the two runs are told apart.
    pub ai: Option<AiSummary>,
    /// Car 0's own result — the car the advisor is allowed to touch. Always
    /// present, so a baseline run and an advised run are read identically.
    pub advised: Option<AdvisedCar>,
    /// Phase 8 weather report. `None` = the run had no weather at all (the
    /// Phase 5/7 engine), so a dry baseline and a wet run are told apart
    /// exactly the way an advised and a baseline run are.
    pub weather: Option<WeatherSummary>,
    /// Phase 9 sound report. `None` = no cue log was recorded, so the Phase 8
    /// result reads identically to before.
    pub sound: Option<SoundSummary>,
    /// The sound config that produced `sound`, so a replay rebuilds the same
    /// extractor and the log can round-trip it.
    pub sound_config: Option<SoundConfig>,
    /// Every cue the run produced, in order. Kept whole rather than summarised
    /// because the gate compares the replayed log against it tick for tick.
    pub sound_events: Option<Vec<sound::CueEvent>>,
    /// Phase 9 UDP link counters at the end of the run, `None` when no link
    /// was opened.
    pub telemetry: Option<telemetry::LinkStats>,
    pub telemetry_target: Option<String>,
    pub telemetry_hz: u32,
    /// Why the link could not be opened, if one was asked for and failed.
    pub telemetry_error: Option<String>,
    /// The cue-log hash the log says the original run produced. `Some` only
    /// when this outcome came from a replay, which is what makes the replayed
    /// `sound.log_hash` comparable against something.
    pub sound_expected_hash: Option<u64>,
    /// Phase 11: the incident record, when the aid was enabled. `None` for a
    /// race that did not ask for it — which is every race by default, and every
    /// log written before this phase.
    pub steward_incidents: Option<Vec<steward::Incident>>,
    /// The rule set, thresholds and totals behind `steward_incidents`.
    pub steward: Option<steward::StewardSummary>,
    /// Thresholds the decisions were made with, so a report can state the rule
    /// it applied rather than only its verdict.
    pub steward_config: Option<steward::StewardConfig>,
    /// Digest of the incident record, compared on replay.
    pub steward_expected_hash: Option<u64>,
}

impl RaceOutcome {
    fn sound_events_len(&self) -> usize {
        self.sound_events.as_ref().map_or(0, |v| v.len())
    }
}

/// The director panel's live view of the sliders, as the dashboard sees them.
/// Distinct from [`WeatherSummary`], which is the end-of-race report: this one
/// changes every tick and is what the sliders are drawn from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeatherState {
    /// The session's scenario name, for the panel's headline.
    pub label: &'static str,
    pub water: f64,
    pub track_temp_c: f64,
    pub grip_scale: f64,
    /// Overrides queued but not yet reached — a slider moved while paused.
    pub queued: usize,
    /// Overrides that have taken effect so far this session.
    pub applied: u64,
}

/// What the weather did to this race, in the terms a race engineer would use.
#[derive(Clone, Debug)]
pub struct WeatherSummary {
    pub label: &'static str,
    pub water: f64,
    pub track_temp_c: f64,
    pub grip_scale: f64,
    /// Every override that fired, with the tick it took effect on.
    pub applied: Vec<AppliedOverride>,
    /// Overrides still queued when the flag fell.
    pub unapplied: usize,
    /// Configured mistake rate per car per lap (0 = none).
    pub driver_error_rate: f64,
    /// Mistakes actually delivered, summed over every car and lap.
    pub driver_errors: u32,
    /// Per-car driver-error totals, for the "which car made the mistake" case.
    pub driver_errors_by_car: Vec<(u32, u32)>,
    /// Mean grip multiplier sampled over car 0's race — the headline number
    /// for "how much of the lap was actually wet".
    pub mean_grip_car0: f64,
}

impl RaceOutcome {
    /// Whether the session ran to the flag, as opposed to being cut short by
    /// the cool-down cap. Every outcome, advised or not, must be true.
    pub fn phase_finished(&self) -> bool {
        self.classification.iter().all(|r| r.finished || r.retired)
    }
}

/// What happened to the advised car, for the helps/ties/hurts comparison.
#[derive(Clone, Debug)]
pub struct AdvisedCar {
    pub id: u32,
    pub position: Option<u32>,
    pub finish_time_s: Option<f64>,
    pub total_time_s: Option<f64>,
    pub best_lap_s: Option<f64>,
    pub laps: u32,
    pub retired: bool,
    pub stops: u32,
    pub compounds: Vec<&'static str>,
}

/// Run a full race to completion (realtime pacing optional). Returns the
/// outcome plus the recorded 50 Hz input stream for all cars.
pub fn run_race(cfg: RaceConfig, realtime: bool) -> (RaceOutcome, Vec<(u32, u64, Controls)>) {
    let (o, i, _) = run_race_full(cfg, realtime);
    (o, i)
}

/// As [`run_race`], but also returns the full advisor exchange log. Phase 7
/// needs it; a non-AI run returns an empty vector.
pub fn run_race_full(
    cfg: RaceConfig,
    realtime: bool,
) -> (RaceOutcome, Vec<(u32, u64, Controls)>, Vec<strategy::AiExchange>) {
    let _timer = if realtime { Some(crate::sim::TimerPeriod::acquire()) } else { None };
    let mut race = Race::new(cfg.clone());
    // Realtime protects the 250 Hz budget, so the advisor stays
    // fire-and-forget and late answers are stale-rejected. A max-speed run
    // covers 250 s of race time in about a second, so it waits for the
    // model at each lap boundary — otherwise the answer would always
    // arrive after the flag.
    race.set_blocking_advisor(!realtime);
    race.start();
    let start = Instant::now();
    let mut next_deadline = start;
    while race.state.phase != RacePhase::Finished {
        if realtime {
            next_deadline += std::time::Duration::from_secs_f64(DT);
            let now = Instant::now();
            if now < next_deadline {
                crate::sim::wait_until(next_deadline);
            }
        }
        race.tick();
    }
    // Drain whatever the model answered before the flag — a late reply is
    // still on the record, just recorded as stale by the legality gate.
    race.ai_drain();
    let wall = start.elapsed().as_secs_f64();
    let inputs = race.take_recorded();
    // Order matters: `outcome()` tallies `ai_log`, so the tally is built
    // before the log is handed out.
    let outcome = race.outcome(wall);
    let exchange_log = std::mem::take(&mut race.ai_log);
    (outcome, inputs, exchange_log)
}

impl Race {
    pub fn outcome(&mut self, wall_time_s: f64) -> RaceOutcome {
        let stats = self.tick_summary();
        // Phase 11 first: the incident record must be closed before anything
        // reads it, and closing it is the only step that mutates the recorder.
        // It changes no car state — the simulated result stays exactly what the
        // cars did, and the penalty is reported alongside it, never folded into
        // it.
        let incidents = self.steward_incidents();
        let steward_cfg = self.steward_config().copied();
        let classification = self.state.leaderboard.clone();
        let dnf_causes: Vec<(u32, &'static str)> = self
            .state
            .cars
            .iter()
            .filter(|e| e.retired && e.dnf.is_some())
            .map(|e| (e.id, e.dnf.unwrap()))
            .collect();
        let retired = self
            .state
            .cars
            .iter()
            .filter(|e| e.retired)
            .map(|e| e.id)
            .collect();
        let leader_laps = self
            .state
            .cars
            .iter()
            .max_by(|a, b| a.total_progress.partial_cmp(&b.total_progress).unwrap())
            .map(|e| e.lap_times.clone())
            .unwrap_or_default();
        let mut pit_stops: Vec<PitStopRecord> = self
            .state
            .cars
            .iter()
            .flat_map(|e| e.pit_stops.iter().copied())
            .collect();
        pit_stops.sort_by_key(|r| r.entered_t.to_bits());
        let ai = self.ai.as_ref().map(|rt| {
            let mut lat: Vec<f64> = self.ai_log.iter().map(|x| x.latency_ms).collect();
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f64| -> f64 {
                if lat.is_empty() {
                    0.0
                } else {
                    let i = ((lat.len() as f64 - 1.0) * p).round() as usize;
                    lat[i.min(lat.len() - 1)]
                }
            };
            let count = |o: &str| self.ai_log.iter().filter(|x| x.outcome == o).count();
            AiSummary {
                exchanges: self.ai_log.len(),
                applied: count("applied"),
                rejected: count("rejected"),
                stale: count("stale"),
                low_confidence: count("low_confidence"),
                bad_answer: count("bad_answer"),
                timeout: count("timeout"),
                error: count("error"),
                latency_ms_p50: pct(0.50),
                latency_ms_p99: pct(0.99),
                transport_errors: rt.transport_errors,
                pace_scale_final: rt.pace_scale,
                asks_suppressed: 0,
            }
        });
        // Car 0's own result. Populated for every run, advised or not — the
        // baseline and the advised run must be comparable, and that means
        // reading both the same way. `ai` above is what distinguishes them.
        let advised = match self.state.cars.first() {
            Some(e) => {
            let row = self.state.leaderboard.iter().find(|r| r.id == e.id);
            // Stint history: start compound plus whatever was actually
            // fitted at each completed stop.
            let mut compounds = vec![self.config.start_compound.label()];
            for rec in &e.pit_stops {
                compounds.push(rec.fitted.label());
            }
            Some(AdvisedCar {
                id: e.id,
                position: row.map(|r| r.pos),
                finish_time_s: e.finish_time_s,
                total_time_s: e.finish_time_s,
                best_lap_s: e.best_lap,
                laps: e.laps_crossings,
                retired: e.retired,
                stops: e.stops_done,
                compounds,
            })
            }
            None => None,
        };
        RaceOutcome {
            weather: self.weather_summary(),
            cars: self.config.cars,
            laps: self.config.laps,
            ticks: self.state.tick,
            sim_time_s: self.state.sim_time,
            wall_time_s,
            state_hash: self.state_hash,
            tick_stats: stats,
            deadline_misses: self.deadline_misses,
            classification,
            contacts: self.state.contacts.clone(),
            leader_lap_times: leader_laps,
            retired_ids: retired,
            dnf_causes,
            mode: self.config.mode.label(),
            fastest: self.state.fastest,
            pit_stops,
            ai,
            advised,
            sound: self.sound.as_ref().map(|rt| rt.summary()),
            sound_config: self.sound.as_ref().map(|rt| *rt.config()),
            sound_events: self.sound.as_ref().map(|rt| rt.events.clone()),
            telemetry: self.telemetry_stats(),
            telemetry_target: self.telemetry_target().map(str::to_string),
            telemetry_hz: self.telemetry_hz(),
            telemetry_error: self.telemetry_error().map(str::to_string),
            sound_expected_hash: None,
            steward_incidents: steward_cfg.map(|_| incidents.clone()),
            steward: steward_cfg.map(|_| steward::StewardSummary::of(&incidents)),
            steward_config: steward_cfg,
            steward_expected_hash: None,
        }
    }

    /// Phase 8 results-sheet summary, or `None` when the run had no weather.
    pub fn weather_summary(&self) -> Option<WeatherSummary> {
        let cfg = self.weather_cfg.as_ref()?;
        let rt = self.weather.as_ref()?;
        let by_car: Vec<(u32, u32)> = self
            .state
            .cars
            .iter()
            .zip(self.weather_errors_by_car.iter())
            .map(|(e, &n)| (e.id, n))
            .filter(|&(_, n)| n > 0)
            .collect();
        Some(WeatherSummary {
            label: cfg.label(),
            water: rt.water,
            track_temp_c: rt.track_temp_c,
            grip_scale: rt.grip_scale,
            applied: self.weather_applied.clone(),
            unapplied: rt.remaining(),
            driver_error_rate: cfg.driver_errors.rate_per_lap,
            driver_errors: self.weather_errors_by_car.iter().sum(),
            driver_errors_by_car: by_car,
            mean_grip_car0: if self.weather_grip_samples > 0 {
                self.weather_grip_sum / self.weather_grip_samples as f64
            } else {
                1.0
            },
        })
    }

    /// Drive N ticks from the standing start without racing to completion —
    /// the grid-scaling benchmark primitive.
    pub fn bench_ticks(&mut self, ticks: u64) -> Summary {
        self.state.phase = RacePhase::Running;
        for _ in 0..ticks {
            self.tick();
        }
        self.tick_summary()
    }
}

/// v3 run log: same numeric-line style as Phase 1, one input stream per car,
/// plus an M line carrying the session/tire/pit configuration.
pub fn write_race_log(
    path: &std::path::Path,
    cfg: &RaceConfig,
    inputs: &[(u32, u64, Controls)],
    outcome: &RaceOutcome,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        w,
        "# simlog v5 (H=header M=session W=weather T=telemetry/sound D=steward P=incident I=input@50Hz R=result)"
    )?;
    let mut fields: Vec<String> = vec![
        cfg.cars.to_string(),
        cfg.laps.to_string(),
        cfg.seed.to_string(),
        ((DT * 1e6) as u64).to_string(),
        DRIVER_EVERY.to_string(),
        "0".into(), // realtime flag irrelevant for replay
        format!("{:e}", cfg.grid_row_gap_m),
        format!("{:e}", cfg.contact_retire_cs),
        format!("{:e}", cfg.pace_spread),
        format!("{:e}", cfg.noise),
    ];
    let p = CarParams::default();
    fields.extend(
        [
            p.mass, p.inertia_z, p.wheelbase, p.cg_to_front, p.cg_to_rear, p.cg_height,
            p.tire_grip, p.corner_stiffness_front, p.corner_stiffness_rear,
            p.engine_force_max, p.engine_power_max, p.brake_force_max, p.brake_bias_front,
            p.drag_area, p.roll_resistance, p.max_steer, p.air_density,
        ]
        .iter()
        .map(|f| format!("{f:e}")),
    );
    writeln!(w, "H {}", fields.join(" "))?;
    // Phase 10: the circuit. A log written before this line replays on the
    // engine default (an absent `G` line reads as `TrackSpec::default()`),
    // which is exactly the layout those logs were produced on. A custom
    // circuit (version 2) writes its control points as `C` lines right
    // after, so the log rebuilds the same tarmac anywhere.
    match &cfg.track.custom {
        None => writeln!(
            w,
            "G {} {:.17e} {:.17e} {:.17e}",
            cfg.track.version, cfg.track.straight_len, cfg.track.radius, cfg.track.half_width
        )?,
        Some(c) => {
            writeln!(
                w,
                "G {} {:.17e} {:.17e} {:.17e} {}",
                cfg.track.version, 0.0, 0.0, cfg.track.half_width, c.name
            )?;
            // The pit fractions ride the log too (the `B` line — `P` is the
            // steward's): Race::new resolves the lane from them, so a replay
            // without them would rebuild the lane in the wrong place (default
            // fractions instead of this track's). Absent = the reader's own
            // default, which is exactly what a track file without a pit block
            // means.
            if let Some(f) = &c.pit {
                writeln!(w, "B {:e} {:e} {:e}", f.entry, f.box_frac, f.exit)?;
            }
            for p in &c.points {
                writeln!(w, "C {:e} {:e}", p[0], p[1])?;
            }
        }
    }
    // Grid order (Phase E): written only when qualifying set it, so every
    // pre-weekend log stays byte-identical (same convention as the F teams
    // line). Pole first.
    if !cfg.grid_order.is_empty() {
        let ids: Vec<String> = cfg.grid_order.iter().map(|c| c.to_string()).collect();
        writeln!(w, "O {}", ids.join(" "))?;
    }
    // Session configuration (Phase 5): mode, formation, tires, strategy,
    // pit lane. Values the replay path needs to rebuild the same race.
    let (strat, stop_lap, switch) = match cfg.strategy {
        PitStrategy::None => ("none", 0, TireCompound::Medium),
        PitStrategy::Auto => ("auto", 0, TireCompound::Medium),
        PitStrategy::Fixed { lap, switch_to } => ("fixed", lap, switch_to),
    };
    writeln!(
        w,
        "M {} {} {} {} {} {} {} {:e} {:e} {:e} {:e} {:e} {:e}",
        cfg.mode.label(),
        cfg.formation as u8,
        cfg.start_compound.label(),
        strat,
        stop_lap,
        switch.label(),
        cfg.pit.enabled as u8,
        cfg.pit.entry_s,
        cfg.pit.box_s,
        cfg.pit.exit_s,
        cfg.pit.offset_m,
        cfg.pit.speed_limit,
        cfg.pit.service_time_s,
    )?;
    // Field model. Written only when the race used team machinery, so a
    // classic race's log stays byte-identical to the pre-teams format and an
    // absent F line reads as classic — exactly the W-line convention.
    if cfg.field == FieldMode::Teams {
        writeln!(w, "F {}", cfg.field.label())?;
    }
    // Realism pack posture (the `J` line). Written only when the pack ran,
    // so a pack-off log stays byte-identical to the pre-realism format and
    // an absent J reads as the pack off — the F/W convention.
    if let Some(r) = &cfg.realism {
        // EVERY flag is written explicitly. Deriving some from others (as an
        // earlier format did) lets a replayed posture silently differ from
        // the run's: e.g. spin_crashes=off + safety_car=on was read back as
        // race control OFF, and the replay then disagreed with the run the
        // moment race control touched the trajectory (SC cap, DRS gate).
        writeln!(
            w,
            "J {} {:e} {:e} {} {} {} {} {} {} {} {} {}",
            r.spin_crashes as u8,
            r.error_rate,
            r.mechanical_rate,
            r.contact_damage as u8,
            r.lockups as u8,
            r.safety_car as u8,
            r.yellow_zones as u8,
            r.blue_flags as u8,
            r.formation_lap as u8,
            r.drs_zones as u8,
            r.fuel as u8,
            r.tyre_temp as u8,
        )?;
    }
    // Weather (Phase 8). Written only when the run actually had weather, so a
    // dry race still logs v3-shaped lines and replays identically. A v4 reader
    // that sees no W line sets `weather: None`, which is exactly the engine
    // that produced it.
    if let Some(cfgw) = cfg.weather.as_ref().filter(|w| !w.is_inert()) {
        write!(
            w,
            "W {} {} {} {} {} {}",
            format_args!("{:e}", cfgw.water),
            format_args!("{:e}", cfgw.track_temp_c),
            format_args!("{:e}", cfgw.grip_scale),
            cfgw.zones.len(),
            cfgw.driver_errors.rate_per_lap,
            cfgw.preset.as_deref().unwrap_or("-"),
        )?;
        for z in &cfgw.zones {
            write!(w, " {:.17e} {:.17e} {}", z.start, z.end, z.surface.code())?;
        }
        write!(w, " {}", cfgw.schedule.len())?;
        for o in &cfgw.schedule {
            write!(w, " {} {} {:.17e}", o.tick, o.slider.code(), o.value)?;
        }
        writeln!(w)?;
    }
    // Phase 9. Written whenever the run produced a cue log, so a replay can
    // rebuild the same extractor and be checked against the original. The
    // telemetry half is recorded for the report only — a replay never opens a
    // socket, because publishing must never depend on a receiver existing.
    if let Some(snd) = outcome.sound.as_ref() {
        let sc = outcome.sound_config.unwrap_or_default();
        writeln!(
            w,
            "T {} {} {} {:.17e} {:.17e} {:.17e} {} {} {:016x}",
            outcome.telemetry_target.as_deref().unwrap_or("-"),
            outcome.telemetry_hz,
            outcome.telemetry.map_or(0, |t| t.sent),
            sc.slip_threshold_g,
            sc.wind_band_mps,
            sc.wind_start_mps,
            sc.max_cues_per_s,
            outcome.sound_events_len(),
            snd.log_hash,
        )?;
        if let Some(events) = outcome.sound_events.as_ref() {
            for e in events {
                writeln!(
                    w,
                    "E {} {} {} {:.17e}",
                    e.tick,
                    e.car,
                    e.cue.code(),
                    e.value
                )?;
            }
        }
    }
    // Phase 11: the stewarding record. The `D` line carries the rule set, its
    // version and EVERY threshold the decisions rest on, so a finding is
    // reproducible from the file alone — a rule whose numbers live only in the
    // binary is not auditable. Each `P` line is one incident's evidence digest
    // and outcome. A replay rebuilds the aid from the `D` line and re-derives
    // the record; a log with no `D` line is a race that never had the aid, and
    // replays on exactly the engine that wrote it.
    if let (Some(incidents), Some(sc)) =
        (outcome.steward_incidents.as_ref(), outcome.steward_config)
    {
        writeln!(w, "D {}", steward::header_line(incidents, &sc))?;
        for inc in incidents {
            writeln!(w, "P {}", steward::incident_line(inc))?;
        }
    }
    for (car, tick, c) in inputs {
        writeln!(w, "I {car} {tick} {:.17e} {:.17e} {:.17e}", c.throttle, c.brake, c.steer_cmd)?;
    }
    writeln!(
        w,
        "R {} {:016x} {} {}",
        outcome.ticks, outcome.state_hash, outcome.contacts.len(), outcome.retired_ids.len()
    )?;
    w.flush()?;
    Ok(())
}

/// Replay a race log: re-execute every car's recorded inputs and verify
/// the whole-field trajectory hash. v2 logs (no M line) replay as legacy
/// sessions: no formation, no stops — exactly how they were recorded.
pub fn replay_race_log(path: &std::path::Path) -> Result<(bool, RaceOutcome), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let mut grid_order: Vec<u32> = Vec::new();
    let mut cars = 0u32;
    let mut laps = 0u32;
    let mut seed = 0u64;
    let mut grid_gap = 9.0;
    let mut retire_cs = 12.0;
    let mut pace = 0.002;
    let mut noise = 0.002;
    // Legacy v2 defaults (an M line overrides all of these).
    let mut mode = SessionMode::Race;
    let mut formation = false;
    let mut start_compound = TireCompound::Soft;
    let mut strategy = PitStrategy::None;
    let mut pit = PitConfig { enabled: false, ..PitConfig::default() };
    let mut track = TrackSpec::default();
    let mut field = FieldMode::Classic;
    let mut realism: Option<RealismConfig> = None;
    let mut saw_session_line = false;
    let mut weather: Option<WeatherConfig> = None;
    let mut expected_hash = 0u64;
    let mut expected_ticks = 0u64;
    // Phase 9: rebuilt from the T line so a replay re-derives the same cue
    // extractor, then compared against the hash that run recorded.
    let mut sound_cfg: Option<SoundConfig> = None;
    let mut sound_expected_hash: Option<u64> = None;
    let mut sound_events_logged = 0usize;
    // Phase 11: rebuilt from the D line so a replay re-derives the same aid
    // against the same thresholds, then compared against the hash that run
    // recorded. Absent D line = the run never had the aid.
    let mut steward_cfg: Option<steward::StewardConfig> = None;
    let mut steward_expected_hash: Option<u64> = None;
    let mut steward_incidents_logged = 0usize;
    let mut steward_lines: Vec<String> = Vec::new();
    let mut inputs: Vec<(u32, u64, Controls)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tok = line.split_whitespace();
        match tok.next() {
            Some("H") => {
                let vals: Vec<f64> = tok
                    .map(|t| t.parse::<f64>().map_err(|e| format!("H {t}: {e}")))
                    .collect::<Result<_, _>>()?;
                if vals.len() != 10 + 17 {
                    return Err(format!("H: {} fields, expected 27", vals.len()));
                }
                cars = vals[0] as u32;
                laps = vals[1] as u32;
                seed = vals[2] as u64;
                grid_gap = vals[6];
                retire_cs = vals[7];
                pace = vals[8];
                noise = vals[9];
            }
            Some("G") => {
                // v1: version straight radius half_width. v2 adds the custom
                // circuit's name, with its control points on following `C`
                // lines. Absent on pre-Phase-10 logs, which were all produced
                // on the built-in layout.
                let words: Vec<&str> = tok.collect();
                let version: u32 =
                    words[0].parse().map_err(|e| format!("G version: {e}"))?;
                let num = |idx: usize| -> Result<f64, String> {
                    words[idx].parse::<f64>().map_err(|e| format!("G field {idx}: {e}"))
                };
                track = TrackSpec {
                    version,
                    straight_len: num(1)?,
                    radius: num(2)?,
                    half_width: num(3)?,
                    custom: None,
                };
                if version >= 2 {
                    if words.len() != 5 {
                        return Err(format!(
                            "G v{version}: {} fields, expected 5 (half_width and name)",
                            words.len()
                        ));
                    }
                    track.custom = Some(CustomTrack {
                        name: words[4].to_string(),
                        points: Vec::new(), // filled by the C lines
                        pit: None,          // fractions resolve in Race::new
                    });
                }
            }
            Some("B") => {
                // Pit fractions (entry box exit) of the custom circuit named
                // by the G v2 line. The steward's incident records are `P`;
                // the letters must not collide.
                let words: Vec<&str> = tok.collect();
                if words.len() != 3 {
                    return Err(format!("P: {} fields, expected 3", words.len()));
                }
                let num = |i: usize| -> Result<f64, String> {
                    words[i].parse::<f64>().map_err(|e| format!("P field {i}: {e}"))
                };
                match &mut track {
                    TrackSpec { custom: Some(c), .. } => {
                        c.pit = Some(PitFractions {
                            entry: num(0)?,
                            box_frac: num(1)?,
                            exit: num(2)?,
                        });
                    }
                    other => {
                        return Err(format!(
                            "P line without a custom G line (version {})",
                            other.version
                        ))
                    }
                }
            }
            Some("C") => {
                // Control point of the custom circuit declared by the G v2 line.
                let words: Vec<&str> = tok.collect();
                if words.len() != 2 {
                    return Err(format!("C: {} fields, expected 2", words.len()));
                }
                let x: f64 = words[0].parse().map_err(|e| format!("C x: {e}"))?;
                let y: f64 = words[1].parse().map_err(|e| format!("C y: {e}"))?;
                match &mut track {
                    TrackSpec { custom: Some(c), .. } => c.points.push([x, y]),
                    other => {
                        return Err(format!(
                            "C line without a custom G line (version {})",
                            other.version
                        ))
                    }
                }
            }
            Some("O") => {
                // Grid order, pole first (Phase E). Absent on every log
                // written before qualifying existed — those fields ran in
                // identity order, which is exactly what an absent line means.
                grid_order = tok
                    .map(|t| t.parse::<u32>().map_err(|e| format!("O {t}: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            Some("M") => {
                // mode formation compound strategy stop_lap switch pit_enabled entry box exit offset limit service
                let words: Vec<&str> = tok.collect();
                if words.len() != 13 {
                    return Err(format!("M: {} fields, expected 13", words.len()));
                }
                mode = SessionMode::parse(words[0]).ok_or_else(|| format!("M: mode '{}'", words[0]))?;
                formation = words[1] != "0";
                start_compound =
                    TireCompound::parse(words[2]).ok_or_else(|| format!("M: compound '{}'", words[2]))?;
                strategy = match words[3] {
                    "none" => PitStrategy::None,
                    "auto" => PitStrategy::Auto,
                    "fixed" => PitStrategy::Fixed {
                        lap: words[4].parse().map_err(|e| format!("M lap: {e}"))?,
                        switch_to: TireCompound::parse(words[5])
                            .ok_or_else(|| format!("M: switch '{}'", words[5]))?,
                    },
                    other => return Err(format!("M: strategy '{other}'")),
                };
                pit.enabled = words[6] != "0";
                let num = |idx: usize| -> Result<f64, String> {
                    words[idx].parse::<f64>().map_err(|e| format!("M field {idx}: {e}"))
                };
                pit.entry_s = num(7)?;
                pit.box_s = num(8)?;
                pit.exit_s = num(9)?;
                pit.offset_m = num(10)?;
                pit.speed_limit = num(11)?;
                pit.service_time_s = num(12)?;
                saw_session_line = true;
            }
            Some("T") => {
                // target hz sent slip wind_start wind_band max_cues ncues loghash
                // No trailing separator: the reader counts whitespace-separated
                // fields, and a stray token here made every Phase 9 log
                // unreplayable. Caught by tests/phase9_gate.rs.
                let words: Vec<&str> = tok.collect();
                if words.len() != 9 {
                    return Err(format!("T: {} fields, expected 9", words.len()));
                }
                let num = |i: usize| -> Result<f64, String> {
                    words[i].parse::<f64>().map_err(|e| format!("T field {i}: {e}"))
                };
                sound_cfg = Some(SoundConfig {
                    enabled: true,
                    max_cues_per_s: words[6].parse::<u32>().map_err(|e| format!("T max_cues: {e}"))?,
                    slip_threshold_g: num(3)?,
                    wind_band_mps: num(4)?,
                    wind_start_mps: num(5)?,
                });
                sound_expected_hash = Some(
                    u64::from_str_radix(words[8], 16)
                        .map_err(|e| format!("T log hash '{}': {e}", words[8]))?,
                );
            }
            // The recorded cue log. Re-read rather than trusted: the replay
            // re-derives the log and compares hashes, and counting the lines
            // here is what lets a truncated log say so instead of quietly
            // comparing a shorter list.
            Some("E") => {
                let words: Vec<&str> = tok.collect();
                if words.len() != 4 {
                    return Err(format!("E: {} fields, expected 4", words.len()));
                }
                sound_events_logged += 1;
            }
            Some("W") => {
                // water temp grip nzones err_rate preset | start end code ... | nover | tick code value ...
                let words: Vec<&str> = tok.collect();
                let num = |i: usize| -> Result<f64, String> {
                    words[i].parse::<f64>().map_err(|e| format!("W field {i}: {e}"))
                };
                if words.len() < 6 {
                    return Err(format!("W: {} fields, expected at least 6", words.len()));
                }
                let nzones = num(3)? as usize;
                let preset = match words[5] {
                    "-" => None,
                    p => Some(p.to_string()),
                };
                // Zones begin at field 6, after the preset token.
                let zbase = 6usize;
                let mut zones = Vec::with_capacity(nzones);
                for z in 0..nzones {
                    let b = zbase + z * 3;
                    zones.push(weather::SurfaceZone {
                        start: num(b)?,
                        end: num(b + 1)?,
                        surface: weather::Surface::from_code(words[b + 2])
                            .ok_or_else(|| format!("W: surface '{}'", words[b + 2]))?,
                    });
                }
                let base = zbase + nzones * 3;
                let nover = num(base)? as usize;
                let mut schedule = Vec::with_capacity(nover);
                for o in 0..nover {
                    let b = base + 1 + o * 3;
                    schedule.push(Override {
                        tick: words[b].parse().map_err(|e| format!("W tick: {e}"))?,
                        slider: weather::Slider::from_code(words[b + 1])
                            .ok_or_else(|| format!("W: slider '{}'", words[b + 1]))?,
                        value: num(b + 2)?,
                    });
                }
                let mut wc = WeatherConfig {
                    water: num(0)?,
                    track_temp_c: num(1)?,
                    grip_scale: num(2)?,
                    zones,
                    schedule,
                    driver_errors: weather::DriverErrorConfig {
                        rate_per_lap: num(4)?,
                        ..Default::default()
                    },
                    preset,
                    // `live` is a dashboard affordance, not physics: a replay
                    // re-derives grip from the values and the schedule, and a
                    // log whose sliders were left neutral replays as the
                    // no-weather engine that produced the same trajectory.
                    live: false,
                };
                wc.normalise();
                weather = Some(wc);
            }
            Some("J") => {
                // Realism pack posture. 5/7-field lines predate the full
                // posture record; their race-control and formation flags are
                // defaulted ON (the posture those formats implied). 10 fields
                // is the explicit posture — every flag as it ran.
                let words: Vec<&str> = tok.collect();
                if !(words.len() == 5
                    || words.len() == 7
                    || words.len() == 10
                    || words.len() == 12)
                {
                    return Err(format!(
                        "J: {} fields, expected 5, 7, 10 or 12",
                        words.len()
                    ));
                }
                let flag = |i: usize| words.get(i).map(|w| *w != "0").unwrap_or(true);
                realism = Some(RealismConfig {
                    spin_crashes: words[0] != "0",
                    error_rate: words[1].parse().map_err(|e| format!("J rate: {e}"))?,
                    mechanical_rate: words[2].parse().map_err(|e| format!("J mech: {e}"))?,
                    contact_damage: flag(3),
                    lockups: flag(4),
                    safety_car: flag(5),
                    yellow_zones: flag(6),
                    blue_flags: flag(7),
                    formation_lap: flag(8),
                    drs_zones: flag(9),
                    fuel: flag(10),
                    tyre_temp: flag(11),
                });
            }
            Some("F") => {
                // Field model ("teams"). Absent on pre-teams logs, which are
                // all classic fields.
                let words: Vec<&str> = tok.collect();
                if words.len() != 1 {
                    return Err(format!("F: {} fields, expected 1", words.len()));
                }
                field = FieldMode::parse(words[0])
                    .ok_or_else(|| format!("F: field '{}'", words[0]))?;
            }
            Some("I") => {
                let car: u32 = tok.next().ok_or("I: car")?.parse().map_err(|e| format!("I: {e}"))?;
                let tick: u64 = tok.next().ok_or("I: tick")?.parse().map_err(|e| format!("I: {e}"))?;
                let mut f = [0.0f64; 3];
                for slot in &mut f {
                    *slot = tok.next().ok_or("I: field")?.parse().map_err(|e| format!("I: {e}"))?;
                }
                inputs.push((car, tick, Controls { throttle: f[0], brake: f[1], steer_cmd: f[2] }));
            }
            Some("R") => {
                expected_ticks = tok.next().ok_or("R: ticks")?.parse().map_err(|e| format!("R: {e}"))?;
                expected_hash = u64::from_str_radix(tok.next().ok_or("R: hash")?, 16)
                    .map_err(|e| format!("R hash: {e}"))?;
            }
            // Phase 11 header. The thresholds are read back rather than assumed,
            // so a replay judges the incident with the rule that actually
            // produced it — a decision replayed against today's defaults would
            // be a different decision wearing the same evidence.
            Some("D") => {
                let set = tok.next().ok_or("D: rule set")?;
                if set != steward::RULE_SET {
                    return Err(format!(
                        "D: rule set {set:?} is not this engine's {:?} — the incident record \
                         cannot be reproduced by a different rule set",
                        steward::RULE_SET
                    ));
                }
                let _ver: u32 = tok.next().ok_or("D: version")?.parse().map_err(|e| format!("D: {e}"))?;
                let n: usize = tok.next().ok_or("D: count")?.parse().map_err(|e| format!("D: {e}"))?;
                let hash = u64::from_str_radix(tok.next().ok_or("D: hash")?, 16)
                    .map_err(|e| format!("D hash: {e}"))?;
                let mut f = [0.0f64; 6];
                for slot in &mut f {
                    *slot = tok.next().ok_or("D: threshold")?.parse().map_err(|e| format!("D: {e}"))?;
                }
                let mut u = [0u64; 2];
                for slot in &mut u {
                    *slot = tok.next().ok_or("D: tick threshold")?.parse().map_err(|e| format!("D: {e}"))?;
                }
                let mut tail = [0.0f64; 3];
                for slot in &mut tail {
                    *slot = tok.next().ok_or("D: threshold")?.parse().map_err(|e| format!("D: {e}"))?;
                }
                steward_cfg = Some(steward::StewardConfig {
                    enabled: true,
                    history_s: f[0],
                    review_s: f[1],
                    contact_cs_penalty: f[2],
                    contact_cs_inconclusive: f[3],
                    contact_fault_margin: f[4],
                    track_limit_dist_m: f[5],
                    track_limit_ticks: u[0],
                    rejoin_min_off_ticks: u[1],
                    rejoin_danger_cs: tail[0],
                    rejoin_avoidable_gap_m: tail[1],
                    penalty_seconds: tail[2],
                });
                steward_expected_hash = Some(hash);
                steward_incidents_logged = n;
            }
            // Per-incident records, kept verbatim and compared against what
            // the replay re-derives. Counting them is not enough: the header
            // hash covers what *this engine* concluded, so an edited decision
            // line leaves that hash untouched and would slip through.
            Some("P") => steward_lines.push(line["P ".len()..].to_string()),
            _ => {}
        }
    }
    if track.custom.as_ref().is_some_and(|c| c.points.len() < 6) {
        return Err(format!(
            "log declares custom circuit '{}' with only {} control points (needs >= 6)",
            track.custom.as_ref().map(|c| c.name.as_str()).unwrap_or("?"),
            track.custom.as_ref().map(|c| c.points.len()).unwrap_or(0)
        ));
    }
    let cfg = RaceConfig {
        cars,
        laps,
        seed,
        grid_row_gap_m: grid_gap,
        contact_retire_cs: retire_cs,
        pace_spread: pace,
        noise,
        mode,
        formation,
        start_compound,
        strategy,
        pit,
        // No session line = Phase 4 log: reproduce that engine exactly.
        legacy_v2: !saw_session_line,
        // A replay never consults the model: the logged controls are the truth.
        ai: None,
        // Absent a W line this is None, and `Race::new` then runs the exact
        // engine that wrote the log — no legacy flag needed.
        weather,
        // A replay re-derives the cue log from the same states the original
        // run saw, so it carries the sound config and opens no socket: a
        // replay must never publish, and must never need a receiver.
        sound: sound_cfg,
        telemetry: None,
        track,
        // Absent a J line this is None: the exact pack posture that wrote
        // every pre-realism log.
        realism,
        // Absent an F line this is Classic: the exact field that wrote
        // every pre-teams log.
        field,
        // A D line means the run had the aid; the replay rebuilds it from the
        // thresholds that run recorded. No D line means it did not, and this
        // stays None — the exact engine that wrote the log.
        steward: steward_cfg,
        // Absent an O line the field ran in identity order, as every
        // pre-qualifying race did. A replay never re-runs qualifying: the
        // recorded order IS the result.
        grid_order,
        solo_car: None,
    };
    // Group inputs by driver-rate tick and re-execute. The loop runs through
    // the formation phase too (its ticks consume no recorded inputs).
    let mut race = Race::new(cfg);
    race.start();
    let start = Instant::now();
    let mut idx = 0usize;
    let mut held: Vec<Controls> = vec![Controls::default(); cars as usize];
    // Inputs are recorded in (tick asc, car asc) order.
    // Bound the loop by the run's own tick count (the R line), not by the
    // input stream: a race that ends the instant its last car retires stops
    // RECORDING one physics tick before it stops RUNNING, and a replay
    // bounded by inputs would silently skip that final tick — a retirement
    // (or any state change) on it would never reproduce. The held controls
    // carry the state past the last logged driver tick.
    while race.state.phase != RacePhase::Finished && race.state.tick < expected_ticks.max(1) {
        if (race.state.phase == RacePhase::Running
                || race.state.phase == RacePhase::FormationLap)
                && race.state.tick.is_multiple_of(DRIVER_EVERY as u64)
            {
            while idx < inputs.len() && inputs[idx].1 == race.state.tick {
                let (car, _, c) = inputs[idx];
                held[car as usize] = c;
                idx += 1;
            }
        }
        let n_cars = race.state.cars.len();
        race.controls[..n_cars].copy_from_slice(&held[..n_cars]);
        // one tick without re-running driver decisions
        race.tick_replay();
    }
    let wall = start.elapsed().as_secs_f64();
    let mut outcome = race.outcome(wall);
    // The cue log is compared on its own terms, separately from the
    // trajectory hash: a run can reproduce its physics exactly and still
    // produce a different sound, and that is a distinct failure worth naming.
    outcome.sound_expected_hash = sound_expected_hash;
    if let (Some(exp), Some(got)) = (sound_expected_hash, outcome.sound.as_ref()) {
        if got.log_hash != exp || got.events != sound_events_logged {
            return Err(format!(
                "cue log did not reproduce: replay produced {} cues (hash {:016x}),                  log recorded {sound_events_logged} (hash {exp:016x})",
                got.events, got.log_hash
            ));
        }
    }
    // The incident record is checked on its own terms, like the cue log: a
    // replay can reproduce every tick of physics and still reach a different
    // verdict, and that is a distinct failure worth naming rather than hiding
    // behind a passing trajectory hash.
    outcome.steward_expected_hash = steward_expected_hash;
    if let (Some(exp), Some(got)) =
        (steward_expected_hash, outcome.steward_incidents.as_ref())
    {
        let hash = steward::incidents_hash(got);
        if hash != exp || got.len() != steward_incidents_logged {
            return Err(format!(
                "incident record did not reproduce: replay reached {} finding(s) (hash \
                 {:016x}), log recorded {steward_incidents_logged} (hash {exp:016x})",
                got.len(),
                hash
            ));
        }
        // Field-by-field, per incident. The hash above proves the replay
        // reached the same record; this proves the *log says so*. Without it
        // the P lines are decoration.
        if steward_lines.len() != got.len() {
            return Err(format!(
                "incident record did not reproduce: log carries {} P line(s), replay derived {}",
                steward_lines.len(),
                got.len()
            ));
        }
        for (i, (logged, derived)) in steward_lines.iter().zip(got.iter()).enumerate() {
            let expected = steward::incident_line(derived);
            if *logged != expected {
                return Err(format!(
                    "incident record did not reproduce at finding #{i}:
  log:    {logged}
  replay: {expected}"
                ));
            }
        }
    }
    Ok((outcome.state_hash == expected_hash && outcome.ticks == expected_ticks, outcome))
}

impl Race {
    /// One tick using pre-set controls (replay path; no driver calls).
    /// Structurally identical to `tick` — every override and state advance
    /// goes through the same pure helpers, on the same cadence.
    pub fn tick_replay(&mut self) {
        if self.state.phase == RacePhase::Ready {
            return;
        }
        // Same cadence as the live loop, formation included.
        self.advance_weather();
        if self.state.phase == RacePhase::FormationLap {
            // The rolling lap's drivers are part of the recorded stream: the
            // outer replay loop drains inputs and copies held controls, then
            // this runs the same physics/transitions as the live lap.
            self.tick_formation_lap(true);
            self.tick_costs_us.push(0.0);
            return;
        }
        if self.state.phase == RacePhase::Formation {
            // Same locked-grid law as the live formation tick.
            let t0 = Instant::now();
            let locked = locked_controls();
            let params = self.car_params;
            for e in &mut self.state.cars {
                e.car = physics::step(&e.car, &locked, &params, DT);
            }
            self.state.tick += 1;
            self.state.sim_time = self.state.tick as f64 * DT;
            let n = (self.state.sim_time.floor() as u64).min(5) as u8;
            if n != self.state.lights {
                self.state.lights = n;
                self.state.rc_events.push(RcEvent::Lights(n));
            }
            if self.state.tick >= self.state.green_tick {
                self.state.phase = RacePhase::Running;
                self.state.lights = 0;
                self.state.rc_events.push(RcEvent::Green);
            }
            self.mix_state();
            self.observe();
            self.tick_costs_us.push(t0.elapsed().as_secs_f64() * 1e6);
            return;
        }
        let t0 = Instant::now();
        if self.state.tick.is_multiple_of(DRIVER_EVERY as u64) {
            self.recompute_progress();
            self.traffic_awareness(); // retired flag + caps affect nothing here:
            // replay holds controls, caps_hint is only read by drivers.
            self.update_leaderboard();
        }
        for i in 0..self.state.cars.len() {
            let ctrl = self.control_for(i);
            let params = self.params_for(i);
            // Same grip sampling as the live loop, so a replay's weather
            // summary matches the run it is replaying.
            if self.weather.is_some() && self.state.tick.is_multiple_of(DRIVER_EVERY as u64) {
                self.weather_grip_sum += self.surface_grip(0);
                self.weather_grip_samples += 1;
            }
            self.state.cars[i].car = physics::step(&self.state.cars[i].car, &ctrl, &params, DT);
            self.accrue_wear(i);
            self.service_countdown(i);
            self.realism_step(i);
        }
        self.contact_pass();
        self.state.tick += 1;
        self.state.sim_time = self.state.tick as f64 * DT;
        self.mix_state();
        // Phase 9 observers, same position as the live loop. A replay opens
        // no socket — `telemetry` is None here — but it re-derives the whole
        // cue log, which is exactly what makes the T line checkable.
        self.observe();
        let cost = t0.elapsed().as_secs_f64() * 1e6;
        if cost > DT * 1e6 {
            self.deadline_misses += 1;
        }
        self.tick_costs_us.push(cost);

        let all_done = self.state.cars.iter().all(|e| e.retired || e.finish_time_s.is_some());
        let leader_done = self
            .state
            .cars
            .iter()
            .filter(|e| !e.retired)
            .map(|e| e.finish_time_s.unwrap_or(f64::INFINITY))
            .fold(f64::INFINITY, f64::min);
        if all_done
            || leader_done < self.state.sim_time - 60.0
            || self.state.sim_time > self.config.laps as f64 * 60.0 + 120.0
        {
            // Same three conditions as the live tick — including the
            // red-flag cap, or a capped race could never replay.
            self.state.phase = RacePhase::Finished;
            self.update_leaderboard();
        }
    }
}

/// The circuit's DRS zones: runs of near-straight track (`|curvature|` at or
/// under [`DRS_K_MAX`]) at least [`DRS_MIN_STRAIGHT_M`] long, taking the
/// longest one or two runs and keeping each run's MIDDLE 70% (the ends stay
/// honest braking and turn-in zones). Pure from the geometry.
pub fn derive_drs_zones_pub(track: &crate::track::Track) -> Vec<(f64, f64)> {
    derive_drs_zones(track)
}

fn derive_drs_zones(track: &crate::track::Track) -> Vec<(f64, f64)> {
    const DRS_K_MAX: f64 = 1.0 / 400.0; // radius >= 400 m reads as "straight"
    const DRS_MIN_STRAIGHT_M: f64 = 300.0;
    let len = track.length();
    let step = 5.0_f64;
    let n = (len / step).floor() as usize;
    if n < 8 {
        return Vec::new();
    }
    // Runs of straight samples (wrapped over the lap).
    let mut runs: Vec<(f64, f64)> = Vec::new(); // (start_s, length)
    let mut run_start: Option<f64> = None;
    for i in 0..=n {
        let s = (i as f64 * step).rem_euclid(len);
        let straight = i < n && track.curvature_at(s).abs() <= DRS_K_MAX;
        match (run_start, straight) {
            (None, true) => run_start = Some(s),
            (Some(start), false) => {
                let from = start;
                let to = s;
                let run_len = (to - from).rem_euclid(len);
                if run_len >= DRS_MIN_STRAIGHT_M {
                    runs.push((from, run_len));
                }
                run_start = None;
            }
            _ => {}
        }
    }
    // A run that spans the wrap (never closed) — close it leniently.
    if let Some(start) = run_start {
        let run_len = len - start;
        if run_len >= DRS_MIN_STRAIGHT_M {
            runs.push((start, run_len));
        }
    }
    runs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    runs.truncate(2);
    runs.iter()
        .map(|(from, run_len)| {
            // Middle 70% of the run.
            let pad = run_len * 0.15;
            let start = (*from + pad).rem_euclid(len);
            let zone = (run_len - 2.0 * pad).max(50.0);
            (start, zone)
        })
        .collect()
}
