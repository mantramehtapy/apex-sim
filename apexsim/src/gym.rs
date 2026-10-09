//! The Gymnasium-facing training environment, and the persistent synchronous
//! interface Python drives it through.
//!
//! ## What the plan asked for, and where it lives
//!
//! * *"Persistent, synchronous headless stepping interface from Python to Rust;
//!   avoid starting a process for every action."* — [`GymEnv`] is an in-process
//!   object reached over a C ABI ([`ffi`]), so a Python `ctypes` handle calls
//!   straight into Rust. No process, no socket, no serialisation of the race
//!   state per action. A second transport, an NDJSON line protocol on a
//!   long-lived subprocess's stdio, exists for callers that cannot load a
//!   DLL, and it drives the identical [`GymEnv`] — so the two cannot diverge
//!   in behaviour, only in transport.
//! * *"Define reset, action frequency, observation units/normalization,
//!   rewards, and episode termination."* — [`EnvSpec`] and [`RewardSpec`] are
//!   that definition, in one serialisable place. They are not implicit in the
//!   code and not hard-coded in the trainer: the spec is written into every
//!   artifact, and a result produced under a different spec is refused.
//! * *"Record observations with the actions that produced them; split training
//!   and evaluation episodes."* — [`GymEnv::step`] appends to the episode log
//!   in the same call that produced the transition, so the pairing cannot be
//!   lost by a trainer that crashes midway. Seeds are owned by [`SplitLedger`],
//!   not by the trainer: a seed claimed for training can never be handed out
//!   for evaluation.
//!
//! ## The environment is a race, not a toy
//!
//! It drives car 0 of a real [`Race`](crate::race::Race) with the field running
//! the deterministic Phase 4/5 driver around it. That is deliberate: the plan
//! says to compare a learned driver against the Phase 1/4 baseline and warns
//! against assuming self-play makes a good teacher. Imitating the existing
//! deterministic driver, in the environment it will actually be evaluated in,
//! is the comparison the plan asks for — and the traffic observation channels
//! are populated by real cars rather than being dead inputs a single-car env
//! could never teach the policy to read.

use crate::policy::{self, ObsSpec, TrafficContext};
use crate::race::{Race, RaceConfig, RacePhase};
use crate::physics::Controls;
use crate::race::SessionMode;
use serde::{Deserialize, Serialize};

/// Physics ticks per action. 5 × 4 ms = 20 ms, so the action rate is 50 Hz —
/// the same rate at which the local driver decides, which keeps the learned
/// driver and the baseline on equal footing.
pub const ACTION_EVERY_TICKS: u32 = 5;
/// The action rate in hertz, for the spec and for every artifact.
pub const ACTION_HZ: f64 = 1.0 / (ACTION_EVERY_TICKS as f64 * crate::sim::DT);

/// Reward weights. Explicit, serialised, and never implicit in a formula.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RewardSpec {
    /// Metres of track progress per unit of reward.
    pub progress_per_m: f64,
    /// Charged once per action step spent beyond the track edge (negative).
    pub off_track_per_step: f64,
    /// Speed bonus per action step, at `speed_ref_mps` (positive).
    pub speed_per_step: f64,
    /// The speed at which the speed bonus is worth `speed_per_step`.
    pub speed_ref_mps: f64,
}

impl Default for RewardSpec {
    fn default() -> Self {
        RewardSpec {
            progress_per_m: 1.0,
            off_track_per_step: -0.5,
            speed_per_step: 0.02,
            speed_ref_mps: 75.0,
        }
    }
}

/// Termination thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerminationSpec {
    /// Seconds below `stall_speed_mps` after the start that count as a stall.
    pub stall_after_s: f64,
    pub stall_speed_mps: f64,
    /// Seconds continuously beyond the track edge *while not making progress*
    /// that count as off and gone.
    ///
    /// The speed condition is not a refinement, it is the whole rule. The
    /// Phase 4/5 baseline genuinely runs wide: on the default circuit it spends
    /// up to 8.5 s continuously outside the white line at 25-30 m/s and
    /// recovers, still setting its fastest lap. A rule that ended the episode
    /// for being off the road would therefore terminate the expert on its
    /// second lap and train on nothing but the opening. What actually ends a
    /// run is a car that is off the road AND stopped — beached, spun, or
    /// parked in the gravel — because from there no amount of throttle recovers
    /// it and every further transition is noise.
    pub off_track_after_s: f64,
    /// Speed below which being off the road counts as beached rather than
    /// running wide. Matches `stall_speed_mps` by default.
    pub off_track_speed_mps: f64,
}

impl Default for TerminationSpec {
    fn default() -> Self {
        TerminationSpec {
            stall_after_s: 3.0,
            stall_speed_mps: 0.5,
            off_track_after_s: 2.0,
            off_track_speed_mps: 0.5,
        }
    }
}

/// The environment definition. Serialised into every artifact; two runs are
/// only comparable when their specs match.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvSpec {
    pub cars: u32,
    pub laps: u32,
    pub seed: u64,
    /// Physics ticks advanced per action.
    pub action_every_ticks: u32,
    /// Hard cap on actions per episode; exceeding it truncates.
    pub max_steps: u64,
    pub formation: bool,
    /// Per-car driver-input noise, seeded per car.
    ///
    /// Defaults to the engine's own default rather than zero. At zero the seed
    /// would only vary the length of the lights hold, every seed would produce
    /// the same episode to the millisecond, and a train/evaluation split over
    /// seeds would be a split over identical data — leakage-proof and
    /// meaningless at once. With the engine default each seed is a genuinely
    /// different race, and "the evaluation seeds were never trained on" means
    /// something.
    pub noise: f64,
    pub obs: ObsSpec,
    pub reward: RewardSpec,
    pub termination: TerminationSpec,
}

impl Default for EnvSpec {
    fn default() -> Self {
        EnvSpec {
            cars: 20,
            laps: 4,
            seed: 42,
            action_every_ticks: ACTION_EVERY_TICKS,
            max_steps: 12_000,
            formation: true,
            // The engine's own default, so the environment is the race the
            // baseline is actually measured in rather than a quieter one.
            noise: RaceConfig::default().noise,
            obs: ObsSpec::default(),
            reward: RewardSpec::default(),
            termination: TerminationSpec::default(),
        }
    }
}

impl EnvSpec {


    fn race_config(&self) -> RaceConfig {
        RaceConfig {
            cars: self.cars,
            laps: self.laps,
            seed: self.seed,
            mode: SessionMode::Race,
            formation: self.formation,
            noise: self.noise,
            // Nothing else is switched on: a trainer environment that also ran
            // weather, an advisor or a steward would be measuring three other
            // subsystems as much as the policy.
            ..RaceConfig::default()
        }
    }
}

/// What one `step` reports beyond the observation and the reward.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepInfo {
    pub terminated: bool,
    pub truncated: bool,
    /// Why the episode ended; empty while it runs.
    pub reason: &'static str,
    pub sim_time_s: f64,
    pub step: u64,
}

/// Everything an evaluation run needs to say about one episode.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct EpisodeMetrics {
    pub seed: u64,
    pub steps: u64,
    pub sim_time_s: f64,
    pub total_reward: f64,
    /// Track metres car 0 covered along its own progress.
    pub progress_m: f64,
    pub laps: u32,
    pub lap_times_s: Vec<f64>,
    pub best_lap_s: Option<f64>,
    pub off_track_steps: u64,
    /// Contacts involving car 0.
    pub contacts: u32,
    /// Final classification position of car 0 (1 = won).
    pub position: Option<u32>,
    pub retired: bool,
    pub finished: bool,
    pub reason: String,
}

/// One recorded transition, appended by `step` itself.
#[derive(Clone, Debug)]
pub struct Transition {
    pub obs: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

/// The persistent environment handle Python holds.
pub struct GymEnv {
    pub spec: EnvSpec,
    race: Race,
    steps: u64,
    last_ctrl: Controls,
    prev_progress: f64,
    total_reward: f64,
    off_track_steps: u64,
    stall_ticks: u64,
    off_track_ticks: u64,
    obs: Vec<f32>,
    log: Vec<Transition>,
    info: StepInfo,
    /// Lapped-out flags, so a lap time is not recorded twice.
    seen_crossings: u32,
}

impl GymEnv {
    pub fn new(spec: EnvSpec) -> GymEnv {
        let mut env = GymEnv {
            spec,
            race: Race::new(spec.race_config()),
            steps: 0,
            last_ctrl: Controls::default(),
            prev_progress: 0.0,
            total_reward: 0.0,
            off_track_steps: 0,
            stall_ticks: 0,
            off_track_ticks: 0,
            obs: vec![0.0; policy::OBS_DIM],
            log: Vec::new(),
            info: StepInfo::default(),
            seen_crossings: 0,
        };
        env.reset(spec.seed);
        env
    }

    /// Seeded reset. Same seed, same spec, same first observation — the
    /// property a trainer needs to compare two policies on identical episodes.
    pub fn reset(&mut self, seed: u64) -> Vec<f32> {
        let mut spec = self.spec;
        spec.seed = seed;
        self.spec = spec;
        self.race = Race::new(spec.race_config());
        self.race.start();
        self.steps = 0;
        self.last_ctrl = Controls::default();
        self.prev_progress = self.car0_progress();
        self.total_reward = 0.0;
        self.off_track_steps = 0;
        self.stall_ticks = 0;
        self.off_track_ticks = 0;
        self.log.clear();
        self.seen_crossings = 0;
        self.info = StepInfo { reason: "", ..StepInfo::default() };
        // Car 0 is under the policy from the first tick, so the environment
        // never quietly hands the opening laps to the baseline driver.
        self.race.learned_controls = Some(self.last_ctrl);
        self.obs = self.observe();
        self.obs.clone()
    }

    pub fn observation(&self) -> &[f32] {
        &self.obs
    }

    pub fn info(&self) -> StepInfo {
        self.info
    }

    pub fn total_reward(&self) -> f64 {
        self.total_reward
    }

    pub fn recorded(&self) -> &[Transition] {
        &self.log
    }

    pub fn race(&self) -> &Race {
        &self.race
    }

    /// Mutable race access. Lets a caller drive the baseline teacher directly
    /// instead of through `step`, which is what the policy-parity harness does.
    pub fn race_mut(&mut self) -> &mut Race {
        &mut self.race
    }





    /// Take the episode log, leaving the environment ready for another
    /// episode. This is how the trainer gets its (obs, action) pairs out.
    pub fn take_log(&mut self) -> Vec<Transition> {
        std::mem::take(&mut self.log)
    }

    fn car0(&self) -> &crate::race::CarEntry {
        &self.race.state.cars[0]
    }

    fn car0_progress(&self) -> f64 {
        self.car0().total_progress
    }

    /// Build car 0's observation, filling the traffic channels from the field.
    fn observe(&mut self) -> Vec<f32> {
        let me = self.car0();
        let car = me.car;
        let s = me.s;
        let heading_error = policy::heading_error(&car, &self.race.track, s);
        let track = &self.race.track;

        // Nearest car ahead and behind, by track progress rather than by grid
        // id: the race order is what a driver actually reads.
        let mut ahead: Option<(f64, f64, f64)> = None;
        let mut behind: Option<f64> = None;
        for other in &self.race.state.cars {
            if other.id == 0 || other.retired {
                continue;
            }
            let ds = other.total_progress - me.total_progress;
            if ds > 0.0 {
                let lat = (other.lateral - me.lateral).abs();
                let closing = car.vel_long - other.car.vel_long;
                if ahead.is_none_or(|(g, _, _)| ds < g) {
                    ahead = Some((ds, lat, closing));
                }
            } else if behind.is_none_or(|g| -ds < g) {
                behind = Some(-ds);
            }
        }

        let track_len = track.length();
        let ctx = TrafficContext {
            ahead,
            behind,
            lap_fraction: if track_len > 0.0 {
                (s / track_len).clamp(0.0, 1.0)
            } else {
                0.0
            },
            budget_left: if self.spec.max_steps == 0 {
                1.0
            } else {
                1.0 - self.steps as f64 / self.spec.max_steps as f64
            },
            grip_scale: self.race.grip_scale_of(0),
            v_target: self.race.v_target_of(0),
        };
        policy::observe(
            &car,
            track,
            s,
            heading_error,
            self.last_ctrl.steer_cmd,
            &ctx,
            &self.spec.obs,
        )
    }

    /// Advance one action. Synchronous: returns when the physics has moved.
    pub fn step(&mut self, action: &[f32]) -> Result<StepInfo, String> {
        if action.len() < policy::ACTION_DIM {
            return Err(format!(
                "action has {} values, expected at least {}",
                action.len(),
                policy::ACTION_DIM
            ));
        }
        if self.info.terminated || self.info.truncated {
            return Err("episode has ended; call reset() first".into());
        }
        let max_steer = crate::physics::CarParams::default().max_steer;
        let ctrl = policy::decode(action, max_steer);
        self.last_ctrl = ctrl;
        self.race.learned_controls = Some(ctrl);

        for _ in 0..self.spec.action_every_ticks {
            self.race.tick();
        }
        self.steps += 1;

        // ---- reward ----
        let progress = self.car0_progress();
        let d_progress = progress - self.prev_progress;
        self.prev_progress = progress;
        let (_, lateral) = self.race.track.project(self.car0().car.pos_x, self.car0().car.pos_y);
        let off = self.race.track.is_off_track(lateral);
        if off {
            self.off_track_steps += 1;
        }
        let r = &self.spec.reward;
        let reward = r.progress_per_m * d_progress
            + r.speed_per_step * (self.car0().car.vel_long / r.speed_ref_mps).clamp(-1.0, 1.0)
            + if off { r.off_track_per_step } else { 0.0 };
        self.total_reward += reward;

        // ---- termination ----
        let t = self.spec.termination;
        // Copied out of the borrow before any field is written: the phase and
        // the car's flags are read once, here, and the counters below need
        // `self` mutably.
        let phase = self.race.state.phase;
        let (car_retired, v_long) = {
            let me = self.car0();
            (me.retired, me.car.vel_long)
        };
        // Before green the car is held still by the lights on purpose, so a
        // stall or an off-track reading during formation is the race working,
        // not the policy failing. Counting those ticks would end every episode
        // about three seconds in, during the standing start, and the whole
        // training set would be five lights and nothing else.
        let racing = phase != RacePhase::Formation;
        if racing {
            if !car_retired && v_long < t.stall_speed_mps {
                self.stall_ticks += self.spec.action_every_ticks as u64;
            } else {
                self.stall_ticks = 0;
            }
            if off && v_long.abs() < t.off_track_speed_mps {
                self.off_track_ticks += self.spec.action_every_ticks as u64;
            } else {
                self.off_track_ticks = 0;
            }
        }
        let secs = |ticks: u64| ticks as f64 * crate::sim::DT;
        let sim_time = self.race.state.sim_time;
        let (term, trunc, reason) = if phase == RacePhase::Finished {
            (true, false, "race_finished")
        } else if car_retired {
            (true, false, "car_retired")
        } else if secs(self.stall_ticks) > t.stall_after_s {
            (true, false, "stalled")
        } else if secs(self.off_track_ticks) > t.off_track_after_s {
            (true, false, "off_track")
        } else if self.steps >= self.spec.max_steps {
            (false, true, "step_budget")
        } else {
            (false, false, "")
        };
        self.info = StepInfo {
            terminated: term,
            truncated: trunc,
            reason,
            sim_time_s: sim_time,
            step: self.steps,
        };

        self.obs = self.observe();

        // Recorded here, in the call that produced the transition, so an
        // observation can never end up paired with the wrong action.
        self.log.push(Transition {
            obs: self.obs.clone(),
            action: action[..policy::ACTION_DIM].to_vec(),
            reward,
            terminated: term,
            truncated: trunc,
        });

        Ok(self.info)
    }

    /// Car 0's race metrics. Callable mid-episode for progress reporting.
    pub fn metrics(&self) -> EpisodeMetrics {
        let me = self.car0();
        let position = self
            .race
            .state
            .leaderboard
            .iter()
            .find(|r| r.id == 0)
            .map(|r| r.pos);
        let contacts = self
            .race
            .state
            .contacts
            .iter()
            .filter(|c| c.a == 0 || c.b == 0)
            .count() as u32;
        EpisodeMetrics {
            seed: self.spec.seed,
            steps: self.steps,
            sim_time_s: self.race.state.sim_time,
            total_reward: self.total_reward,
            progress_m: me.total_progress,
            laps: me.laps_crossings,
            lap_times_s: me.lap_times.clone(),
            best_lap_s: me.best_lap,
            off_track_steps: self.off_track_steps,
            contacts,
            position,
            retired: me.retired,
            finished: self.race.state.phase == RacePhase::Finished,
            reason: self.info.reason.to_string(),
        }
    }

    /// Run one episode to its end with a given policy.
    ///
    /// `f` maps an observation to an action. This is the loop a Gymnasium
    /// caller would write by hand, kept here so the Rust-side gate tests and
    /// the Python trainer exercise the same code path.
    pub fn rollout(
        &mut self,
        mut f: impl FnMut(&[f32]) -> Vec<f32>,
    ) -> Result<EpisodeMetrics, String> {
        loop {
            let obs = self.observation().to_vec();
            let action = f(&obs);
            let info = self.step(&action)?;
            if info.terminated || info.truncated {
                break;
            }
        }
        Ok(self.metrics())
    }
}

// ---------------------------------------------------------------------------
// Evaluating a learned driver against the baseline
// ---------------------------------------------------------------------------

/// One race's worth of car-0 results, in the shape the comparison is made in.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PolicyRaceResult {
    pub seed: u64,
    pub best_lap_s: Option<f64>,
    pub laps_done: u32,
    pub position: Option<u32>,
    pub finished: bool,
    pub retired: bool,
    pub progress_m: f64,
    /// Inference failures that made the baseline take over. A policy that
    /// errors is a policy that did not drive, and this counts it.
    pub fallbacks: u64,
    pub actions: u64,
}

/// Aggregate over a split, for one controller.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PolicyScore {
    pub episodes: usize,
    pub finished: usize,
    pub retired: usize,
    pub mean_best_lap_s: f64,
    pub mean_laps: f64,
    pub mean_position: f64,
    pub mean_progress_m: f64,
    pub fallbacks: u64,
    pub actions: u64,
}

/// Build car 0's observation from a live race, without owning the race.
///
/// Deliberately duplicated in shape from [`GymEnv::observe`] rather than shared
/// with it: one borrows the race mutably to read the driver, the other does not,
/// and a shared helper would have to take the weaker signature and force the
/// environment to give up the observation it builds every tick. The gate test
/// `observations_agree_between_env_and_race_harness` pins the two together.
fn observe_car0(race: &Race, spec: &EnvSpec, last: Controls, step: u64) -> Vec<f32> {
    let me = &race.state.cars[0];
    let car = me.car;
    let s = me.s;
    let heading_error = policy::heading_error(&car, &race.track, s);
    let mut ahead = None;
    let mut behind = None;
    for other in &race.state.cars {
        if other.id == 0 || other.retired {
            continue;
        }
        let ds = other.total_progress - me.total_progress;
        if ds > 0.0 {
            let lat = (other.lateral - me.lateral).abs();
            let closing = car.vel_long - other.car.vel_long;
            if ahead.is_none_or(|(g, _, _)| ds < g) {
                ahead = Some((ds, lat, closing));
            }
        } else if behind.is_none_or(|g| -ds < g) {
            behind = Some(-ds);
        }
    }
    let track_len = race.track.length();
    let ctx = TrafficContext {
        ahead,
        behind,
        lap_fraction: if track_len > 0.0 {
            (s / track_len).clamp(0.0, 1.0)
        } else {
            0.0
        },
        budget_left: if spec.max_steps == 0 {
            1.0
        } else {
            1.0 - step as f64 / spec.max_steps as f64
        },
        grip_scale: race.grip_scale_of(0),
        v_target: race.v_target_of(0),
    };
    policy::observe(&car, &race.track, s, heading_error, last.steer_cmd, &ctx, &spec.obs)
}

/// Run one race with `controller` driving car 0 and the local driver driving
/// everyone else.
///
/// This is deliberately a whole [`Race`] run rather than a [`GymEnv`] episode.
/// The gate asks whether the learned driver improves declared metrics, and the
/// declared metrics are race metrics — laps, best lap, finishing position —
/// which only exist once the race has run to its own conclusion. The
/// environment's stall and off-track termination rules would cut a race short
/// and quietly change what is being measured.
pub fn run_policy_race(
    spec: &EnvSpec,
    seed: u64,
    controller: &mut policy::Controller,
) -> PolicyRaceResult {
    let mut cfg = spec.race_config();
    cfg.seed = seed;
    let mut race = Race::new(cfg);
    race.start();
    race.learned_controls = Some(Controls::default());

    let max_steer = crate::physics::CarParams::default().max_steer;
    let mut last = Controls::default();
    let mut result = PolicyRaceResult { seed, ..Default::default() };

    while race.state.phase != RacePhase::Finished {
        if result.actions >= spec.max_steps {
            break;
        }
        let obs = observe_car0(&race, spec, last, result.actions);
        let (ctrl, err) = {
            let mut fallback = || race.baseline_control_for_car0();
            controller.act(&obs, max_steer, &mut fallback)
        };
        if err.is_some() {
            result.fallbacks += 1;
        }
        result.actions += 1;
        last = ctrl;
        race.learned_controls = Some(ctrl);
        for _ in 0..spec.action_every_ticks {
            race.tick();
        }
    }

    let me = &race.state.cars[0];
    result.best_lap_s = me.best_lap;
    result.laps_done = me.laps_crossings;
    result.progress_m = me.total_progress;
    result.retired = me.retired;
    result.finished = race.state.phase == RacePhase::Finished;
    result.position = race
        .state
        .leaderboard
        .iter()
        .find(|r| r.id == 0)
        .map(|r| r.pos);
    result
}

/// Aggregate per-race results.
///
/// `None` best laps are skipped rather than counted as zero: a driver that set
/// no lap did not set a slow lap, and averaging a zero in would quietly flatter
/// it against a controller that did.
pub fn score(results: &[PolicyRaceResult]) -> PolicyScore {
    let n = results.len();
    if n == 0 {
        return PolicyScore::default();
    }
    let laps: Vec<f64> = results.iter().filter_map(|r| r.best_lap_s).collect();
    let positions: Vec<f64> = results
        .iter()
        .filter_map(|r| r.position)
        .map(|p| p as f64)
        .collect();
    PolicyScore {
        episodes: n,
        finished: results.iter().filter(|r| r.finished).count(),
        retired: results.iter().filter(|r| r.retired).count(),
        mean_best_lap_s: if laps.is_empty() {
            0.0
        } else {
            laps.iter().sum::<f64>() / laps.len() as f64
        },
        mean_laps: results.iter().map(|r| r.laps_done as f64).sum::<f64>() / n as f64,
        mean_position: if positions.is_empty() {
            0.0
        } else {
            positions.iter().sum::<f64>() / positions.len() as f64
        },
        mean_progress_m: results.iter().map(|r| r.progress_m).sum::<f64>() / n as f64,
        fallbacks: results.iter().map(|r| r.fallbacks).sum(),
        actions: results.iter().map(|r| r.actions).sum(),
    }
}

/// Signed differences. Every field is learned-minus-baseline, so a positive
/// value always means the same direction: more laps, further down the road,
/// higher up the order — except `best_lap_s`, where positive means *slower*,
/// because that is the one metric where bigger is worse.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Deltas {
    pub best_lap_s: f64,
    pub laps: f64,
    pub position: f64,
    pub progress_m: f64,
}

/// The metrics the enable rule is written against, named in the artifact so a
/// reader can see what "improves" was decided on.
pub const DECLARED_METRICS: [&str; 4] = ["best_lap_s", "laps", "position", "progress_m"];

#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    pub seeds: Vec<u64>,
    pub cars: u32,
    pub laps: u32,
    pub declared_metrics: [&'static str; 4],
    pub baseline: PolicyScore,
    pub learned: PolicyScore,
    pub deltas: Deltas,
    pub improves: bool,
    /// Why the learned driver was not enabled. Empty exactly when it was.
    pub reasons: Vec<String>,
}

/// Run baseline and learned over the same seeds and say whether the learned
/// driver is worth turning on.
///
/// The rule is fixed in code rather than left to the reader, because the gate
/// is a decision — "A trained driver is enabled only if it improves declared
/// metrics within the runtime budget" — and a decision whose criteria are
/// chosen after seeing the numbers is not a gate.
pub fn compare(
    spec: &EnvSpec,
    seeds: &[u64],
    model: Option<&std::path::Path>,
) -> Result<Comparison, String> {
    if seeds.is_empty() {
        return Err("comparison needs at least one seed".into());
    }
    let mut learned_ctrl = match model {
        Some(p) => policy::Controller::Learned(crate::onnx::OnnxSession::load(p)?),
        None => policy::Controller::Baseline,
    };
    let mut baseline_runs = Vec::new();
    let mut learned_runs = Vec::new();
    for &seed in seeds {
        // Baseline first on the same seed, so the field, the grid and the
        // weather-free conditions are identical and any difference is the
        // driver rather than the race.
        baseline_runs.push(run_policy_race(spec, seed, &mut policy::Controller::Baseline));
        learned_runs.push(run_policy_race(spec, seed, &mut learned_ctrl));
    }
    let baseline = score(&baseline_runs);
    let learned = score(&learned_runs);

    let deltas = Deltas {
        best_lap_s: learned.mean_best_lap_s - baseline.mean_best_lap_s,
        laps: learned.mean_laps - baseline.mean_laps,
        position: learned.mean_position - baseline.mean_position,
        progress_m: learned.mean_progress_m - baseline.mean_progress_m,
    };

    let mut reasons = Vec::new();
    // A baseline that set no lap cannot be compared against; say so rather than
    // comparing two zeros and declaring parity.
    if baseline.mean_best_lap_s <= 0.0 {
        reasons.push("the baseline set no lap, so there is nothing to improve on".into());
    } else if learned.mean_best_lap_s <= 0.0 {
        reasons.push("the learned driver set no lap".into());
    } else if deltas.best_lap_s >= 0.0 {
        reasons.push(format!(
            "mean best lap did not improve: baseline {:.3}s vs learned {:.3}s",
            baseline.mean_best_lap_s, learned.mean_best_lap_s
        ));
    }
    // A policy that trades pace for reliability is not faster, but covering
    // more road is the other half of racing well, so distance counts — with a
    // lap gain of at least a tenth of a lap, which is noise if it is not.
    if deltas.laps <= 0.1 && deltas.progress_m <= 0.0 {
        reasons.push("no further down the road".into());
    }
    if deltas.position > 0.0 {
        reasons.push(format!(
            "classified {:.2} places worse on average",
            deltas.position
        ));
    }
    if learned.fallbacks > 0 {
        reasons.push(format!(
            "{} inference errors fell back to the baseline",
            learned.fallbacks
        ));
    }
    if learned.retired > baseline.retired {
        reasons.push(format!(
            "{} retirements against the baseline's {}",
            learned.retired, baseline.retired
        ));
    }
    let improves = reasons.is_empty();

    Ok(Comparison {
        seeds: seeds.to_vec(),
        cars: spec.cars,
        laps: spec.laps,
        declared_metrics: DECLARED_METRICS,
        baseline,
        learned,
        deltas,
        improves,
        reasons,
    })
}

// ---------------------------------------------------------------------------
// End-to-end inference latency
// ---------------------------------------------------------------------------

/// Latency distribution for one policy across a grid, in microseconds.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LatencyReport {
    pub cars: u32,
    pub runs: u64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
    /// The action period the whole grid has to fit inside.
    pub budget_us: f64,
    pub within_budget: bool,
    pub model: String,
    pub op_nodes: usize,
}

/// Measure end-to-end inference latency: model load is excluded (once, up
/// front), and everything after it — building the observation, the ONNX walk,
/// the matmuls — is inside the timed region, because that is what the race
/// loop actually pays.
///
/// `cars` is the grid the policy is intended for, and it matters: the report
/// is per-action, and a race asks for one action per car per action period. A
/// single evaluation being fast says nothing about a 20-car grid.
pub fn benchmark_latency(
    model: &std::path::Path,
    obs_dim: usize,
    cars: u32,
    runs: u64,
    action_every_ticks: u32,
) -> Result<LatencyReport, String> {
    let mut session = crate::onnx::OnnxSession::load(model)?;
    let op_nodes = session.model.nodes.len();
    let mut obs = vec![0.0f32; obs_dim];
    // A deterministic, non-degenerate observation: a vector of exact zeros is
    // the one input a network can fast-path, and benchmarking it would flatter
    // the result.
    let mut x: u32 = 0x1234_5678;
    for v in obs.iter_mut() {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *v = ((x >> 8) as f32 / 8_388_608.0) - 1.0;
    }
    // Warm the pages and the branch predictors; the first call on a cold cache
    // is a file-system fact, not a latency property of the model.
    for _ in 0..32 {
        session.run(&obs)?;
    }

    let mut samples: Vec<f64> = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let t = std::time::Instant::now();
        session.run(&obs)?;
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f64| -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        // Nearest-rank, so p99 is a sample that really happened rather than an
        // interpolation between two that did.
        let rank = (q * samples.len() as f64).ceil() as usize;
        samples[rank.clamp(1, samples.len()) - 1]
    };
    let budget_us = action_every_ticks as f64 * crate::sim::DT * 1e6;
    let p99 = pick(0.99);
    let _ = cars; // the grid is reported, not multiplied in: the budget is per action
    Ok(LatencyReport {
        cars,
        runs,
        p50_us: pick(0.50),
        p99_us: p99,
        max_us: samples.last().copied().unwrap_or(0.0),
        mean_us: samples.iter().sum::<f64>() / samples.len().max(1) as f64,
        budget_us,
        within_budget: p99 < budget_us,
        model: model.display().to_string(),
        op_nodes,
    })
}
// ---------------------------------------------------------------------------
// Episode splits — owned by Rust so leakage is impossible, not merely checked
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Eval,
}

impl Split {
    fn from_i32(v: i32) -> Option<Split> {
        match v {
            0 => Some(Split::Train),
            1 => Some(Split::Eval),
            _ => None,
        }
    }
}

/// Records which split each seed episode belongs to, and refuses any claim
/// that would put one episode in both.
///
/// The rule is enforced here rather than in the trainer because the trainer is
/// the thing that would be trusted: a trainer that picks its own seeds can
/// leak by accident, and a leaked evaluation set makes every number it later
/// reports meaningless.
#[derive(Clone, Debug, Default)]
pub struct SplitLedger {
    claimed: Vec<(Split, u64)>,
}

impl SplitLedger {
    pub fn new() -> SplitLedger {
        SplitLedger::default()
    }

    pub fn claim(&mut self, split: Split, seed: u64) -> Result<(), String> {
        if let Some((s, _)) = self.claimed.iter().find(|(_, d)| *d == seed) {
            return Err(format!(
                "seed {seed} is already claimed by the {} split; an episode may not appear in both, \
                 or twice in one",
                match s {
                    Split::Train => "training",
                    Split::Eval => "evaluation",
                }
            ));
        }
        self.claimed.push((split, seed));
        Ok(())
    }

    pub fn seeds(&self, split: Split) -> Vec<u64> {
        self.claimed.iter().filter(|(s, _)| *s == split).map(|(_, d)| *d).collect()
    }

    pub fn clear(&mut self) {
        self.claimed.clear();
    }
}

// ---------------------------------------------------------------------------
// The C ABI Python drives
// ---------------------------------------------------------------------------
//
// Everything below is `extern "C"` over raw pointers. Two rules keep it safe
// under this crate's `panic = "abort"` release profile, where a panic would
// take the Python process down with no unwinding and no message:
//
//   1. no exported function may panic — every fallible path returns an error
//      code or a null pointer and records a message;
//   2. every `extern "C"` function takes raw types only, so no Rust destructor
//      can run across the boundary.

pub mod ffi {
    use super::{EnvSpec, EpisodeMetrics, Split, SplitLedger};
    use std::ffi::{c_char, c_float, c_int, CString};
    use std::sync::{LazyLock, Mutex};

    // Last error message, per thread, kept alive so the returned pointer
    // stays valid until the next FFI call on the same thread.
    thread_local! {
        static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
    }

    fn set_error(msg: impl Into<String>) {
        let s = msg.into().replace('\0', " ");
        LAST_ERROR.with(|c| *c.borrow_mut() = CString::new(s).ok());
    }

    /// The process-wide seed ledger. One ledger, so no caller can get a
    /// second one and hand itself a seed the first already owns.
    static LEDGER: LazyLock<Mutex<SplitLedger>> = LazyLock::new(|| Mutex::new(SplitLedger::new()));

    /// Observation width.
    #[no_mangle]
    pub extern "C" fn f1_obs_dim() -> usize {
        super::policy::OBS_DIM
    }

    /// Action width.
    #[no_mangle]
    pub extern "C" fn f1_action_dim() -> usize {
        super::policy::ACTION_DIM
    }

    /// The default environment spec as JSON. Python reads this rather than
    /// hard-coding the definition on its side.
    #[no_mangle]
    pub extern "C" fn f1_env_spec_default_json() -> *const c_char {
        let json = serde_json::to_string_pretty(&EnvSpec::default())
            .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
        store(&json)
    }

    /// Create an environment from a JSON [`EnvSpec`]. Null on failure; call
    /// `f1_last_error` for the reason.
    ///
    /// # Safety
    ///
    /// spec_json is read as a NUL-terminated UTF-8 string and must be valid
    /// for reads of that string. The returned handle is owned by the caller and
    /// must be released exactly once with [`f1_env_close`].
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_new(spec_json: *const c_char) -> *mut super::GymEnv {
        let text = match unsafe { read_str(spec_json) } {
            Ok(t) => t,
            Err(e) => {
                set_error(e);
                return std::ptr::null_mut();
            }
        };
        let spec: EnvSpec = match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(e) => {
                set_error(format!("env spec is not valid JSON for EnvSpec: {e}"));
                return std::ptr::null_mut();
            }
        };
        Box::into_raw(Box::new(super::GymEnv::new(spec)))
    }

    /// Reset, returning nothing; read the observation with `f1_env_obs`.
    #[no_mangle]
    pub extern "C" fn f1_env_reset(env: *mut super::GymEnv, seed: u64) -> c_int {
        with_env(env, |e| {
            e.reset(seed);
            0
        })
    }

    /// One action. Returns 0 on success, or -1 with `f1_last_error` set.
    ///
    /// # Safety
    ///
    /// env must be a live handle, and action must be valid for reads of
    /// `n` consecutive `f32` values, with `n` at least [`ACTION_DIM`].
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_step(
        env: *mut super::GymEnv,
        action: *const c_float,
        n: usize,
    ) -> c_int {
        if n < super::policy::ACTION_DIM || action.is_null() {
            set_error(format!(
                "action must have at least {} values",
                super::policy::ACTION_DIM
            ));
            return -1;
        }
        let slice = unsafe { std::slice::from_raw_parts(action, n) };
        with_env(env, |e| match e.step(slice) {
            Ok(_) => 0,
            Err(err) => {
                set_error(err);
                -1
            }
        })
    }

    /// Pointer to the current observation. Valid until the next call that
    /// changes it.
    ///
    /// # Safety
    ///
    /// env must be a live handle. The returned pointer borrows the
    /// environment: it is valid only until the next call that changes the
    /// observation (reset, step), so copy out of it before calling again.
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_obs(env: *const super::GymEnv) -> *const c_float {
        match unsafe { env.as_ref() } {
            Some(e) => e.observation().as_ptr(),
            None => {
                set_error("null environment handle");
                std::ptr::null()
            }
        }
    }

    /// The last step's terminated / truncated flags: bit 0 terminated,
    /// bit 1 truncated.
    ///
    /// # Safety
    ///
    /// env must be a live handle.
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_flags(env: *const super::GymEnv) -> c_int {
        match unsafe { env.as_ref() } {
            Some(e) => {
                let i = e.info();
                (i.terminated as c_int) | ((i.truncated as c_int) << 1)
            }
            None => {
                set_error("null environment handle");
                -1
            }
        }
    }

    /// Cumulative reward so far this episode.
    ///
    /// # Safety
    ///
    /// env must be a live handle.
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_total_reward(env: *const super::GymEnv) -> f64 {
        match unsafe { env.as_ref() } {
            Some(e) => e.total_reward(),
            None => {
                set_error("null environment handle");
                0.0
            }
        }
    }

    /// Episode metrics as JSON.
    ///
    /// # Safety
    ///
    /// env must be a live handle.
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_metrics_json(env: *const super::GymEnv) -> *const c_char {
        match unsafe { env.as_ref() } {
            Some(e) => store(&serde_json::to_string(&e.metrics()).unwrap_or_default()),
            None => {
                set_error("null environment handle");
                std::ptr::null()
            }
        }
    }

    /// Copy the episode log out. `obs_out`, `act_out` and `rew_out` must each
    /// hold `n * dims` room, or pass 0 to query the length first.
    ///
    /// # Safety
    ///
    /// env must be a live handle.
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_log_len(env: *const super::GymEnv) -> usize {
        match unsafe { env.as_ref() } {
            Some(e) => e.recorded().len(),
            None => {
                set_error("null environment handle");
                0
            }
        }
    }


    ///
    /// # Safety
    ///
    /// env must be a live handle, and out must be valid for writes of
    /// `cap` [`Transition`] values. Call [`f1_env_log_len`] first; the function
    /// writes nothing and returns -1 if `cap` is short.
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_log_copy(
        env: *const super::GymEnv,
        obs_out: *mut c_float,
        act_out: *mut c_float,
        rew_out: *mut f64,
        cap: usize,
    ) -> usize {
        let Some(e) = (unsafe { env.as_ref() }) else {
            set_error("null environment handle");
            return 0;
        };
        let log = e.recorded();
        let n = log.len().min(cap);
        let od = super::policy::OBS_DIM;
        let ad = super::policy::ACTION_DIM;
        for (i, t) in log.iter().take(n).enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    t.obs.as_ptr(),
                    obs_out.add(i * od),
                    od,
                );
                std::ptr::copy_nonoverlapping(
                    t.action.as_ptr(),
                    act_out.add(i * ad),
                    ad,
                );
                *rew_out.add(i) = t.reward;
            }
        }
        n
    }

    /// Discard the recorded episode, keeping the environment usable.
    #[no_mangle]
    pub extern "C" fn f1_env_clear_log(env: *mut super::GymEnv) -> c_int {
        with_env(env, |e| {
            e.take_log();
            0
        })
    }

    /// Car 0's action from the deterministic baseline driver, for the current
    /// state. This is the expert a behaviour-cloning trainer labels with: it
    /// comes from the engine, so the labels cannot drift from the behaviour
    /// they imitate.
    ///
    /// # Safety
    ///
    /// env must be a live handle, and out must be valid for writes of
    /// [`ACTION_DIM`] `f32` values.
    #[no_mangle]
    pub unsafe extern "C" fn f1_teacher_action(
        env: *mut super::GymEnv,
        out: *mut c_float,
        cap: usize,
    ) -> c_int {
        if out.is_null() || cap < super::policy::ACTION_DIM {
            set_error(format!(
                "teacher action needs room for {} values",
                super::policy::ACTION_DIM
            ));
            return -1;
        }
        with_env(env, |e| {
            // `baseline_control_for_car0` handles formation itself.
            let ctrl = e.race.baseline_control_for_car0();
            let max_steer = crate::physics::CarParams::default().max_steer;
            let norm = [
                (ctrl.steer_cmd / max_steer) as f32,
                if ctrl.brake > 0.0 {
                    -ctrl.brake as f32
                } else {
                    ctrl.throttle as f32
                },
                0.0f32,
            ];
            unsafe {
                std::ptr::copy_nonoverlapping(norm.as_ptr(), out, norm.len());
            }
            0
        })
    }


    ///
    /// # Safety
    ///
    /// env must be a handle from [`f1_env_new`] or null. After this call
    /// the handle is freed: using it again, including a second close, is
    /// use-after-free. Null is accepted and ignored so callers can close a
    /// failed creation unconditionally.
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_close(env: *mut super::GymEnv) {
        if !env.is_null() {
            // SAFETY: the handle came from Box::into_raw and is closed once.
            unsafe { drop(Box::from_raw(env)) };
        }
    }

    // ---- the split ledger ----

    /// Claim a seed for a split (`0` = train, `1` = eval). 0 on success,
    /// -1 if the seed is already owned — including by the other split.
    #[no_mangle]
    pub extern "C" fn f1_split_claim(which: c_int, seed: u64) -> c_int {
        let Some(split) = Split::from_i32(which) else {
            set_error(format!("split must be 0 (train) or 1 (eval), got {which}"));
            return -1;
        };
        match LEDGER.lock() {
            Ok(mut l) => match l.claim(split, seed) {
                Ok(()) => 0,
                Err(e) => {
                    set_error(e);
                    -1
                }
            },
            Err(_) => {
                set_error("split ledger is poisoned");
                -1
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn f1_split_count(which: c_int) -> isize {
        let Some(split) = Split::from_i32(which) else { return -1 };
        match LEDGER.lock() {
            Ok(l) => l.seeds(split).len() as isize,
            Err(_) => -1,
        }
    }

    /// The i-th seed of a split, so a caller can enumerate exactly what the
    /// ledger owns instead of inventing its own.
    #[no_mangle]
    pub extern "C" fn f1_split_seed(which: c_int, i: usize) -> u64 {
        let Some(split) = Split::from_i32(which) else { return 0 };
        match LEDGER.lock() {
            Ok(l) => l.seeds(split).get(i).copied().unwrap_or(0),
            Err(_) => 0,
        }
    }

    /// Release every claim. For tests; a training run never calls it.
    #[no_mangle]
    pub extern "C" fn f1_split_reset() {
        if let Ok(mut l) = LEDGER.lock() {
            l.clear();
        }
    }

    #[no_mangle]
    pub extern "C" fn f1_last_error() -> *const c_char {
        LAST_ERROR.with(|c| match &*c.borrow() {
            Some(s) => s.as_ptr(),
            None => std::ptr::null(),
        })
    }

    // ---- helpers ----

    fn store(s: &str) -> *const c_char {
        let owned = CString::new(s.replace('\0', " ")).unwrap_or_default();
        LAST_ERROR.with(|c| {
            *c.borrow_mut() = Some(owned);
            c.borrow().as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null())
        })
    }

    unsafe fn read_str(p: *const c_char) -> Result<String, String> {
        if p.is_null() {
            return Err("null string pointer".into());
        }
        let mut len = 0usize;
        while *p.add(len) != 0 {
            len += 1;
        }
        let bytes = std::slice::from_raw_parts(p as *const u8, len);
        String::from_utf8(bytes.to_vec()).map_err(|e| format!("argument is not UTF-8: {e}"))
    }

    /// Runs `f` on a live handle, turning a panic into an error code would
    /// require `catch_unwind`, which `panic = "abort"` forbids. So instead the
    /// bodies below are written not to panic, and the null check is the only
    /// validation they need.
    fn with_env(env: *mut super::GymEnv, f: impl FnOnce(&mut super::GymEnv) -> c_int) -> c_int {
        match unsafe { env.as_mut() } {
            Some(e) => f(e),
            None => {
                set_error("null environment handle");
                -1
            }
        }
    }

    /// Metrics for a finished episode, for callers that want to serialise
    /// without going through JSON.
    ///
    /// # Safety
    ///
    /// env must be a live handle.
    #[no_mangle]
    pub unsafe extern "C" fn f1_env_progress_m(env: *const super::GymEnv) -> f64 {
        match unsafe { env.as_ref() } {
            Some(e) => e.race().state.cars[0].total_progress,
            None => {
                set_error("null environment handle");
                0.0
            }
        }
    }

    /// Expose the metric type so the signature above cannot drift from it.
    #[allow(dead_code)]
    fn _assert_metrics_type(_: &EpisodeMetrics) {}
}
