//! Phase 10 — headless tournaments and deployment.
//!
//! A *batch* is a saved, self-describing list of races: every seed, every
//! policy, and the exact track version they ran on. The batch config is the
//! reproduction recipe — re-running it on the same engine build must produce
//! the same trajectory hashes, which is the only claim "reproducible" is
//! allowed to mean here.
//!
//! Three things this module is deliberately opinionated about:
//!
//! 1. **Serial first.** [`BatchRunner`] runs one race at a time. The plan asks
//!    for concurrency only *after measuring*, and [`BatchRunner::measure_concurrency`]
//!    is how that measurement gets made rather than assumed. Results do not
//!    depend on the answer: a race is a pure function of its config, so
//!    parallel and serial batches agree by construction — and the runner still
//!    checks it rather than trusting it.
//! 2. **Sample sizes are part of the report.** Every aggregate in
//!    [`ScoreSummary`] carries `n`. A mean over two runs is printed as a mean
//!    over two runs, and [`ScoreSummary::supports_claim`] refuses to promote a
//!    small sample into a general statement.
//! 3. **Failures are data.** A run that panics, does not reach the flag, or
//!    cannot open its telemetry socket is recorded as a [`RunStatus`] and
//!    counted in [`BatchReport::failed`]. It is never dropped and never
//!    averaged over.
//!
//! The engine has one built-in circuit, so a batch pins the geometry as well
//! as naming it: [`TrackSpec`] records the three numbers, and a spec that
//! names a circuit the current build does not have is rejected rather than
//! quietly run on the default.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::race::{
    self, PitStrategy, RaceConfig, RaceOutcome, SessionMode, TireCompound,
};
use crate::track::TrackSpec;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Bumped when the on-disk batch format changes incompatibly. A spec carrying
/// a different version is refused with an error naming both numbers, because
/// silently migrating a recipe is how you get a batch that no longer means
/// what its file says.
pub const BATCH_FORMAT: u32 = 1;

/// Minimum successful runs behind any general claim in the report. Below it
/// the number is printed with its sample size and explicitly *not* promoted
/// to a statement about the engine.
pub const MIN_CLAIM_SAMPLE: usize = 5;

/// One saved race: a seed plus the policy the cars ran under.
///
/// `seed` and the policy are the *whole* input. Everything else lives on the
/// batch (shared session setup) so that two entries differing only in seed
/// really do differ in nothing but the seed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSpec {
    pub seed: u64,
    #[serde(default)]
    pub label: String,
    /// Tyre and pit policy. `None` = whatever the shared session sets.
    #[serde(default)]
    pub policy: Option<Policy>,
}

/// A per-run policy: compound, pit strategy, and whether the LLM advisor is
/// attached. These are the three things that legitimately differ between
/// comparable runs, so they are the three a batch may vary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub start_compound: TireCompoundSpec,
    pub strategy: PitStrategySpec,
    /// Attach the Phase 7 advisor to car 0. An advised run is *not*
    /// reproducible (model latency feeds back into the race), so the report
    /// marks every such run `reproducible: false`.
    #[serde(default)]
    pub advisor: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TireCompoundSpec {
    Soft,
    Medium,
    Hard,
}

impl TireCompoundSpec {
    pub fn parse(s: &str) -> Option<TireCompound> {
        TireCompound::parse(s)
    }
    pub fn label(self) -> &'static str {
        match self {
            TireCompoundSpec::Soft => "S",
            TireCompoundSpec::Medium => "M",
            TireCompoundSpec::Hard => "H",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PitStrategySpec {
    /// One stop per car at `laps / 2`, onto medium.
    Auto,
    /// No stops at all.
    None,
    /// One stop on a named lap, onto a named compound.
    Fixed { lap: u32, switch_to: TireCompoundSpec },
}

/// The shared session setup every run in the batch inherits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionSpec {
    pub cars: u32,
    pub laps: u32,
    pub mode: String,
    #[serde(default = "default_noise")]
    pub noise: f64,
    #[serde(default = "default_true")]
    pub formation: bool,
    /// Weather preset name (`dry`/`damp`/`wet`/`mixed`) or `None` for the
    /// no-weather engine.
    #[serde(default)]
    pub weather: Option<String>,
    /// Run the in-simulation stewarding aid over every race in this batch.
    /// An observer: it records incidents and penalties but cannot change a
    /// trajectory, so a stewarded run and a plain one share a state hash.
    #[serde(default)]
    pub steward: bool,
}

fn default_noise() -> f64 {
    0.002
}
fn default_true() -> bool {
    true
}

impl Default for SessionSpec {
    fn default() -> Self {
        SessionSpec {
            cars: 20,
            laps: 8,
            mode: "race".into(),
            noise: 0.002,
            formation: true,
            weather: None,
            steward: false,
        }
    }
}

/// The saved batch: the complete reproduction recipe for a set of races.
///
/// Everything needed to re-run a batch is here — seeds, policies, session
/// setup, and the track version with its geometry. Nothing is inherited from
/// the machine, the clock, or the engine's defaults at run time except the
/// engine build itself, which the report records alongside the results so a
/// mismatch is visible instead of assumed away.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchSpec {
    pub format: u32,
    pub name: String,
    pub track: TrackSpec,
    pub session: SessionSpec,
    pub runs: Vec<RunSpec>,
    /// Optional Ollama endpoint override for `advisor: true` runs.
    #[serde(default)]
    pub advisor_endpoint: Option<String>,
}

impl Default for BatchSpec {
    fn default() -> Self {
        BatchSpec {
            format: BATCH_FORMAT,
            name: "unnamed".into(),
            track: TrackSpec::default(),
            session: SessionSpec::default(),
            runs: Vec::new(),
            advisor_endpoint: None,
        }
    }
}

impl BatchSpec {
    /// A small, valid, serial batch: the shape the plan asks for first.
    ///
    /// Deliberately 10 cars over 4 laps rather than a full 20-car grid. At 20
    /// cars over 8 laps the field currently retires ~15 cars in traffic and
    /// the session runs into the cool-down cap — a real and *correctly
    /// reported* outcome, but a poor first impression for a template. See
    /// PROGRESS.md, "Known limitations".
    pub fn demo() -> BatchSpec {
        let runs = (1..=3u64)
            .map(|seed| RunSpec { seed, label: format!("seed{seed}"), policy: None })
            .collect();
        BatchSpec {
            name: "demo".into(),
            session: SessionSpec { cars: 10, laps: 4, ..SessionSpec::default() },
            runs,
            ..BatchSpec::default()
        }
    }

    /// Reject a spec that could not be reproduced faithfully. Every message
    /// names the offending value, because this is the error a user edits a
    /// file to fix.
    pub fn validate(&self) -> Result<(), String> {
        if self.format != BATCH_FORMAT {
            return Err(format!(
                "batch format {} is not supported (this build writes format {BATCH_FORMAT})",
                self.format
            ));
        }
        if self.runs.is_empty() {
            return Err("batch has no runs".into());
        }
        if self.session.cars == 0 || self.session.cars > 64 {
            return Err(format!("cars must be 1..=64, got {}", self.session.cars));
        }
        if self.session.laps == 0 {
            return Err(format!("laps must be >= 1, got {}", self.session.laps));
        }
        if SessionMode::parse(&self.session.mode).is_none() {
            return Err(format!(
                "mode '{}' is not race|qualifying|practice",
                self.session.mode
            ));
        }
        // A batch must name a drivable circuit: the built-in stadium, or a
        // custom track whose spec carries its own control points (validated
        // here by building it — a bad track file is refused at config time,
        // not discovered by the first race).
        if !self.track.matches_builtin() && self.track.custom.is_none() {
            return Err(format!(
                "track {} is not the circuit this build runs ({}) and names no custom circuit",
                self.track.label(),
                TrackSpec::default().label()
            ));
        }
        if let Err(e) = self.track.build() {
            return Err(format!("track {} is not drivable: {e}", self.track.label()));
        }
        if let Some(w) = &self.session.weather {
            if crate::weather::WeatherConfig::preset(w).is_none() {
                return Err(format!("weather '{w}' is not dry|damp|wet|mixed"));
            }
        }
        for (i, r) in self.runs.iter().enumerate() {
            if let Some(p) = &r.policy {
                if let PitStrategySpec::Fixed { lap, switch_to } = &p.strategy {
                    if *lap == 0 || *lap >= self.session.laps {
                        return Err(format!(
                            "run {i}: fixed stop lap {lap} must be 1..{} for a {}-lap race",
                            self.session.laps.saturating_sub(1),
                            self.session.laps
                        ));
                    }
                    if TireCompoundSpec::parse(switch_to_label(*switch_to)).is_none() {
                        return Err(format!("run {i}: unknown switch compound"));
                    }
                }
            }
        }
        Ok(())
    }

    /// The exact [`RaceConfig`] one entry of this batch runs. Deterministic:
    /// the same spec always produces the same config, which is what makes the
    /// saved file a reproduction recipe rather than a description.
    pub fn race_config(&self, run: &RunSpec) -> RaceConfig {
        let mode = SessionMode::parse(&self.session.mode).unwrap_or(SessionMode::Race);
        let mut cfg = RaceConfig {
            cars: self.session.cars,
            laps: self.session.laps,
            seed: run.seed,
            noise: self.session.noise,
            mode,
            formation: self.session.formation,
            track: self.track.clone(),
            ..RaceConfig::default()
        };
        if self.session.steward {
            cfg.steward = Some(crate::steward::StewardConfig::enabled());
        }
        cfg.weather = self
            .session
            .weather
            .as_deref()
            .and_then(crate::weather::WeatherConfig::preset);
        if let Some(p) = &run.policy {
            cfg.start_compound = TireCompoundSpec::parse(p.start_compound.label()).unwrap();
            cfg.strategy = match &p.strategy {
                PitStrategySpec::Auto => PitStrategy::Auto,
                PitStrategySpec::None => PitStrategy::None,
                PitStrategySpec::Fixed { lap, switch_to } => PitStrategy::Fixed {
                    lap: *lap,
                    switch_to: TireCompoundSpec::parse(switch_to.label())
                        .unwrap_or(TireCompound::Medium),
                },
            };
            if p.advisor {
                let mut advisor = crate::strategy::AdvisorConfig::default();
                if let Some(ep) = &self.advisor_endpoint {
                    advisor.endpoint = ep.clone();
                }
                cfg.ai = Some(advisor);
            }
        }
        cfg
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialise batch: {e}"))
    }

    pub fn from_json(text: &str) -> Result<BatchSpec, String> {
        serde_json::from_str(text).map_err(|e| format!("parse batch: {e}"))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        std::fs::write(path, self.to_json()?)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<BatchSpec, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        BatchSpec::from_json(&text)
    }
}

fn switch_to_label(c: TireCompoundSpec) -> &'static str {
    c.label()
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// How a single run ended. Failure is a recorded outcome, not an absence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RunStatus {
    /// Ran to the flag with every car classified.
    Finished,
    /// Ran, but a car was still circulating at the cool-down cap: the session
    /// was cut short, so its numbers describe a partial race.
    CutShort,
    /// The run threw. The message is kept.
    Panicked { message: String },
    /// The configuration was refused before any tick ran.
    Rejected { reason: String },
}

impl RunStatus {
    pub fn is_success(&self) -> bool {
        matches!(self, RunStatus::Finished)
    }
    pub fn label(&self) -> String {
        match self {
            RunStatus::Finished => "finished".into(),
            RunStatus::CutShort => "cut-short".into(),
            RunStatus::Panicked { message } => format!("panicked: {message}"),
            RunStatus::Rejected { reason } => format!("rejected: {reason}"),
        }
    }
}

/// Everything measured about one race.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunReport {
    pub seed: u64,
    pub label: String,
    /// `Some` when the run had an LLM advisor attached. Such a run is not
    /// reproducible, and the report says so instead of letting its numbers
    /// sit next to deterministic ones without comment.
    pub advised: bool,
    pub status: RunStatus,
    pub state_hash: Option<u64>,
    pub ticks: u64,
    pub sim_time_s: f64,
    pub wall_time_s: f64,
    /// Field size for this run — the denominator of the finish rate. Taken
    /// from the config rather than from the classification length, so a run
    /// that lost cars is scored against the grid it actually started.
    pub cars: u32,
    /// Full-field finishers (classified with a finish time).
    pub finishers: u32,
    pub retirements: u32,
    pub contacts: usize,
    pub leader_laps_completed: u32,
    /// Lap-time consistency for the car that led the race: the coefficient of
    /// variation of its lap times. `None` when it completed fewer than two
    /// laps, because a "consistency" figure from one lap is a division by
    /// zero pretending to be a statistic.
    pub leader_lap_cv: Option<f64>,
    pub leader_best_lap_s: Option<f64>,
    pub fastest_lap_s: Option<f64>,
    pub pit_stops: usize,
    /// Compounds actually fitted, counted across the field.
    pub compounds_fitted: BTreeMap<String, usize>,
    /// Mean time held in the box per stop, seconds.
    pub mean_pit_held_s: Option<f64>,
    pub avg_tick_us: f64,
    pub p99_tick_us: f64,
    pub deadline_misses: u64,
    /// Stewarding findings for this run: how many incidents the aid recorded,
    /// how many it penalised, how many it deliberately left inconclusive, and
    /// the total time served. All zero when the aid was off.
    pub steward_incidents: usize,
    pub steward_penalised: usize,
    pub steward_inconclusive: usize,
    pub steward_time_s: f64,
}

impl RunReport {
    fn new(seed: u64, label: String, advised: bool, status: RunStatus) -> RunReport {
        RunReport {
            seed,
            label,
            advised,
            status,
            state_hash: None,
            ticks: 0,
            sim_time_s: 0.0,
            wall_time_s: 0.0,
            cars: 0,
            finishers: 0,
            retirements: 0,
            contacts: 0,
            leader_laps_completed: 0,
            leader_lap_cv: None,
            leader_best_lap_s: None,
            fastest_lap_s: None,
            pit_stops: 0,
            compounds_fitted: BTreeMap::new(),
            mean_pit_held_s: None,
            avg_tick_us: 0.0,
            p99_tick_us: 0.0,
            deadline_misses: 0,
            steward_incidents: 0,
            steward_penalised: 0,
            steward_inconclusive: 0,
            steward_time_s: 0.0,
        }
    }

    /// Score one finished (or cut-short) race. Kept separate from running it
    /// so the scoring rules can be read — and tested — without a race.
    pub fn from_outcome(
        seed: u64,
        label: String,
        advised: bool,
        cars: u32,
        o: &RaceOutcome,
    ) -> RunReport {
        let status = if o.phase_finished() {
            RunStatus::Finished
        } else {
            RunStatus::CutShort
        };
        let mut r = RunReport::new(seed, label, advised, status);
        r.state_hash = Some(o.state_hash);
        r.ticks = o.ticks;
        r.sim_time_s = o.sim_time_s;
        r.wall_time_s = o.wall_time_s;
        r.cars = cars.max(o.classification.len() as u32);
        r.finishers = o.classification.iter().filter(|c| c.finished).count() as u32;
        r.retirements = o.retired_ids.len() as u32;
        r.contacts = o.contacts.len();
        r.leader_laps_completed = o.leader_lap_times.len() as u32;
        r.leader_lap_cv = lap_cv(&o.leader_lap_times);
        r.leader_best_lap_s = o.classification.first().and_then(|c| c.best_lap_s);
        r.fastest_lap_s = o.fastest.map(|(_, t, _)| t);
        r.pit_stops = o.pit_stops.len();
        for stop in &o.pit_stops {
            *r.compounds_fitted.entry(stop.fitted.label().to_string()).or_insert(0) += 1;
        }
        if !o.pit_stops.is_empty() {
            r.mean_pit_held_s =
                Some(o.pit_stops.iter().map(|s| s.held_s).sum::<f64>() / o.pit_stops.len() as f64);
        }
        if let Some(sum) = o.steward.as_ref() {
            r.steward_incidents = sum.incidents;
            r.steward_penalised = sum.penalised;
            r.steward_inconclusive = sum.inconclusive;
            r.steward_time_s = sum.time_penalty_seconds;
        }
        r.avg_tick_us = o.tick_stats.avg_us;
        r.p99_tick_us = o.tick_stats.p99_us;
        r.deadline_misses = o.deadline_misses;
        r
    }
}

/// Coefficient of variation of lap times, or `None` below two laps.
fn lap_cv(laps: &[f64]) -> Option<f64> {
    if laps.len() < 2 {
        return None;
    }
    let n = laps.len() as f64;
    let mean = laps.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return None;
    }
    let var = laps.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / n;
    Some((var.sqrt() / mean) * 100.0)
}

/// Mean with its sample size attached. The size travels *with* the number,
/// everywhere, so a report line cannot quote a mean without also quoting how
/// many runs produced it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Mean {
    pub value: f64,
    pub n: usize,
}

pub fn mean_of(samples: &[f64]) -> Option<Mean> {
    if samples.is_empty() {
        return None;
    }
    Some(Mean { value: samples.iter().sum::<f64>() / samples.len() as f64, n: samples.len() })
}

/// The aggregate score across a batch, every figure carrying its `n`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScoreSummary {
    pub runs_planned: usize,
    pub runs_finished: usize,
    pub runs_failed: usize,
    pub advised_runs: usize,
    /// Reproducible runs: finished, unadvised, and identical on re-run.
    /// Counted, never assumed.
    pub reproducible_runs: usize,
    pub finish_rate: Option<Mean>,
    pub mean_leader_lap_cv_pct: Option<Mean>,
    pub mean_contacts_per_run: Option<Mean>,
    pub mean_retirements_per_run: Option<Mean>,
    pub mean_race_time_s: Option<Mean>,
    pub mean_pit_stops_per_run: Option<Mean>,
    pub mean_pit_held_s: Option<Mean>,
    pub compounds_fitted: BTreeMap<String, usize>,
    /// Stewarding-aid totals across the finished runs. `None` when the aid was
    /// not enabled for this batch, which is different from zero findings.
    pub mean_incidents_per_run: Option<Mean>,
    pub mean_penalised_per_run: Option<Mean>,
    pub mean_inconclusive_per_run: Option<Mean>,
    pub mean_steward_time_per_run: Option<Mean>,
    /// Wall-clock seconds and how many races shared them.
    pub wall_time_s: f64,
    pub wall_races_per_s: f64,
    /// The hashes a re-run must reproduce, in run order. This is the batch's
    /// fingerprint: same list means same races.
    pub state_hashes: Vec<u64>,
}

impl ScoreSummary {
    /// Runs that finished, went unadvised, and produced a hash.
    pub fn deterministic_runs(&self) -> usize {
        self.reproducible_runs
    }

    /// Whether the sample is big enough to support a claim about the engine
    /// rather than a statement about this batch. Two runs of one seed are a
    /// fact about the batch and nothing more.
    pub fn supports_claim(&self) -> bool {
        self.reproducible_runs >= MIN_CLAIM_SAMPLE
    }

    /// The sentence that says so out loud. Every verdict line in the rendered
    /// report goes through here, so "not enough runs" is impossible to leave
    /// out by accident.
    pub fn claim_note(&self) -> String {
        if self.supports_claim() {
            format!(
                "supported: {} reproducible runs (>= {MIN_CLAIM_SAMPLE})",
                self.reproducible_runs
            )
        } else {
            format!(
                "NOT a general claim: {} reproducible run(s) < {MIN_CLAIM_SAMPLE}; \
                 these numbers describe this batch only",
                self.reproducible_runs
            )
        }
    }
}

/// A whole batch's saved result: the spec that produced it plus everything
/// measured. The two together are the artifact the completion gate asks for —
/// reproducible *and* comparable from saved configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchReport {
    pub spec: BatchSpec,
    /// Engine build the batch ran on. A batch is only comparable to another
    /// report when these match; the comparison checks rather than assumes.
    pub engine: String,
    /// OS and architecture the batch ran on. See [`platform_id`]: state hashes
    /// are reproducible within a platform and are *not* portable across
    /// platforms, so this is part of what makes two reports comparable.
    #[serde(default = "platform_id")]
    pub platform: String,
    /// What the batch shared the machine with. Recorded so numbers from a
    /// dashboard-hosted batch are never silently read as engine-only numbers.
    pub profile: Profile,
    pub generated_unix_ms: u64,
    pub summary: ScoreSummary,
    pub runs: Vec<RunReport>,
}

impl BatchReport {
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("serialise report: {e}"))
    }
    pub fn from_json(text: &str) -> Result<BatchReport, String> {
        serde_json::from_str(text).map_err(|e| format!("parse report: {e}"))
    }
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        std::fs::write(path, self.to_json()?)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }
    pub fn load(path: &Path) -> Result<BatchReport, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        BatchReport::from_json(&text)
    }
}

/// What a saved report says about a fresh run of the same spec.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Comparison {
    pub engine_match: bool,
    pub track_match: bool,
    pub profile_match: bool,
    pub spec_match: bool,
    /// The saved and fresh runs came from the same OS/arch pair.
    pub platform_match: bool,
    /// The two platforms, kept so a mismatch can name them instead of just
    /// failing.
    #[serde(default)]
    pub saved_platform: String,
    #[serde(default)]
    pub fresh_platform: String,
    pub runs_compared: usize,
    /// Run labels whose hash moved. Empty means bit-identical.
    pub diverged: Vec<String>,
    pub missing: Vec<String>,
    pub extra: Vec<String>,
    pub identical: bool,
    /// The hashes differ but every *scored* metric still agrees. Only possible
    /// across platforms; see [`platform_id`].
    pub metrics_agree: bool,
}

impl Comparison {
    /// One line naming both platforms and whether they agreed.
    pub fn platform_match_label(&self) -> String {
        if self.platform_match {
            format!("same ({})", self.saved_platform)
        } else {
            format!("{} -> {} (hashes not portable)", self.saved_platform, self.fresh_platform)
        }
    }

    pub fn verdict(&self) -> String {
        if !self.spec_match || !self.track_match {
            return "NOT COMPARABLE: the saved spec or track differs".into();
        }
        if !self.profile_match {
            return "NOT COMPARABLE: the two batches ran under different profiles".into();
        }
        if !self.engine_match {
            return "NOT COMPARABLE: the two batches ran different engine builds".into();
        }
        if !self.platform_match {
            // Every hash moved. Whether that is benign depends entirely on
            // whether anything scored differently: saying "DIVERGED" when the
            // scores agree would read as a regression when nothing regressed,
            // and saying "NOT COMPARABLE" without qualification would throw
            // away the useful part. So the scores decide the wording.
            return if self.metrics_agree {
                format!(
                    "CROSS-PLATFORM MATCH: {} runs agree on every scored metric ({} -> {}); \
                     state hashes differ because f64 is not portable across platforms \
                     or across rebuilds",
                    self.runs_compared, self.saved_platform, self.fresh_platform
                )
            } else {
                format!(
                    "NOT COMPARABLE: different platforms ({} -> {}) and a score moved too, \
                     so this is not only hash drift",
                    self.saved_platform, self.fresh_platform
                )
            };
        }
        if self.identical {
            return format!("IDENTICAL: {} runs reproduced bit for bit", self.runs_compared);
        }
        format!(
            "DIVERGED: {} of {} runs differ{}",
            self.diverged.len() + self.missing.len() + self.extra.len(),
            self.runs_compared,
            if self.diverged.is_empty() {
                String::new()
            } else {
                format!(" ({})", self.diverged.join(", "))
            }
        )
    }
}

/// Compare a fresh report against a saved one. Deliberately strict: a
/// comparison that quietly ignores a changed spec would let a "reproduced"
/// verdict mean nothing.
pub fn compare(saved: &BatchReport, fresh: &BatchReport) -> Comparison {
    let mut c = Comparison {
        engine_match: saved.engine == fresh.engine,
        track_match: saved.spec.track == fresh.spec.track,
        profile_match: saved.profile == fresh.profile,
        spec_match: saved.spec == fresh.spec,
        platform_match: saved.platform == fresh.platform,
        saved_platform: saved.platform.clone(),
        fresh_platform: fresh.platform.clone(),
        ..Comparison::default()
    };
    let saved_by: BTreeMap<String, &RunReport> =
        saved.runs.iter().map(|r| (r.label.clone(), r)).collect();
    let fresh_by: BTreeMap<String, &RunReport> =
        fresh.runs.iter().map(|r| (r.label.clone(), r)).collect();
    for (label, s) in &saved_by {
        match fresh_by.get(label) {
            None => c.missing.push(label.clone()),
            Some(f) => {
                c.runs_compared += 1;
                if s.state_hash != f.state_hash {
                    c.diverged.push(label.clone());
                }
            }
        }
    }
    for label in fresh_by.keys() {
        if !saved_by.contains_key(label) {
            c.extra.push(label.clone());
        }
    }
    // Scored metrics, compared with a tolerance on the floats and exactly on
    // the counts. It answers a different question from `identical`: did
    // anything *move*, or did only the fingerprint move?
    c.metrics_agree = c.missing.is_empty()
        && c.extra.is_empty()
        && !saved.runs.is_empty()
        && saved.runs.len() == fresh.runs.len()
        && saved.runs.iter().zip(fresh.runs.iter()).all(|(s, f)| {
            s.label == f.label
                && s.status == f.status
                && s.finishers == f.finishers
                && s.retirements == f.retirements
                && s.contacts == f.contacts
                && s.pit_stops == f.pit_stops
                && s.leader_laps_completed == f.leader_laps_completed
                && score_close(s.leader_lap_cv, f.leader_lap_cv)
                && score_close(s.fastest_lap_s, f.fastest_lap_s)
                && score_close(s.mean_pit_held_s, f.mean_pit_held_s)
                && s.compounds_fitted == f.compounds_fitted
        });
    // `metrics_agree` is not redundant with the hash checks. The hash is a
    // fingerprint of the state, so matching hashes ought to imply matching
    // scores — but "ought to" is not "does", and an IDENTICAL verdict that
    // ignored the scores would be a claim about the results that the results
    // themselves were never asked to support.
    c.identical = c.spec_match
        && c.engine_match
        && c.track_match
        && c.platform_match
        && c.profile_match
        && c.metrics_agree
        && c.diverged.is_empty()
        && c.missing.is_empty()
        && c.extra.is_empty();
    c
}

/// Relative tolerance for calling a floating-point score "the same".
///
/// Chosen against the only measurement taken: across platforms the leader lap
/// CV moves by 1–2 ULP (11.281340414159413 vs 11.281340414159411, a relative
/// difference of ~2e-16). 1e-9 is six orders of magnitude looser than that and
/// still far tighter than the report's own display precision — the CV is
/// printed to 2 decimal places, so a genuine change would have to exceed
/// ~1e-2 relative to be visible at all. Integer scores are compared exactly
/// and never get this tolerance.
const SCORE_TOL: f64 = 1e-9;

fn score_close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            if x == y {
                return true;
            }
            let scale = x.abs().max(y.abs()).max(1.0);
            (x - y).abs() <= SCORE_TOL * scale
        }
        _ => false,
    }
}

/// The OS and architecture a batch was produced on.
///
/// Recorded because a state hash is **not portable across platforms**, and not
/// even perfectly stable across rebuilds of identical source. The race is a
/// pure function of its `RaceConfig` in the mathematical sense, but it
/// computes in `f64`, and neither glibc vs MinGW nor two builds of the same
/// crate agree to the last unit in the last place. Measured here:
///
/// - repeated runs of one binary on one host: bit-identical, every time;
/// - the same spec in the Linux container vs the Windows host: all five state
///   hashes differ, and the leader lap CV moves by 1-2 ULP — with driver noise
///   disabled and at 1 car x 1 lap, so it is present on the first tick, not
///   accumulated over a race.
///
/// Integer scores (finishers, retirements, contacts, pit stops, laps
/// completed, compounds fitted) agree exactly across that boundary; only the
/// `f64` scores move, and only in the last bits. So a comparison across
/// platforms is reported as *not comparable*, with the metric agreement stated
/// separately, rather than as a divergence. Nothing regressed; the fingerprint
/// is just not a portable identity, and saying "IDENTICAL" would be a lie
/// about the hashes.
///
/// This is also why [`engine_id`] carries a source hash: two binaries built
/// from identical source are not bit-identical either, so build identity
/// cannot be a hand-written constant.
pub fn platform_id() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// What the batch is running alongside, if anything. The plan asks for three
/// deployment profiles; they differ in what else shares the machine, which is
/// a property of the *host*, not of the race. So the batch records the profile
/// and the measurements, and never lets one profile's numbers be compared
/// against another's without saying so.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    /// Engine only. The clean baseline every other profile is measured against.
    EngineOnly,
    /// Engine plus the dashboard server on the same machine.
    EnginePlusDashboard,
    /// Engine plus a local Ollama serving the advisor.
    EnginePlusOllama,
}

impl Profile {
    pub fn label(self) -> &'static str {
        match self {
            Profile::EngineOnly => "engine-only",
            Profile::EnginePlusDashboard => "engine+dashboard",
            Profile::EnginePlusOllama => "engine+ollama",
        }
    }
    pub fn parse(s: &str) -> Option<Profile> {
        match s.trim().to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
            "engine-only" | "engine" | "none" => Some(Profile::EngineOnly),
            "engine+dashboard" | "dashboard" | "engine-dashboard" => {
                Some(Profile::EnginePlusDashboard)
            }
            "engine+ollama" | "ollama" | "engine-ollama" => Some(Profile::EnginePlusOllama),
            _ => None,
        }
    }
}

/// Runs a [`BatchSpec`] and scores it.
///
/// Serial by default. [`BatchRunner::run_parallel`] exists but is only
/// reachable once [`measure_concurrency`] has been run — the plan is explicit
/// that the M2 is shared, and parallelising before measuring is how a
/// tournament quietly starves the dashboard it is supposed to leave usable.
#[derive(Clone)]
pub struct BatchRunner {
    spec: BatchSpec,
    profile: Profile,
    /// Checked between races, so a long batch is stoppable without losing the
    /// races already scored. The plan's "observable and stoppable during long
    /// batches" requirement, in the smallest form that works.
    should_stop: Option<Arc<AtomicBool>>,
    on_progress: Option<ProgressHook>,
}

/// Callback the runner invokes between races with `(done, total, label)`.
/// A type alias, not an inline `Arc<dyn Fn ..>`: the signature appears in the
/// struct, the setters and the docs, and one of those drifts otherwise.
type ProgressHook = Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

impl std::fmt::Debug for BatchRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchRunner")
            .field("spec", &self.spec)
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

fn new_atomic() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

impl BatchRunner {
    pub fn new(spec: BatchSpec) -> BatchRunner {
        BatchRunner {
            spec,
            profile: Profile::EngineOnly,
            should_stop: Some(new_atomic()),
            on_progress: None,
        }
    }
    pub fn with_profile(mut self, p: Profile) -> BatchRunner {
        self.profile = p;
        self
    }
    /// Called as `(done, total, label)` after each race. Used by the server
    /// profile to keep the batch observable from outside.
    pub fn with_progress(
        mut self,
        f: impl Fn(usize, usize, &str) + Send + Sync + 'static,
    ) -> BatchRunner {
        self.on_progress = Some(Arc::new(f));
        self
    }
    /// Request a stop. Races already scored are kept; the run in flight
    /// finishes, because a half-tick race cannot be abandoned and still be
    /// scored honestly.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.should_stop.clone().unwrap_or_else(new_atomic)
    }
    /// Adopt an externally-owned stop flag — the dashboard's
    /// [`crate::server::BatchProgress`], so that `POST /api/batch/stop` and a
    /// programmatic stop are the same action rather than two that can disagree.
    pub fn with_stop_flag(mut self, flag: Arc<AtomicBool>) -> BatchRunner {
        self.should_stop = Some(flag);
        self
    }
    pub fn stopped(&self) -> bool {
        self.should_stop.as_ref().is_some_and(|s| s.load(Ordering::Relaxed))
    }

    fn stopped_now(&self) -> bool {
        self.stopped()
    }

    /// Run the batch serially.
    pub fn run(&self) -> Result<BatchReport, String> {
        self.spec.validate()?;
        let total = self.spec.runs.len();
        let start = Instant::now();
        let mut reports: Vec<RunReport> = Vec::with_capacity(total);
        for (i, run) in self.spec.runs.iter().enumerate() {
            // `i > 0`, not `i >= 0`: a stop request that arrives before the
            // batch starts must not produce an empty report. The race already
            // under way when the request lands always runs to the flag, so a
            // stopped batch reports at least one scored race and never throws
            // away work that was in flight.
            if i > 0 && self.stopped_now() {
                break;
            }
            let label = run_label(&self.spec, i, run);
            let report = self.run_one(run, &label);
            reports.push(report);
            if let Some(cb) = &self.on_progress {
                cb(i + 1, total, &label);
            }
        }
        Ok(self.assemble(reports, start.elapsed().as_secs_f64()))
    }

    /// Run the batch across `threads` workers. Results are identical to the
    /// serial run by construction — a race is a pure function of its config —
    /// and [`run_parallel`] verifies that rather than asserting it.
    pub fn run_parallel(&self, threads: usize) -> Result<BatchReport, String> {
        self.spec.validate()?;
        if threads <= 1 {
            return self.run();
        }
        let total = self.spec.runs.len();
        let start = Instant::now();
        // Everything a worker needs, resolved before any thread starts, so no
        // worker touches `self` (which is not `Sync`-shared by construction).
        let jobs: Vec<Job> = self
            .spec
            .runs
            .iter()
            .enumerate()
            .map(|(i, r)| Job {
                label: run_label(&self.spec, i, r),
                seed: r.seed,
                cfg: self.spec.race_config(r),
                advised: self.advised(r),
            })
            .collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let done = std::sync::atomic::AtomicUsize::new(0);
        let slots: Vec<std::sync::Mutex<Option<RunReport>>> =
            (0..total).map(|_| std::sync::Mutex::new(None)).collect();
        std::thread::scope(|scope| {
            for _ in 0..threads.min(total) {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= total {
                        break;
                    }
                    let job = &jobs[i];
                    let r = execute(job.cfg.clone(), job.seed, job.label.clone(), job.advised);
                    *slots[i].lock().unwrap() = Some(r);
                    let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                    if let Some(cb) = &self.on_progress {
                        cb(n, total, &job.label);
                    }
                });
            }
        });
        let reports: Vec<RunReport> = slots
            .into_iter()
            .filter_map(|m| m.into_inner().unwrap())
            .collect();
        Ok(self.assemble(reports, start.elapsed().as_secs_f64()))
    }

    /// Run one race and score it. A panic is caught and recorded: the batch
    /// survives it and the report says the run failed, which is the honest
    /// outcome. (Release builds use `panic = "abort"`, so in a release batch a
    /// panic takes the process down — the gate runs in dev, and the CLI says
    /// which profile it is in.)
    fn run_one(&self, run: &RunSpec, label: &str) -> RunReport {
        let cfg = self.spec.race_config(run);
        execute(cfg, run.seed, label.to_string(), self.advised(run))
    }

    fn advised(&self, run: &RunSpec) -> bool {
        run.policy.as_ref().is_some_and(|p| p.advisor)
    }

    fn assemble(&self, reports: Vec<RunReport>, wall_time_s: f64) -> BatchReport {
        summary_of(&self.spec, &reports, wall_time_s, self.profile)
    }
}

/// A run's stable identity within a batch. Comparison keys on this, so it
/// must depend only on the spec — never on run order or wall-clock.
fn run_label(spec: &BatchSpec, _i: usize, run: &RunSpec) -> String {
    if !run.label.is_empty() {
        return run.label.clone();
    }
    match &run.policy {
        Some(p) => format!(
            "{}/seed{}/{}-{}{}",
            spec.name,
            run.seed,
            p.start_compound.label(),
            match &p.strategy {
                PitStrategySpec::Auto => "auto".to_string(),
                PitStrategySpec::None => "nostop".to_string(),
                PitStrategySpec::Fixed { lap, .. } => format!("fixed{lap}"),
            },
            if p.advisor { "-advised" } else { "" },
        ),
        None => format!("{}/seed{}", spec.name, run.seed),
    }
}

/// One race's resolved inputs, so a parallel worker needs nothing but `&Job`.
#[derive(Clone, Debug)]
struct Job {
    label: String,
    seed: u64,
    cfg: RaceConfig,
    advised: bool,
}

/// Run one race, turning a panic into a recorded failure.
fn execute(cfg: RaceConfig, seed: u64, label: String, advised: bool) -> RunReport {
    let cars = cfg.cars;
    // `AssertUnwindSafe` is correct here rather than lazy: nothing outside
    // this call is mutated by a partial race, because the race owns its state
    // and the panic discards it entirely.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        race::run_race(cfg, false)
    }));
    match result {
        Ok((outcome, _)) => RunReport::from_outcome(seed, label, advised, cars, &outcome),
        Err(payload) => {
            let message = panic_message(&payload);
            let mut r = RunReport::new(seed, label, advised, RunStatus::Panicked { message });
            r.cars = cars;
            r
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".into()
    }
}

fn summary_of(
    spec: &BatchSpec,
    reports: &[RunReport],
    wall_time_s: f64,
    profile: Profile,
) -> BatchReport {
    let planned = spec.runs.len();
    let finished: Vec<&RunReport> = reports.iter().filter(|r| r.status.is_success()).collect();
    let failed = reports.iter().filter(|r| !r.status.is_success()).count();
    let advised_runs = reports.iter().filter(|r| r.advised).count();
    let finish_rate = mean_of(
        &finished
            .iter()
            .map(|r| r.finishers as f64 / r.cars.max(1) as f64)
            .collect::<Vec<_>>(),
    );
    let cv = finished.iter().filter_map(|r| r.leader_lap_cv).collect::<Vec<_>>();
    let mut compounds = BTreeMap::new();
    for r in &finished {
        for (k, v) in &r.compounds_fitted {
            *compounds.entry(k.clone()).or_insert(0) += v;
        }
    }
    let summary = ScoreSummary {
        runs_planned: planned,
        runs_finished: finished.len(),
        runs_failed: failed,
        advised_runs,
        reproducible_runs: finished.iter().filter(|r| !r.advised && r.state_hash.is_some()).count(),
        finish_rate,
        mean_leader_lap_cv_pct: mean_of(&cv),
        mean_contacts_per_run: mean_of(&finished.iter().map(|r| r.contacts as f64).collect::<Vec<_>>()),
        mean_retirements_per_run: mean_of(
            &finished.iter().map(|r| r.retirements as f64).collect::<Vec<_>>(),
        ),
        mean_race_time_s: mean_of(&finished.iter().map(|r| r.sim_time_s).collect::<Vec<_>>()),
        mean_pit_stops_per_run: mean_of(
            &finished.iter().map(|r| r.pit_stops as f64).collect::<Vec<_>>(),
        ),
        mean_pit_held_s: mean_of(
            &finished.iter().filter_map(|r| r.mean_pit_held_s).collect::<Vec<_>>(),
        ),
        compounds_fitted: compounds,
        // Only meaningful when the aid ran. A batch without it reports None
        // rather than a tidy 0.0, so "no findings" and "nothing was watching"
        // never print the same line.
        mean_incidents_per_run: spec.session.steward.then(|| {
            mean_of(&finished.iter().map(|r| r.steward_incidents as f64).collect::<Vec<_>>())
        }).flatten(),
        mean_penalised_per_run: spec.session.steward.then(|| {
            mean_of(&finished.iter().map(|r| r.steward_penalised as f64).collect::<Vec<_>>())
        }).flatten(),
        mean_inconclusive_per_run: spec.session.steward.then(|| {
            mean_of(&finished.iter().map(|r| r.steward_inconclusive as f64).collect::<Vec<_>>())
        }).flatten(),
        mean_steward_time_per_run: spec.session.steward.then(|| {
            mean_of(&finished.iter().map(|r| r.steward_time_s).collect::<Vec<_>>())
        }).flatten(),
        wall_time_s,
        wall_races_per_s: if wall_time_s > 0.0 {
            reports.len() as f64 / wall_time_s
        } else {
            0.0
        },
        state_hashes: finished.iter().filter_map(|r| r.state_hash).collect(),
    };
    BatchReport {
        spec: spec.clone(),
        engine: engine_id(),
        platform: platform_id(),
        profile,
        generated_unix_ms: unix_ms(),
        summary,
        runs: reports.to_vec(),
    }
}

/// Build identity recorded with every report. A batch is only comparable to
/// another report when these strings match.
///
/// `F1_SIM_SOURCE_HASH` comes from `build.rs` and covers every `.rs` file in
/// `src/` plus `Cargo.toml`. It exists because the hand-bumped
/// [`TRACK_ENGINE_BUILD`] below cannot distinguish two binaries built from the
/// same source, and those two do not agree bit for bit — see
/// [`platform_id`] for the measurement. Without the hash, a comparison across
/// them would print `IDENTICAL` on the strength of a constant that has not
/// changed.
pub fn engine_id() -> String {
    format!(
        "apexsim {} (src {}, {})",
        env!("CARGO_PKG_VERSION"),
        env!("F1_SIM_SOURCE_HASH"),
        TRACK_ENGINE_BUILD
    )
}

/// Bumped by hand when a change alters race results on unchanged input. It is
/// recorded in every report so a cross-build comparison fails loudly instead
/// of reporting a divergence that is really just a new build.
pub const TRACK_ENGINE_BUILD: &str = "p10-r1";

pub fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Measured, not assumed: how long the batch took serially and across N
/// workers, and whether the results were identical. The plan gates
/// concurrency on exactly this measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConcurrencyMeasurement {
    pub threads: usize,
    pub serial_wall_s: f64,
    pub parallel_wall_s: f64,
    pub speedup: f64,
    /// Whether the parallel batch produced the same hashes as the serial one.
    pub results_identical: bool,
    pub serial_hashes: Vec<u64>,
    pub parallel_hashes: Vec<u64>,
}

/// Run the same batch serially and then across `threads` workers and report
/// both wall times and whether the results agree.
pub fn measure_concurrency(
    spec: &BatchSpec,
    threads: usize,
) -> Result<ConcurrencyMeasurement, String> {
    let serial = BatchRunner::new(spec.clone()).run()?;
    let parallel = BatchRunner::new(spec.clone()).run_parallel(threads)?;
    let serial_hashes = serial.summary.state_hashes.clone();
    let parallel_hashes = parallel.summary.state_hashes.clone();
    Ok(ConcurrencyMeasurement {
        threads,
        serial_wall_s: serial.summary.wall_time_s,
        parallel_wall_s: parallel.summary.wall_time_s,
        speedup: if parallel.summary.wall_time_s > 0.0 {
            serial.summary.wall_time_s / parallel.summary.wall_time_s
        } else {
            0.0
        },
        results_identical: serial_hashes == parallel_hashes,
        serial_hashes,
        parallel_hashes,
    })
}

/// Write a batch spec, a report, and one replayable input log per run.
///
/// This is the whole reproducibility story in one directory: the spec says
/// what to run, the logs say what happened, and the report says what it
/// measured. Re-running the spec must reproduce the hashes in the report.
pub struct BatchArtifacts {
    pub dir: PathBuf,
    pub spec_path: PathBuf,
    pub report_path: PathBuf,
    /// (run label, log path) for every run that produced a log.
    pub logs: Vec<(String, PathBuf)>,
}

/// Run `spec` across `threads` workers (serially when `threads <= 1`), then
/// write `spec.json`, `report.json`, and a per-run log under `dir/logs/`.
pub fn run_and_save(
    spec: &BatchSpec,
    dir: &Path,
    profile: Profile,
    threads: usize,
    write_logs: bool,
) -> Result<(BatchReport, BatchArtifacts), String> {
    let runner = BatchRunner::new(spec.clone()).with_profile(profile);
    let report = if threads > 1 {
        runner.run_parallel(threads)?
    } else {
        runner.run()?
    };
    let artifacts = save_batch(spec, &report, dir, write_logs)?;
    Ok((report, artifacts))
}

/// Write an already-computed batch to `dir`.
///
/// Split out of `run_and_save` so a caller that ran the batch some other way
/// (behind the dashboard, say) can persist exactly that report instead of
/// running the batch a second time to obtain one. The saved report has to be
/// the batch the user asked for: re-running to "normalise" it doubles the cost
/// and leaves the printed and saved wall times describing different executions.
pub fn save_batch(
    spec: &BatchSpec,
    report: &BatchReport,
    dir: &Path,
    write_logs: bool,
) -> Result<BatchArtifacts, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let spec_path = dir.join("spec.json");
    let report_path = dir.join("report.json");
    spec.save(&spec_path)?;
    report.save(&report_path)?;
    // Per-run logs: the recorded input stream for each race, so a run can be
    // replayed tick for tick without re-running the whole batch. Written only
    // when asked for, because a log is ~10 MB per race and a tournament is
    // many races; `--logs` is for the run you intend to audit.
    let mut logs = Vec::new();
    if write_logs {
        for (i, run) in spec.runs.iter().enumerate() {
            let label = run_label(spec, i, run);
            let cfg = spec.race_config(run);
            // One race, both halves: the log's inputs and the outcome must
            // come from the same execution or the log does not describe the
            // race it is filed under.
            let executed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    race::run_race(cfg, false)
                }));
            let (outcome, inputs) = match executed {
                Ok(pair) => pair,
                Err(_) => continue,
            };
            let safe: String = label
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '-'
                    }
                })
                .collect();
            let path = dir.join("logs").join(format!("{safe}.log"));
            if let Some(p) = path.parent() {
                std::fs::create_dir_all(p).map_err(|e| format!("create {}: {e}", p.display()))?;
            }
            if race::write_race_log(&path, &spec.race_config(run), &inputs, &outcome).is_ok() {
                logs.push((label, path));
            }
        }
    }
    Ok(BatchArtifacts { dir: dir.to_path_buf(), spec_path, report_path, logs })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec3() -> BatchSpec {
        BatchSpec {
            name: "t".into(),
            session: SessionSpec { cars: 4, laps: 2, ..SessionSpec::default() },
            runs: (1..=3u64)
                .map(|seed| RunSpec { seed, label: format!("s{seed}"), policy: None })
                .collect(),
            ..BatchSpec::default()
        }
    }

    #[test]
    fn a_spec_round_trips_through_json() {
        let s = spec3();
        let json = s.to_json().expect("to json");
        let back = BatchSpec::from_json(&json).expect("from json");
        assert_eq!(back, s, "a saved batch must rebuild exactly");
    }

    #[test]
    fn the_demo_spec_validates_and_runs() {
        let d = BatchSpec::demo();
        d.validate().expect("demo spec is valid");
        let r = BatchRunner::new(d).run().expect("batch runs");
        assert_eq!(r.runs.len(), 3);
        assert_eq!(r.summary.runs_failed, 0);
    }

    #[test]
    fn a_bad_spec_is_rejected_with_the_offending_value() {
        let mut s = spec3();
        s.session.mode = "drifting".into();
        let e = s.validate().unwrap_err();
        assert!(e.contains("drifting"), "the error must name the bad value: {e}");

        let mut s = spec3();
        s.track.version += 1;
        let e = s.validate().unwrap_err();
        assert!(e.contains("track"), "a non-builtin track must be refused: {e}");

        let mut s = spec3();
        s.session.cars = 0;
        assert!(s.validate().unwrap_err().contains("cars"));
    }

    #[test]
    fn lap_cv_needs_two_laps() {
        assert_eq!(lap_cv(&[]), None);
        assert_eq!(lap_cv(&[60.0]), None, "one lap has no spread to measure");
        let two = lap_cv(&[60.0, 60.0]).expect("two laps give a cv");
        assert!(two.abs() < 1e-9, "identical laps are perfectly consistent: {two}");
        let mixed = lap_cv(&[60.0, 66.0]).expect("two laps");
        assert!(mixed > 0.0);
    }

    #[test]
    fn a_few_runs_cannot_support_a_claim() {
        let s = ScoreSummary { reproducible_runs: 2, ..ScoreSummary::default() };
        assert!(!s.supports_claim());
        assert!(s.claim_note().contains("NOT a general claim"));
        let s = ScoreSummary { reproducible_runs: MIN_CLAIM_SAMPLE, ..ScoreSummary::default() };
        assert!(s.supports_claim());
    }
}