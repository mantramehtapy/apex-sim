//! Phase 11 — in-simulation stewarding and incident records.
//!
//! **What this is.** A rule aid that watches a race, records what happened
//! around each incident, applies a small set of *stated* simulator rules, and
//! writes down the evidence, the rule version, the decision and the service for
//! every one of them. It is a *stewarding aid inside the simulation*, not a
//! rules authority: the official driving standards judge the whole incident and are
//! guidelines rather than a single apex-overlap formula, so nothing here claims
//! to reproduce a real stewards' decision. Every finding names the rule that
//! produced it, and any case the rules cannot separate is recorded
//! [`Decision::Inconclusive`] rather than being rounded into a fault finding.
//!
//! **Calibration, measured not guessed.** On this engine's default track
//! (half width 7.0 m) the drivers never take a car more than ~0.46 m past the
//! edge outside the pit lane — measured over 20-car/12-lap and 12-car/10-lap
//! races, with a continuous distribution and no gap in it. Everything further
//! out than that is the pit lane itself, which sits at a designed 7.5 m. So
//! the track-limit threshold of 1.00 m sits above what these cars currently do:
//! the shortcut and rejoin rules are reachable in the recorder and in the
//! judges, but they do not fire on this circuit with this driver model. That is
//! stated rather than papered over, and lowering the threshold to make them
//! fire would penalise cars running half a metre wide in normal racing.
//!
//! **What this is not.** It is not a threshold that stands in for a human. The
//! ambiguity branches are load-bearing: a closing speed inside the stated band,
//! or two cars closing on each other with no margin between them, are *not*
//! findings, and the code is arranged so those branches cannot be skipped.
//!
//! **Determinism.** Everything here is a pure function of the tick stream. The
//! recorder is an observer: it is reachable from nowhere in the race's state
//! hash, so a stewarded race drives exactly the same trajectory as an
//! un-stewarded one and both hash identically. A run log records the rule set,
//! the thresholds and every decision, and a replay re-derives them from the same
//! states and fails loudly if the evidence or the outcome moved.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::race::ContactEvent;

/// Identifies the rule set. Bumped by hand when a stated rule changes meaning;
/// the thresholds travel separately in the run log, so a decision is always
/// reproducible from the file that recorded it.
pub const RULE_SET: &str = "apexsim-steward";
pub const RULE_SET_VERSION: u32 = 1;

/// Half the overlap distance used for the "inside the contact circle" figure.
/// Mirrors `race::CONTACT_DIST`, which is what actually resolves a contact.
const CONTACT_DIST_M: f64 = 3.0;

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

/// The three incidents this aid has an explicit rule for. Every [`Incident`]
/// names one of these, and every one of these has a written statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    /// Two cars touched and at least one was closing on the other.
    Contact,
    /// A car ran beyond the track edge for long enough, or far enough, to matter.
    Shortcut,
    /// A car that had been off the road came back into the path of another.
    UnsafeRejoin,
}

impl Rule {
    pub const ALL: [Rule; 3] = [Rule::Contact, Rule::Shortcut, Rule::UnsafeRejoin];

    pub fn id(self) -> &'static str {
        match self {
            Rule::Contact => "contact",
            Rule::Shortcut => "shortcut",
            Rule::UnsafeRejoin => "unsafe-rejoin",
        }
    }

    pub fn parse(s: &str) -> Option<Rule> {
        Rule::ALL.into_iter().find(|r| r.id() == s)
    }

    /// Per-rule version, so a penalty links to one exact rule and not merely to
    /// "the rule set".
    pub fn version(self) -> u32 {
        1
    }

    pub fn title(self) -> &'static str {
        match self {
            Rule::Contact => "Contact causing a collision",
            Rule::Shortcut => "Track limits / shortcut",
            Rule::UnsafeRejoin => "Unsafe rejoin",
        }
    }

    /// The rule in words, with this run's thresholds substituted in. This text
    /// is what makes a decision auditable: a reviewer can read the sentence the
    /// engine applied, not a rule number they have to look up.
    pub fn statement(self, cfg: &StewardConfig) -> String {
        match self {
            Rule::Contact => format!(
                "A contact whose closing speed is below {:.2} m/s is not an incident. \
                 A closing speed within {:.2} m/s of the {:.2} m/s threshold is INCONCLUSIVE — \
                 this aid declines to find fault there. Above the band, the car closing on the \
                 other is at fault only if it exceeds the other's closing component by at \
                 least {:.2} m/s; otherwise the case is INCONCLUSIVE. Fault is a {:.2} s time \
                 penalty, served at the race end.",
                cfg.contact_cs_penalty,
                cfg.contact_cs_inconclusive,
                cfg.contact_cs_penalty,
                cfg.contact_fault_margin,
                cfg.penalty_seconds,
            ),
            Rule::Shortcut => format!(
                "Time beyond the track edge is measured per car. A car that spends at least {} \
                 ticks ({:.2} s) at least {:.2} m beyond the edge has taken a shortcut and \
                 receives a {:.2} s time penalty, served at the race end. The duration and the \
                 distance are measured over the same stretch, so a car that only clips the edge \
                 for a moment raises no case at all. \
                 An excursion while RETIRED, or inside the pit lane, is NOT a shortcut: the lane \
                 is a separate road, and a retired car is a hazard rather than a competitor.",
                cfg.track_limit_ticks,
                cfg.track_limit_ticks as f64 * crate::sim::DT,
                cfg.track_limit_dist_m,
                cfg.penalty_seconds,
            ),
            Rule::UnsafeRejoin => format!(
                "A car that was at least {:.2} m beyond the track edge for at least {} ticks ({:.2} s) and then returns is evaluated over a {:.2} s review window. If another car closes on it to within {:.2} m at {:.2} m/s or more AND that car still had {:.2} m of room at the moment of the rejoin (so it could have steered away), the rejoining car receives a {:.2} s time penalty. The room is measured at the rejoin and the closing rate at the closest approach; each number names its own time. A car in the pit lane does not count as closing: it is on a separate road. If the closing car was already alongside, or itself off the road, the case is INCONCLUSIVE — this aid does not decide which of two committed cars should have moved.",
                cfg.track_limit_dist_m,
                cfg.rejoin_min_off_ticks,
                cfg.rejoin_min_off_ticks as f64 * crate::sim::DT,
                cfg.review_s,
                cfg.rejoin_avoidable_gap_m,
                cfg.rejoin_danger_cs,
                cfg.rejoin_avoidable_gap_m,
                cfg.penalty_seconds,
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration — the thresholds ARE the rule, so they are logged
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StewardConfig {
    pub enabled: bool,
    /// Seconds of car state kept *before* an incident as evidence.
    pub history_s: f64,
    /// Seconds kept *after* an incident before the rule is evaluated. A rejoin
    /// cannot honestly be judged before the rejoining car has had time to place
    /// itself, so the decision is deferred by this much rather than guessed at
    /// the instant of the contact.
    pub review_s: f64,
    /// Closing speed at or above which a contact is notable (m/s).
    pub contact_cs_penalty: f64,
    /// Half-width of the band around that threshold where no finding is made.
    pub contact_cs_inconclusive: f64,
    /// Difference in closing components needed to name one car at fault (m/s).
    pub contact_fault_margin: f64,
    /// Ticks beyond the edge that count as an excursion.
    pub track_limit_ticks: u64,
    /// Metres beyond the edge that count as an excursion.
    pub track_limit_dist_m: f64,
    /// Ticks off the road before a return counts as a rejoin.
    pub rejoin_min_off_ticks: u64,
    /// Closing speed on the rejoining car that makes the return dangerous (m/s).
    pub rejoin_danger_cs: f64,
    /// Separation at the rejoin above which the other car could have avoided it.
    pub rejoin_avoidable_gap_m: f64,
    /// The time penalty this aid imposes, in seconds.
    pub penalty_seconds: f64,
}

impl Default for StewardConfig {
    /// Off by default and inert: a race with no `--steward` is bit-identical to
    /// the same race without this phase, which is what keeps every earlier gate
    /// and artifact valid.
    fn default() -> Self {
        StewardConfig {
            enabled: false,
            history_s: 3.0,
            review_s: 2.0,
            contact_cs_penalty: 8.0,
            contact_cs_inconclusive: 2.0,
            contact_fault_margin: 3.0,
            track_limit_ticks: 25,
            track_limit_dist_m: 1.0,
            rejoin_min_off_ticks: 25,
            rejoin_danger_cs: 6.0,
            rejoin_avoidable_gap_m: 12.0,
            penalty_seconds: 5.0,
        }
    }
}

impl StewardConfig {
    /// The configuration the `--steward` flag turns on.
    pub fn enabled() -> Self {
        StewardConfig { enabled: true, ..StewardConfig::default() }
    }

    /// An unconfigured or explicitly-disabled aid is normalised away, so
    /// `Some(inert)` and `None` cannot mean two different engines.
    pub fn is_inert(&self) -> bool {
        !self.enabled
    }

    pub fn history_ticks(&self) -> u64 {
        (self.history_s / crate::sim::DT).round() as u64
    }

    pub fn review_ticks(&self) -> u64 {
        (self.review_s / crate::sim::DT).round() as u64
    }
}

// ---------------------------------------------------------------------------
// Outcome vocabulary
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// A stated rule was met and a penalty follows.
    Penalised,
    /// A stated rule was considered and did not fire.
    NoPenalty,
    /// The rule could not separate the cars on the recorded evidence. Carries
    /// no penalty, by construction: an ambiguous case must never quietly become
    /// a certain fault finding.
    Inconclusive,
}

impl Decision {
    pub fn label(self) -> &'static str {
        match self {
            Decision::Penalised => "PENALISED",
            Decision::NoPenalty => "NO PENALTY",
            Decision::Inconclusive => "INCONCLUSIVE",
        }
    }

    pub fn parse(s: &str) -> Option<Decision> {
        match s {
            "penalised" => Some(Decision::Penalised),
            "no-penalty" => Some(Decision::NoPenalty),
            "inconclusive" => Some(Decision::Inconclusive),
            _ => None,
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Decision::Penalised => "penalised",
            Decision::NoPenalty => "no-penalty",
            Decision::Inconclusive => "inconclusive",
        }
    }
}

/// What the aid imposes. Kept deliberately small: the simulator has no
/// drive-through path and does not pretend to have one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PenaltyKind {
    None,
    Reprimand,
    Time,
}

impl PenaltyKind {
    pub fn label(self) -> &'static str {
        match self {
            PenaltyKind::None => "none",
            PenaltyKind::Reprimand => "reprimand",
            PenaltyKind::Time => "time",
        }
    }

    pub fn parse(s: &str) -> Option<PenaltyKind> {
        match s {
            "none" => Some(PenaltyKind::None),
            "reprimand" => Some(PenaltyKind::Reprimand),
            "time" => Some(PenaltyKind::Time),
            _ => None,
        }
    }

    pub fn code(self) -> &'static str {
        self.label()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Penalty {
    pub kind: PenaltyKind,
    pub seconds: f64,
}

impl Penalty {
    pub fn none() -> Self {
        Penalty { kind: PenaltyKind::None, seconds: 0.0 }
    }

    pub fn time(seconds: f64) -> Self {
        Penalty { kind: PenaltyKind::Time, seconds }
    }

    pub fn reprimand() -> Self {
        Penalty { kind: PenaltyKind::Reprimand, seconds: 0.0 }
    }
}

/// Where the penalty is served. Recorded, not invented: this simulator has no
/// drive-through loop and no post-race jury, so a time penalty is served at the
/// race end and nothing pretends otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServicePoint {
    NotServed,
    AtRaceEnd,
    NextPitEntry,
}

impl ServicePoint {
    pub fn label(self) -> &'static str {
        match self {
            ServicePoint::NotServed => "not-served",
            ServicePoint::AtRaceEnd => "at-race-end",
            ServicePoint::NextPitEntry => "next-pit-entry",
        }
    }

    pub fn parse(s: &str) -> Option<ServicePoint> {
        match s {
            "not-served" => Some(ServicePoint::NotServed),
            "at-race-end" => Some(ServicePoint::AtRaceEnd),
            "next-pit-entry" => Some(ServicePoint::NextPitEntry),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

/// Everything the rules are allowed to look at for one car, over one window.
/// The rules read *only* this — which is what makes an incident auditable and
/// keeps a rule from quietly consulting live race state it cannot reproduce.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarEvidence {
    pub car: u32,
    pub speed_mps: f64,
    pub max_speed_mps: f64,
    pub lateral_m: f64,
    /// Worst distance beyond the track edge over the window (0 = never off).
    pub max_excursion_m: f64,
    /// Ticks spent beyond the edge over the window.
    pub off_track_ticks: u64,
    /// Separation to the other car at the incident tick (m).
    pub gap_m: f64,
    /// How far inside the contact circle the pair was at the incident tick (m).
    pub overlap_m: f64,
    /// Component of this car's velocity directed at the other car (m/s).
    pub closing_into_other_mps: f64,
    pub retired: bool,
    pub in_pit: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Evidence {
    pub rule: Rule,
    pub incident_tick: u64,
    pub sim_time_s: f64,
    /// The recorded window: `[window_start_tick, window_end_tick]`. An incident
    /// is reviewed over what came *before* and after it, not at one instant.
    pub window_start_tick: u64,
    pub window_end_tick: u64,
    pub a: CarEvidence,
    pub b: Option<CarEvidence>,
    /// Closing speed of the pair as a whole (m/s).
    pub closing_speed_mps: f64,
    /// How much more one car closed on the other than the reverse (m/s). Small
    /// means the evidence does not separate them.
    pub fault_margin_mps: f64,
}

impl Evidence {
    pub fn fault_margin(&self) -> f64 {
        self.fault_margin_mps.abs()
    }

    pub fn rule_version(&self) -> u32 {
        self.rule.version()
    }
}

/// What a rule decided, and why in words.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub decision: Decision,
    pub penalty: Penalty,
    /// The car that serves the penalty, if any.
    pub served_to: Option<u32>,
    pub service: ServicePoint,
    pub basis: String,
}

impl Finding {
    fn none(decision: Decision, basis: String) -> Finding {
        // A structural guarantee, not a convention: no decision other than
        // Penalised can ever carry a penalty, so an inconclusive case cannot
        // acquire one further down the pipeline.
        debug_assert!(decision != Decision::Penalised);
        Finding { decision, penalty: Penalty::none(), served_to: None, service: ServicePoint::NotServed, basis }
    }
}

// ---------------------------------------------------------------------------
// The rules
// ---------------------------------------------------------------------------

/// Contact causing a collision. See [`Rule::Contact`] for the statement.
pub fn judge_contact(ev: &Evidence, cfg: &StewardConfig) -> Finding {
    let cs = ev.closing_speed_mps;
    let margin = ev.fault_margin();
    // A contact incident always names two cars. If the evidence ever arrives
    // with only one there is nothing to attribute the closing speed to, and
    // "cannot attribute it" is the honest answer — not a finding by default.
    let Some(b) = ev.b.as_ref() else {
        return Finding::none(
            Decision::Inconclusive,
            format!(
                "the incident record names one car only, so the {:.2} m/s closing speed cannot \
                 be attributed to either of them",
                cs
            ),
        );
    };
    let nums = format!(
        "closing {:.2} m/s against a {:.2} m/s threshold with a +-{:.2} m/s no-finding band; \
         fault margin {:.2} m/s against a required {:.2} m/s; overlap {:.2} m into a {:.1} m \
         contact circle",
        cs, cfg.contact_cs_penalty, cfg.contact_cs_inconclusive, margin, cfg.contact_fault_margin,
        ev.a.overlap_m, CONTACT_DIST_M
    );
    // Below the band: rubbing, not an incident.
    if cs + cfg.contact_cs_inconclusive <= cfg.contact_cs_penalty {
        return Finding::none(
            Decision::NoPenalty,
            format!("closing speed is below the no-finding band: {nums}"),
        );
    }
    // Inside the band: explicitly not a finding.
    if cs - cfg.contact_cs_inconclusive < cfg.contact_cs_penalty {
        return Finding::none(
            Decision::Inconclusive,
            format!(
                "closing speed sits inside the +-{:.2} m/s band around the {:.2} m/s threshold, \
                 which this rule defines as not a finding: {nums}",
                cfg.contact_cs_inconclusive, cfg.contact_cs_penalty
            ),
        );
    }
    // Clearly notable, but the evidence may still not say which car did it.
    if margin < cfg.contact_fault_margin {
        return Finding::none(
            Decision::Inconclusive,
            format!(
                "both cars were closing on each other ({nums}); the evidence does not separate \
                 them, so this aid records no fault"
            ),
        );
    }
    let culprit = if ev.a.closing_into_other_mps >= b.closing_into_other_mps {
        ev.a.car
    } else {
        b.car
    };
    Finding {
        decision: Decision::Penalised,
        penalty: Penalty::time(cfg.penalty_seconds),
        served_to: Some(culprit),
        service: ServicePoint::AtRaceEnd,
        basis: format!("car {culprit} was the closing party by a clear margin: {nums}"),
    }
}

/// Track limits. See [`Rule::Shortcut`] for the statement.
pub fn judge_shortcut(ev: &Evidence, cfg: &StewardConfig) -> Finding {
    let a = ev.a;
    let nums = format!(
        "car {} spent {} ticks ({:.2} s) beyond the edge and reached {:.3} m beyond it, against \
         thresholds of {} ticks ({:.2} s) and {:.2} m",
        a.car,
        a.off_track_ticks,
        a.off_track_ticks as f64 * crate::sim::DT,
        a.max_excursion_m,
        cfg.track_limit_ticks,
        cfg.track_limit_ticks as f64 * crate::sim::DT,
        cfg.track_limit_dist_m,
    );
    if a.in_pit {
        return Finding::none(
            Decision::NoPenalty,
            format!("the car was in the pit lane, which is a separate road, not the racing surface: {nums}"),
        );
    }
    if a.retired {
        return Finding::none(
            Decision::NoPenalty,
            format!("the car was already retired, so it is a hazard rather than a competitor: {nums}"),
        );
    }
    if a.off_track_ticks >= cfg.track_limit_ticks && a.max_excursion_m >= cfg.track_limit_dist_m {
        return Finding {
            decision: Decision::Penalised,
            penalty: Penalty::time(cfg.penalty_seconds),
            served_to: Some(a.car),
            service: ServicePoint::AtRaceEnd,
            basis: format!("both track-limits thresholds were met: {nums}"),
        };
    }
    Finding::none(
        Decision::NoPenalty,
        format!("at least one track-limits threshold was not met: {nums}"),
    )
}

/// Unsafe rejoin. See [`Rule::UnsafeRejoin`] for the statement.
pub fn judge_rejoin(ev: &Evidence, cfg: &StewardConfig) -> Finding {
    let a = &ev.a;
    if a.off_track_ticks < cfg.rejoin_min_off_ticks {
        return Finding::none(
            Decision::NoPenalty,
            format!(
                "the car was not beyond the track edge for long enough to have rejoined:                  {} ticks off the road against a {} tick minimum",
                a.off_track_ticks, cfg.rejoin_min_off_ticks
            ),
        );
    }
    let Some(b) = ev.b.as_ref() else {
        return Finding::none(
            Decision::NoPenalty,
            "no car closed on the rejoining car during the review window".to_string(),
        );
    };
    let closing = b.closing_into_other_mps;
    let nums = format!(
        "car {} rejoined after {} ticks ({:.2} s) off the road with car {} {:.2} m back at that          moment; over the review window car {} came within {:.2} m while closing at {:.2} m/s          against a {:.2} m/s threshold (and was itself off the road for {} ticks)",
        a.car,
        a.off_track_ticks,
        a.off_track_ticks as f64 * crate::sim::DT,
        b.car,
        a.gap_m,
        b.car,
        b.gap_m,
        closing,
        cfg.rejoin_danger_cs,
        b.off_track_ticks,
    );
    if closing < cfg.rejoin_danger_cs {
        return Finding::none(
            Decision::NoPenalty,
            format!("no car closed on the rejoining car quickly enough to call the return unsafe: {nums}"),
        );
    }
    // Avoidable turns on the room at the *rejoin*, not on the closest
    // approach. A car still far back when the joining car came back is a car
    // that could have moved; a car already alongside is not at fault for being
    // alongside. Reading the closest approach here instead makes the two halves
    // of the rule contradictory — a car cannot be both within 12 m and 12 m
    // clear — and the rule can then never penalise anything.
    let avoidable = a.gap_m >= cfg.rejoin_avoidable_gap_m && b.off_track_ticks == 0;
    if !avoidable {
        return Finding::none(
            Decision::Inconclusive,
            format!(
                "at the rejoin car {} was only {:.2} m back and was {} itself off the road, so                  the evidence does not show it could have avoided the return; this aid records                  no fault: {nums}",
                b.car,
                a.gap_m,
                if b.off_track_ticks > 0 { "also" } else { "not" }
            ),
        );
    }
    Finding {
        decision: Decision::Penalised,
        penalty: Penalty::time(cfg.penalty_seconds),
        served_to: Some(a.car),
        service: ServicePoint::AtRaceEnd,
        basis: format!(
            "car {} rejoined into a car that still had {:.2} m of room to steer away: {nums}",
            a.car, a.gap_m
        ),
    }
}

pub fn judge(ev: &Evidence, cfg: &StewardConfig) -> Finding {
    match ev.rule {
        Rule::Contact => judge_contact(ev, cfg),
        Rule::Shortcut => judge_shortcut(ev, cfg),
        Rule::UnsafeRejoin => judge_rejoin(ev, cfg),
    }
}

// ---------------------------------------------------------------------------
// Incidents
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Incident {
    pub rule: Rule,
    pub rule_version: u32,
    pub tick: u64,
    pub sim_time_s: f64,
    pub car_a: u32,
    pub car_b: Option<u32>,
    pub decision: Decision,
    pub penalty: Penalty,
    pub served_to: Option<u32>,
    pub service: ServicePoint,
    pub basis: String,
    pub evidence: Evidence,
    pub evidence_hash: u64,
}

impl Incident {
    /// A stable digest of the evidence *and* the outcome. A replay compares
    /// this, so a divergence in either the recorded facts or the rule applied to
    /// them fails loudly instead of passing as a different-but-valid run.
    pub fn compute_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x1000_0000_01b3);
            }
        };
        mix(self.tick);
        mix(self.rule as u64);
        mix(self.rule_version as u64);
        mix(self.car_a as u64);
        mix(self.car_b.unwrap_or(u32::MAX) as u64);
        mix(self.decision as u64);
        mix(self.penalty.kind as u64);
        mix(self.penalty.seconds.to_bits());
        mix(self.served_to.unwrap_or(u32::MAX) as u64);
        mix(self.service as u64);
        for c in [Some(self.evidence.a), self.evidence.b] {
            match c {
                Some(c) => {
                    mix(c.car as u64);
                    mix(c.speed_mps.to_bits());
                    mix(c.max_speed_mps.to_bits());
                    mix(c.max_excursion_m.to_bits());
                    mix(c.off_track_ticks);
                    mix(c.gap_m.to_bits());
                    mix(c.overlap_m.to_bits());
                    mix(c.closing_into_other_mps.to_bits());
                    mix(c.retired as u64);
                    mix(c.in_pit as u64);
                }
                None => mix(u64::MAX),
            }
        }
        mix(self.evidence.window_start_tick);
        mix(self.evidence.window_end_tick);
        h
    }
}

/// The digest over a whole incident list, in order.
/// The `D` header line: rule set, version, incident count, record hash, and
/// every threshold any decision rests on. Written here rather than at the call
/// site so the reader and the writer cannot drift apart.
pub fn header_line(incidents: &[Incident], cfg: &StewardConfig) -> String {
    format!(
        "{} {} {} {:016x} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {} {} {:.17e} {:.17e} {:.17e}",
        RULE_SET,
        RULE_SET_VERSION,
        incidents.len(),
        incidents_hash(incidents),
        cfg.history_s,
        cfg.review_s,
        cfg.contact_cs_penalty,
        cfg.contact_cs_inconclusive,
        cfg.contact_fault_margin,
        cfg.track_limit_dist_m,
        cfg.track_limit_ticks,
        cfg.rejoin_min_off_ticks,
        cfg.rejoin_danger_cs,
        cfg.rejoin_avoidable_gap_m,
        cfg.penalty_seconds,
    )
}

/// One `P` line: an incident's digest and outcome, as the reviewer sees it.
///
/// This exists so that a replay can *compare* rather than merely count. A
/// reader that checks the aggregate hash of its own re-derived record against
/// the aggregate hash in the header will happily accept a log whose per-
/// incident lines were edited, because editing them does not change what the
/// replay concludes. Comparing the rendered line field by field is what makes
/// the recorded decisions load-bearing.
pub fn incident_line(inc: &Incident) -> String {
    format!(
        "{} {} {} {} {} {} {} {:.17e} {} {} {:016x}",
        inc.tick,
        inc.rule.id(),
        inc.rule_version,
        inc.car_a,
        inc.car_b.map_or("-".to_string(), |b| b.to_string()),
        inc.decision.code(),
        inc.penalty.kind.code(),
        inc.penalty.seconds,
        inc.served_to.map_or("-".to_string(), |c| c.to_string()),
        inc.service.label(),
        inc.evidence_hash,
    )
}

pub fn incidents_hash(incidents: &[Incident]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for i in incidents {
        for b in i.evidence_hash.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// Per-car penalty ledger — who serves what, and when.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CarPenalty {
    pub car: u32,
    pub seconds: f64,
    pub incidents: u32,
    pub penalised: u32,
    pub inconclusive: u32,
}

#[derive(Clone, Debug)]
pub struct StewardSummary {
    pub rule_set: &'static str,
    pub rule_set_version: u32,
    pub incidents: usize,
    pub penalised: usize,
    pub no_penalty: usize,
    pub inconclusive: usize,
    pub time_penalty_seconds: f64,
    pub by_car: Vec<CarPenalty>,
}

impl StewardSummary {
    pub fn of(incidents: &[Incident]) -> StewardSummary {
        let mut by_car: Vec<CarPenalty> = Vec::new();
        let mut penalised = 0;
        let mut no_penalty = 0;
        let mut inconclusive = 0;
        let mut time_penalty_seconds = 0.0;
        for inc in incidents {
            match inc.decision {
                Decision::Penalised => penalised += 1,
                Decision::NoPenalty => no_penalty += 1,
                Decision::Inconclusive => inconclusive += 1,
            }
            if let Some(car) = inc.served_to {
                let e = match by_car.iter_mut().find(|e| e.car == car) {
                    Some(e) => e,
                    None => {
                        by_car.push(CarPenalty { car, ..CarPenalty::default() });
                        by_car.last_mut().expect("just pushed")
                    }
                };
                e.incidents += 1;
                e.seconds += inc.penalty.seconds;
                time_penalty_seconds += inc.penalty.seconds;
                match inc.decision {
                    Decision::Penalised => e.penalised += 1,
                    Decision::Inconclusive => e.inconclusive += 1,
                    Decision::NoPenalty => {}
                }
            }
        }
        by_car.sort_by_key(|e| e.car);
        StewardSummary {
            rule_set: RULE_SET,
            rule_set_version: RULE_SET_VERSION,
            incidents: incidents.len(),
            penalised,
            no_penalty,
            inconclusive,
            time_penalty_seconds,
            by_car,
        }
    }

    pub fn seconds_for(&self, car: u32) -> f64 {
        self.by_car.iter().find(|e| e.car == car).map_or(0.0, |e| e.seconds)
    }

    /// Cars in classification order, then re-sorted by the time each serves.
    ///
    /// The simulated classification is deliberately left alone: it records what
    /// the cars did on track. A stewarding penalty is a separate record about
    /// that, so it is presented as a second ordering rather than quietly
    /// rewriting the result.
    /// The finishing order once time penalties are served.
    ///
    /// This is a *derived view*, never an edit: `classification` is exactly
    /// what the race produced and stays that way. A penalty is time, so it is
    /// added to a car's gap to the leader and the field re-sorted on the total
    /// — the same arithmetic a real time penalty gets.
    ///
    /// An earlier version sorted on `pos` first, which made the whole function
    /// return the classification unchanged: the position dominated, the
    /// penalty could never move anyone, and a test asserting the two orders
    /// differed caught it. A derived order that can never differ from its input
    /// is not an order.
    ///
    /// Laps still come first, and retirements still come last — a car a lap
    /// down does not overtake on a five-second penalty.
    pub fn post_penalty_order(&self, classification: &[crate::race::LeaderRow]) -> Vec<u32> {
        // (dnf, Reverse(laps), adjusted gap, id)
        let mut rows: Vec<(u8, std::cmp::Reverse<u32>, f64, u32)> = classification
            .iter()
            .map(|r| {
                let dnf = u8::from(r.retired && !r.finished);
                let gap = r.gap_s.unwrap_or(0.0) + self.seconds_for(r.id);
                (dnf, std::cmp::Reverse(r.laps_done), gap, r.id)
            })
            .collect();
        rows.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.3.cmp(&b.3))
        });
        rows.into_iter().map(|r| r.3).collect()
    }
}

// ---------------------------------------------------------------------------
// Human-review display
// ---------------------------------------------------------------------------

/// One incident as a review sheet: what happened, which rule, what it decided,
/// and the evidence it was allowed to look at.
pub fn render_incident(inc: &Incident, cfg: &StewardConfig) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "\n-- incident #{} | tick {} | t={:.2} s | rule {} v{} ({})\n",
        inc.evidence_hash,
        inc.tick,
        inc.sim_time_s,
        inc.rule.id(),
        inc.rule_version,
        inc.rule.title()
    ));
    s.push_str(&format!(
        "   rule in force (v{}, this race's thresholds): {}\n",
        inc.rule_version,
        inc.rule.statement(cfg)
    ));
    s.push_str(&format!(
        "   window:      ticks {}..{} (span {:.2} s; the incident sits +{:.2} s in)\n",
        inc.evidence.window_start_tick,
        inc.evidence.window_end_tick,
        (inc.evidence.window_end_tick.saturating_sub(inc.evidence.window_start_tick)) as f64
            * crate::sim::DT,
        inc.evidence
            .incident_tick
            .saturating_sub(inc.evidence.window_start_tick) as f64
            * crate::sim::DT,
    ));
    // Only the columns this rule actually read are printed. A column the rule
    // ignored, printed as 0.00, reads as a measurement of zero — which would be
    // a fabrication of exactly the kind this aid exists to avoid.
    let car_line = |c: &CarEvidence, label: &str| {
        let mut line = format!(
            "   {label} car {:>2}:  v {:>6.2} m/s (max {:>6.2})  lateral {:>6.2} m  \
             excursion {:>5.2} m for {:>4} ticks",
            c.car,
            c.speed_mps,
            c.max_speed_mps,
            c.lateral_m,
            c.max_excursion_m,
            c.off_track_ticks,
        );
        if inc.rule == Rule::Contact {
            line.push_str(&format!(
                "  gap {:>5.2} m  overlap {:>5.2} m  closing {:>6.2} m/s",
                c.gap_m,
                c.overlap_m,
                c.closing_into_other_mps
            ));
        }
        if inc.rule == Rule::UnsafeRejoin {
            line.push_str(&format!(
                "  gap {:>5}  closing {:>6.2} m/s",
                if c.gap_m.is_finite() {
                    format!("{:.2} m", c.gap_m)
                } else {
                    // No other car closed on this one, so there is no gap to
                    // report. Printing `inf` would leak the sentinel into the
                    // sheet a human is meant to read.
                    "none in range".to_string()
                },
                c.closing_into_other_mps
            ));
        }
        if c.retired {
            line.push_str("  retired");
        }
        if c.in_pit {
            line.push_str("  in-pit");
        }
        line.push('\n');
        line
    };
    s.push_str(&car_line(&inc.evidence.a, "subject "));
    if let Some(b) = inc.evidence.b.as_ref() {
        s.push_str(&car_line(b, "other   "));
    }
    s.push_str(&format!("   decision:    {} — {}\n", inc.decision.label(), inc.basis));
    s.push_str(&format!(
        "   penalty:    {}",
        match inc.penalty.kind {
            PenaltyKind::None => "none".to_string(),
            PenaltyKind::Reprimand => "reprimand (recorded, no effect on the race)".to_string(),
            PenaltyKind::Time => format!("+{:.2} s", inc.penalty.seconds),
        }
    ));
    s.push_str(&format!(
        "  service: {}\n",
        match inc.served_to {
            Some(car) => format!("car {car} at {}", inc.service.label()),
            None => "not served".to_string(),
        }
    ));
    s.push_str(&format!("   evidence:    hash {:016x}\n", inc.evidence_hash));
    s
}

// ---------------------------------------------------------------------------
// The recorder
// ---------------------------------------------------------------------------

/// One car as the recorder sees it, sampled every physics tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct CarSample {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub s: f64,
    pub lateral: f64,
    pub speed_mps: f64,
    pub retired: bool,
    pub in_pit: bool,
}

#[derive(Clone, Debug)]
struct Frame {
    tick: u64,
    cars: Vec<CarSample>,
}

#[derive(Clone, Copy, Debug)]
struct Excursion {
    started_tick: u64,
    ticks: u64,
    max_m: f64,
}

#[derive(Clone, Debug)]
enum PendingKind {
    Contact { other: u32, closing_speed: f64, into_a: f64, into_b: f64 },
    Shortcut,
    Rejoin,
}

#[derive(Clone, Debug)]
struct Pending {
    kind: PendingKind,
    /// The tick the incident is *about* — what a reviewer cares about.
    incident_tick: u64,
    incident_time_s: f64,
    /// The tick at which enough after-the-fact evidence exists to decide.
    close_tick: u64,
    car_a: u32,
    excursion: Excursion,
    retired_at_incident: bool,
}

/// Watches a race and produces the incident record.
///
/// Called once per physics tick from the observer stage, after the state hash
/// has been folded, so it can neither influence nor be influenced by the
/// trajectory. It reads state and writes only its own buffer.
pub struct Recorder {
    cfg: StewardConfig,
    half_width: f64,
    history: VecDeque<Frame>,
    /// Per car: the excursion beyond the edge currently running, if any.
    excursions: Vec<Option<Excursion>>,
    /// Excursions that ended, awaiting the rejoin/shortcut review on the
    /// frame the car came back. Keyed by car id.
    closed: std::collections::BTreeMap<u32, Excursion>,
    pending: Vec<Pending>,
    contacts_seen: usize,
    /// Pair (ordered by car id) -> the tick it last raised a contact. Used to
    /// fold a sustained grind into the single incident the review window
    /// already judges it as.
    last_contact: BTreeMap<(u32, u32), u64>,
    incidents: Vec<Incident>,
}

impl Recorder {
    pub fn new(cfg: StewardConfig, half_width: f64) -> Recorder {
        Recorder {
            cfg,
            half_width,
            history: VecDeque::new(),
            excursions: Vec::new(),
            closed: std::collections::BTreeMap::new(),
            pending: Vec::new(),
            contacts_seen: 0,
            last_contact: BTreeMap::new(),
            incidents: Vec::new(),
        }
    }

    pub fn config(&self) -> &StewardConfig {
        &self.cfg
    }

    pub fn incidents(&self) -> &[Incident] {
        &self.incidents
    }

    /// Take the incident record, closing anything still open. Called once, at
    /// the end of the race: a case still inside its review window when the flag
    /// falls is judged on the evidence that exists rather than dropped.
    pub fn finish(&mut self) -> Vec<Incident> {
        while let Some(p) = self.pending.first().cloned() {
            self.pending.remove(0);
            self.resolve(p);
        }
        std::mem::take(&mut self.incidents)
    }

    /// Observe one tick. `contacts` is the whole contact list; the recorder
    /// consumes what it has not seen, so it needs no cursor from the caller.
    ///
    /// `under_steward` is the caller's ruling on whether this tick is stewarded
    /// at all — the formation lap under a locked grid is not, and treating the
    /// grid as track limits would file a finding against every car every start.
    pub fn observe(
        &mut self,
        tick: u64,
        sim_time_s: f64,
        under_steward: bool,
        cars: &[CarSample],
        contacts: &[ContactEvent],
    ) {
        if !under_steward {
            return;
        }
        if self.excursions.len() != cars.len() {
            self.excursions = vec![None; cars.len()];
        }
        self.track_excursions(tick, cars);
        self.collect_contacts(tick, sim_time_s, contacts);
        self.history.push_back(Frame { tick, cars: cars.to_vec() });
        self.trim_history(tick);
        self.collect_excursion_endings(tick, sim_time_s, cars);
        self.close_due(tick);
    }

    fn trim_history(&mut self, tick: u64) {
        let keep = self.cfg.history_ticks().max(self.cfg.review_ticks()) + 2;
        while self.history.len() > 2 && self.history[0].tick + keep < tick {
            self.history.pop_front();
        }
    }

    fn track_excursions(&mut self, tick: u64, cars: &[CarSample]) {
        for (i, c) in cars.iter().enumerate() {
            let beyond = (c.lateral.abs() - self.half_width).max(0.0);
            // The pit lane is a parallel road at -7.5 m, deliberately outside the
            // track edge. Without this every routine pit stop reads as a
            // shortcut, which is the first thing that would make the aid a lie.
            //
            // The distance gate is the rule's own definition, applied at the
            // source rather than filtered out afterwards: an excursion only
            // exists once a car is at least track_limit_dist_m out. Cars
            // routinely run a few centimetres wide, and one NO-PENALTY case per
            // car per race buries the findings a reviewer is meant to read.
            // Ticks and distance are then measured over the same stretch — "at
            // least 1.00 m beyond the edge" — so the two halves of the shortcut
            // rule cannot disagree about what was counted.
            let counts = beyond > self.cfg.track_limit_dist_m && !c.in_pit;
            match (&mut self.excursions[i], counts) {
                (Some(e), true) => {
                    e.ticks += 1;
                    e.max_m = e.max_m.max(beyond);
                }
                (slot @ None, true) => {
                    *slot = Some(Excursion { started_tick: tick, ticks: 1, max_m: beyond })
                }
                (slot @ Some(_), false) => {
                    // The car is back on the road: park the excursion for review
                    // and clear the running slot.
                    if let Some(e) = slot.take() {
                        self.closed.insert(c.id, e);
                    }
                }
                (None, false) => {}
            }
        }
    }

    /// A car that was off the road long enough and is now back has both a
    /// *shortcut* to answer for (the excursion itself) and a *rejoin* to answer
    /// for (the return). Both open here, on the same tick, from the same
    /// evidence.
    fn collect_excursion_endings(&mut self, tick: u64, sim_time_s: f64, cars: &[CarSample]) {
        let review = self.cfg.review_ticks();
        let min_off = self.cfg.rejoin_min_off_ticks;
        let mut fired: Vec<(u32, Excursion, bool)> = Vec::new();
        for c in cars.iter() {
            if c.in_pit {
                continue;
            }
            let Some(e) = self.closed.get(&c.id).copied() else { continue };
            if e.ticks < min_off {
                continue;
            }
            fired.push((c.id, e, c.retired));
        }
        for (car, e, retired) in fired {
            self.closed.remove(&car);
            // Under the track-limit thresholds this is still recorded: the rule
            // is what decides, not the trigger.
            self.pending.push(Pending {
                kind: PendingKind::Shortcut,
                incident_tick: e.started_tick,
                incident_time_s: e.started_tick as f64 * crate::sim::DT,
                close_tick: tick,
                car_a: car,
                excursion: e,
                retired_at_incident: retired,
            });
            self.pending.push(Pending {
                kind: PendingKind::Rejoin,
                incident_tick: tick,
                incident_time_s: sim_time_s,
                close_tick: tick + review,
                car_a: car,
                excursion: e,
                retired_at_incident: retired,
            });
        }
    }

    fn collect_contacts(&mut self, _tick: u64, sim_time_s: f64, contacts: &[ContactEvent]) {
        let review = self.cfg.review_ticks();
        while self.contacts_seen < contacts.len() {
            let c = &contacts[self.contacts_seen];
            self.contacts_seen += 1;
            // Two cars grinding together raise a contact event on many
            // consecutive ticks. Each would otherwise become its own incident,
            // and a pack reshuffling on the formation lap produced 17 records
            // for one moment of contact. A repeat from the same pair inside
            // the review window is the same incident — the window is already
            // the span over which the rule is judged — so it is folded in.
            let pair = (c.a.min(c.b), c.a.max(c.b));
            if self.last_contact.get(&pair).is_some_and(|&last| c.tick < last + review) {
                continue;
            }
            self.last_contact.insert(pair, c.tick);
            self.pending.push(Pending {
                kind: PendingKind::Contact {
                    other: c.b,
                    closing_speed: c.closing_speed,
                    into_a: c.closing_into_a,
                    into_b: c.closing_into_b,
                },
                incident_tick: c.tick,
                incident_time_s: sim_time_s,
                close_tick: c.tick + review,
                car_a: c.a,
                excursion: Excursion { started_tick: c.tick, ticks: 0, max_m: 0.0 },
                retired_at_incident: c.retired.is_some(),
            });
        }
    }

    fn close_due(&mut self, tick: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].close_tick <= tick {
                let p = self.pending.remove(i);
                self.resolve(p);
            } else {
                i += 1;
            }
        }
    }

    fn resolve(&mut self, p: Pending) {
        let history_ticks = self.cfg.history_ticks();
        let review_ticks = self.cfg.review_ticks();
        let ev = match p.kind {
            PendingKind::Contact { other, closing_speed, into_a, into_b } => {
                let zero = Excursion { started_tick: p.incident_tick, ticks: 0, max_m: 0.0 };
                let mut a = self.car_evidence(p.car_a, p.incident_tick, zero);
                let mut b = self.car_evidence(other, p.incident_tick, zero);
                a.closing_into_other_mps = into_b;
                b.closing_into_other_mps = into_a;
                let gap = self.gap_at(p.incident_tick, p.car_a, other);
                a.gap_m = gap;
                b.gap_m = gap;
                a.overlap_m = (CONTACT_DIST_M - gap).max(0.0);
                b.overlap_m = a.overlap_m;
                Evidence {
                    rule: Rule::Contact,
                    incident_tick: p.incident_tick,
                    sim_time_s: p.incident_time_s,
                    window_start_tick: p.incident_tick.saturating_sub(history_ticks),
                    window_end_tick: p.incident_tick + review_ticks,
                    a,
                    b: Some(b),
                    closing_speed_mps: closing_speed,
                    // How much more one car closed on the other than the
                    // reverse. Near zero means the evidence does not separate
                    // them, which the contact rule must then treat as no finding.
                    fault_margin_mps: into_b - into_a,
                }
            }
            PendingKind::Shortcut => {
                let mut a = self.car_evidence(p.car_a, p.incident_tick, p.excursion);
                a.retired = p.retired_at_incident;
                Evidence {
                    rule: Rule::Shortcut,
                    incident_tick: p.incident_tick,
                    sim_time_s: p.incident_time_s,
                    window_start_tick: p.incident_tick.saturating_sub(history_ticks),
                    window_end_tick: p.incident_tick,
                    a,
                    b: None,
                    closing_speed_mps: 0.0,
                    fault_margin_mps: 0.0,
                }
            }
            PendingKind::Rejoin => {
                let a_id = p.car_a;
                // (other car, room at the rejoin, closest approach, closing there)
                let other = self.dangerous_closer(a_id, p.incident_tick, review_ticks);
                let mut a = self.car_evidence(a_id, p.incident_tick, p.excursion);
                // The room the joining car had when it came back — this is the
                // quantity the "could have steered away" test turns on.
                a.gap_m = other.map_or(f64::INFINITY, |(_, room, _, _)| room);
                let b = other.map(|(id, _room, closest, closing)| {
                    let zero = Excursion { started_tick: p.incident_tick, ticks: 0, max_m: 0.0 };
                    let mut e = self.car_evidence(id, p.incident_tick, zero);
                    // Closest approach and the closing rate there: one instant,
                    // described by two numbers.
                    e.gap_m = closest;
                    e.closing_into_other_mps = closing;
                    e
                });
                Evidence {
                    rule: Rule::UnsafeRejoin,
                    incident_tick: p.incident_tick,
                    sim_time_s: p.incident_time_s,
                    window_start_tick: p.incident_tick,
                    window_end_tick: p.incident_tick + review_ticks,
                    a,
                    closing_speed_mps: b.map_or(0.0, |b| b.closing_into_other_mps),
                    fault_margin_mps: b.map_or(0.0, |b| b.closing_into_other_mps),
                    b,
                }
            }
        };
        let finding = judge(&ev, &self.cfg);
        let mut inc = Incident {
            rule: ev.rule,
            rule_version: ev.rule.version(),
            tick: ev.incident_tick,
            sim_time_s: ev.sim_time_s,
            car_a: ev.a.car,
            car_b: ev.b.map(|b| b.car),
            decision: finding.decision,
            penalty: finding.penalty,
            served_to: finding.served_to,
            service: finding.service,
            basis: finding.basis,
            evidence: ev,
            evidence_hash: 0,
        };
        inc.evidence_hash = inc.compute_hash();
        self.incidents.push(inc);
    }

    fn car_evidence(&self, car: u32, at_tick: u64, exc: Excursion) -> CarEvidence {
        let mut out = CarEvidence { car, ..CarEvidence::default() };
        let floor = at_tick.saturating_sub(self.cfg.history_ticks());
        let mut seen = false;
        for f in self.history.iter() {
            if f.tick < floor || f.tick > at_tick {
                continue;
            }
            let Some(c) = f.cars.iter().find(|c| c.id == car) else { continue };
            seen = true;
            out.max_speed_mps = out.max_speed_mps.max(c.speed_mps);
            if f.tick == at_tick {
                out.speed_mps = c.speed_mps;
                out.lateral_m = c.lateral;
                out.retired = c.retired;
                out.in_pit = c.in_pit;
            }
        }
        if !seen {
            // The incident tick can predate the trimmed history on a very short
            // run; take the nearest frame there is rather than reporting zeroes.
            if let Some(c) = self
                .history
                .iter()
                .find(|f| f.cars.iter().any(|c| c.id == car))
                .and_then(|f| f.cars.iter().find(|c| c.id == car))
            {
                out.speed_mps = c.speed_mps;
                out.max_speed_mps = c.speed_mps;
                out.lateral_m = c.lateral;
                out.retired = c.retired;
                out.in_pit = c.in_pit;
            }
        }
        out.max_excursion_m = exc.max_m;
        out.off_track_ticks = exc.ticks;
        out
    }

    fn frame_at(&self, tick: u64) -> Option<&Frame> {
        self.history.iter().find(|f| f.tick == tick)
    }

    fn gap_at(&self, tick: u64, a: u32, b: u32) -> f64 {
        let Some(f) = self.frame_at(tick) else { return f64::INFINITY };
        let (Some(ca), Some(cb)) = (
            f.cars.iter().find(|c| c.id == a),
            f.cars.iter().find(|c| c.id == b),
        ) else {
            return f64::INFINITY;
        };
        ((cb.x - ca.x).powi(2) + (cb.y - ca.y).powi(2)).sqrt()
    }

    /// The car that came closest to the rejoining car during the review
    /// window, and three numbers that each mean something at a *named* time.
    ///
    /// Measuring the separation and the closing rate at one instant is right
    /// in principle and wrong for this rule: "came within 12 m" and "had 12 m
    /// to steer away" are mutually exclusive, so a single instant can never
    /// satisfy both halves and the rule can never penalise anything. An earlier
    /// version had the opposite fault, pairing the gap at the rejoin with the
    /// largest closing rate seen anywhere in the window, which reads as "269 m
    /// apart, closing at 29 m/s" — arithmetically possible, and meaningless
    /// without saying when.
    ///
    /// So each number is tied to its own time and named in the finding:
    ///   - `gap_rejoin`  — separation at the moment of the rejoin: how much
    ///     room the joining car had. Decides *avoidable*.
    ///   - `min_gap`     — closest approach over the review window.
    ///   - `closing`     — closing rate at that closest approach. Decides
    ///     *dangerous*. Together these two describe one instant.
    ///
    /// Returns (other car, gap at rejoin m, closest approach m, closing rate
    /// at that approach m/s). Measured from recorded positions rather than
    /// live state, so a replay measures exactly the same approach.
    fn dangerous_closer(
        &self,
        a: u32,
        at_tick: u64,
        review_ticks: u64,
    ) -> Option<(u32, f64, f64, f64)> {
        let sep = |f: &Frame, id: u32| -> Option<f64> {
            let c = f.cars.iter().find(|c| c.id == id)?;
            let o = f.cars.iter().find(|o| o.id == a)?;
            Some(((o.x - c.x).powi(2) + (o.y - c.y).powi(2)).sqrt())
        };
        let frames: Vec<&Frame> = self
            .history
            .iter()
            .filter(|f| f.tick >= at_tick && f.tick <= at_tick + review_ticks)
            .collect();
        // Best = closest approach, and the closing rate at that same approach.
        let mut best: Option<(u32, f64, f64)> = None;
        for w in frames.windows(2) {
            let (f0, f1) = (w[0], w[1]);
            for cb in f1.cars.iter() {
                if cb.id == a {
                    continue;
                }
                // A car in the pit lane is on a parallel road. It did not close
                // on anyone; it drove where the pit lane goes, which is
                // deliberately outside the track edge.
                if cb.in_pit {
                    continue;
                }
                let (Some(s0), Some(s1)) = (sep(f0, cb.id), sep(f1, cb.id)) else { continue };
                if s1 >= s0 {
                    continue; // opening, not closing, on this tick
                }
                let closing = (s0 - s1) / crate::sim::DT;
                let closer = best.is_none_or(|(_, gap, _)| s1 < gap);
                if closer {
                    best = Some((cb.id, s1, closing));
                }
            }
        }
        let (id, min_gap, closing) = best?;
        let room = self.gap_at(at_tick, a, id);
        Some((id, room, min_gap, closing))
    }
}
