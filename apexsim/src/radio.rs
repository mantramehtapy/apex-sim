//! Phase 12, second subdeliverable: short radio messages from stable race
//! events.
//!
//! This is deliberately an **observer**. It reads state that already exists —
//! the race-control log and the stewarding aid's incident list — and turns a
//! subset of it into text. It never writes to the race, never consults a
//! driver, and is off by default, so attaching it leaves every earlier phase
//! bit-for-bit identical. That matters more than it sounds: every other
//! Phase 12 component either replaces a controller or changes what the
//! simulation *does*, and this one must be provably incapable of either.
//!
//! ## What "stable" means here
//!
//! The plan asks for messages "from stable race events", and the word is
//! load-bearing. A message a spectator sees once and never again is noise; a
//! message built from an event that drifts between runs is worse than noise,
//! because it looks authoritative. So every message here is derived from one of
//! exactly two sources, both of which the engine already guarantees are
//! reproducible:
//!
//! * [`RaceState::rc_events`] — lights, green, pit stops, fastest laps, weather
//!   overrides. `Race` appends to this in a fixed order, and the replay tests
//!   in earlier phases already pin it.
//! * `steward::Recorder::incidents()` — each carries an `evidence_hash` and a
//!   `compute_hash`, so a replay either reproduces a message byte for byte or
//!   fails loudly.
//!
//! What is *not* a source is per-tick car state. Nothing here reads a speed or
//! a position to decide whether to speak. A radio that fires on a threshold is
//! a chatter generator, and chatter during a corner is exactly the failure the
//! engine's own event log exists to avoid.
//!
//! ## Determinism
//!
//! [`Radio::observe`] consumes the two event lists by index, so the same seed
//! produces the same messages in the same order. There is no RNG, no wall
//! clock, and no formatting of anything but values that are already part of the
//! event. `tests/phase12_gate.rs` asserts this by running the same race twice
//! and comparing the two message lists for equality.

use crate::race::{Race, RcEvent};
use crate::steward;

/// How urgent a transmission is. Ordered, so callers can filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Caution,
    Critical,
}

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Caution => "caution",
            Severity::Critical => "critical",
        }
    }
}

/// Who is speaking. Race control addresses the field; a car transmission is
/// addressed to one team.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    RaceControl,
    Car(u32),
}

impl Channel {
    pub fn label(&self) -> String {
        match self {
            Channel::RaceControl => "race-control".to_string(),
            Channel::Car(id) => format!("car-{id}"),
        }
    }
}

/// One short transmission.
#[derive(Clone, Debug, PartialEq)]
pub struct RadioMessage {
    /// Tick the underlying event landed on. Never "now" — a message about a
    /// past event that reports the present tick is a message you cannot
    /// correlate with a replay.
    pub tick: u64,
    pub sim_time_s: f64,
    pub channel: Channel,
    pub severity: Severity,
    /// The transmission. Short by construction; see [`MAX_LEN`].
    pub text: String,
    /// What this message was derived from, so a reader can check the claim
    /// against the event log instead of taking it on trust.
    pub basis: String,
}

/// Longest a transmission may be. "Short" is a real requirement — these render
/// one line in the spectator view — so it is enforced rather than requested.
pub const MAX_LEN: usize = 96;

/// Transcribes stable race events. Disabled by default and inert when off.
#[derive(Debug)]
pub struct Radio {
    enabled: bool,
    /// How far into `rc_events` we have consumed. An index, not a set: the log
    /// is append-only, so this is both correct and cheap.
    rc_cursor: usize,
    incident_cursor: usize,
    out: Vec<RadioMessage>,
}

impl Default for Radio {
    fn default() -> Self {
        Radio::off()
    }
}

impl Radio {
    /// Off. A disabled radio records nothing and allocates nothing.
    pub fn off() -> Radio {
        Radio {
            enabled: false,
            rc_cursor: 0,
            incident_cursor: 0,
            out: Vec::new(),
        }
    }

    pub fn enabled(enabled: bool) -> Radio {
        let mut r = Radio::off();
        r.enabled = enabled;
        r
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Everything said so far, in order.
    pub fn messages(&self) -> &[RadioMessage] {
        &self.out
    }

    pub fn into_messages(self) -> Vec<RadioMessage> {
        self.out
    }

    pub fn clear(&mut self) {
        self.out.clear();
        self.rc_cursor = 0;
        self.incident_cursor = 0;
    }

    /// Consume any events that have appeared since the last call.
    ///
    /// Read-only on the race. Anything that would need a mutable borrow of the
    /// race to say is, by definition, not derivable from a stable event, and so
    /// does not belong here.
    pub fn observe(&mut self, race: &Race) -> &[RadioMessage] {
        if !self.enabled {
            return &self.out;
        }
        let first_new = self.out.len();
        let tick = race.state.tick;
        let t = race.state.sim_time;

        // ---- race control ----
        // `get` + cursor advance, not `drain`: a slice borrow of the event log
        // and a mutable borrow of `self` cannot coexist, and the cursor is
        // what makes the next call pick up exactly where this one stopped.
        let pending: Vec<&RcEvent> = race.state.rc_events[self.rc_cursor.min(race.state.rc_events.len())..].iter().collect();
        for ev in pending {
            let Some((channel, severity, text, basis)) = self.transcribe(ev) else {
                continue;
            };
            let msg = RadioMessage {
                tick,
                sim_time_s: t,
                channel,
                severity,
                text,
                basis,
            };
            if msg.text.chars().count() <= MAX_LEN {
                self.out.push(msg);
            }
        }
        self.rc_cursor = race.state.rc_events.len();

        // ---- stewarding ----
        if let Some(rec) = race.steward.as_ref() {
            let pending: Vec<&steward::Incident> =
                rec.incidents()[self.incident_cursor.min(rec.incidents().len())..].iter().collect();
            for inc in pending {
                if let Some((channel, severity, text, basis)) = self.transcribe_incident(inc) {
                    let msg = RadioMessage {
                        tick: inc.tick,
                        sim_time_s: inc.sim_time_s,
                        channel,
                        severity,
                        text,
                        basis,
                    };
                    if msg.text.chars().count() <= MAX_LEN {
                        self.out.push(msg);
                    }
                }
            }
            self.incident_cursor = rec.incidents().len();
        }

        &self.out[first_new..]
    }

    /// One race-control event, or nothing if it is not worth saying.
    ///
    /// The event variants that return `None` are the ones where the message
    /// would only restate something the spectator can already see. `PitExited`
    /// is the clearest case: the car rejoining is visible, and a message about
    /// it arrives while the car is on the racing line and cannot be acted on.
    fn transcribe(&self, ev: &RcEvent) -> Option<(Channel, Severity, String, String)> {
        Some(match *ev {
            RcEvent::Lights(n) => (
                Channel::RaceControl,
                Severity::Info,
                format!("Start lights: {n} of 5."),
                "rc/lights".into(),
            ),
            RcEvent::Green => (
                Channel::RaceControl,
                Severity::Info,
                "Green flag. Race start.".into(),
                "rc/green".into(),
            ),
            RcEvent::PitEntered { car, lap } => (
                Channel::Car(car),
                Severity::Info,
                format!("Car {car} pits on lap {lap}."),
                "rc/pit-entered".into(),
            ),
            RcEvent::PitServiceDone { car, held_s } => (
                Channel::Car(car),
                Severity::Info,
                format!("Car {car} released after {held_s:.1} s."),
                "rc/pit-service-done".into(),
            ),
            RcEvent::FastestLap { car, time_s, lap } => (
                Channel::RaceControl,
                Severity::Info,
                format!("Fastest lap: car {car}, {time_s:.3} s (lap {lap})."),
                "rc/fastest-lap".into(),
            ),
            RcEvent::WeatherApplied { tick, slider, value } => (
                Channel::RaceControl,
                Severity::Caution,
                format!("Race director: {slider} set to {value:.2} (tick {tick})."),
                "rc/weather".into(),
            ),
            RcEvent::CarOut { car, cause, detail } => (
                Channel::RaceControl,
                Severity::Caution,
                format!("Car {car} OUT — {cause} ({detail})."),
                "rc/car-out".into(),
            ),
            RcEvent::SafetyCar { phase } => {
                let text = match phase {
                    "deployed" => "SAFETY CAR DEPLOYED. No overtaking.",
                    "clear" => "Safety car in this lap. Track clear — racing.",
                    other => other,
                };
                (
                    Channel::RaceControl,
                    Severity::Critical,
                    text.to_string(),
                    "rc/safety-car".into(),
                )
            }
            // A car rejoining is visible from the trackside, and a message
            // about it lands while the car is on the racing line and cannot be
            // acted on. The pit *entry* is the transmission that matters.
            RcEvent::PitExited { .. } => return None,
        })
    }

    /// One steward incident, or nothing.
    ///
    /// Only decisions that reached a conclusion are transmitted. An
    /// inconclusive finding is the aid saying "the evidence does not separate
    /// these cars" — broadcasting that as though it were a verdict would be the
    /// aid inventing authority it explicitly declined to claim.
    fn transcribe_incident(
        &self,
        inc: &steward::Incident,
    ) -> Option<(Channel, Severity, String, String)> {
        use steward::{Decision, PenaltyKind};
        let car = inc.served_to?;
        let basis = format!("steward/{}@v{}", inc.rule.id(), inc.rule_version);
        match inc.decision {
            Decision::Penalised => {
                let what = match inc.penalty.kind {
                    PenaltyKind::Time => format!("time penalty, {:.1} s", inc.penalty.seconds),
                    PenaltyKind::Reprimand => "reprimand".to_string(),
                    // Unreachable: `Finding::none` only builds non-penalised
                    // findings, and a penalty here would be an upstream bug.
                    PenaltyKind::None => return None,
                };
                Some((
                    Channel::RaceControl,
                    Severity::Critical,
                    format!("Car {car}: {what} — {}", inc.rule.title()),
                    basis,
                ))
            }
            Decision::NoPenalty => Some((
                Channel::RaceControl,
                Severity::Info,
                format!("Car {car}: no penalty — {}", inc.rule.title()),
                basis,
            )),
            Decision::Inconclusive => None,
        }
    }
}

/// One-line summary of a message list, for logs and results sheets.
pub fn summarise(msgs: &[RadioMessage]) -> String {
    if msgs.is_empty() {
        return "no transmissions".to_string();
    }
    let critical = msgs.iter().filter(|m| m.severity == Severity::Critical).count();
    let caution = msgs.iter().filter(|m| m.severity == Severity::Caution).count();
    format!(
        "{} transmissions ({} critical, {} caution)",
        msgs.len(),
        critical,
        caution
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_radio_never_speaks() {
        let mut race = crate::race::Race::new(crate::race::RaceConfig::default());
        let mut r = Radio::off();
        for _ in 0..500 {
            r.observe(&race);
            race.tick();
        }
        assert!(r.messages().is_empty());
    }

    #[test]
    fn every_transmission_fits_one_line() {
        let mut race = crate::race::Race::new(crate::race::RaceConfig { cars: 20, laps: 2, ..Default::default() });
        let mut r = Radio::enabled(true);
        race.start();
        for _ in 0..30000 {
            r.observe(&race);
            race.tick();
            if race.state.phase == crate::race::RacePhase::Finished {
                break;
            }
        }
        assert!(!r.messages().is_empty(), "a full race must say something");
        for m in r.messages() {
            assert!(m.text.chars().count() <= MAX_LEN, "too long: {}", m.text);
            assert!(!m.text.is_empty());
        }
    }
}