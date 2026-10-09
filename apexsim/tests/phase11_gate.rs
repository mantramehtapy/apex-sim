//! Phase 11 gate — stewarding and incident records.
//!
//! The plan's completion gate is three sentences, and they are the three
//! headings below, quoted verbatim:
//!
//!   1. Replay produces the same evidence and rule outcome.
//!   2. Each penalty links to a stated simulator rule and can be audited.
//!   3. Ambiguous cases do not silently become certain fault findings.
//!
//! Two things about this file are deliberate and worth stating up front.
//!
//! First, several tests drive the judges with *constructed* evidence rather
//! than with a race. That is not a shortcut around testing the recorder; it is
//! the only way to reach the branches that matter most. No race in this engine
//! reliably produces a case that sits inside the ambiguous bands, so a suite
//! that only ran races would never execute the "decline to find fault" paths
//! and would report them green.
//!
//! Second, several tests assert that something is *refused* — a tampered
//! incident line, a foreign rule set. A verification check that has never been
//! shown to fail is not evidence that anything is verified, so each one is
//! paired with the corrupted input that must be rejected.

use apexsim::race::{self, RaceConfig};
use apexsim::steward::{
    self, CarEvidence, Decision, Evidence, PenaltyKind, Rule, ServicePoint, StewardConfig,
};

fn cfg() -> StewardConfig {
    StewardConfig::enabled()
}

/// A car with sensible defaults; every field is set explicitly at the call
/// site that matters, so a test never depends on `Default` meaning anything.
fn car(id: u32) -> CarEvidence {
    CarEvidence { car: id, speed_mps: 40.0, ..Default::default() }
}

/// Collapse all whitespace, so a phrase broken across two lines is still
/// found. Checking raw source text for a multi-word phrase is otherwise a
/// test of the formatter, not of the disclaimer.
fn flatten(s: &str) -> String {
    s.replace("//!", " ").replace("//", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn evidence(rule: Rule) -> Evidence {    Evidence {
        rule,
        incident_tick: 1000,
        sim_time_s: 4.0,
        window_start_tick: 250,
        window_end_tick: 1500,
        a: car(1),
        b: None,
        closing_speed_mps: 0.0,
        fault_margin_mps: 0.0,
    }
}

// ---------------------------------------------------------------------------
// Gate 1: replay produces the same evidence and rule outcome
// ---------------------------------------------------------------------------

/// A stewarded race writes a D header (rule set, version, count, hash, and
/// every threshold) plus one P line per incident, and a replay of that log
/// re-derives the identical list.
#[test]
fn gate1_replay_reproduces_the_same_incident_record() {
    let dir = std::env::temp_dir().join("f1_p11_replay");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("stewarded.simlog");

    let rc = RaceConfig {
        cars: 16,
        laps: 8,
        seed: 21,
        steward: Some(cfg()),
        ..Default::default()
    };
    let (outcome, inputs) = race::run_race(rc.clone(), false);
    let before = outcome.steward_incidents.clone().expect("steward ran");
    assert!(!before.is_empty(), "this race must raise at least one incident");

    race::write_race_log(&path, &rc, &inputs, &outcome).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let logged: Vec<&str> = text.lines().filter(|l| l.starts_with("P ")).collect();
    assert_eq!(
        logged.len(),
        before.len(),
        "the log must carry one P line per incident, not a count it trusts"
    );
    assert!(
        text.lines().any(|l| l.starts_with("D ")),
        "the log must carry the D header, or a reader has no rule set"
    );

    let (ok, replayed) = race::replay_race_log(&path).unwrap();
    let after = replayed.steward_incidents.clone().expect("replay ran the steward");
    assert_eq!(
        steward::incidents_hash(&before),
        steward::incidents_hash(&after),
        "the replayed evidence differs from the recorded evidence"
    );
    assert!(ok, "the trajectory itself must still replay bit-identically");
    for (a, b) in before.iter().zip(after.iter()) {
        assert_eq!(a.rule, b.rule);
        assert_eq!(a.rule_version, b.rule_version);
        assert_eq!(a.decision, b.decision);
        assert_eq!(a.penalty.kind, b.penalty.kind);
        assert_eq!(a.served_to, b.served_to);
        assert_eq!(a.evidence_hash, b.evidence_hash);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The check above is only worth anything if it can fail. Flipping one
/// decision in a P line must make the replay refuse, loudly and by name.
#[test]
fn gate1b_a_tampered_incident_line_is_refused() {
    let dir = std::env::temp_dir().join("f1_p11_tamper");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("stewarded.simlog");

    let rc = RaceConfig {
        cars: 16,
        laps: 8,
        seed: 21,
        steward: Some(cfg()),
        ..Default::default()
    };
    let (outcome, inputs) = race::run_race(rc.clone(), false);
    race::write_race_log(&path, &rc, &inputs, &outcome).unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    let tampered = text.lines().map(|l| {
        if l.starts_with("P ") {
            // no-penalty <-> penalised, in whichever direction is legal.
            if l.contains(" no-penalty ") {
                l.replace(" no-penalty ", " penalised ")
            } else if l.contains(" penalised ") {
                l.replace(" penalised ", " no-penalty ")
            } else {
                l.to_string()
            }
        } else {
            l.to_string()
        }
    });
    let bad = dir.join("tampered.simlog");
    std::fs::write(&bad, tampered.collect::<Vec<_>>().join("\n")).unwrap();

    let err = race::replay_race_log(&bad).expect_err("a tampered P line must not replay clean");
    assert!(
        err.contains("incident record did not reproduce"),
        "the error must name the failure, not surface as a bare divergence: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A record produced under a different rule set cannot be reproduced by this
/// engine's rules, and must not be quietly reinterpreted as if it could.
#[test]
fn gate1c_a_foreign_rule_set_is_refused() {
    let dir = std::env::temp_dir().join("f1_p11_foreign");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("foreign.simlog");

    let rc = RaceConfig {
        cars: 12,
        laps: 6,
        seed: 5,
        steward: Some(cfg()),
        ..Default::default()
    };
    let (outcome, inputs) = race::run_race(rc.clone(), false);
    race::write_race_log(&path, &rc, &inputs, &outcome).unwrap();

    let text = std::fs::read_to_string(&path).unwrap();
    let bad = dir.join("foreign_ruleset.simlog");
    std::fs::write(
        &bad,
        text.lines()
            .map(|l| {
                if l.starts_with("D ") {
                    l.replacen(steward::RULE_SET, "some-other-steward", 1)
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();

    let err = race::replay_race_log(&bad).expect_err("a foreign rule set must not be accepted");
    assert!(err.contains("some-other-steward"), "the error must name the rule set: {err}");
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Gate 2: each penalty links to a stated simulator rule and can be audited
// ---------------------------------------------------------------------------

/// Every penalised incident names its rule and version, carries the numbers it
/// decided on, and re-decides itself from its own recorded evidence.
#[test]
fn gate2_every_penalty_names_a_rule_and_audits_from_its_own_evidence() {
    // Seed 10 is what exercises the penalty path now. Phase 12 fixed the
    // driver's noise handling (noise landing on an unused brake channel could
    // zero the throttle), so the cars accelerate as intended and stay on the
    // road more often, and the old seed stopped producing penalties at all.
    // The gate is that penalties are reachable and auditable, not that one
    // particular seed produces them.
    let rc = RaceConfig {
        cars: 20,
        laps: 12,
        seed: 10,
        steward: Some(cfg()),
        ..Default::default()
    };
    let (outcome, _) = race::run_race(rc.clone(), false);
    let incidents = outcome.steward_incidents.clone().expect("steward ran");
    let summary = outcome.steward.clone().expect("summary");

    let penalised: Vec<&steward::Incident> =
        incidents.iter().filter(|i| i.decision == Decision::Penalised).collect();
    assert!(!penalised.is_empty(), "this race must exercise the penalty path");

    for inc in &penalised {
        // (a) links to a stated rule, at a stated version
        assert_eq!(inc.rule_version, steward::RULE_SET_VERSION);
        assert!(inc.rule.version() == inc.rule_version);
        assert_eq!(inc.rule, Rule::parse(inc.rule.id()).expect("rule id round-trips"));

        // (b) a real penalty, served somewhere real
        assert_ne!(inc.penalty.kind, PenaltyKind::None, "penalised with no penalty");
        assert!(inc.penalty.seconds > 0.0);
        assert!(inc.served_to.is_some(), "a penalty that names no car is unauditable");
        assert_ne!(inc.service, ServicePoint::NotServed);

        // (c) auditable: the decision is derivable from the recorded evidence,
        //     and the ledger agrees with the incidents.
        let refinding = steward::judge(&inc.evidence, &cfg());
        assert_eq!(refinding.decision, inc.decision, "decision not derivable from evidence");
        assert_eq!(refinding.penalty.kind, inc.penalty.kind);
        assert_eq!(refinding.served_to, inc.served_to);
        assert_eq!(
            summary.seconds_for(inc.served_to.unwrap()),
            incidents
                .iter()
                .filter(|i| i.served_to == inc.served_to && i.decision == Decision::Penalised)
                .count() as f64
                * inc.penalty.seconds,
            "the ledger must agree with the incident list"
        );

        // (d) the basis states the numbers the rule turned on, not a verdict
        assert!(inc.basis.len() > 40, "a bare verdict is not auditable: {}", inc.basis);
        assert!(inc.basis.contains(&format!("car {}", inc.served_to.unwrap())));
        assert!(inc.evidence_hash != 0);

        // (e) the human-review sheet carries the rule, the numbers and the car
        let sheet = steward::render_incident(inc, &cfg());
        assert!(sheet.contains(inc.rule.id()));
        assert!(sheet.contains(&format!("v{}", inc.rule_version)));
        assert!(sheet.contains(&inc.basis));
        assert!(sheet.contains(&format!("car {}", inc.served_to.unwrap())));
        assert!(sheet.contains("rule in force"), "the sheet must state the rule it applied");
    }
}

/// The review sheet must not read as a verdict from an authority. The plan is
/// explicit: label this a stewarding aid, not an official rules authority.
#[test]
fn gate2b_the_aid_does_not_claim_to_be_a_rules_authority() {
    let text = std::path::Path::new("src/steward.rs");
    let src = flatten(&std::fs::read_to_string(text).unwrap());
    assert!(
        src.contains("not a rules authority"),
        "the module must disclaim official authority in its own documentation"
    );
    // The disclaimer must reach the operator, not just the source: it is
    // printed by `apexsim race --steward` and `apexsim steward`. Whitespace is
    // collapsed first because a phrase split across two `//!` lines is still
    // the same phrase to a reader.
    let main = std::fs::read_to_string("src/main.rs").unwrap();
    assert!(
        flatten(&main).contains("not an official rules authority"),
        "the CLI must print the disclaimer a reviewer will read"
    );
    for rule in Rule::ALL {
        let s = rule.statement(&cfg());
        assert!(!s.is_empty());
        assert!(
            s.contains("INCONCLUSIVE") || rule == Rule::Shortcut,
            "each rule must name its inconclusive outcome where it has one"
        );
    }
}

// ---------------------------------------------------------------------------
// Gate 3: ambiguous cases do not silently become certain fault findings
// ---------------------------------------------------------------------------

/// The contact rule has two distinct ambiguity branches. Both must decline.
#[test]
fn gate3_an_ambiguous_contact_is_not_a_fault_finding() {
    let c = cfg();

    // Inside the +/- band around the threshold: explicitly not a finding.
    let mut ev = evidence(Rule::Contact);
    ev.closing_speed_mps = c.contact_cs_penalty - 1.0;
    let mut b = car(2);
    b.closing_into_other_mps = 0.0;
    ev.a.closing_into_other_mps = ev.closing_speed_mps;
    ev.a.overlap_m = 1.0;
    ev.b = Some(b);
    let f = steward::judge_contact(&ev, &c);
    assert_eq!(f.decision, Decision::Inconclusive, "inside the band must be inconclusive");
    assert_eq!(f.penalty.kind, PenaltyKind::None);
    assert_eq!(f.served_to, None);

    // Clearly notable, but both cars were closing: the evidence does not say
    // which one did it.
    let mut ev = evidence(Rule::Contact);
    ev.closing_speed_mps = 20.0;
    ev.fault_margin_mps = 0.5; // below contact_fault_margin
    ev.a.closing_into_other_mps = 10.0;
    let mut b = car(2);
    b.closing_into_other_mps = 9.5;
    ev.b = Some(b);
    let f = steward::judge_contact(&ev, &c);
    assert_eq!(f.decision, Decision::Inconclusive, "a symmetric contact must not pick a culprit");
    assert_eq!(f.penalty.kind, PenaltyKind::None);
    assert_eq!(f.served_to, None);

    // A contact with no second car is a record of a car, not of an incident.
    let mut ev = evidence(Rule::Contact);
    ev.closing_speed_mps = 40.0;
    ev.a.closing_into_other_mps = 40.0;
    let f = steward::judge_contact(&ev, &c);
    assert_eq!(f.decision, Decision::Inconclusive);
    assert_eq!(f.penalty.kind, PenaltyKind::None);
}

/// The rejoin rule's ambiguity branch: the closing car was already alongside,
/// or was itself off the road, so nothing in the record says it could have
/// moved.
#[test]
fn gate3b_an_unavoidable_rejoin_is_not_a_fault_finding() {
    let c = cfg();

    // Committed alongside: too little separation to have steered away.
    let mut ev = evidence(Rule::UnsafeRejoin);
    ev.a.off_track_ticks = c.rejoin_min_off_ticks * 2;
    ev.a.max_excursion_m = 2.0;
    let mut b = car(2);
    b.gap_m = 1.5; // below rejoin_avoidable_gap_m
    b.closing_into_other_mps = 12.0;
    ev.b = Some(b);
    ev.closing_speed_mps = 12.0;
    let f = steward::judge_rejoin(&ev, &c);
    assert_eq!(f.decision, Decision::Inconclusive, "a committed car must not be blamed");
    assert_eq!(f.penalty.kind, PenaltyKind::None);
    assert_eq!(f.served_to, None);

    // The closing car was itself off the road.
    let mut ev = evidence(Rule::UnsafeRejoin);
    ev.a.off_track_ticks = c.rejoin_min_off_ticks * 2;
    let mut b = car(2);
    b.gap_m = 40.0;
    b.off_track_ticks = 500;
    b.closing_into_other_mps = 15.0;
    ev.b = Some(b);
    ev.closing_speed_mps = 15.0;
    let f = steward::judge_rejoin(&ev, &c);
    assert_eq!(f.decision, Decision::Inconclusive, "two cars off the road must not pick one");
    assert_eq!(f.penalty.kind, PenaltyKind::None);
}

/// The structural guarantee, swept across a wide band of evidence: any
/// decision that is not Penalised carries no penalty and names no car. This is
/// what stops ambiguity from leaking into a sanction somewhere downstream of
/// the judges.
#[test]
fn gate3c_only_a_penalised_decision_can_carry_a_penalty() {
    let c = cfg();
    let mut checked = 0usize;

    for cs10 in 0..400 {
        let cs = cs10 as f64 * 0.1;
        for margin10 in 0..60 {
            let margin = margin10 as f64 * 0.1;
            let mut ev = evidence(Rule::Contact);
            ev.closing_speed_mps = cs;
            ev.fault_margin_mps = margin;
            ev.a.closing_into_other_mps = 10.0;
            let mut b = car(2);
            b.closing_into_other_mps = (10.0 - margin).max(0.0);
            ev.b = Some(b);
            let f = steward::judge_contact(&ev, &c);
            assert_eq!(
                f.decision == Decision::Penalised,
                f.penalty.kind != PenaltyKind::None,
                "penalty/decision disagree at closing {cs} margin {margin}: {:?} + {:?}",
                f.decision,
                f.penalty.kind
            );
            assert_eq!(f.decision == Decision::Penalised, f.served_to.is_some());
            assert_eq!(f.decision == Decision::Penalised, f.service != ServicePoint::NotServed);
            checked += 1;
        }
    }
    assert!(checked > 1000, "the sweep must be wide enough to mean something");

    // And the same for the shortcut and rejoin rules.
    for ticks in [0u64, 1, 24, 25, 26, 100, 10_000] {
        for dist in [0.0f64, 0.5, 1.0, 2.0] {
            let mut ev = evidence(Rule::Shortcut);
            ev.a.off_track_ticks = ticks;
            ev.a.max_excursion_m = dist;
            let f = steward::judge_shortcut(&ev, &c);
            assert_eq!(f.decision == Decision::Penalised, f.penalty.kind != PenaltyKind::None);
            for gap in [0.0f64, 5.0, 12.0, 100.0] {
                for closing in [0.0f64, 5.9, 6.0, 30.0] {
                    let mut ev = evidence(Rule::UnsafeRejoin);
                    ev.a.off_track_ticks = ticks;
                    let mut b = car(2);
                    b.gap_m = gap;
                    b.closing_into_other_mps = closing;
                    ev.b = Some(b);
                    ev.closing_speed_mps = closing;
                    let f = steward::judge_rejoin(&ev, &c);
                    assert_eq!(
                        f.decision == Decision::Penalised,
                        f.penalty.kind != PenaltyKind::None
                    );
                    assert_eq!(f.decision == Decision::Penalised, f.served_to.is_some());
                }
            }
        }
    }
}

/// Every ambiguity branch must be reachable and must decline. If a branch were
/// dead code, the "does not silently become certain" claim would be vacuous.
#[test]
fn gate3d_every_inconclusive_branch_is_reachable() {
    let c = cfg();
    let mut branches = 0usize;

    // contact: no second car, inside the band, symmetric
    for ev in [
        { let mut e = evidence(Rule::Contact); e.closing_speed_mps = 40.0; e.a.closing_into_other_mps = 40.0; e },
        { let mut e = evidence(Rule::Contact); e.closing_speed_mps = c.contact_cs_penalty - 1.0; e.b = Some(car(2)); e },
        { let mut e = evidence(Rule::Contact); e.closing_speed_mps = 20.0; e.fault_margin_mps = 0.1; e.b = Some(car(2)); e },
    ] {
        assert_eq!(steward::judge_contact(&ev, &c).decision, Decision::Inconclusive);
        branches += 1;
    }

    // rejoin: already alongside, and the other car off the road
    for other_ticks in [0u64, 500] {
        let mut ev = evidence(Rule::UnsafeRejoin);
        ev.a.off_track_ticks = 500;
        let mut b = car(2);
        b.gap_m = 1.0;
        b.closing_into_other_mps = 12.0;
        b.off_track_ticks = other_ticks;
        ev.b = Some(b);
        ev.closing_speed_mps = 12.0;
        assert_eq!(steward::judge_rejoin(&ev, &c).decision, Decision::Inconclusive);
        branches += 1;
    }
    assert_eq!(branches, 5, "all five declared ambiguity branches must be exercised");
}

// ---------------------------------------------------------------------------
// Build items: the aid records what it saw, and the record is auditable
// ---------------------------------------------------------------------------

/// The recorder must capture position, speed, overlap, track limits and
/// contact events around an incident — the things the plan lists, not a
/// summary verdict.
#[test]
fn gate4_incident_records_carry_the_evidence_the_plan_lists() {
    let rc = RaceConfig {
        cars: 20,
        laps: 10,
        seed: 7,
        steward: Some(cfg()),
        ..Default::default()
    };
    let (outcome, _) = race::run_race(rc, false);
    let incidents = outcome.steward_incidents.clone().unwrap();
    assert!(!incidents.is_empty());

    let mut saw_contact = false;
    for inc in &incidents {
        let e = &inc.evidence;
        // position: recorded as a window, and every car has a lateral offset
        assert!(e.window_end_tick >= e.incident_tick);
        assert!(e.window_start_tick <= e.incident_tick);
        assert!(e.a.lateral_m.is_finite());
        // speed
        assert!(e.a.speed_mps.is_finite() && e.a.max_speed_mps >= e.a.speed_mps);
        assert!(e.sim_time_s >= 0.0 && inc.evidence_hash != 0);
        match inc.rule {
            Rule::Contact => {
                saw_contact = true;
                let b = e.b.as_ref().expect("a contact names both cars");
                assert!(e.a.overlap_m.is_finite() && b.overlap_m.is_finite());
                assert!(e.closing_speed_mps.is_finite());
                assert!(e.a.closing_into_other_mps.is_finite());
            }
            Rule::Shortcut | Rule::UnsafeRejoin => {
                // track limits: how long and how far past the edge
                assert!(e.a.max_excursion_m >= 0.0);
                assert!(e.a.off_track_ticks > 0);
            }
        }
        // decision, rule version and penalty service are always present
        assert!(!inc.basis.is_empty());
        assert_eq!(inc.rule_version, inc.rule.version());
        assert!(Decision::parse(inc.decision.code()).is_some());
    }
    assert!(saw_contact, "no contact was recorded in this race");
}

/// The track-limits path, driven through the recorder directly.
///
/// No race in this engine produces one: measured over 20-car/12-lap and
/// 12-car/10-lap runs, the furthest a car gets past the edge outside the pit
/// lane is about 0.46 m, against a 1.00 m threshold. That is a real property of
/// the driver model, not a gap in the recorder — but it does mean a
/// race-only suite would report the shortcut and rejoin rules as covered while
/// never executing them. Feeding the recorder synthetic samples is what
/// actually exercises the path.
#[test]
fn gate4a_the_recorder_raises_a_track_limits_case_when_a_car_goes_off() {
    let c = cfg();
    let mut rec = steward::Recorder::new(c, 7.0);
    let min = c.rejoin_min_off_ticks;

    // 200 ticks (0.8 s) at 9 m lateral — i.e. 2 m past the 7 m edge — then
    // back on the road, then coasting long enough for the closing car to be
    // seen inside the review window.
    const TOTAL: u64 = 600;
    for tick in 0..TOTAL {
        let off = (10..210).contains(&tick);
        let lat = if off { -9.0 } else { -1.0 };
        // Car 0 rejoins. Car 1 approaches from 60 m away at 40 m/s, so it is
        // inside the review window and well clear of the car it closes on.
        // 200 m back at the rejoin (so there was room to steer away),
        // approaching at 40 m/s and still 120 m away when the window ends.
        let x1 = 200.0 - 40.0 * f64::from(tick as u32) * 0.004;
        rec.observe(
            tick,
            f64::from(tick as u32) * 0.004,
            true,
            &[
                steward::CarSample {
                    id: 0,
                    x: 0.0,
                    y: 0.0,
                    s: 0.0,
                    lateral: lat,
                    speed_mps: 20.0,
                    retired: false,
                    in_pit: false,
                },
                steward::CarSample {
                    id: 1,
                    x: x1,
                    y: 0.0,
                    s: 0.0,
                    lateral: 1.0,
                    speed_mps: 40.0,
                    retired: false,
                    in_pit: false,
                },
            ],
            &[],
        );
    }
    let incidents = rec.finish();
    assert!(
        incidents.len() >= 2,
        "an excursion and a return must raise both rules, got {incidents:?}"
    );

    let shortcut = incidents
        .iter()
        .find(|i| i.rule == Rule::Shortcut)
        .expect("a car 2 m past the edge must raise the shortcut rule");
    let rejoin = incidents
        .iter()
        .find(|i| i.rule == Rule::UnsafeRejoin)
        .expect("a car returning to the road must raise the rejoin rule");

    // Both thresholds of the shortcut rule were met, so this is a penalty.
    assert_eq!(shortcut.decision, Decision::Penalised);
    assert!(shortcut.evidence.a.max_excursion_m >= c.track_limit_dist_m);
    assert!(shortcut.evidence.a.off_track_ticks >= min);
    assert_eq!(shortcut.served_to, Some(0));

    // The rejoin saw a car that could have steered away.
    assert_eq!(rejoin.decision, Decision::Penalised);
    let b = rejoin.evidence.b.as_ref().expect("the rejoin names the closing car");
    assert_eq!(b.car, 1);
    // Two distinct moments, each named: the room at the rejoin (on the
    // joining car) and the closest approach (on the closing car).
    assert!(
        rejoin.evidence.a.gap_m > cfg().rejoin_avoidable_gap_m,
        "the closing car must still have had room when the car rejoined"
    );
    assert!(b.gap_m.is_finite() && b.gap_m > 0.0);
    assert!(
        b.gap_m < rejoin.evidence.a.gap_m,
        "the closest approach must be nearer than the gap at the rejoin"
    );
    assert!(b.closing_into_other_mps > cfg().rejoin_danger_cs);
}

/// The calibration claim, pinned. If the driver model ever starts putting cars
/// a metre off, this fails and the thresholds need revisiting rather than
/// silently still matching nothing.
#[test]
fn gate4b_drivers_stay_within_half_a_metre_of_the_edge_outside_the_pit_lane() {
    let mut worst = 0.0f64;
    let (mut pit_below, mut track_below) = (0u64, 0u64);
    for &(cars, laps, seed) in &[(20u32, 12u32, 7u64), (12, 10, 5)] {
        let mut race = race::Race::new(RaceConfig { cars, laps, seed, ..Default::default() });
        race.start();
        let cap = ((f64::from(laps) * 60.0 + 150.0) / 0.004) as u64;
        let mut n = 0u64;
        while race.state.phase != race::RacePhase::Finished && n < cap {
            race.tick();
            n += 1;
            for e in &race.state.cars {
                let over = e.lateral.abs() - 7.0;
                if over <= 0.0 {
                    continue;
                }
                if e.pit_phase.in_pit() {
                    pit_below += 1;
                } else {
                    track_below += 1;
                    worst = worst.max(over);
                }
            }
        }
    }
    assert!(track_below > 100, "the measurement saw too few on-track excursions to mean anything");
    assert!(
        worst < 0.75,
        "a car reached {worst:.2} m past the edge outside the pit lane; the shortcut threshold \
         ({}) is no longer above normal running, and re-check it against this measurement",
        cfg().track_limit_dist_m
    );
    assert!(
        pit_below > track_below,
        "if the pit lane stopped dominating the excursion histogram, the measurement above is \
         measuring the wrong thing"
    );
}

/// The thresholds ARE the rule, so they travel with the record. A decision
/// whose numbers live only in the binary cannot be audited.
#[test]
fn gate4c_the_thresholds_travel_with_the_record() {
    let rc = RaceConfig {
        cars: 12,
        laps: 6,
        seed: 3,
        steward: Some(cfg()),
        ..Default::default()
    };
    let (outcome, inputs) = race::run_race(rc.clone(), false);
    let dir = std::env::temp_dir().join("f1_p11_thresh");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.simlog");
    race::write_race_log(&path, &rc, &inputs, &outcome).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();

    let d = text.lines().find(|l| l.starts_with("D ")).expect("a D header");
    let toks: Vec<&str> = d.split_whitespace().collect();
    assert_eq!(toks[1], steward::RULE_SET);
    assert_eq!(toks[2].parse::<u32>().unwrap(), steward::RULE_SET_VERSION);
    assert_eq!(
        u64::from_str_radix(toks[4], 16).expect("the D header hash is hex"),
        steward::incidents_hash(&outcome.steward_incidents.clone().unwrap()),
        "the D header must record the hash the reader will check"
    );

    // Every threshold the rules use, written out.
    let c = cfg();
    for want in [
        c.history_s,
        c.review_s,
        c.contact_cs_penalty,
        c.contact_cs_inconclusive,
        c.contact_fault_margin,
        c.track_limit_dist_m,
        c.track_limit_ticks as f64,
        c.rejoin_min_off_ticks as f64,
        c.rejoin_danger_cs,
        c.rejoin_avoidable_gap_m,
        c.penalty_seconds,
    ] {
        assert!(
            toks[5..].iter().any(|t| t.parse::<f64>().is_ok_and(|v| (v - want).abs() < 1e-9)),
            "threshold {want} is not in the D header: {d}"
        );
    }

    // And the reader rebuilds them from the file, not from the binary.
    let (_, replayed) = race::replay_race_log(&path).unwrap();
    let back = replayed.steward_config.expect("the replay rebuilt a config");
    assert_eq!(back, c, "the log must round-trip the exact thresholds that ran");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The pit lane sits outside the track edge by design. Every routine pit stop
/// would otherwise be recorded as a shortcut, which is the first thing that
/// would make the aid a lie about its own track.
#[test]
fn gate4d_the_pit_lane_is_not_a_track_limits_case() {
    let rc = RaceConfig {
        cars: 12,
        laps: 8,
        seed: 3,
        steward: Some(cfg()),
        ..Default::default()
    };
    let (outcome, _) = race::run_race(rc, false);
    assert!(
        !outcome.pit_stops.is_empty(),
        "this race must stop in the lane, or the test proves nothing"
    );
    for inc in outcome.steward_incidents.clone().unwrap() {
        assert!(
            !inc.evidence.a.in_pit,
            "car {} was recorded for a case while inside the pit lane",
            inc.evidence.a.car
        );
        assert!(
            inc.evidence.a.max_excursion_m <= 1.0 || inc.evidence.a.off_track_ticks == 0,
            "an excursion in the lane must not become a track-limits case"
        );
    }
}

/// The aid is an observer. A stewarded race and a plain one must produce the
/// same trajectory, or every pinned hash in the repository is conditional on a
/// feature most runs do not use.
#[test]
fn gate5_stewarding_cannot_change_the_trajectory() {
    let base = RaceConfig { cars: 16, laps: 8, seed: 21, ..Default::default() };
    let (plain, _) = race::run_race(base.clone(), false);
    let mut stewarded_cfg = base.clone();
    stewarded_cfg.steward = Some(cfg());
    let (stewarded, _) = race::run_race(stewarded_cfg, false);

    assert_eq!(
        plain.state_hash,
        stewarded.state_hash,
        "the stewarding aid reached the trajectory; it must be unreachable from mix_state"
    );
    assert_eq!(plain.ticks, stewarded.ticks);
    assert_eq!(plain.classification.len(), stewarded.classification.len());
    // ...and it did actually run, so the comparison is not vacuous.
    assert!(stewarded.steward_incidents.is_some());

    // Inert configs are normalised away, so "enabled but inert" cannot smuggle
    // in an observer either.
    let mut inert = base.clone();
    inert.steward = Some(StewardConfig::default());
    let (out, _) = race::run_race(inert, false);
    assert!(out.steward_incidents.is_none(), "an inert config must record nothing");
}

/// Penalties move the order of the results but never edit the classification
/// itself: the race that was simulated is the race that is reported.
#[test]
fn gate6_penalties_reorder_a_separate_view_and_never_edit_the_classification() {
    let rc = RaceConfig {
        cars: 20,
        laps: 12,
        seed: 7,
        steward: Some(cfg()),
        ..Default::default()
    };
    let (outcome, _) = race::run_race(rc, false);
    let summary = outcome.steward.clone().expect("summary");
    let with = summary.post_penalty_order(&outcome.classification);

    assert_eq!(
        with.len(),
        outcome.classification.len(),
        "the post-penalty view must cover the same field"
    );
    let mut sorted = with.clone();
    let mut seen = std::collections::BTreeSet::new();
    for id in &sorted {
        assert!(seen.insert(*id), "car {id} appears twice in the post-penalty order");
    }
    sorted.sort_unstable();
    let mut plain: Vec<u32> = outcome.classification.iter().map(|r| r.id).collect();
    plain.sort_unstable();
    assert_eq!(sorted, plain, "the post-penalty order must be a permutation of the field");

    // The classification itself is untouched, and its positions are what the
    // penalty view sorts on first.
    for row in &outcome.classification {
        assert!(with.contains(&row.id));
    }
    if summary.time_penalty_seconds > 0.0 {
        assert_ne!(
            with,
            outcome.classification.iter().map(|r| r.id).collect::<Vec<_>>(),
            "with time served, the two orders should differ"
        );
    }
}
