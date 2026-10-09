//! Phase 5 completion-gate tests, mapped to the plan's gate criteria:
//!
//! 1. "A seeded race starts, completes, and produces a correct
//!    classification."
//! 2. "Pit stops and tire choices affect the result in repeatable ways."
//! 3. "A stopped/retired car is represented consistently."
//!
//! Plus build items: race weekend (grid formation, start sequence, lap
//! counting, finish, results), basic tire wear and compound state, pit
//! entry / speed limit / service / exit / time loss, and practice,
//! qualifying, and race as separate session modes.

use apexsim::race::{
    PitPhase, PitStrategy, Race, RaceConfig, RacePhase, RcEvent, SessionMode, TireCompound,
};

fn cfg(cars: u32) -> RaceConfig {
    RaceConfig { cars, laps: 3, seed: 42, ..Default::default() }
}

/// Drive a session from `start()` to `Finished`, live path.
fn run_to_end(cfg: &RaceConfig) -> Race {
    let mut race = Race::new(cfg.clone());
    race.start();
    let cap = ((cfg.laps as f64 * 60.0 + 120.0 + 30.0) / 0.004) as u64; // generous tick bound
    let mut n = 0u64;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
    }
    assert_eq!(race.state.phase, RacePhase::Finished, "session never finished");
    race
}

// ---------------------------------------------------------------------------
// Gate 1: a seeded race starts (lights), completes, classifies correctly.
// ---------------------------------------------------------------------------

#[test]
fn gate1_start_sequence_lights_then_green() {
    let mut race = Race::new(cfg(4));
    race.start();
    assert_eq!(race.state.phase, RacePhase::Formation, "race mode opens in formation");
    assert!(race.state.green_tick > 0, "green tick resolved at construction");

    // The locked grid: no motion while the lights count.
    let start_progress: Vec<f64> = race.state.cars.iter().map(|e| e.total_progress).collect();
    let mut light_seq: Vec<u8> = Vec::new();
    let mut green_seen = false;
    let mut seen_events = 0usize;
    for _ in 0..2500 {
        race.tick();
        // Consume only NEW race-control events (the vec is append-only).
        for ev in &race.state.rc_events[seen_events..] {
            match ev {
                RcEvent::Lights(n) => light_seq.push(*n),
                RcEvent::Green => green_seen = true,
                _ => {}
            }
        }
        seen_events = race.state.rc_events.len();
        if race.state.phase == RacePhase::Running {
            break;
        }
    }
    assert_eq!(light_seq, vec![1, 2, 3, 4, 5], "lights must count 1..5");
    assert!(green_seen, "green flag event must fire");
    assert_eq!(race.state.phase, RacePhase::Running);

    // Grid was locked: cars have not moved a metre before green.
    for (i, e) in race.state.cars.iter().enumerate() {
        let travelled = e.total_progress - start_progress[i];
        assert!(travelled.abs() < 0.01, "car {} moved {travelled} m during formation", e.id);
        assert!(e.car.vel_long < 0.1, "car {} rolled during formation", e.id);
    }

    // And they launch: every car is under power shortly after green.
    for _ in 0..300 {
        race.tick();
    }
    for e in &race.state.cars {
        assert!(e.car.vel_long > 1.0, "car {} did not launch at green", e.id);
    }
}

#[test]
fn gate1b_green_tick_is_seed_deterministic() {
    let a = Race::new(RaceConfig { seed: 42, ..cfg(4) });
    let b = Race::new(RaceConfig { seed: 42, ..cfg(4) });
    let c = Race::new(RaceConfig { seed: 43, ..cfg(4) });
    assert_eq!(a.state.green_tick, b.state.green_tick, "same seed, same hold");
    assert_ne!(a.state.green_tick, c.state.green_tick, "hold varies with seed");
    // Both runs reproduce the identical early trajectory through green.
    let mut ra = a;
    let mut rb = b;
    ra.start();
    rb.start();
    for _ in 0..2200 {
        ra.tick();
        rb.tick();
    }
    assert_eq!(ra.state_hash(), rb.state_hash());
}

#[test]
fn gate1c_seeded_race_completes_with_correct_classification() {
    let cfg = cfg(20); // laps 3, default Auto strategy, formation on
    let race = run_to_end(&cfg);
    // Natural finish: under the red-flag cap (laps*60 + 120 s).
    assert!(
        race.state.sim_time < cfg.laps as f64 * 60.0 + 120.0,
        "race hit the red-flag cap at {:.1} s",
        race.state.sim_time
    );
    let lb = &race.state.leaderboard;
    assert_eq!(lb.len(), 20, "whole field classified");
    let positions: Vec<u32> = lb.iter().map(|r| r.pos).collect();
    let mut sorted = positions.clone();
    sorted.sort();
    assert_eq!(positions, sorted, "positions must be 1..=20 in order");

    // Winner completed the race distance.
    assert_eq!(lb[0].laps_done, cfg.laps, "P1 must have all laps done");

    // DNF cars (retired before the flag) at the tail. A car retired AFTER
    // taking the flag stays classified among finishers by its flag time.
    let is_dnf = |r: &apexsim::race::LeaderRow| r.retired && !r.finished;
    let first_dnf = lb.iter().position(&is_dnf).unwrap_or(lb.len());
    assert!(
        lb[first_dnf..].iter().all(is_dnf),
        "DNF cars must occupy the tail of the classification"
    );

    // Finished cars in increasing finish-time order (winner = min time).
    let fin: Vec<(u32, u32, Option<f64>)> = lb.iter().map(|r| {
        let e = race.state.cars.iter().find(|e| e.id == r.id).unwrap();
        (r.pos, r.id, e.finish_time_s)
    }).collect();
    let mut last = 0.0;
    for (_, id, t) in fin.iter().take_while(|(_, _, t)| t.is_some()) {
        let t = t.unwrap();
        assert!(t >= last, "finished cars out of time order: car {id} at {t:.2} after {last:.2}\n{fin:?}");
        last = t;
    }

    // Results: a session-fastest lap that really is someone's best lap.
    let (fcar, ftime, flap) = race.state.fastest.expect("a completed race has a fastest lap");
    assert!(ftime > 0.0 && flap >= 1);
    let row = lb.iter().find(|r| r.id == fcar).expect("fastest-lap car classified");
    assert_eq!(row.best_lap_s, Some(ftime), "fastest lap must equal that car's best lap");
}

// ---------------------------------------------------------------------------
// Gate 2: pit stops and tire choices affect the result, repeatably.
// ---------------------------------------------------------------------------

#[test]
fn gate2_pit_stops_happen_lose_time_and_repeat_bit_exactly() {
    let mut no_stops = cfg(10);
    no_stops.laps = 8;
    no_stops.strategy = PitStrategy::None;
    let mut auto = no_stops.clone();
    auto.strategy = PitStrategy::Auto;

    // A: no stops.
    let (out_a, _) = apexsim::race::run_race(no_stops.clone(), false);
    assert!(out_a.pit_stops.is_empty(), "strategy None must not stop");

    // B: one-stop Auto — run twice; the result must repeat bit-exactly.
    let (out_b1, _) = apexsim::race::run_race(auto.clone(), false);
    let (out_b2, _) = apexsim::race::run_race(auto.clone(), false);
    assert_eq!(out_b1.state_hash, out_b2.state_hash, "pit strategy must replay identically");
    assert_eq!(out_b1.classification.len(), out_b2.classification.len());
    for (r1, r2) in out_b1.classification.iter().zip(out_b2.classification.iter()) {
        assert_eq!(r1.id, r2.id, "classification must repeat exactly");
        assert_eq!(r1.gap_s, r2.gap_s);
    }

    // B actually stopped, and every stop cost real lane time.
    assert!(out_b1.pit_stops.len() >= 5, "expected most of the 10-car field to stop, got {}",
        out_b1.pit_stops.len());
    for s in &out_b1.pit_stops {
        let lane_s = s.exited_t - s.entered_t;
        assert!(lane_s >= 5.0, "stop for car {} took only {lane_s:.1} s in lane", s.car);
        assert!((s.held_s - auto.pit.service_time_s).abs() < 1e-9, "held time must equal config");
    }

    // Stops affect the result: the leader's total race time differs.
    let total = |laps: &[f64]| laps.iter().sum::<f64>();
    assert!(
        (total(&out_a.leader_lap_times) - total(&out_b1.leader_lap_times)).abs() > 3.0,
        "pit strategy must change the leader's race time"
    );

    // And the mechanism is visible per car: the stop lap is the slow lap.
    let mut race = Race::new(auto.clone());
    race.start();
    while race.state.phase != RacePhase::Finished {
        race.tick();
    }
    for e in &race.state.cars {
        if let (Some(lap), Some(best)) = (e.stop_lap, e.best_lap) {
            if e.stops_done > 0 {
                let stop_lap_time = e.lap_times[(lap - 1) as usize];
                assert!(
                    stop_lap_time > best + 5.0,
                    "car {} stop lap {} was {:.1} s vs best {:.1} s — no time loss?",
                    e.id, lap, stop_lap_time, best
                );
            }
        }
    }
}

#[test]
fn gate2b_tire_compound_and_wear_change_pace() {
    // Compound: same seed, Soft vs Hard, no stops, no formation noise —
    // soft must be measurably faster, and each choice must repeat exactly.
    let base = RaceConfig {
        cars: 10,
        laps: 3,
        seed: 7,
        strategy: PitStrategy::None,
        ..Default::default()
    };
    let run = |c: TireCompound| {
        let cfg = RaceConfig { start_compound: c, ..base.clone() };
        let (o1, _) = apexsim::race::run_race(cfg.clone(), false);
        let (o2, _) = apexsim::race::run_race(cfg, false);
        assert_eq!(o1.state_hash, o2.state_hash, "{:?} must be deterministic", c);
        o1
    };
    let soft = run(TireCompound::Soft);
    let hard = run(TireCompound::Hard);
    assert_ne!(soft.state_hash, hard.state_hash, "compound must change the trajectory");
    let best_soft = soft.classification.iter().filter_map(|r| r.best_lap_s).fold(f64::MAX, f64::min);
    let best_hard = hard.classification.iter().filter_map(|r| r.best_lap_s).fold(f64::MAX, f64::min);
    assert!(
        best_soft < best_hard - 0.3,
        "soft best {best_soft:.2} s must beat hard best {best_hard:.2} s"
    );

    // Wear: a long solo stint on Soft degrades — late laps slower than
    // early ones (1-car field, clean air, deterministic driver noise).
    let stint = RaceConfig {
        cars: 1,
        laps: 12,
        seed: 5,
        strategy: PitStrategy::None,
        ..Default::default()
    };
    let (out, _) = apexsim::race::run_race(stint, false);
    assert_eq!(out.leader_lap_times.len(), 12, "12 timed laps");
    let early: f64 = out.leader_lap_times[1..4].iter().sum::<f64>() / 3.0;
    let late: f64 = out.leader_lap_times[9..12].iter().sum::<f64>() / 3.0;
    assert!(
        late > early + 0.05,
        "worn soft tires must cost time: early avg {early:.2} vs late avg {late:.2}"
    );
}

// ---------------------------------------------------------------------------
// Gate 2 (build items): session modes.
// ---------------------------------------------------------------------------

#[test]
fn gate3_qualifying_and_practice_classify_by_best_lap() {
    for mode in [SessionMode::Qualifying, SessionMode::Practice] {
        let cfg = RaceConfig { cars: 10, laps: 3, seed: 7, mode, ..Default::default() };
        let (out, _) = apexsim::race::run_race(cfg, false);
        assert_eq!(out.mode, mode.label());
        // Timed sessions don't take race stops.
        assert!(out.pit_stops.is_empty(), "{:?} must not pit", mode);
        // Classification IS the best-lap order: timed rows ascending, and
        // rows without a time (incl. retired ones) after them. A retired
        // car with a valid time keeps its place — the DNF flag rides along.
        let mut seen_best = 0.0f64;
        let mut seen_nolap = false;
        for r in &out.classification {
            match r.best_lap_s {
                Some(b) => {
                    assert!(!seen_nolap, "timed cars must precede no-time cars");
                    assert!(
                        b >= seen_best,
                        "best-lap classification violated: {b:.2} after {seen_best:.2}"
                    );
                    seen_best = b;
                }
                None => seen_nolap = true,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gate 3: a stopped/retired car is represented consistently.
// ---------------------------------------------------------------------------

#[test]
fn gate4_retired_cars_are_consistently_represented() {
    // Low retire threshold → guaranteed incidents (any real touch retires).
    let cfg = RaceConfig {
        cars: 20,
        laps: 2,
        seed: 99,
        contact_retire_cs: 1.5,
        ..Default::default()
    };
    let mut race = Race::new(cfg.clone());
    race.start();
    let cap = ((cfg.laps as f64 * 60.0 + 120.0 + 30.0) / 0.004) as u64; // generous tick bound
    let mut n = 0u64;
    // Car -> (tick it retired, speed at that tick). Recorded because "a
    // retired car has stopped" is only a meaningful claim about a car that
    // had time to stop — one can retire on the flag itself.
    let mut retired_at: std::collections::BTreeMap<u32, (u64, f64)> = Default::default();
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        for e in &race.state.cars {
            if e.retired {
                retired_at.entry(e.id).or_insert((n, e.car.vel_long));
            }
        }
    }
    // 5.0 s of scrubbing at DT = 4 ms. Measured, not guessed: in this race
    // every car retiring at least this long before the flag had stopped, and
    // the one retiring 3.6 s before it had not.
    const SCRUB_TICKS: u64 = 1250;

    let retired: Vec<u32> = race.state.cars.iter().filter(|e| e.retired).map(|e| e.id).collect();
    assert!(!retired.is_empty(), "low retire threshold must produce retirements");

    for e in &race.state.cars {
        if e.retired {
            let (tick, v_at) = retired_at[&e.id];
            let since = n.saturating_sub(tick);
            if since >= SCRUB_TICKS {
                assert!(
                    e.car.vel_long <= 0.5,
                    "retired car {} still rolling at {:.2} m/s {} ticks after retiring",
                    e.id,
                    e.car.vel_long,
                    since
                );
            } else {
                // Too little elapsed time to have scrubbed to a stop. What
                // must still hold is that retirement is terminal: the car
                // coasts and never picks speed back up.
                assert!(
                    e.car.vel_long <= v_at + 1e-9,
                    "retired car {} accelerated after retiring ({:.3} -> {:.3} m/s)",
                    e.id,
                    v_at,
                    e.car.vel_long
                );
            }
            // No stop left half-done: every completed service has its
            // lane-exit record. (A car that stopped on lap 1 and retired on
            // lap 2 is fine — stops_done may exceed zero. A car retired IN the
            // lane is a lane hazard — consistent only if its service never
            // began.)
            if e.pit_phase.in_pit() {
                assert_eq!(e.stops_done, 0, "retired car {} in lane with a served stop", e.id);
                assert!(e.pit_stops.is_empty(), "retired car {} in lane with stop records", e.id);
            }
            assert_eq!(
                e.pit_stops.len(),
                e.stops_done as usize,
                "retired car {} has an incomplete pit sequence",
                e.id
            );
            assert!(e.car.pos_x.is_finite() && e.car.vel_long.is_finite());
        }
    }
    // And the classification agrees: DNF rows (retired before the flag) at
    // the tail; a retired row showing PIT never shows a completed stop.
    let lb = &race.state.leaderboard;
    let is_dnf = |r: &apexsim::race::LeaderRow| r.retired && !r.finished;
    let first_dnf = lb.iter().position(&is_dnf).unwrap_or(lb.len());
    assert!(lb[first_dnf..].iter().all(is_dnf), "DNF rows must be the tail");
    assert!(lb.iter().filter(|r| r.retired).all(|r| !r.in_pit || r.stops == 0));
    assert_eq!(lb.iter().filter(|r| r.retired).count(), retired.len());
}

// ---------------------------------------------------------------------------
// Replay: every phase-5 session type replays bit-identically from its log,
// and the phase-4 v2 artifact (legacy format) still replays.
// ---------------------------------------------------------------------------

#[test]
fn gate5_replay_round_trip_race_formation_pits() {
    let cfg = cfg(20); // race mode: formation + lights + Auto stops
    let (outcome, inputs) = apexsim::race::run_race(cfg.clone(), false);
    let dir = std::env::temp_dir().join(format!("apexsim_p5_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("race20_v3.simlog");
    apexsim::race::write_race_log(&path, &cfg, &inputs, &outcome).unwrap();
    let (ok, replayed) = apexsim::race::replay_race_log(&path).unwrap();
    assert!(
        ok,
        "v3 race replay diverged: {:016x} vs {:016x}",
        replayed.state_hash, outcome.state_hash
    );
    assert!(!replayed.pit_stops.is_empty(), "replay must reproduce the pit log");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gate5b_replay_round_trip_qualifying() {
    let cfg = RaceConfig { cars: 10, laps: 3, seed: 9, mode: SessionMode::Qualifying, ..Default::default() };
    let (outcome, inputs) = apexsim::race::run_race(cfg.clone(), false);
    let dir = std::env::temp_dir().join(format!("apexsim_p5q_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("quali10.simlog");
    apexsim::race::write_race_log(&path, &cfg, &inputs, &outcome).unwrap();
    let (ok, replayed) = apexsim::race::replay_race_log(&path).unwrap();
    assert!(ok, "qualifying replay diverged: {:016x} vs {:016x}",
        replayed.state_hash, outcome.state_hash);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gate5c_legacy_v2_artifact_still_replays() {
    // The phase-4 artifact has no M line: it must replay as a legacy
    // session (no formation, no stops) — bit-identical, as before.
    //
    // Its pinned state hash moved once: f704e58130814f75 -> 7d07a37e51c2e69e,
    // when Phase 11 fixed car j's world-frame lateral velocity in the contact
    // pass (it used `uj * jhx` where it needed `vj * jhx`, inflating the
    // component by up to sqrt(2)). The artifact was regenerated rather than
    // relaxed: this gate exists to catch exactly that class of change, and a
    // fixture pinned to a known-wrong physics would stop being able to.
    let path = std::path::Path::new("artifacts/phase4_race_20car.simlog");
    let (ok, _outcome) = apexsim::race::replay_race_log(path).expect("artifact readable");
    assert!(ok, "phase-4 v2 artifact replay must still PASS (back-compat)");
}

// ---------------------------------------------------------------------------
// Pit-lane mechanics: the lane state machine, seen from the field.
// ---------------------------------------------------------------------------

#[test]
fn gate6_pit_lane_transitions_obey_limits_and_service() {
    let cfg = RaceConfig {
        cars: 6,
        laps: 4,
        seed: 3,
        strategy: PitStrategy::Fixed { lap: 2, switch_to: TireCompound::Hard },
        ..Default::default()
    };
    let mut race = Race::new(cfg);
    race.start();
    let mut saw_in_lane_slow = false;
    let mut saw_service = false;
    let cap = ((4.0 * 60.0 + 150.0) / 0.004) as u64;
    let mut n = 0u64;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        for e in &race.state.cars {
            if let PitPhase::InLane = e.pit_phase {
                assert!(
                    e.car.vel_long <= race.config.pit.speed_limit + 1.0,
                    "car {} exceeded the pit limit: {:.1} m/s",
                    e.id,
                    e.car.vel_long
                );
                saw_in_lane_slow = true;
            }
            if let PitPhase::Servicing { remaining_s } = e.pit_phase {
                saw_service = true;
                assert!(remaining_s >= 0.0);
                // Well into the stop the car must be standing (it brakes
                // from the pit limit over the first moments of service).
                if remaining_s < race.config.pit.service_time_s * 0.5 {
                    assert!(
                        e.car.vel_long <= 0.5,
                        "servicing car {} still rolling at {:.1} m/s mid-stop",
                        e.id,
                        e.car.vel_long
                    );
                }
            }
        }
    }
    assert!(saw_in_lane_slow, "no car was ever observed in the pit lane");
    assert!(saw_service, "no car was ever observed being serviced");
    assert_eq!(race.state.phase, RacePhase::Finished);
    // Every planned stop executed, onto the fixed compound, wear reset.
    for e in &race.state.cars {
        if e.finish_time_s.is_some() {
            assert!(e.stops_done >= 1, "finished car {} missed its planned stop", e.id);
            assert_eq!(e.compound, TireCompound::Hard, "car {} must wear the Hard after its stop", e.id);
        }
    }
    // Race-control log records the full lane journey.
    let entered = race.state.rc_events.iter().filter(|e| matches!(e, RcEvent::PitEntered { .. })).count();
    let exited = race.state.rc_events.iter().filter(|e| matches!(e, RcEvent::PitExited { .. })).count();
    assert!(entered >= 5, "pit entries logged: {entered}");
    assert_eq!(entered, exited, "every entry must have an exit (or be still in flight)");
}
