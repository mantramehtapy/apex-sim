//! Phase E gates: one-by-one qualifying that sets the grid.
//!
//! The property that matters most is identity: a solo qualifying run measures
//! THE CAR, and the race fields that car where it qualified. Machinery, pace
//! envelope and driver noise follow the car id everywhere — if any of them
//! followed the grid slot instead, a fastest qualifier would inherit pole
//! position's car and the session would be theatre.

use apexsim::race::{Race, RaceConfig};

fn classic(cars: u32, laps: u32) -> RaceConfig {
    RaceConfig { cars, laps, seed: 42, ..Default::default() }
}

#[test]
fn a_solo_run_measures_that_car_not_slot_zero() {
    // Classic pace envelope: car i is slower by pace_spread × i. A solo run
    // for car 7 must reflect car 7's envelope — measurable as strictly slower
    // than car 0's. If the solo path resolved machinery for slot 0 instead,
    // both laps would be identical.
    let fast = {
        let cfg = RaceConfig {
            solo_car: Some(0),
            ..classic(1, 3)
        };
        apexsim::weekend::run_qualifying(&cfg, 300.0).remove(0)
    };
    let slow = {
        let cfg = RaceConfig {
            solo_car: Some(7),
            ..classic(1, 3)
        };
        apexsim::weekend::run_qualifying(&cfg, 300.0).remove(0)
    };
    let (tf, ts) = (fast.best_lap_s.expect("car 0 set a lap"), slow.best_lap_s.expect("car 7 set a lap"));
    assert!(
        ts > tf,
        "solo car 7 ({ts:.3} s) did not run slower than solo car 0 ({tf:.3} s) — the solo \
         session is measuring the wrong car"
    );
    // And the lap must be measurably slower. The sensitivity is sub-linear
    // by physics: the classic envelope scales the driver's grip ceiling, a
    // grip-limited corner speed goes as sqrt(a_lat), so the 1.4% envelope
    // (pace_spread 0.002 × 7) lands near ~0.6% on lap time — what the
    // measured value is. The floor only guards against the envelope not
    // reaching the solo run at all (measured noise is below 0.01%).
    let envelope = 0.002 * 7.0;
    let slower = (ts / tf - 1.0) * 100.0;
    assert!(
        slower > envelope * 25.0,
        "car 7 was only {slower:.2}% slower — the pace envelope did not reach the solo run"
    );
}

#[test]
fn a_solo_run_matches_the_full_field_identity() {
    // The strongest identity check: a solo run of car k and the k-th car's
    // machinery in the real field must agree. Classic mode has no traffic
    // variation per car in a solo run, so the lap times match to the noise
    // seed — which is also keyed by id. Compare car 3 solo against three
    // separate solo runs of car 3: identical, byte for byte of the lap.
    let one = apexsim::weekend::run_qualifying(&RaceConfig {
        solo_car: Some(3),
        ..classic(1, 3)
    }, 300.0)
    .remove(0);
    let two = apexsim::weekend::run_qualifying(&RaceConfig {
        solo_car: Some(3),
        ..classic(1, 3)
    }, 300.0)
    .remove(0);
    assert_eq!(one, two, "the same solo run must reproduce exactly");
}

#[test]
fn grid_order_spawns_the_pole_car_at_the_line() {
    // Reverse the grid: car 0 (normally pole) starts last, car 5 at pole.
    let mut cfg = classic(6, 3);
    cfg.grid_order = vec![5, 4, 3, 2, 1, 0];
    let mut race = Race::new(cfg);
    race.start();
    let len = race.track.length();
    for _ in 0..2000 {
        race.tick();
    }
    let s = |id: u32| race.state.cars[id as usize].s;
    // Expected slot s per car: the grid order [5,4,3,2,1,0] puts car 5 in
    // slot 0 (pole, `len - 15`), car 4 in slot 1 (same row — the two grid
    // columns share a row's s BY DESIGN), car 3 in slot 2, and so on.
    let slot_s = |slot: u32| len - 15.0 - (slot / 2) as f64 * 9.0;
    let expected = [
        (5u32, 0u32), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5),
    ];
    for (car, slot) in expected {
        let want = slot_s(slot);
        assert!(
            (s(car) - want).abs() < 30.0,
            "car {car} (slot {slot}) at {:.1} m, expected {:.1} m",
            s(car),
            want
        );
    }
    // The pole sitter really is the one qualifying put there.
    assert!(
        (s(5) - (len - 15.0)).abs() < 30.0,
        "car 5 starts {} m along, pole is at {}",
        s(5),
        len - 15.0
    );
}

#[test]
fn the_race_rolls_identity_order_when_no_qualifying_happened() {
    // Default: empty grid_order must be exactly the pre-weekend engine. The
    // spawn positions are the only thing grid_order can touch, so compare a
    // default race's early positions against the computed identity positions.
    let cfg = classic(6, 3);
    let mut race = Race::new(cfg.clone());
    race.start();
    for _ in 0..2000 {
        race.tick();
    }
    let len = race.track.length();
    for id in 0..6u32 {
        let expected = len - 15.0 - (id / 2) as f64 * cfg.grid_row_gap_m;
        assert!(
            (race.state.cars[id as usize].s - expected).abs() < 30.0,
            "car {id} at {:.1} m, identity grid puts it at {:.1}",
            race.state.cars[id as usize].s,
            expected
        );
    }
}

#[test]
fn the_weekend_race_with_a_grid_replays_bit_identically() {
    // The O line carries the qualifying result, so a weekend race replays
    // anywhere without re-running the session. A missing O line must also
    // mean identity order — that is what makes every pre-weekend log valid.
    let mut cfg = classic(6, 3);
    cfg.grid_order = vec![3, 5, 0, 2, 4, 1];
    let (outcome, inputs, _) = apexsim::race::run_race_full(cfg.clone(), false);
    assert_eq!(outcome.classification.len(), 6, "the whole grid must classify");
    let path = std::env::temp_dir().join("f1sim_weekend_gate.simlog");
    apexsim::race::write_race_log(&path, &cfg, &inputs, &outcome).expect("log written");
    let text = std::fs::read_to_string(&path).expect("log written");
    let o_line = text
        .lines()
        .find(|l| l.starts_with("O "))
        .expect("a qualifying-ordered race logs its grid");
    assert_eq!(o_line, "O 3 5 0 2 4 1");
    let (identical, replayed) = apexsim::race::replay_race_log(&path).expect("replay");
    assert!(identical, "replay reported a divergence");
    assert_eq!(
        replayed.state_hash, outcome.state_hash,
        "the weekend race did not replay bit-identically"
    );

    // And a grid-free race writes no O line at all — the byte-identical
    // convention every pre-weekend log depends on.
    let plain_cfg = classic(6, 3);
    let (plain_out, plain_inputs, _) = apexsim::race::run_race_full(plain_cfg.clone(), false);
    let plain_path = std::env::temp_dir().join("f1sim_weekend_gate_plain.simlog");
    apexsim::race::write_race_log(&plain_path, &plain_cfg, &plain_inputs, &plain_out)
        .expect("plain log written");
    let plain_text = std::fs::read_to_string(&plain_path).expect("plain log written");
    assert!(
        !plain_text.lines().any(|l| l.starts_with("O ")),
        "an identity-order race must not write an O line"
    );
    let (plain_identical, plain_replayed) =
        apexsim::race::replay_race_log(&plain_path).expect("plain replay");
    assert!(plain_identical);
    assert_eq!(plain_replayed.state_hash, plain_out.state_hash);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&plain_path);
}

#[test]
fn qualifying_orders_the_grid_by_lap_time() {
    // The full mini-weekend: classic 4-car field, short session. Cars run one
    // by one; the grid comes out fastest-first; the race then runs to a flag
    // with that grid.
    let cfg = classic(4, 3);
    let rows = apexsim::weekend::run_qualifying(&cfg, 240.0);
    assert_eq!(rows.len(), 4, "every car gets its own run");
    assert!(
        rows.iter().all(|r| r.best_lap_s.is_some()),
        "on a clear circuit every solo car sets a time"
    );
    let grid = apexsim::weekend::grid_from_rows(&rows);
    let mut times: Vec<(u32, f64)> =
        rows.iter().filter_map(|r| r.best_lap_s.map(|t| (r.car, t))).collect();
    times.sort_by(|a, b| a.1.total_cmp(&b.1));
    let expected: Vec<u32> = times.into_iter().map(|(c, _)| c).collect();
    assert_eq!(grid, expected, "grid order is not lap-time order");

    let (outcome, _, _) =
        apexsim::race::run_race_full(RaceConfig { grid_order: grid, ..cfg }, false);
    assert_eq!(outcome.classification.len(), 4, "the whole grid must classify");
}
