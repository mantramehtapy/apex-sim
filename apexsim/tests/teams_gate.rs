//! Team-field gates: the 2026 team machinery must change the physics, must
//! agree with the driver envelope, must stay deterministic, must round-trip
//! the log format, and must leave classic mode exactly as it was.

use apexsim::race::{FieldMode, Race, RaceConfig, RacePhase};
use apexsim::teams;

fn cfg(field: FieldMode, seed: u64) -> RaceConfig {
    RaceConfig { cars: 20, laps: 3, seed, field, ..RaceConfig::default() }
}

fn running(field: FieldMode, seed: u64) -> Race {
    let mut race = Race::new(cfg(field, seed));
    race.state.phase = RacePhase::Running;
    race
}

#[test]
fn classic_field_is_physically_identical() {
    let race = running(FieldMode::Classic, 7);
    let p0 = race.params_of(0);
    let p15 = race.params_of(15);
    assert_eq!(p0.engine_force_max, p15.engine_force_max, "classic engines must match");
    assert_eq!(p0.drag_area, p15.drag_area, "classic drag must match");
    assert_eq!(p0.brake_force_max, p15.brake_force_max, "classic brakes must match");
    assert_eq!(p0.tire_grip, p15.tire_grip, "classic tyres must match");
}

#[test]
fn teams_field_gives_teams_their_own_machinery() {
    let race = running(FieldMode::Teams, 7);
    // The file's first two teams (ids 0/1 vs 2/3) have different engines,
    // and top speed maps to drag.
    let a = race.params_of(0);
    let b = race.params_of(2);
    assert_ne!(a.engine_force_max, b.engine_force_max);
    assert_ne!(a.drag_area, b.drag_area);
    // Both cars of one team share the machinery: every attribute identical.
    let a0 = race.params_of(0);
    let a1 = race.params_of(1);
    assert_eq!(a0.engine_force_max, a1.engine_force_max);
    assert_eq!(a0.drag_area, a1.drag_area);
    assert_eq!(a0.brake_force_max, a1.brake_force_max);
    assert_eq!(a0.tire_grip, a1.tire_grip);
}

#[test]
fn top_speed_maps_to_drag_through_the_cube() {
    // Power-limited vmax ~ (P/cdA)^(1/3): dividing drag by s^3 raises vmax
    // by exactly s. The driver envelope takes s directly, so this is the
    // check that the physics half uses the cube and not the raw multiplier.
    let s = teams::for_car(0).top_speed;
    let base = apexsim::physics::CarParams::default().drag_area;
    let race = running(FieldMode::Teams, 7);
    let p = race.params_of(0);
    let ratio = base / p.drag_area;
    let want = s * s * s;
    assert!((ratio - want).abs() < 1e-12, "drag ratio {ratio} != top_speed^3 {want}");
}

#[test]
fn driver_and_physics_agree_on_tyre_grip() {
    // The invariant that keeps Teams mode drivable: the grip the physics
    // grants (params_of's tyre factor, relative to base grip) is exactly the
    // grip the driver aims at (grip_scale) — car for car.
    let race = running(FieldMode::Teams, 9);
    let base_grip = apexsim::physics::CarParams::default().tire_grip;
    for i in 0..20usize {
        let physics = race.params_of(i).tire_grip / base_grip;
        let driver = race.grip_scale_of(i);
        assert!(
            ((physics - driver) / driver).abs() < 1e-9,
            "car {i}: physics tyre factor {physics} != driver grip scale {driver}"
        );
    }
}

#[test]
fn teams_mode_changes_results() {
    // Same seed, same everything but the field: the teams race must differ
    // from the classic one (machinery now decides), and its best laps must
    // span a real spread across the field.
    let run = |field| {
        let (outcome, _) = apexsim::race::run_race(cfg(field, 11), false);
        outcome
    };
    let classic = run(FieldMode::Classic);
    let teams = run(FieldMode::Teams);
    assert_ne!(classic.state_hash, teams.state_hash, "team machinery must change the racing");
    let laps: Vec<f64> = teams.classification.iter().filter_map(|r| r.best_lap_s).collect();
    assert!(!laps.is_empty());
    let spread = laps.iter().cloned().fold(f64::MIN, f64::max)
        - laps.iter().cloned().fold(f64::MAX, f64::min);
    assert!(spread > 0.1, "teams field best-lap spread only {spread:.3} s");
}

#[test]
fn seeded_form_is_deterministic_per_seed_and_car() {
    // Same seed = same grid, session for session. Team machinery is
    // seed-independent (it comes from the file); the form delta lives only
    // in the driver envelope (its seeding is unit-tested in `teams`), so the
    // race-level checks are determinism and stable team attributes.
    let run = |seed| {
        let (outcome, _) = apexsim::race::run_race(cfg(FieldMode::Teams, seed), false);
        outcome.state_hash
    };
    assert_eq!(run(21), run(21), "same seed must produce the same teams race");
    for i in [0usize, 7, 15] {
        let a = running(FieldMode::Teams, 21);
        let c = running(FieldMode::Teams, 22);
        assert_eq!(
            a.params_of(i).engine_force_max,
            c.params_of(i).engine_force_max,
            "engine is pure team: seed-independent"
        );
        assert_eq!(
            a.params_of(i).tire_grip,
            c.params_of(i).tire_grip,
            "tyres are pure team: seed-independent"
        );
    }
}

#[test]
fn teams_log_round_trips_and_replays() {
    // A teams race logged to disk must read back as a teams race and replay
    // to the same trajectory hash. The F line carries the field model; the
    // team table and seeded form are pure functions of (seed, id), so the
    // replay needs nothing else.
    let dir = std::env::temp_dir().join(format!("apexsim_teams_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("teams_race.simlog");
    let race_cfg = RaceConfig {
        cars: 6,
        laps: 2,
        seed: 5,
        field: FieldMode::Teams,
        ..RaceConfig::default()
    };
    let (outcome, inputs) = apexsim::race::run_race(race_cfg.clone(), false);
    apexsim::race::write_race_log(&path, &race_cfg, &inputs, &outcome).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\nF teams"), "log must record the field model");

    let (ok, replayed) = apexsim::race::replay_race_log(&path).unwrap();
    assert!(ok, "teams race replay diverged: {:016x} vs {:016x}",
        replayed.state_hash, outcome.state_hash);
    assert_eq!(replayed.state_hash, outcome.state_hash);

    // And a classic race must NOT grow an F line — byte-identical old format.
    let classic_cfg = RaceConfig { cars: 6, laps: 2, seed: 5, ..RaceConfig::default() };
    let (out_c, in_c) = apexsim::race::run_race(classic_cfg.clone(), false);
    let path_c = dir.join("classic_race.simlog");
    apexsim::race::write_race_log(&path_c, &classic_cfg, &in_c, &out_c).unwrap();
    let text_c = std::fs::read_to_string(&path_c).unwrap();
    assert!(!text_c.contains("\nF "), "classic logs must stay in the old format");
}
