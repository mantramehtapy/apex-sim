//! Realism-pack gates (R1): crashes, mechanicals, damage and lockups.
//!
//! The pack's contract is "on = cars can go out for driver and mechanical
//! reasons; off = the pre-pack engine bit-for-bit". Every gate here guards
//! one half of that sentence.

use apexsim::race::{
    run_race, FieldMode, Race, RaceConfig, RacePhase, RealismConfig, SessionMode,
};
use apexsim::weather::{DriverErrorConfig, WeatherConfig};

fn base(seed: u64, realism: Option<RealismConfig>) -> RaceConfig {
    RaceConfig {
        cars: 10,
        laps: 4,
        seed,
        field: FieldMode::Teams,
        realism,
        ..RaceConfig::default()
    }
}

fn pack() -> Option<RealismConfig> {
    Some(RealismConfig::default())
}

/// Force frequent mistakes so spin/mechanical paths are exercised on demand:
/// a weather error block at 2.5 mistakes/lap (the realism pack adopts the
/// weather block when it carries one).
fn error_heavy(mut cfg: RaceConfig) -> RaceConfig {
    let w = WeatherConfig {
        driver_errors: DriverErrorConfig {
            rate_per_lap: 2.5,
            ..DriverErrorConfig::default()
        },
        ..WeatherConfig::default()
    };
    cfg.weather = Some(w);
    cfg
}

#[test]
fn pack_off_is_the_old_engine_exactly() {
    // An inert pack config normalises to None in Race::new, so a caller who
    // passes Some(all-off) gets the pre-pack trajectory, and the pack-on race
    // is genuinely a different race.
    let inert = RealismConfig {
        spin_crashes: false,
        error_rate: 0.0,
        mechanical_rate: 0.0,
        contact_damage: false,
        lockups: false,
        safety_car: false,
        yellow_zones: false,
        blue_flags: false,
        formation_lap: false,
        drs_zones: false,
        fuel: false,
        tyre_temp: false,
    };
    assert!(inert.is_inert());

    let cfg_off = base(31, None);
    let (a, _) = run_race(cfg_off.clone(), false);
    let cfg_inert = base(31, Some(inert));
    let (b, _) = run_race(cfg_inert, false);
    assert_eq!(a.state_hash, b.state_hash, "an inert pack must be no pack");

    let (c, _) = run_race(base(31, pack()), false);
    assert_ne!(a.state_hash, c.state_hash, "the pack must change the racing");
}

#[test]
fn spins_put_cars_out_when_mistakes_are_frequent() {
    // With 2.5 mistakes/lap across a 10-car, 4-lap race and the spin kick
    // live, at least one driver must go out for driver reasons. Swept over
    // six seeds so the claim does not lean on one lucky seed.
    let mut races_with_spin_dnf = 0;
    for seed in 1..=6u64 {
        let cfg = error_heavy(base(seed, pack()));
        let (outcome, _) = run_race(cfg, false);
        let spun = outcome
            .dnf_causes
            .iter()
            .filter(|(_, c)| *c == "spun" || *c == "beached")
            .count();
        if spun > 0 {
            races_with_spin_dnf += 1;
        }
    }
    assert!(
        races_with_spin_dnf >= 3,
        "driver-error spins fired in only {races_with_spin_dnf}/6 error-heavy races"
    );
}

#[test]
fn without_the_pack_errors_never_retire_anyone() {
    // The same error-heavy races with the pack OFF: mistakes still wobble the
    // cars (that is Phase 8), but nothing loses control and nobody is out.
    for seed in 1..=6u64 {
        let cfg = error_heavy(base(seed, None));
        let (outcome, _) = run_race(cfg, false);
        assert!(
            outcome.dnf_causes.is_empty(),
            "pack-off race (seed {seed}) had retirements with causes: {:?}",
            outcome.dnf_causes
        );
    }
}

#[test]
fn mechanical_failures_are_seeded_and_deterministic() {
    // rate 3.0 per car per race = essentially everyone fails, deterministically.
    // Spins stay off so the race lives long enough for the per-lap checks to
    // arm (a spun car retires at lap 0, before the first crossing).
    let mut cfg = base(9, pack());
    cfg.realism.as_mut().unwrap().spin_crashes = false;
    cfg.realism.as_mut().unwrap().error_rate = 0.0;
    cfg.realism.as_mut().unwrap().mechanical_rate = 3.0;
    let (a, _) = run_race(cfg.clone(), false);
    let (b, _) = run_race(cfg, false);
    assert_eq!(a.dnf_causes, b.dnf_causes, "failures must be seed-deterministic");
    let mech = a
        .dnf_causes
        .iter()
        .filter(|(_, c)| *c == "mechanical")
        .count();
    assert!(mech >= 5, "rate 3.0/field of 10 produced only {mech} failures — causes: {:?}", a.dnf_causes);

    // And at rate 0 there are none.
    let mut cfg0 = base(9, pack());
    cfg0.realism.as_mut().unwrap().spin_crashes = false;
    cfg0.realism.as_mut().unwrap().error_rate = 0.0;
    cfg0.realism.as_mut().unwrap().mechanical_rate = 0.0;
    let (c, _) = run_race(cfg0, false);
    assert!(
        !c.dnf_causes.iter().any(|(_, x)| *x == "mechanical"),
        "rate 0 must never fail an engine"
    );
}

#[test]
fn damage_and_flatspots_keep_the_driver_lockstep() {
    // After a damage-heavy race, every surviving car's physics grip must
    // still equal what its driver aims at — the invariant a dented car's
    // driver would violate first.
    let cfg = error_heavy(base(17, pack()));
    let mut race = Race::new(cfg);
    race.start();
    let cap = ((4.0 * 60.0 + 120.0 + 30.0) / 0.004) as u64;
    let mut n = 0u64;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
    }
    let base_grip = apexsim::physics::CarParams::default().tire_grip;
    for i in 0..race.state.cars.len() {
        let p = race.params_of(i);
        let physics = p.tire_grip / base_grip;
        let driver = race.grip_scale_of(i);
        assert!(
            ((physics - driver) / driver).abs() < 1e-9,
            "car {i}: physics tyre factor {physics} != driver scale {driver}"
        );
    }
    // At least one car actually carried damage or a flat spot, so the check
    // above exercised the new channels rather than trivially passing.
    let hurt = race
        .state
        .cars
        .iter()
        .filter(|e| e.damage > 0.0 || e.flatspot > 0.0)
        .count();
    assert!(hurt > 0, "a contact-heavy race damaged nobody — channels never fired");
}

#[test]
fn a_realism_race_logs_and_replays_bit_exactly() {
    let dir = std::env::temp_dir().join(format!("apexsim_realism_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("realism.simlog");
    let cfg = error_heavy(base(23, pack()));
    let (outcome, inputs) = run_race(cfg.clone(), false);
    apexsim::race::write_race_log(&path, &cfg, &inputs, &outcome).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\nJ "), "the pack posture must ride the log");
    let (ok, replayed) = apexsim::race::replay_race_log(&path).unwrap();
    assert!(
        ok,
        "realism race replay diverged: {:016x} vs {:016x}",
        replayed.state_hash, outcome.state_hash
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn timed_sessions_never_lose_cars_to_the_pack() {
    // Mechanicals and spins are RACE decisions: a qualifying or practice
    // session measures pace and must not hand out DNFs.
    let mut cfg = error_heavy(base(33, pack()));
    cfg.mode = SessionMode::Qualifying;
    cfg.realism.as_mut().unwrap().mechanical_rate = 3.0;
    let mut race = Race::new(cfg);
    race.start();
    for _ in 0..20000 {
        if race.state.phase == RacePhase::Finished {
            break;
        }
        race.tick();
    }
    assert!(
        !race.state.cars.iter().any(|e| e.retired),
        "a timed session retired cars to the realism pack"
    );
}
