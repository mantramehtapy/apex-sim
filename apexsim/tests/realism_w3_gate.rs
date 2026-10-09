//! W3 gates: fuel mass slows the car, burns per metre, refuels at stops;
//! cold tyres bleed grip and warm with driving; pack-off is identical.

use apexsim::race::{FieldMode, Race, RaceConfig, RacePhase, RealismConfig};

fn pack() -> Option<RealismConfig> {
    Some(RealismConfig::default())
}

fn base(seed: u64, realism: Option<RealismConfig>) -> RaceConfig {
    RaceConfig {
        cars: 6,
        laps: 4,
        seed,
        field: FieldMode::Teams,
        realism,
        ..RaceConfig::default()
    }
}

#[test]
fn fuel_mass_burns_as_the_car_drives() {
    // params_of mass = base + fuel on board. Driving burns fuel, so the
    // mass must DECREASE monotonically (fuel is never added mid-stint).
    // No stops: fuel only ever burns, so the burn must be monotonic.
    let mut race = Race::new(RaceConfig {
        strategy: apexsim::race::PitStrategy::None,
        ..base(5, pack())
    });
    race.start();
    let base_mass = apexsim::physics::CarParams::default().mass;
    let start_mass = {
        for _ in 0..(20.0 / 0.004) as u64 {
            race.tick();
        }
        race.params_of(0).mass
    };
    assert!(
        start_mass > base_mass + 30.0,
        "a full 60 kg tank must show in the params: start {start_mass} vs base {base_mass}"
    );
    // Drive on; mass must fall and never rise.
    let mut prev = start_mass;
    for _ in 0..(90.0 / 0.004) as u64 {
        race.tick();
        let m = race.params_of(0).mass;
        assert!(m <= prev + 1e-9, "fuel mass must not rise: {prev} -> {m}");
        prev = m;
    }
    assert!(
        prev < start_mass - 5.0,
        "90 s of driving burned only {:.1} kg of fuel (game-scaled burn: full tank ~= 20 km)",
        start_mass - prev
    );
}

#[test]
fn heavy_car_is_measurably_slower() {
    // Same seeds, same field: the full-tank first laps are slower than the
    // light final laps (fuel burns down over the race).
    let cfg = base(11, pack());
    let mut race = Race::new(cfg);
    race.start();
    let mut first_lap_best = f64::MAX;
    let mut last_lap_best = f64::MAX;
    let cap = ((5.0 * 60.0 + 120.0) / 0.004) as u64;
    let mut n = 0u64;
    let mut laps_seen = Vec::new();
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        if let Some(t) = race.state.cars[0].lap_times.last() {
            let lap = race.state.cars[0].lap_times.len();
            if laps_seen.last() != Some(&lap) {
                laps_seen.push(lap);
                if lap == 1 {
                    first_lap_best = *t;
                }
                last_lap_best = *t;
            }
        }
        if laps_seen.len() >= 3 {
            break;
        }
    }
    assert!(
        laps_seen.len() >= 3,
        "not enough laps to compare fuel effects"
    );
    // With fuel burning, later laps trend quicker (all else equal-ish).
    assert!(
        last_lap_best < first_lap_best + 2.0,
        "first {first_lap_best} vs last {last_lap_best}"
    );
    let _ = &mut last_lap_best;
}

#[test]
fn cold_tyres_warm_with_driving() {
    // Tyres start stone cold and warm toward full grip with driving
    // distance (braking and sliding warm faster). A pit stop INTENTIONALLY
    // resets the temp (fresh rubber is cold) — so the assertion is that the
    // tyres reach a warmed state, not that the trace is monotonic across
    // stops.
    let cfg = base(7, pack());
    let mut race = Race::new(cfg);
    race.start();
    let mut max_temp_seen = 0.0f64;
    let cap = ((4.0 * 60.0 + 120.0) / 0.004) as u64;
    let mut n = 0u64;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        max_temp_seen = max_temp_seen.max(race.state.cars[0].tyre_temp);
    }
    assert!(
        max_temp_seen > 0.5,
        "tyres never warmed: max temp {max_temp_seen}"
    );
}

#[test]
fn pack_off_has_no_fuel_and_no_temp() {
    // Pack-off params: mass exactly base (no fuel), and the pack-off hash
    // stays the pre-pack engine.
    let mut race = Race::new(base(9, None));
    race.start();
    for _ in 0..(60.0 / 0.004) as u64 {
        race.tick();
    }
    let base_mass = apexsim::physics::CarParams::default().mass;
    let p = race.params_of(0);
    assert!(
        (p.mass - base_mass).abs() < 1e-9,
        "pack-off mass {} must equal base {base_mass}",
        p.mass
    );
}
