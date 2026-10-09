//! W4 sweep: full-race measurements for the realism pack's rates.

use apexsim::race::{FieldMode, Race, RaceConfig, RacePhase, RealismConfig};

#[test]
fn full_race_sweep_reports_fuel_tyres_and_dnf() {
    for seed in [42u64, 7, 99] {
        let cfg = RaceConfig {
            cars: 20,
            laps: 12,
            seed,
            field: FieldMode::Teams,
            realism: Some(RealismConfig::default()),
            ..RaceConfig::default()
        };
        let mut race = Race::new(cfg);
        race.start();
        let cap = ((13.0 * 60.0 + 120.0) / 0.004) as u64;
        let mut n = 0u64;
        let start_fuels: Vec<f64> =
            race.state.cars.iter().map(|e| e.fuel_mass).collect();
        let mut max_temp = 0.0f64;
        while race.state.phase != RacePhase::Finished && n < cap {
            race.tick();
            n += 1;
            max_temp = max_temp.max(race.state.cars[0].tyre_temp);
        }
        let retired = race.state.cars.iter().filter(|e| e.retired).count();
        // A SURVIVING car must have burned a believable chunk of its tank;
        // early retirees freeze wherever they ran out of race.
        let burn: f64 = race
            .state
            .cars
            .iter()
            .zip(start_fuels.iter())
            .filter(|(e, _)| !e.retired)
            .map(|(e, s0)| s0 - e.fuel_mass)
            .fold(0.0, f64::max);
        println!(
            "seed {seed}: retired={retired} max burn={burn:.1} kg, max temp {max_temp:.2}"
        );
        assert!(burn > 10.0, "seed {seed}: best survivor burned only {burn:.1} kg");
        assert!(max_temp > 0.9, "seed {seed}: tyres never fully warmed");
    }
}

