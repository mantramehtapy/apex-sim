//! Phase 8 gates — weather, race-director controls, and driver errors.
//!
//! The gate the plan sets:
//!   1. two runs with the same seed and override schedule match;
//!   2. wet-weather tests show measured changes in braking and cornering;
//!   3. overrides can be replayed and do not corrupt session state.
//!
//! Each is pinned by a test below. (1) is hash equality over the whole field,
//! not "the leader was slower" — a weather model that is merely *similar*
//! under repeat runs is not deterministic and cannot be replayed at all.
//! (2) measures the physics, it does not assert a constant: the numbers
//! printed by `weatherbench` come from the same code paths.
//! (3) is a full log → replay round trip through the v4 `W` line, plus a
//! check that a queued-but-never-fired command leaves no trace.

use apexsim::physics::{self, CarParams, CarState, Controls};
use apexsim::race::{RaceConfig, RacePhase};
use apexsim::weather::{
    surface_grip, DriverErrorConfig, Override, Slider, Surface, SurfaceZone, WeatherConfig,
};
use std::path::PathBuf;

fn tmp_log(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("f1_p8_{}_{}.log", name, std::process::id()));
    p
}

fn base_cfg(seed: u64) -> RaceConfig {
    RaceConfig { cars: 8, laps: 4, seed, ..Default::default() }
}

// ---------------------------------------------------------------------------
// Gate 1 — determinism
// ---------------------------------------------------------------------------

#[test]
fn gate1_same_seed_and_override_schedule_reproduce_exactly() {
    // The schedule has to be non-trivial: an override that never fires would
    // make this test pass without testing anything.
    let schedule = || {
        vec![
            Override::new(500, Slider::Water, 0.8),
            Override::new(1500, Slider::TrackTemp, 18.0),
            Override::new(3000, Slider::Grip, 0.9),
        ]
    };
    let mut a = base_cfg(7);
    a.weather = Some(WeatherConfig::preset("damp").unwrap().with_schedule(schedule()));
    let b = a.clone();
    let (ra, _, _) = apexsim::race::run_race_full(a.clone(), false);
    let (rb, _, _) = apexsim::race::run_race_full(b, false);

    // Same config object → same run.
    assert_eq!(ra.state_hash, rb.state_hash, "same seed + schedule must reproduce");
    assert_eq!(ra.ticks, rb.ticks);

    // And the schedule really did fire, in both.
    for o in [&ra, &rb] {
        let w = o.weather.as_ref().expect("weather summary present");
        assert_eq!(w.applied.len(), 3, "all three overrides fired");
        assert_eq!(w.applied[0].tick, 500);
        assert_eq!(w.applied[1].tick, 1500);
        assert_eq!(w.applied[2].tick, 3000);
        assert_eq!(w.unapplied, 0);
    }

    // A different seed must give a different race — otherwise the equality
    // above would be trivially satisfied by a constant hash.
    let mut c = a.clone();
    c.seed = 8;
    let (rc, _, _) = apexsim::race::run_race_full(c, false);
    assert_ne!(ra.state_hash, rc.state_hash, "the seed must still matter");
}

#[test]
fn gate1_weather_is_off_by_default_and_changes_nothing() {
    let dry = base_cfg(3);
    let (a, _, _) = apexsim::race::run_race_full(dry.clone(), false);
    // An explicitly-passed inert config is normalised away, so it must be the
    // same race — not a different-but-equally-fast one.
    let inert = RaceConfig {
        weather: Some(WeatherConfig::default()),
        ..dry.clone()
    };
    assert!(WeatherConfig::default().is_inert());
    let (b, _, _) = apexsim::race::run_race_full(inert, false);
    assert_eq!(a.state_hash, b.state_hash, "a dry config must be a no-op");
    assert!(b.weather.is_none(), "an inert config normalises to no weather");

    // And weather: None is the Phase 7 engine — the 76 pre-existing tests
    // depend on that, and this pins it explicitly.
    let none_cfg = RaceConfig { weather: None, ..dry };
    let (c, _, _) = apexsim::race::run_race_full(none_cfg, false);
    assert_eq!(a.state_hash, c.state_hash);
}

#[test]
fn gate1_a_different_override_schedule_gives_a_different_race() {
    // The mirror of gate1: the schedule must be able to change the outcome,
    // otherwise "reproduces exactly" would be trivially true.
    let mut a = base_cfg(11);
    a.weather = Some(WeatherConfig::preset("wet").unwrap());
    let mut b = a.clone();
    b.weather = Some(WeatherConfig::preset("wet").unwrap().with_schedule(vec![
        Override::new(200, Slider::Grip, 0.7),
    ]));
    let (ra, _, _) = apexsim::race::run_race_full(a, false);
    let (rb, _, _) = apexsim::race::run_race_full(b, false);
    assert_ne!(
        ra.state_hash, rb.state_hash,
        "a grip override must actually change the race"
    );
}

// ---------------------------------------------------------------------------
// Gate 2 — measured wet-weather effects on braking and cornering
// ---------------------------------------------------------------------------

/// Braking distance from a fixed speed at a given grip multiplier, integrated
/// from the real velocity trace. Same physics step the race uses.
fn braking_distance(grip: f64, v0: f64) -> f64 {
    let p = CarParams { tire_grip: CarParams::default().tire_grip * grip, ..Default::default() };
    let mut s = CarState { vel_long: v0, ..Default::default() };
    let ctrl = Controls { brake: 1.0, ..Default::default() };
    let mut dist = 0.0;
    for _ in 0..50_000 {
        let prev = s.vel_long;
        s = physics::step(&s, &ctrl, &p, apexsim::race::DT);
        dist += (prev + s.vel_long) / 2.0 * apexsim::race::DT;
        if s.vel_long < 0.5 {
            break;
        }
    }
    dist
}

/// Peak steady lateral acceleration at a fixed speed, found by sweeping the
/// steering input. Sweeping matters: one steer angle either under- or
/// over-saturates depending on grip, so a single-input comparison is not fair.
fn peak_lateral_accel(grip: f64, v: f64) -> f64 {
    let mut best = 0.0f64;
    for si in 0..40 {
        let steer = 0.05 + si as f64 * 0.02;
        let p = CarParams { tire_grip: CarParams::default().tire_grip * grip, ..Default::default() };
        let mut s = CarState { vel_long: v, ..Default::default() };
        let mut held = true;
        for _ in 0..3000 {
            let err = v - s.vel_long;
            let ctrl = Controls {
                steer_cmd: steer,
                throttle: (err * 0.5).clamp(0.0, 1.0),
                brake: (-err * 0.5).clamp(0.0, 1.0),
            };
            s = physics::step(&s, &ctrl, &p, apexsim::race::DT);
            if s.vel_long < v * 0.4 {
                held = false; // slid off or spun
                break;
            }
        }
        if held {
            best = best.max(s.yaw_rate * s.vel_long);
        }
    }
    best
}

#[test]
fn gate2_wet_measurably_lengthens_braking() {
    let v0 = 200.0 / 3.6;
    let dry = braking_distance(surface_grip(Surface::Dry, 0.0, 35.0, 1.0, v0), v0);
    let damp = braking_distance(surface_grip(Surface::Damp, 0.35, 35.0, 1.0, v0), v0);
    let wet = braking_distance(surface_grip(Surface::Wet, 1.0, 35.0, 1.0, v0), v0);

    assert!(dry < damp, "dry {dry:.1} m must brake shorter than damp {damp:.1} m");
    assert!(damp < wet, "damp {damp:.1} m must brake shorter than wet {wet:.1} m");
    // A real margin, not a rounding artefact: ~24% longer on wet is roughly
    // what a 40% grip loss buys you.
    assert!(
        wet > dry * 1.15,
        "wet braking {wet:.1} m is not enough longer than dry {dry:.1} m"
    );
    println!(
        "braking from 200 km/h: dry {dry:.1} m | damp {damp:.1} m | wet {wet:.1} m \
         (+{:.0}% wet)",
        (wet / dry - 1.0) * 100.0
    );
}

#[test]
fn gate2_wet_measurably_lowers_the_cornering_limit() {
    let v = 25.0;
    let dry = peak_lateral_accel(surface_grip(Surface::Dry, 0.0, 35.0, 1.0, v), v);
    let damp = peak_lateral_accel(surface_grip(Surface::Damp, 0.35, 35.0, 1.0, v), v);
    let wet = peak_lateral_accel(surface_grip(Surface::Wet, 1.0, 35.0, 1.0, v), v);

    assert!(dry > 0.0, "the dry car must corner at all");
    assert!(dry > damp, "dry {dry:.2} must beat damp {damp:.2} m/s²");
    assert!(damp > wet, "damp {damp:.2} must beat wet {wet:.2} m/s²");
    assert!(
        wet < dry * 0.9,
        "wet cornering {wet:.2} m/s² is not clearly below dry {dry:.2} m/s²"
    );
    println!(
        "peak lateral at 90 km/h: dry {:.2} g | damp {:.2} g | wet {:.2} g",
        dry / 9.81,
        damp / 9.81,
        wet / 9.81
    );
}

#[test]
fn gate2_wet_conditions_cost_lap_time_in_a_real_race() {
    // The isolated physics tests prove grip changed; this proves the driver
    // and the race loop actually respond to it.
    let mut dry = base_cfg(5);
    dry.laps = 6;
    let mut wet = dry.clone();
    wet.weather = Some(WeatherConfig::preset("wet").unwrap());
    let (d, _, _) = apexsim::race::run_race_full(dry, false);
    let (w, _, _) = apexsim::race::run_race_full(wet, false);

    let d_best = d.leader_lap_times.iter().cloned().fold(0.0f64, f64::max);
    let w_best = w.leader_lap_times.iter().cloned().fold(0.0f64, f64::max);
    assert!(w_best > d_best, "wet best lap {w_best:.2} s must be slower than dry {d_best:.2} s");

    let d_sum: f64 = d.leader_lap_times.iter().sum();
    let w_sum: f64 = w.leader_lap_times.iter().sum();
    assert!(w_sum > d_sum, "wet race must take longer overall");

    let grip = w.weather.as_ref().unwrap().mean_grip_car0;
    assert!(grip > 0.5 && grip < 0.8, "mean car 0 grip {grip:.3} is not wet-like");
    println!(
        "6-lap race: leader best {d_best:.2} s dry -> {w_best:.2} s wet \
         (+{:.1}%)| total {d_sum:.1} -> {w_sum:.1} s (+{:.0} s)",
        (w_best / d_best - 1.0) * 100.0,
        w_sum - d_sum
    );
}

#[test]
fn gate2_hydroplaning_only_applies_above_its_threshold() {
    // The plan is explicit that fixed hydroplaning thresholds are an
    // assumption until tested, so the assumption needs a test.
    for surf in [Surface::Dry, Surface::Damp, Surface::Wet] {
        let slow = surface_grip(surf, 1.0, 35.0, 1.0, 20.0);
        let fast = surface_grip(surf, 1.0, 35.0, 1.0, 80.0);
        if surf == Surface::Dry {
            assert!((slow - fast).abs() < 1e-12, "a dry line does not hydroplane");
        } else {
            assert!(fast < slow, "{} must lose grip at speed", surf.label());
        }
    }
}

// ---------------------------------------------------------------------------
// Gate 3 — replay and session-state integrity
// ---------------------------------------------------------------------------

#[test]
fn gate3_overrides_replay_bit_identically() {
    let path = tmp_log("replay");
    let mut cfg = base_cfg(13);
    cfg.laps = 5;
    cfg.weather = Some(WeatherConfig::preset("damp").unwrap().with_schedule(vec![
        Override::new(400, Slider::Water, 0.75),
        Override::new(2000, Slider::TrackTemp, 20.0),
        Override::new(6000, Slider::Grip, 0.88),
    ]));
    let run_cfg = cfg.clone();
    let (live, inputs, _) = apexsim::race::run_race_full(cfg, false);
    apexsim::race::write_race_log(&path, &run_cfg, &inputs, &live).expect("write log");

    let (ok, replayed) = apexsim::race::replay_race_log(&path).expect("replay");
    assert!(ok, "replay diverged: {} vs {}", live.state_hash, replayed.state_hash);

    // The replay rebuilt the weather from the W line, not from memory.
    let lw = live.weather.as_ref().unwrap();
    let rw = replayed.weather.as_ref().expect("replay kept the weather");
    assert_eq!(lw.applied.len(), rw.applied.len());
    for (a, b) in lw.applied.iter().zip(rw.applied.iter()) {
        assert_eq!(a.tick, b.tick, "override tick must survive the round trip");
        assert_eq!(a.slider, b.slider);
    }
    assert!((lw.mean_grip_car0 - rw.mean_grip_car0).abs() < 1e-9);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gate3_a_dry_log_still_replays_and_carries_no_weather_line() {
    // Absence of the W line is what identifies a v3-shaped run; the reader
    // must not invent weather for it.
    let path = tmp_log("dry");
    let cfg = base_cfg(17);
    let (live, inputs, _) = apexsim::race::run_race_full(cfg.clone(), false);
    apexsim::race::write_race_log(&path, &cfg, &inputs, &live).expect("write log");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.lines().any(|l| l.starts_with("W ")), "a dry log must not carry a W line");

    let (ok, replayed) = apexsim::race::replay_race_log(&path).expect("replay");
    assert!(ok, "a dry log must still replay");
    assert!(replayed.weather.is_none(), "no W line means no weather");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gate3_an_override_past_the_flag_leaves_no_trace() {
    // A command queued for a tick the race never reaches must not fire, must
    // not appear in the results, and must not change the outcome — that is
    // what "does not corrupt session state" means for an unfired command.
    let queued = RaceConfig {
        weather: Some(WeatherConfig::preset("damp").unwrap().with_schedule(vec![
            Override::new(50_000_000, Slider::Water, 0.99),
        ])),
        ..base_cfg(19)
    };
    let none = RaceConfig { weather: Some(WeatherConfig::preset("damp").unwrap()), ..base_cfg(19) };
    let (a, _, _) = apexsim::race::run_race_full(queued, false);
    let (b, _, _) = apexsim::race::run_race_full(none, false);
    assert_eq!(a.state_hash, b.state_hash, "an unfired override changed the race");

    let w = a.weather.as_ref().unwrap();
    assert!(w.applied.is_empty(), "nothing should have fired");
    assert_eq!(w.unapplied, 1, "but it is still on the books as queued");
    assert!(a.phase_finished(), "the race still completed");
}

#[test]
fn gate3_overrides_land_on_their_queued_tick_and_not_earlier() {
    // The plan's architectural rule: never inject UI values into
    // unsynchronized physics state — queue them for a known tick. So the
    // grip car 0 sees must be unchanged right up to the tick before.
    let tick = 2000u64;
    let mut race = apexsim::race::Race::new(RaceConfig {
        cars: 1,
        laps: 5,
        weather: Some(WeatherConfig {
            water: 0.0,
            grip_scale: 1.0,
            zones: vec![SurfaceZone { start: 0.0, end: 1.0, surface: Surface::Dry }],
            schedule: vec![Override::new(tick, Slider::Grip, 0.6)],
            ..WeatherConfig::default()
        }),
        ..base_cfg(23)
    });
    race.start();
    while race.state.phase == RacePhase::Ready || race.state.phase == RacePhase::Formation {
        race.tick();
    }
    // Every tick below the queued one must show the old value — an override
    // that arrives early is the bug this whole design exists to prevent.
    while race.state.tick < tick - 1 {
        race.tick();
        let w = race.weather_summary().expect("weather on");
        assert!(
            (w.grip_scale - 1.0).abs() < 1e-12,
            "slider moved early at tick {}: grip {}",
            race.state.tick,
            w.grip_scale
        );
        assert!(w.applied.is_empty(), "override fired before its tick");
    }
    // One more tick and the counter reads the queued tick — which is the moment
    // the value is in effect, so an observer reading the state at tick N sees
    // the grip the physics is using at tick N.
    race.tick();
    assert_eq!(race.state.tick, tick);
    let w = race.weather_summary().expect("weather on");
    assert_eq!(w.applied.len(), 1, "fired exactly on its tick");
    assert_eq!(w.applied[0].tick, tick);
    assert!((w.grip_scale - 0.6).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Driver errors — seeded, pure, and separate from vehicle physics
// ---------------------------------------------------------------------------

#[test]
fn gate4_driver_errors_are_off_unless_asked_for() {
    let off = DriverErrorConfig::default();
    assert!(off.is_off());
    assert_eq!(off.errors_on_lap(3, 5), 0);

    let mut cfg = base_cfg(29);
    cfg.weather = Some(WeatherConfig::preset("wet").unwrap());
    let (o, _, _) = apexsim::race::run_race_full(cfg, false);
    let w = o.weather.as_ref().unwrap();
    assert_eq!(w.driver_errors, 0, "wet weather alone must not make mistakes");
}

#[test]
fn gate4_seeded_errors_reproduce_exactly() {
    let mut a = base_cfg(31);
    a.laps = 6;
    a.weather = Some(
        WeatherConfig::preset("damp").unwrap().with_driver_error(DriverErrorConfig {
            rate_per_lap: 1.5,
            ..Default::default()
        }),
    );
    let b = a.clone();
    let (ra, _, _) = apexsim::race::run_race_full(a, false);
    let (rb, _, _) = apexsim::race::run_race_full(b, false);
    assert_eq!(ra.state_hash, rb.state_hash, "seeded errors must be reproducible");

    let wa = ra.weather.as_ref().unwrap();
    let wb = rb.weather.as_ref().unwrap();
    assert_eq!(wa.driver_errors, wb.driver_errors);
    assert_eq!(wa.driver_errors_by_car, wb.driver_errors_by_car);
    assert!(wa.driver_errors > 0, "a rate of 1.5/lap must produce mistakes");
    assert!(
        (wa.driver_error_rate - 1.5).abs() < 1e-12,
        "the reported rate must be the configured one"
    );
}

#[test]
fn gate4_errors_are_a_control_layer_perturbation_only() {
    // The plan requires driver errors stay separate from vehicle physics so a
    // scenario remains explainable. Concretely: the error must change the
    // controls and the resulting trajectory, and nothing else — the same run
    // with errors disabled must have a different hash, and turning errors on
    // must not alter the weather the tyres see.
    let mut with_err = base_cfg(37);
    with_err.laps = 5;
    with_err.weather = Some(WeatherConfig::preset("damp").unwrap());
    let without_err = with_err.clone();
    let (a, _, _) = apexsim::race::run_race_full(with_err.clone(), false);
    let (b, _, _) = apexsim::race::run_race_full(without_err, false);
    assert_eq!(
        a.state_hash, b.state_hash,
        "errors off must leave the race bit-for-bit alone"
    );

    with_err.weather = Some(
        WeatherConfig::preset("damp")
            .unwrap()
            .with_driver_error(DriverErrorConfig { rate_per_lap: 3.0, ..Default::default() }),
    );
    let (c, _, _) = apexsim::race::run_race_full(with_err, false);
    assert_ne!(
        a.state_hash, c.state_hash,
        "errors on must actually change the race"
    );
    // Same conditions underneath: the surface layout and slider values are a
    // property of the track and the director, not of the driver. (Mean grip is
    // deliberately NOT compared — it is sampled along the trajectory, and a
    // mistake moves the car through those samples differently.)
    let (wa, wc) = (a.weather.as_ref().unwrap(), c.weather.as_ref().unwrap());
    assert_eq!(wa.label, wc.label);
    assert_eq!(wa.water, wc.water);
    assert_eq!(wa.grip_scale, wc.grip_scale);
    assert_eq!(wa.track_temp_c, wc.track_temp_c);
    assert_eq!(wa.unapplied, wc.unapplied);
}

#[test]
fn gate4_a_mistake_is_a_bounded_control_offset_not_a_physics_change() {
    let cfg = DriverErrorConfig { rate_per_lap: 4.0, ..DriverErrorConfig::default() };
    // Find a (lap, phase) where a mistake is active and check the offset is
    // bounded by the configured magnitudes — a mistake cannot demand more
    // than the config allows.
    let mut found = 0;
    for lap in 0..20u32 {
        for step in 0..200 {
            let phase = step as f64 / 200.0;
            let mut steer = 0.0;
            let mut throttle = 0.0;
            for k in 0..DriverErrorConfig::MAX_PER_LAP {
                let (ds, dt) = cfg.offset(4, lap, k, phase, 0.1);
                steer += ds;
                throttle += dt;
            }
            assert!(
                steer.abs() <= cfg.steer_mag * DriverErrorConfig::MAX_PER_LAP as f64 + 1e-12,
                "steer offset {steer} exceeds the budget at lap {lap} phase {phase}"
            );
            assert!(
                throttle.abs() <= cfg.throttle_mag * DriverErrorConfig::MAX_PER_LAP as f64 + 1e-12,
                "throttle offset {throttle} exceeds the budget"
            );
            if steer != 0.0 || throttle != 0.0 {
                found += 1;
            }
        }
    }
    assert!(found > 0, "a rate of 4.0/lap produced no active mistake anywhere");
}

// ---------------------------------------------------------------------------
// Timing — the Phase 0 budget must survive the new per-tick work
// ---------------------------------------------------------------------------

#[test]
fn gate5_wet_weather_stays_inside_the_physics_budget() {
    // The per-tick cost is now surface_grip() per car per tick. That must not
    // push p99 past the 4 ms the engine rate allows.
    let mut cfg = base_cfg(41);
    cfg.laps = 8;
    cfg.weather = Some(
        WeatherConfig::preset("wet").unwrap()
            .with_driver_error(DriverErrorConfig { rate_per_lap: 2.0, ..Default::default() }),
    );
    let (o, _, _) = apexsim::race::run_race_full(cfg, false);
    let s = &o.tick_stats;
    assert!(
        s.p99_us < 4000.0,
        "p99 tick cost {:.1} µs exceeds the 4000 µs budget",
        s.p99_us
    );
    assert!(o.phase_finished(), "the wet race must still run to completion");
    println!(
        "wet + driver errors, {} cars {} laps: avg {:.1} µs | p50 {:.1} | p99 {:.1} | worst {:.1}",
        o.cars, o.laps, s.avg_us, s.p50_us, s.p99_us, s.worst_us
    );
}
// ---------------------------------------------------------------------------
// The live race-director panel
//
// Everything above is reachable from the command line, which is enough for the
// plan's completion gate but not for its architecture rule: "do not inject UI
// values directly into unsynchronized physics state — queue commands for a
// known simulation tick." A rule about the UI is only met if there is a UI.
// These gates drive the same code the dashboard's sliders do.
// ---------------------------------------------------------------------------

#[test]
fn gate6_a_live_panel_left_alone_changes_nothing() {
    // The dashboard arms the panel in every served session, so a neutral panel
    // has to be free. Not "close to" the no-weather engine — the same hash.
    let base = base_cfg(51);
    let (plain, _, _) = apexsim::race::run_race_full(base.clone(), false);
    let mut live = base;
    live.weather = Some(WeatherConfig::director());
    let (panel, _, _) = apexsim::race::run_race_full(live, false);
    assert_eq!(
        plain.state_hash, panel.state_hash,
        "an untouched director panel must not move the simulation"
    );

    // Neutral means exactly 1.0 at every speed, so no slider position is quietly
    // pre-loading a penalty.
    for v in [0.0, 15.0, 45.0, 90.0] {
        assert_eq!(surface_grip(Surface::Dry, 0.0, apexsim::weather::Assumptions::IDEAL_TEMP_C, 1.0, v), 1.0);
    }
    let d = WeatherConfig::director();
    assert!(d.live, "the director panel must stay live at neutral");
    assert_eq!(d.label(), "dry");

    // ...and the scenario name has to follow the sliders, or a director who
    // has just put water on the track is still told the session is dry.
    let mut cfg = base_cfg(55);
    cfg.laps = 4;
    cfg.weather = Some(WeatherConfig::director());
    let mut race = apexsim::race::Race::new(cfg);
    race.start();
    for _ in 0..1000 {
        race.tick();
    }
    assert_eq!(race.weather_state().unwrap().label, "dry");
    apexsim::server::apply_slider_command(&mut race, "slider:water:0.6");
    race.tick();
    let st = race.weather_state().unwrap();
    assert_eq!(st.water, 0.6);
    assert_eq!(st.label, "damp", "the label must follow the live water level");
}

#[test]
fn gate6_a_slider_command_is_queued_for_a_known_tick_and_never_injected() {
    let mut cfg = base_cfg(52);
    cfg.laps = 6;
    cfg.weather = Some(WeatherConfig::director());
    let mut race = apexsim::race::Race::new(cfg);
    race.start();
    for _ in 0..3000 {
        race.tick();
    }
    let at = race.state.tick;
    assert_eq!(race.weather_state().unwrap().water, 0.0);

    // Exactly what the dashboard sends when the water slider is released.
    let out = apexsim::server::apply_slider_command(&mut race, "slider:water:0.90");
    match out {
        apexsim::server::SliderOutcome::Queued { slider, value, tick } => {
            assert_eq!(slider, Slider::Water);
            assert_eq!(value, 0.90);
            assert_eq!(tick, at + apexsim::server::SLIDER_LEAD_TICKS, "queued for a named tick");
        }
        other => panic!("expected the move to queue, got {other:?}"),
    }

    // The physics has not moved: the value is on a queue, not in the state.
    let now = race.weather_state().unwrap();
    assert_eq!(now.water, 0.0, "a queued slider must not take effect early");
    assert_eq!(now.applied, 0);
    assert_eq!(now.queued, 1);

    // ... and it lands on exactly that tick, and is announced with that tick.
    race.tick();
    let after = race.weather_state().unwrap();
    assert_eq!(after.water, 0.90);
    assert_eq!(after.applied, 1);
    assert_eq!(after.queued, 0);
    let applied = race
        .state
        .rc_events
        .iter()
        .find_map(|e| match e {
            apexsim::race::RcEvent::WeatherApplied { tick, slider, value } => Some((*tick, *slider, *value)),
            _ => None,
        })
        .expect("the applied move must be announced");
    assert_eq!(applied.0, at + apexsim::server::SLIDER_LEAD_TICKS);
    assert_eq!(applied.1, "water");
    assert_eq!(applied.2, 0.90);
}

#[test]
fn gate6_a_slider_moved_mid_race_replays_bit_identically() {
    // The reason a live move is queued rather than applied: it has to be as
    // replayable as one set before the start.
    let path = tmp_log("live");
    let mut cfg = base_cfg(53);
    cfg.laps = 6;
    cfg.weather = Some(WeatherConfig::director());
    let mut race = apexsim::race::Race::new(cfg.clone());
    race.start();
    for _ in 0..2500 {
        race.tick();
    }
    apexsim::server::apply_slider_command(&mut race, "slider:grip:0.85");
    apexsim::server::apply_slider_command(&mut race, "slider:track_temp:18");
    let queued_at = race.state.tick;
    while race.state.phase != RacePhase::Finished {
        race.tick();
    }
    let inputs = race.take_recorded();
    let live = race.outcome(0.0);

    // The log is written from the config as it now stands — the caller's copy
    // predates both moves.
    let mut logged = cfg.clone();
    logged.weather = race.weather_config().cloned();
    apexsim::race::write_race_log(&path, &logged, &inputs, &live).expect("write log");

    let (ok, replayed) = apexsim::race::replay_race_log(&path).expect("replay");
    assert!(ok, "a UI-moved session diverged on replay: {} vs {}", live.state_hash, replayed.state_hash);

    let lw = live.weather.as_ref().expect("the live run had weather");
    assert_eq!(lw.applied.len(), 2, "both moves took effect");
    assert_eq!(lw.applied[0].tick, queued_at + apexsim::server::SLIDER_LEAD_TICKS);
    assert_eq!(lw.applied[0].slider, Slider::Grip);
    assert_eq!(lw.applied[1].slider, Slider::TrackTemp);

    let rw = replayed.weather.as_ref().expect("the replay kept the weather");
    for (a, b) in lw.applied.iter().zip(rw.applied.iter()) {
        assert_eq!((a.tick, a.slider, a.value), (b.tick, b.slider, b.value));
    }
    assert!((lw.mean_grip_car0 - rw.mean_grip_car0).abs() < 1e-9);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gate6_the_wire_protocol_accepts_a_slider_and_says_why_it_will_not() {
    use apexsim::protocol::{validate_command, CommandError};
    for cmd in ["start", "pause", "resume", "reset", "slider:water:0.5", "slider:track_temp:-4.5"] {
        assert!(validate_command(cmd).is_ok(), "{cmd} should be accepted");
    }
    match validate_command("turbo") {
        Err(CommandError::Unknown(d)) => assert!(d.contains("turbo"), "{d}"),
        other => panic!("an unknown verb must be refused as unknown, got {other:?}"),
    }
    // Out of range is refused with the range, not clamped into silence.
    match validate_command("slider:water:3") {
        Err(CommandError::Invalid(d)) => {
            assert!(d.contains("water") && d.contains("0"), "the reason should name the slider and its range: {d}")
        }
        other => panic!("an out-of-range slider must be refused, got {other:?}"),
    }
    match validate_command("slider:mud:0.5") {
        Err(CommandError::Invalid(d)) => assert!(d.contains("mud"), "{d}"),
        other => panic!("an unknown slider must be refused, got {other:?}"),
    }
    match validate_command("slider:water") {
        Err(CommandError::Invalid(_)) => {}
        other => panic!("a valueless slider must be refused, got {other:?}"),
    }
    // The two refusals are distinguishable on the wire.
    assert_eq!(validate_command("turbo").unwrap_err().status(), "rejected_unknown_command");
    assert_eq!(validate_command("slider:water:9").unwrap_err().status(), "rejected_bad_command");
}

#[test]
fn gate6_a_slider_without_a_live_panel_is_refused_rather_than_dropped() {
    // The single-car / no-weather engine has no sliders; the director is told
    // so instead of watching the handle move and nothing happen.
    let mut cfg = base_cfg(54);
    cfg.weather = None;
    let mut race = apexsim::race::Race::new(cfg);
    race.start();
    let before = race.state.tick;
    match apexsim::server::apply_slider_command(&mut race, "slider:water:0.8") {
        apexsim::server::SliderOutcome::Rejected(reason) => {
            assert!(reason.contains("no weather"), "{reason}")
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(race.state.tick, before, "a refused command must not tick the simulation");
    assert!(race.weather_state().is_none());
}
