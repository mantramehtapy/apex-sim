//! Phase 1 completion-gate tests, mapped to the plan's gate criteria:
//!
//! 1. "Repeat the same seeded lap with the same inputs and replay it."
//! 2. "The car completes laps without unstable motion; steering and braking
//!    have predictable effects." (stability + physics unit tests)
//! 3. "Record average, p99, p99.9, worst tick time, and memory use."
//! 4. Seeded reset and pause hold state (build items).

use apexsim::replay::{read_log, write_log};
use apexsim::driver::DriverParams;
use apexsim::sim::{replay_session, run_session, Phase, Sim, SimConfig, DT};

fn cfg(laps: u32, seed: u64, noise: f64) -> SimConfig {
    SimConfig {
        laps,
        seed,
        driver: DriverParams {
            noise_amplitude: noise,
            ..DriverParams::default()
        },
        ..SimConfig::default()
    }
}

#[test]
fn gate1_same_seed_replays_identically() {
    let (o1, i1) = run_session(cfg(2, 7, 0.002), None);
    let (o2, i2) = run_session(cfg(2, 7, 0.002), None);
    assert_eq!(o1.state_hash, o2.state_hash, "identical seeds must give identical trajectories");
    assert_eq!(o1.lap_times, o2.lap_times);
    assert_eq!(i1.len(), i2.len());
    // The recorded inputs themselves are the replay medium — feed them back.
    let rp = replay_session(cfg(2, 7, 0.002), &i1);
    assert_eq!(rp.state_hash, o1.state_hash, "replay must be bit-identical to the live run");
    assert_eq!(rp.ticks, o1.ticks);
    assert_eq!(rp.lap_times, o1.lap_times);
}

#[test]
fn different_seed_diverges_when_noise_present() {
    let (o1, _) = run_session(cfg(1, 7, 0.002), None);
    let (o2, _) = run_session(cfg(1, 8, 0.002), None);
    assert_ne!(o1.state_hash, o2.state_hash, "seed must influence the run when noise > 0");
}

#[test]
fn gate2_laps_complete_without_instability() {
    let (o, _) = run_session(cfg(3, 42, 0.0), None);
    assert_eq!(o.laps_done, 3, "must complete the requested laps");
    assert_eq!(o.lap_times.len(), 3);
    for (n, &t) in o.lap_times.iter().enumerate() {
        assert!(t.is_finite(), "lap {} time not finite: {t}", n + 1);
        assert!(
            (10.0..=60.0).contains(&t),
            "lap {} time {t:.2} s outside plausible band",
            n + 1
        );
    }
    // Settled laps should be repeatable: compare laps 2 and 3.
    let a = o.lap_times[1];
    let b = o.lap_times[2];
    assert!((a - b).abs() / a < 0.15, "settled laps differ {a:.2} vs {b:.2} (>15%)");
    assert_eq!(o.offtrack_ticks, 0, "line follower should stay on track");
    let f = o.final_state;
    assert!(f.pos_x.is_finite() && f.pos_y.is_finite() && f.vel_long.is_finite());
    assert!(f.vel_long < 80.0, "speed {} exceeds model limits", f.vel_long);
}

#[test]
fn gate3_timing_report_present() {
    let (o, _) = run_session(cfg(1, 42, 0.0), None);
    let ts = o.tick_stats;
    assert!(ts.count > 0);
    assert_eq!(ts.count as u64, o.ticks, "one cost sample per physics tick");
    assert!(ts.avg_us > 0.0 && ts.p50_us >= ts.avg_us * 0.05);
    assert!(ts.p99_us >= ts.p50_us, "percentiles must be ordered");
    assert!(ts.p999_us >= ts.p99_us);
    assert!(ts.worst_us >= ts.p999_us);
    // Sanity on this hardware class: headroom vs the 4000 us budget.
    let avg = ts.avg_us;
    let p999 = ts.p999_us;
    assert!(avg < 400.0, "avg tick cost {avg:.0} us too high");
    assert!(p999 < 4000.0, "p99.9 tick cost {p999:.0} us exceeds budget");
}

#[test]
fn log_round_trip_and_replay_from_disk() {
    let (o, inputs) = run_session(cfg(2, 99, 0.001), None);
    let path = std::env::temp_dir().join(format!("apexsim_gate_{}.log", std::process::id()));
    write_log(&path, &cfg(2, 99, 0.001), &inputs, &o).expect("write log");
    let loaded = read_log(&path).expect("read log");
    assert_eq!(loaded.config.seed, 99);
    assert_eq!(loaded.config.laps, 2);
    assert_eq!(loaded.inputs.len(), inputs.len());
    assert_eq!(loaded.expected.state_hash, o.state_hash);
    let rp = replay_session(loaded.config, &loaded.inputs);
    assert_eq!(rp.state_hash, o.state_hash, "disk replay must match original hash");
    assert_eq!(rp.lap_times, o.lap_times);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn pause_holds_state_and_resume_continues() {
    let mut sim = Sim::new(cfg(1, 42, 0.0));
    sim.state.phase = Phase::Running;
    for _ in 0..1000 {
        sim.tick_with_driver();
    }
    let before = sim.state.car;
    let tick_before = sim.state.tick;

    sim.pause();
    assert_eq!(sim.state.phase, Phase::Paused);
    for _ in 0..50 {
        sim.tick_with_driver(); // must be a no-op while paused
    }
    assert_eq!(sim.state.tick, tick_before, "paused sim must not advance ticks");
    assert_eq!(sim.state.car, before, "paused sim must not alter state");

    sim.resume();
    assert_eq!(sim.state.phase, Phase::Running);
    for _ in 0..1000 {
        sim.tick_with_driver();
    }
    assert!(sim.state.tick > tick_before);
    assert!(sim.state.car.vel_long > 5.0, "should be moving after resume");
}

#[test]
fn seeded_reset_reproduces_initial_run() {
    let mut sim = Sim::new(cfg(1, 1234, 0.0));
    sim.state.phase = Phase::Running;
    for _ in 0..5000 {
        sim.tick_with_driver();
    }
    let hash_a = sim.state_hash();
    let car_a = sim.state.car;

    sim.reset(1234);
    assert_eq!(sim.state.tick, 0, "reset must rewind the clock");
    sim.state.phase = Phase::Running;
    for _ in 0..5000 {
        sim.tick_with_driver();
    }
    assert_eq!(sim.state_hash(), hash_a, "reset(seed) must reproduce the run");
    assert_eq!(sim.state.car, car_a);
}

#[test]
fn dt_is_the_plan_value() {
    assert!((DT - 0.004).abs() < 1e-12, "Phase 1 runs the provisional 250 Hz fixed step");
}
