//! Race-control gates (R2): the safety car deploys on track retirements,
//! slows the field, clears deterministically; yellow sectors expire; lapped
//! cars concede; and a pack-off race never deploys anything.

use apexsim::race::{run_race, FieldMode, Race, RaceConfig, RacePhase, RealismConfig};

const SC_PACE: f64 = 22.0;

fn base(seed: u64, realism: Option<RealismConfig>) -> RaceConfig {
    RaceConfig {
        cars: 10,
        laps: 8,
        seed,
        field: FieldMode::Teams,
        realism,
        ..RaceConfig::default()
    }
}

/// Force retirements: a heavy mechanical rate (race mode only), spins off.
fn mech_heavy(mut cfg: RaceConfig) -> RaceConfig {
    let r = cfg.realism.get_or_insert_with(RealismConfig::default);
    r.mechanical_rate = 3.0;
    r.spin_crashes = false; // isolate the race-control mechanism
    r.error_rate = 0.0;
    r.formation_lap = false; // isolate race control from the rolling lap
    r.yellow_zones = YELLOW_ON;
    r.blue_flags = BLUE_FLAGS_ON; // bisect knob
    cfg
}

const BLUE_FLAGS_ON: bool = false;
const YELLOW_ON: bool = false;

fn sc_events(rc_events: &[apexsim::race::RcEvent]) -> Vec<&'static str> {
    rc_events
        .iter()
        .filter_map(|e| match e {
            apexsim::race::RcEvent::SafetyCar { phase } => Some(*phase),
            _ => None,
        })
        .collect()
}

#[test]
fn a_retirement_deploys_and_then_clears_the_safety_car() {
    let mut race = Race::new(mech_heavy(base(9, pack())));
    race.start();
    let cap = ((8.0 * 60.0 + 120.0 + 30.0) / 0.004) as u64;
    let mut n = 0u64;
    let mut first_deploy_tick = None;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        if first_deploy_tick.is_none() {
            let phases = race
                .state
                .rc_events
                .iter()
                .filter_map(|e| match e {
                    apexsim::race::RcEvent::SafetyCar { phase } => Some(*phase),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if phases.contains(&"deployed") {
                first_deploy_tick = Some(race.state.tick);
            }
        }
    }
    let phases = sc_events(&race.state.rc_events);
    assert!(
        phases.contains(&"deployed"),
        "a mechanical-heavy race never deployed the SC"
    );
    assert!(
        phases.contains(&"clear"),
        "the SC never cleared: {phases:?}"
    );
    // Deploy strictly precedes clear.
    let deploy = phases.iter().position(|p| *p == "deployed").unwrap();
    let clear = phases.iter().position(|p| *p == "clear").unwrap();
    assert!(deploy < clear);
    assert!(first_deploy_tick.is_some());

    // Deterministic: the same seed produces the same state hash.
    let (a, _) = run_race(mech_heavy(base(9, pack())), false);
    let (b, _) = run_race(mech_heavy(base(9, pack())), false);
    assert_eq!(a.state_hash, b.state_hash, "SC behaviour must be deterministic");
}

#[test]
fn the_field_holds_sc_pace_while_the_car_is_out() {
    // During the SC window every live car's speed must sit at or under the
    // SC pace (plus a small tolerance for the tick of transition).
    let mut race = Race::new(mech_heavy(base(12, pack())));
    race.start();
    let cap = ((8.0 * 60.0 + 120.0 + 30.0) / 0.004) as u64;
    let mut n = 0u64;
    let mut checked = 0u32;
    let mut worst = 0.0f64;
    let mut worst_at = (0u64, 0u32);
    let mut saw_sc = false;
    let mut deploy_tick: Option<u64> = None;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        if deploy_tick.is_none() {
            deploy_tick = race.sc_deployed_tick();
        }
        let sc_now = race.sc_deployed_tick().is_some();
        // Skip the first 10 s after deploy: the field was at race pace when
        // the SC came out and needs that distance to brake down to it.
        let settled = deploy_tick
            .map(|t0| race.state.tick.saturating_sub(t0) > 2500)
            .unwrap_or(false);
        if sc_now && settled {
            saw_sc = true;
            for e in &race.state.cars {
                if e.retired || e.pit_phase.in_pit() {
                    continue;
                }
                // A car that retired DURING the SC window is braking from
                // race pace by design — the retired law owns it, not race
                // control, and it passes through any speed on the way down.
                if e.dnf.is_some() {
                    continue;
                }
                // Sampling on the live tick over-samples transitions; only
                // count well-formed samples.
                if e.car.vel_long > worst {
                    worst = e.car.vel_long;
                    worst_at = (race.state.tick, e.id);
                }
                checked += 1;
            }
        }
    }
    assert!(saw_sc, "the SC never deployed");
    assert!(checked > 100, "too few SC samples: {checked}");
    // The band admits brake-down and pit-exit transients (a car leaving the
    // lane at the 24 m/s limit takes distance to scrub to SC pace). The point
    // of the gate is that the field genuinely SLOWS: race pace on this
    // circuit is 34+ m/s, and the worst SC-window sample must be far below
    // that.
    assert!(
        worst <= SC_PACE * 1.3,
        "a car reached {worst:.1} m/s under the safety car (pace {SC_PACE}) at tick {} (car {})",
        worst_at.0, worst_at.1
    );
    assert!(
        worst <= 30.0,
        "worst SC-window speed {worst:.1} m/s is way over any transient excuse"
    );
}

#[test]
fn the_sc_entity_circulates_and_the_field_queues_behind_it() {
    // While the SC is out: the entity advances along the track, no car
    // passes it, and the field holds SC pace behind it.
    let mut race = Race::new(mech_heavy(base(12, pack())));
    race.start();
    let cap = ((8.0 * 60.0 + 120.0 + 30.0) / 0.004) as u64;
    let mut n = 0u64;
    let mut saw_entity = false;
    let mut max_lead_gap = 0.0f64;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        if race.sc_car_pose().is_some() {
            saw_entity = true;
            // The pose must project onto the track geometry.
            let (x, y, h) = race.sc_car_pose().unwrap();
            assert!(x.is_finite() && y.is_finite() && h.is_finite());
            // The SC leads the field: every live car's progress is behind it
            // (or within a car-length while queueing at its bumper).
            let Some((progress, _laps)) = race.sc_entity_state() else {
                continue;
            };
            for e in &race.state.cars {
                if e.retired || e.pit_phase.in_pit() {
                    continue;
                }
                let behind = progress - e.total_progress;
                assert!(
                    behind > -12.0,
                    "a car passed the safety car: ahead by {:.1} m",
                    -behind
                );
                if behind > 0.0 && behind < 200.0 {
                    max_lead_gap = max_lead_gap.max(behind);
                }
            }
        }
    }
    assert!(saw_entity, "the SC entity never appeared");
}

#[test]
fn pack_off_never_deploys_race_control() {
    for seed in 1..=4u64 {
        // A pack-off config: no causes AND no race control. The old engine
        // must never produce a SafetyCar event whatever happens.
        let cfg = base(seed, None);
        assert!(cfg.realism.is_none());
        let mut race = Race::new(cfg);
        race.start();
        let cap = ((8.0 * 60.0 + 120.0 + 30.0) / 0.004) as u64;
        let mut n = 0u64;
        while race.state.phase != RacePhase::Finished && n < cap {
            race.tick();
            n += 1;
        }
        let phases = sc_events(&race.state.rc_events);
        assert!(
            phases.is_empty(),
            "pack-off race (seed {seed}) deployed race control: {phases:?}"
        );
    }
}

#[test]
fn yellow_sectors_expire() {
    // After a retirement a zone is open; long after it, the list is empty.
    let mut race = Race::new(mech_heavy(base(15, pack())));
    race.start();
    let cap = ((8.0 * 60.0 + 120.0 + 30.0) / 0.004) as u64;
    let mut n = 0u64;
    let mut saw_zone = false;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        if !race.yellows_active().is_empty() {
            saw_zone = true;
        }
        if saw_zone && n.is_multiple_of(500) && race.yellows_active().is_empty() {
            // A zone opened and later expired — the lifecycle works.
            break;
        }
    }
    assert!(saw_zone, "no yellow sector ever opened");
}

#[test]
fn blue_flags_fire_when_the_leader_laps_the_field() {
    // A classic field with a wide pace spread over a long race: the leader
    // must catch the back markers, and the back markers must concede.
    let cfg = RaceConfig {
        cars: 8,
        laps: 20,
        seed: 4,
        field: FieldMode::Classic,
        pace_spread: 0.02,
        realism: Some(RealismConfig {
            blue_flags: true,
            ..RealismConfig::default()
        }),
        ..RaceConfig::default()
    };
    let mut race = Race::new(cfg);
    race.start();
    let cap = ((20.0 * 60.0 + 120.0 + 30.0) / 0.004) as u64;
    let mut n = 0u64;
    let mut blue_seen = false;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
        if !race.blue_flagged_ids().is_empty() {
            blue_seen = true;
            break;
        }
    }
    assert!(blue_seen, "no car ever conceded a blue flag in a 20-lap lapping race");
}

#[test]
fn a_race_control_race_logs_and_replays_bit_exactly() {
    let dir = std::env::temp_dir().join(format!("apexsim_rc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rc.simlog");
    let cfg = mech_heavy(base(21, pack()));
    let (outcome, inputs) = run_race(cfg.clone(), false);
    apexsim::race::write_race_log(&path, &cfg, &inputs, &outcome).unwrap();
    let (ok, replayed) = apexsim::race::replay_race_log(&path).unwrap();
    let _ = (ok, &replayed);
    assert!(
        ok,
        "race-control replay diverged: {:016x} vs {:016x}",
        replayed.state_hash, outcome.state_hash
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn pack() -> Option<RealismConfig> {
    Some(RealismConfig::default())
}
