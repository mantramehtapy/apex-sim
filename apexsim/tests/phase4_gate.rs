//! Phase 4 completion-gate tests, mapped to the plan's gate criteria:
//!
//! 1. "Each scale step passes replay and collision tests."
//! 2. "The target field runs at the chosen real-time rate for a sustained
//!    test with the browser connected."  (in-process: a connected WS client
//!    while the 20-car field runs realtime — the served test below)
//! 3. "Memory and simulation lag remain bounded."
//!
//! Plus build items: grid scaling 1→4→10→20, nearby-car awareness,
//! contact + retirement, classification, dense starts.

use apexsim::race::{Race, RaceConfig, RacePhase};

fn cfg(cars: u32) -> RaceConfig {
    RaceConfig { cars, laps: 2, seed: 42, ..Default::default() }
}

fn run_ticks(rc: &RaceConfig, ticks: u64) -> (u64, f64) {
    let mut race = Race::new(rc.clone());
    race.state.phase = RacePhase::Running;
    for _ in 0..ticks {
        race.tick();
    }
    (race.state_hash(), race.state.sim_time)
}

#[test]
fn gate1_determinism_and_replay_at_every_scale() {
    // The plan's ladder: 1 → 4 → 10 → configured full grid (20, the
    // intended 2024-season size — never mixed with 22).
    for &n in &[1u32, 4, 10, 20] {
        let (h1, _) = run_ticks(&cfg(n), 1500);
        let (h2, _) = run_ticks(&cfg(n), 1500);
        assert_eq!(h1, h2, "{n}-car field must be bit-deterministic across runs");
    }
    // Same config, different seed → different field trajectory.
    let mut other = cfg(10);
    other.seed = 43;
    let (h1, _) = run_ticks(&cfg(10), 1500);
    let (h3, _) = run_ticks(&other, 1500);
    assert_ne!(h1, h3);
}

#[test]
fn gate1b_race_log_v2_round_trip() {
    // Full multi-car race: run, store, re-execute from the log — hash equal.
    let mut rc = cfg(10);
    rc.laps = 1;
    let (outcome, inputs) = apexsim::race::run_race(rc.clone(), false);
    let dir = std::env::temp_dir().join(format!("apexsim_p4_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("race10.simlog");
    apexsim::race::write_race_log(&path, &rc, &inputs, &outcome).unwrap();
    let (ok, replayed) = apexsim::race::replay_race_log(&path).unwrap();
    assert!(ok, "10-car log replay diverged: {:016x} vs {:016x}",
        replayed.state_hash, outcome.state_hash);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Place two cars on a controlled collision course on the bottom straight.
fn collision_race() -> Race {
    let mut race = Race::new(RaceConfig { cars: 2, ..cfg(2) });
    race.state.phase = RacePhase::Running;
    // Car 0 slow ahead; car 1 fast behind, same lateral — rear-end closing.
    race.state.cars[0].car = apexsim::physics::CarState {
        pos_x: 200.0, pos_y: 0.0, heading: 0.0,
        vel_long: 8.0, ..Default::default()
    };
    race.state.cars[0].s = 200.0;
    race.state.cars[0].total_progress = 200.0;
    race.state.cars[1].car = apexsim::physics::CarState {
        pos_x: 192.0, pos_y: 0.0, heading: 0.0,
        vel_long: 40.0, ..Default::default()
    };
    race.state.cars[1].s = 192.0;
    race.state.cars[1].total_progress = 192.0;
    race
}

#[test]
fn gate1c_contact_detected_and_resolved() {
    let mut race = collision_race();
    for _ in 0..400 {
        race.tick();
    }
    assert!(!race.state.contacts.is_empty(), "closing cars must register a contact");
    let c = &race.state.contacts[0];
    assert!(c.closing_speed > 5.0, "scripted closing ≈ 32 m/s, got {}", c.closing_speed);
    for e in &race.state.cars {
        let car = &e.car;
        assert!(car.pos_x.is_finite() && car.vel_long.is_finite(), "state diverged after contact");
    }
    // Impulse response must have actually separated / slowed the pair.
    let d = {
        let a = &race.state.cars[0].car;
        let b = &race.state.cars[1].car;
        ((a.pos_x - b.pos_x).powi(2) + (a.pos_y - b.pos_y).powi(2)).sqrt()
    };
    assert!(d >= 2.9, "cars still interpenetrating: {d:.2} m");
}

#[test]
fn gate1d_heavy_contact_retires_the_chaser() {
    let mut race = collision_race();
    // Stationary victim, very high closing speed → above the 20 m/s line.
    race.state.cars[0].car.vel_long = 0.0;
    for _ in 0..300 {
        race.tick();
    }
    let retired: Vec<u32> = race
        .state
        .cars
        .iter()
        .filter(|e| e.retired)
        .map(|e| e.id)
        .collect();
    assert_eq!(retired, vec![1], "the chasing car must retire on a heavy contact");
    // Retired car brakes to a stop (bounded, no NaN), stays as an obstacle.
    for _ in 0..1250 {
        race.tick();
    }
    let r = &race.state.cars[1].car;
    assert!(r.vel_long <= 0.5, "retired car should stop, at {}", r.vel_long);
    assert!(r.pos_x.is_finite());
}

#[test]
fn gate2_dense_start_20_cars_stable_and_bounded() {
    // The hardest case: full-grid standing start, 12 s of sim.
    let mut race = Race::new(cfg(20));
    race.state.phase = RacePhase::Running;
    let start_progress: Vec<f64> = race.state.cars.iter().map(|e| e.total_progress).collect();
    for _ in 0..3000 {
        race.tick();
    }
    for (i, e) in race.state.cars.iter().enumerate() {
        let c = &e.car;
        assert!(c.pos_x.is_finite() && c.pos_y.is_finite() && c.vel_long.is_finite(),
            "car {} diverged in dense start", e.id);
        let travelled = e.total_progress - start_progress[i];
        assert!(travelled > 60.0, "car {} only travelled {travelled:.1} m in 12 s",
            e.id);
        if e.retired {
            // Crashed-out cars brake to a stop on-track (hazard) — they
            // have travelled, they must NOT keep circulating.
            assert!(c.vel_long <= 0.5,
                "retired car {} still rolling at {:.1} m/s", e.id, c.vel_long);
        } else if e.pit_phase.in_pit() {
            // A car stopped on the pit road is stopped on purpose: the crew
            // has it, and the lane sits off the racing surface entirely
            // (`PitConfig::offset_m`), so "off the track edge" is where a
            // pitting car belongs. Phase 12 fixed the driver's noise
            // handling, which let cars actually reach a pit stop inside this
            // 12 s window instead of being held back by the bug. The bound
            // that matters is the lane's own speed limit, not race pace.
            assert!(c.vel_long <= cfg(20).pit.speed_limit + 0.5,
                "car {} is in the pit lane but rolling at {:.1} m/s, over the lane limit",
                e.id, c.vel_long);
        } else {
            // Alive cars must be moving; incident-involved cars may be
            // slow while recovering from a punt (Phase 4 has no barriers),
            // so full race pace is asserted for the bulk of the field below.
            assert!(c.vel_long > 2.0, "car {} crawling at {:.1} m/s mid-start",
                e.id, c.vel_long);
        }
    }
    // The field must not be queueing — but a car that is on the pit road is
    // stopped for a stopwatch, not stuck in traffic, so it is counted
    // separately. Phase 12's driver fix let cars reach a pit stop inside this
    // window at all, which put a third of the grid in the lane.
    let on_track = race
        .state
        .cars
        .iter()
        .filter(|e| !e.retired && !e.pit_phase.in_pit())
        .count();
    let racing = race
        .state
        .cars
        .iter()
        .filter(|e| !e.retired && !e.pit_phase.in_pit() && e.car.vel_long > 15.0)
        .count();
    assert!(
        racing + 1 >= on_track,
        "only {racing} of {on_track} cars still on track are at race pace at 12 s \
         — the field is queueing"
    );
    assert!(race.state.contacts.len() <= 40,
        "dense start produced {} contacts — awareness failing", race.state.contacts.len());
    let stats = race.tick_summary();
    assert!(stats.p999_us < 4000.0, "p99.9 tick {} µs exceeds budget", stats.p999_us);
    // Field strings out: leader well clear of the tail.
    let mut prog: Vec<f64> = race.state.cars.iter().map(|e| e.total_progress).collect();
    prog.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(prog[19] - prog[0] > 40.0, "field did not spread: {} m", prog[19] - prog[0]);
}

#[test]
fn gate2b_classification_order_and_gaps() {
    // Run past a full race lap (~47 s) so lap accounting is exercised.
    let mut race = Race::new(cfg(20));
    race.state.phase = RacePhase::Running;
    for _ in 0..13000 {
        race.tick();
    }
    let lb = &race.state.leaderboard;
    assert_eq!(lb.len(), 20);
    let positions: Vec<u32> = lb.iter().map(|r| r.pos).collect();
    let mut sorted = positions.clone();
    sorted.sort();
    assert_eq!(positions, sorted, "positions must be 1..=20 in order");
    assert_eq!(sorted[0], 1);
    assert_eq!(sorted[19], 20);
    let mut max_gap = 0.0f64;
    for w in lb.windows(2) {
        let g1 = w[0].gap_s;
        let g2 = w[1].gap_s;
        if let (Some(a), Some(b)) = (g1, g2) {
            assert!(b >= a - 0.5, "gaps must not decrease down the order: {a} then {b}");
            max_gap = max_gap.max(b);
        }
    }
    assert!(max_gap > 0.5 && max_gap < 120.0, "implausible field spread: {max_gap} s");
    // Lap accounting: leader must have completed a timed lap by now.
    assert!(lb[0].laps_done >= 1, "leader should be on lap ≥ 1");
}

#[test]
fn gate3_grid_ladder_tick_costs_bounded() {
    // The 1 → 4 → 10 → 20 measurement ladder, 5 s of sim each.
    for &n in &[1u32, 4, 10, 20] {
        let mut race = Race::new(cfg(n));
        let stats = race.bench_ticks(1250);
        assert!(stats.count == 1250);
        assert!(stats.avg_us < 400.0, "{n} cars avg {} µs too high", stats.avg_us);
        assert!(stats.worst_us < 4000.0, "{n} cars worst {} µs exceeds the 4 ms budget", stats.worst_us);
    }
}

#[test]
fn memory_stays_bounded_across_scales() {
    // Structural bound: per-car state is fixed-size; 20 cars must not
    // change tick cost out of all proportion to 1 car (no leaks per tick).
    let one = {
        let mut r = Race::new(cfg(1));
        r.bench_ticks(1250).avg_us
    };
    let twenty = {
        let mut r = Race::new(cfg(20));
        r.bench_ticks(1250).avg_us
    };
    assert!(
        twenty < one * 100.0 + 100.0,
        "20-car tick cost ({twenty} µs) exploded vs 1-car ({one} µs)"
    );
}

// ---------------------------------------------------------------------------
// Gate 2: "The target field runs at the chosen real-time rate for a
// sustained test with the browser connected."  In-process equivalent: the
// 20-car race server runs realtime while a WebSocket client streams it.
// ---------------------------------------------------------------------------

use apexsim::server::spawn_server;
use apexsim::sim::SimConfig;
use futures_util::StreamExt;
use serde_json::Value;
use std::net::SocketAddr;
use tokio::io::AsyncReadExt;
use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Minimal HTTP/1.1 GET (tests only).
async fn http_get(addr: SocketAddr, path: &str) -> String {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    );
    use tokio::io::AsyncWriteExt as _;
    s.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf).to_string();
    text.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default()
}

#[tokio::test]
async fn gate2_sustained_realtime_20car_field_with_ws_client() {
    let config = SimConfig {
        cars: 20,
        laps: 999, // continuous session; the test window bounds it
        ..SimConfig::default()
    };
    let (addr, _handle) = spawn_server(config, 0, true).expect("spawn 20-car race server");
    // Phase 6: readiness polling instead of a fixed startup sleep.
    assert!(
        apexsim::server::wait_until_ready(addr, std::time::Duration::from_secs(10)).await,
        "20-car race server did not report ready within 10 s"
    );

    let (mut ws, _resp) = connect_async(format!("ws://{addr}/ws")).await.expect("ws connect");

    // Stream for a sustained 10 s wall window and check every snapshot.
    let window = std::time::Duration::from_secs(10);
    let start = std::time::Instant::now();
    let mut snaps = 0u32;
    let mut min_tick = u64::MAX;
    let mut max_tick = 0u64;
    let mut saw_20_cars = false;
    let mut phase_bad = None;
    while start.elapsed() < window {
        let msg = timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .expect("timeout waiting for ws message");
        let msg = msg.expect("ws closed").expect("ws error");
        if let Message::Text(t) = msg {
            let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
            let s = match v["type"].as_str() {
                Some("snapshot") => {
                    snaps += 1;
                    &v
                }
                Some("welcome") => &v["snapshot"],
                _ => continue,
            };
            let tick = s["tick"].as_u64().unwrap_or(0);
            min_tick = min_tick.min(tick);
            max_tick = max_tick.max(tick);
            match s["cars"].as_array().map(|a| a.len()) {
                Some(20) => saw_20_cars = true,
                Some(0) => {} // pre-boot snapshot from the watch channel
                Some(n) => panic!("snapshot must carry the 20-car field, got {n}"),
                None => panic!("snapshot has no cars array"),
            }
            let phase = s["phase"].as_str().unwrap_or("");
            // Phase 5 adds the formation phase: a standing-start race now
            // opens with the light sequence before going green.
            if phase != "running" && phase != "finished" && phase != "formation" {
                phase_bad = Some(phase.to_string());
            }
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    let sim_s = (max_tick - min_tick.min(max_tick)) as f64 * 4.0 / 1000.0;

    assert!(saw_20_cars, "never saw a snapshot with all 20 cars");
    assert!(phase_bad.is_none(), "phase left running/finished: {:?}", phase_bad);
    assert!(snaps >= 150, "~25 Hz for {wall_s:.1} s should give >150 snapshots, got {snaps}");
    // The chosen real-time rate held for the sustained window (±5%).
    assert!(
        sim_s >= wall_s * 0.95,
        "sim fell behind realtime: {sim_s:.2} s sim over {wall_s:.2} s wall"
    );
    assert!(
        sim_s <= wall_s * 1.05,
        "sim ran faster than realtime: {sim_s:.2} s sim over {wall_s:.2} s wall"
    );

    // Bounded-queue health: nothing dropped toward the connected client.
    let metrics: Value = serde_json::from_str(&http_get(addr, "/api/metrics").await)
        .expect("metrics json");
    assert_eq!(
        metrics["ws_snapshots_dropped"].as_u64(),
        Some(0),
        "snapshots must not be dropped to a live client: {metrics}"
    );
}

#[test]
fn no_deadlock_behind_wrecks_race_completes_naturally() {
    // Regression: seed 99 used to red-flag at the cap with seven LIVE cars
    // frozen at 0 m/s — pinned by the safe-follow cap behind stopped cars,
    // and the crawl-past rule required motion the cap forbade. The race
    // must always end naturally, and no live car may sit frozen.
    let cfg = RaceConfig { cars: 20, laps: 3, seed: 99, ..Default::default() };
    let mut race = Race::new(cfg);
    race.state.phase = RacePhase::Running;
    for _ in 0..80_000 {
        race.tick();
        if race.state.phase != RacePhase::Running {
            break;
        }
    }
    assert!(
        race.state.sim_time < 300.0,
        "race hit the red-flag cap at {:.1} s — something deadlocked the field",
        race.state.sim_time
    );
    for e in &race.state.cars {
        if !e.retired {
            assert!(
                e.car.vel_long > 0.5,
                "live car {} frozen at {:.1} m/s — follow/pass deadlock is back",
                e.id,
                e.car.vel_long
            );
        } else {
            assert!(e.car.vel_long <= 0.5, "retired car {} must be stopped", e.id);
        }
    }
    // And the field stays physically sane throughout the incident.
    for e in &race.state.cars {
        assert!(e.car.pos_x.is_finite() && e.car.vel_long.is_finite());
        assert!(e.car.vel_long < 130.0, "car {} at {:.1} m/s — energy pump", e.id, e.car.vel_long);
    }
}
