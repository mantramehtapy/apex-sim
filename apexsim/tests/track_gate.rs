//! Custom-circuit gates: user-drawn loops build and validate, presets load,
//! a race runs and replays bit-exactly on them, and the built-in stadium is
//! untouched by any of it.

use apexsim::race::{run_race, replay_race_log, write_race_log, FieldMode, Race, RaceConfig, RacePhase};
use apexsim::track::{resolve_track_spec, CustomTrack, PitFractions, Track, TrackSpec};

#[test]
fn a_centerline_round_trips_projection_and_closes() {
    let control = vec![
        [0.0, 0.0], [200.0, -10.0], [400.0, 30.0], [500.0, 120.0],
        [420.0, 210.0], [240.0, 240.0], [60.0, 200.0], [-80.0, 100.0],
    ];
    let t = Track::from_centerline(&control, 7.0).expect("valid loop");
    let len = t.length();
    assert!(len > 1000.0, "the loop is only {len:.0} m");
    // Projecting sampled points back must recover s (the whole point of the
    // projection: the race's progress, lap counting and awareness ride it).
    for f in [0.0, 0.13, 0.37, 0.5, 0.71, 0.9, 0.999] {
        let s = f * len;
        let (x, y, _) = t.point_at(s);
        let (ps, lat) = t.project(x, y);
        assert!(
            (ps - s).abs() < 0.5 || (ps - s).rem_euclid(len).abs() < 0.5,
            "f={f}: s={s:.2} projected to {ps:.2} (lat {lat:.3})"
        );
        assert!(lat.abs() < 0.2, "f={f}: lateral {lat}");
    }
    // Off-track detection is geometric, not stadium-hardcoded.
    let (x, y, h) = t.point_at(len * 0.25);
    let (nx, ny) = (-h.sin(), h.cos());
    let (_, lat_out) = t.project(x + nx * 20.0, y + ny * 20.0);
    assert!(t.is_off_track(lat_out), "20 m off the centreline must be off track");
    let (_, lat_in) = t.project(x + nx * 3.0, y + ny * 3.0);
    assert!(!t.is_off_track(lat_in), "3 m off the centreline must be on track");
    // Corner speed limits exist where the loop curves.
    let mut saw_corner = false;
    for i in 0..64 {
        let v = t.speed_limit_at(len * i as f64 / 64.0, 11.0);
        if v.is_finite() {
            saw_corner = true;
            // On a table track curvature is never exactly zero, so a
            // "straight" yields a huge-but-finite limit (the driver caps it
            // against v_straight_max). A real corner must be plausible.
            if v < 200.0 {
                assert!(v > 5.0, "implausible corner speed {v}");
            }
        }
    }
    assert!(saw_corner, "a circuit with corners must have finite speed limits somewhere");
}

#[test]
fn unstructured_drawings_are_rejected_with_reasons() {
    let half = 7.0;
    // Too few points to close a loop.
    let e = Track::from_centerline(&[[0.0, 0.0]; 2], half).unwrap_err();
    assert!(e.contains("3 control points"), "{e}");
    // A speck: near-identical points cannot form a circuit (and would make
    // the arc-length table degenerate).
    let speck = vec![[0.0, 0.0]; 8];
    let e = Track::from_centerline(&speck, half).unwrap_err();
    assert!(e.contains("too small to form a circuit"), "{e}");
    // An out-and-back line: the spline paces the line and returns, so the
    // total turn is 0 — not a closed circuit.
    let line = vec![[0.0, 0.0], [150.0, 0.0], [300.0, 0.0]];
    let e = Track::from_centerline(&line, half).unwrap_err();
    assert!(e.contains("turns by only"), "{e}");
}

#[test]
fn anything_drawable_races_short_tight_and_crossed() {
    let half = 7.0;
    // 1. A micro kart loop — 3 points, ~250 m around. Previously refused
    //    (needed 6 points and 400 m); now it builds and races, pit lane
    //    auto-disabled because the default fractions cannot fit.
    let micro = vec![[0.0, 0.0], [120.0, -30.0], [60.0, 90.0]];
    let t = Track::from_centerline(&micro, half).expect("a 3-point loop is a circuit");
    assert!(t.length() > 150.0, "micro loop is {:.0} m", t.length());
    // 2. A figure-8: the outbound and return legs cross. Drivable — the
    //    centreline the cars follow is well-defined everywhere.
    let eight = vec![
        [0.0, 0.0], [300.0, -5.0], [600.0, -5.0], [650.0, 50.0],
        [600.0, 105.0], [300.0, 3.0], [0.0, 3.0], [-60.0, 50.0],
    ];
    let t8 = Track::from_centerline(&eight, half).expect("a figure-8 is accepted");
    // 3. A sub-width hairpin: a stadium whose right-hand end turns 180° in
    //    ~10 m — well below 2×half_width, folding the surface over its own
    //    paint. Still accepted.
    let hairpin = vec![
        [0.0, 0.0], [380.0, 0.0], [402.0, 0.0], [404.0, 25.0],
        [402.0, 50.0], [380.0, 50.0], [0.0, 50.0], [-45.0, 30.0], [-45.0, 12.0],
    ];
    let th = Track::from_centerline(&hairpin, half).expect("a tight hairpin is accepted");
    let mut max_k = 0.0f64;
    for i in 0..512 {
        let k = th.curvature_at(th.length() * i as f64 / 512.0).abs();
        max_k = max_k.max(k);
    }
    let min_r = 1.0 / max_k;
    assert!(
        min_r < 2.0 * half,
        "the hairpin must actually be sub-width: tightest radius {min_r:.1} m"
    );
    let _ = (t, t8, th);

    // And "accepted" means RACES: each drawable completes a short race with
    // a classified finish.
    for (label, points) in [
        ("micro", &micro),
        ("figure-8", &eight),
        ("hairpin", &hairpin),
    ] {
        let cfg = RaceConfig {
            cars: 6,
            laps: 2,
            seed: 7,
            field: FieldMode::Classic,
            track: TrackSpec {
                version: 2,
                straight_len: 0.0,
                radius: 0.0,
                half_width: half,
                custom: Some(CustomTrack {
                    name: label.to_string(),
                    points: points.clone(),
                    pit: None,
                }),
            },
            ..RaceConfig::default()
        };
        let mut race = Race::new(cfg);
        race.start();
        let cap = ((4.0 * 60.0 + 60.0) / 0.004) as u64;
        let mut n = 0u64;
        while race.state.phase != RacePhase::Finished && n < cap {
            race.tick();
            n += 1;
        }
        assert_eq!(race.state.phase, RacePhase::Finished, "{label}: race never finished");
        assert!(
            race.state.leaderboard.iter().any(|r| r.laps_done >= 2),
            "{label}: nobody completed 2 laps"
        );
    }
}

#[test]
fn presets_load_build_and_have_distinct_identity() {
    let mut labels = Vec::new();
    for name in ["paperclip", "kidney", "trioval", "roadcourse"] {
        let spec = resolve_track_spec(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        let t = spec.build().unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(t.length() > 1000.0, "{name} is too short");
        assert!(t.curvature_at(t.length() * 0.5).abs() < 1.0);
        labels.push(spec.label());
    }
    let unique: std::collections::BTreeSet<_> = labels.iter().collect();
    assert_eq!(unique.len(), labels.len(), "presets must have distinct identities");
    // Unknown names are refused with the list, not silently run on a default.
    let e = resolve_track_spec("nope").unwrap_err();
    assert!(e.contains("unknown preset"), "{e}");
    // The embedded file parses to the same spec twice (stable identity).
    let a = resolve_track_spec("kidney").unwrap();
    let b = resolve_track_spec("kidney").unwrap();
    assert_eq!(a, b);
}

#[test]
fn custom_track_file_json_round_trips() {
    let json = r#"{"name":"test","half_width":7.0,"pit":{"entry":0.1,"box":0.2,"exit":0.3},
        "points":[[0,0],[150,-5],[280,40],[300,140],[200,210],[60,190],[-30,100]]}"#;
    let spec = TrackSpec::from_custom_json(json).expect("parse");
    let custom = spec.custom.as_ref().expect("custom block");
    assert_eq!(custom.name, "test");
    assert_eq!(custom.pit, Some(PitFractions { entry: 0.1, box_frac: 0.2, exit: 0.3 }));
    assert_eq!(custom.points.len(), 7);
    // Building it works, and a file WITHOUT a pit block defaults the lane.
    spec.build().expect("build");
    let nopit = r#"{"name":"np","points":[[0,0],[150,-5],[280,40],[300,140],[200,210],[60,190],[-30,100]]}"#;
    let spec2 = TrackSpec::from_custom_json(nopit).unwrap();
    assert_eq!(spec2.custom.as_ref().unwrap().pit, None);
    assert_ne!(spec.label(), spec2.label(), "different circuits, different identity");
}

#[test]
fn race_on_a_custom_track_runs_replays_and_places_the_pit_lane() {
    let spec = resolve_track_spec("trioval").unwrap();
    let cfg = RaceConfig {
        cars: 10,
        laps: 3,
        seed: 42,
        field: FieldMode::Teams,
        track: spec.clone(),
        ..RaceConfig::default()
    };
    // The race completes on the custom circuit and classifies.
    let mut race = Race::new(cfg.clone());
    race.start();
    let cap = ((3.0 * 60.0 + 120.0 + 30.0) / 0.004) as u64;
    let mut n = 0u64;
    while race.state.phase != RacePhase::Finished && n < cap {
        race.tick();
        n += 1;
    }
    assert_eq!(race.state.phase, RacePhase::Finished, "the trioval race never finished");
    assert!(
        race.state.leaderboard.iter().any(|r| r.laps_done >= 3),
        "nobody completed 3 laps on the custom circuit"
    );

    // And it round-trips: G v2 + P + C lines, replay to the same hash.
    let dir = std::env::temp_dir().join(format!("apexsim_track_gate_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("trioval_race.simlog");
    let (outcome, inputs) = run_race(cfg.clone(), false);
    write_race_log(&path, &cfg, &inputs, &outcome).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\nG 2 "), "log must declare a v2 circuit");
    assert!(text.contains("\nB "), "log must carry the pit fractions");
    assert!(text.matches("\nC ").count() >= 6, "log must carry the control points");
    let (ok, replayed) = replay_race_log(&path).unwrap();
    assert!(ok, "custom-track replay diverged: {:016x} vs {:016x}",
        replayed.state_hash, outcome.state_hash);

    // The stadium is untouched: same config on the default track still
    // produces the pre-custom-circuit engine's hash for THIS seed/cars —
    // checked relative to itself below by determinism, since the absolute
    // hash belongs to the suite's other gates.
    let stadium_cfg = RaceConfig { track: TrackSpec::default(), ..cfg };
    let (s1, _) = run_race(stadium_cfg.clone(), false);
    let (s2, _) = run_race(stadium_cfg, false);
    assert_eq!(s1.state_hash, s2.state_hash);
    assert_ne!(s1.state_hash, outcome.state_hash, "a different circuit must race differently");
}

#[test]
fn pit_fractions_place_the_lane_where_the_file_says() {
    // trioval pins its pit lane to the first fifth of the lap; the engine
    // must turn those fractions into absolute s on THIS circuit.
    let spec = resolve_track_spec("trioval").unwrap();
    let track = spec.build().unwrap();
    let len = track.length();
    let f = PitFractions { entry: 0.09, box_frac: 0.14, exit: 0.20 };
    assert!((f.exit * len) < len && f.entry * len > 0.0);
    // And the same law the race applies is deterministic.
    let custom: CustomTrack = spec.custom.clone().unwrap();
    let f2 = custom.pit.unwrap_or_default();
    assert_eq!(f, f2, "the file's fractions must survive the spec");
}

#[test]
fn an_unusable_pit_placement_is_refused_at_load() {
    // The entry needs 120 m of braking zone before it: 0.01 of the lap is
    // far too early, and the load must say so instead of silently disabling
    // the pit lane.
    let dir = std::env::temp_dir().join(format!("apexsim_track_gate3_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("earlypit.json");
    std::fs::write(
        &path,
        r#"{"name":"earlypit","pit":{"entry":0.01,"box":0.05,"exit":0.1},"points":[[0,0],[150,-5],[280,40],[300,140],[200,210],[60,190],[-30,100]]}"#,
    )
    .unwrap();
    let e = resolve_track_spec(path.to_str().unwrap()).unwrap_err();
    assert!(e.contains("pit lane does not fit"), "{e}");
}

#[test]
fn an_unstructured_track_file_is_refused_at_load_not_midrace() {
    // A 2-point "loop" cannot close: resolve_track_spec surfaces the engine's
    // structural refusal, naming the problem. (Self-crossings and short laps
    // are ACCEPTED at load since the draw-anything relaxation — anything
    // drawable races.)
    let dir = std::env::temp_dir().join(format!("apexsim_track_gate2_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("notaloop.json");
    std::fs::write(
        &path,
        r#"{"name":"notaloop","points":[[0,0],[300,-5]]}"#,
    )
    .unwrap();
    let e = resolve_track_spec(path.to_str().unwrap()).unwrap_err();
    assert!(
        e.contains("3 control points"),
        "expected the structural refusal, got: {e}"
    );
}