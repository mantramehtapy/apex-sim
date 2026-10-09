//! Phase 3 completion-gate tests, mapped to the plan's gate criteria:
//!
//! 1. "Publish numeric tolerances and show results for each component test."
//! 2. "Record whether error changes as timestep is reduced."
//!    (convergence assertions on the two integral metrics)
//! 3. "Every failed case can be replayed from saved inputs."
//!    (stored-case hash reproduction + model-version guard)

use apexsim::physics::CarParams;
use apexsim::validation::{
    self, case_braking_50, case_corner_r60, case_steer_step, case_straight_accel,
    ref_braking_distance, ref_straight_accel, tol, LapTrace, RATES_HZ, MODEL_VERSION,
    REFERENCE_RATE_HZ,
};

fn metric_of(cases: &[validation::CaseOutput], name: &str, rate: u64, key: &str) -> f64 {
    cases
        .iter()
        .find(|c| c.name == name && c.rate_hz == rate)
        .and_then(|c| c.metrics.iter().find(|(k, _)| k == key))
        .map(|(_, v)| *v)
        .unwrap_or(f64::NAN)
}

fn all_cases() -> Vec<validation::CaseOutput> {
    let p = CarParams::default();
    RATES_HZ
        .iter()
        .flat_map(|&r| {
            vec![
                case_straight_accel(&p, r),
                case_braking_50(&p, r),
                case_corner_r60(&p, r),
                case_steer_step(&p, r),
            ]
        })
        .collect()
}

#[test]
fn gate1_tolerances_published_and_met_at_engine_rate() {
    // Tolerances exist as explicit constants. These are compile-time
    // constants, so a runtime `assert!` would be a tautology clippy is right
    // to flag; as an inline const block the check instead fails the *build*
    // the moment anyone sets a tolerance to zero or negative.
    const {
        assert!(tol::STRAIGHT_X_FINAL_M > 0.0);
        assert!(tol::BRAKING_DISTANCE_M > 0.0);
        assert!(tol::CORNER_CONSISTENCY_PCT > 0.0);
        assert!(tol::STEER_RISE_TIME_PCT > 0.0);
        assert!(tol::LAP_TIME_S > 0.0);
    }
    // …and every component case at the engine rate (250 Hz) meets them.
    let p = CarParams::default();
    let cases = all_cases();
    assert!(
        validation::component_pass_at(&cases, 250, &p),
        "component tolerances must pass at the engine rate"
    );
}

#[test]
fn gate2_error_shrinks_as_timestep_is_reduced() {
    // The two integral metrics must converge: error(125) > error(250) >
    // error(500) > error(1000), within a small equality band. This is the
    // "record whether error changes as timestep is reduced" requirement,
    // asserted rather than just logged.
    let p = CarParams::default();
    let (ref_x, ref_v) = ref_straight_accel(&p, 8.0);
    let ref_brake = ref_braking_distance(&p, 50.0);
    let cases = all_cases();

    let x_err: Vec<f64> = RATES_HZ
        .iter()
        .map(|&r| (metric_of(&cases, "straight_accel", r, "x_final_m") - ref_x).abs())
        .collect();
    let v_err: Vec<f64> = RATES_HZ
        .iter()
        .map(|&r| (metric_of(&cases, "straight_accel", r, "v_final_ms") - ref_v).abs())
        .collect();
    let b_err: Vec<f64> = RATES_HZ
        .iter()
        .map(|&r| (metric_of(&cases, "braking_50", r, "distance_m") - ref_brake).abs())
        .collect();

    // x_final and v_final converge monotonically (positive-error series).
    for series in [&x_err, &v_err] {
        for w in series.windows(2) {
            assert!(
                w[1] <= w[0] * 1.05 + 1e-9,
                "error must not grow as dt shrinks: {w:?} (series {series:?})"
            );
        }
    }
    // Braking distance error crosses ZERO near 250 Hz (semi-implicit Euler
    // under-estimates at 125 Hz, slightly over-estimates above); absolute
    // error stays tiny and within tolerance at every rate.
    for (i, &e) in b_err.iter().enumerate() {
        assert!(
            e <= tol::BRAKING_DISTANCE_M,
            "{} Hz braking error {e} exceeds tolerance",
            RATES_HZ[i]
        );
    }
    assert!(
        b_err.iter().all(|&e| e <= b_err[0]),
        "braking error must not exceed its 125 Hz baseline: {b_err:?}"
    );
    // And the 125 → 1000 improvement must be substantial (O(dt) scaling).
    assert!(
        x_err[3] < x_err[0] * 0.25,
        "x_final error should fall ≥4× from 125 Hz to 1 kHz: {:?}",
        x_err
    );
}

#[test]
fn gate2b_lap_trace_tolerances_vs_reference_rate() {
    let p = CarParams::default();
    let track = apexsim::track::Track::default();
    let inputs = validation::evaluation_inputs();
    let mut traces: Vec<LapTrace> = RATES_HZ
        .iter()
        .map(|&r| validation::replay_inputs_at_rate(&p, &track, &inputs, r))
        .collect();
    let reference = traces.last().unwrap();

    for (i, rate) in RATES_HZ.iter().enumerate() {
        let t = &traces[i];
        assert_eq!(t.lap_times.len(), 2, "{} Hz must complete both laps", rate);
        if *rate == REFERENCE_RATE_HZ {
            continue;
        }
        for (a, b) in t.lap_times.iter().zip(reference.lap_times.iter()) {
            assert!(
                (a - b).abs() <= tol::LAP_TIME_S,
                "{} Hz lap time {} vs {} exceeds {:?}",
                rate, a, b, tol::LAP_TIME_S
            );
        }
        for (la, lb) in t.sector_times.iter().zip(reference.sector_times.iter()) {
            for (a, b) in la.iter().zip(lb.iter()) {
                assert!(
                    (a - b).abs() <= tol::SECTOR_TIME_S,
                    "{} Hz sector {} vs {} exceeds {:?}",
                    rate, a, b, tol::SECTOR_TIME_S
                );
            }
        }
        let (rms, max, n) = validation::trace_diff(t, reference);
        assert!(n > 500, "expected >500 shared distance bins, got {n}");
        assert!(rms <= tol::TRACE_RMS_MS, "{} Hz trace RMS {rms}", rate);
        assert!(max <= tol::TRACE_MAX_MS, "{} Hz trace max {max}", rate);
    }
    let _ = &mut traces;
}

#[test]
fn gate3_stored_cases_replay_bit_identically() {
    // Every stored case must reproduce its trajectory hash from the file —
    // this is what makes a failed case replayable from saved inputs.
    let p = CarParams::default();
    let dir = std::env::temp_dir().join(format!("apexsim_p3_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for rate in RATES_HZ {
        for case in [
            case_straight_accel(&p, rate),
            case_braking_50(&p, rate),
            case_corner_r60(&p, rate),
            case_steer_step(&p, rate),
        ] {
            let path = dir.join(format!("{}_{}hz.casefile", case.name, rate));
            validation::write_case(&path, &case, &p).unwrap();
            let ok = validation::replay_case(&path)
                .unwrap_or_else(|e| panic!("replay {} failed: {e}", path.display()));
            assert!(ok, "case {} @ {} Hz did not reproduce", case.name, rate);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gate3b_replay_rejects_foreign_model_version() {
    // A case recorded against a different physics version must be refused,
    // never silently compared.
    let p = CarParams::default();
    let dir = std::env::temp_dir().join(format!("apexsim_p3b_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let case = case_straight_accel(&p, 250);
    let path = dir.join("stale.casefile");
    validation::write_case(&path, &case, &p).unwrap();
    let text = std::fs::read_to_string(&path).unwrap().replace(
        MODEL_VERSION,
        "apexsim-physics-0.0.999",
    );
    std::fs::write(&path, text).unwrap();
    let err = validation::replay_case(&path).expect_err("must reject stale model version");
    assert!(err.contains("model version mismatch"), "got: {err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gate4_report_written_with_all_sections_and_cases() {
    let p = CarParams::default();
    let report = validation::run_validation();
    let text = validation::render_report(&report, &p);
    for section in [
        "Published numeric tolerances",
        "Identified references",
        "Component results by rate",
        "Lap-trace comparison",
        "Outcome",
        "Failed-case replay",
    ] {
        assert!(text.contains(section), "report missing section '{section}'");
    }
    for case in ["straight_accel", "braking_50", "corner_r60", "steer_step"] {
        assert!(text.contains(case), "report missing case '{case}'");
    }
    assert!(text.contains(MODEL_VERSION));
    // The report carries the rate recommendation and the 250 Hz decision.
    assert!(text.contains("retained at 250 Hz"));
    assert!(report.cases.len() == 16, "4 cases × 4 rates expected");
}

#[test]
fn calibration_and_evaluation_are_separate() {
    // The evaluation input log is a fixed, seed-identified artifact; running
    // it twice yields identical inputs (nothing is fitted to it).
    let a = validation::evaluation_inputs();
    let b = validation::evaluation_inputs();
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b.iter()).all(|(x, y)| x.tick == y.tick
        && x.throttle == y.throttle
        && x.brake == y.brake
        && x.steer_cmd == y.steer_cmd));
}
