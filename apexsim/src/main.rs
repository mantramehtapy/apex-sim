//! apexsim CLI — Phase 1 gate runner.
//!
//!   apexsim run     [--laps N] [--seed S] [--noise A] [--realtime] [--log PATH] [--quiet]
//!   apexsim replay  <log-path>
//!
//! `run` executes the race on a dedicated simulation thread and prints the
//! Phase 1 gate report (lap times, tick-cost percentiles, deadline misses,
//! memory, trajectory hash). `replay` re-executes a saved input log and
//! verifies the trajectory hash matches, exiting non-zero on divergence.

use apexsim::replay::{read_log, write_log};
use apexsim::sim::{replay_session, run_session, Event, SimConfig, SimOutcome, DT};
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::sync::mpsc;
use std::thread;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => match cmd_run(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("serve") => match cmd_serve(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("dev") => match cmd_dev(&args[1..]) {
            Ok(0) => ExitCode::SUCCESS,
            Ok(code) => ExitCode::from(code as u8),
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("weekend") => match cmd_weekend(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("race") => match cmd_race(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("validate") => match cmd_validate(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("aibench") => match cmd_aibench(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("weatherbench") => match cmd_weatherbench(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("tournament") => match cmd_tournament(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("steward") => match cmd_steward(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("policy-eval") => match cmd_policy_eval(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("policy-latency") => match cmd_policy_latency(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        Some("replay") => match cmd_replay(&args[1..]) {
            Ok(passed) => {
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(2)
            }
        },
        _ => {
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn print_usage() {
    eprintln!("usage: apexsim run [--laps N] [--seed S] [--noise A] [--realtime] [--log PATH] [--quiet]");
    eprintln!("       apexsim dev [--config FILE] [--init] [--duration SECS] [--port N] [--cars N]");
    eprintln!("                  [--laps N] [--seed S] [--mode M] [--noise A] [--no-autostart]");
    eprintln!("       apexsim serve [--port N] [--laps N] [--seed S] [--cars N] [--mode M] [--noise A]");
    eprintln!("                    [--config FILE] [--structured] [--duration SECS] [--no-autostart]");
    eprintln!("       apexsim race [--cars N] [--laps N] [--seed S] [--mode race|qualifying|practice]");
    eprintln!("                   [--compound soft|medium|hard] [--no-pits] [--stop-lap N [--switch-to C]]");
    eprintln!("                   [--realtime] [--log PATH] | --scale-bench");
    eprintln!("       apexsim race ... --ai [--ai-model M] [--ai-endpoint URL] [--ai-timeout MS]");
    eprintln!("                   [--ai-max-tokens N] [--ai-every-laps N] [--ai-min-confidence C] [--ai-log PATH]");
    eprintln!("       apexsim aibench [--model M] [--endpoint URL] [--seeds 1,2,3] [--laps N] [--cars N]");
    eprintln!("                    [--timeout MS] [--max-tokens N] [--every-laps N] [--min-confidence C]");
    eprintln!("                    [--ai-log PATH] [--quiet]");
    eprintln!("       apexsim race ... --weather dry|damp|wet|mixed [--water W] [--track-temp C] [--grip G]");
    eprintln!("                    [--overrides 'TICK:SLIDER:VALUE', ...] [--driver-error RATE]");
    eprintln!("       apexsim weatherbench [--seeds 1,2,3] [--laps N] [--cars N] [--speed KMH]");
    eprintln!("       apexsim race/serve ... --telemetry [HOST:PORT] [--telemetry-hz N] [--telemetry-player N]");
    eprintln!("                             [--telemetry-uid N]");
    eprintln!("       apexsim race/serve ... --sound [--sound-max-cues N] [--slip-threshold G]");
    eprintln!("                             [--wind-band MPS] [--wind-start MPS]");
    eprintln!("       apexsim validate [--out DIR] | --replay-case FILE");
    eprintln!("       apexsim tournament [--spec FILE] [--out DIR] [--seeds 1,2,3] [--policies soft,no-stop] [--steward]");
    eprintln!("                       [--cars N] [--laps N] [--mode M] [--weather W | --no-weather]");
    eprintln!("                       [--profile engine-only|engine+dashboard|engine+ollama] [--port N]");
    eprintln!("                       [--threads N] [--measure N] [--logs] [--quiet]");
    eprintln!("       apexsim tournament --write-spec FILE | --replay DIR");
    eprintln!("       apexsim race ... --steward [--steward-verbose]");
    eprintln!("       apexsim steward <race-log> [--verbose]");
    eprintln!("       apexsim replay <log-path>");
    eprintln!("       apexsim weekend --cars 20 --laps 12 --seed 42 [--track NAME|FILE]
                      [--quali-s 240] [--no-quali]
       apexsim policy-eval --model PATH --seeds 9001,9002 [--laps N] [--cars N] [--json]");
    eprintln!("       apexsim policy-latency --model PATH [--cars N] [--runs N] [--json]");
}

/// Phase 12: `policy-eval` — race the baseline and the learned driver on the
/// same seeds and apply the enable rule.
fn cmd_policy_eval(args: &[String]) -> Result<bool, String> {
    use apexsim::gym::{self, EnvSpec};
    let mut model: Option<PathBuf> = None;
    let mut seeds: Vec<u64> = Vec::new();
    let mut laps: u32 = 4;
    let mut cars: u32 = 20;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        let need = |name: &str| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match args[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(need("--model")?));
                i += 2;
            }
            "--seeds" => {
                seeds = need("--seeds")?
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().parse::<u64>().map_err(|e| format!("--seeds: {e}")))
                    .collect::<Result<Vec<u64>, String>>()?;
                i += 2;
            }
            "--laps" => {
                laps = need("--laps")?.parse().map_err(|e| format!("--laps: {e}"))?;
                i += 2;
            }
            "--cars" => {
                cars = need("--cars")?.parse().map_err(|e| format!("--cars: {e}"))?;
                i += 2;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            other => return Err(format!("policy-eval: unknown flag {other}")),
        }
    }
    if seeds.is_empty() {
        // Seeds are required rather than defaulted: this command exists to
        // evaluate on the evaluation split, and a default that silently picked
        // training seeds would make the whole gate meaningless.
        return Err("policy-eval: --seeds is required (list the evaluation split)".into());
    }
    let spec = EnvSpec {
        laps,
        cars,
        ..EnvSpec::default()
    };

    let cmp = gym::compare(&spec, &seeds, model.as_deref())?;

    if json {
        println!("{}", serde_json::to_string_pretty(&cmp).map_err(|e| e.to_string())?);
    } else {
        println!("policy-eval: {} cars, {} laps, seeds {:?}", cmp.cars, cmp.laps, cmp.seeds);
        println!("  baseline: best lap {:.3}s  laps {:.2}  position {:.2}  progress {:.0}m  finished {}/{}",
            cmp.baseline.mean_best_lap_s, cmp.baseline.mean_laps, cmp.baseline.mean_position,
            cmp.baseline.mean_progress_m, cmp.baseline.finished, cmp.baseline.episodes);
        println!("  learned : best lap {:.3}s  laps {:.2}  position {:.2}  progress {:.0}m  finished {}/{}",
            cmp.learned.mean_best_lap_s, cmp.learned.mean_laps, cmp.learned.mean_position,
            cmp.learned.mean_progress_m, cmp.learned.finished, cmp.learned.episodes);
        println!("  delta   : best lap {:+.3}s (negative is faster)  laps {:+.2}  position {:+.2}  progress {:+.0}m",
            cmp.deltas.best_lap_s, cmp.deltas.laps, cmp.deltas.position, cmp.deltas.progress_m);
        println!("  metrics : {}", cmp.declared_metrics.join(", "));
        if cmp.improves {
            println!("  verdict : learned driver IMPROVES the declared metrics");
        } else {
            println!("  verdict : learned driver does NOT improve; not enabled:");
            for r in &cmp.reasons {
                println!("             - {r}");
            }
        }
    }
    Ok(cmp.improves)
}

/// Phase 12: `policy-latency` — end-to-end p99 over the intended grid.
fn cmd_policy_latency(args: &[String]) -> Result<bool, String> {
    use apexsim::gym::{self, EnvSpec};
    let mut model: Option<PathBuf> = None;
    let mut cars: u32 = 20;
    let mut runs: u64 = 2_000;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        let need = |name: &str| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match args[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(need("--model")?));
                i += 2;
            }
            "--cars" => {
                cars = need("--cars")?.parse().map_err(|e| format!("--cars: {e}"))?;
                i += 2;
            }
            "--runs" => {
                runs = need("--runs")?.parse().map_err(|e| format!("--runs: {e}"))?;
                i += 2;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            other => return Err(format!("policy-latency: unknown flag {other}")),
        }
    }
    let Some(model) = model else {
        return Err("policy-latency: --model is required".into());
    };
    let spec = EnvSpec::default();
    let report = gym::benchmark_latency(
        &model,
        apexsim::policy::OBS_DIM,
        cars,
        runs,
        spec.action_every_ticks,
    )?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    } else {
        println!(
            "policy-latency: {} ({} nodes), {} runs at the {}-car grid",
            report.model, report.op_nodes, report.runs, report.cars
        );
        println!(
            "  p50 {:.1} us   p99 {:.1} us   max {:.1} us   mean {:.1} us",
            report.p50_us, report.p99_us, report.max_us, report.mean_us
        );
        println!(
            "  budget {:.0} us per action ({:.0} ms): p99 is {:.1}% of it -> {}",
            report.budget_us,
            report.budget_us / 1000.0,
            100.0 * report.p99_us / report.budget_us,
            if report.within_budget { "WITHIN BUDGET" } else { "OVER BUDGET" }
        );
    }
    Ok(report.within_budget)
}

/// Phase E: the race weekend — one-by-one timed qualifying, then the race on
/// the grid it produced. Each qualifier is a real solo session (formation
/// start, the car's own machinery, form and noise seed) run for a fixed
/// window; lap times set the grid, fastest at pole.
fn cmd_weekend(args: &[String]) -> Result<bool, String> {
    use apexsim::race::{FieldMode, RealismConfig};
    let mut cfg = apexsim::race::RaceConfig {
        field: FieldMode::Teams,
        realism: Some(RealismConfig::default()),
        ..Default::default()
    };
    let mut quali_s = 240.0f64;
    let mut do_quali = true;
    let mut log: Option<std::path::PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |name: &str| -> Result<String, String> {
            args.get(i + 1).cloned().ok_or_else(|| format!("{name} requires a value"))
        };
        match args[i].as_str() {
            "--cars" => {
                let v: u32 = need("--cars")?.parse().map_err(|e| format!("--cars: {e}"))?;
                if !(1..=30).contains(&v) {
                    return Err("--cars must be 1..=30".into());
                }
                cfg.cars = v;
                i += 2;
            }
            "--laps" => {
                cfg.laps = need("--laps")?.parse().map_err(|e| format!("--laps: {e}"))?;
                i += 2;
            }
            "--seed" => {
                cfg.seed = need("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?;
                i += 2;
            }
            "--noise" => {
                cfg.noise = need("--noise")?.parse().map_err(|e| format!("--noise: {e}"))?;
                i += 2;
            }
            "--track" => {
                cfg.track = apexsim::track::resolve_track_spec(&need("--track")?)?;
                i += 2;
            }
            "--field" => {
                cfg.field = FieldMode::parse(&need("--field")?)
                    .ok_or_else(|| "--field: teams | classic".to_string())?;
                i += 2;
            }
            "--quali-s" => {
                quali_s = need("--quali-s")?.parse().map_err(|e| format!("--quali-s: {e}"))?;
                if !(10.0..=3600.0).contains(&quali_s) {
                    return Err("--quali-s must be 10..=3600 sim seconds per car".into());
                }
                i += 2;
            }
            "--error-rate" => {
                let v: f64 = args.get(i + 1).cloned().ok_or("--error-rate requires a value")?.parse().map_err(|e| format!("--error-rate: {e}"))?;
                let r = cfg.realism.get_or_insert_with(apexsim::race::RealismConfig::default);
                r.error_rate = v;
                if v <= 0.0 { r.spin_crashes = false; }
                i += 2;
            }
            "--mechanical-rate" => {
                let v: f64 = args.get(i + 1).cloned().ok_or("--mechanical-rate requires a value")?.parse().map_err(|e| format!("--mechanical-rate: {e}"))?;
                cfg.realism.get_or_insert_with(apexsim::race::RealismConfig::default).mechanical_rate = v;
                i += 2;
            }
            "--no-quali" => {
                do_quali = false;
                i += 1;
            }
            "--log" => {
                log = Some(std::path::PathBuf::from(need("--log")?));
                i += 2;
            }
            other => return Err(format!("weekend: unknown flag {other} (the race legs' own options live on `race`; this command takes cars/laps/seed/noise/track/field/quali-s/no-quali/log)")),
        }
    }
    if do_quali {
        // Qualifying runs the race's formation; laps must give the window
        // room to matter on any circuit.
        cfg.laps = cfg.laps.max(2);
        println!(
            "== apexsim weekend | qualifying: {} solo runs x {:.0} s | {} cars | {} laps | seed {} | {} ==",
            cfg.cars, quali_s, cfg.cars, cfg.laps, cfg.seed,
            if cfg.field == FieldMode::Teams { "2026 teams" } else { "classic" }
        );
        let t0 = std::time::Instant::now();
        let rows = apexsim::weekend::run_qualifying(&cfg, quali_s);
        println!("  {:>4}  {:>9}  {:>9}  {:>5}", "car", "best lap", "top spd", "laps");
        for r in &rows {
            let lap = r.best_lap_s.map(|t| format!("{t:.3} s")).unwrap_or_else(|| "--".into());
            println!(
                "  {:>4}  {:>9}  {:>6.0} km/h  {:>5}",
                r.car,
                lap,
                r.top_speed_mps * 3.6,
                r.laps
            );
        }
        let grid = apexsim::weekend::grid_from_rows(&rows);
        let pole = rows.iter().find(|r| r.car == grid[0]);
        if let Some(p) = pole {
            if let Some(t) = p.best_lap_s {
                println!("POLE: car {} — {:.3} s (qualifying ran {:.1} s wall)
", grid[0], t, t0.elapsed().as_secs_f64());
            }
        }
        cfg.grid_order = grid;
    } else {
        println!(
            "== apexsim weekend | qualifying skipped | {} cars | {} laps | seed {} ==",
            cfg.cars, cfg.laps, cfg.seed
        );
    }

    // The race leg: same session, now carrying the qualifying order. The
    // log records the grid as an O line, so the weekend's race replays
    // bit-identically without re-running qualifying.
    let (outcome, inputs, _exchanges) = apexsim::race::run_race_full(cfg.clone(), false);
    println!(
        "
== race | {} cars | {} laps | grid by qualifying ==",
        cfg.cars, cfg.laps
    );
    println!("classification:");
    for row in &outcome.classification {
        let gap = row
            .gap_s
            .map(|g| format!("{g:+.1}"))
            .unwrap_or_else(|| if row.pos == 1 { "leader".into() } else { "—".into() });
        let status = if row.retired {
            match row.dnf {
                Some(why) => format!("DNF {why}"),
                None => "DNF".into(),
            }
        } else if row.finished {
            "FIN".into()
        } else {
            format!("lap {}", row.laps_done)
        };
        let best = row.best_lap_s.map(|b| format!(" {b:.2}")).unwrap_or_default();
        let stops = if row.stops > 0 { format!(" {}p", row.stops) } else { String::new() };
        println!(
            "  P{:>2} car {:>2}  {:>8}  gap {:>8} s  {}{} {}{stops}",
            row.pos, row.id, status, gap, row.compound, best,
            format_args!("{:.0} km/h", row.v * 3.6),
        );
    }
    println!("state hash: {:016x}", outcome.state_hash);
    if let Some(path) = log {
        apexsim::race::write_race_log(&path, &cfg, &inputs, &outcome)
            .map_err(|e| format!("write log: {e}"))?;
        println!("log: {}", path.display());
    }
    Ok(true)
}

fn cmd_race(args: &[String]) -> Result<bool, String> {
    use apexsim::race::{FieldMode, PitStrategy, RealismConfig, SessionMode, TireCompound};
    // A race from the CLI is the 2026 championship with the realism pack on
    // unless asked otherwise; --field classic / --realism off step back.
    let mut cfg = apexsim::race::RaceConfig {
        field: FieldMode::Teams,
        realism: Some(RealismConfig::default()),
        ..Default::default()
    };
    let mut realtime = false;
    let mut log: Option<std::path::PathBuf> = None;
    let mut scale_bench = false;
    let mut probe_ticks: Option<u64> = None;
    let mut stop_lap: Option<u32> = None;
    let mut switch_to: Option<TireCompound> = None;
    // Phase 11: turn the stewarding aid on for this race. Off by default, and
    // provably inert when off — a race without it hashes identically.
    let mut steward_on = false;
    let mut steward_verbose = false;
    // Phase 7 advisor flags. Any --ai-* flag implies --ai, so `--ai-model x`
    // alone is a complete request.
    let mut ai_on = false;
    let mut ai_model: Option<String> = None;
    let mut ai_endpoint: Option<String> = None;
    let mut ai_timeout_ms: Option<u64> = None;
    let mut ai_max_tokens: Option<u32> = None;
    let mut ai_every_laps: Option<u32> = None;
    let mut ai_min_conf: Option<f64> = None;
    let mut ai_log: Option<std::path::PathBuf> = None;
    // Phase 8 weather flags. Any --weather* flag implies weather on, so
    // `--water 1.0` alone is a complete request.
    let mut weather_on = false;
    let mut w_preset: Option<String> = None;
    let mut w_water: Option<f64> = None;
    let mut w_temp: Option<f64> = None;
    let mut w_grip: Option<f64> = None;
    let mut w_overrides: Vec<String> = Vec::new();
    let mut w_errors: Option<f64> = None;
    // Phase 9 flags. Any --telemetry*/--sound* flag implies the subsystem is
    // on, so `--telemetry-hz 60` alone is a complete request and
    // `--sound-max-cues 10` alone is too.
    let mut tel_target: Option<String> = None;
    let mut tel_hz: Option<u32> = None;
    let mut tel_player: Option<u32> = None;
    let mut tel_uid: Option<u64> = None;
    let mut sound_on = false;
    let mut s_max_cues: Option<u32> = None;
    let mut s_slip_g: Option<f64> = None;
    let mut s_wind_band: Option<f64> = None;
    let mut s_wind_start: Option<f64> = None;
    let mut i = 0;
    while i < args.len() {
        let need = |name: &str| -> Result<String, String> {
            args.get(i + 1).cloned().ok_or_else(|| format!("{name} requires a value"))
        };
        match args[i].as_str() {
            "--cars" => {
                let v: u32 = need("--cars")?.parse().map_err(|e| format!("--cars: {e}"))?;
                if !(1..=30).contains(&v) {
                    return Err("--cars must be 1..=30 (pick the intended season's grid; do not mix assumptions)".into());
                }
                cfg.cars = v;
                i += 2;
            }
            "--laps" => {
                cfg.laps = need("--laps")?.parse().map_err(|e| format!("--laps: {e}"))?;
                i += 2;
            }
            "--seed" => {
                cfg.seed = need("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?;
                i += 2;
            }
            "--mode" => {
                let v = need("--mode")?;
                cfg.mode = SessionMode::parse(&v)
                    .ok_or_else(|| format!("--mode must be race | qualifying | practice, got '{v}'"))?;
                i += 2;
            }
            "--field" => {
                let v = need("--field")?;
                cfg.field = FieldMode::parse(&v)
                    .ok_or_else(|| format!("--field must be teams | classic, got '{v}'"))?;
                i += 2;
            }
            "--track" => {
                let v = need("--track")?;
                cfg.track = apexsim::track::resolve_track_spec(&v)?;
                i += 2;
            }
            "--realism" => {
                let v = need("--realism")?;
                cfg.realism = match v.as_str() {
                    "on" | "full" => Some(RealismConfig::default()),
                    "off" | "none" => None,
                    other => return Err(format!("--realism must be on | off, got '{other}'")),
                };
                i += 2;
            }
            "--error-rate" => {
                let v: f64 = need("--error-rate")?.parse().map_err(|e| format!("--error-rate: {e}"))?;
                let r = cfg.realism.get_or_insert_with(RealismConfig::default);
                r.error_rate = v;
                if v <= 0.0 {
                    r.spin_crashes = false;
                }
                i += 2;
            }
            "--mechanical-rate" => {
                let v: f64 =
                    need("--mechanical-rate")?.parse().map_err(|e| format!("--mechanical-rate: {e}"))?;
                cfg.realism.get_or_insert_with(RealismConfig::default).mechanical_rate = v;
                i += 2;
            }
            "--compound" => {
                let v = need("--compound")?;
                cfg.start_compound = TireCompound::parse(&v)
                    .ok_or_else(|| format!("--compound must be soft | medium | hard, got '{v}'"))?;
                i += 2;
            }
            "--no-pits" => {
                cfg.strategy = PitStrategy::None;
                i += 1;
            }
            "--stop-lap" => {
                stop_lap = Some(need("--stop-lap")?.parse().map_err(|e| format!("--stop-lap: {e}"))?);
                i += 2;
            }
            "--switch-to" => {
                let v = need("--switch-to")?;
                switch_to = Some(TireCompound::parse(&v)
                    .ok_or_else(|| format!("--switch-to must be soft | medium | hard, got '{v}'"))?);
                i += 2;
            }
            "--realtime" => {
                realtime = true;
                i += 1;
            }
            "--steward" => {
                steward_on = true;
                i += 1;
            }
            "--steward-verbose" => {
                // Implies --steward, so `--steward-verbose` alone is a
                // complete request rather than a silently ignored flag.
                steward_on = true;
                steward_verbose = true;
                i += 1;
            }
            "--ai" => {
                ai_on = true;
                i += 1;
            }
            "--ai-model" => {
                ai_model = Some(need("--ai-model")?);
                ai_on = true;
                i += 2;
            }
            "--ai-endpoint" => {
                ai_endpoint = Some(need("--ai-endpoint")?);
                ai_on = true;
                i += 2;
            }
            "--ai-timeout" => {
                ai_timeout_ms = Some(need("--ai-timeout")?.parse().map_err(|e| format!("--ai-timeout: {e}"))?);
                ai_on = true;
                i += 2;
            }
            "--ai-max-tokens" => {
                ai_max_tokens = Some(need("--ai-max-tokens")?.parse().map_err(|e| format!("--ai-max-tokens: {e}"))?);
                ai_on = true;
                i += 2;
            }
            "--ai-every-laps" => {
                ai_every_laps = Some(need("--ai-every-laps")?.parse().map_err(|e| format!("--ai-every-laps: {e}"))?);
                ai_on = true;
                i += 2;
            }
            "--ai-min-confidence" => {
                ai_min_conf = Some(need("--ai-min-confidence")?.parse().map_err(|e| format!("--ai-min-confidence: {e}"))?);
                ai_on = true;
                i += 2;
            }
            "--ai-log" => {
                ai_log = Some(std::path::PathBuf::from(need("--ai-log")?));
                ai_on = true;
                i += 2;
            }
            "--log" => {
                log = Some(std::path::PathBuf::from(need("--log")?));
                i += 2;
            }
            "--weather" => {
                w_preset = Some(need("--weather")?);
                weather_on = true;
                i += 2;
            }
            "--water" => {
                w_water = Some(need("--water")?.parse().map_err(|e| format!("--water: {e}"))?);
                weather_on = true;
                i += 2;
            }
            "--track-temp" => {
                w_temp = Some(need("--track-temp")?.parse().map_err(|e| format!("--track-temp: {e}"))?);
                weather_on = true;
                i += 2;
            }
            "--grip" => {
                w_grip = Some(need("--grip")?.parse().map_err(|e| format!("--grip: {e}"))?);
                weather_on = true;
                i += 2;
            }
            "--overrides" => {
                // Comma-separated, so a whole director's schedule is one flag.
                w_overrides.extend(
                    need("--overrides")?
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty()),
                );
                weather_on = true;
                i += 2;
            }
            "--driver-error" => {
                w_errors = Some(need("--driver-error")?.parse().map_err(|e| format!("--driver-error: {e}"))?);
                weather_on = true;
                i += 2;
            }
            // ---- Phase 9: telemetry over the wire, sound as an observer ----
            "--telemetry" => {
                tel_target = Some(need("--telemetry")?);
                i += 2;
            }
            "--telemetry-hz" => {
                tel_hz = Some(need("--telemetry-hz")?.parse().map_err(|e| format!("--telemetry-hz: {e}"))?);
                i += 2;
            }
            "--telemetry-player" => {
                tel_player = Some(need("--telemetry-player")?.parse().map_err(|e| format!("--telemetry-player: {e}"))?);
                i += 2;
            }
            "--telemetry-uid" => {
                tel_uid = Some(need("--telemetry-uid")?.parse().map_err(|e| format!("--telemetry-uid: {e}"))?);
                i += 2;
            }
            "--sound" => {
                sound_on = true;
                i += 1;
            }
            "--sound-max-cues" => {
                s_max_cues = Some(need("--sound-max-cues")?.parse().map_err(|e| format!("--sound-max-cues: {e}"))?);
                sound_on = true;
                i += 2;
            }
            "--slip-threshold" => {
                s_slip_g = Some(need("--slip-threshold")?.parse().map_err(|e| format!("--slip-threshold: {e}"))?);
                sound_on = true;
                i += 2;
            }
            "--wind-band" => {
                s_wind_band = Some(need("--wind-band")?.parse().map_err(|e| format!("--wind-band: {e}"))?);
                sound_on = true;
                i += 2;
            }
            "--wind-start" => {
                s_wind_start = Some(need("--wind-start")?.parse().map_err(|e| format!("--wind-start: {e}"))?);
                sound_on = true;
                i += 2;
            }
            "--scale-bench" => {
                scale_bench = true;
                i += 1;
            }
            "--probe" => {
                probe_ticks = Some(need("--probe")?.parse().map_err(|e| format!("--probe: {e}"))?);
                i += 2;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    if stop_lap.is_some() || switch_to.is_some() {
        cfg.strategy = PitStrategy::Fixed {
            lap: stop_lap.unwrap_or(1),
            switch_to: switch_to.unwrap_or(TireCompound::Medium),
        };
    }

    // Build the advisor config. Two hard guards, because an advisor run is
    // not reproducible: it can't be replayed from a log (model latency
    // feeds back into the controls), and it only makes sense on a real
    // race session with laps to strategise over.
    let advisor = if ai_on {
        let mut a = apexsim::strategy::AdvisorConfig::default();
        if let Some(m) = ai_model {
            a.model = m;
        }
        if let Some(e) = ai_endpoint {
            a.endpoint = e;
        }
        if let Some(t) = ai_timeout_ms {
            a.timeout_ms = t;
        }
        if let Some(t) = ai_max_tokens {
            a.max_tokens = t;
        }
        if let Some(n) = ai_every_laps {
            if n == 0 {
                return Err("--ai-every-laps must be >= 1".into());
            }
            a.every_laps = n;
        }
        if let Some(c) = ai_min_conf {
            if !(0.0..=1.0).contains(&c) {
                return Err("--ai-min-confidence must be 0.0..=1.0".into());
            }
            a.min_confidence = c;
        }
        if cfg.mode != SessionMode::Race {
            return Err("--ai needs --mode race (timed sessions have no strategy)".into());
        }
        if cfg.laps < 3 {
            return Err(format!(
                "--ai needs at least 3 laps to strategise over, got {}",
                cfg.laps
            ));
        }
        cfg.ai = Some(a.clone());
        Some(a)
    } else {
        None
    };

    // Weather config. Sliders are clamped to their legal range here rather
    // than silently accepted — a race director's box has limits, and a value
    // outside them is a mistake worth naming.
    if weather_on {
        use apexsim::weather::{Override, Slider, WeatherConfig};
        let mut w = match &w_preset {
            Some(name) => WeatherConfig::preset(name)
                .ok_or_else(|| format!("--weather must be dry | damp | wet | mixed, got '{name}'"))?,
            None => WeatherConfig::default(),
        };
        if let Some(v) = w_water {
            let c = Slider::Water.clamp(v);
            if (c - v).abs() > 1e-12 {
                eprintln!("note: --water {v} clamped to {c} (slider range 0.0..=1.0)");
            }
            w.water = c;
        }
        if let Some(v) = w_temp {
            let c = Slider::TrackTemp.clamp(v);
            if (c - v).abs() > 1e-12 {
                eprintln!("note: --track-temp {v} clamped to {c} °C (slider range -10..=70)");
            }
            w.track_temp_c = c;
        }
        if let Some(v) = w_grip {
            let c = Slider::Grip.clamp(v);
            if (c - v).abs() > 1e-12 {
                eprintln!("note: --grip {v} clamped to {c} (slider range 0.5..=1.2)");
            }
            w.grip_scale = c;
        }
        for spec in &w_overrides {
            // "tick:slider:value", e.g. "2500:water:0.4" — seconds, or a raw
            // tick with a leading '@'. Seconds are what a race engineer means.
            let parts: Vec<&str> = spec.split(':').collect();
            if parts.len() != 3 {
                return Err(format!(
                    "--overrides '{spec}': expected tick:slider:value (e.g. 30:water:0.4)"
                ));
            }
            let (tick, slider) = if let Some(t) = parts[0].strip_prefix('@') {
                (t.parse::<u64>().map_err(|e| format!("--overrides tick: {e}"))?, parts[1].to_string())
            } else {
                let secs: f64 = parts[0]
                    .parse()
                    .map_err(|e| format!("--overrides '{spec}': seconds or @tick: {e}"))?;
                if secs < 0.0 {
                    return Err(format!("--overrides '{spec}': time cannot be negative"));
                }
                ((secs / apexsim::race::DT).round() as u64, parts[1].to_string())
            };
            let sl = Slider::parse(&slider).ok_or_else(|| {
                format!("--overrides '{spec}': slider must be water | temp | grip")
            })?;
            let val: f64 = parts[2]
                .parse()
                .map_err(|e| format!("--overrides '{spec}': value: {e}"))?;
            w.schedule.push(Override::new(tick, sl, val));
        }
        if let Some(rate) = w_errors {
            if rate < 0.0 {
                return Err("--driver-error cannot be negative".into());
            }
            w.driver_errors.rate_per_lap = rate;
        }
        w.normalise();
        for o in &w.schedule {
            if o.tick > (cfg.laps as u64 + 1) * 250 * 60 {
                eprintln!(
                    "note: override at tick {} is past the end of a {}-lap race and will never fire",
                    o.tick, cfg.laps
                );
            }
        }
        cfg.weather = Some(w);
    }

    // ---- Phase 9 wiring -------------------------------------------------
    // Telemetry is opt-in on *any* telemetry flag; `--telemetry` alone uses
    // the documented default target. An explicit `--telemetry ""` is a
    // deliberate way to say "no wire", and normalises to None.
    if tel_target.is_some() || tel_hz.is_some() || tel_player.is_some() || tel_uid.is_some() {
        use apexsim::telemetry::TelemetryConfig;
        let mut t = TelemetryConfig::default();
        if let Some(target) = &tel_target {
            t.target = TelemetryConfig::parse_target(target)?;
        }
        if let Some(hz) = tel_hz {
            if !(1..=250).contains(&hz) {
                return Err(format!("--telemetry-hz must be 1..=250 (the engine runs at 250 Hz), got {hz}"));
            }
            t.hz = hz;
        }
        if let Some(p) = tel_player {
            if p >= cfg.cars {
                return Err(format!(
                    "--telemetry-player {p} is not on a {}-car grid (valid 0..={})",
                    cfg.cars,
                    cfg.cars - 1
                ));
            }
            t.player_car = p;
        }
        if let Some(u) = tel_uid {
            if u == 0 {
                return Err("--telemetry-uid 0 is reserved as 'no session'".into());
            }
            t.session_uid = u;
        }
        cfg.telemetry = Some(t);
    }

    if sound_on {
        use apexsim::sound::SoundConfig;
        let mut s = SoundConfig::default();
        if let Some(n) = s_max_cues {
            s.max_cues_per_s = n;
        }
        if let Some(g) = s_slip_g {
            if !(0.1..=6.0).contains(&g) {
                return Err(format!("--slip-threshold must be 0.1..=6.0 g, got {g}"));
            }
            s.slip_threshold_g = g;
        }
        if let Some(b) = s_wind_band {
            if !(0.0..=200.0).contains(&b) {
                return Err(format!("--wind-band must be 0.0..=200 m/s, got {b}"));
            }
            s.wind_band_mps = b;
        }
        if let Some(v) = s_wind_start {
            if !(0.0..=400.0).contains(&v) {
                return Err(format!("--wind-start must be 0.0..=400 m/s, got {v}"));
            }
            s.wind_start_mps = v;
        }
        cfg.sound = Some(s);
    }

    if steward_on {
        cfg.steward = Some(apexsim::steward::StewardConfig::enabled());
    }

    if let Some(ticks) = probe_ticks {
        // Diagnostic: run N ticks from the grid and dump the field state.
        // Pure traffic diagnostics — no formation sequence, no stops.
        let mut pcfg = cfg.clone();
        pcfg.formation = false;
        pcfg.strategy = apexsim::race::PitStrategy::None;
        let mut race = apexsim::race::Race::new(pcfg);
        race.state.phase = apexsim::race::RacePhase::Running;
        for _ in 0..ticks {
            race.tick();
        }
        println!("== probe @ {} ticks (t = {:.1} s) ==", ticks, race.state.sim_time);
        println!("contacts: {}", race.state.contacts.len());
        let mut order: Vec<&apexsim::race::CarEntry> = race.state.cars.iter().collect();
        order.sort_by(|a, b| b.total_progress.partial_cmp(&a.total_progress).unwrap());
        for e in order {
            println!(
                "car {:>2}: progress {:>8.1} m | lap {} | v {:>5.1} m/s | lat {:+.1} | lat_target {:+.1} | retired {}",
                e.id, e.total_progress, e.laps_crossings, e.car.vel_long,
                e.lateral, e.line_offset, e.retired
            );
        }
        let last_contact = race.state.contacts.last().cloned();
        if let Some(c) = last_contact {
            println!("last contact: tick {} cars {}&{} cs {:.1}", c.tick, c.a, c.b, c.closing_speed);
        }
        return Ok(true);
    }

    if scale_bench {
        // The plan's ladder: 1 → 4 → 10 → configured full grid, measured.
        println!("== apexsim Phase 4 grid-scaling ladder (10 s sim, max-speed, seed {}) ==", cfg.seed);
        println!("{:>6} | {:>9} {:>9} {:>10} {:>9} | {:>10}", "cars", "avg µs", "p99 µs", "p99.9 µs", "worst µs", "budget %");
        for n in [1u32, 4, 10, cfg.cars] {
            let mut c = cfg.clone();
            c.cars = n;
            let mut race = apexsim::race::Race::new(c);
            let stats = race.bench_ticks(2500); // 10 s at 250 Hz
            let worst_pct = stats.worst_us / (apexsim::race::DT * 1e6) * 100.0;
            println!(
                "{:>6} | {:>9.2} {:>9.2} {:>10.2} {:>9.1} | {:>9.2}%",
                n, stats.avg_us, stats.p99_us, stats.p999_us, stats.worst_us, worst_pct
            );
        }
        println!("budget: 4000 µs per tick (250 Hz engine rate)");
        return Ok(true);
    }

    let mode = if realtime { "realtime" } else { "max-speed" };
    println!(
        "== apexsim race | {} mode | {} cars | {} laps | seed {} | {} ==",
        cfg.mode.label(), cfg.cars, cfg.laps, cfg.seed, mode
    );
    let (outcome, inputs, exchanges) = apexsim::race::run_race_full(cfg.clone(), realtime);
    println!("ticks: {} | sim {:.1} s | wall {:.2} s", outcome.ticks, outcome.sim_time_s, outcome.wall_time_s);
    let ts = &outcome.tick_stats;
    println!(
        "tick cost (us): avg {:.1} | p50 {:.1} | p99 {:.1} | p99.9 {:.1} | worst {:.1} | n {}",
        ts.avg_us, ts.p50_us, ts.p99_us, ts.p999_us, ts.worst_us, ts.count
    );
    if realtime {
        let pct = if ts.count > 0 { outcome.deadline_misses as f64 / ts.count as f64 * 100.0 } else { 0.0 };
        println!("deadline misses: {} ({pct:.3}%)", outcome.deadline_misses);
    }
    println!("contacts: {} | retired: {:?}", outcome.contacts.len(), outcome.retired_ids);
    println!("leader laps: [{}]",
        outcome.leader_lap_times.iter().map(|t| format!("{t:.2}")).collect::<Vec<_>>().join(", "));
    if let Some((car, t, lap)) = outcome.fastest {
        println!("fastest lap: car {car} — {t:.2} s (lap {lap})");
    }
    if !outcome.pit_stops.is_empty() {
        println!(
            "pit stops: {} total | lane time {:.1}-{:.1} s",
            outcome.pit_stops.len(),
            outcome.pit_stops.iter().map(|r| r.exited_t - r.entered_t).fold(f64::INFINITY, f64::min),
            outcome.pit_stops.iter().map(|r| r.exited_t - r.entered_t).fold(0.0f64, f64::max),
        );
    }
    println!("classification:");
    for row in &outcome.classification {
        let gap = row
            .gap_s
            .map(|g| format!("{g:+.1}"))
            .unwrap_or_else(|| if row.pos == 1 { "leader".into() } else { "—".into() });
        let status = if row.retired {
            row.dnf.map(|why| format!("DNF {why}")).unwrap_or_else(|| "DNF".into())
        }
            else if row.finished { "FIN".into() }
            else { format!("lap {}", row.laps_done) };
        let best = row
            .best_lap_s
            .map(|b| format!(" {b:.2}"))
            .unwrap_or_default();
        let stops = if row.stops > 0 { format!(" {}p", row.stops) } else { String::new() };
        println!(
            "  P{:>2} car {:>2}  {:>8}  gap {:>8} s  {}{} {}{stops}",
            row.pos, row.id, status, gap, row.compound, best, format_args!("{:.0} km/h", row.v * 3.6),
        );
    }
    println!("state hash: {:016x}", outcome.state_hash);
    print_steward_record(&outcome, steward_verbose);

    // Phase 8 report. Every override is shown with the tick it actually took
    // effect on — not when it was requested — because that is the whole point
    // of queueing rather than injecting.
    if let Some(w) = outcome.weather.as_ref() {
        println!(
            "\n== weather | {} | water {:.2} | track {:.1} °C | grip trim {:.2} | mean car 0 grip {:.3} ==",
            w.label, w.water, w.track_temp_c, w.grip_scale, w.mean_grip_car0
        );
        if w.applied.is_empty() && w.unapplied == 0 {
            println!("no overrides: the race director's sliders never moved");
        }
        for a in &w.applied {
            println!(
                "  override applied @ tick {:>7} ({:>7.1} s): {:<10} -> {:.3}",
                a.tick,
                a.tick as f64 * apexsim::race::DT,
                a.slider.label(),
                a.value
            );
        }
        if w.unapplied > 0 {
            println!("  {} override(s) were queued past the flag and never applied", w.unapplied);
        }
        if w.driver_error_rate > 0.0 {
            let by_car: Vec<String> = w
                .driver_errors_by_car
                .iter()
                .map(|(id, n)| format!("car{id}:{n}"))
                .collect();
            println!(
                "driver errors: {} delivered (rate {:.2}/car/lap) | {}",
                w.driver_errors,
                w.driver_error_rate,
                by_car.join(" ")
            );
        }
    }

    // Phase 9 report. Two separate things are shown on purpose: what the
    // publisher *tried* (offered) and what the wire *did* (sent, bytes,
    // errors). A gap between offered and sent is the drop counter doing its
    // job, and is only visible if both numbers are on the page.
    if let (Some(target), Some(st)) = (outcome.telemetry_target.as_ref(), outcome.telemetry.as_ref()) {
        let (pc, uid) = cfg
            .telemetry
            .as_ref()
            .map_or((0, 1), |t| (t.player_car, t.session_uid));
        println!(
            "\n== telemetry | racing UDP layout (game year {}, m_packetFormat {}) | {} | {} Hz | player slot {} | session uid {} ==",
            2000 + apexsim::telemetry::GAME_YEAR as u32,
            apexsim::telemetry::PACKET_FORMAT,
            target,
            outcome.telemetry_hz,
            pc,
            uid,
        );
        println!(
            "offered {} | sent {} | bytes {} | dropped {} | send errors {}",
            st.offered, st.sent, st.bytes, st.dropped, st.send_errors
        );
        if st.send_errors > 0 {
            println!(
                "  send errors are the expected signature of nothing listening on that port — \
                 the race is unaffected (see the state hash above)"
            );
        }
        if st.dropped > 0 {
            println!(
                "  {} datagram(s) dropped because the sender thread fell behind; physics never waited for them",
                st.dropped
            );
        }
    }
    if let Some(err) = outcome.telemetry_error.as_ref() {
        if outcome.telemetry.is_some() {
            println!("telemetry: {err} (publisher continues; the race is unaffected)");
        }
    }

    if let (Some(sum), Some(cfg)) = (outcome.sound.as_ref(), outcome.sound_config.as_ref()) {
        println!(
            "\n== sound | {} cues extracted | slip {:.2} g | wind {:.0} m/s band from {:.0} m/s | playback ceiling {} cue/s ==",
            sum.events, cfg.slip_threshold_g, cfg.wind_band_mps, cfg.wind_start_mps, cfg.max_cues_per_s
        );
        let lines = sum.lines();
        if lines.is_empty() {
            println!("no cue fired — the car never reached a shift, the limiter, the slip threshold or the wind start speed");
        } else {
            println!("  {}", lines.join(" | "));
        }
        println!("cue log hash: {:016x}", sum.log_hash);
    }

    // Phase 7 report. The advisor's whole value is auditable: what was
    // asked, what came back, how long it took, and what the legality gate
    // did with it.
    if let (Some(a), Some(sum)) = (advisor.as_ref(), outcome.ai.as_ref()) {
        println!(
            "\n== strategy advisor | model {} | every {} laps | timeout {} ms | min confidence {:.2} ==",
            a.model, a.every_laps, a.timeout_ms, a.min_confidence
        );
        if let Some(c) = outcome.advised.as_ref() {
            let pos = c.position.map(|p| format!("P{p}")).unwrap_or_else(|| "—".into());
            println!(
                "advised car {}: {} | best {:?} | {} stops | stints {}",
                c.id,
                pos,
                c.best_lap_s.map(|t| format!("{t:.2} s")),
                c.stops,
                c.compounds.join("→")
            );
        }
        println!(
            "exchanges {} | applied {} | rejected {} | stale {} | low-conf {} | bad answer {} | timeout {} | error {}",
            sum.exchanges, sum.applied, sum.rejected, sum.stale, sum.low_confidence,
            sum.bad_answer, sum.timeout, sum.error
        );
        println!(
            "latency ms: p50 {:.1} | p99 {:.1} | final pace scale {:.3}",
            sum.latency_ms_p50, sum.latency_ms_p99, sum.pace_scale_final
        );
        if sum.transport_errors > 0 {
            println!("transport errors: {} (endpoint down or HTTP failure)", sum.transport_errors);
        }
        if sum.exchanges == 0 {
            println!(
                "no exchange was ever completed — the advisor changed nothing and the race is the deterministic baseline."
            );
        }
        for (i, x) in exchanges.iter().enumerate() {
            let resp: String = x.response.chars().take(160).collect();
            println!(
                "  #{:<2} tick {:>6} | {:<14} | {:>7.0} ms | {} | {}",
                i + 1,
                x.request_tick,
                x.outcome,
                x.latency_ms,
                x.detail,
                resp.replace('\n', " ")
            );
        }
        if let Some(path) = ai_log {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| format!("create ai-log dir: {e}"))?;
                }
            }
            let mut w = std::io::BufWriter::new(
                std::fs::File::create(&path).map_err(|e| format!("create ai log: {e}"))?,
            );
            use std::io::Write;
            for x in &exchanges {
                writeln!(
                    w,
                    "{}",
                    serde_json::json!({
                        "request_tick": x.request_tick,
                        "applied_tick": x.applied_tick,
                        "prompt_summary": x.prompt_summary,
                        "model": x.model,
                        "response": x.response,
                        "latency_ms": (x.latency_ms * 10.0).round() / 10.0,
                        "outcome": x.outcome,
                        "detail": x.detail,
                    })
                )
                .map_err(|e| format!("write ai log: {e}"))?;
            }
            println!("ai log: {}", path.display());
        }
    }

    if let Some(path) = log {
        if cfg.ai.is_some() {
            return Err(
                "--log is refused on an --ai run: model latency makes the run non-reproducible, \
                 so a replay would not match. Use --ai-log for the exchange record instead."
                    .into(),
            );
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create log dir: {e}"))?;
        }
        apexsim::race::write_race_log(&path, &cfg, &inputs, &outcome)
            .map_err(|e| format!("write log: {e}"))?;
        println!("log: {}", path.display());
    }
    Ok(true)
}

/// Phase 8 benchmark: measure what the surface actually costs, rather than
/// asserting it. Three measurements, all in isolation so the numbers are
/// attributable:
///   1. braking distance from a fixed speed, dry vs damp vs wet;
///   2. steady-state cornering radius at a fixed lateral acceleration;
///   3. a full race's lap times and total time, dry vs wet.
///
/// (1) and (2) use the *same* physics step the race uses, with a fixed
/// control input — so the difference is grip and nothing else. (3) is the
/// emergent result, which includes the driver correctly slowing down.
fn cmd_weatherbench(args: &[String]) -> Result<bool, String> {
    use apexsim::physics::{self, CarParams, CarState, Controls};
    use apexsim::weather::{surface_grip, Surface};

    let mut seeds: Vec<u64> = Vec::new();
    let mut laps = 8u32;
    let mut cars = 20u32;
    let mut speed_kmh = 200.0f64;
    let mut i = 0;
    while i < args.len() {
        let need = |name: &str| -> Result<String, String> {
            args.get(i + 1).cloned().ok_or_else(|| format!("{name} requires a value"))
        };
        match args[i].as_str() {
            "--seeds" => {
                seeds = need("--seeds")?
                    .split(',')
                    .map(|s| s.trim().parse::<u64>().map_err(|e| format!("--seeds: {e}")))
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "--laps" => {
                laps = need("--laps")?.parse().map_err(|e| format!("--laps: {e}"))?;
                i += 2;
            }
            "--cars" => {
                cars = need("--cars")?.parse().map_err(|e| format!("--cars: {e}"))?;
                i += 2;
            }
            "--speed" => {
                speed_kmh = need("--speed")?.parse().map_err(|e| format!("--speed: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    if seeds.is_empty() {
        seeds = (1..=3).collect();
    }
    let v0 = speed_kmh / 3.6;
    if v0 <= 5.0 {
        return Err("--speed must be above 5 km/h for a braking measurement".into());
    }

    println!("== apexsim Phase 8 weather benchmark ==");
    println!(
        "scenario: {cars} cars | {laps} laps | {} seeds | braking measured from {speed_kmh:.0} km/h\n",
        seeds.len()
    );

    // ---- 1. braking distance -------------------------------------------------
    // Fixed brake input from a fixed speed, full brake force, no cornering.
    // Distance is integrated from the actual velocity trace.
    let brake_distance = |mu: f64| -> f64 {
        let p = CarParams { tire_grip: CarParams::default().tire_grip * mu, ..Default::default() };
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
    };
    println!("-- braking distance from {:.0} km/h --", speed_kmh);
    println!("  surface |  grip  |  distance |  vs dry");
    // Each surface at the water level its own preset uses — a "dry" row
    // measured with the water slider at 1.0 would not be dry at all.
    let dry_mu = surface_grip(Surface::Dry, 0.0, 35.0, 1.0, v0);
    let dry_d = brake_distance(dry_mu);
    let mut brake_rows: Vec<(Surface, f64, f64)> = Vec::new();
    for (surf, water) in [
        (Surface::Dry, 0.0),
        (Surface::Damp, 0.35),
        (Surface::Wet, 1.0),
    ] {
        let mu = surface_grip(surf, water, 35.0, 1.0, v0);
        let d = brake_distance(mu);
        brake_rows.push((surf, mu, d));
        println!(
            "  {:<7} | {:.3}  | {:>7.1} m  | {:+.0}%",
            surf.label(),
            mu,
            d,
            (d / dry_d - 1.0) * 100.0
        );
    }

    // ---- 2. cornering -------------------------------------------------------
    // The peak lateral acceleration the car can actually hold at a fixed speed
    // — the grip envelope, found by sweeping the steering input and keeping the
    // fastest steady state the car reaches without spinning away. A single
    // steer angle would not be a fair comparison: the bicycle model goes
    // unstable past a point that moves with grip, so one input either under-
    // or over-saturates depending on surface.
    let corner_v = 25.0f64;
    let peak_lat = |grip: f64| -> (f64, f64, u32) {
        let mut best = 0.0f64;
        let mut best_steer = 0.0f64;
        let mut lost = 0u32;
        for si in 0..40 {
            let steer = 0.05 + si as f64 * 0.02;
            let p = CarParams { tire_grip: CarParams::default().tire_grip * grip, ..Default::default() };
            let mut s = CarState { vel_long: corner_v, ..Default::default() };
            let mut ok = true;
            for _ in 0..3000 {
                // Proportional speed hold, so the only thing changing between
                // surfaces is what the tires can do — not how fast we went in.
                let err = corner_v - s.vel_long;
                let ctrl = Controls {
                    steer_cmd: steer,
                    throttle: (err * 0.5).clamp(0.0, 1.0),
                    brake: (-err * 0.5).clamp(0.0, 1.0),
                };
                s = physics::step(&s, &ctrl, &p, apexsim::race::DT);
                if s.vel_long < corner_v * 0.4 {
                    ok = false; // slid off, or spun
                    break;
                }
            }
            if !ok {
                lost += 1;
                continue;
            }
            let lat = s.yaw_rate * s.vel_long;
            if lat > best {
                best = lat;
                best_steer = steer;
            }
        }
        (best, best_steer, lost)
    };
    println!(
        "\n-- cornering: peak steady lateral acceleration at {:.0} km/h --",
        corner_v * 3.6
    );
    println!("  surface |  grip  |  peak lat accel |  at steer |  lost the car");
    let mut corner_rows: Vec<(Surface, f64, f64)> = Vec::new();
    for (surf, water) in [
        (Surface::Dry, 0.0),
        (Surface::Damp, 0.35),
        (Surface::Wet, 1.0),
    ] {
        let grip = surface_grip(surf, water, 35.0, 1.0, corner_v);
        let (lat, steer, lost) = peak_lat(grip);
        corner_rows.push((surf, grip, lat));
        println!(
            "  {:<7} | {:.3}  | {:>13.2} m/s² | {:>8.2}  | {:>6}/40 inputs",
            surf.label(),
            grip,
            lat,
            steer,
            lost
        );
    }

    // ---- 3. full race, dry vs wet -------------------------------------------
    println!("\n-- full race | {} cars | {} laps | leader + car 0 --", cars, laps);
    println!("  weather | seed |  leader time |  leader best |  car 0 mean lap |  mean grip");
    let run = |w: Option<&str>, seed: u64| -> (f64, f64, f64, f64) {
        let cfg = apexsim::race::RaceConfig {
            cars,
            laps,
            seed,
            // An unknown preset name is a caller typo: fail loudly, as the
            // original `unwrap` did, rather than silently running a dry race.
            weather: w.map(|name| {
                apexsim::weather::WeatherConfig::preset(name)
                    .expect("valid weather preset name")
            }),
            ..apexsim::race::RaceConfig::default()
        };
        let (outcome, _, _) = apexsim::race::run_race_full(cfg, false);
        let leader = outcome
            .leader_lap_times
            .iter()
            .copied()
            .fold(0.0, f64::max);
        let lead_total = outcome.leader_lap_times.iter().sum::<f64>();
        let mean_lap = if outcome.advised.as_ref().map(|c| c.laps).unwrap_or(0) > 0 {
            outcome
                .classification
                .iter()
                .find(|r| r.id == 0)
                .and_then(|r| r.best_lap_s)
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        (
            lead_total,
            leader,
            mean_lap,
            outcome.weather.as_ref().map(|w| w.mean_grip_car0).unwrap_or(1.0),
        )
    };
    let mut wet_slower = 0;
    let mut compared = 0;
    for seed in &seeds {
        let (dt, db, _, dg) = run(None, *seed);
        let (wt, wb, wg, gg) = run(Some("wet"), *seed);
        compared += 1;
        if wt > dt {
            wet_slower += 1;
        }
        println!(
            "  dry      | {:>4} | {:>10.1} s | {:>11.2} s | {:>13} | {:.3}",
            seed,
            dt,
            db,
            "—",
            dg
        );
        println!(
            "  wet      | {:>4} | {:>10.1} s | {:>11.2} s | {:>13.2} s | {:.3}",
            seed,
            wt,
            wb,
            wg,
            gg
        );
        println!(
            "  delta    |      | {:>+9.1} s | {:>+11.2} s |",
            wt - dt,
            wb - db
        );
    }

    // ---- verdict -------------------------------------------------------------
    let dry_brake = brake_rows[0].2;
    let wet_brake = brake_rows[2].2;
    let dry_lat = corner_rows[0].2;
    let wet_lat = corner_rows[2].2;
    let brake_ok = wet_brake > dry_brake * 1.2;
    let corner_ok = wet_lat < dry_lat * 0.9 && dry_lat > 0.0;
    let race_ok = wet_slower == compared && compared > 0;
    println!("\n-- verdict --");
    println!(
        "  wet braking {:.0}% longer than dry: {}",
        (wet_brake / dry_brake - 1.0) * 100.0,
        if brake_ok { "MEASURED" } else { "NOT MEASURED" }
    );
    println!(
        "  wet cornering limit {:.0}% lower than dry ({:.2} g vs {:.2} g): {}",
        (wet_lat / dry_lat - 1.0) * 100.0,
        wet_lat / 9.81,
        dry_lat / 9.81,
        if corner_ok { "MEASURED" } else { "NOT MEASURED" }
    );
    println!(
        "  wet slower in {wet_slower}/{compared} race(s): {}",
        if race_ok { "MEASURED" } else { "NOT MEASURED" }
    );
    Ok(brake_ok && corner_ok && race_ok)
}

/// Phase 7 benchmark: run the same saved scenario twice — once on the
/// deterministic Phase 5 strategy, once with the model advising car 0 — and
/// report where the advice gained, changed nothing, or cost time.
///
/// The comparison is only meaningful because everything except the advisor
/// is identical: same seed, same track, same grid, same physics.
fn cmd_aibench(args: &[String]) -> Result<bool, String> {

    let mut cfg = apexsim::race::RaceConfig::default();
    let mut advisor = apexsim::strategy::AdvisorConfig::default();
    let mut seeds: Vec<u64> = Vec::new();
    let mut ai_log: Option<std::path::PathBuf> = None;
    let mut quiet = false;
    let mut i = 0;
    while i < args.len() {
        let need = |name: &str| -> Result<String, String> {
            args.get(i + 1).cloned().ok_or_else(|| format!("{name} requires a value"))
        };
        match args[i].as_str() {
            "--seeds" => {
                seeds = need("--seeds")?
                    .split(',')
                    .map(|s| s.trim().parse::<u64>().map_err(|e| format!("--seeds: {e}")))
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "--model" => {
                advisor.model = need("--model")?;
                i += 2;
            }
            "--endpoint" => {
                advisor.endpoint = need("--endpoint")?;
                i += 2;
            }
            "--timeout" => {
                advisor.timeout_ms = need("--timeout")?.parse().map_err(|e| format!("--timeout: {e}"))?;
                i += 2;
            }
            "--max-tokens" => {
                advisor.max_tokens = need("--max-tokens")?.parse().map_err(|e| format!("--max-tokens: {e}"))?;
                i += 2;
            }
            "--every-laps" => {
                advisor.every_laps = need("--every-laps")?.parse().map_err(|e| format!("--every-laps: {e}"))?;
                i += 2;
            }
            "--min-confidence" => {
                advisor.min_confidence = need("--min-confidence")?.parse().map_err(|e| format!("--min-confidence: {e}"))?;
                i += 2;
            }
            "--cars" => {
                cfg.cars = need("--cars")?.parse().map_err(|e| format!("--cars: {e}"))?;
                i += 2;
            }
            "--laps" => {
                cfg.laps = need("--laps")?.parse().map_err(|e| format!("--laps: {e}"))?;
                i += 2;
            }
            "--ai-log" => {
                ai_log = Some(std::path::PathBuf::from(need("--ai-log")?));
                i += 2;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    if seeds.is_empty() {
        seeds = (1..=6).collect();
    }
    if advisor.every_laps == 0 {
        return Err("--every-laps must be >= 1".into());
    }

    println!(
        "== apexsim Phase 7 strategy benchmark ==",
    );
    println!(
        "model {} | endpoint {} | every {} laps | timeout {} ms | max tokens {} | min confidence {:.2}",
        advisor.model, advisor.endpoint, advisor.every_laps, advisor.timeout_ms,
        advisor.max_tokens, advisor.min_confidence
    );
    println!(
        "scenario: {} cars | {} laps | {} seeds | car 0 advised (deterministic strategy owns everything else)\n",
        cfg.cars, cfg.laps, seeds.len()
    );
    println!(
        "{:>6} | {:>16} | {:>16} | {:>9} | {:>22}",
        "seed", "baseline P / time", "advised P / time", "verdict", "advisor outcome tally"
    );

    let mut helps = 0usize;
    let mut ties = 0usize;
    let mut hurts = 0usize;
    let mut worse_unknown = 0usize;
    let mut all_exchanges: Vec<apexsim::strategy::AiExchange> = Vec::new();
    let mut t_base = 0.0f64;
    let mut t_adv = 0.0f64;
    let mut compared = 0usize;
    let mut dnf_flips = 0usize;

    for seed in &seeds {
        let mut base_cfg = cfg.clone();
        base_cfg.seed = *seed;
        base_cfg.ai = None;
        let (base, _, _) = apexsim::race::run_race_full(base_cfg, false);

        let mut ai_cfg = cfg.clone();
        ai_cfg.seed = *seed;
        ai_cfg.ai = Some(advisor.clone());
        let (adv, _, log) = apexsim::race::run_race_full(ai_cfg, false);
        all_exchanges.extend(log.iter().cloned());

        let (bp, bt) = car0(&base);
        let (ap, at) = car0(&adv);
        let verdict = match (bp, ap) {
            (Some(b), Some(a)) if a < b => {
                helps += 1;
                "HELPED"
            }
            (Some(b), Some(a)) if a > b => {
                hurts += 1;
                "HURT"
            }
            _ => {
                ties += 1;
                "tie"
            }
        };
        if verdict == "tie" {
            // A tie in position can still be a time loss — record it so the
            // summary doesn't quietly call a 2 s loss a tie.
            if let (Some(b), Some(a)) = (bt, at) {
                if (a - b).abs() > 0.05 {
                    worse_unknown += 1;
                }
            }
        }
        // Time is only comparable where BOTH runs produced a finish time.
        // A seed where one run DNFs is reported, not silently averaged in.
        match (bt, at) {
            (Some(b), Some(a)) => {
                t_base += b;
                t_adv += a;
                compared += 1;
            }
            _ => {
                if bt.is_none() != at.is_none() {
                    dnf_flips += 1;
                }
            }
        }

        let tally = match adv.ai.as_ref() {
            Some(s) => format!(
                "app {} rej {} stale {} lo {} bad {} to {} err {}",
                s.applied, s.rejected, s.stale, s.low_confidence, s.bad_answer, s.timeout, s.error
            ),
            None => "—".into(),
        };
        println!(
            "{:>6} | {:>16} | {:>16} | {:>9} | {:>22}",
            seed,
            fmt_pos_time(bp, bt),
            fmt_pos_time(ap, at),
            verdict,
            tally
        );
    }

    let applied = all_exchanges.iter().filter(|x| x.outcome == "applied").count();
    let mut lat: Vec<f64> = all_exchanges.iter().map(|x| x.latency_ms).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| -> f64 {
        if lat.is_empty() { 0.0 } else {
            lat[((lat.len() as f64 - 1.0) * q).round() as usize].min(lat[lat.len() - 1])
        }
    };

    println!("\n-- verdict --");
    println!("helped: {helps} | tied: {ties} | hurt: {hurts}");
    if ties > 0 && worse_unknown > 0 {
        println!("  ({worse_unknown} of the ties still lost more than 0.05 s)");
    }
    if compared > 0 {
        println!(
            "car 0 total time over the {} comparable seed(s): baseline {:.2} s | advised {:.2} s | delta {:+.2} s",
            compared,
            t_base,
            t_adv,
            t_adv - t_base
        );
    }
    if dnf_flips > 0 {
        println!(
            "  ({dnf_flips} seed(s): one run finished and the other did not — excluded from the time total, already counted as a position change)"
        );
    }
    println!("\n-- advisor --");
    println!(
        "exchanges {} | applied {} | p50 {:.0} ms | p99 {:.0} ms",
        all_exchanges.len(),
        applied,
        p(0.5),
        p(0.99)
    );
    let verdict_line = match (helps, hurts) {
        (0, 0) => "the model never changed the result — every answer fell back.".to_string(),
        (h, 0) if h > 0 => format!("the model helped in {h} scenario(s) and hurt in none."),
        (0, u) if u > 0 => format!("the model hurt in {u} scenario(s) and helped in none."),
        (h, u) => format!("the model helped in {h} and hurt in {u} scenario(s) — no consistent edge."),
    };
    println!("verdict: {verdict_line}");

    if !quiet {
        println!("\n-- every exchange --");
        for (i, x) in all_exchanges.iter().enumerate() {
            let resp: String = x.response.chars().take(110).collect();
            println!(
                "  #{:<2} seed? tick {:>6} | {:<14} | {:>6.0} ms | {}",
                i + 1,
                x.request_tick,
                x.outcome,
                x.latency_ms,
                x.detail
            );
            println!("        prompt: {}", x.prompt_summary);
            println!("        reply : {}", resp.replace('\n', " "));
        }
    }

    if let Some(path) = ai_log {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| format!("create ai-log dir: {e}"))?;
            }
        }
        let mut w = std::io::BufWriter::new(
            std::fs::File::create(&path).map_err(|e| format!("create ai log: {e}"))?,
        );
        use std::io::Write;
        for x in &all_exchanges {
            writeln!(
                w,
                "{}",
                serde_json::json!({
                    "request_tick": x.request_tick,
                    "applied_tick": x.applied_tick,
                    "prompt_summary": x.prompt_summary,
                    "model": x.model,
                    "response": x.response,
                    "latency_ms": (x.latency_ms * 10.0).round() / 10.0,
                    "outcome": x.outcome,
                    "detail": x.detail,
                })
            )
            .map_err(|e| format!("write ai log: {e}"))?;
        }
        println!("\nai log: {}", path.display());
    }
    Ok(true)
}

/// Car 0's finishing position and time from an outcome.
fn car0(o: &apexsim::race::RaceOutcome) -> (Option<u32>, Option<f64>) {
    (
        o.advised.as_ref().and_then(|a| a.position),
        o.advised.as_ref().and_then(|a| a.total_time_s),
    )
}

fn fmt_pos_time(p: Option<u32>, t: Option<f64>) -> String {
    match (p, t) {
        (Some(p), Some(t)) => format!("P{p} / {t:.2}s"),
        (Some(p), None) => format!("P{p} / —"),
        _ => "DNF / —".into(),
    }
}

struct RunOpts {
    laps: u32,
    seed: u64,
    noise: f64,
    realtime: bool,
    log: Option<PathBuf>,
    quiet: bool,
}

fn parse_run_opts(args: &[String]) -> Result<RunOpts, String> {
    let mut o = RunOpts { laps: 5, seed: 42, noise: 0.0, realtime: false, log: None, quiet: false };
    let mut i = 0;
    let need = |name: &str, val: Option<&String>| -> Result<String, String> {
        val.ok_or_else(|| format!("{name} requires a value")).cloned()
    };
    while i < args.len() {
        match args[i].as_str() {
            "--laps" => {
                o.laps = need("--laps", args.get(i + 1))?.parse().map_err(|e| format!("--laps: {e}"))?;
                i += 2;
            }
            "--seed" => {
                o.seed = need("--seed", args.get(i + 1))?.parse().map_err(|e| format!("--seed: {e}"))?;
                i += 2;
            }
            "--noise" => {
                o.noise = need("--noise", args.get(i + 1))?.parse().map_err(|e| format!("--noise: {e}"))?;
                i += 2;
            }
            "--realtime" => {
                o.realtime = true;
                i += 1;
            }
            "--log" => {
                o.log = Some(PathBuf::from(need("--log", args.get(i + 1))?));
                i += 2;
            }
            "--quiet" => {
                o.quiet = true;
                i += 1;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    Ok(o)
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let opts = parse_run_opts(args)?;
    let config = SimConfig {
        laps: opts.laps,
        seed: opts.seed,
        realtime: opts.realtime,
        driver: apexsim::driver::DriverParams {
            noise_amplitude: opts.noise,
            ..apexsim::driver::DriverParams::default()
        },
        ..SimConfig::default()
    };

    let (tx, rx) = mpsc::channel::<Event>();
    let sim_cfg = config.clone();
    let handle = thread::Builder::new()
        .name("sim".into())
        .spawn(move || run_session(sim_cfg, Some(&tx)))
        .map_err(|e| format!("spawn sim thread: {e}"))?;

    let mut done = false;
    while !done {
        let ev = match rx.recv() {
            Ok(ev) => ev,
            Err(_) => break,
        };
        match ev {
            Event::Started { seed, laps, realtime } => {
                if !opts.quiet {
                    println!(
                        "== apexsim Phase 1 | mode {} | laps {laps} | seed {seed} ==",
                        if realtime { "realtime" } else { "max-speed" }
                    );
                }
            }
            Event::Lap { n, time_s } => {
                if !opts.quiet {
                    println!("  lap {n}: {time_s:.2} s");
                }
            }
            Event::Progress { sim_time, u, s } => {
                if !opts.quiet {
                    println!("  t={sim_time:6.1}s  v={u:5.1} m/s  s={s:7.1} m");
                }
            }
            Event::Finished(_) => {
                done = true;
            }
        }
    }
    let (outcome, inputs) = handle.join().map_err(|_| "sim thread panicked")?;

    let log_path = opts.log.clone().unwrap_or_else(|| {
        let name = if opts.realtime { "phase1_realtime.log" } else { "phase1_maxspeed.log" };
        PathBuf::from("artifacts").join(name)
    });
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create artifacts dir: {e}"))?;
    }
    write_log(&log_path, &config, &inputs, &outcome).map_err(|e| format!("write log: {e}"))?;

    print_report(&outcome, opts.realtime);
    println!("  memory: {}", memory_label());
    println!("  state hash: {:016x}", outcome.state_hash);
    println!("  log: {}", log_path.display());
    Ok(())
}

fn print_report(o: &SimOutcome, realtime: bool) {
    let laps: Vec<String> = o.lap_times.iter().map(|t| format!("{t:.2}")).collect();
    println!("laps done: {}/{}", o.laps_done, laps.len().max(o.laps_done as usize));
    println!("  lap times: [{}]", laps.join(", "));
    if o.lap_times.len() >= 2 {
        let rest = &o.lap_times[1..];
        let best = rest.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst = rest.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("  best {best:.2} s | worst {worst:.2} s | spread (lap 2+) {:.1}%", (worst - best) / best * 100.0);
    }
    println!("offtrack ticks: {}", o.offtrack_ticks);
    let speedup = o.sim_time / o.wall_time_s.max(1e-9);
    println!(
        "physics ticks: {} | sim {:.1} s | wall {:.2} s ({}x realtime)",
        o.ticks, o.sim_time, o.wall_time_s,
        if realtime { "1.0".to_string() } else { format!("{speedup:.0}") }
    );
    let ts = &o.tick_stats;
    println!(
        "tick cost (us): avg {:.1} | p50 {:.1} | p99 {:.1} | p99.9 {:.1} | worst {:.1} | n {}",
        ts.avg_us, ts.p50_us, ts.p99_us, ts.p999_us, ts.worst_us, ts.count
    );
    if realtime {
        let total_dt_us = DT * 1e6;
        let miss_pct = if ts.count > 0 { o.deadline_misses as f64 / ts.count as f64 * 100.0 } else { 0.0 };
        println!(
            "deadline misses (> {total_dt_us:.0} us): {} ({miss_pct:.3}%) | max start lag {:.0} us",
            o.deadline_misses, o.max_lag_us
        );
    }
}

fn memory_label() -> String {
    match self_memory_mb() {
        Some(mb) => format!("{mb:.1} MB"),
        None => "unavailable".to_string(),
    }
}

/// Best-effort working-set size of this process via `tasklist`.
/// Returns None rather than failing the run if the parse falls apart.
fn self_memory_mb() -> Option<f64> {
    let pid = std::process::id();
    let out = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.contains(".exe"))?;
    // Fields are quoted; the memory field is last, e.g. "84,296 K".
    let last = line.rsplit("\",\"").next()?;
    let digits: String = last.chars().filter(|c| c.is_ascii_digit()).collect();
    let kb: f64 = digits.parse().ok()?;
    Some(kb / 1024.0)
}

fn cmd_serve(args: &[String]) -> Result<(), String> {
    use apexsim::server::ServeOpts;
    let mut port: u16 = 8080;
    // The config file below overrides most of these, so this binding stays
    // mutable; only the default session length is fixed at construction.
    let mut config = SimConfig {
        laps: 999, // continuous demo session by default
        // The dashboard serves the 2026 championship, realism pack on.
        field: apexsim::race::FieldMode::Teams,
        realism: Some(apexsim::race::RealismConfig::default()),
        ..SimConfig::default()
    };
    let mut autostart = true;
    let mut structured = false;
    let mut duration: Option<std::time::Duration> = None;
    let mut run_id: Option<String> = None;
    let mut log_level: Option<String> = None;
    let mut log_file: Option<PathBuf> = None;
    // Phase 9, same flags as `apexsim race`, so a dashboard session can be
    // driven onto the wire and heard at the same time.
    let mut tel_target: Option<String> = None;
    let mut tel_hz: Option<u32> = None;
    let mut tel_player: Option<u32> = None;
    let mut sound_on = false;
    let mut s_max_cues: Option<u32> = None;

    // Pass 1: a config file (if any) seeds every value...
    let mut cfg_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--config" {
            cfg_path = Some(
                args.get(i + 1)
                    .cloned()
                    .ok_or_else(|| "--config requires a value".to_string())?,
            );
            i += 2;
        } else {
            i += 1;
        }
    }
    if let Some(p) = &cfg_path {
        let text = std::fs::read_to_string(p).map_err(|e| format!("read config {p}: {e}"))?;
        let dev = apexsim::config::parse_config(&text).map_err(|e| format!("{p}: {e}"))?;
        port = dev.port;
        config.cars = dev.cars;
        config.laps = dev.laps;
        config.seed = dev.seed;
        config.driver.noise_amplitude = dev.noise;
        config.mode = dev.mode.clone();
        autostart = dev.autostart;
        log_level = Some(dev.log_level.clone());
        if let Some(f) = dev.log_file {
            log_file = Some(PathBuf::from(f));
        }
    }

    // Pass 2: explicit flags override the file.
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--config" {
            i += 2; // consumed in pass 1
            continue;
        }
        let need = |name: &str| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match args[i].as_str() {
            "--port" => {
                port = need("--port")?.parse().map_err(|e| format!("--port: {e}"))?;
                i += 2;
            }
            "--laps" => {
                config.laps = need("--laps")?.parse().map_err(|e| format!("--laps: {e}"))?;
                i += 2;
            }
            "--seed" => {
                config.seed = need("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?;
                i += 2;
            }
            "--noise" => {
                config.driver.noise_amplitude =
                    need("--noise")?.parse().map_err(|e| format!("--noise: {e}"))?;
                i += 2;
            }
            "--cars" => {
                let n: u32 = need("--cars")?.parse().map_err(|e| format!("--cars: {e}"))?;
                if !(1..=30).contains(&n) {
                    return Err(format!("--cars must be 1..=30, got {n}"));
                }
                config.cars = n;
                i += 2;
            }
            "--mode" => {
                let v = need("--mode")?;
                if apexsim::race::SessionMode::parse(&v).is_none() {
                    return Err(format!("--mode must be race | qualifying | practice, got '{v}'"));
                }
                config.mode = v;
                i += 2;
            }
            "--field" => {
                let v = need("--field")?;
                config.field = apexsim::race::FieldMode::parse(&v)
                    .ok_or_else(|| format!("--field must be teams | classic, got '{v}'"))?;
                i += 2;
            }
            "--realism" => {
                let v = need("--realism")?;
                config.realism = match v.as_str() {
                    "on" | "full" => Some(apexsim::race::RealismConfig::default()),
                    "off" | "none" => None,
                    other => return Err(format!("--realism must be on | off, got '{other}'")),
                };
                i += 2;
            }
            "--weekend" => {
                config.weekend_quali = true;
                i += 1;
            }
            "--quali-s" => {
                let v: f64 = need("--quali-s")?.parse().map_err(|e| format!("--quali-s: {e}"))?;
                if !(10.0..=1800.0).contains(&v) {
                    return Err(format!("--quali-s must be 10..=1800, got {v}"));
                }
                config.quali_window_s = v;
                i += 2;
            }
            "--track" => {
                let v = need("--track")?;
                let spec = apexsim::track::resolve_track_spec(&v)?;
                if let Some(c) = &spec.custom {
                    config.track_custom = Some(c.clone());
                    config.track_half_width = spec.half_width;
                } else {
                    config.track_straight = spec.straight_len;
                    config.track_radius = spec.radius;
                    config.track_half_width = spec.half_width;
                }
                i += 2;
            }
            "--structured" => {
                structured = true;
                i += 1;
            }
            "--run-id" => {
                run_id = Some(need("--run-id")?);
                i += 2;
            }
            "--log-level" => {
                log_level = Some(need("--log-level")?);
                i += 2;
            }
            "--log-file" => {
                log_file = Some(PathBuf::from(need("--log-file")?));
                i += 2;
            }
            "--duration" => {
                let secs: f64 = need("--duration")?.parse().map_err(|e| format!("--duration: {e}"))?;
                if !(0.1..=86_400.0).contains(&secs) {
                    return Err(format!("--duration must be 0.1..=86400 seconds, got {secs}"));
                }
                duration = Some(std::time::Duration::from_secs_f64(secs));
                i += 2;
            }
            "--no-autostart" => {
                autostart = false;
                i += 1;
            }
            "--telemetry" => {
                tel_target = Some(need("--telemetry")?);
                i += 2;
            }
            "--telemetry-hz" => {
                tel_hz = Some(need("--telemetry-hz")?.parse().map_err(|e| format!("--telemetry-hz: {e}"))?);
                i += 2;
            }
            "--telemetry-player" => {
                tel_player = Some(need("--telemetry-player")?.parse().map_err(|e| format!("--telemetry-player: {e}"))?);
                i += 2;
            }
            "--sound" => {
                sound_on = true;
                i += 1;
            }
            "--sound-max-cues" => {
                s_max_cues = Some(need("--sound-max-cues")?.parse().map_err(|e| format!("--sound-max-cues: {e}"))?);
                sound_on = true;
                i += 2;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }

    if tel_target.is_some() || tel_hz.is_some() || tel_player.is_some() {
        use apexsim::telemetry::TelemetryConfig;
        let mut t = TelemetryConfig::default();
        if let Some(target) = &tel_target {
            t.target = TelemetryConfig::parse_target(target)?;
        }
        if let Some(hz) = tel_hz {
            if !(1..=250).contains(&hz) {
                return Err(format!("--telemetry-hz must be 1..=250, got {hz}"));
            }
            t.hz = hz;
        }
        if let Some(p) = tel_player {
            if p >= config.cars {
                return Err(format!(
                    "--telemetry-player {p} is not on a {}-car grid (valid 0..={})",
                    config.cars,
                    config.cars - 1
                ));
            }
            t.player_car = p;
        }
        config.telemetry = Some(t);
    }
    if sound_on {
        let mut s = apexsim::sound::SoundConfig::default();
        if let Some(n) = s_max_cues {
            s.max_cues_per_s = n;
        }
        config.sound = Some(s);
    }

    let level = log_level
        .map(|s| {
            apexsim::logging::Level::parse(&s)
                .ok_or_else(|| format!("log level must be debug | info | warn | error, got '{s}'"))
        })
        .transpose()?;
    let opts = ServeOpts {
        structured,
        duration,
        run_id,
        log_level: level,
        log_file,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("tokio runtime: {e}"))?;
    runtime
        .block_on(apexsim::server::serve(config, port, autostart, opts))
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 6 development launcher: `apexsim dev`
//
// One command that starts the dashboard as a CHILD PROCESS IT OWNS, waits
// for real readiness (health polling, not a fixed sleep), relays structured
// logs, and — on duration end, Ctrl-C, or stdin EOF — shuts the child down.
// Shutdown stops only what this run started; an unrelated server on another
// port is untouched.
// ---------------------------------------------------------------------------

enum DevEvent {
    StdinEof,
    CtrlC,
}

fn cmd_dev(args: &[String]) -> Result<u32, String> {
    // --init: write the documented config file and stop.
    if args.iter().any(|a| a == "--init") {
        let path = config_path_arg(args)?.unwrap_or_else(|| PathBuf::from("f1_dev.toml"));
        if path.exists() {
            return Err(format!(
                "{} already exists — delete it first (the file documents itself; no silent overwrites)",
                path.display()
            ));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create {}: {e}", parent.display()))?;
            }
        }
        std::fs::write(&path, apexsim::config::CONFIG_TEMPLATE)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        println!("wrote {} — edit it, then run `apexsim dev`", path.display());
        return Ok(0);
    }

    // Config file seeds values; explicit flags override.
    let mut dev = apexsim::config::DevConfig::default();
    let mut duration: Option<f64> = None;
    if let Some(p) = config_path_arg(args)? {
        let text = std::fs::read_to_string(&p).map_err(|e| format!("read config {}: {e}", p.display()))?;
        dev = apexsim::config::parse_config(&text).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--config" {
            i += 2;
            continue;
        }
        let need = |name: &str| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match args[i].as_str() {
            "--port" => {
                dev.port = need("--port")?.parse().map_err(|e| format!("--port: {e}"))?;
                i += 2;
            }
            "--cars" => {
                dev.cars = need("--cars")?.parse().map_err(|e| format!("--cars: {e}"))?;
                i += 2;
            }
            "--laps" => {
                dev.laps = need("--laps")?.parse().map_err(|e| format!("--laps: {e}"))?;
                i += 2;
            }
            "--seed" => {
                dev.seed = need("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?;
                i += 2;
            }
            "--noise" => {
                dev.noise = need("--noise")?.parse().map_err(|e| format!("--noise: {e}"))?;
                i += 2;
            }
            "--mode" => {
                dev.mode = need("--mode")?;
                i += 2;
            }
            "--no-autostart" => {
                dev.autostart = false;
                i += 1;
            }
            "--duration" => {
                duration = Some(need("--duration")?.parse().map_err(|e| format!("--duration: {e}"))?);
                i += 2;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    if let Some(secs) = duration {
        if !(0.1..=86_400.0).contains(&secs) {
            return Err(format!("--duration must be 0.1..=86400 seconds, got {secs}"));
        }
    }
    dev.validate().map_err(|e| format!("invalid configuration: {e}"))?;

    let level = apexsim::logging::Level::parse(&dev.log_level)
        .ok_or_else(|| format!("log level must be debug | info | warn | error, got '{}'", dev.log_level))?;
    if let Some(f) = &dev.log_file {
        if let Some(parent) = std::path::Path::new(f).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create log dir {}: {e}", parent.display()))?;
            }
        }
    }
    let run_id = format!("r{:x}-{}", unix_ms(), std::process::id());
    let logger = apexsim::logging::Logger::new(run_id.clone(), level, dev.log_file.as_deref().map(std::path::Path::new))
        .map_err(|e| format!("open log file: {e}"))?;
    logger.log(apexsim::logging::Level::Info, None, None, &format!(
        "dev launcher starting (port {}, cars {}, laps {}, seed {}, mode {})",
        dev.port, dev.cars, dev.laps, dev.seed, dev.mode
    ));
    println!(
        "== apexsim dev | run {run_id} | port {} | {} cars | {} laps | {} mode ==",
        dev.port, dev.cars, dev.laps, dev.mode
    );

    // Spawn the server as a child this run owns, sharing the run id.
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let mut cmd_args = vec![
        "serve".to_string(),
        "--port".to_string(), dev.port.to_string(),
        "--cars".to_string(), dev.cars.to_string(),
        "--laps".to_string(), dev.laps.to_string(),
        "--seed".to_string(), dev.seed.to_string(),
        "--noise".to_string(), format!("{}", dev.noise),
        "--mode".to_string(), dev.mode.clone(),
        "--structured".to_string(),
        "--run-id".to_string(), run_id.clone(),
        "--log-level".to_string(), dev.log_level.clone(),
    ];
    if let Some(f) = &dev.log_file {
        cmd_args.push("--log-file".into());
        cmd_args.push(f.clone());
    }
    if !dev.autostart {
        cmd_args.push("--no-autostart".into());
    }
    if let Some(secs) = duration {
        cmd_args.push("--duration".into());
        cmd_args.push(format!("{secs}"));
    }
    let mut child = std::process::Command::new(&exe)
        .args(&cmd_args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn server: {e}"))?;
    let child_id = child.id();

    // Relay the child's output verbatim (it is structured NDJSON under
    // --structured; keep the combined stream uniform).
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        use std::io::{Read, Write as _};
        let mut buf = [0u8; 4096];
        let mut out = std::io::stdout();
        while let Ok(n) = stdout.read(&mut buf) {
            if n == 0 {
                break;
            }
            let _ = out.write_all(&buf[..n]);
            let _ = out.flush();
        }
    });
    std::thread::spawn(move || {
        use std::io::{Read, Write as _};
        let mut buf = [0u8; 4096];
        let mut err = std::io::stderr();
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            let _ = err.write_all(&buf[..n]);
        }
    });

    // Watchers feeding the stop channel: Ctrl-C always; our own stdin
    // reaching EOF only when stdin IS a terminal (the console closed /
    // Ctrl-Z). A pipe or null stdin — CI, `dev < NUL`, tool harnesses —
    // hits EOF instantly and must not be mistaken for a shutdown request;
    // those contexts use --duration or Ctrl-C.
    let (tx, rx) = std::sync::mpsc::channel::<DevEvent>();
    {
        use std::io::IsTerminal as _;
        if std::io::stdin().is_terminal() {
            let tx = tx.clone();
            std::thread::spawn(move || {
                use std::io::Read;
                let mut buf = [0u8; 64];
                let mut stdin = std::io::stdin();
                loop {
                    match stdin.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                }
                let _ = tx.send(DevEvent::StdinEof);
            });
        }
    }
    {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            if let Ok(rt) = rt {
                rt.block_on(async {
                    if tokio::signal::ctrl_c().await.is_ok() {
                        let _ = tx.send(DevEvent::CtrlC);
                    }
                });
            }
        });
    }
    drop(tx); // main loop holds the last receiver

    // ---- Readiness: poll /api/health, never a fixed startup sleep ----
    let started = std::time::Instant::now();
    let timeout = std::time::Duration::from_millis(dev.readiness_timeout_ms);
    let ready = loop {
        if health_once(dev.port) {
            break true;
        }
        if let Some(code) = child.try_wait().ok().flatten() {
            logger.error(&format!("server exited before readiness (status {code:?})"));
            return Ok(3);
        }
        if started.elapsed() >= timeout {
            break false;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    if !ready {
        let msg = format!(
            "readiness timeout: /api/health on port {} did not answer within {} ms",
            dev.port, dev.readiness_timeout_ms
        );
        logger.error(&msg);
        let _ = child.kill();
        let _ = child.wait();
        return Ok(3);
    }
    let elapsed_ms = started.elapsed().as_millis() as u64;
    logger.log(apexsim::logging::Level::Info, None, None, &format!(
        "ready: dashboard at http://127.0.0.1:{}/ (ready in {elapsed_ms} ms)", dev.port
    ));
    println!(
        "dashboard ready: http://127.0.0.1:{}/ (ready in {elapsed_ms} ms, server pid {child_id})",
        dev.port
    );

    // ---- Run until duration / Ctrl-C / stdin EOF / child death ----
    let deadline = duration.map(|s| std::time::Instant::now() + std::time::Duration::from_secs_f64(s));
    let stop_reason = loop {
        // Poll in short slices so a crashed child is noticed promptly even
        // without a duration cap.
        let now = std::time::Instant::now();
        if let Some(d) = deadline {
            if now >= d {
                break "duration elapsed";
            }
        }
        match rx.recv_timeout(std::time::Duration::from_millis(250)) {
            Ok(ev) => match ev {
                DevEvent::StdinEof => break "stdin closed",
                DevEvent::CtrlC => break "ctrl-c",
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break "internal",
        }
        if let Some(status) = child.try_wait().ok().flatten() {
            // With a duration cap the child is TOLD to exit at that time —
            // a clean exit there is the planned shutdown, not a crash.
            if status.success() && deadline.is_some() {
                break "server finished its run";
            }
            logger.error(&format!(
                "server exited unexpectedly (status {})",
                status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into())
            ));
            return Ok(3);
        }
    };

    // ---- Shutdown: stop only the processes this run started ----
    // The child was given the same --duration, so it usually exits on its
    // own, gracefully. Allow it a grace window; only then terminate it.
    logger.log(apexsim::logging::Level::Info, None, None, &format!("shutdown ({stop_reason})"));
    println!("shutting down ({stop_reason})…");
    let grace = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {
                if std::time::Instant::now() >= grace {
                    let _ = child.kill();
                    break child.wait().ok();
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => {
                logger.warn(&format!("wait on server failed: {e}"));
                break None;
            }
        }
    };
    let clean = status.map(|s| s.success()).unwrap_or(false);
    logger.log(apexsim::logging::Level::Info, None, None, &format!(
        "run complete (server status {})",
        status.map(|s| s.code().unwrap_or(-1)).map(|c| c.to_string()).unwrap_or_else(|| "signal".into())
    ));
    println!("run {run_id} complete");
    Ok(if clean { 0 } else { 3 })
}

fn config_path_arg(args: &[String]) -> Result<Option<PathBuf>, String> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--config" {
            let v = it.next().ok_or_else(|| "--config requires a value".to_string())?;
            return Ok(Some(PathBuf::from(v)));
        }
    }
    Ok(None)
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// One blocking health probe (std TCP, no HTTP client dependency). Bounded:
/// a port that accepts but never answers (e.g. a non-apexsim listener) must
/// read as "not ready", not hang the launcher.
fn health_once(port: u16) -> bool {
    use std::io::{Read, Write};
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut s) = std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(500))
    else {
        return false;
    };
    let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
    let req = format!("GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if s.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = Vec::new();
    match s.read_to_end(&mut buf) {
        Ok(_) => String::from_utf8_lossy(&buf).contains("\"ok\":true"),
        Err(_) => false,
    }
}

fn cmd_validate(args: &[String]) -> Result<bool, String> {
    // --replay-case FILE: re-execute one stored case and verify its hash.
    if let Some(pos) = args.iter().position(|a| a == "--replay-case") {
        let path = args
            .get(pos + 1)
            .ok_or_else(|| "--replay-case requires a file".to_string())?;
        let ok = apexsim::validation::replay_case(std::path::Path::new(path))?;
        println!("case replay: {}", if ok { "PASS — trajectory hash reproduced" } else { "FAIL — trajectory diverged" });
        return Ok(ok);
    }
    let mut out_dir = std::path::PathBuf::from("artifacts/validation");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out_dir = std::path::PathBuf::from(
                    args.get(i + 1).ok_or("--out requires a value")?,
                );
                i += 2;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }

    let p = apexsim::physics::CarParams::default();
    let report = apexsim::validation::run_validation();
    let text = apexsim::validation::render_report(&report, &p);

    let cases_dir = out_dir.join("cases");
    std::fs::create_dir_all(&cases_dir).map_err(|e| format!("create {}: {e}", cases_dir.display()))?;
    for case in &report.cases {
        let path = cases_dir.join(format!("{}_{}hz.casefile", case.name, case.rate_hz));
        apexsim::validation::write_case(&path, case, &p)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    let report_path = out_dir.join("report.md");
    std::fs::write(&report_path, &text).map_err(|e| format!("write report: {e}"))?;

    // Compact console summary (full detail in the report file).
    println!("== apexsim Phase 3 validation | model {} ==", apexsim::validation::MODEL_VERSION);
    println!("all tolerances pass at 250 Hz: {}", if report.all_pass_at_250 { "YES" } else { "NO" });
    println!("lowest passing rate: {} Hz (engine retains 250 Hz)", report.recommended_rate_hz);
    println!("report: {}", report_path.display());
    println!("cases:  {} files in {}", report.cases.len(), cases_dir.display());
    Ok(report.all_pass_at_250)
}

/// Phase 11 — the human-review display.
///
/// Prints the incident record as a review sheet: the rule that was applied, in
/// words; the evidence it was allowed to look at; the decision; and who serves
/// what. The counts come first so a reader sees the shape of the race before
/// the detail, and the disclaimer is printed unconditionally — this is a
/// stewarding aid inside a simulation, not a rules authority, and a report
/// that omitted that would be claiming more than the rules can support.
fn print_steward_record(outcome: &apexsim::race::RaceOutcome, verbose: bool) {
    use apexsim::steward;
    let Some(summary) = outcome.steward.as_ref() else { return };
    let cfg = outcome.steward_config.unwrap_or_default();
    let incidents = outcome.steward_incidents.as_deref().unwrap_or(&[]);
    println!(
        "\n== stewarding aid | rule set {} v{} | {} finding(s) ==",
        summary.rule_set,
        summary.rule_set_version,
        summary.incidents
    );
    println!(
        "   this is an in-simulation stewarding aid, not an official rules authority; it records \
         what the rules below say, not what a human steward would decide"
    );
    println!(
        "   penalised: {} | no penalty: {} | inconclusive: {} | time served: {:.1} s",
        summary.penalised, summary.no_penalty, summary.inconclusive, summary.time_penalty_seconds
    );
    if !summary.by_car.is_empty() {
        let ledger: Vec<String> = summary
            .by_car
            .iter()
            .map(|e| {
                format!(
                    "car {}: +{:.1} s over {} finding(s) ({} penalised, {} inconclusive)",
                    e.car, e.seconds, e.incidents, e.penalised, e.inconclusive
                )
            })
            .collect();
        println!("   penalty ledger: {}", ledger.join(" | "));
    }
    for rule in steward::Rule::ALL {
        let n = incidents.iter().filter(|i| i.rule == rule).count();
        println!(
            "\n   rule {} v{} — {} ({} case(s) this race)",
            rule.id(),
            rule.version(),
            rule.title(),
            n
        );
        for line in wrap(&rule.statement(&cfg), 88) {
            println!("     {line}");
        }
    }
    if summary.inconclusive > 0 {
        println!(
            "\n   {} case(s) were left INCONCLUSIVE on purpose: the recorded evidence does not \
             separate the cars, and this aid does not turn that into a fault finding.",
            summary.inconclusive
        );
    }
    if verbose {
        for inc in incidents {
            print!("{}", steward::render_incident(inc, &cfg));
        }
    }
    // The classification is what the cars did on track. The penalty is a
    // separate record about it, so it is presented as a second ordering rather
    // than folded into the result — the two must never be confused.
    if summary.time_penalty_seconds > 0.0 {
        let order = summary.post_penalty_order(&outcome.classification);
        let list: Vec<String> = order.iter().map(|c| c.to_string()).collect();
        let served: Vec<String> = summary
            .by_car
            .iter()
            .filter(|e| e.seconds > 0.0)
            .map(|e| format!("car {} +{:.0} s", e.car, e.seconds))
            .collect();
        println!("\n   on-track classification: [{}]", list.join(", "));
        println!("   served by:               {}", served.join(", "));
        println!(
            "   (the penalty is a separate record, reported alongside the result rather than \
             folded into it)"
        );
        let _ = order;
    }
}

/// Greedy wrap, so a long rule statement stays readable in a terminal.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if cur.is_empty() {
            cur.push_str(word);
        } else if cur.len() + 1 + word.len() <= width {
            cur.push(' ');
            cur.push_str(word);
        } else {
            lines.push(std::mem::take(&mut cur));
            cur.push_str(word);
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// `apexsim steward <log>` — replay a run log and print its incident record for
/// review. Replaying rather than re-simulating is deliberate: the record comes
/// out of the same file the race wrote, so a reviewer reads exactly what was
/// recorded, and a log whose record cannot be reproduced fails here instead of
/// quietly producing a different one.
fn cmd_steward(args: &[String]) -> Result<bool, String> {
    let mut path: Option<std::path::PathBuf> = None;
    let mut verbose = false;
    for a in args {
        match a.as_str() {
            "--verbose" | "-v" => verbose = true,
            other => {
                if path.is_some() {
                    return Err(format!("unexpected argument {other:?}"));
                }
                path = Some(std::path::PathBuf::from(other));
            }
        }
    }
    let path = path.ok_or("usage: apexsim steward <race-log> [--verbose]")?;
    let (ok, outcome) = apexsim::race::replay_race_log(&path)?;
    println!(
        "== steward review | {} | replay {} ==",
        path.display(),
        if ok { "bit-identical" } else { "DIVERGED" }
    );
    if !ok {
        println!("  the trajectory itself did not reproduce; the record below is not trustworthy");
    }
    if outcome.steward_config.is_none() {
        println!(
            "\nno incident record in this log: it was written by a run without the stewarding \
             aid ({}).\nRe-run that race with --steward to produce one.",
            path.display()
        );
        return Ok(ok);
    }
    // A short record is shown in full by default — a reviewer who asked for a
    // review sheet should not also have to ask twice to see it.
    let n = outcome.steward_incidents.as_ref().map_or(0, |v| v.len());
    print_steward_record(&outcome, verbose || n <= 12);
    Ok(ok)
}

fn cmd_replay(args: &[String]) -> Result<bool, String> {
    let path = args
        .first()
        .ok_or_else(|| "replay requires a log path".to_string())?
        .clone();
    // Multi-car race logs (v2/v3) carry their own format marker.
    let head = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {path}: {e}"))?
        .lines()
        .take(2)
        .collect::<Vec<_>>()
        .join("\n");
    if head.contains("simlog v") {
        let (ok, outcome) = apexsim::race::replay_race_log(std::path::Path::new(&path))?;
        println!(
            "== race replay | {} cars | ticks {} | hash {:016x} ==",
            outcome.cars, outcome.ticks, outcome.state_hash
        );
        println!("  RESULT: {}", if ok { "PASS — whole-field replay is bit-identical" } else { "FAIL — trajectory diverged" });
        return Ok(ok);
    }
    let loaded = read_log(std::path::Path::new(&path))?;
    println!(
        "== replay | seed {} | laps {} | inputs {} ==",
        loaded.config.seed, loaded.config.laps, loaded.inputs.len()
    );
    let outcome = replay_session(loaded.config.clone(), &loaded.inputs);
    let exp = &loaded.expected;

    let hash_ok = outcome.state_hash == exp.state_hash;
    let ticks_ok = outcome.ticks == exp.ticks;
    let laps_ok = outcome.laps_done == exp.laps_done;
    let laptimes_ok = outcome.lap_times.len() == exp.lap_times.len()
        && outcome
            .lap_times
            .iter()
            .zip(exp.lap_times.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9);

    println!("  ticks:   {} (expected {})", outcome.ticks, exp.ticks);
    println!("  laps:    {} (expected {})", outcome.laps_done, exp.laps_done);
    println!("  hash:    {:016x} (expected {:016x})", outcome.state_hash, exp.state_hash);
    let all_ok = hash_ok && ticks_ok && laps_ok && laptimes_ok;
    println!("  RESULT: {}", if all_ok { "PASS — replay is bit-identical" } else { "FAIL — trajectory diverged" });
    Ok(all_ok)
}
// ---------------------------------------------------------------------------
// Phase 10 — headless tournaments and deployment
// ---------------------------------------------------------------------------

/// `apexsim tournament` — run a saved batch of races and score it.
///
/// Modes:
///   * default        run the batch in `--spec` (or a built-in demo) and score it
///   * `--write-spec` emit a spec template to a file and stop
///   * `--replay DIR` re-run a saved batch and compare it against its report
///
/// Flags mirror the plan's build items: explicit seeds and policies
/// (`--seeds`, `--policies`), a pinned track version (in the spec), the three
/// deployment profiles (`--profile`), and the concurrency measurement the plan
/// requires before parallelising (`--measure N`).
fn cmd_tournament(args: &[String]) -> Result<bool, String> {
    use apexsim::tournament as T;

    let mut spec_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut replay_dir: Option<PathBuf> = None;
    let mut write_spec: Option<PathBuf> = None;
    let mut seeds: Option<Vec<u64>> = None;
    let mut policies: Vec<String> = Vec::new();
    let mut cars: Option<u32> = None;
    let mut laps: Option<u32> = None;
    let mut mode: Option<String> = None;
    let mut weather: Option<Option<String>> = None;
    let mut profile = T::Profile::EngineOnly;
    let mut threads = 1usize;
    let mut measure: Option<usize> = None;
    let mut logs = false;
    let mut steward = false;
    let mut port: Option<u16> = None;
    let mut quiet = false;

    let mut i = 0;
    let need = |name: &str, val: Option<&String>| -> Result<String, String> {
        val.ok_or_else(|| format!("{name} requires a value")).cloned()
    };
    while i < args.len() {
        match args[i].as_str() {
            "--spec" => {
                spec_path = Some(PathBuf::from(need("--spec", args.get(i + 1))?));
                i += 2;
            }
            "--out" => {
                out_dir = Some(PathBuf::from(need("--out", args.get(i + 1))?));
                i += 2;
            }
            "--replay" => {
                replay_dir = Some(PathBuf::from(need("--replay", args.get(i + 1))?));
                i += 2;
            }
            "--write-spec" => {
                write_spec = Some(PathBuf::from(need("--write-spec", args.get(i + 1))?));
                i += 2;
            }
            "--seeds" => {
                let raw = need("--seeds", args.get(i + 1))?;
                let mut v = Vec::new();
                for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    v.push(part.parse::<u64>().map_err(|e| format!("--seeds '{part}': {e}"))?);
                }
                if v.is_empty() {
                    return Err("--seeds needs at least one seed".into());
                }
                seeds = Some(v);
                i += 2;
            }
            "--policies" => {
                let raw = need("--policies", args.get(i + 1))?;
                for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    policy_from_name(part)?;
                    policies.push(part.to_string());
                }
                i += 2;
            }
            "--cars" => {
                cars = Some(need("--cars", args.get(i + 1))?.parse().map_err(|e| format!("--cars: {e}"))?);
                i += 2;
            }
            "--laps" => {
                laps = Some(need("--laps", args.get(i + 1))?.parse().map_err(|e| format!("--laps: {e}"))?);
                i += 2;
            }
            "--mode" => {
                mode = Some(need("--mode", args.get(i + 1))?);
                i += 2;
            }
            "--weather" => {
                weather = Some(Some(need("--weather", args.get(i + 1))?));
                i += 2;
            }
            "--no-weather" => {
                weather = Some(None);
                i += 1;
            }
            "--profile" => {
                profile = T::Profile::parse(&need("--profile", args.get(i + 1))?)
                    .ok_or("--profile must be engine-only, engine+dashboard, or engine+ollama")?;
                i += 2;
            }
            "--threads" => {
                threads = need("--threads", args.get(i + 1))?.parse().map_err(|e| format!("--threads: {e}"))?;
                i += 2;
            }
            "--measure" => {
                measure = Some(need("--measure", args.get(i + 1))?.parse().map_err(|e| format!("--measure: {e}"))?);
                i += 2;
            }
            "--steward" => {
                steward = true;
                i += 1;
            }
            "--logs" => {
                logs = true;
                i += 1;
            }
            "--port" => {
                port = Some(need("--port", args.get(i + 1))?.parse().map_err(|e| format!("--port: {e}"))?);
                i += 2;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    if threads == 0 {
        return Err("--threads must be >= 1".into());
    }

    // ---- --write-spec: emit a template and stop ---------------------------
    if let Some(path) = write_spec {
        let spec = T::BatchSpec::demo();
        spec.save(&path)?;
        println!("wrote spec template {}", path.display());
        return Ok(true);
    }

    // ---- load or build the spec ------------------------------------------
    let mut spec: T::BatchSpec = match (&spec_path, &replay_dir) {
        (Some(p), _) => T::BatchSpec::load(p)?,
        (None, Some(d)) => T::BatchSpec::load(&d.join("spec.json"))?,
        (None, None) => T::BatchSpec::demo(),
    };
    // Command-line overrides refine the loaded spec without mutating the file.
    if let Some(c) = cars {
        spec.session.cars = c;
    }
    if let Some(l) = laps {
        spec.session.laps = l;
    }
    if let Some(m) = mode {
        spec.session.mode = m;
    }
    if let Some(w) = weather {
        spec.session.weather = w;
    }
    if steward {
        spec.session.steward = true;
    }
    if let Some(s) = seeds {
        spec.runs = expand_runs(&s, &policies, spec.session.laps)?;
    } else if !policies.is_empty() {
        let existing: Vec<u64> = spec.runs.iter().map(|r| r.seed).collect();
        spec.runs = expand_runs(&existing, &policies, spec.session.laps)?;
    }

    // ---- concurrency measurement, before any parallel run -----------------
    if let Some(n) = measure {
        println!("== concurrency measurement (serial vs {n} workers) ==");
        let m = T::measure_concurrency(&spec, n)?;
        println!("  serial:   {:.2}s", m.serial_wall_s);
        println!("  parallel: {:.2}s", m.parallel_wall_s);
        println!("  speedup:  {:.2}x", m.speedup);
        println!("  results identical: {}", m.results_identical);
        println!(
            "  -- verdict -- {}",
            if m.results_identical {
                "PARALLEL SAFE on this machine"
            } else {
                "PARALLEL UNSAFE: results differ"
            }
        );
        if !m.results_identical {
            return Ok(false);
        }
    }

    // ---- run the batch, optionally behind the dashboard ------------------
    // One execution, whatever the profile and worker count. The report that is
    // printed is the report that is saved, so its wall time describes the batch
    // the user actually asked for rather than a second, serial re-run of it.
    let artifacts = match &out_dir {
        Some(d) => {
            let (r, a) = if profile == T::Profile::EnginePlusDashboard {
                let (r, _) = run_batch_behind_dashboard(&spec, port, quiet)?;
                let a = T::save_batch(&spec, &r, d, logs)?;
                (r, Some(a))
            } else {
                let (r, a) = T::run_and_save(&spec, d, profile, threads, logs)?;
                (r, Some(a))
            };
            (r, a)
        }
        None => {
            let runner = T::BatchRunner::new(spec.clone()).with_profile(profile);
            let r = if threads > 1 {
                runner.run_parallel(threads)?
            } else {
                runner.run()?
            };
            (r, None)
        }
    };
    let report = artifacts.0;

    print_batch_report(&report, quiet);

    // ---- say where the batch was written ---------------------------------
    if let Some(art) = &artifacts.1 {
        println!("\nwrote {}", art.spec_path.display());
        println!("wrote {}", art.report_path.display());
        if !art.logs.is_empty() {
            println!(
                "wrote {} per-run log(s) under {}",
                art.logs.len(),
                art.dir.join("logs").display()
            );
        }
    }

    // ---- replay: compare against a saved report ---------------------------
    if let Some(d) = replay_dir {
        let saved = T::BatchReport::load(&d.join("report.json"))?;
        let cmp = T::compare(&saved, &report);
        println!("\n== comparison against {} ==", d.display());
        println!("  spec match:    {}", cmp.spec_match);
        println!("  engine match:  {}", cmp.engine_match);
        println!("  platform:      {}", cmp.platform_match_label());
        println!("  track match:   {}", cmp.track_match);
        println!("  profile match: {}", cmp.profile_match);
        println!("  runs compared: {}", cmp.runs_compared);
        println!("  scores agree:  {}", cmp.metrics_agree);
        println!("  RESULT: {}", cmp.verdict());
        return Ok(cmp.identical);
    }

    // Non-zero when anything failed: a batch with a failed run is a failed
    // batch, and the exit code is the machine-readable version of that.
    Ok(report.summary.runs_failed == 0
        && report.summary.runs_finished == report.summary.runs_planned)
}

/// Resolve a policy name to a [`apexsim::tournament::Policy`].
fn policy_from_name(name: &str) -> Result<apexsim::tournament::Policy, String> {
    use apexsim::tournament::{PitStrategySpec, Policy, TireCompoundSpec as C};
    let (compound, strategy, advisor) = match name.trim().to_ascii_lowercase().as_str() {
        "soft" | "one-stop-soft" => (C::Soft, PitStrategySpec::Auto, false),
        "medium" | "one-stop-medium" => (C::Medium, PitStrategySpec::Auto, false),
        "hard" | "one-stop-hard" => (C::Hard, PitStrategySpec::Auto, false),
        "no-stop" | "nostop" => (C::Soft, PitStrategySpec::None, false),
        "advised" | "soft-advised" => (C::Soft, PitStrategySpec::Auto, true),
        other => {
            return Err(format!("unknown policy '{other}' (use soft|medium|hard|no-stop|advised)"))
        }
    };
    Ok(Policy { start_compound: compound, strategy, advisor })
}

/// The cross product of seeds x policies, seed-major so a batch can be
/// extended with seeds without reshuffling the runs that came before.
fn expand_runs(
    seeds: &[u64],
    policies: &[String],
    laps: u32,
) -> Result<Vec<apexsim::tournament::RunSpec>, String> {
    use apexsim::tournament::{PitStrategySpec, Policy, RunSpec};
    if policies.is_empty() {
        return Ok(seeds
            .iter()
            .map(|&seed| RunSpec { seed, label: format!("seed{seed}"), policy: None })
            .collect());
    }
    let built: Vec<Policy> =
        policies.iter().map(|p| policy_from_name(p)).collect::<Result<_, _>>()?;
    let mut out = Vec::new();
    for &seed in seeds {
        for (pi, p) in built.iter().enumerate() {
            // `Auto` scales with the race length; on a 1-lap race it has no
            // legal stop lap, so it degrades to no-stop rather than producing
            // a spec that `validate` would reject.
            let strategy = match &p.strategy {
                PitStrategySpec::Auto if laps <= 1 => PitStrategySpec::None,
                s => s.clone(),
            };
            out.push(RunSpec {
                seed,
                label: format!("seed{seed}/{}", policies[pi]),
                policy: Some(Policy { strategy, ..p.clone() }),
            });
        }
    }
    Ok(out)
}

/// Run a batch on a worker thread with the dashboard serving alongside it.
///
/// This is the plan's `engine+dashboard` profile: batch and server share the
/// machine, and the server is the batch's window — `GET /api/batch` shows
/// progress, `POST /api/batch/stop` asks it to stop. The stop flag the batch
/// checks is the server's own, so a stop request actually reaches the batch.
fn run_batch_behind_dashboard(
    spec: &apexsim::tournament::BatchSpec,
    port: Option<u16>,
    quiet: bool,
) -> Result<(apexsim::tournament::BatchReport, Option<u16>), String> {
    use apexsim::tournament::{BatchRunner, Profile};
    use std::sync::mpsc;
    use std::sync::atomic::Ordering;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| format!("runtime: {e}"))?;
    // The dashboard serves an idle single-car session; the interesting race is
    // the batch, observed through /api/batch.
    let sim_cfg = apexsim::sim::SimConfig {
        cars: 1,
        laps: spec.session.laps.max(1),
        ..Default::default()
    };
    let (addr, handle, progress) = {
        let _guard = rt.enter();
        apexsim::server::spawn_server_with_batch(sim_cfg, port.unwrap_or(0), false)
            .map_err(|e| format!("bind dashboard: {e}"))?
    };
    println!("dashboard on http://{addr}  (GET /api/batch, POST /api/batch/stop)");

    let total = spec.runs.len();
    let total_c = total;
    let quiet2 = quiet;
    let prog = progress.clone();
    let runner = BatchRunner::new(spec.clone())
        .with_profile(Profile::EnginePlusDashboard)
        .with_progress(move |done, total, label| {
            prog.progress(done, total, label);
            if !quiet2 {
                println!("  [{done}/{total}] {label}");
            }
        });
    // The batch's stop flag IS the server's flag object, so a POST to
    // /api/batch/stop reaches the batch without a second channel.
    let runner = runner.with_stop_flag(progress.stop_flag());
    progress.begin(&spec.name, total);

    let (tx, rx) =
        mpsc::channel::<Result<apexsim::tournament::BatchReport, String>>();
    let worker = std::thread::Builder::new()
        .name("batch".into())
        .spawn(move || {
            let _ = tx.send(runner.run());
        })
        .map_err(|e| format!("spawn batch thread: {e}"))?;
    let report = rx
        .recv()
        .map_err(|_| "batch thread died".to_string())??;
    progress.end();
    handle.abort();
    let _ = worker.join();
    let _ = Ordering::SeqCst;
    drop(rt);
    let _ = total_c;
    Ok((report, Some(addr.port())))
}

fn print_batch_report(r: &apexsim::tournament::BatchReport, quiet: bool) {
    let s = &r.summary;
    if !quiet {
        println!("\n== batch '{}' ==", r.spec.name);
        println!("  engine:  {}", r.engine);
        println!("  profile: {}", r.profile.label());
        println!("  track:   {}", r.spec.track.label());
        println!(
            "  session: {} cars x {} laps, {}",
            r.spec.session.cars, r.spec.session.laps, r.spec.session.mode
        );
        println!(
            "  runs:    {} planned, {} finished, {} failed",
            s.runs_planned, s.runs_finished, s.runs_failed
        );
        for run in &r.runs {
            println!(
                "    {} [{}] finishers {}/{}  contacts {}  cv {}  hash {}",
                run.label,
                run.status.label(),
                run.finishers,
                run.cars,
                run.contacts,
                run.leader_lap_cv
                    .map(|c| format!("{c:.2}%"))
                    .unwrap_or_else(|| "-".into()),
                run.state_hash
                    .map(|h| format!("{h:016x}"))
                    .unwrap_or_else(|| "-".into()),
            );
        }
        println!("-- scoring --");
        match s.finish_rate {
            Some(m) => println!("  finish rate:        {:.1}% (n={})", m.value * 100.0, m.n),
            None => println!("  finish rate:        - (no finished runs)"),
        }
        match s.mean_leader_lap_cv_pct {
            Some(m) => println!("  leader lap CV:      {:.2}% (n={})", m.value, m.n),
            // The reason matters. A blank here is either "nothing finished",
            // which is a failure to report, or "every run was one lap long",
            // which is a batch that cannot answer the question at all. Saying
            // "no finished runs" for the second case states something false
            // about a batch that in fact ran clean.
            None if s.runs_finished == 0 => println!("  leader lap CV:      - (no finished runs)"),
            None => println!(
                "  leader lap CV:      - (n=0: every run finished on 1 lap, a consistency \
                 figure needs 2+)"
            ),
        }
        match s.mean_contacts_per_run {
            Some(m) => println!("  contacts/run:       {:.2} (n={})", m.value, m.n),
            None => println!("  contacts/run:       -"),
        }
        match s.mean_retirements_per_run {
            Some(m) => println!("  retirements/run:    {:.2} (n={})", m.value, m.n),
            None => println!("  retirements/run:    -"),
        }
        match s.mean_race_time_s {
            Some(m) => println!("  mean race time:     {:.1}s (n={})", m.value, m.n),
            None => println!("  mean race time:     -"),
        }
        match s.mean_pit_stops_per_run {
            Some(m) => println!("  pit stops/run:      {:.2} (n={})", m.value, m.n),
            None => println!("  pit stops/run:      -"),
        }
        match s.mean_pit_held_s {
            Some(m) => println!("  mean pit held:      {:.2}s (n={})", m.value, m.n),
            None => println!("  mean pit held:      - (no stops)"),
        }
        // Printed only when the aid ran, so a batch without it never shows a
        // row of zeroes that could be read as "nothing happened".
        if s.mean_incidents_per_run.is_some() {
            let line = |m: &apexsim::tournament::Mean, unit: &str| format!("{:.2}{} (n={})", m.value, unit, m.n);
            match (&s.mean_incidents_per_run, &s.mean_penalised_per_run) {
                (Some(i), Some(p)) => println!(
                    "  incidents/run:      {}  ->  penalised {}, inconclusive {}, served {}",
                    line(i, ""),
                    line(p, ""),
                    s.mean_inconclusive_per_run.map_or_else(|| "-".into(), |m| line(&m, "")),
                    s.mean_steward_time_per_run.map_or_else(|| "-".into(), |m| format!("{:.2}s (n={})", m.value, m.n)),
                ),
                _ => println!("  incidents/run:      -"),
            }
        }
        if !s.compounds_fitted.is_empty() {
            let fitted: Vec<String> =
                s.compounds_fitted.iter().map(|(c, n)| format!("{c}x{n}")).collect();
            println!("  compounds fitted:   {}", fitted.join(" "));
        }
        println!(
            "  wall:               {:.2}s for {} race(s) ({:.2}/s)",
            s.wall_time_s,
            r.runs.len(),
            s.wall_races_per_s
        );
    }
    println!("-- verdict -- {}", s.claim_note());
    if s.runs_failed > 0 {
        println!("FAILURES: {} run(s) did not finish cleanly", s.runs_failed);
    }
}