//! Run-log serialization for the replay gate.
//!
//! Format is a plain numeric line format (not JSON) so both writer and
//! reader are trivially exact — no float formatting/parsing ambiguity:
//!
//!   # comment header
//!   H <seed> <laps> <dt_us> <driver_every> <realtime> <car:17> <driver:16> <track:3>
//!   I <tick> <throttle> <brake> <steer_cmd>          (one per 50 Hz boundary)
//!   R <ticks> <laps_done> <state_hash> <offtrack> <lap_times...>

use crate::physics::CarParams;
use crate::sim::{SimConfig, SimOutcome, DT, DRIVER_EVERY};
use crate::driver::DriverParams;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordedInput {
    pub tick: u64,
    pub throttle: f64,
    pub brake: f64,
    pub steer_cmd: f64,
}

fn car_to_vec(p: &CarParams) -> Vec<f64> {
    vec![
        p.mass, p.inertia_z, p.wheelbase, p.cg_to_front, p.cg_to_rear, p.cg_height,
        p.tire_grip, p.corner_stiffness_front, p.corner_stiffness_rear,
        p.engine_force_max, p.engine_power_max, p.brake_force_max, p.brake_bias_front,
        p.drag_area, p.roll_resistance, p.max_steer, p.air_density,
    ]
}

fn car_from_vec(v: &[f64]) -> Option<CarParams> {
    if v.len() != 17 {
        return None;
    }
    Some(CarParams {
        mass: v[0], inertia_z: v[1], wheelbase: v[2], cg_to_front: v[3], cg_to_rear: v[4],
        cg_height: v[5], tire_grip: v[6], corner_stiffness_front: v[7],
        corner_stiffness_rear: v[8], engine_force_max: v[9], engine_power_max: v[10],
        brake_force_max: v[11], brake_bias_front: v[12], drag_area: v[13],
        roll_resistance: v[14], max_steer: v[15], air_density: v[16],
    })
}

fn driver_to_vec(p: &DriverParams) -> Vec<f64> {
    vec![
        p.wheelbase, p.lookahead_gain, p.lookahead_base, p.lookahead_min, p.lookahead_max,
        p.a_lat_max, p.a_brake_anticipate, p.v_straight_max, p.horizon_margin,
        p.speed_margin, p.throttle_gain, p.brake_gain, p.deadband, p.steer_rate_limit,
        p.max_steer, p.noise_amplitude,
    ]
}

fn driver_from_vec(v: &[f64]) -> Option<DriverParams> {
    if v.len() != 16 {
        return None;
    }
    Some(DriverParams {
        wheelbase: v[0], lookahead_gain: v[1], lookahead_base: v[2], lookahead_min: v[3],
        lookahead_max: v[4], a_lat_max: v[5], a_brake_anticipate: v[6], v_straight_max: v[7],
        horizon_margin: v[8], speed_margin: v[9], throttle_gain: v[10], brake_gain: v[11],
        deadband: v[12], steer_rate_limit: v[13], max_steer: v[14], noise_amplitude: v[15],
    })
}

pub fn write_log(
    path: &Path,
    config: &SimConfig,
    inputs: &[RecordedInput],
    outcome: &SimOutcome,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "# apexsim run log v1 (H=header I=input@50Hz R=result)")?;
    let dt_us = (DT * 1e6) as u64;
    let realtime = if config.realtime { 1 } else { 0 };
    let mut fields: Vec<String> = vec![
        config.seed.to_string(),
        config.laps.to_string(),
        dt_us.to_string(),
        DRIVER_EVERY.to_string(),
        realtime.to_string(),
    ];
    fields.extend(car_to_vec(&config.car).iter().map(|f| format!("{f:e}")));
    fields.extend(driver_to_vec(&config.driver).iter().map(|f| format!("{f:e}")));
    fields.extend([
        config.track_straight,
        config.track_radius,
        config.track_half_width,
    ].iter().map(|f| format!("{f:e}")));
    writeln!(w, "H {}", fields.join(" "))?;
    for i in inputs {
        writeln!(
            w,
            "I {} {:.17e} {:.17e} {:.17e}",
            i.tick, i.throttle, i.brake, i.steer_cmd
        )?;
    }
    let mut r = vec![
        outcome.ticks.to_string(),
        outcome.laps_done.to_string(),
        format!("{:016x}", outcome.state_hash),
        outcome.offtrack_ticks.to_string(),
    ];
    r.extend(outcome.lap_times.iter().map(|t| format!("{t:e}")));
    writeln!(w, "R {}", r.join(" "))?;
    w.flush()?;
    Ok(())
}

pub struct LoadedLog {
    pub config: SimConfig,
    pub inputs: Vec<RecordedInput>,
    pub expected: SimOutcomeLite,
}

/// The parts of the original outcome needed to verify a replay.
#[derive(Clone, Debug)]
pub struct SimOutcomeLite {
    pub ticks: u64,
    pub laps_done: u32,
    pub state_hash: u64,
    pub offtrack_ticks: u64,
    pub lap_times: Vec<f64>,
}

pub fn read_log(path: &Path) -> Result<LoadedLog, String> {
    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    let reader = BufReader::new(file);
    let mut config: Option<SimConfig> = None;
    let mut inputs = Vec::new();
    let mut expected: Option<SimOutcomeLite> = None;

    for (n, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read line {n}: {e}"))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tok = line.split_whitespace();
        let kind = tok.next().ok_or(format!("line {n}: empty"))?;
        match kind {
            "H" => {
                let vals: Vec<f64> = tok
                    .map(|t| t.parse::<f64>().map_err(|e| format!("line {n}: bad number {t}: {e}")))
                    .collect::<Result<_, _>>()?;
                if vals.len() != 5 + 17 + 16 + 3 {
                    return Err(format!("line {n}: header has {} fields, expected 41", vals.len()));
                }
                let seed = vals[0] as u64;
                let laps = vals[1] as u32;
                let dt = vals[2] / 1e6;
                let driver_every = vals[3] as u32;
                let realtime = vals[4] != 0.0;
                if (dt - DT).abs() > 1e-12 {
                    return Err(format!("line {n}: dt {dt}s does not match engine DT {DT}s"));
                }
                if driver_every != DRIVER_EVERY {
                    return Err(format!("line {n}: driver_every {driver_every} mismatch"));
                }
                config = Some(SimConfig {
                    seed,
                    laps,
                    cars: 1, // v1 logs are single-car (Phase 4 races use simlog v2)
                    mode: "race".into(), // v1 logs predate session modes
                    field: crate::race::FieldMode::Classic, // and team fields
                    realism: None, // and the pack (v1 logs predate both)
                    weekend_quali: false, // and weekend mode
                    quali_window_s: 240.0,
                    realtime,
                    car: car_from_vec(&vals[5..22]).ok_or("car params wrong length")?,
                    driver: driver_from_vec(&vals[22..38]).ok_or("driver params wrong length")?,
                    track_straight: vals[38],
                    track_radius: vals[39],
                    track_half_width: vals[40],
                    track_custom: None, // v1 logs are always the stadium
                    weather: None,
                    telemetry: None,
                    sound: None,
                });
            }
            "I" => {
                let tick: u64 = tok.next().ok_or(format!("line {n}: missing tick"))?
                    .parse().map_err(|e| format!("line {n}: tick: {e}"))?;
                let f = |tok: &mut std::str::SplitWhitespace| -> Result<f64, String> {
                    tok.next().ok_or(format!("line {n}: missing field"))?
                        .parse().map_err(|e| format!("line {n}: field: {e}"))
                };
                inputs.push(RecordedInput {
                    tick,
                    throttle: f(&mut tok)?,
                    brake: f(&mut tok)?,
                    steer_cmd: f(&mut tok)?,
                });
            }
            "R" => {
                let ticks: u64 = tok.next().ok_or(format!("line {n}: missing ticks"))?
                    .parse().map_err(|e| format!("line {n}: ticks: {e}"))?;
                let laps_done: u32 = tok.next().ok_or(format!("line {n}: missing laps"))?
                    .parse().map_err(|e| format!("line {n}: laps: {e}"))?;
                let hash_str = tok.next().ok_or(format!("line {n}: missing hash"))?;
                let state_hash = u64::from_str_radix(hash_str, 16)
                    .map_err(|e| format!("line {n}: hash {hash_str}: {e}"))?;
                let offtrack: u64 = tok.next().ok_or(format!("line {n}: missing offtrack"))?
                    .parse().map_err(|e| format!("line {n}: offtrack: {e}"))?;
                let lap_times: Vec<f64> = tok
                    .map(|t| t.parse::<f64>().map_err(|e| format!("line {n}: lap time: {e}")))
                    .collect::<Result<_, _>>()?;
                expected = Some(SimOutcomeLite { ticks, laps_done, state_hash, offtrack_ticks: offtrack, lap_times });
            }
            other => return Err(format!("line {n}: unknown record type '{other}'")),
        }
    }
    Ok(LoadedLog {
        config: config.ok_or("missing H header record")?,
        inputs,
        expected: expected.ok_or("missing R result record")?,
    })
}
