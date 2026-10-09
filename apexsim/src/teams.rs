//! Team-based field differentiation (fictional 2026-style grid).
//!
//! Before this module every car was physically identical: one shared
//! `CarParams`, and the only per-car variation was the driver-envelope
//! `pace_spread`. This gives each *team* its own engine power, top speed,
//! braking and tyre grip, with both cars of a team identical — the way a real
//! championship works — plus a tiny seeded per-driver form delta so teammates
//! are not interchangeable.
//!
//! The stats live in a committed, human-editable `teams.json` rather than
//! in code, so tuning a team does not mean recompiling the engine. The file's
//! team order must match the dashboard's `GRID_2026` sprite mapping: car id
//! `i` belongs to team `i / 2`, so ids 0 and 1 are the first team's pair, ids
//! 2 and 3 the second's, and so on. A field larger than the table (more than
//! two cars per the last team) clamps to the final entry rather than panicking.
//!
//! Determinism: the team attributes come from the file and the driver form
//! delta from `(race seed, car id)`, so the same file plus the same seed
//! reproduces the same field bit-for-bit. Everything here is a pure function —
//! nothing about a team changes over a race except what the tyres already did
//! (compound and wear).

use serde::Deserialize;
use std::sync::OnceLock;

/// The per-team multipliers. All four are dimensionless factors around 1.0:
/// 1.0 means exactly the baseline `CarParams::default()` car.
#[derive(Clone, Debug, Deserialize)]
pub struct TeamStats {
    pub name: String,
    /// Engine force and power multiplier — acceleration out of corners.
    pub engine: f64,
    /// Top-speed multiplier. The driver envelope takes it directly; the
    /// physics takes it as drag scaled by the CUBE (power-limited top speed
    /// goes as (P/cdA)^(1/3), so dividing drag by s³ raises vmax by s).
    pub top_speed: f64,
    /// Brake force multiplier.
    pub brakes: f64,
    /// Tyre grip multiplier — the one attribute that must be applied to the
    /// physics AND the driver envelope in lockstep (see `Race::params_for`
    /// and `Race::grip_scale`): a driver who aims at grip his tyres do not
    /// have simply slides, and the failure looks like a physics bug.
    pub tyres: f64,
}

#[derive(Deserialize)]
struct TeamFile {
    teams: Vec<TeamStats>,
}

impl TeamStats {
    /// A hand-typed multiplier outside this band is a typo, not a tuning
    /// choice; reject it where it can be explained instead of letting a car
    /// lap at half speed.
    fn validate(&self) -> Result<(), String> {
        for (attr, v) in [
            ("engine", self.engine),
            ("top_speed", self.top_speed),
            ("brakes", self.brakes),
            ("tyres", self.tyres),
        ] {
            if !(0.90..=1.10).contains(&v) || !v.is_finite() {
                return Err(format!(
                    "team {} attribute {attr} = {v} is outside 0.90..=1.10",
                    self.name
                ));
            }
        }
        Ok(())
    }
}

/// The parsed team table, built once per process.
pub struct TeamBook {
    teams: Vec<TeamStats>,
}

impl TeamBook {
    pub fn from_json(text: &str) -> Result<TeamBook, String> {
        let file: TeamFile = serde_json::from_str(text)
            .map_err(|e| format!("teams file is not valid JSON: {e}"))?;
        if file.teams.is_empty() {
            return Err("teams file declares no teams".into());
        }
        for t in &file.teams {
            t.validate()?;
        }
        Ok(TeamBook { teams: file.teams })
    }

    /// The stats of the team car id drives. Two ids per team, in
    /// `GRID_2026` order; ids past the table clamp to the last team.
    pub fn for_car(&self, id: u32) -> &TeamStats {
        let idx = (id as usize / 2).min(self.teams.len() - 1);
        &self.teams[idx]
    }

    pub fn len(&self) -> usize {
        self.teams.len()
    }

    pub fn is_empty(&self) -> bool {
        self.teams.is_empty()
    }
}

fn book() -> &'static TeamBook {
    static BOOK: OnceLock<TeamBook> = OnceLock::new();
    BOOK.get_or_init(|| {
        TeamBook::from_json(include_str!("../teams.json"))
            .expect("embedded teams.json must parse; it is validated by tests")
    })
}

/// The team stats car `id` drives, from the embedded teams file.
pub fn for_car(id: u32) -> &'static TeamStats {
    book().for_car(id)
}

/// Per-driver form delta, ±`DELTA`, deterministic in (seed, id). Both cars of
/// a team share the machinery; this is the only thing that separates
/// teammates, and it is two orders of magnitude smaller than the team spread.
pub const FORM_DELTA: f64 = 0.003;

pub fn driver_form(seed: u64, id: u32) -> f64 {
    let s = seed ^ (0x5DEECE66D_u64.wrapping_mul(id as u64 + 1));
    1.0 + (crate::rng::Lcg::new(s).next_f64() * 2.0 - 1.0) * FORM_DELTA
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_file_is_valid_and_covers_the_2026_grid() {
        let b = book();
        assert!(b.len() >= 11, "the 2026 grid has 11 entries in GRID_2026");
        // Ids 0..22 must resolve without panicking (Polaris is the 11th pair).
        for id in 0..22u32 {
            let t = b.for_car(id);
            assert!(!t.name.is_empty());
            t.validate().expect("embedded stats in range");
        }
        // Both cars of a team share the stats.
        assert_eq!(b.for_car(0).name, b.for_car(1).name);
        assert_ne!(b.for_car(0).name, b.for_car(2).name);
    }

    #[test]
    fn field_spread_stays_near_two_percent() {
        let b = book();
        for t in &b.teams {
            for v in [t.engine, t.top_speed, t.brakes, t.tyres] {
                assert!(
                    (0.975..=1.025).contains(&v),
                    "{} attr {v} outside the designed ±2.5% band",
                    t.name
                );
            }
        }
    }

    #[test]
    fn driver_form_is_seeded_bounded_and_pairwise_distinct() {
        for id in 0..20u32 {
            let f = driver_form(42, id);
            assert!((1.0 - FORM_DELTA..=1.0 + FORM_DELTA).contains(&f));
        }
        // Different cars of one team get different form (these fixed seeds do).
        assert_ne!(driver_form(42, 0), driver_form(42, 1));
        // Same car, different seed, can differ.
        assert_ne!(driver_form(42, 3), driver_form(43, 3));
    }

    #[test]
    fn a_typo_in_the_file_is_rejected_not_run() {
        let bad = r#"{"teams":[{"name":"X","engine":1.5,"top_speed":1.0,"brakes":1.0,"tyres":1.0}]}"#;
        assert!(TeamBook::from_json(bad).is_err());
        let malformed = r#"{"teams": [{"name": "Y"}]}"#;
        assert!(TeamBook::from_json(malformed).is_err());
        let empty = r#"{"teams": []}"#;
        assert!(TeamBook::from_json(empty).is_err());
    }
}
