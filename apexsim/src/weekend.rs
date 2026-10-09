//! Phase E: the race weekend — one-by-one timed qualifying that sets the grid.
//!
//! The format the plan (and the stand-in) asks for: each car goes out **alone**
//! for a fixed session window, is timed, and the lap times set the grid —
//! fastest at pole, slowest at the back. A qualifying session is therefore a
//! sequence of [`RaceConfig::solo_car`] sessions, each a real race with the
//! real formation start, real machinery, and the car's own driver form and
//! noise seed. Nothing about a solo run is a simulation shortcut: the only
//! difference from the race is that the pit lane is closed and there is nobody
//! else out there to hit.
//!
//! Determinism: runs are independent, so rows reproduce exactly from the
//! seed. Grid construction is a stable sort over (best lap, car id), so a car
//! with no time (never crossed the line, or couldn't put a lap together in
//! the window) starts behind every timed car, ordered by id — reproducible,
//! and honest about who failed to set a time.

use crate::race::{Race, RaceConfig};

/// One car's qualifying sheet.
#[derive(Clone, Debug, PartialEq)]
pub struct QualiRow {
    pub car: u32,
    /// Best lap of the session. `None` = no time set.
    pub best_lap_s: Option<f64>,
    /// Fastest trap speed of the session, m/s — the "top speed" each car's
    /// engine was tested for.
    pub top_speed_mps: f64,
    /// Out-laps started (crossings of the line).
    pub laps: u32,
}

impl QualiRow {
    /// Sort key: timed cars fastest-first; untimed cars to the back by id.
    pub fn key(&self) -> (u8, f64, u32) {
        match self.best_lap_s {
            Some(t) => (0, t, self.car),
            None => (1, 0.0, self.car),
        }
    }
}

impl QualiRow {
    /// Build a row from a finished solo session (the server's harvest).
    pub fn from_session(car: u32, best_lap_s: Option<f64>, top_speed_mps: f64, laps: u32) -> QualiRow {
        QualiRow { car, best_lap_s, top_speed_mps, laps }
    }
}

/// Run the one-by-one qualifying session.
///
/// `base` carries the field, circuit, seed and session setup the race will
/// use; this function overrides only what a solo run must change (one car,
/// pits closed, no steward — nothing out there to infringe). `window_s` is
/// each car's session length in sim time; the run also ends at the natural
/// race end (a car that retires has simply set no more times).
pub fn run_qualifying(base: &RaceConfig, window_s: f64) -> Vec<QualiRow> {
    // A base that already names its car runs exactly that car — a caller
    // asking for car 7 must never get a re-labelled car 0 (this gate caught
    // the loop overwriting the caller's identity, which made every solo run
    // measure the same machinery).
    match base.solo_car {
        Some(id) => vec![run_solo(base, id, window_s)],
        None => (0..base.cars).map(|car| run_solo(base, car, window_s)).collect(),
    }
}

/// One car's session: a real race with formation and its own machinery, run
/// solo for `window_s` of sim time.
fn run_solo(base: &RaceConfig, car: u32, window_s: f64) -> QualiRow {
    let mut cfg = base.clone();
    cfg.solo_car = Some(car);
    cfg.grid_order.clear();
    // Nobody to infringe against out there, and nobody to serve a penalty
    // to: the aid watches races, not time-attack runs.
    cfg.steward = None;
    // A solo run is not a pit-stop practice: the box is closed so the whole
    // window is flying laps.
    cfg.strategy = crate::race::PitStrategy::None;
    let mut race = Race::new(cfg);
    race.start();
    let cap_ticks = (window_s / crate::sim::DT).max(1.0) as u64;
    let mut top = 0.0f64;
    for _ in 0..cap_ticks {
        race.tick();
        let v = race.state.cars[0].car.vel_long;
        if v > top {
            top = v;
        }
        if race.state.phase == crate::race::RacePhase::Finished {
            break;
        }
    }
    let entry = &race.state.cars[0];
    QualiRow {
        car,
        best_lap_s: entry.best_lap,
        top_speed_mps: top,
        laps: entry.laps_crossings,
    }
}

/// The qualifying result as a grid order: pole first, untimed cars at the
/// back by car id. Feeds [`RaceConfig::grid_order`] verbatim.
pub fn grid_from_rows(rows: &[QualiRow]) -> Vec<u32> {
    let mut sorted: Vec<&QualiRow> = rows.iter().collect();
    // f64 has no Ord; total_cmp gives the total order the sort needs (NaN
    // lap times cannot occur — a lap is either set and finite, or absent).
    sorted.sort_by(|a, b| {
        let (ka, kb) = (a.key(), b.key());
        ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then(ka.2.cmp(&kb.2))
    });
    sorted.iter().map(|r| r.car).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untimed_cars_sort_behind_timed_ones() {
        let rows = vec![
            QualiRow { car: 2, best_lap_s: None, top_speed_mps: 50.0, laps: 1 },
            QualiRow { car: 0, best_lap_s: Some(45.0), top_speed_mps: 80.0, laps: 4 },
            QualiRow { car: 3, best_lap_s: Some(44.0), top_speed_mps: 81.0, laps: 4 },
            QualiRow { car: 1, best_lap_s: None, top_speed_mps: 49.0, laps: 0 },
        ];
        assert_eq!(grid_from_rows(&rows), vec![3, 0, 1, 2]);
    }
}
