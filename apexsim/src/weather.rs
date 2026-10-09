//! Phase 8: weather, race-director controls, and driver-error scenarios.
//!
//! Three separate things, deliberately kept apart:
//!
//! * **Surface** — a deterministic wet/damp/dry layout in track distance.
//!   Purely a function of `s`, so a car entering the same corner always
//!   finds the same surface. This is what makes a wet race reproducible.
//! * **Race-director sliders** — water, track temperature, and a grip trim.
//!   These are *not* applied immediately from a UI: each one is queued and
//!   takes effect at a named simulation tick (see [`Override`]).
//! * **Driver errors** — seeded mistakes, modelled as a control-layer
//!   perturbation rather than a change to the vehicle. A scenario where car
//!   7 spins has to be explainable as "the driver got it wrong", not "the
//!   physics was different".
//!
//! ## On the thresholds
//!
//! The grip numbers below are **assumptions, not measurements** — the plan
//! says to treat fixed hydroplaning and temperature thresholds that way
//! until they are tested. They are collected in [`Assumptions`] and
//! deliberately kept in one place so a calibration pass has a single site to
//! change, and so a scenario report can say exactly which constants produced
//! it.

use crate::race::TireCompound;

// ---------------------------------------------------------------------------
// Surface
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Dry,
    Damp,
    Wet,
}

impl Surface {
    pub fn label(self) -> &'static str {
        match self {
            Surface::Dry => "dry",
            Surface::Damp => "damp",
            Surface::Wet => "wet",
        }
    }
    pub fn parse(s: &str) -> Option<Surface> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dry" => Some(Surface::Dry),
            "damp" => Some(Surface::Damp),
            "wet" => Some(Surface::Wet),
            _ => None,
        }
    }
    /// Single-character code for the run log (D/W are ambiguous, so damp is B).
    pub fn code(self) -> char {
        match self {
            Surface::Dry => 'D',
            Surface::Damp => 'B',
            Surface::Wet => 'W',
        }
    }
    pub fn from_code(c: &str) -> Option<Surface> {
        match c.trim() {
            "D" => Some(Surface::Dry),
            "B" => Some(Surface::Damp),
            "W" => Some(Surface::Wet),
            _ => None,
        }
    }
    /// ASSUMPTION: fraction of dry grip left on this surface. Real-world wet
    /// tyres lose roughly 30–50% of dry grip; these are the mid-range
    /// values, not a measurement of any specific tyre.
    pub fn base_grip(self) -> f64 {
        match self {
            Surface::Dry => 1.0,
            Surface::Damp => 0.82,
            Surface::Wet => 0.66,
        }
    }
}

/// A band of track distance carrying one surface. Positions are FRACTIONS of
/// the track length so a zone layout is track-independent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceZone {
    pub start: f64,
    pub end: f64,
    pub surface: Surface,
}

/// Named weather presets. Each is a complete, deterministic scenario.
pub fn preset_zones(name: &str) -> Option<Vec<SurfaceZone>> {
    let zone = |a: f64, b: f64, s: Surface| SurfaceZone { start: a, end: b, surface: s };
    match name.trim().to_ascii_lowercase().as_str() {
        "dry" => Some(vec![zone(0.0, 1.0, Surface::Dry)]),
        // Drying-out line: the main straight stays dry, the last corner is
        // still standing in water.
        "damp" => Some(vec![
            zone(0.0, 0.45, Surface::Dry),
            zone(0.45, 0.75, Surface::Damp),
            zone(0.75, 1.0, Surface::Wet),
        ]),
        // Full wet: dry patches only on the two longest straights.
        "wet" => Some(vec![
            zone(0.0, 0.18, Surface::Dry),
            zone(0.18, 0.40, Surface::Damp),
            zone(0.40, 0.78, Surface::Wet),
            zone(0.78, 0.86, Surface::Damp),
            zone(0.86, 1.0, Surface::Dry),
        ]),
        // Mixed: what a passing shower actually leaves behind.
        "mixed" => Some(vec![
            zone(0.0, 0.25, Surface::Dry),
            zone(0.25, 0.50, Surface::Wet),
            zone(0.50, 0.70, Surface::Damp),
            zone(0.70, 0.88, Surface::Wet),
            zone(0.88, 1.0, Surface::Damp),
        ]),
        _ => None,
    }
}

/// Surface at track distance `s`, wrapping at the start/finish line.
pub fn surface_at(zones: &[SurfaceZone], s: f64, track_len: f64) -> Surface {
    if track_len <= 0.0 || zones.is_empty() {
        return Surface::Dry;
    }
    let f = (s.rem_euclid(track_len)) / track_len;
    for z in zones {
        if f >= z.start && f < z.end {
            return z.surface;
        }
    }
    Surface::Dry
}

// ---------------------------------------------------------------------------
// Race-director controls, and the queued overrides that change them
// ---------------------------------------------------------------------------

/// Which slider an [`Override`] moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slider {
    Water,
    TrackTemp,
    Grip,
}

impl Slider {
    pub fn label(self) -> &'static str {
        match self {
            Slider::Water => "water",
            Slider::TrackTemp => "track_temp",
            Slider::Grip => "grip",
        }
    }
    pub fn parse(s: &str) -> Option<Slider> {
        match s.trim().to_ascii_lowercase().as_str() {
            "water" | "rain" => Some(Slider::Water),
            "temp" | "track_temp" | "track-temp" => Some(Slider::TrackTemp),
            "grip" => Some(Slider::Grip),
            _ => None,
        }
    }
    /// Legal range of the slider, as the race director's box allows it.
    pub fn bounds(self) -> (f64, f64) {
        match self {
            Slider::Water => (0.0, 1.0),
            Slider::TrackTemp => (-10.0, 70.0),
            Slider::Grip => (0.5, 1.2),
        }
    }
    /// Single-character code for the run log.
    pub fn code(self) -> char {
        match self {
            Slider::Water => 'w',
            Slider::TrackTemp => 't',
            Slider::Grip => 'g',
        }
    }
    pub fn from_code(c: &str) -> Option<Slider> {
        match c.trim() {
            "w" => Some(Slider::Water),
            "t" => Some(Slider::TrackTemp),
            "g" => Some(Slider::Grip),
            _ => None,
        }
    }
    pub fn clamp(self, v: f64) -> f64 {
        let (lo, hi) = self.bounds();
        v.clamp(lo, hi)
    }
}

/// Parse a director's slider command: `slider:<name>:<value>`.
///
/// The dashboard sends this as an ordinary `cmd` string, so the shape is shared
/// with the command-line `--overrides` syntax and needs one parser, not two.
/// Out-of-range values are rejected rather than clamped here — a slider that
/// silently reads 1.0 when the director asked for 3.0 is a bug report nobody
/// can act on. [`Slider::clamp`] does the clamping on the way into the queue.
pub fn parse_slider_cmd(cmd: &str) -> Result<(Slider, f64), String> {
    let rest = cmd
        .strip_prefix("slider:")
        .ok_or_else(|| format!("not a slider command: '{cmd}'"))?;
    let (name, value) = rest
        .split_once(':')
        .ok_or_else(|| format!("slider command needs slider:<name>:<value>, got '{cmd}'"))?;
    let slider =
        Slider::parse(name).ok_or_else(|| format!("unknown slider '{}'", name.trim()))?;
    let value: f64 = value
        .trim()
        .parse()
        .map_err(|_| format!("slider '{}' needs a number, got '{}'", slider.label(), value.trim()))?;
    if !value.is_finite() {
        return Err(format!("slider '{}' value is not a finite number", slider.label()));
    }
    let (lo, hi) = slider.bounds();
    if value < lo || value > hi {
        return Err(format!(
            "slider '{}' must be {}..={}, got {}",
            slider.label(),
            lo,
            hi,
            value
        ));
    }
    Ok((slider, value))
}

/// A queued race-director command. It does nothing until the simulation
/// reaches `tick` — the plan is explicit that UI values must never be pushed
/// straight into unsynchronized physics state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Override {
    /// The simulation tick at which this takes effect.
    pub tick: u64,
    pub slider: Slider,
    pub value: f64,
}

impl Override {
    pub fn new(tick: u64, slider: Slider, value: f64) -> Override {
        Override { tick, slider, value: slider.clamp(value) }
    }
}

/// The live slider values, plus the schedule still to come.
#[derive(Clone, Debug, PartialEq)]
pub struct WeatherConfig {
    pub water: f64,
    pub track_temp_c: f64,
    /// Race-director grip trim, applied on top of surface and tire.
    pub grip_scale: f64,
    pub zones: Vec<SurfaceZone>,
    /// Commands applied at a known tick. Sorted by tick on construction.
    pub schedule: Vec<Override>,
    pub driver_errors: DriverErrorConfig,
    /// Name of the preset this came from, if any. Kept so the results sheet and
    /// the run log can say what the scenario was — `damp` and `wet` share zone
    /// types and differ only in layout, so they are not derivable.
    pub preset: Option<String>,
    /// Keep the weather runtime alive even when every slider sits at a
    /// grip-neutral value. A live race-director session has to be able to move
    /// a slider on tick 1, which is impossible if a neutral config collapses to
    /// "no weather". Physics is unchanged while the values are neutral — this
    /// only decides whether the sliders exist.
    pub live: bool,
}

/// Preset name → display label. `&'static` because a report wants a literal.
const PRESET_LABELS: &[(&str, &str)] =
    &[("dry", "dry"), ("damp", "damp"), ("wet", "wet"), ("mixed", "mixed")];

impl Default for WeatherConfig {
    fn default() -> Self {
        WeatherConfig {
            water: 0.0,
            track_temp_c: 30.0,
            grip_scale: 1.0,
            zones: preset_zones("dry").unwrap(),
            schedule: Vec::new(),
            driver_errors: DriverErrorConfig::default(),
            preset: None,
            live: false,
        }
    }
}

impl WeatherConfig {
    /// A race-director console session: dry, neutral, and *live*, so the
    /// dashboard's sliders are live from the first tick. The temperature starts
    /// at the grip optimum rather than at [`WeatherConfig::default`]'s 30 °C,
    /// because a panel that opened 5 °C off the optimum would quietly hand the
    /// cars 3% less grip than a session with no weather at all — the sliders
    /// have to read zero-change at their neutral positions.
    pub fn director() -> WeatherConfig {
        WeatherConfig {
            track_temp_c: Assumptions::IDEAL_TEMP_C,
            live: true,
            ..WeatherConfig::default()
        }
    }

    pub fn preset(name: &str) -> Option<WeatherConfig> {
        let key = name.trim().to_ascii_lowercase();
        let zones = preset_zones(&key)?;
        let water = match key.as_str() {
            "damp" => 0.35,
            "wet" => 1.0,
            "mixed" => 0.8,
            _ => 0.0,
        };
        Some(WeatherConfig {
            water,
            zones,
            preset: Some(key),
            ..WeatherConfig::default()
        })
    }

    /// Sort the schedule and drop commands that land on the same tick for
    /// the same slider — last one wins, so a queue built by the UI cannot
    /// apply two contradictory values on one tick.
    pub fn normalise(&mut self) {
        for o in &mut self.schedule {
            o.value = o.slider.clamp(o.value);
        }
        self.schedule.sort_by_key(|o| (o.tick, o.slider.label()));
        // Two commands for one slider on one tick contradict each other, and
        // the later one is the correction — so the later value must win.
        // `dedup_by` passes (current, previous) and drops `current`, so the
        // survivor is `b`; copy `a` (the later value) into it before dropping.
        self.schedule.dedup_by(|a, b| {
            if a.tick == b.tick && a.slider == b.slider {
                *b = *a;
                true
            } else {
                false
            }
        });
    }

    pub fn with_schedule(mut self, schedule: Vec<Override>) -> WeatherConfig {
        self.schedule = schedule;
        self.normalise();
        self
    }

    /// Builder for the driver-error config, so a scenario can be described in
    /// one expression rather than a `mut` block.
    pub fn with_driver_error(mut self, errors: DriverErrorConfig) -> WeatherConfig {
        self.driver_errors = errors;
        self.normalise();
        self
    }

    /// True when this configuration changes nothing — the Phase 7 engine,
    /// bit for bit. Used so a dry race with no schedule stays identical.
    /// A `live` session is never inert: its sliders are the point.
    pub fn is_inert(&self) -> bool {
        if self.live {
            return false;
        }
        self.water == 0.0
            && self.grip_scale == 1.0
            && self.track_temp_c == 30.0
            && self.schedule.is_empty()
            && self.driver_errors.is_off()
            && self.zones.iter().all(|z| z.surface == Surface::Dry)
    }

    /// What to call this scenario in a report. The preset name when there is one —
    /// `damp` and `wet` differ in their zone *layout*, not just in water, so no
    /// derivation from the zones could tell them apart. Without a preset, fall
    /// back to describing the zones themselves.
    pub fn label(&self) -> &'static str {
        self.label_at(self.water)
    }

    /// The scenario name for a given water level. Separate from [`Self::label`]
    /// because a live panel's sliders have moved since the config was built,
    /// and a director who has just put water on the track should not still be
    /// told the session is dry.
    pub fn label_at(&self, water: f64) -> &'static str {
        if self.is_inert() {
            return "dry";
        }
        if let Some(name) = self.preset.as_deref() {
            return PRESET_LABELS
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| *v)
                .unwrap_or("custom");
        }
        let has = |s: Surface| self.zones.iter().any(|z| z.surface == s);
        match (has(Surface::Dry), has(Surface::Damp), has(Surface::Wet)) {
            (true, _, true) => "mixed", // dry patches and standing water
            (true, _, false) if water > 0.0 => "damp", // dry layout, water on it
            (true, _, false) => "dry",
            (false, true, false) => "damp",
            (false, _, true) => "wet",
            _ => "custom",
        }
    }
}

/// A race-director's change, as it actually reached the simulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AppliedOverride {
    pub tick: u64,
    pub slider: Slider,
    pub value: f64,
}

/// Applies the schedule exactly once per tick, in tick order. The remaining
/// commands are consumed as the simulation reaches their tick, so the
/// effective tick of every command is known and logged — never "whenever the
/// UI happened to send it".
#[derive(Clone, Debug, Default)]
pub struct WeatherRuntime {
    /// Slider values at the current tick (start from the config).
    pub water: f64,
    pub track_temp_c: f64,
    pub grip_scale: f64,
    pending: Vec<Override>,
    applied: Vec<AppliedOverride>,
}

impl WeatherRuntime {
    pub fn new(cfg: &WeatherConfig) -> WeatherRuntime {
        let mut pending = cfg.schedule.clone();
        pending.sort_by_key(|o| (o.tick, o.slider.label()));
        WeatherRuntime {
            water: Slider::Water.clamp(cfg.water),
            track_temp_c: Slider::TrackTemp.clamp(cfg.track_temp_c),
            grip_scale: Slider::Grip.clamp(cfg.grip_scale),
            pending,
            applied: Vec::new(),
        }
    }

    /// Apply every command scheduled for this tick. Returns the ones that
    /// took effect here, in application order.
    pub fn advance_to(&mut self, tick: u64) -> Vec<AppliedOverride> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let mut fired = Vec::new();
        let mut keep = Vec::with_capacity(self.pending.len());
        for o in self.pending.drain(..) {
            if o.tick <= tick {
                match o.slider {
                    Slider::Water => self.water = o.value,
                    Slider::TrackTemp => self.track_temp_c = o.value,
                    Slider::Grip => self.grip_scale = o.value,
                }
                self.applied.push(AppliedOverride { tick, slider: o.slider, value: o.value });
                fired.push(AppliedOverride { tick, slider: o.slider, value: o.value });
            } else {
                keep.push(o);
            }
        }
        self.pending = keep;
        fired
    }

    pub fn applied(&self) -> &[AppliedOverride] {
        &self.applied
    }

    pub fn remaining(&self) -> usize {
        self.pending.len()
    }

    /// Add a director command while the race is already running. It lands on
    /// `o.tick` and nowhere else — the queue is what makes a live slider move
    /// as reproducible as one set on the command line before the start.
    pub fn push(&mut self, o: Override) {
        self.pending.push(o);
        self.pending.sort_by_key(|o| (o.tick, o.slider.label()));
    }
}

// ---------------------------------------------------------------------------
// Grip as a function of surface, water, temperature, and speed
// ---------------------------------------------------------------------------

/// The constants the grip model is built on. All ASSUMPTIONS — collected so
/// a calibration pass has one site, and so a scenario can state its basis.
pub struct Assumptions;

impl Assumptions {
    /// ASSUMPTION: grip optimum for a race tyre.
    pub const IDEAL_TEMP_C: f64 = 35.0;
    /// ASSUMPTION: full grip loss per °C away from the optimum, either side.
    pub const TEMP_LOSS_PER_C: f64 = 0.006;
    /// ASSUMPTION: speed (m/s) at which a fully wet surface begins to
    /// hydroplane. Deliberately crude — it is a threshold to be tested, not
    /// a claim.
    pub const HYDROPLANE_START_MS: f64 = 45.0;
    /// ASSUMPTION: grip lost at 60 m/s of speed over that threshold.
    pub const HYDROPLANE_LOSS_AT_60: f64 = 0.35;
}

/// Effective grip multiplier for one point on the track.
///
/// Returns a multiplier applied to the car's tire grip, in roughly
/// `[0.25, 1.2]`. Every term is a pure function of its arguments, so a replay
/// reconstructs it exactly from the logged values.
pub fn surface_grip(surface: Surface, water: f64, track_temp_c: f64, grip_scale: f64, v: f64) -> f64 {
    // Water slider: standing water on top of whatever the zone already is.
    // Dry stays dry — you cannot wet a dry line with the rain slider.
    let wetted = match surface {
        Surface::Dry => 1.0 - 0.25 * water.clamp(0.0, 1.0),
        Surface::Damp => 1.0 - 0.12 * water.clamp(0.0, 1.0),
        Surface::Wet => 1.0 - 0.06 * water.clamp(0.0, 1.0),
    };
    let base = surface.base_grip() * wetted;

    // Temperature: grip peaks at the optimum and falls off either side.
    let dt = track_temp_c - Assumptions::IDEAL_TEMP_C;
    let thermal = 1.0 - Assumptions::TEMP_LOSS_PER_C * dt.abs();

    // Hydroplaning: only a genuinely wet surface, only at speed.
    let wetness = match surface {
        Surface::Wet => 1.0,
        Surface::Damp => 0.35,
        Surface::Dry => 0.0,
    };
    let over = (v - Assumptions::HYDROPLANE_START_MS).max(0.0);
    let hydro = 1.0
        - Assumptions::HYDROPLANE_LOSS_AT_60
            * (over / 60.0).min(1.0)
            * wetness
            * water.clamp(0.0, 1.0);

    (base * thermal * hydro * grip_scale.clamp(0.5, 1.2)).clamp(0.25, 1.2)
}

// ---------------------------------------------------------------------------
// Driver errors — seeded, and separate from the vehicle
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriverErrorConfig {
    /// Expected mistakes per completed lap. 0 = no errors at all.
    pub rate_per_lap: f64,
    /// How long one mistake lasts.
    pub duration_s: f64,
    /// Peak steering error, rad.
    pub steer_mag: f64,
    /// Peak throttle error (signed — negative is lifting too much).
    pub throttle_mag: f64,
    pub seed: u64,
}

impl Default for DriverErrorConfig {
    fn default() -> Self {
        DriverErrorConfig {
            rate_per_lap: 0.0,
            duration_s: 0.45,
            steer_mag: 0.05,
            throttle_mag: 0.35,
            seed: 0x00D8_171E,
        }
    }
}

impl DriverErrorConfig {
    pub fn is_off(&self) -> bool {
        self.rate_per_lap <= 0.0
    }

    /// Upper bound on mistakes in one lap, so the count is a finite sum of
    /// independent draws rather than a single threshold.
    pub const MAX_PER_LAP: u32 = 4;

    /// How many mistakes this car makes on this lap. A pure function of the
    /// car's id, the lap, and the config seed — which is what lets a replay
    /// reproduce the identical mistake without any RNG state to carry.
    ///
    /// Each of [`MAX_PER_LAP`](Self::MAX_PER_LAP) opportunities is an
    /// independent Bernoulli draw with p = rate/MAX, so the expected count is
    /// exactly `rate_per_lap`. (Taking `floor(hash * rate)` instead would
    /// give zero mistakes almost always at rate 1.0 — the mean would be 0.5,
    /// not 1.)
    pub fn errors_on_lap(&self, car: u32, lap: u32) -> u32 {
        if self.is_off() {
            return 0;
        }
        let p = (self.rate_per_lap / Self::MAX_PER_LAP as f64).clamp(0.0, 1.0);
        (0..Self::MAX_PER_LAP)
            .filter(|k| self.draw(car, lap, *k) < p)
            .count() as u32
    }

    /// One independent uniform draw for opportunity `k` of this lap.
    fn draw(&self, car: u32, lap: u32, k: u32) -> f64 {
        crate::rng::Lcg::new(
            self.seed ^ ((car as u64) << 32) ^ (lap as u64) ^ ((k as u64) << 8) ^ 0x9E37,
        )
        .next_f64()
    }

    /// Where in the lap mistake `k` sits, as a fraction of the lap, with the
    /// window each one occupies left free around it. Pure in (seed, car, lap, k)
    /// so the mistake happens at the same place in every replay.
    fn placement(&self, car: u32, lap: u32, k: u32, window: f64) -> f64 {
        let mut lcg = crate::rng::Lcg::new(
            self.seed ^ ((car as u64) << 32) ^ (lap as u64) ^ ((k as u64) << 8) ^ 0x9E37,
        );
        lcg.next_u64();
        let span = (1.0 - window * Self::MAX_PER_LAP as f64).max(0.05);
        span * lcg.next_f64()
    }

    /// The control offset from mistake `k` of this lap, given the car's
    /// current `phase` through the lap (0..1) and `window` (the fraction of
    /// the lap one mistake lasts). Zero outside the mistake's window, and zero
    /// for any k past the count, so callers can sum over a fixed range.
    pub fn offset(&self, car: u32, lap: u32, k: u32, phase: f64, window: f64) -> (f64, f64) {
        if k >= self.errors_on_lap(car, lap) {
            return (0.0, 0.0);
        }
        let window = window.clamp(1e-3, 0.5);
        let start = self.placement(car, lap, k, window);
        let u = (phase - start) / window;
        if !(0.0..=1.0).contains(&u) {
            return (0.0, 0.0);
        }
        let mut lcg = crate::rng::Lcg::new(
            self.seed ^ ((car as u64) << 32) ^ (lap as u64) ^ ((k as u64) << 8) ^ 0x9E37,
        );
        lcg.next_u64();
        let dir = if lcg.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        let mag = 0.55 + 0.45 * lcg.next_f64();
        // Triangular envelope so the mistake ramps in and out instead of
        // snapping on — a step input at 250 Hz is a shock, not a slip.
        let env = (1.0 - (2.0 * u - 1.0).abs()).clamp(0.0, 1.0);
        (
            dir * self.steer_mag * mag * env,
            dir * self.throttle_mag * mag * env,
        )
    }
}

// ---------------------------------------------------------------------------
// Stint-aware surface naming (for reports)
// ---------------------------------------------------------------------------

/// One-word description of what a car is on, including its tire, for the
/// results sheet.
pub fn describe(surface: Surface, compound: TireCompound, wear: f64) -> String {
    format!(
        "{}·{}{}",
        surface.label(),
        compound.label(),
        if wear >= 1.0 {
            " (worn)"
        } else if wear > 0.6 {
            " (ageing)"
        } else {
            ""
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zones_wrap_at_the_start_line() {
        let z = preset_zones("wet").unwrap();
        let len = 1102.65;
        assert_eq!(surface_at(&z, 0.0, len), Surface::Dry);
        assert_eq!(surface_at(&z, len + 5.0, len), surface_at(&z, 5.0, len));
        assert_eq!(surface_at(&z, len - 1.0, len), Surface::Dry);
        assert_eq!(surface_at(&z, 0.5 * len, len), Surface::Wet);
    }

    #[test]
    fn dry_preset_is_inert() {
        assert!(WeatherConfig::preset("dry").unwrap().is_inert());
        assert!(WeatherConfig::default().is_inert());
        assert!(!WeatherConfig::preset("wet").unwrap().is_inert());
    }

    #[test]
    fn overrides_fire_on_their_tick_and_not_before() {
        let cfg = WeatherConfig::default().with_schedule(vec![
            Override::new(500, Slider::Water, 0.8),
            Override::new(100, Slider::Grip, 0.9),
        ]);
        let mut rt = WeatherRuntime::new(&cfg);
        assert_eq!(rt.advance_to(99), vec![]);
        assert_eq!(rt.water, 0.0, "must not act before its tick");
        assert_eq!(rt.advance_to(100).len(), 1);
        assert!((rt.grip_scale - 0.9).abs() < 1e-12);
        assert_eq!(rt.advance_to(499), vec![]);
        assert_eq!(rt.water, 0.0, "still not wet at tick 499");
        let fired = rt.advance_to(500);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].tick, 500);
        assert_eq!(fired[0].slider, Slider::Water);
        assert!((rt.water - 0.8).abs() < 1e-12);
        assert_eq!(rt.applied().len(), 2);
        assert_eq!(rt.remaining(), 0);
        // Idempotent: a tick with nothing pending changes nothing.
        assert_eq!(rt.advance_to(900), vec![]);
    }

    #[test]
    fn overrides_are_clamped_and_deduplicated() {
        let mut cfg = WeatherConfig::default().with_schedule(vec![
            Override::new(10, Slider::Water, 5.0),
            Override::new(10, Slider::Water, 0.2),
            Override::new(10, Slider::TrackTemp, 500.0),
        ]);
        cfg.normalise();
        assert_eq!(cfg.schedule.len(), 2, "same slider same tick collapses");
        // normalise() sorts by (tick, slider), so look the values up by slider
        // rather than assuming the caller's insertion order survives.
        let val = |s: Slider| cfg.schedule.iter().find(|o| o.slider == s).unwrap().value;
        assert!((val(Slider::Water) - 0.2).abs() < 1e-12, "last write wins");
        assert!((val(Slider::TrackTemp) - 70.0).abs() < 1e-12, "temp is clamped");
        assert!(cfg.schedule.windows(2).all(|w| (w[0].tick, w[0].slider.label()) <= (w[1].tick, w[1].slider.label())));
    }

    #[test]
    fn each_preset_reports_its_own_name() {
        // damp and wet share zone *types* and differ only in layout, so the
        // name has to be carried, not derived — that is what regressed once.
        for name in ["dry", "damp", "wet", "mixed"] {
            let w = WeatherConfig::preset(name).unwrap();
            assert_eq!(w.label(), name, "preset {name} mislabelled as {}", w.label());
        }
        // A slider-only config is described by what it does, not by a name.
        let sliders = WeatherConfig { water: 0.6, ..WeatherConfig::default() };
        assert_eq!(sliders.preset, None);
        assert_eq!(sliders.label(), "damp");
        // A grip-only trim is still a dry track, just a slower car.
        let trim = WeatherConfig { grip_scale: 0.8, ..WeatherConfig::default() };
        assert_eq!(trim.label(), "dry");
    }

    #[test]
    fn wet_is_slower_than_dry_and_temperature_has_an_optimum() {
        let v = 30.0;
        let dry = surface_grip(Surface::Dry, 0.0, 35.0, 1.0, v);
        let damp = surface_grip(Surface::Damp, 0.4, 35.0, 1.0, v);
        let wet = surface_grip(Surface::Wet, 1.0, 35.0, 1.0, v);
        assert!(dry > damp && damp > wet, "dry {dry} damp {damp} wet {wet}");

        // Equal distances either side of the 35 °C optimum.
        let hot = surface_grip(Surface::Dry, 0.0, 55.0, 1.0, v);
        let cold = surface_grip(Surface::Dry, 0.0, 15.0, 1.0, v);
        assert!(hot < dry && cold < dry, "optimum is a peak, not a slope");
        assert!((hot - cold).abs() < 1e-12, "symmetric around the optimum");
    }

    #[test]
    fn hydroplaning_only_bites_when_wet_and_fast() {
        let slow = surface_grip(Surface::Wet, 1.0, 35.0, 1.0, 40.0);
        let fast = surface_grip(Surface::Wet, 1.0, 35.0, 1.0, 70.0);
        assert!(fast < slow, "wet at speed must lose grip");
        let dry_fast = surface_grip(Surface::Dry, 1.0, 35.0, 1.0, 70.0);
        let dry_slow = surface_grip(Surface::Dry, 1.0, 35.0, 1.0, 40.0);
        assert!((dry_fast - dry_slow).abs() < 1e-12, "a dry line does not hydroplane");
    }

    #[test]
    fn grip_never_leaves_its_legal_band() {
        for &w in &[0.0, 0.5, 1.0] {
            for &t in &[-10.0, 0.0, 35.0, 70.0] {
                for &g in &[0.5, 1.0, 1.2] {
                    for &v in &[0.0, 30.0, 80.0] {
                        let g = surface_grip(Surface::Wet, w, t, g, v);
                        assert!(
                            (0.25..=1.2).contains(&g),
                            "grip {g} out of band at w={w} t={t} g={g} v={v}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn driver_errors_are_off_by_default_and_seed_stable() {
        let off = DriverErrorConfig::default();
        assert!(off.is_off());
        assert_eq!(off.errors_on_lap(3, 5), 0);
        assert_eq!(off.offset(3, 5, 0, 0.5, 0.1), (0.0, 0.0));

        let on = DriverErrorConfig { rate_per_lap: 1.0, ..DriverErrorConfig::default() };
        // Same inputs, same mistake, every time — this is what makes a
        // replay reproduce the driver's error exactly.
        assert_eq!(on.errors_on_lap(3, 5), on.errors_on_lap(3, 5));
        // A rate of 1.0 is the *expected* count, not a guarantee for any one
        // (car, lap) — each lap draws MAX_PER_LAP independent Bernoulli trials.
        // So find a lap that actually drew one rather than pinning a magic pair.
        let laps: Vec<u32> = (0..40).map(|l| on.errors_on_lap(3, l)).collect();
        let some = laps.iter().position(|&n| n > 0).expect("a rate of 1.0 produced no errors at all");
        let lap = some as u32;
        let w = 0.1;
        let start = on.placement(3, lap, 0, w);
        let mid = start + w / 2.0;
        let a = on.offset(3, lap, 0, mid, w);
        let b = on.offset(3, lap, 0, mid, w);
        assert_eq!(a, b);
        assert!(a.0 != 0.0 || a.1 != 0.0, "a mistake must actually perturb");
        // Envelope is zero outside the mistake's window and largest mid-way.
        assert_eq!(on.offset(3, lap, 0, 0.0, w), (0.0, 0.0));
        assert_eq!(on.offset(3, lap, 0, 1.0, w), (0.0, 0.0));
        assert!(on.offset(3, lap, 0, mid + w * 0.4, w).0.abs() < a.0.abs());
        // Mistakes past the count are inert, so a caller can sum a fixed range.
        let n = on.errors_on_lap(3, lap);
        for k in n..DriverErrorConfig::MAX_PER_LAP {
            assert_eq!(on.offset(3, lap, k, mid, w), (0.0, 0.0), "k={k} is past the count");
        }
        // The mean over many laps must actually track the requested rate.
        let total: u32 = laps.iter().sum();
        let mean = total as f64 / laps.len() as f64;
        assert!((mean - 1.0).abs() < 0.35, "mean {mean} at rate 1.0");
        // A mistake occupies a small slice of the lap, not all of it.
        let active = (0..=200)
            .filter(|p| on.offset(3, lap, 0, *p as f64 / 200.0, w) != (0.0, 0.0))
            .count();
        assert!(active > 0 && active < 200, "{active}/200 of the lap was a mistake");
    }

    #[test]
    fn driver_error_rate_reacts_to_the_slider() {
        let few = DriverErrorConfig { rate_per_lap: 0.2, ..DriverErrorConfig::default() };
        let many = DriverErrorConfig { rate_per_lap: 2.0, ..DriverErrorConfig::default() };
        let sum = |c: DriverErrorConfig| (0..40).map(|l| c.errors_on_lap(1, l)).sum::<u32>();
        assert!(sum(many) > sum(few), "a higher rate must produce more mistakes");
    }
}