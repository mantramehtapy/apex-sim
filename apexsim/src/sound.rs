//! Phase 9 — event-driven sound.
//!
//! The plan asks for "RPM, tire slip, and wind cues" and then says to
//! "prefer a small validated sound vocabulary before procedural synthesis".
//! That is the distinction this module is built around:
//!
//! * **Extraction is total and deterministic.** Every cue the race produces is
//!   derived from `(tick, car state)` and written to an ordered log. It does
//!   not depend on wall-clock time, on how fast the machine is, or on whether
//!   anything is listening. That is what makes a replay reproduce the same
//!   sound log bit-for-bit, and what keeps a paused race silent.
//! * **Playback is lossy and bounded.** The consumer may start at most
//!   [`SoundConfig::max_cues_per_s`] cues per wall-clock second, shedding the
//!   lowest priority first. In accelerated mode the race generates cues far
//!   faster than a person could hear them; dropping them is correct, queueing
//!   them would build an ever-growing backlog and eventually stall the race.
//!
//! The vocabulary is deliberately thirteen fixed cues, each backed by a
//! validated audio asset rather than by runtime synthesis. See
//! `tools/make_sounds.py` and [`validate_wav`].

use crate::telemetry::{CarSample, Gearbox};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// The vocabulary
// ---------------------------------------------------------------------------

/// The complete set of sounds this engine can make. Order is the wire code
/// and must stay stable: it is recorded in the race log and compared against
/// on replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SoundCue {
    ShiftUp = 0,
    ShiftDown = 1,
    RevLimiter = 2,
    TyreSlip = 3,
    Wind = 4,
    LapComplete = 5,
    StartLight = 6,
    GreenFlag = 7,
    PitEntered = 8,
    Contact = 9,
    WeatherChange = 10,
    SessionFinish = 11,
    FormationTick = 12,
}

pub const CUE_COUNT: usize = 13;

/// The vocabulary, in code order. Every entry must have a validated asset.
pub const ALL_CUES: [SoundCue; CUE_COUNT] = [
    SoundCue::ShiftUp,
    SoundCue::ShiftDown,
    SoundCue::RevLimiter,
    SoundCue::TyreSlip,
    SoundCue::Wind,
    SoundCue::LapComplete,
    SoundCue::StartLight,
    SoundCue::GreenFlag,
    SoundCue::PitEntered,
    SoundCue::Contact,
    SoundCue::WeatherChange,
    SoundCue::SessionFinish,
    SoundCue::FormationTick,
];

impl SoundCue {
    pub fn code(self) -> u8 {
        self as u8
    }

    pub fn from_code(c: u8) -> Option<SoundCue> {
        ALL_CUES.iter().copied().find(|x| x.code() == c)
    }

    /// Stable snake_case name, used for the audio file, the dashboard and the
    /// log line. Changing one breaks saved logs and the asset map.
    pub fn name(self) -> &'static str {
        match self {
            SoundCue::ShiftUp => "shift_up",
            SoundCue::ShiftDown => "shift_down",
            SoundCue::RevLimiter => "rev_limiter",
            SoundCue::TyreSlip => "tyre_slip",
            SoundCue::Wind => "wind",
            SoundCue::LapComplete => "lap_complete",
            SoundCue::StartLight => "start_light",
            SoundCue::GreenFlag => "green_flag",
            SoundCue::PitEntered => "pit_entered",
            SoundCue::Contact => "contact",
            SoundCue::WeatherChange => "weather_change",
            SoundCue::SessionFinish => "session_finish",
            SoundCue::FormationTick => "formation_tick",
        }
    }

    /// The audio asset for this cue.
    pub fn asset(self) -> String {
        format!("{}.wav", self.name())
    }

    /// 3 = never shed, 0 = shed first.
    ///
    /// Race events outrank continuous cues because losing "green flag" makes
    /// the race unfollowable, while losing a wind gust costs nothing.
    pub fn priority(self) -> u8 {
        match self {
            SoundCue::SessionFinish | SoundCue::GreenFlag | SoundCue::StartLight => 3,
            SoundCue::LapComplete
            | SoundCue::Contact
            | SoundCue::PitEntered
            | SoundCue::WeatherChange
            | SoundCue::FormationTick => 2,
            SoundCue::RevLimiter | SoundCue::ShiftUp | SoundCue::ShiftDown => 1,
            SoundCue::TyreSlip | SoundCue::Wind => 0,
        }
    }

    /// Shortest gap, in **simulation** seconds, before the same cue may fire
    /// again for the same car. Sim time, not wall-clock: a replay runs at a
    /// different speed and must still produce the same log.
    pub fn min_gap_s(self) -> f64 {
        match self {
            SoundCue::RevLimiter => 0.25,
            SoundCue::ShiftUp | SoundCue::ShiftDown => 0.15,
            SoundCue::TyreSlip => 0.35,
            SoundCue::Wind => 1.0,
            // Race events are already rare and meaningful; spacing them would
            // hide real ones.
            _ => 0.0,
        }
    }
}

/// Whether `cue` may fire for this car at `now`. A free function because it
/// reads only the car's own bookkeeping and the cue's own gap — no runtime
/// state, and no borrow of one.
fn ready(st: &CarCueState, cue: SoundCue, now: f64) -> bool {
    now - st.last_fired[cue.code() as usize] >= cue.min_gap_s()
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// One cue the race produced. Car 0 means a session-wide event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CueEvent {
    pub tick: u64,
    pub car: u32,
    pub cue: SoundCue,
    /// The quantity that triggered it: rev/min for a shift, lateral g for a
    /// slip, m/s for wind. Kept so a report can show *why* a cue fired.
    pub value: f64,
}

impl CueEvent {
    /// Order-independent hash of the whole log. Two runs agree only if every
    /// cue fired at the same tick, for the same car, in the same order.
    pub fn mix_into(&self, mut h: u64) -> u64 {
        let code = [self.cue.code()];
        let value = self.value.to_bits().to_le_bytes();
        for b in self
            .tick
            .to_le_bytes()
            .iter()
            .chain(self.car.to_le_bytes().iter())
            .chain(code.iter())
            .chain(value.iter())
        {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

pub fn hash_log(events: &[CueEvent]) -> u64 {
    events.iter().fold(0xcbf2_9ce4_8422_2325, |h, e| e.mix_into(h))
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundConfig {
    pub enabled: bool,
    /// Playback ceiling per wall-clock second. Zero means "no limit", which is
    /// only sensible for a real-time race.
    pub max_cues_per_s: u32,
    /// Lateral acceleration, in g, at which the tyres are considered to be
    /// scrubbing.
    pub slip_threshold_g: f64,
    /// Speed band width for the wind cue, m/s.
    pub wind_band_mps: f64,
    /// Speed above which the wind cue starts, m/s.
    pub wind_start_mps: f64,
}

impl Default for SoundConfig {
    fn default() -> Self {
        SoundConfig {
            enabled: true,
            max_cues_per_s: 40,
            slip_threshold_g: 0.9,
            wind_band_mps: 10.0,
            wind_start_mps: 25.0,
        }
    }
}

impl SoundConfig {
    pub fn is_inert(&self) -> bool {
        !self.enabled
    }
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct CarCueState {
    last_gear: i8,
    last_lap: u32,
    last_wind_band: i32,
    /// Sim time each cue last fired for this car, indexed by cue code.
    last_fired: [f64; CUE_COUNT],
    limiter_on: bool,
}

/// What the phase calls for, extracted from simulation state only.
pub struct SoundRuntime {
    pub cfg: SoundConfig,
    gearbox: Gearbox,
    cars: Vec<CarCueState>,
    /// The complete, ordered, deterministic cue log for the session.
    pub events: Vec<CueEvent>,
    counts: [u32; CUE_COUNT],
}

impl SoundRuntime {
    pub fn new(cfg: SoundConfig, cars: usize, gearbox: Gearbox) -> SoundRuntime {
        SoundRuntime {
            cfg,
            gearbox,
            cars: vec![CarCueState { last_fired: [f64::NEG_INFINITY; CUE_COUNT], ..Default::default() }; cars],
            events: Vec::new(),
            counts: [0; CUE_COUNT],
        }
    }

    pub fn config(&self) -> &SoundConfig {
        &self.cfg
    }

    /// How many cars the per-car cue bookkeeping currently covers.
    pub fn cue_car_count(&self) -> usize {
        self.cars.len()
    }

    /// Called once per car before the race starts, so car 0 does not report a
    /// shift on the first tick just because its gear went from 0.
    pub fn prime(&mut self, samples: &[CarSample]) {
        if self.cars.len() != samples.len() {
            self.cars = vec![CarCueState { last_fired: [f64::NEG_INFINITY; CUE_COUNT], ..Default::default() }; samples.len()];
        }
        for (i, s) in samples.iter().enumerate() {
            self.cars[i].last_gear = s.gear;
        }
    }

    /// Extract continuous cues (shift, limiter, slip, wind) for every car at
    /// `tick`. Race events go through [`Self::race_event`] instead.
    pub fn extract(&mut self, tick: u64, sim_time_s: f64, samples: &[CarSample]) {
        if !self.cfg.enabled {
            return;
        }
        if self.cars.len() != samples.len() {
            self.prime(samples);
        }
        // Car state is read, cues are decided, then applied — so the per-car
        // bookkeeping and the firing stay in one place each and the borrow
        // stays local.
        let mut fired: Vec<(SoundCue, f64)> = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            fired.clear();
            let car = i as u32;
            let st = &mut self.cars[i];

            // Gear change. Below walking pace a car is being shuffled around
            // the grid, not shifting, so the noise is suppressed.
            if s.vel_long >= 5.0 && st.last_gear > 0 {
                if s.gear > st.last_gear && ready(st, SoundCue::ShiftUp, sim_time_s) {
                    fired.push((SoundCue::ShiftUp, f64::from(s.gear)));
                } else if s.gear < st.last_gear && ready(st, SoundCue::ShiftDown, sim_time_s) {
                    fired.push((SoundCue::ShiftDown, f64::from(s.gear)));
                }
            }
            if s.vel_long >= 5.0 {
                st.last_gear = s.gear;
            }

            // Rev limiter. Edge-triggered so it does not buzz once per tick.
            let at_limiter = f64::from(s.engine_rpm) >= self.gearbox.limiter_rpm - 1.0;
            if at_limiter && !st.limiter_on && ready(st, SoundCue::RevLimiter, sim_time_s) {
                fired.push((SoundCue::RevLimiter, f64::from(s.engine_rpm)));
            }
            st.limiter_on = at_limiter;

            // Tyre scrub. Lateral load beyond what the tyres have left is what
            // a driver hears as scrub, so this reads the same lateral g the
            // telemetry packet publishes.
            if s.g_lat.abs() >= self.cfg.slip_threshold_g
                && ready(st, SoundCue::TyreSlip, sim_time_s)
            {
                fired.push((SoundCue::TyreSlip, s.g_lat.abs()));
            }

            // Wind, one cue per speed band crossed.
            if s.vel_long >= self.cfg.wind_start_mps && self.cfg.wind_band_mps > 0.0 {
                let band = (s.vel_long / self.cfg.wind_band_mps) as i32;
                if band != st.last_wind_band {
                    st.last_wind_band = band;
                    if ready(st, SoundCue::Wind, sim_time_s) {
                        fired.push((SoundCue::Wind, s.vel_long));
                    }
                }
            }

            for (cue, value) in fired.drain(..) {
                self.fire(tick, car, cue, value, sim_time_s);
            }
        }
    }

    /// A discrete race event. Priority 2–3 and no minimum gap.
    pub fn race_event(&mut self, tick: u64, car: u32, cue: SoundCue, value: f64) {
        if !self.cfg.enabled {
            return;
        }
        self.events.push(CueEvent { tick, car, cue, value });
        self.counts[cue.code() as usize] += 1;
    }

    /// A per-car lap completion, tracked here so the runtime does not need to
    /// be told about laps it could derive itself.
    pub fn lap_completed(&mut self, tick: u64, car: u32, lap: u32, time_s: f64) {
        if !self.cfg.enabled {
            return;
        }
        if let Some(st) = self.cars.get_mut(car as usize) {
            st.last_lap = lap;
        }
        self.race_event(tick, car, SoundCue::LapComplete, time_s);
    }

    fn fire(&mut self, tick: u64, car: u32, cue: SoundCue, value: f64, now: f64) {
        if let Some(st) = self.cars.get_mut(car as usize) {
            st.last_fired[cue.code() as usize] = now;
        }
        self.events.push(CueEvent { tick, car, cue, value });
        self.counts[cue.code() as usize] += 1;
    }

    pub fn counts(&self) -> &[u32; CUE_COUNT] {
        &self.counts
    }

    pub fn log_hash(&self) -> u64 {
        hash_log(&self.events)
    }

    pub fn summary(&self) -> SoundSummary {
        SoundSummary {
            events: self.events.len(),
            counts: self.counts,
            log_hash: self.log_hash(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundSummary {
    pub events: usize,
    pub counts: [u32; CUE_COUNT],
    pub log_hash: u64,
}

impl SoundSummary {
    /// `name: count` pairs, for the CLI report.
    pub fn lines(&self) -> Vec<String> {
        ALL_CUES
            .iter()
            .filter(|c| self.counts[c.code() as usize] > 0)
            .map(|c| format!("{} {}", c.name(), self.counts[c.code() as usize]))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Playback: the bounded, lossy side
// ---------------------------------------------------------------------------

/// How many cues the consumer actually started, and how many it shed.
///
/// In a real-time race nothing is shed. In a max-speed replay at 5 000×
/// real time, most cues are — and that is the design, not a failure.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlaybackStats {
    pub started: u64,
    pub shed: u64,
}

impl PlaybackStats {
    /// Fraction of cues that reached a listener, 0..=1.
    pub fn delivery_ratio(&self) -> f64 {
        let total = self.started + self.shed;
        if total == 0 {
            1.0
        } else {
            self.started as f64 / total as f64
        }
    }
}

/// Nanoseconds in one playback window. Every cue a consumer starts occupies
/// an equal share of it, which is what turns `max_cues_per_s` into a rate
/// rather than a suggestion.
const WINDOW_NS: f64 = 1e9;

/// How deep the backlog may get before [`AudioConsumer::offer`] starts
/// evicting. Bounded on purpose: an accelerated race produces cues far faster
/// than any listener can hear them, and a queue that only grew would be a
/// memory leak that eventually stalls the simulation thread that feeds it.
pub const BACKLOG_LIMIT: usize = 256;

/// Accepts cues as fast as the race produces them and starts at most
/// `max_per_s` per wall-clock second, shedding the lowest priority first.
///
/// This type is what "audio events remain synchronized during accelerated
/// mode" actually means. Three properties, each pinned by a gate test:
///
/// * **Started cues keep the order the race produced them in.** The queue is
///   arrival-ordered, not priority-sorted, so a listener hears the race as it
///   unfolded rather than a reshuffle of it.
/// * **What is shed is the least important thing queued.** Eviction picks the
///   lowest priority, breaking ties towards the *newest* entry — an older cue
///   of equal importance is nearer its moment and worth more keeping.
/// * **The queue cannot grow without bound.** [`Self::pending`] is capped, so
///   a fast producer costs memory nothing and blocks the race nothing.
pub struct AudioConsumer {
    max_per_s: u32,
    /// Not yet started, in arrival order: front is next to play.
    pending: VecDeque<CueEvent>,
    stats: PlaybackStats,
    /// Cursor within the current window: the earliest ns at which the next cue
    /// may start. Reset by [`Self::begin_second`].
    budget_ns: f64,
}

impl AudioConsumer {
    pub fn new(max_per_s: u32) -> AudioConsumer {
        AudioConsumer {
            max_per_s,
            pending: VecDeque::new(),
            stats: PlaybackStats::default(),
            budget_ns: 0.0,
        }
    }

    pub fn stats(&self) -> PlaybackStats {
        self.stats
    }

    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Mark a fresh one-second window. `max_cues_per_s` of zero disables the
    /// ceiling entirely.
    pub fn begin_second(&mut self, max_per_s: u32) {
        self.max_per_s = max_per_s;
        self.budget_ns = 0.0;
    }

    /// Offer a cue. O(1) unless the backlog is full, and it never blocks.
    pub fn offer(&mut self, e: CueEvent) {
        self.pending.push_back(e);
        while self.pending.len() > BACKLOG_LIMIT {
            self.shed_one();
        }
    }

    /// Drop one queued cue, choosing the newest among those of the lowest
    /// priority present.
    ///
    /// Both halves of that choice matter. Dropping the *newest* keeps the queue
    /// in step with the race: what is shed is the cue furthest ahead, never the
    /// one about to be heard. Restricting the choice to the lowest priority
    /// means a start light is only ever discarded when there is nothing less
    /// important left to discard instead.
    fn shed_one(&mut self) {
        let victim = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(i, c)| (c.cue.priority(), std::cmp::Reverse(*i)))
            .map(|(i, _)| i)
            .expect("queue is non-empty");
        self.pending.remove(victim);
        self.stats.shed += 1;
    }

    /// Start as many cues as the rest of this second allows, in play order.
    ///
    /// `now_ns` is how far into the window the caller believes it is; a cue
    /// cannot start before that, so a consumer polling late in a window cannot
    /// burst the whole window's allowance at once.
    ///
    /// Cues are played in **arrival order**, oldest first, and never
    /// reordered by priority. "Synchronized" means in step with the race, and a
    /// listener hearing the green flag from tick 50 after a tyre-slip cue from
    /// tick 500 is not in step with anything. Priority earns its place at
    /// [`Self::shed_one`], where something has to be thrown away.
    pub fn drain(&mut self, now_ns: f64) -> Vec<CueEvent> {
        let cost_ns = if self.max_per_s == 0 {
            0.0
        } else {
            WINDOW_NS / self.max_per_s as f64
        };
        let mut out = Vec::new();
        while !self.pending.is_empty() {
            let start = self.budget_ns.max(now_ns);
            if start + cost_ns > WINDOW_NS {
                break;
            }
            self.budget_ns = start + cost_ns;
            let e = self.pending.pop_front().expect("non-empty");
            self.stats.started += 1;
            out.push(e);
        }
        out
    }

    /// Close the window: shed everything still queued, counting it. A cue that
    /// did not start within its own second was never going to be heard at the
    /// right moment, so holding it over would only make it late.
    ///
    /// Priority still decides *which* go first on the way out, via
    /// [`Self::shed_one`], so a green flag outlives a wind gust.
    pub fn expire(&mut self) {
        while !self.pending.is_empty() {
            self.shed_one();
        }
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        self.stats = PlaybackStats::default();
        self.budget_ns = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Asset validation
// ---------------------------------------------------------------------------

/// What `tools/make_sounds.py` asserts, re-checked in Rust so a badly
/// regenerated asset cannot quietly ship.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits: u16,
    pub frames: u32,
    pub duration_s: f64,
    pub peak: f64,
}

impl WavInfo {
    pub fn name(&self) -> String {
        format!(
            "{} Hz, {} ch, {} bit, {:.0} ms, peak {:.2}",
            self.sample_rate,
            self.channels,
            self.bits,
            self.duration_s * 1000.0,
            self.peak
        )
    }
}

/// Parse and validate a 16-bit PCM WAV against the vocabulary's rules.
///
/// The rules are what "validated" means here: every cue must be a short mono
/// clip at a known rate, loud enough to hear and quiet enough not to clip when
/// two of them overlap.
pub fn validate_wav(bytes: &[u8]) -> Result<WavInfo, String> {
    if bytes.len() < 44 {
        return Err(format!("wav is {} bytes, too short for a header", bytes.len()));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_string());
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u32, u16)> = None;
    let mut data: Option<(usize, usize)> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if body + size > bytes.len() {
            return Err("chunk runs past end of file".to_string());
        }
        if id == b"fmt " {
            if size < 16 {
                return Err("fmt chunk is truncated".to_string());
            }
            let channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap());
            let rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
            let bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap());
            let audio_format = u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap());
            if audio_format != 1 {
                return Err(format!("expected uncompressed PCM (format 1), got {audio_format}"));
            }
            fmt = Some((channels, rate, bits));
        } else if id == b"data" {
            data = Some((body, size));
        }
        // Chunks are word-aligned; a RIFF chunk body is padded to an even size.
        pos = body + size + (size & 1);
    }
    let (channels, sample_rate, bits) =
        fmt.ok_or_else(|| "no fmt chunk".to_string())?;
    let (dstart, dsize) = data.ok_or_else(|| "no data chunk".to_string())?;
    if channels != 1 {
        return Err(format!("expected mono, got {channels} channels"));
    }
    if sample_rate != 48_000 {
        return Err(format!("expected 48000 Hz, got {sample_rate}"));
    }
    if bits != 16 {
        return Err(format!("expected 16-bit, got {bits}-bit"));
    }
    if dsize < 2 {
        return Err("data chunk is empty".to_string());
    }
    let frames = (dsize / 2) as u32;
    let duration_s = frames as f64 / sample_rate as f64;
    if duration_s > 1.5 {
        return Err(format!("clip is {:.0} ms, longer than the 1500 ms limit", duration_s * 1000.0));
    }
    if duration_s < 0.02 {
        return Err(format!("clip is {:.0} ms, too short to hear", duration_s * 1000.0));
    }
    let mut peak = 0.0f64;
    for i in (0..dsize).step_by(2) {
        let v = i16::from_le_bytes([bytes[dstart + i], bytes[dstart + i + 1]]) as f64 / 32768.0;
        peak = peak.max(v.abs());
    }
    if peak > 0.98 {
        return Err(format!("peak {:.3} would clip", peak));
    }
    if peak < 0.05 {
        return Err(format!("peak {:.3} is too quiet", peak));
    }
    Ok(WavInfo {
        sample_rate,
        channels,
        bits,
        frames,
        duration_s,
        peak,
    })
}