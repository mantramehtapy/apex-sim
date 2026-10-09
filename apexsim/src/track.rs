//! Circuit geometry: the built-in analytic stadium, and any user-drawn
//! closed loop.
//!
//! Two construction paths, one API:
//!
//! * [`Track::new`] — the Phase 1 stadium (two straights + two 180° arcs),
//!   still analytic to the last ulp so every replay hash and validation
//!   fixture ever recorded stays bit-exact.
//! * [`Track::from_centerline`] — a closed loop of control points, smoothed
//!   with a centripetal Catmull-Rom spline and resampled to a uniform
//!   arc-length table that then answers `point_at`/`curvature_at`.
//!
//! Everything downstream (drivers, physics, race logic, policy, stewarding)
//! consumes only `point_at`/`curvature_at`/`project`/`speed_limit_at`, all of
//! which were already shape-agnostic in contract — so a custom circuit needs
//! no changes anywhere else.
//!
//! s measures distance along the centreline from the start/finish line. For
//! the stadium that is the middle of the bottom straight, heading +x; for a
//! custom loop it is wherever the loop's first control point is, heading
//! along the loop.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct Track {
    pub straight_len: f64, // m, each straight (stadium; 0 for a custom loop)
    pub radius: f64,       // m, each 180° corner (stadium; 0 for a custom loop)
    pub half_width: f64,   // m, from centreline to track edge
    samples: Vec<Sample>,  // uniform centreline samples for projection
    /// The smoothed centreline table of a custom circuit. `None` = the
    /// analytic stadium, which answers geometry from closed-form formulas.
    custom: Option<CustomGeom>,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    x: f64,
    y: f64,
    s: f64,
}

/// Uniform arc-length table of a custom circuit: position, unwrapped heading
/// and curvature per sample. Heading is stored UNWRAPPED (continuous across
/// the seam) so interpolation never jumps by 2π; one full lap turns by
/// exactly ±2π.
#[derive(Clone, Debug)]
struct CustomGeom {
    length: f64,
    pts: Vec<[f64; 4]>, // x, y, heading, curvature — N entries, uniform ds
}

impl Track {
    const N_SAMPLES: usize = 2048;

    pub fn new(straight_len: f64, radius: f64, half_width: f64) -> Self {
        let mut track = Track {
            straight_len,
            radius,
            half_width,
            samples: Vec::with_capacity(Self::N_SAMPLES),
            custom: None,
        };
        let len = track.length();
        for i in 0..Self::N_SAMPLES {
            let s = i as f64 * len / Self::N_SAMPLES as f64;
            let (x, y, _) = track.raw_point_at(s);
            track.samples.push(Sample { x, y, s });
        }
        track
    }

    /// Build a circuit from control points of a closed loop.
    ///
    /// The loop is smoothed with a closed Catmull-Rom spline through the
    /// control points and resampled to a uniform arc-length table. Only the
    /// STRUCTURAL laws are enforced: enough points to close a loop, actually
    /// a closed loop (total turn ±2π), and not degenerate to a speck. Hairpins
    /// tighter than the width, self-crossings and short laps are all accepted
    /// — the editor draws it, the engine races it. Errors name the problem —
    /// a drawing that cannot form a circuit must be refused at load, not
    /// discovered mid-race.
    pub fn from_centerline(control: &[[f64; 2]], half_width: f64) -> Result<Track, String> {
        if control.len() < 3 {
            return Err(format!(
                "a circuit needs at least 3 control points to close a loop, got {}",
                control.len()
            ));
        }
        if !(1.0..=30.0).contains(&half_width) {
            return Err(format!("half_width {half_width} outside 1..30 m"));
        }

        // --- Catmull-Rom through the closed loop, densified 40x per segment.
        const STEPS: usize = 40;
        let n = control.len();
        let mut dense: Vec<[f64; 2]> = Vec::with_capacity(n * STEPS);
        for i in 0..n {
            let p0 = control[(i + n - 1) % n];
            let p1 = control[i];
            let p2 = control[(i + 1) % n];
            let p3 = control[(i + 2) % n];
            for k in 0..STEPS {
                let t = k as f64 / STEPS as f64;
                let t2 = t * t;
                let t3 = t2 * t;
                // Standard (uniform) Catmull-Rom basis.
                let x = 0.5
                    * ((2.0 * p1[0])
                        + (-p0[0] + p2[0]) * t
                        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3);
                let y = 0.5
                    * ((2.0 * p1[1])
                        + (-p0[1] + p2[1]) * t
                        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3);
                dense.push([x, y]);
            }
        }

        // --- Resample to uniform arc length.
        let mut seg: Vec<f64> = Vec::with_capacity(dense.len() + 1);
        let mut total = 0.0;
        for i in 0..dense.len() {
            seg.push(total);
            let a = dense[i];
            let b = dense[(i + 1) % dense.len()];
            total += ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        }
        // Degeneracy floor, not a length rule: a 60 m kart loop races (fast),
        // but a speck of near-identical points would make ds≈0 and poison the
        // arc-length table.
        if total < 50.0 {
            return Err(format!(
                "the drawing is only {total:.1} m around — too small to form a circuit"
            ));
        }
        const N: usize = Track::N_SAMPLES;
        let ds = total / N as f64;
        let mut pts: Vec<[f64; 2]> = Vec::with_capacity(N);
        let mut seg_i = 0usize;
        for k in 0..N {
            let target = k as f64 * ds;
            while seg_i + 1 < seg.len() && seg[seg_i + 1] <= target {
                seg_i += 1;
            }
            let a = dense[seg_i];
            // The table records each segment's START; the final entry closes
            // the loop back to dense[0], so its end is `total` itself.
            let end = if seg_i + 1 < seg.len() { seg[seg_i + 1] } else { total };
            let b = dense[(seg_i + 1) % dense.len()];
            let span = (end - seg[seg_i]).max(1e-9);
            let f = (target - seg[seg_i]) / span;
            pts.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
        }

        // --- Heading (unwrapped) and curvature by finite difference.
        let mut heading = vec![0.0f64; N];
        let mut prev_h = 0.0f64;
        for i in 0..N {
            let a = pts[(i + N - 1) % N];
            let b = pts[(i + 1) % N];
            let mut h = (b[1] - a[1]).atan2(b[0] - a[0]);
            // Unwrap: choose the 2π branch nearest the previous sample.
            while h - prev_h > std::f64::consts::PI {
                h -= 2.0 * std::f64::consts::PI;
            }
            while h - prev_h < -std::f64::consts::PI {
                h += 2.0 * std::f64::consts::PI;
            }
            heading[i] = h;
            prev_h = h;
        }
        // A closed loop must turn by exactly ±2π; a drawing that doubles back
        // (an out-and-back line, a scatter) fails here — the one structural
        // law left: what is drawn must actually be a circuit.
        let turn = heading[N - 1] - heading[0];
        if turn.abs() < std::f64::consts::PI {
            return Err(format!(
                "the loop turns by only {turn:.2} rad over a lap — control points must form a single closed circuit (a full lap turns ±2π)"
            ));
        }

        // Curvature by central difference — with the SEAM handled: heading
        // is unwrapped, so h[N-1] sits a full ±2π away from h[0], and a naive
        // modulo neighbour difference would book that lap's worth of turn at
        // one sample (a 0.2 m "radius"). Extend one sample past both ends by
        // the lap's total turn instead.
        let turn = heading[N - 1] - heading[0];
        let h_ext = |i: i64| -> f64 {
            if i < 0 {
                heading[(i + N as i64) as usize] - turn
            } else if i as usize >= N {
                heading[i as usize - N] + turn
            } else {
                heading[i as usize]
            }
        };
        let mut table: Vec<[f64; 4]> = Vec::with_capacity(N);
        for i in 0..N {
            let k = (h_ext(i as i64 + 1) - h_ext(i as i64 - 1)) / (2.0 * ds);
            table.push([pts[i][0], pts[i][1], heading[i], k]);
        }

        // Self-crossings and sub-width hairpins are legal since the
        // draw-anything relaxation: the surface may fold over its own paint,
        // but the centerline (what the cars actually follow) is well-defined
        // everywhere, so the race proceeds. No refusal, no warning needed.

        let mut samples = Vec::with_capacity(N);
        for (k, p) in table.iter().enumerate() {
            samples.push(Sample { x: p[0], y: p[1], s: k as f64 * ds });
        }
        Ok(Track {
            straight_len: 0.0,
            radius: 0.0,
            half_width,
            samples,
            custom: Some(CustomGeom { length: total, pts: table }),
        })
    }

    /// Total centreline length, m.
    pub fn length(&self) -> f64 {
        match &self.custom {
            Some(g) => g.length,
            None => 2.0 * self.straight_len + 2.0 * std::f64::consts::PI * self.radius,
        }
    }

    /// Exact centreline position and heading (rad, ccw from +x) at distance s.
    pub fn point_at(&self, s: f64) -> (f64, f64, f64) {
        let s = s.rem_euclid(self.length());
        match &self.custom {
            None => self.raw_point_at(s),
            Some(g) => {
                let n = g.pts.len();
                let f = s / g.length * n as f64;
                let i = (f as usize).min(n - 1);
                let a = g.pts[i];
                let b_raw = g.pts[(i + 1) % n];
                // Heading is stored UNWRAPPED (table[n-1] ~= table[0] + 2pi),
                // so the seam destination must be lifted into a's branch: a
                // raw lerp table[n-1] -> table[0] would swing the tangent by
                // the full lap turn across the last few metres of every lap.
                let b_heading = if i + 1 == n {
                    let d = b_raw[2] - a[2];
                    let d = d - 2.0 * std::f64::consts::PI * (d / std::f64::consts::PI / 2.0).round();
                    a[2] + d
                } else {
                    b_raw[2]
                };
                let t = f - i as f64;
                let h = a[2] + (b_heading - a[2]) * t;
                // Returned heading is WRAPPED to [0, 2pi) — the same contract
                // as the analytic stadium, whose tangents are always wrapped.
                // Consumers compare car heading against this; the car's own
                // heading is continuous and only ever differs from the tangent
                // by a wrapped amount.
                (
                    a[0] + (b_raw[0] - a[0]) * t,
                    a[1] + (b_raw[1] - a[1]) * t,
                    h.rem_euclid(2.0 * std::f64::consts::PI),
                )
            }
        }
    }

    fn raw_point_at(&self, s: f64) -> (f64, f64, f64) {
        let l = self.straight_len;
        let r = self.radius;
        let arc = std::f64::consts::PI * r;
        if s < l {
            // Bottom straight, heading +x.
            (s, 0.0, 0.0)
        } else if s < l + arc {
            // Right-end corner, centre (l, r), turning left.
            let phi = (s - l) / r;
            (l + r * phi.sin(), r - r * phi.cos(), phi)
        } else if s < 2.0 * l + arc {
            // Top straight (driven right-to-left), heading pi.
            let t = s - l - arc;
            (l - t, 2.0 * r, std::f64::consts::PI)
        } else {
            // Left-end corner, centre (0, r), turning left.
            let psi = (s - 2.0 * l - arc) / r;
            (-r * psi.sin(), r + r * psi.cos(), std::f64::consts::PI + psi)
        }
    }

    /// Signed curvature at s (1/m; positive = left turn).
    pub fn curvature_at(&self, s: f64) -> f64 {
        let s = s.rem_euclid(self.length());
        match &self.custom {
            Some(g) => {
                let n = g.pts.len();
                let f = s / g.length * n as f64;
                let i = (f as usize).min(n - 1);
                let a = g.pts[i][3];
                let b = g.pts[(i + 1) % n][3];
                a + (b - a) * (f - i as f64)
            }
            None => {
                let l = self.straight_len;
                let arc = std::f64::consts::PI * self.radius;
                let in_corner = (s >= l && s < l + arc) || s >= 2.0 * l + arc;
                if in_corner {
                    1.0 / self.radius
                } else {
                    0.0
                }
            }
        }
    }

    /// Maximum speed that keeps lateral acceleration within `a_lat_max` at s.
    pub fn speed_limit_at(&self, s: f64, a_lat_max: f64) -> f64 {
        let k = self.curvature_at(s);
        if k <= 1e-9 {
            f64::INFINITY
        } else {
            (a_lat_max / k).sqrt()
        }
    }

    /// Project a world position onto the centreline.
    /// Returns (s, signed lateral offset in m — positive = left of centreline).
    pub fn project(&self, x: f64, y: f64) -> (f64, f64) {
        // Nearest uniform sample, then refine along the local tangent.
        let mut best_i = 0;
        let mut best_d2 = f64::INFINITY;
        for (i, sm) in self.samples.iter().enumerate() {
            let dx = x - sm.x;
            let dy = y - sm.y;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best_i = i;
            }
        }
        let sm = self.samples[best_i];
        let (_, _, heading) = self.point_at(sm.s);
        let (tx, ty) = (heading.cos(), heading.sin());
        let along = (x - sm.x) * tx + (y - sm.y) * ty;
        let lateral = -(x - sm.x) * ty + (y - sm.y) * tx;
        let s = (sm.s + along).rem_euclid(self.length());
        (s, lateral)
    }

    pub fn is_off_track(&self, lateral: f64) -> bool {
        lateral.abs() > self.half_width
    }
}

impl Default for Track {
    fn default() -> Self {
        // Centreline length = 600 + 2*pi*80 ≈ 1102.65 m.
        Track::new(300.0, 80.0, 7.0)
    }
}

// ---------------------------------------------------------------------------
// Track identity (Phase 10)
// ---------------------------------------------------------------------------

/// Bumped by hand whenever the physics-facing geometry or the sampling of it
/// changes. It is part of the batch configuration and is stamped into every
/// run log, so a tournament scored against an older circuit is never silently
/// compared with one scored against a newer circuit.
pub const TRACK_VERSION: u32 = 1;

/// A user-defined circuit: control points the engine smooths into the racing
/// loop, plus where the pit lane sits. Embedded in a [`TrackSpec`] so a run
/// log or batch config carries everything needed to rebuild the circuit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomTrack {
    /// File/preset name, for display and log identity.
    pub name: String,
    /// Control points of the closed loop (>= 6). The engine smooths these
    /// with a Catmull-Rom spline — they are corners-to-be, not the tarmac.
    pub points: Vec<[f64; 2]>,
    /// Pit lane placement as fractions of lap length. `None` = defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pit: Option<PitFractions>,
}

/// Pit entry/box/exit as fractions of the lap. The engine turns these into
/// absolute s values against the circuit length and validates that the
/// region is straight enough to house a pit lane.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PitFractions {
    pub entry: f64,
    #[serde(rename = "box")]
    pub box_frac: f64,
    pub exit: f64,
}

impl Default for PitFractions {
    fn default() -> Self {
        PitFractions { entry: 0.10, box_frac: 0.15, exit: 0.22 }
    }
}

/// A circuit a batch can name: geometry plus the identity of that geometry.
///
/// For the built-in stadium the three numbers and the version pin the layout
/// exactly. For a custom circuit the `custom` block carries the control
/// points themselves, so a log or batch is self-contained: the same JSON
/// rebuilds the same tarmac to the last resampled sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackSpec {
    pub version: u32,
    pub straight_len: f64,
    pub radius: f64,
    pub half_width: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<CustomTrack>,
}

impl Default for TrackSpec {
    fn default() -> Self {
        TrackSpec {
            version: TRACK_VERSION,
            straight_len: 300.0,
            radius: 80.0,
            half_width: 7.0,
            custom: None,
        }
    }
}

/// Embedded preset circuits (`apexsim/tracks/*.json`), compiled in so a name
/// works from any working directory.
pub const TRACK_PRESETS: &[(&str, &str)] = &[
    ("paperclip", include_str!("../tracks/paperclip.json")),
    ("kidney", include_str!("../tracks/kidney.json")),
    ("trioval", include_str!("../tracks/trioval.json")),
    ("roadcourse", include_str!("../tracks/roadcourse.json")),
];

fn default_half_width() -> f64 {
    7.0
}

/// Resolve a `--track` argument: a preset name, or a path to a track JSON
/// (anything containing a path separator or ending in `.json`).
pub fn resolve_track_spec(arg: &str) -> Result<TrackSpec, String> {
    let looks_like_path = arg.contains('/') || arg.contains('\\') || arg.ends_with(".json");
    let spec = if looks_like_path {
        let text = std::fs::read_to_string(arg)
            .map_err(|e| format!("--track: cannot read {arg}: {e}"))?;
        TrackSpec::from_custom_json(&text).map_err(|e| format!("--track: {arg}: {e}"))?
    } else {
        let preset = TRACK_PRESETS
            .iter()
            .find(|(name, _)| *name == arg.to_ascii_lowercase())
            .ok_or_else(|| {
                format!(
                    "--track: unknown preset '{arg}' (presets: {})",
                    TRACK_PRESETS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
                )
            })?;
        TrackSpec::from_custom_json(preset.1).map_err(|e| format!("preset {arg}: {e}"))?
    };
    validate_track_spec(&spec).map_err(|e| format!("--track {arg}: {e}"))?;
    Ok(spec)
}

/// The load-time law for any circuit spec: it must build, and a custom pit
/// lane must fit the circuit it sits on. Shared by `--track`, the track-file
/// endpoint and the dashboard's staged track change, so all three refuse a
/// bad circuit in the same words.
pub fn validate_track_spec(spec: &TrackSpec) -> Result<(), String> {
    // Refusal happens here, where the user can read it — a track that cannot
    // be driven must not survive the load and surface mid-race instead.
    let t = spec.build().map_err(|e| format!("not drivable: {e}"))?;
    if let Some(c) = &spec.custom {
        let len = t.length();
        // An EXPLICIT pit block that does not fit is an authoring error and
        // is refused. A file without one gets the engine defaults, and a
        // circuit too short for the defaults simply races without a pit
        // lane (Race::new disables it) — refusing every sub-1200 m loop
        // would make short circuits unspeakable.
        if let Some(f) = c.pit {
            let pc = crate::race::PitConfig {
                entry_s: f.entry * len,
                box_s: f.box_frac * len,
                exit_s: f.exit * len,
                ..crate::race::PitConfig::default()
            };
            if !pc.usable(len) {
                return Err(format!(
                    "pit lane does not fit (entry at {:.0} m needs >{:.0} m of braking room, and entry<box<exit<{len:.0}))",
                    pc.entry_s, pc.brake_zone_m
                ));
            }
        }
    }
    Ok(())
}

impl TrackSpec {
    /// Parse a track-file JSON (what `tracks/*.json` and the dashboard editor
    /// write) into a spec with `version`/stadium numbers zeroed — the custom
    /// block is the identity. The file's `half_width` is honoured: it is part
    /// of the circuit (validation thresholds and the drawn tarmac both use
    /// it), and ignoring it silently widened or narrowed what the author drew.
    pub fn from_custom_json(text: &str) -> Result<TrackSpec, String> {
        #[derive(Deserialize)]
        struct TrackFile {
            name: String,
            #[serde(default = "default_half_width")]
            half_width: f64,
            points: Vec<[f64; 2]>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pit: Option<PitFractions>,
        }
        let f: TrackFile = serde_json::from_str(text)
            .map_err(|e| format!("track file is not valid JSON: {e}"))?;
        if f.name.trim().is_empty() {
            return Err("track file has an empty name".into());
        }
        if !(1.0..=30.0).contains(&f.half_width) {
            return Err(format!("half_width {} outside 1..30 m", f.half_width));
        }
        Ok(TrackSpec {
            version: 2,
            straight_len: 0.0,
            radius: 0.0,
            half_width: f.half_width,
            custom: Some(CustomTrack { name: f.name, points: f.points, pit: f.pit }),
        })
    }

    /// Human-readable identity, e.g. `v1/300.0/80.0/7.0` for the stadium or
    /// `v2/kidney/a1b2…` for a custom circuit. Printed in the report and
    /// compared by [`TrackSpec::matches`].
    pub fn label(&self) -> String {
        match &self.custom {
            Some(c) => format!("v{}/{}/{:08x}", self.version, c.name, self.content_hash()),
            None => format!(
                "v{}/{}/{}/{}",
                self.version, self.straight_len, self.radius, self.half_width
            ),
        }
    }

    /// A stable digest of the circuit definition: control-point coordinates
    /// for a custom loop, the stadium numbers otherwise. Two runs are on the
    /// same tarmac iff this matches.
    pub fn content_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100_0000_01b3);
        };
        mix(self.version as u64);
        mix(self.straight_len.to_bits());
        mix(self.radius.to_bits());
        mix(self.half_width.to_bits());
        if let Some(c) = &self.custom {
            mix(c.name.len() as u64);
            for p in &c.points {
                mix(p[0].to_bits());
                mix(p[1].to_bits());
            }
            if let Some(pit) = c.pit {
                mix(pit.entry.to_bits());
                mix(pit.box_frac.to_bits());
                mix(pit.exit.to_bits());
            }
        }
        h
    }

    /// Whether this spec selects the engine's current built-in layout. A spec
    /// that does not is rejected rather than quietly run on the wrong circuit.
    pub fn matches_builtin(&self) -> bool {
        self.custom.is_none() && {
            let d = TrackSpec::default();
            self.version == d.version
                && self.straight_len == d.straight_len
                && self.radius == d.radius
                && self.half_width == d.half_width
        }
    }

    /// Build the circuit. Custom tracks are validated here, so a bad track
    /// file is refused where the user can read the error.
    pub fn build(&self) -> Result<Track, String> {
        match &self.custom {
            Some(c) => Track::from_centerline(&c.points, self.half_width),
            None => Ok(Track::new(self.straight_len, self.radius, self.half_width)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_closed_and_oriented() {
        let t = Track::default();
        let l = t.length();
        let (x0, y0, h0) = t.point_at(0.0);
        assert!((x0 - 0.0).abs() < 1e-9 && y0.abs() < 1e-9 && h0.abs() < 1e-9);
        // One full lap returns to the start with heading wrapped to 0.
        let (x1, y1, h1) = t.point_at(l);
        assert!((x1 - 0.0).abs() < 1e-9 && y1.abs() < 1e-9);
        assert!((h1.rem_euclid(2.0 * std::f64::consts::PI)).abs() < 1e-9);
        // Mid-top straight sits at y = 160, heading pi.
        let s_mid_top = 300.0 + std::f64::consts::PI * 80.0 + 150.0;
        let (xm, ym, hm) = t.point_at(s_mid_top);
        assert!((ym - 160.0).abs() < 1e-9 && (xm - 150.0).abs() < 1e-6);
        assert!((hm - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn projection_round_trips() {
        let t = Track::default();
        for s in [0.0, 123.4, 300.0, 500.0, 800.0, 1050.0] {
            let (x, y, _) = t.point_at(s);
            let (ps, lat) = t.project(x, y);
            assert!(
                (ps - s).abs() < 0.6,
                "s={s} projected to {ps} (lat {lat})"
            );
            assert!(lat.abs() < 0.5, "lateral {lat} at s={s}");
        }
    }

    #[test]
    fn corner_speed_limit() {
        let t = Track::default();
        let v = t.speed_limit_at(400.0, 11.0); // mid-corner
        assert!((v - (11.0 * 80.0f64).sqrt()).abs() < 1e-9);
        assert!(t.speed_limit_at(100.0, 11.0).is_infinite()); // straight
    }

    #[test]
    fn a_track_spec_round_trips_and_identifies_the_builtin() {
        let spec = TrackSpec::default();
        assert!(spec.matches_builtin());
        assert!(spec.build().unwrap().length() > 1102.0);
        let json = serde_json::to_string(&spec).expect("serialise");
        let back: TrackSpec = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(back, spec);
        // A different circuit version is recognised as *not* the built-in one
        // rather than being run on the stadium and reported under its own name.
        let other = TrackSpec { version: TRACK_VERSION + 1, ..spec.clone() };
        assert!(!other.matches_builtin());
        assert_ne!(other.label(), spec.label());
    }
}
