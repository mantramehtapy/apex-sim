//! The learned-driver contract: what a policy sees, what it may say, and how
//! its answer becomes controls.
//!
//! One observation builder serves both the training environment
//! ([`crate::gym`], a single car) and the race ([`crate::race`], twenty of
//! them). That is deliberate. If the environment and the race each built their
//! own feature vector, a policy could pass a parity test against the
//! environment and then meet a different world in a race — and the parity
//! check would be worthless. There is one builder, one normaliser, and one
//! action decoder, and every caller goes through them.
//!
//! ## Units and normalisation
//!
//! Observations are SI and then divided by the scale in [`ObsSpec`], so every
//! channel is roughly in `[-1, 1]`. The scales are data, not literals: they are
//! serialised into the environment spec and echoed into every training and
//! evaluation artifact, so a policy trained against one set of scales cannot
//! be silently evaluated against another.
//!
//! ## Action
//!
//! Three values, all in `[-1, 1]`:
//!
//! 0. steering — multiplied by the car's maximum steer angle;
//! 1. longitudinal — positive is throttle, negative is brake. One pedal, not
//!    two, so the policy cannot demand both at once.
//!
//! Decoding clamps and never mixes: a positive longitudinal value sets the
//! throttle and zeroes the brake, exactly as the local driver does.

use crate::physics::{CarState, Controls};
use crate::track::Track;
use serde::{Deserialize, Serialize};

/// Number of observation channels. Fixed, and asserted against the ONNX
/// graph's input width before anything runs.
pub const OBS_DIM: usize = 22;

/// Number of action channels: steer, throttle(+)/brake(−).
pub const ACTION_DIM: usize = 3;

/// Channel names, in observation order. Kept beside the scales rather than
/// inside them: names are documentation for a human reading an artifact, and
/// a policy's meaning lives in the numbers. `OBS_NAMES.len() == OBS_DIM` is
/// asserted by the gate.
pub const OBS_NAMES: [&str; OBS_DIM] = [
    "v_long",
    "v_body",
    "lateral_offset",
    "heading_error",
    "curvature_here",
    "curvature_50m",
    "curvature_100m",
    "curvature_200m",
    "curvature_300m",
    "steer_angle",
    "steer_cmd",
    "v_lat",
    "yaw_rate",
    "lat_accel",
    "gap_ahead",
    "closing_ahead",
    "lateral_ahead",
    "gap_behind",
    "lap_fraction",
    "budget_left",
    "grip_scale",
    "v_target",
];

/// The scales that define an observation.
///
/// `scale` is the interface: two runs with different scales are different
/// experiments, so this struct is written into the training and evaluation
/// artifacts and compared before a result is believed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObsSpec {
    /// Divisor for each channel, in order. SI units.
    pub scale: [f64; OBS_DIM],
}

impl Default for ObsSpec {
    fn default() -> Self {
        ObsSpec {
            scale: [
                75.0, // 0  forward speed, m/s (v_straight_max)
                45.0, // 1  speed along the car, m/s
                7.0,  // 2  lateral offset from the centreline, m (half-width)
                0.5,  // 3  heading error vs the track tangent, rad
                0.05, // 4  track curvature here, 1/m
                0.05, // 5  track curvature 50 m ahead, 1/m
                0.05, // 6  track curvature 100 m ahead, 1/m
                0.05, // 7  track curvature 200 m ahead, 1/m
                0.05, // 8  track curvature 300 m ahead, 1/m
                0.45, // 9  front wheel angle, rad (max_steer)
                0.45, // 10 commanded steering, rad (max_steer)
                75.0, // 11 lateral speed, m/s
                4.0,  // 12 yaw rate, rad/s
                12.0, // 13 lateral acceleration, m/s² (a_lat_max)
                60.0, // 14 gap to the car ahead, m
                75.0, // 15 closing speed on the car ahead, m/s
                7.0,  // 16 lateral gap to the car ahead, m (half-width)
                60.0, // 17 gap to the car behind, m
                1.0,  // 18 lap fraction of the track, 0..1
                1.0,  // 19 step budget remaining, 0..1
                1.0,  // 20 available grip as a fraction of new, 0..1
                75.0, // 21 driver's target speed, m/s (v_straight_max)
            ],
        }
    }
}

impl ObsSpec {
    pub fn names(&self) -> [&'static str; OBS_DIM] {
        OBS_NAMES
    }

    /// True when two runs are comparable — the gate refuses to compare a
    /// policy evaluated under different scales with one trained under others.
    pub fn same_as(&self, other: &ObsSpec) -> bool {
        self.scale
            .iter()
            .zip(other.scale.iter())
            .all(|(a, b)| (a - b).abs() <= 1e-12 * a.abs().max(1.0))
    }
}

/// What the environment knows about the car that is not in `CarState`.
///
/// The race fills the traffic half from the leaderboard; a lone car has none,
/// and says so with an explicit sentinel rather than a zero that reads as
/// "a car is exactly touching me".
#[derive(Clone, Copy, Debug, Default)]
pub struct TrafficContext {
    /// (gap along the track, lateral gap, closing speed) for the car ahead.
    pub ahead: Option<(f64, f64, f64)>,
    /// Gap along the track for the car behind.
    pub behind: Option<f64>,
    /// 0 at the start of an episode, 1 at the end.
    pub lap_fraction: f64,
    /// 1.0 at reset, falling to 0.0 at the step budget.
    pub budget_left: f64,
    /// Grip available to this car as a fraction of a fresh tire — compound,
    /// wear and surface, already multiplied together.
    ///
    /// This is not a nice-to-have. The local driver scales its own speed
    /// target, brake anticipation and lateral limit by exactly this number, so
    /// without it the teacher's pedal choice is not a function of the
    /// observation: the same corner at the same speed needs full brake on a
    /// worn tire and none on a fresh one, and a policy that cannot tell them
    /// apart can only learn the average of the two. Measured on the Phase 12
    /// run, omitting it left the pedal channel at RMSE 0.32 with the error
    /// concentrated in a large bias toward coasting.
    pub grip_scale: f64,
    /// The speed the local driver is aiming for right now, m/s.
    ///
    /// This is the driver's entire longitudinal decision, and it is a minimum
    /// over roughly 65 speed-limit samples spread at 5 m intervals across a
    /// braking horizon that grows with the car's speed. No small set of
    /// curvature lookups recovers a min over that many samples, which is why
    /// the steering channel cloned cleanly (RMSE 0.004) while the pedal channel
    /// sat at 0.32 no matter how wide the network was — 330k training rows,
    /// train error equal to val error, and a memorizing k-NN predictor scoring
    /// *worse* (0.36). That is the signature of a target that is not a function
    /// of the observation, not of a model too small to learn it. Handing the
    /// policy the number the teacher actually used fixes it at the source,
    /// and it is information a real driver has: it comes off the track map.
    pub v_target: f64,
}

/// Sentinel for "there is no car there". A policy reading 0.0 for a gap would
/// be told a car is bumper to bumper; -1 in the normalised space says absent.
pub const NO_CAR: f64 = -1.0;

/// Build the observation vector for one car.
///
/// Deterministic and side-effect free: it reads the car, the track and the
/// supplied context, and nothing else. That is what lets the same call serve
/// the trainer, the race, and the parity check.
pub fn observe(
    car: &CarState,
    track: &Track,
    s: f64,
    heading_error: f64,
    steer_cmd: f64,
    ctx: &TrafficContext,
    spec: &ObsSpec,
) -> Vec<f32> {
    let k = |i: usize| spec.scale[i];
    let mut o = [0.0f32; OBS_DIM];
    let put = |o: &mut [f32; OBS_DIM], i: usize, v: f64| {
        o[i] = (v / k(i)) as f32;
    };

    let (ahead_gap, ahead_lat, ahead_close) = ctx.ahead.unwrap_or((0.0, 0.0, 0.0));
    let has_ahead = ctx.ahead.is_some();

    put(&mut o, 0, car.vel_long);
    put(&mut o, 1, (car.vel_long * car.vel_long + car.vel_lat * car.vel_lat).sqrt());
    let (_, lateral) = track.project(car.pos_x, car.pos_y);
    put(&mut o, 2, lateral);
    put(&mut o, 3, heading_error);
    // Curvature is sampled out to 300 m because that is how far the local
    // driver looks before choosing a brake point: its horizon is
    // v²/(2·a_brake) + margin, which at 75 m/s is roughly 350 m. Sampling only
    // the near field left the driver's own throttle target genuinely
    // unobservable — two states with identical 0/50/100 m curvature can need
    // opposite pedals because of a corner the observation never saw, and no
    // amount of training fixes a label that is not a function of the input.
    put(&mut o, 4, track.curvature_at(s));
    put(&mut o, 5, track.curvature_at(s + 50.0));
    put(&mut o, 6, track.curvature_at(s + 100.0));
    put(&mut o, 7, track.curvature_at(s + 200.0));
    put(&mut o, 8, track.curvature_at(s + 300.0));
    put(&mut o, 9, car.steer);
    put(&mut o, 10, steer_cmd);
    put(&mut o, 11, car.vel_lat);
    put(&mut o, 12, car.yaw_rate);
    put(&mut o, 13, car.vel_long * car.yaw_rate);
    // An absent car is -1 after normalisation, not 0.
    o[14] = if has_ahead { (ahead_gap / k(14)) as f32 } else { NO_CAR as f32 };
    put(&mut o, 15, ahead_close);
    o[16] = if has_ahead { (ahead_lat / k(16)) as f32 } else { NO_CAR as f32 };
    o[17] = match ctx.behind {
        Some(g) => (g / k(17)) as f32,
        None => NO_CAR as f32,
    };
    put(&mut o, 18, ctx.lap_fraction);
    put(&mut o, 19, ctx.budget_left);
    put(&mut o, 20, ctx.grip_scale);
    put(&mut o, 21, ctx.v_target);
    debug_assert_eq!(o.len(), OBS_DIM);
    o.to_vec()
}

/// Signed heading error: how far the car points away from the track tangent.
/// Positive is left of the tangent.
pub fn heading_error(car: &CarState, track: &Track, s: f64) -> f64 {
    let (_, _, theta) = track.point_at(s);
    let mut e = car.heading - theta;
    // Wrap to (-pi, pi] so the error never jumps when the tangent wraps.
    while e > std::f64::consts::PI {
        e -= 2.0 * std::f64::consts::PI;
    }
    while e <= -std::f64::consts::PI {
        e += 2.0 * std::f64::consts::PI;
    }
    e
}

/// Turn a policy output into controls.
///
/// Clamped, not trusted: a policy that saturates every channel is a policy
/// that will spin the car, and the decoder must not be the thing that turns an
/// out-of-range float into a NaN in the steering actuator.
pub fn decode(action: &[f32], max_steer: f64) -> Controls {
    let clamp = |v: f32| (v as f64).clamp(-1.0, 1.0);
    let steer = clamp(action.first().copied().unwrap_or(0.0));
    let long = clamp(action.get(1).copied().unwrap_or(0.0));
    let mut ctrl = Controls {
        throttle: long.max(0.0),
        brake: (-long).max(0.0),
        steer_cmd: steer * max_steer,
    };
    if ctrl.brake > 0.0 {
        ctrl.throttle = 0.0; // never both pedals
    }
    ctrl
}

/// Which controller drives a car.
pub enum Controller {
    /// The Phase 1/4 deterministic line-following driver. The baseline every
    /// learned driver is measured against, and the fallback whenever the
    /// learned driver is absent, fails, or overruns its latency budget.
    Baseline,
    /// A learned policy loaded from ONNX.
    Learned(crate::onnx::OnnxSession),
    /// Both, recording the difference. Used by the parity gate and by the
    /// latency benchmark; never used to drive a result.
    Shadow(Box<Controller>, Box<Controller>),
}

impl Controller {
    pub fn name(&self) -> &'static str {
        match self {
            Controller::Baseline => "baseline",
            Controller::Learned(_) => "learned",
            Controller::Shadow(..) => "shadow",
        }
    }

    /// Run a policy, falling back to `baseline` on any error.
    ///
    /// The fallback is not a convenience. A learned driver that can wedge the
    /// 250 Hz loop when its file is missing has taken away the deterministic
    /// engine's main property, so an error is reported and the baseline
    /// drives instead.
    pub fn act(
        &mut self,
        obs: &[f32],
        max_steer: f64,
        baseline: &mut dyn FnMut() -> Controls,
    ) -> (Controls, Option<String>) {
        match self {
            Controller::Baseline => (baseline(), None),
            Controller::Learned(sess) => match sess.run(obs) {
                Ok(a) if a.len() >= ACTION_DIM => (decode(&a, max_steer), None),
                Ok(a) => (
                    baseline(),
                    Some(format!("policy returned {} actions, expected {ACTION_DIM}", a.len())),
                ),
                Err(e) => (baseline(), Some(e)),
            },
            Controller::Shadow(a, b) => {
                // Shadow mode drives with the first controller and evaluates
                // the second, so the two can be compared step for step on the
                // same observations. Only the first one's controls reach the
                // car; a comparison that let both drive would measure whichever
                // ran last.
                let (ca, _) = a.act(obs, max_steer, baseline);
                let _ = b.act(obs, max_steer, baseline);
                (ca, None)
            }
        }
    }
}
