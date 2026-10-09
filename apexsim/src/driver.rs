//! Local line-following driver: pure-pursuit steering plus curvature-aware
//! speed control with braking anticipation. Runs at 50 Hz (every 5th physics
//! tick per the plan's provisional targets) and is fully deterministic.
//!
//! Seeded noise (zero amplitude by default in Phase 1) is added to the
//! controls so the seed has observable meaning; replay reproduces it exactly
//! because the recorded inputs already contain it.

use crate::physics::{CarState, Controls};
use crate::rng::Lcg;
use crate::track::Track;

#[derive(Clone, Copy, Debug)]
pub struct DriverParams {
    pub wheelbase: f64,         // m, for pure-pursuit geometry
    pub lookahead_gain: f64,    // Ld = gain·speed + base
    pub lookahead_base: f64,    // m
    pub lookahead_min: f64,     // m
    pub lookahead_max: f64,     // m
    pub a_lat_max: f64,         // m/s², lateral budget for corner speed
    pub a_brake_anticipate: f64, // m/s², assumed decel when looking ahead
    pub v_straight_max: f64,    // m/s
    pub horizon_margin: f64,    // m added to braking-distance horizon
    pub speed_margin: f64,      // fraction, safety factor on corner speed
    pub throttle_gain: f64,
    pub brake_gain: f64,
    pub deadband: f64,          // m/s
    pub steer_rate_limit: f64,  // rad/s on the commanded angle
    pub max_steer: f64,         // rad
    pub noise_amplitude: f64,   // fraction of full control range
}

impl Default for DriverParams {
    fn default() -> Self {
        DriverParams {
            wheelbase: 3.4,
            lookahead_gain: 0.6,
            lookahead_base: 8.0,
            lookahead_min: 10.0,
            lookahead_max: 45.0,
            a_lat_max: 11.0,
            a_brake_anticipate: 11.0,
            v_straight_max: 75.0,
            horizon_margin: 30.0,
            speed_margin: 0.97,
            throttle_gain: 0.45,
            brake_gain: 0.30,
            deadband: 0.25,
            steer_rate_limit: 2.5,
            max_steer: 0.45,
            noise_amplitude: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct Driver {
    params: DriverParams,
    /// Construction-time params: the base the grip scale multiplies from,
    /// so `set_grip_scale` is absolute (idempotent) wherever it is called.
    base: DriverParams,
    rng: Lcg,
    prev_steer_cmd: f64,
}

pub struct DriverInput {
    /// Centreline distance of the car, m.
    pub s: f64,
    /// Target lateral offset from the centreline (m; + = left). The pursuit
    /// point is shifted by this, giving per-car racing lines / overtakes.
    pub lateral: f64,
    /// Traffic speed cap (m/s); INFINITY when no car is nearby. Applied on
    /// top of the internal curvature-based target.
    pub v_cap: f64,
}

impl Default for DriverInput {
    fn default() -> Self {
        DriverInput { s: 0.0, lateral: 0.0, v_cap: f64::INFINITY }
    }
}

impl Driver {
    pub fn new(params: DriverParams, seed: u64) -> Self {
        Driver { base: params, params, rng: Lcg::new(seed ^ 0xD12), prev_steer_cmd: 0.0 }
    }

    /// Scale the performance envelope (cornering, braking anticipation,
    /// straight-line speed) by a grip factor. Tire compound and wear act
    /// through this: the driver aims at what the tires can actually hold,
    /// so degradation costs lap time instead of spinning the car. Absolute
    /// from the constructor params — calling it every driver tick is fine.
    pub fn set_grip_scale(&mut self, scale: f64) {
        self.params.a_lat_max = self.base.a_lat_max * scale;
        self.params.a_brake_anticipate = self.base.a_brake_anticipate * scale;
        self.params.v_straight_max = self.base.v_straight_max * scale;
    }

    /// The speed this driver is aiming for at `pos`, in m/s.
    ///
    /// Split out of `drive` because it is the driver's whole longitudinal
    /// decision, and a learned policy needs to see it: the target is a minimum
    /// over ~65 speed-limit samples spread across a braking horizon that grows
    /// with speed, so it is not recoverable from a handful of curvature
    /// lookups. Everything a real driver computes off the track map, this
    /// method computes from the same map.
    pub fn v_target(&self, car: &CarState, track: &Track, pos: DriverInput) -> f64 {
        let dp = &self.params;
        let brake_dist = car.vel_long * car.vel_long / (2.0 * dp.a_brake_anticipate);
        let horizon = brake_dist + dp.horizon_margin;
        let mut v_target = dp.v_straight_max;
        let mut d = 5.0f64;
        while d < horizon {
            let v_allow = track
                .speed_limit_at(pos.s + d, dp.a_lat_max)
                .min(dp.v_straight_max)
                * dp.speed_margin;
            let candidate = (v_allow * v_allow + 2.0 * dp.a_brake_anticipate * d).sqrt();
            v_target = v_target.min(candidate);
            d += 5.0;
        }
        // Also respect the limit right where we stand, and any traffic cap.
        let v_here = track
            .speed_limit_at(pos.s, dp.a_lat_max)
            .min(dp.v_straight_max)
            * dp.speed_margin;
        v_target.min(v_here).min(pos.v_cap)
    }

    /// Compute controls for this 50 Hz update from the car state and its
    /// position on the track. Deterministic for a fixed seed.
    pub fn drive(&mut self, car: &CarState, track: &Track, pos: DriverInput) -> Controls {
        let dp = &self.params;

        // ---- Pure-pursuit steering toward a lookahead point ----
        let ld = (dp.lookahead_gain * car.vel_long + dp.lookahead_base)
            .clamp(dp.lookahead_min, dp.lookahead_max);
        let (tx, ty, th) = track.point_at(pos.s + ld);
        // Shift the pursuit point sideways for the car's chosen line.
        let left_x = -th.sin();
        let left_y = th.cos();
        let tx = tx + left_x * pos.lateral;
        let ty = ty + left_y * pos.lateral;
        let dx = tx - car.pos_x;
        let dy = ty - car.pos_y;
        let cos_h = car.heading.cos();
        let sin_h = car.heading.sin();
        let fx = dx * cos_h + dy * sin_h; // forward component in car frame
        let fy = -dx * sin_h + dy * cos_h; // left component
        let alpha = fy.atan2(fx.max(1e-6));
        let steer_target = (2.0 * dp.wheelbase * alpha.sin()).atan2(ld);

        // ---- Target speed: min over horizon of sqrt(v_allow² + 2·a·d) ----
        let v_target = self.v_target(car, track, pos);

        // ---- Longitudinal pedals ----
        let err = v_target - car.vel_long;
        let mut throttle = 0.0f64;
        let mut brake = 0.0f64;
        if err > dp.deadband {
            throttle = (err * dp.throttle_gain).clamp(0.0, 1.0);
        } else if err < -dp.deadband {
            brake = (-err * dp.brake_gain).clamp(0.0, 1.0);
        }

        // ---- Friction-circle pedal caps ----
        // Tires spending lateral grip on cornering have proportionally less
        // left for longitudinal force. Without this cap the driver floors it
        // at corner exit, the rear axle loses all lateral capacity, and the
        // car spins (power oversteer).
        let a_lat_now = (car.vel_long * car.yaw_rate).abs();
        let a_long_avail = (dp.a_lat_max * dp.a_lat_max - a_lat_now * a_lat_now).max(0.04).sqrt();
        let throttle_cap = (a_long_avail / 10.0).clamp(0.0, 1.0); // 10 m/s² ≈ full-throttle accel
        let brake_cap = (a_long_avail / 15.0).clamp(0.0, 1.0); // 15 m/s² ≈ full-brake decel
        throttle = throttle.min(throttle_cap);
        brake = brake.min(brake_cap);

        // ---- Rate-limited steering command, then seeded noise ----
        let dt_driver = 0.02; // 50 Hz
        let mut steer_cmd = steer_target.clamp(-dp.max_steer, dp.max_steer);
        steer_cmd = self.prev_steer_cmd
            + (steer_cmd - self.prev_steer_cmd).clamp(-dp.steer_rate_limit * dt_driver, dp.steer_rate_limit * dt_driver);
        self.prev_steer_cmd = steer_cmd;

        if dp.noise_amplitude > 0.0 {
            let amp = dp.noise_amplitude;
            // Draw all three first, in a fixed order, so the generator's
            // advance per tick does not depend on which pedal is in use.
            let n_throttle = (self.rng.next_f64() * 2.0 - 1.0) * amp;
            let n_brake = (self.rng.next_f64() * 2.0 - 1.0) * amp;
            let n_steer = (self.rng.next_f64() * 2.0 - 1.0) * amp * 0.05;
            steer_cmd = (steer_cmd + n_steer).clamp(-dp.max_steer, dp.max_steer);
            // Noise goes only on the pedal already in use. Adding it to the
            // idle one instead would let a positive speck on an unused brake
            // channel conjure a brake command, which the both-pedals rule below
            // then resolves by zeroing the throttle — silently cancelling full
            // throttle on nearly half of all steps, for no physical reason.
            if brake > 0.0 {
                brake = (brake + n_brake).clamp(0.0, 1.0);
            } else {
                throttle = (throttle + n_throttle).clamp(0.0, 1.0);
            }
        }

        if brake > 0.0 {
            throttle = 0.0; // never both pedals
        }
        Controls { throttle, brake, steer_cmd }
    }
}
