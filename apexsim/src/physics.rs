//! Planar vehicle dynamics: dynamic bicycle model with saturating linear
//! tires, a friction circle per axle, quasi-static longitudinal load
//! transfer, and a low-speed blend to kinematic steering.
//!
//! Integrated with semi-implicit Euler at a fixed dt (4 ms). The step
//! function is pure: same (state, controls, params) in, same state out —
//! this is what makes the replay gate bit-exact.

/// Physical parameters of the single Phase 1 car. SI units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarParams {
    pub mass: f64,        // kg
    pub inertia_z: f64,   // kg·m² (yaw)
    pub wheelbase: f64,   // m
    pub cg_to_front: f64, // m (a)
    pub cg_to_rear: f64,  // m (b)
    pub cg_height: f64,   // m
    pub tire_grip: f64,   // friction coefficient µ
    pub corner_stiffness_front: f64, // N/rad
    pub corner_stiffness_rear: f64,  // N/rad
    pub engine_force_max: f64, // N, at low speed
    pub engine_power_max: f64, // W
    pub brake_force_max: f64,  // N, total both axles
    pub brake_bias_front: f64, // fraction of brake force on the front axle
    pub drag_area: f64,        // Cd·A, m²
    pub roll_resistance: f64,  // coefficient c_rr (force = c_rr · m · g)
    pub max_steer: f64,        // rad
    pub air_density: f64,      // kg/m³
}

impl Default for CarParams {
    fn default() -> Self {
        CarParams {
            mass: 720.0,
            inertia_z: 1000.0,
            wheelbase: 3.4,
            cg_to_front: 1.7,
            cg_to_rear: 1.7,
            cg_height: 0.35,
            tire_grip: 1.6,
            corner_stiffness_front: 90_000.0,
            corner_stiffness_rear: 100_000.0,
            engine_force_max: 8_000.0,
            engine_power_max: 400_000.0,
            brake_force_max: 18_000.0,
            brake_bias_front: 0.62,
            drag_area: 1.3,
            roll_resistance: 0.012,
            max_steer: 0.45,
            air_density: 1.225,
        }
    }
}

/// Vehicle state in the plane. World frame: x east, y north, heading ccw.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarState {
    pub pos_x: f64,      // m
    pub pos_y: f64,      // m
    pub heading: f64,    // rad
    pub vel_long: f64,   // m/s, forward in body frame
    pub vel_lat: f64,    // m/s, positive = left in body frame
    pub yaw_rate: f64,   // rad/s
    pub steer: f64,      // rad, actual front-wheel angle
}

/// Driver inputs, normalized to [0, 1] except steering.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Controls {
    pub throttle: f64, // [0, 1]
    pub brake: f64,    // [0, 1]
    pub steer_cmd: f64 // rad, desired front-wheel angle
}

/// One fixed physics step. Semi-implicit Euler; dt must stay fixed.
pub fn step(state: &CarState, ctrl: &Controls, p: &CarParams, dt: f64) -> CarState {
    let m = p.mass;
    let g = 9.81;
    let u = state.vel_long;
    let v = state.vel_lat;
    let r = state.yaw_rate;
    let a = p.cg_to_front;
    let b = p.cg_to_rear;
    let l = p.wheelbase;

    // Steering actuator: first-order rate limit toward the command.
    let max_rate = 4.0; // rad/s — fast but not instantaneous
    let delta = ctrl.steer_cmd.clamp(-p.max_steer, p.max_steer);
    let steer = (state.steer + (delta - state.steer).clamp(-max_rate * dt, max_rate * dt))
        .clamp(-p.max_steer, p.max_steer);

    // ---- Longitudinal tire forces (rear-wheel drive) ----
    let drive = ctrl.throttle * (p.engine_force_max.min(p.engine_power_max / u.max(3.0)));
    let brake_total = ctrl.brake * p.brake_force_max;
    let fx_front = -p.brake_bias_front * brake_total;
    let fx_rear = drive - (1.0 - p.brake_bias_front) * brake_total;

    // Body-level drag and rolling resistance (do not consume tire grip here).
    let fx_body = -0.5 * p.air_density * p.drag_area * u * u - p.roll_resistance * m * g * u.signum();

    // ---- Axle loads with quasi-static longitudinal transfer ----
    let ax_prev = (fx_front + fx_rear + fx_body) / m;
    let transfer = (m * ax_prev * p.cg_height / l).clamp(-0.4 * m * g, 0.4 * m * g);
    let fz_front = (m * g * b / l - transfer).max(100.0);
    let fz_rear = (m * g * a / l + transfer).max(100.0);

    // ---- Friction circle per axle ----
    // Longitudinal force is capped at 90% of axle grip, leaving a lateral
    // floor of sqrt(1-0.9²) ≈ 44% under combined loading. Fully saturating
    // the circle (lat → 0 under peak braking/power) makes the car
    // uncontrollable in transients; real slip-limited tires behave the same
    // way under ABS-style intervention.
    let cap_f = p.tire_grip * fz_front;
    let cap_r = p.tire_grip * fz_rear;
    let fx_front_c = fx_front.clamp(-0.9 * cap_f, 0.9 * cap_f);
    let fx_rear_c = fx_rear.clamp(-0.9 * cap_r, 0.9 * cap_r);
    let lat_cap_f = (cap_f * cap_f - fx_front_c * fx_front_c).max(0.0).sqrt();
    let lat_cap_r = (cap_r * cap_r - fx_rear_c * fx_rear_c).max(0.0).sqrt();

    // ---- Lateral tire forces (saturating linear), low-speed blend ----
    // u_dyn floors the speed in the slip-angle denominators so standstill
    // cannot blow them up; the kinematic blend below takes over down there.
    let w = (u / 3.0).clamp(0.0, 1.0); // 0 = kinematic regime, 1 = dynamic
    let u_dyn = if u.abs() < 1.5 { 1.5 } else { u };

    let alpha_f = (v + a * r).atan2(u_dyn) - steer;
    let alpha_r = (v - b * r).atan2(u_dyn);
    let fy_front_dyn = (-p.corner_stiffness_front * alpha_f).clamp(-lat_cap_f, lat_cap_f);
    let fy_rear_dyn = (-p.corner_stiffness_rear * alpha_r).clamp(-lat_cap_r, lat_cap_r);

    let fx_total = fx_front_c + fx_rear_c + fx_body;
    let du_dt = (fx_total - fy_front_dyn * steer.sin()) / m + v * r;
    let dv_dt_dyn = (fy_front_dyn * steer.cos() + fy_rear_dyn) / m - u * r;
    let dr_dt_dyn = (a * fy_front_dyn * steer.cos() - b * fy_rear_dyn) / p.inertia_z;

    // Kinematic targets blended in at low speed so nothing divides by ~0.
    let r_kin = u / l * steer.tan();
    let v_kin = r_kin * b;
    let relax = 0.15; // s, first-order blend toward kinematic response
    let dv_dt = w * dv_dt_dyn + (1.0 - w) * (v_kin - v) / relax;
    let dr_dt = w * dr_dt_dyn + (1.0 - w) * (r_kin - r) / relax;

    // ---- Integrate ----
    let mut new = *state;
    new.vel_lat = v + dv_dt * dt;
    new.yaw_rate = r + dr_dt * dt;
    new.vel_long = (u + du_dt * dt).max(0.0); // no reverse gear in Phase 1
    new.pos_x += (new.vel_long * state.heading.cos() - new.vel_lat * state.heading.sin()) * dt;
    new.pos_y += (new.vel_long * state.heading.sin() + new.vel_lat * state.heading.cos()) * dt;
    new.heading += new.yaw_rate * dt;
    new.steer = steer;
    new
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coast_state(u: f64) -> CarState {
        CarState { vel_long: u, ..Default::default() }
    }

    #[test]
    fn full_throttle_accelerates() {
        let p = CarParams::default();
        let mut s = coast_state(0.0);
        let ctrl = Controls { throttle: 1.0, ..Default::default() };
        for _ in 0..2000 {
            s = step(&s, &ctrl, &p, 0.004); // 8 s
        }
        assert!(s.vel_long > 30.0, "only reached {} m/s", s.vel_long);
        assert!(s.pos_x > 100.0);
    }

    #[test]
    fn full_brake_from_50_stops_within_250m() {
        let p = CarParams::default();
        let mut s = coast_state(50.0);
        let ctrl = Controls { brake: 1.0, ..Default::default() };
        let mut dist = 0.0;
        for _ in 0..10_000 {
            let prev = s.vel_long;
            s = step(&s, &ctrl, &p, 0.004);
            dist += (prev + s.vel_long) / 2.0 * 0.004;
            if s.vel_long < 0.5 {
                break;
            }
        }
        assert!(s.vel_long < 0.5, "still rolling at {} m/s", s.vel_long);
        assert!(dist < 250.0, "braking distance {} m", dist);
    }

    #[test]
    fn steering_turns_left_at_speed() {
        let p = CarParams::default();
        let mut s = coast_state(30.0);
        let ctrl = Controls { steer_cmd: 0.1, ..Default::default() };
        for _ in 0..1250 {
            s = step(&s, &ctrl, &p, 0.004); // 5 s
        }
        assert!(s.heading > 0.05, "heading {}", s.heading);
        assert!(s.pos_y > 5.0, "moved left by {} m", s.pos_y);
    }

    #[test]
    fn no_nans_under_random_valid_inputs() {
        let p = CarParams::default();
        let mut rng = crate::rng::Lcg::new(1234);
        let mut s = CarState::default();
        for _ in 0..20_000 {
            let ctrl = Controls {
                throttle: rng.next_f64(),
                brake: rng.next_f64(),
                steer_cmd: (rng.next_f64() * 2.0 - 1.0) * p.max_steer,
            };
            s = step(&s, &ctrl, &p, 0.004);
            assert!(
                s.pos_x.is_finite() && s.pos_y.is_finite() && s.vel_long.is_finite(),
                "state diverged: {:?}",
                s
            );
        }
    }
}
