//! apexsim — Phase 1 Rust simulation core.
//!
//! One car, one track, dry conditions, a local line-following driver.
//! Fixed 4 ms timestep (250 Hz provisional), one authoritative in-memory
//! race state, seeded reset, pause, state hashing, and input replay.
//!
//! Units are SI throughout: metres, seconds, kilograms, newtons, radians.

pub mod config;
pub mod driver;
pub mod gym;
pub mod logging;
pub mod onnx;
pub mod physics;
pub mod policy;
pub mod protocol;
pub mod race;
pub mod radio;
pub mod replay;
pub mod rng;
pub mod server;
pub mod sim;
pub mod sound;
pub mod stats;
pub mod steward;
pub mod strategy;
pub mod teams;
pub mod telemetry;
pub mod tournament;
pub mod track;
pub mod validation;
pub mod weather;
pub mod weekend;

pub use physics::{CarParams, CarState, Controls};
pub use sim::{Phase, RaceState, Sim, SimConfig, SimOutcome};
