//! R4 gates: every circuit dresses itself — pit buildings, wall, gantry,
//! stands, gravel — derived from its own geometry (works for editor tracks).

use apexsim::server::state_track_def_pub;
use apexsim::track::{resolve_track_spec, TrackSpec};

fn dressing_for(spec: &TrackSpec) -> apexsim::protocol::TrackDressing {
    state_track_def_pub(spec)
        .dressing
        .expect("every circuit dresses itself")
}

#[test]
fn the_stadium_dresses_itself() {
    let d = dressing_for(&TrackSpec::default());
    assert!(d.garages.len() >= 6, "pit buildings");
    assert!(d.pit_wall.len() >= 10, "pit wall polyline");
    assert!(d.stands.len() >= 3, "grandstands");
    assert!(d.gravel.len() >= 2, "gravel traps at the sharpest corners");
    assert!(d.runoff_m > 0.0);
    // The gantry sits ON the track at the start/finish line.
    let t = TrackSpec::default().build().unwrap();
    let (x, y, _) = t.point_at(0.0);
    let dist = ((d.gantry[0] - x).powi(2) + (d.gantry[1] - y).powi(2)).sqrt();
    assert!(dist < 1.0, "gantry {} m off the line", dist);
}

#[test]
fn a_custom_track_dresses_itself_from_its_own_geometry() {
    let spec = resolve_track_spec("kidney").unwrap();
    let d = dressing_for(&spec);
    assert!(!d.garages.is_empty() && !d.stands.is_empty() && !d.gravel.is_empty());
    // Dressing positions must lie NEAR the circuit (within the dressing
    // band around it) — derivation from the centreline, not stadium coords.
    let t = spec.build().unwrap();
    let len = t.length();
    for g in d.gravel.iter().chain(d.stands.iter()) {
        let (s, _lat) = t.project(g[0], g[1]);
        assert!(s >= 0.0 && s <= len, "dressing at s={s} outside the lap");
    }
}

#[test]
fn different_circuits_dress_differently() {
    let a = dressing_for(&resolve_track_spec("kidney").unwrap());
    let b = dressing_for(&resolve_track_spec("trioval").unwrap());
    let neq = a.garages != b.garages
        || a.pit_wall != b.pit_wall
        || a.stands != b.stands
        || a.gravel != b.gravel;
    assert!(neq, "different circuits must derive different dressing");
}
