//! Weekend-on-the-dashboard gates (W1): a served session in weekend mode
//! runs qualifying car-by-car, forms the grid from the quali order, and
//! hands off to the race leg — all visible on /api/state.

use std::time::Duration;

#[tokio::test]
async fn a_weekend_server_runs_quali_then_hands_off_to_the_race() {
    let cfg = apexsim::sim::SimConfig {
        laps: 3,
        cars: 3,
        seed: 42,
        weekend_quali: true,
        quali_window_s: 20.0,
        field: apexsim::race::FieldMode::Teams,
        realism: Some(apexsim::race::RealismConfig {
            formation_lap: false, // keep the race leg short for the gate
            ..apexsim::race::RealismConfig::default()
        }),
        ..apexsim::sim::SimConfig::default()
    };
    let (addr, _h) = apexsim::server::spawn_server(cfg, 0, true).expect("spawn server");
    assert!(
        apexsim::server::wait_until_ready(addr, Duration::from_secs(5)).await,
        "server not ready"
    );

    // Watch /api/state until the weekend completes: quali runs advance
    // (brief ticks through cars), then the session becomes a race.
    let deadline = std::time::Instant::now() + Duration::from_secs(180);
    let mut saw_quali_run = [false; 3];
    let mut saw_race_leg = false;
    let mut race_grid_ids: Vec<u32> = Vec::new();
    while std::time::Instant::now() < deadline {
        let body = http_json(addr, "/api/state").await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);

        if let Some(q) = v.get("quali").filter(|q| !q.is_null()) {
            let car = q["car"].as_u64().unwrap_or(u64::MAX) as usize;
            let run = q["run"].as_u64().unwrap_or(0) as usize;
            if (1..=3).contains(&run) && car < 3 {
                saw_quali_run[run - 1] = true;
            }
        } else if v["session"] == "race" {
            saw_race_leg = true;
            // Capture the grid as soon as the leaderboard is populated —
            // right after the handoff it may still be empty.
            if race_grid_ids.is_empty() {
                if let Some(rows) = v["leaderboard"].as_array() {
                    if rows.len() == 3 {
                        race_grid_ids = rows
                            .iter()
                            .filter_map(|r| r["id"].as_u64())
                            .map(|x| x as u32)
                            .collect();
                    }
                }
            }
        }

        // Done observing once the race leg produced laps.
        if saw_race_leg && race_grid_ids.len() == 3 {
            if let Some(rows) = v["leaderboard"].as_array() {
                if rows.iter().any(|r| r["laps_done"].as_u64().unwrap_or(0) >= 1) {
                    break;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    assert!(
        saw_quali_run.iter().all(|b| *b),
        "every car must get a solo quali run: {saw_quali_run:?}"
    );
    assert!(saw_race_leg, "the weekend never handed off to the race");
    assert_eq!(
        race_grid_ids.len(),
        3,
        "the race leg must field all cars: {race_grid_ids:?}"
    );
}

async fn http_json(addr: std::net::SocketAddr, path: &str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    s.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.unwrap();
    String::from_utf8_lossy(&buf)
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default()
}
