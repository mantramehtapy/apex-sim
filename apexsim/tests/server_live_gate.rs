//! R3 regression: the dashboard serve loop's "session is live" gate listed
//! Running and Formation but not FormationLap — a dashboard race froze at
//! tick 0 the moment the rolling formation lap started. The gate must treat
//! the rolling lap as live.

use std::time::Duration;

#[tokio::test]
async fn the_server_advances_a_rolling_lap_race() {
    let cfg = apexsim::sim::SimConfig {
        laps: 999,
        cars: 8,
        seed: 42,
        field: apexsim::race::FieldMode::Teams,
        realism: Some(apexsim::race::RealismConfig {
            formation_lap: true,
            ..apexsim::race::RealismConfig::default()
        }),
        ..apexsim::sim::SimConfig::default()
    };
    let (addr, _h) = apexsim::server::spawn_server(cfg, 0, true).expect("spawn server");
    assert!(
        apexsim::server::wait_until_ready(addr, Duration::from_secs(5)).await,
        "server did not report ready"
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut advanced = false;
    while std::time::Instant::now() < deadline {
        let body = http_get(addr, "/api/state").await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
        if v["tick"].as_u64().unwrap_or(0) > 100 {
            advanced = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        advanced,
        "the race froze at tick 0 during the rolling formation lap"
    );
}

async fn http_get(addr: std::net::SocketAddr, path: &str) -> String {
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
