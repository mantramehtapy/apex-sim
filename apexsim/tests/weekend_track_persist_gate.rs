//! The reported bug, as a gate: an editor track staged via "save & race"
//! must SURVIVE the weekend's quali handoff. The handoff legs used to be
//! built from a stale boot-time config, silently reverting the track.

use std::time::Duration;

#[tokio::test]
async fn an_editor_track_survives_the_quali_handoff() {
    let cfg = apexsim::sim::SimConfig {
        laps: 3,
        cars: 2,
        seed: 42,
        weekend_quali: true,
        quali_window_s: 8.0,
        field: apexsim::race::FieldMode::Teams,
        realism: Some(apexsim::race::RealismConfig {
            formation_lap: false,
            ..apexsim::race::RealismConfig::default()
        }),
        ..apexsim::sim::SimConfig::default()
    };
    let (addr, _h) = apexsim::server::spawn_server(cfg, 0, true).expect("spawn server");
    assert!(
        apexsim::server::wait_until_ready(addr, Duration::from_secs(5)).await,
        "server not ready"
    );

    // Stage an OVAL (obviously different from the boot circuit) through the
    // editor flow: POST /api/config with the full spec, then a WS reset —
    // exactly what the dashboard editor's "Save & race" does.
    let oval = serde_json::json!({
        "track": {
            "version": 2, "straight_len": 0.0, "radius": 0.0, "half_width": 7.0,
            "custom": { "name": "gatetest", "points": [
                [0, 0], [300, -10], [600, 0], [600, 160], [300, 190], [0, 160]
            ]}
        }
    });
    let (status, body) =
        http_json_post(addr, "/api/config", &oval.to_string()).await;
    assert_eq!(status, 200, "{body}");
    ws_send_reset(addr).await;

    // After the reset AND across the quali handoffs, /api/track must still
    // report the staged oval — the handoff legs inherit the current config.
    let stadium = 1102.65_f64;
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let mut last_len = stadium;
    while std::time::Instant::now() < deadline {
        let body = http_json(addr, "/api/track").await;
        let v: serde_json::Value =
            serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
        last_len = v["length"].as_f64().unwrap_or(stadium);
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    assert!(
        (last_len - stadium).abs() > 50.0,
        "the staged track was reverted after the handoffs (last length {last_len})"
    );
}

async fn http_json_post(addr: std::net::SocketAddr, path: &str, body: &str) -> (u16, String) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf).to_string();
    let status: u16 = text
        .split(' ')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("



")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
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

/// Send the editor's reset command over the WS (what "Save & race" does).
async fn ws_send_reset(addr: std::net::SocketAddr) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;
    let (ws, _resp) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .expect("ws connect");
    let (mut write, mut read) = ws.split();

    // Welcome first: the session id.
    let mut sid = String::new();
    for _ in 0..6 {
        if let Ok(Some(Ok(Message::Text(t)))) =
            tokio::time::timeout(Duration::from_secs(3), read.next()).await
        {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                if v["type"] == "welcome" {
                    sid = v["sid"].as_str().unwrap_or("").to_string();
                    break;
                }
            }
        }
    }
    assert!(!sid.is_empty(), "no welcome received");
    write
        .send(Message::Text(
            serde_json::json!({ "v": 1, "sid": sid, "nonce": 1u64, "cmd": "reset" })
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
    // Hold the socket briefly so the command lands before we drop it.
    tokio::time::sleep(Duration::from_millis(300)).await;
}
