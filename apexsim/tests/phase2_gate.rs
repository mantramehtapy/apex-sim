//! Phase 2 completion-gate tests, mapped to the plan's gate criteria:
//!
//! 1. "The dashboard reconnects and catches up to current state."
//! 2. "Closing, hiding, or slowing the browser does not change race
//!    progression."  (client disconnect must not stall the sim)
//! 3. "Snapshot age and dropped/replaced updates are observable."
//!    (metrics counters + per-snapshot seq; UI displays them)
//!
//! Plus the build items: versioned messages, ~25 Hz snapshots, HTTP config,
//! WS commands with stale-session rejection, bounded queues.

use apexsim::sim::SimConfig;
use apexsim::server::spawn_server;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message};

fn test_config() -> SimConfig {
    SimConfig {
        laps: 999, // continuous session; tests stop by dropping the server
        seed: 42,
        ..SimConfig::default()
    }
}

async fn start_server() -> SocketAddr {
    let (addr, _handle) = spawn_server(test_config(), 0, true).expect("spawn server");
    // Phase 6: readiness polling, never a fixed startup sleep.
    assert!(
        apexsim::server::wait_until_ready(addr, Duration::from_secs(5)).await,
        "server did not report ready within 5 s"
    );
    addr
}

/// Minimal HTTP/1.1 client (tests only).
async fn http(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
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
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

async fn ws_connect(addr: SocketAddr) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>> {
    let (ws, _resp) = connect_async(format!("ws://{addr}/ws")).await.expect("ws connect");
    ws
}

async fn next_json(
    ws: &mut tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>,
) -> Value {
    loop {
        let msg = timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("timeout waiting for ws message")
            .expect("ws closed")
            .expect("ws error");
        if let Message::Text(t) = msg {
            if let Ok(v) = serde_json::from_str(&t) {
                return v;
            }
        }
    }
}

async fn next_snapshot(
    ws: &mut tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>,
) -> Value {
    loop {
        let v = next_json(ws).await;
        if v["type"] == "snapshot" {
            return v;
        }
    }
}

/// Next ack with the given nonce (snapshots/events may interleave first).
async fn next_ack(
    ws: &mut tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<TcpStream>>,
    nonce: u64,
) -> Value {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        let v = next_json(ws).await;
        if v["type"] == "ack" && v["nonce"].as_u64() == Some(nonce) {
            return v;
        }
    }
    panic!("no ack with nonce {nonce} within 5 s");
}

#[tokio::test]
async fn gate_http_dashboard_and_endpoints_served() {
    let addr = start_server().await;
    let (code, html) = http(addr, "GET", "/", None).await;
    assert_eq!(code, 200);
    assert!(html.contains("apexsim dashboard"), "dashboard HTML not served");
    assert!(html.contains("<canvas"), "no canvas in dashboard");

    let (code, body) = http(addr, "GET", "/api/health", None).await;
    assert_eq!(code, 200);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["ok"], true);

    let (code, body) = http(addr, "GET", "/api/track", None).await;
    assert_eq!(code, 200);
    let track = serde_json::from_str::<Value>(&body).unwrap();
    assert_eq!(track["points"].as_array().unwrap().len(), 256);
    assert!((track["length"].as_f64().unwrap() - 1102.65).abs() < 1.0);

    let (code, body) = http(addr, "GET", "/api/state", None).await;
    assert_eq!(code, 200);
    let mut snap = serde_json::from_str::<Value>(&body).unwrap();
    assert_eq!(snap["v"].as_u64().unwrap(), 1);
    assert!(snap["sid"].as_str().unwrap().starts_with("s42-"));
    // Phase 6: readiness answers as soon as the HTTP layer is up — poll
    // (no fixed sleep) until the autostarted sim has actually ticked.
    let deadline = Instant::now() + Duration::from_secs(5);
    while snap["tick"].as_u64().unwrap_or(0) == 0 {
        assert!(Instant::now() < deadline, "sim never ticked (autostart)");
        tokio::time::sleep(Duration::from_millis(50)).await;
        let (code, body) = http(addr, "GET", "/api/state", None).await;
        assert_eq!(code, 200);
        snap = serde_json::from_str::<Value>(&body).unwrap();
    }
    assert!(snap["tick"].as_u64().unwrap() > 0, "sim should be running (autostart)");
}

#[tokio::test]
async fn gate_snapshots_versioned_and_flow_at_25hz() {
    let addr = start_server().await;
    let mut ws = ws_connect(addr).await;

    let welcome = next_json(&mut ws).await;
    assert_eq!(welcome["type"], "welcome");
    assert_eq!(welcome["v"].as_u64().unwrap(), 1);
    let sid = welcome["sid"].as_str().unwrap().to_string();
    assert!(welcome["track"]["points"].as_array().is_some());

    // Collect ~0.6 s of snapshots; check versioning, seq continuity, tick growth.
    let mut snaps = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_millis(700);
    while std::time::Instant::now() < deadline && snaps.len() < 25 {
        let v = next_json(&mut ws).await;
        if v["type"] == "snapshot" {
            snaps.push(v);
        }
    }
    assert!(snaps.len() >= 8, "only {} snapshots in 0.7 s — expected ~17 at 25 Hz", snaps.len());
    for w in snaps.windows(2) {
        let a = &w[0];
        let b = &w[1];
        assert_eq!(a["sid"].as_str().unwrap(), sid, "sid must be constant within a session");
        assert_eq!(b["seq"].as_u64().unwrap(), a["seq"].as_u64().unwrap() + 1,
            "seq must advance by exactly 1 (no drops while healthy)");
        assert!(b["tick"].as_u64().unwrap() > a["tick"].as_u64().unwrap());
        assert_eq!(b["v"].as_u64().unwrap(), 1);
        assert_eq!(b["phase"].as_str().unwrap(), "running");
    }
    // ~25 Hz rate check: with 17+ snapshots in 0.7 s we are in the right band.
    let rate = snaps.len() as f64 / 0.7;
    assert!((15.0..=40.0).contains(&rate), "snapshot rate {rate:.1} Hz out of band");
}

#[tokio::test]
async fn gate_ws_commands_pause_resume_apply() {
    let addr = start_server().await;
    let mut ws = ws_connect(addr).await;
    let welcome = next_json(&mut ws).await;
    let sid = welcome["sid"].as_str().unwrap().to_string();

    // Pause
    ws.send(Message::text(format!(r#"{{"v":1,"sid":"{sid}","nonce":101,"cmd":"pause"}}"#)))
        .await
        .unwrap();
    let ack = next_ack(&mut ws, 101).await;
    assert_eq!(ack["type"], "ack");
    assert_eq!(ack["status"], "queued");

    let mut paused_tick = None;
    let mut saw_phases: Vec<String> = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while std::time::Instant::now() < deadline {
        let s = next_snapshot(&mut ws).await;
        saw_phases.push(s["phase"].as_str().unwrap_or("?").to_string());
        if s["phase"] == "paused" {
            let applied = s["last_applied"].as_object().expect("last_applied on command");
            assert_eq!(applied["cmd"], "pause");
            assert_eq!(applied["nonce"].as_u64().unwrap(), 101);
            assert!(applied["applied_tick"].as_u64().is_some());
            paused_tick = Some(s["tick"].as_u64().unwrap());
            break;
        }
    }
    let tick_at_pause = paused_tick.unwrap_or_else(|| {
        panic!("sim never reported paused; phases seen: {saw_phases:?}")
    });

    // While paused the published tick must stay frozen…
    let s2 = next_snapshot(&mut ws).await;
    assert_eq!(s2["tick"].as_u64().unwrap(), tick_at_pause, "paused sim must not advance");

    // …and resume restarts progression.
    ws.send(Message::text(format!(r#"{{"v":1,"sid":"{sid}","nonce":102,"cmd":"resume"}}"#)))
        .await
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let mut resumed = false;
    while std::time::Instant::now() < deadline {
        let s = next_snapshot(&mut ws).await;
        if s["phase"] == "running" && s["tick"].as_u64().unwrap() > tick_at_pause {
            resumed = true;
            break;
        }
    }
    assert!(resumed, "sim did not resume after resume command");
}

#[tokio::test]
async fn gate_client_disconnect_does_not_stop_race() {
    let addr = start_server().await;

    // Observe progression with a client attached…
    let mut ws = ws_connect(addr).await;
    let welcome = next_json(&mut ws).await;
    let _ = welcome;
    let s = next_snapshot(&mut ws).await;
    let tick_with_client = s["tick"].as_u64().unwrap();

    // …then kill the client entirely.
    drop(ws);
    tokio::time::sleep(Duration::from_millis(900)).await;

    let (code, body) = http(addr, "GET", "/api/state", None).await;
    assert_eq!(code, 200);
    let after = serde_json::from_str::<Value>(&body).unwrap();
    let tick_after = after["tick"].as_u64().unwrap();
    assert!(
        tick_after > tick_with_client + 100, // 0.9 s at 250 Hz ≈ 225 ticks
        "race stalled after client disconnect: {tick_with_client} -> {tick_after}"
    );
    assert_eq!(after["phase"], "running");

    let (code, body) = http(addr, "GET", "/api/metrics", None).await;
    assert_eq!(code, 200);
    let m = serde_json::from_str::<Value>(&body).unwrap();
    assert_eq!(m["ws_clients"].as_u64().unwrap(), 0, "client count must return to zero");
    assert!(m["snapshots_published"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn gate_reconnect_catches_up_to_current_state() {
    let addr = start_server().await;

    let mut ws = ws_connect(addr).await;
    let welcome = next_json(&mut ws).await;
    let s1 = next_snapshot(&mut ws).await;
    let tick_before = s1["tick"].as_u64().unwrap();
    let sid = welcome["sid"].as_str().unwrap().to_string();
    drop(ws);

    tokio::time::sleep(Duration::from_millis(700)).await;

    // Reconnect: the welcome snapshot must already reflect current state.
    let mut ws2 = ws_connect(addr).await;
    let welcome2 = next_json(&mut ws2).await;
    assert_eq!(welcome2["type"], "welcome");
    assert_eq!(welcome2["sid"].as_str().unwrap(), sid, "same session across reconnect");
    let catch_up_tick = welcome2["snapshot"]["tick"].as_u64().unwrap();
    assert!(
        catch_up_tick > tick_before,
        "welcome snapshot {catch_up_tick} did not catch up past {tick_before}"
    );
    // And the stream continues from there without gaps.
    let s2 = next_snapshot(&mut ws2).await;
    assert!(s2["tick"].as_u64().unwrap() >= catch_up_tick);
}

#[tokio::test]
async fn gate_stale_session_command_rejected() {
    let addr = start_server().await;
    let mut ws = ws_connect(addr).await;
    let _welcome = next_json(&mut ws).await;

    ws.send(Message::text(r#"{"v":1,"sid":"s99-old","nonce":1,"cmd":"pause"}"#.to_string()))
        .await
        .unwrap();
    let ack = next_ack(&mut ws, 1).await;
    assert_eq!(ack["type"], "ack");
    assert_eq!(ack["status"], "rejected_stale_session");

    // Unknown command name also rejected.
    let sid_now = {
        let (code, body) = http(addr, "GET", "/api/state", None).await;
        assert_eq!(code, 200);
        serde_json::from_str::<Value>(&body).unwrap()["sid"].as_str().unwrap().to_string()
    };
    ws.send(Message::text(format!(
        r#"{{"v":1,"sid":"{sid_now}","nonce":2,"cmd":"turbo"}}"#
    )))
    .await
    .unwrap();
    let ack2 = next_ack(&mut ws, 2).await;
    assert_eq!(ack2["status"], "rejected_unknown_command");
    // And the race is still running — rejections have no side effects.
    let (code, body) = http(addr, "GET", "/api/state", None).await;
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["phase"], "running");
    assert_eq!(code, 200);
}

#[tokio::test]
async fn gate_config_validation_over_http() {
    let addr = start_server().await;

    // Out-of-range values are rejected with a 400 and an error message.
    let (code, body) = http(addr, "POST", "/api/config", Some(r#"{"laps":0}"#)).await;
    assert_eq!(code, 400, "body: {body}");
    assert!(serde_json::from_str::<Value>(&body).unwrap()["error"].is_string());

    let (code, _) = http(addr, "POST", "/api/config", Some(r#"{"noise":0.5}"#)).await;
    assert_eq!(code, 400);

    let (code, _) = http(addr, "POST", "/api/config", Some(r#"{"seed":"not-a-number"}"#)).await;
    assert_eq!(code, 400);

    // Valid settings are staged as pending, not applied mid-race.
    let (code, body) = http(addr, "POST", "/api/config", Some(r#"{"laps":7,"seed":123}"#)).await;
    assert_eq!(code, 200, "body: {body}");
    let v = serde_json::from_str::<Value>(&body).unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(v["pending"]["laps"].as_u64().unwrap(), 7);

    let (code, body) = http(addr, "GET", "/api/config", None).await;
    assert_eq!(code, 200);
    let cfg = serde_json::from_str::<Value>(&body).unwrap();
    assert_eq!(cfg["current"]["laps"].as_u64().unwrap(), 999, "current must be untouched until reset");
    assert_eq!(cfg["pending"]["seed"].as_u64().unwrap(), 123);
}

#[tokio::test]
async fn gate_start_and_reset_with_config_staging() {
    // No autostart: the dashboard's Start button drives the race.
    let (addr, _h) = spawn_server(test_config(), 0, false).expect("spawn server");
    assert!(
        apexsim::server::wait_until_ready(addr, Duration::from_secs(5)).await,
        "server did not report ready within 5 s"
    );

    let mut ws = ws_connect(addr).await;
    let welcome = next_json(&mut ws).await;
    let sid0 = welcome["sid"].as_str().unwrap().to_string();

    // Before start, the sim sits ready and the tick does not advance.
    let s = next_snapshot(&mut ws).await;
    assert_eq!(s["phase"], "ready");
    assert_eq!(s["tick"].as_u64().unwrap(), 0);

    // Stage a different seed over HTTP, then Start.
    let (code, _) = http(addr, "POST", "/api/config", Some(r#"{"seed":777,"laps":3}"#)).await;
    assert_eq!(code, 200);
    ws.send(Message::text(format!(r#"{{"v":1,"sid":"{sid0}","nonce":1,"cmd":"start"}}"#)))
        .await
        .unwrap();
    assert_eq!(next_ack(&mut ws, 1).await["status"], "queued");
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let mut running = false;
    while std::time::Instant::now() < deadline {
        let s = next_snapshot(&mut ws).await;
        if s["phase"] == "running" && s["tick"].as_u64().unwrap() > 0 {
            running = true;
            break;
        }
    }
    assert!(running, "start command did not start the race");

    // Reset applies the staged config and mints a new session id.
    ws.send(Message::text(format!(r#"{{"v":1,"sid":"{sid0}","nonce":2,"cmd":"reset"}}"#)))
        .await
        .unwrap();
    assert_eq!(next_ack(&mut ws, 2).await["status"], "queued");
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let mut new_sid = None;
    while std::time::Instant::now() < deadline {
        let s = next_snapshot(&mut ws).await;
        let sid = s["sid"].as_str().unwrap_or("");
        if sid != sid0 && !sid.is_empty() {
            assert_eq!(s["phase"], "ready");
            assert_eq!(s["tick"].as_u64().unwrap(), 0, "reset must rewind the clock");
            new_sid = Some(sid.to_string());
            break;
        }
    }
    let sid1 = new_sid.expect("reset never minted a new session id");
    assert!(sid1.starts_with("s777-"), "staged seed 777 should apply on reset, got {sid1}");

    // A command carrying the OLD session id must now be rejected as stale.
    ws.send(Message::text(format!(r#"{{"v":1,"sid":"{sid0}","nonce":3,"cmd":"pause"}}"#)))
        .await
        .unwrap();
    assert_eq!(next_ack(&mut ws, 3).await["status"], "rejected_stale_session");
}

#[tokio::test]
async fn gate_snapshot_age_and_drops_observable_via_metrics() {
    let addr = start_server().await;
    let mut ws = ws_connect(addr).await;
    let _ = next_json(&mut ws).await; // welcome

    // Every snapshot carries server_now_ms (age observable per message)…
    let s = next_snapshot(&mut ws).await;
    assert!(s["server_now_ms"].as_u64().unwrap() > 1_000_000_000_000, "server_now_ms should be unix ms");

    // …and delivery/drop counters are exposed.
    for _ in 0..5 {
        let _ = next_snapshot(&mut ws).await;
    }
    let (code, body) = http(addr, "GET", "/api/metrics", None).await;
    assert_eq!(code, 200);
    let m = serde_json::from_str::<Value>(&body).unwrap();
    assert!(m["ws_snapshots_sent"].as_u64().unwrap() >= 5);
    assert!(m["snapshots_published"].as_u64().unwrap() >= m["ws_snapshots_sent"].as_u64().unwrap());
    assert!(m["uptime_ms"].as_u64().unwrap() > 0);
}
