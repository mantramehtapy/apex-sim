//! Track-editor gates (Phase D): the dashboard's save endpoint validates and
//! writes track files, and a circuit staged through /api/config becomes the
//! raced circuit at the reset — with /api/track serving the new shape.

use apexsim::server::spawn_server;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::connect_async;

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
    let status: u16 = text.split(' ').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

async fn start_race_server() -> SocketAddr {
    let cfg = apexsim::sim::SimConfig {
        laps: 999,
        cars: 20,
        seed: 42,
        field: apexsim::race::FieldMode::Teams,
        ..apexsim::sim::SimConfig::default()
    };
    let (addr, _handle) = spawn_server(cfg, 0, true).expect("spawn server");
    assert!(
        apexsim::server::wait_until_ready(addr, Duration::from_secs(5)).await,
        "server did not report ready within 5 s"
    );
    addr
}

const GOOD_TRACK: &str = r#"{"name":"edgate","half_width":7.0,
  "points":[[0,0],[150,-5],[280,40],[300,140],[200,210],[60,190],[-30,100]]}"#;

#[tokio::test]
async fn the_editor_endpoint_saves_any_drawable_track_and_refuses_unstructured_ones() {
    let addr = start_race_server().await;

    // A valid circuit saves.
    let (status, body) = http(addr, "POST", "/api/tracks", Some(GOOD_TRACK)).await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["ok"], true, "{body}");
    let file = v["file"].as_str().expect("file path").to_string();
    let written = std::fs::read_to_string(&file).expect("saved file readable");
    assert!(written.contains("edgate"), "the saved file is the track JSON");
    let mut droppings = vec![file.clone()];

    // Draw-anything: a self-crossing figure-8 and a 3-point micro loop SAVE
    // now (the engine races them); only unstructureable drawings are refused.
    let crossing = r#"{"name":"crossing","points":[[0,0],[300,-5],[600,-5],[650,50],[600,105],[300,3],[0,3],[-60,50]]}"#;
    let (status, body) = http(addr, "POST", "/api/tracks", Some(crossing)).await;
    assert_eq!(status, 200, "a figure-8 is accepted: {body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["ok"], true, "{body}");
    droppings.push(v["file"].as_str().expect("crossing file path").to_string());

    let micro = r#"{"name":"micro","half_width":7.0,"points":[[0,0],[120,-30],[60,90]]}"#;
    let (status, body) = http(addr, "POST", "/api/tracks", Some(micro)).await;
    assert_eq!(status, 200, "a 3-point micro loop is accepted: {body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["ok"], true, "{body}");
    droppings.push(v["file"].as_str().expect("micro file path").to_string());

    // A 2-point non-loop is refused with the reason (the one structural law).
    let notaloop = r#"{"name":"notaloop","points":[[0,0],[300,-5]]}"#;
    let (status, body) = http(addr, "POST", "/api/tracks", Some(notaloop)).await;
    assert_eq!(status, 400, "a non-loop must be refused: {body}");
    assert!(body.contains("3 control points"), "{body}");

    // Garbage is refused as JSON, not half-saved.
    let (status, body) = http(addr, "POST", "/api/tracks", Some("{oops")).await;
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("not valid JSON"), "{body}");
    for f in droppings {
        let _ = std::fs::remove_file(&f);
    }
}

#[tokio::test]
async fn a_staged_track_becomes_the_raced_circuit_at_reset() {
    let addr = start_race_server().await;

    // The server starts on the stadium.
    let (status, body) = http(addr, "GET", "/api/track", None).await;
    assert_eq!(status, 200);
    let before: Value = serde_json::from_str(&body).unwrap();
    assert!((before["length"].as_f64().unwrap() - 1102.65).abs() < 1.0, "{before}");

    // Stage the trioval circuit through the editor's config path.
    let spec = resolve_trioval_spec();
    let staged = serde_json::json!({
        "track": {
            "version": 2, "straight_len": 0.0, "radius": 0.0,
            "half_width": spec["half_width"],
            "custom": { "name": spec["name"], "points": spec["points"] },
        }
    });
    let (status, body) = http(addr, "POST", "/api/config", Some(&staged.to_string())).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["ok"], true, "{body}");

    // And a merely UGLY drawing (self-crossing figure-8) stages fine now —
    // the draw-anything law: only unstructureable drawings are refused, and
    // those fail at /api/tracks (asserted above), so staging accepts it too.
    let ugly = serde_json::json!({
        "track": { "version": 2, "straight_len": 0.0, "radius": 0.0, "half_width": 7.0,
                   "custom": { "name": "ugly", "points": [[0,0],[300,-5],[600,-5],[650,50],[600,105],[300,3],[0,3],[-60,50]] } }
    });
    let (status, body) = http(addr, "POST", "/api/config", Some(&ugly.to_string())).await;
    assert_eq!(status, 200, "a drawable track stages: {body}");
    // Re-stage the trioval so the reset below races the preset, not the 8.
    let (status, body) = http(addr, "POST", "/api/config", Some(&staged.to_string())).await;
    assert_eq!(status, 200, "{body}");

    // Reset over the WS (what the dashboard's Reset button does).
    let (ws, _resp) = connect_async(format!("ws://{addr}/ws")).await.expect("ws");
    let (mut write, mut read) = ws.split();
    use tokio_tungstenite::tungstenite::Message;
    // Drain until the welcome arrives so `sid` is known.
    let mut sid = String::new();
    for _ in 0..8 {
        let msg = timeout(Duration::from_secs(5), read.next())
            .await.expect("timeout").expect("closed").expect("err");
        if let Message::Text(t) = msg {
            let v: Value = serde_json::from_str(&t).unwrap();
            if v["type"] == "welcome" {
                sid = v["sid"].as_str().unwrap().to_string();
                // The welcome describes the CURRENT session (the stadium):
                // a staged circuit lands at the reset, asserted below.
                assert!((v["track"]["length"].as_f64().unwrap() - 1102.65).abs() < 1.0);
                break;
            }
        }
    }
    assert!(!sid.is_empty(), "no welcome received");
    write
        .send(Message::Text(
            serde_json::json!({ "v": 1, "sid": sid, "nonce": 1u64, "cmd": "reset" }).to_string().into(),
        ))
        .await
        .unwrap();
    // Ack, then the served track is the trioval.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        assert!(std::time::Instant::now() < deadline, "no reset ack");
        let msg = timeout(Duration::from_secs(5), read.next()).await.expect("timeout").expect("closed").expect("err");
        if let Message::Text(t) = msg {
            let v: Value = serde_json::from_str(&t).unwrap();
            if v["type"] == "ack" && v["nonce"].as_u64() == Some(1)
                && v["status"] == "queued" {
                break;
            }
        }
    }
    // The ack says "queued"; the command lands on a later sim tick. Poll
    // until the served track is the staged one.
    let mut after: Option<Value> = None;
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {

        let (status, body) = http(addr, "GET", "/api/track", None).await;
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        if (v["length"].as_f64().unwrap() - 1503.88).abs() < 2.0 {
            after = Some(v);
            break;
        }
        timeout(Duration::from_millis(100), futures_util::future::ready(())).await.ok();
    }
    let after = after.expect("the staged track never reached /api/track within 5 s");
    // The pit polyline follows the new circuit's fractions.
    assert!(after["pit"].is_object(), "the switched track must still draw its pit lane");
}

fn resolve_trioval_spec() -> Value {
    // The embedded preset, parsed: the same JSON the CLI resolves by name.
    let presets: &[(&str, &str)] = apexsim::track::TRACK_PRESETS;
    let (_, text) = presets.iter().find(|(n, _)| *n == "trioval").expect("trioval preset");
    serde_json::from_str(text).expect("preset parses")
}