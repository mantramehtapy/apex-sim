//! Phase 6 completion-gate tests, mapped to the plan's gate criteria:
//!
//! 1. "A fresh start reports service readiness."
//! 2. "A failed dependency produces a visible error and non-success status."
//! 3. "Shutdown stops only processes started by this run."
//!
//! Plus build items: one development launcher, a documented configuration
//! file, readiness checks instead of fixed startup sleeps, structured logs
//! with run ID / session ID / simulation tick / severity, graceful shutdown.
//!
//! These tests drive the real `apexsim dev` / `apexsim serve` binaries as
//! child processes (the plan's Phase 6 is about process behaviour, which
//! in-process tests cannot prove). The launcher only arms its stdin-EOF
//! shutdown for interactive terminals, so piped-stdin test runs are governed
//! by their `--duration`; the tests still hold stdin open out of caution.

use serde_json::Value;
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const EXE: &str = env!("CARGO_BIN_EXE_apexsim");

fn free_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0)).expect("bind :0").local_addr().unwrap().port()
}

fn health_once(port: u16) -> bool {
    use std::io::Write as _;
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_millis(500)));
    let req = format!(
        "GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    );
    if s.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = Vec::new();
    match s.read_to_end(&mut buf) {
        Ok(_) => String::from_utf8_lossy(&buf).contains("\"ok\":true"),
        Err(_) => false,
    }
}

fn wait_health(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if health_once(port) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(80));
    }
    false
}

/// The port must NOT come up during the window (shutdown really stopped it).
fn stays_down(port: u16, window: Duration) -> bool {
    let deadline = Instant::now() + window;
    while Instant::now() < deadline {
        if health_once(port) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(80));
    }
    true
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

/// Run a child to completion, holding its stdin OPEN for the whole run
/// (dropping it early would EOF the child's stdin watcher and shut it down).
fn run_child(args: &[String], max_wait: Duration) -> Run {
    let mut child = Command::new(EXE)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::piped())
        .spawn()
        .expect("spawn child");
    let _stdin = child.stdin.take(); // held until the run ends

    fn reader(s: impl std::io::Read + Send + 'static) -> Arc<Mutex<String>> {
        let buf = Arc::new(Mutex::new(String::new()));
        let buf2 = buf.clone();
        std::thread::spawn(move || {
            let mut s = s;
            let mut tmp = [0u8; 8192];
            let mut out = buf2.lock().unwrap();
            while let Ok(n) = s.read(&mut tmp) {
                if n == 0 {
                    break;
                }
                out.push_str(&String::from_utf8_lossy(&tmp[..n]));
            }
        });
        buf
    }
    let out_buf = reader(child.stdout.take().unwrap());
    let err_buf = reader(child.stderr.take().unwrap());

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                assert!(
                    started.elapsed() < max_wait,
                    "child overran {max_wait:?}: {args:?}"
                );
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => panic!("wait failed: {e}"),
        }
    };
    // Give the relay threads a beat to drain, then snapshot.
    std::thread::sleep(Duration::from_millis(150));
    let out_text = out_buf.lock().unwrap().clone();
    let err_text = err_buf.lock().unwrap().clone();
    Run {
        status,
        stdout: out_text,
        stderr: err_text,
    }
}

fn json_lines(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| {
            let t = l.trim();
            if t.starts_with('{') {
                serde_json::from_str(t).ok()
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Gate 1: a fresh start reports service readiness (and structured logs).
// ---------------------------------------------------------------------------

#[test]
fn gate1_fresh_start_reports_readiness_with_structured_logs() {
    let port = free_port();
    let run = run_child(
        &[
            "dev".into(),
            "--port".into(), port.to_string(),
            "--cars".into(), "4".into(),
            "--laps".into(), "2".into(),
            "--seed".into(), "7".into(),
            "--duration".into(), "5".into(),
        ],
        Duration::from_secs(40),
    );
    assert!(run.status.success(), "launcher must exit 0: {:?}", run.status);

    // The human readiness report, with the address and elapsed time.
    assert!(
        run.stdout.contains(&format!("dashboard ready: http://127.0.0.1:{port}/")),
        "no readiness report in stdout:\n{}",
        run.stdout
    );
    assert!(run.stdout.contains("ready in"), "no readiness latency:\n{}", run.stdout);

    // Structured lines: launcher and server share one run_id, the ready
    // event is a JSON line, and the run ends with an explicit shutdown.
    let lines = json_lines(&run.stdout);
    assert!(lines.len() >= 5, "expected structured log lines, got:\n{}", run.stdout);
    for l in &lines {
        assert!(l["level"].is_string() && l["run_id"].is_string(), "line missing level/run_id: {l}");
    }
    let run_id = lines[0]["run_id"].as_str().unwrap().to_string();
    assert!(lines.iter().any(|l| l["msg"].as_str().unwrap_or("").starts_with("dev launcher starting")));
    assert!(
        lines.iter().any(|l| {
            l["msg"].as_str().unwrap_or("").starts_with("ready:") && l["level"] == "info"
        }),
        "no structured ready line:\n{}",
        run.stdout
    );
    // The server child logs under the SAME run id (passed down explicitly).
    assert!(
        lines.iter().any(|l| {
            l["run_id"].as_str() == Some(&run_id)
                && l["msg"].as_str().unwrap_or("").contains("race loop started")
        }),
        "server lifecycle line missing or run_id not shared:\n{}",
        run.stdout
    );
    assert!(
        lines.iter().any(|l| l["msg"].as_str().unwrap_or("").starts_with("shutdown")),
        "no shutdown line:\n{}",
        run.stdout
    );
}

// ---------------------------------------------------------------------------
// Gate 2: a failed dependency → visible error + non-success status.
// ---------------------------------------------------------------------------

#[test]
fn gate2a_occupied_port_fails_visibly_and_nonzero() {
    // The "dependency" this run needs is a free port. Hold it from the test.
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let run = run_child(
        &[
            "dev".into(),
            "--port".into(), port.to_string(),
            "--cars".into(), "2".into(),
            "--duration".into(), "3".into(),
        ],
        Duration::from_secs(30),
    );
    assert!(
        !run.status.success(),
        "launcher must NOT succeed when its port is taken"
    );
    // Visible error, naming the actual problem, relayed from the child.
    assert!(
        run.stderr.contains("already in use") || run.stdout.contains("already in use"),
        "no 'already in use' error surfaced:\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stdout.contains("exited before readiness"),
        "launcher must report the early server death:\n{}",
        run.stdout
    );
    drop(listener);
}

#[test]
fn gate2b_bad_config_is_a_visible_error() {
    let dir = std::env::temp_dir().join(format!("f1_p6_bad_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.toml");
    std::fs::write(&path, "# undocumented key\nbogus_key = 1\n").unwrap();
    let port = free_port();
    let run = run_child(
        &[
            "dev".into(),
            "--config".into(), path.display().to_string(),
            "--port".into(), port.to_string(),
            "--duration".into(), "2".into(),
        ],
        Duration::from_secs(20),
    );
    assert!(!run.status.success(), "unknown config key must fail the run");
    assert!(
        run.stderr.contains("unknown key 'bogus_key'"),
        "error must name the bad key:\n{}",
        run.stderr
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Gate 3: shutdown stops only processes started by this run.
// ---------------------------------------------------------------------------

#[test]
fn gate3_shutdown_stops_only_its_own_children() {
    // An INDEPENDENT server this run did not start.
    let indep_port = free_port();
    let mut indep = Command::new(EXE)
        .args([
            "serve",
            "--port", &indep_port.to_string(),
            "--cars", "2",
            "--laps", "999",
            "--structured",
            "--duration", "60",
            "--log-level", "error",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn independent server");
    assert!(
        wait_health(indep_port, Duration::from_secs(15)),
        "independent server must come up"
    );

    // The launcher run: starts ITS OWN server, then shuts everything down.
    let dev_port = free_port();
    let run = run_child(
        &[
            "dev".into(),
            "--port".into(), dev_port.to_string(),
            "--cars".into(), "2".into(),
            "--laps".into(), "999".into(),
            "--duration".into(), "4".into(),
        ],
        Duration::from_secs(40),
    );
    assert!(run.status.success(), "launcher must exit 0: {:?}", run.status);

    // The launcher's own child is gone (its port stays down)…
    assert!(
        stays_down(dev_port, Duration::from_secs(2)),
        "launcher's port {dev_port} still answers after shutdown — orphaned child!"
    );
    // …while the process this run did NOT start is untouched.
    assert!(
        health_once(indep_port),
        "independent server on {indep_port} was killed by someone else's shutdown"
    );

    indep.kill().ok();
    let _ = indep.wait();
}

// ---------------------------------------------------------------------------
// Build item: one documented configuration file.
// ---------------------------------------------------------------------------

#[test]
fn gate4_init_writes_documented_config_and_launcher_honors_it() {
    let dir = std::env::temp_dir().join(format!("f1_p6_cfg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("f1_dev.toml");

    // --init writes the self-documenting template…
    let init = run_child(
        &["dev".into(), "--init".into(), "--config".into(), path.display().to_string()],
        Duration::from_secs(15),
    );
    assert!(init.status.success(), "--init failed: {}", init.stderr);
    let text = std::fs::read_to_string(&path).unwrap();
    for marker in ["# f1_dev.toml", "port = 8080", "cars = 20", "readiness_timeout_ms", "log_level", "# log_file"] {
        assert!(text.contains(marker), "template missing '{marker}':\n{text}");
    }
    // …and never overwrites silently.
    let again = run_child(
        &["dev".into(), "--init".into(), "--config".into(), path.display().to_string()],
        Duration::from_secs(15),
    );
    assert!(!again.status.success(), "--init must refuse to overwrite");

    // The launcher honors a config file (port/cars come from the file, not
    // from defaults).
    let port = free_port();
    std::fs::write(
        dir.join("custom.toml"),
        format!(
            "# test config\nport = {port}\ncars = 3\nlaps = 1\nseed = 9\nnoise = 0.0\nmode = race\nautostart = true\nreadiness_timeout_ms = 10000\nlog_level = info\n"
        ),
    )
    .unwrap();
    let run = run_child(
        &[
            "dev".into(),
            "--config".into(), dir.join("custom.toml").display().to_string(),
            "--duration".into(), "4".into(),
        ],
        Duration::from_secs(40),
    );
    assert!(run.status.success(), "config-driven launch failed: {}", run.stderr);
    assert!(
        run.stdout.contains(&format!("http://127.0.0.1:{port}/")),
        "launcher did not use the configured port:\n{}",
        run.stdout
    );
    let lines = json_lines(&run.stdout);
    assert!(
        lines
            .iter()
            .any(|l| l["msg"].as_str().unwrap_or("").contains("cars 3, laps 1")),
        "server did not run the configured field:\n{}",
        run.stdout
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Build item: structured logs with run id, session id, tick, severity.
// ---------------------------------------------------------------------------

#[test]
fn gate5_server_structured_logs_carry_run_sid_tick_severity() {
    let port = free_port();
    let mut child = Command::new(EXE)
        .args([
            "serve",
            "--port", &port.to_string(),
            "--cars", "3",
            "--laps", "2",
            "--structured",
            "--duration", "6",
            "--log-level", "info",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
        .expect("spawn structured server");
    let mut stdout = child.stdout.take().unwrap();
    // Wait for readiness, then let some lifecycle events flow.
    assert!(wait_health(port, Duration::from_secs(15)), "server never became ready");
    std::thread::sleep(Duration::from_millis(1500));
    child.kill().ok();
    let _ = child.wait();
    let mut text = String::new();
    stdout.read_to_string(&mut text).unwrap();

    let lines = json_lines(&text);
    assert!(!lines.is_empty(), "no structured lines from the server:\n{text}");
    for l in &lines {
        assert!(l["level"].is_string(), "no severity: {l}");
        assert!(l["run_id"].is_string(), "no run id: {l}");
    }
    // At least one session line carries BOTH the session id and the tick.
    assert!(
        lines.iter().any(|l| l["sid"].is_string() && l["tick"].is_u64()),
        "no line with sid + tick:\n{text}"
    );
    assert!(
        lines.iter().any(|l| l["msg"].as_str().unwrap_or("").contains("session started")),
        "no session-start line:\n{text}"
    );
}
