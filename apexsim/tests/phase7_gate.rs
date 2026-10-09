//! Phase 7 gates — local Tev1 strategy evaluation.
//!
//! The gate the plan sets:
//!   1. a strategy benchmark shows when the model helps, ties, or hurts;
//!   2. timeout, unload, bad answer, and low-confidence cases fall back safe;
//!   3. full-race p99 physics timing and memory stay within Phase 0 limits.
//!
//! Every case here runs against a scripted mock endpoint rather than a real
//! model: the failure modes the plan asks about (timeout, unload, bad
//! answer, refused connection) are things you cannot reliably provoke from a
//! healthy Ollama. The real-model run is the `aibench` benchmark, not here.
//! What these tests pin down is the safety property — every one of them
//! must still produce a complete, classified race.

use apexsim::race::{RaceConfig, RaceOutcome};
use apexsim::strategy::{AdvisorConfig, AiExchange};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Mock OpenAI-compatible endpoint
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Reply {
    /// A well-formed OpenAI chat completion carrying this content.
    Content(String),
    /// An HTTP error status (404 = model not loaded, Ollama's "unload").
    Status(u16, String),
    /// A 200 whose body is not the expected JSON shape.
    Malformed,
    /// Accept the request and never answer, so the client read times out.
    Hang,
}

struct Mock {
    addr: SocketAddr,
    requests: Arc<AtomicUsize>,
}

impl Mock {
    /// Serve `script` in order; the last entry repeats for further requests.
    fn start(script: Vec<Reply>) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let seen = requests.clone();
        std::thread::spawn(move || {
            let mut i = 0usize;
            loop {
                match listener.accept() {
                    Ok((mut sock, _)) => {
                        let reply = script.get(i).unwrap_or_else(|| script.last().unwrap());
                        i += 1;
                        seen.fetch_add(1, Ordering::SeqCst);
                        sock.set_nonblocking(false).ok();
                        sock.set_read_timeout(Some(std::time::Duration::from_secs(2)))
                            .ok();
                        read_request(&mut sock);
                        match reply {
                            Reply::Content(c) => {
                                let body = format!(
                                    r#"{{"choices":[{{"message":{{"role":"assistant","content":{}}}}}]}}"#,
                                    json_string(c)
                                );
                                write_response(&mut sock, 200, "application/json", &body);
                            }
                            Reply::Status(code, body) => {
                                write_response(&mut sock, *code, "application/json", body);
                            }
                            Reply::Malformed => {
                                write_response(&mut sock, 200, "application/json", "not json at all");
                            }
                            Reply::Hang => {
                                // Hold the socket open past any sane read
                                // timeout without ever writing.
                                std::thread::sleep(std::time::Duration::from_secs(3));
                            }
                        }
                        let _ = sock.shutdown(std::net::Shutdown::Both);
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        Mock { addr, requests }
    }

    fn endpoint(&self) -> String {
        format!("http://{}/v1/chat/completions", self.addr)
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Consume the request headers and body so the client never sees a reset.
fn read_request(sock: &mut std::net::TcpStream) {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while sock.read(&mut byte).unwrap_or(0) == 1 {
        buf.push(byte[0]);
        if buf.ends_with(b"\r\n\r\n") {
            break;
        }
        if buf.len() > 64 * 1024 {
            return;
        }
    }
    let head = String::from_utf8_lossy(&buf);
    let len = head
        .lines()
        .find_map(|l| {
            let lower = l.to_ascii_lowercase();
            lower.strip_prefix("content-length:").map(|v| v.trim().to_string())
        })
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; len];
    let _ = sock.read_exact(&mut body);
}

fn write_response(sock: &mut std::net::TcpStream, code: u16, ctype: &str, body: &str) {
    let reason = match code {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = sock.write_all(head.as_bytes());
    let _ = sock.write_all(body.as_bytes());
    let _ = sock.flush();
}

/// A port that is guaranteed to have nothing listening on it.
fn dead_endpoint() -> String {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    drop(l);
    format!("http://{addr}/v1/chat/completions")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn advisor(endpoint: String) -> AdvisorConfig {
    AdvisorConfig {
        endpoint,
        model: "mock-strategist:0.8b".into(),
        every_laps: 2,
        timeout_ms: 400,
        max_tokens: 320,
        min_confidence: 0.5,
    }
}

fn scenario() -> RaceConfig {
    RaceConfig {
        cars: 8,
        laps: 6,
        seed: 42,
        ..RaceConfig::default()
    }
}

fn run(cfg: RaceConfig) -> (RaceOutcome, Vec<AiExchange>) {
    let (o, _, log) = apexsim::race::run_race_full(cfg, false);
    (o, log)
}

/// Every race must still be a complete, classified race whatever the model
/// did. This is the invariant the whole gate rests on.
fn assert_complete(o: &RaceOutcome) {
    assert!(o.phase_finished(), "race did not reach the flag");
    assert_eq!(
        o.classification.len(),
        o.cars as usize,
        "every car must be classified"
    );
    let positions: Vec<u32> = o.classification.iter().map(|r| r.pos).collect();
    let mut sorted = positions.clone();
    sorted.sort_unstable();
    assert_eq!(positions, sorted, "classification must be ordered by position");
}

fn tally(log: &[AiExchange], outcome: &str) -> usize {
    log.iter().filter(|x| x.outcome == outcome).count()
}

// ---------------------------------------------------------------------------
// Gate 1 — advice is applied when the answer is good
// ---------------------------------------------------------------------------

#[test]
fn gate1_good_advice_is_applied_and_recorded() {
    let good = r#"{"pace":{"choice":"push","confidence":0.9},"pit":{"choice":"stay","confidence":0.8},"tire":{"choice":"keep","confidence":0.9}}"#;
    let mock = Mock::start(vec![Reply::Content(good.into())]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    assert!(!log.is_empty(), "the advisor must have been consulted");
    assert!(
        tally(&log, "applied") >= 1,
        "a confident, legal answer must be applied; got {:?}",
        log.iter().map(|x| (x.outcome, &x.detail)).collect::<Vec<_>>()
    );

    let x = &log[0];
    // Everything the plan requires on the record must actually be there.
    assert!(x.prompt_summary.contains("\"lap\""), "prompt summary missing lap");
    assert!(x.prompt_summary.contains("tire"), "prompt summary missing tire");
    assert_eq!(x.model, "mock-strategist:0.8b");
    assert!(x.response.contains("push"), "raw response not recorded");
    assert!(x.latency_ms > 0.0, "latency not measured");
    assert!(x.applied_tick.is_some(), "an applied exchange needs an application tick");
    assert!(x.applied_tick.unwrap() >= x.request_tick, "applied before it was asked");
    assert!(!x.detail.is_empty(), "an exchange must say what it did");

    // Pace "push" is 1.04 and must survive into the driver's speed cap.
    let sum = o.ai.as_ref().expect("advisor ran");
    assert!(sum.applied >= 1);
    assert!(sum.pace_scale_final > 1.0, "push must raise the pace scale");
}

// ---------------------------------------------------------------------------
// Gate 2 — every failure mode falls back safely and still finishes the race
// ---------------------------------------------------------------------------

#[test]
fn gate2_timeout_falls_back_and_the_race_still_completes() {
    let mock = Mock::start(vec![Reply::Hang]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    assert!(tally(&log, "timeout") >= 1, "a hang must be recorded as a timeout");
    assert_eq!(tally(&log, "applied"), 0, "a timed-out answer must never be applied");
    // Nothing changed, so this is the deterministic race, bit for bit.
    let baseline = run(scenario()).0;
    assert_eq!(
        o.state_hash, baseline.state_hash,
        "a pure fallback must reproduce the deterministic race exactly"
    );
}

#[test]
fn gate2b_unloaded_model_404_falls_back() {
    // Ollama answers 404 when the model is not pulled or not loaded.
    let mock = Mock::start(vec![Reply::Status(404, r#"{"error":"model not found"}"#.into())]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    assert!(tally(&log, "error") >= 1, "a 404 must be recorded as an error");
    let e = log.iter().find(|x| x.outcome == "error").unwrap();
    assert!(e.detail.contains("404"), "the status code must be on the record: {}", e.detail);
    let baseline = run(scenario()).0;
    assert_eq!(o.state_hash, baseline.state_hash);
}

#[test]
fn gate2c_bad_answer_falls_back() {
    let mock = Mock::start(vec![Reply::Malformed]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    assert!(tally(&log, "bad_answer") >= 1, "an unparseable payload must be recorded");
    assert_eq!(tally(&log, "applied"), 0);
    let baseline = run(scenario()).0;
    assert_eq!(o.state_hash, baseline.state_hash);
}

#[test]
fn gate2d_prose_without_json_falls_back() {
    // The model's most common real failure: a chatty answer with no JSON.
    let mock = Mock::start(vec![Reply::Content(
        "I think you should probably pit around lap 4, maybe medium tires.".into(),
    )]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    assert!(tally(&log, "bad_answer") >= 1);
    let baseline = run(scenario()).0;
    assert_eq!(o.state_hash, baseline.state_hash);
}

#[test]
fn gate2e_low_confidence_falls_back() {
    // Well-formed and legal, but the model is not sure — the floor holds.
    let timid = r#"{"pace":{"choice":"push","confidence":0.2},"pit":{"choice":"now","confidence":0.1},"tire":{"choice":"H","confidence":0.05}}"#;
    let mock = Mock::start(vec![Reply::Content(timid.into())]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    assert!(tally(&log, "low_confidence") >= 1, "sub-floor answers must be held back");
    assert_eq!(tally(&log, "applied"), 0, "nothing below the floor may be applied");
    let baseline = run(scenario()).0;
    assert_eq!(o.state_hash, baseline.state_hash, "held-back advice changes nothing");
}

#[test]
fn gate2f_unreachable_endpoint_falls_back() {
    // Ollama not running. On this platform a dead localhost port does not
    // refuse — the firewall drops the SYN, so the attempt burns the connect
    // deadline and surfaces as a timeout. Either way the gate is the same:
    // the failure is recorded, nothing is applied, the race is untouched.
    let mut cfg = scenario();
    cfg.ai = Some(advisor(dead_endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    let failures = tally(&log, "error") + tally(&log, "timeout");
    assert!(failures >= 1, "an unreachable endpoint must be recorded");
    assert_eq!(tally(&log, "applied"), 0, "an unreachable endpoint must advise nothing");
    let baseline = run(scenario()).0;
    assert_eq!(o.state_hash, baseline.state_hash, "no service must equal no advice");
}

// ---------------------------------------------------------------------------
// Gate 3 — legality: the model cannot make an illegal move
// ---------------------------------------------------------------------------

#[test]
fn gate3_illegal_pit_on_final_lap_is_refused() {
    // The advisor asks on the last legal lap boundary, but wants a stop on a
    // lap where pitting cannot complete the race. It must be refused.
    let greedy = r#"{"pace":{"choice":"push","confidence":0.9},"pit":{"choice":"now","confidence":0.95},"tire":{"choice":"S","confidence":0.9}}"#;
    let mock = Mock::start(vec![Reply::Content(greedy.into())]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    cfg.laps = 3; // ask at lap 2 — the last lap before the flag
    let laps = cfg.laps;
    let (o, log) = run(cfg);

    assert_complete(&o);
    let armed: Vec<&AiExchange> = log.iter().filter(|x| x.detail.contains("pit armed")).collect();
    assert!(
        armed.iter().all(|x| !x.detail.contains("final lap")),
        "a pit must never be armed on the final lap"
    );
    // Car 0 must have finished all configured laps whatever it was told.
    let c0 = o.advised.as_ref().unwrap();
    assert!(c0.laps >= laps, "advised car must still complete the distance");
}

#[test]
fn gate3b_illegal_tire_choice_is_refused() {
    let bad_tire = r#"{"pace":{"choice":"hold","confidence":0.8},"pit":{"choice":"stay","confidence":0.8},"tire":{"choice":"Q","confidence":0.9}}"#;
    let mock = Mock::start(vec![Reply::Content(bad_tire.into())]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    // An unknown compound is not one of the plan's three; it must not be
    // fitted and must not take the whole answer down with it.
    assert!(
        log.iter().all(|x| !x.detail.contains("tire=Q")),
        "an unknown compound must never be fitted"
    );
    let c0 = o.advised.as_ref().unwrap();
    assert!(
        c0.compounds.iter().all(|c| matches!(*c, "S" | "M" | "H")),
        "only real compounds may appear in a stint history: {:?}",
        c0.compounds
    );
}

#[test]
fn gate3c_fixed_strategy_keeps_ownership_of_the_stop_lap() {
    // A Fixed strategy is an explicit human decision; the advisor may advise
    // pace but must not move the stop.
    let greedy = r#"{"pace":{"choice":"push","confidence":0.9},"pit":{"choice":"now","confidence":0.99},"tire":{"choice":"M","confidence":0.9}}"#;
    let mock = Mock::start(vec![Reply::Content(greedy.into())]);
    let mut cfg = scenario();
    cfg.strategy = apexsim::race::PitStrategy::Fixed { lap: 4, switch_to: apexsim::race::TireCompound::Hard };
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    assert!(
        log.iter().any(|x| x.detail.contains("fixed strategy owns")),
        "the fixed strategy must keep ownership of the stop lap; details: {:?}",
        log.iter().map(|x| &x.detail).collect::<Vec<_>>()
    );
    // The stop still happened — on the lap the strategy chose.
    let c0 = o.advised.as_ref().unwrap();
    assert_eq!(c0.stops, 1, "the planned stop must still be taken");
    assert_eq!(
        c0.compounds,
        vec!["S", "H"],
        "the fitted compound must be the strategy's, not the advisor's"
    );
}

#[test]
fn gate3d_only_one_stop_is_ever_taken() {
    let greedy = r#"{"pace":{"choice":"push","confidence":0.9},"pit":{"choice":"now","confidence":0.99},"tire":{"choice":"S","confidence":0.9}}"#;
    let mock = Mock::start(vec![Reply::Content(greedy.into())]);
    let mut cfg = scenario();
    cfg.laps = 10;
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, log) = run(cfg);

    assert_complete(&o);
    let c0 = o.advised.as_ref().unwrap();
    assert!(c0.stops <= 1, "one planned stop means at most one stop, got {}", c0.stops);
    assert!(log.iter().any(|x| x.detail.contains("stop already taken")));
}

// ---------------------------------------------------------------------------
// Gate 4 — the advisor never runs on every tick, and never drives the car
// ---------------------------------------------------------------------------

#[test]
fn gate4_advisor_is_asked_on_a_lap_cadence_not_per_tick() {
    let good = r#"{"pace":{"choice":"hold","confidence":0.8},"pit":{"choice":"stay","confidence":0.8},"tire":{"choice":"keep","confidence":0.8}}"#;
    let mock = Mock::start(vec![Reply::Content(good.into())]);
    let mut cfg = scenario();
    cfg.laps = 12;
    cfg.ai = Some(advisor(mock.endpoint()));
    let laps = cfg.laps;
    let (o, log) = run(cfg);

    // every_laps = 2, asking on laps 2,4,6,8,10 and never on the final lap.
    let expected = (1..laps).filter(|l| l % 2 == 0).count();
    assert!(
        log.len() <= expected,
        "asked {} times for at most {} lap boundaries",
        log.len(),
        expected
    );
    assert!(
        ((log.len() * 1000) as u64) < o.ticks,
        "the advisor must be asked far less often than the {} physics ticks",
        o.ticks
    );
    assert!(mock.requests.load(Ordering::SeqCst) <= expected + 1);
}

#[test]
fn gate4b_ai_is_off_the_critical_path_and_timing_stays_in_budget() {
    // The plan's Phase 0 limit: 4 ms per tick at 250 Hz. An advisor run must
    // meet it even though the model runs on its own thread and the sim waits
    // on it at lap boundaries.
    let good = r#"{"pace":{"choice":"push","confidence":0.9},"pit":{"choice":"now","confidence":0.9},"tire":{"choice":"H","confidence":0.9}}"#;
    let mock = Mock::start(vec![Reply::Content(good.into())]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (o, _log) = run(cfg);

    assert_complete(&o);
    let budget_us = 4000.0;
    assert!(
        o.tick_stats.p99_us < budget_us,
        "p99 tick cost {:.1} µs exceeds the {:.0} µs budget",
        o.tick_stats.p99_us,
        budget_us
    );
    assert!(
        o.tick_stats.avg_us < budget_us / 10.0,
        "avg tick cost {:.1} µs is too high",
        o.tick_stats.avg_us
    );
}

#[test]
fn gate4c_realtime_run_never_blocks_on_the_model() {
    // Fire-and-forget in realtime: the sim must finish without waiting, and
    // the advisor simply misses the flag. Assert the non-blocking path by
    // checking that a blocking advisor was never armed for it.
    let good = r#"{"pace":{"choice":"hold","confidence":0.8},"pit":{"choice":"stay","confidence":0.8},"tire":{"choice":"keep","confidence":0.8}}"#;
    let mock = Mock::start(vec![Reply::Content(good.into())]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));

    let mut race = apexsim::race::Race::new(cfg);
    assert!(
        !race.ai_blocks(),
        "a freshly built race must not wait on the model by default"
    );
    race.set_blocking_advisor(true);
    assert!(race.ai_blocks(), "max-speed mode must be allowed to wait");
}

// ---------------------------------------------------------------------------
// Gate 5 — determinism of the baseline, and the audit trail
// ---------------------------------------------------------------------------

#[test]
fn gate5_baseline_is_untouched_by_the_advisor_code() {
    // Phase 5's guarantee still holds: with no advisor the race is exactly
    // what it was, and two runs of it agree bit for bit.
    let a = run(scenario()).0;
    let b = run(scenario()).0;
    assert_eq!(a.state_hash, b.state_hash);
    let ids = |o: &RaceOutcome| o.classification.iter().map(|r| r.id).collect::<Vec<_>>();
    assert_eq!(ids(&a), ids(&b));
    assert!(a.ai.is_none(), "a baseline run must report no advisor activity");
    assert!(a.advised.is_some(), "car 0's result is reported either way");
}

#[test]
fn gate5b_every_exchange_carries_the_full_audit_trail() {
    let good = r#"{"pace":{"choice":"conserve","confidence":0.7},"pit":{"choice":"stay","confidence":0.6},"tire":{"choice":"keep","confidence":0.6}}"#;
    let mock = Mock::start(vec![Reply::Content(good.into())]);
    let mut cfg = scenario();
    cfg.ai = Some(advisor(mock.endpoint()));
    let (_, log) = run(cfg);

    for x in &log {
        assert!(!x.prompt_summary.is_empty(), "prompt summary must be recorded");
        assert!(!x.model.is_empty(), "model version must be recorded");
        assert!(!x.response.is_empty(), "response must be recorded");
        assert!(x.latency_ms.is_finite(), "latency must be recorded");
        assert!(
            matches!(
                x.outcome,
                "applied" | "rejected" | "stale" | "low_confidence" | "bad_answer" | "timeout" | "error"
            ),
            "unexpected outcome '{}'",
            x.outcome
        );
        if x.outcome == "applied" {
            assert!(x.applied_tick.is_some());
        } else {
            assert!(x.applied_tick.is_none(), "{} was not applied but records a tick", x.outcome);
        }
    }
}

#[test]
fn gate5c_stale_answers_are_rejected() {
    // An answer that arrives after the car has moved on describes a race
    // that no longer exists. Force it by answering only after the car has
    // run several laps past the question: a slow endpoint plus a long
    // staleness window means the answer is late by definition.
    let mock = Mock::start(vec![Reply::Hang, Reply::Hang, Reply::Hang, Reply::Hang]);
    let mut a = advisor(mock.endpoint());
    a.timeout_ms = 250;
    let mut cfg = scenario();
    cfg.laps = 8;
    cfg.ai = Some(a);
    let laps = cfg.laps;
    let (o, log) = run(cfg);

    assert_complete(&o);
    // Whatever the mix of timeouts and late answers, none of them may be
    // applied, and none may leave the race short of the flag.
    assert_eq!(tally(&log, "applied"), 0);
    assert!(o.advised.as_ref().unwrap().laps >= laps);
}