//! Phase 7: local LLM strategy advisor (Ollama, plan model `tev1:0.8b`).
//!
//! A LOW-FREQUENCY tactical advisor for one car — never a steering loop:
//! the advisor runs on its OWN thread (the 250 Hz sim loop never blocks on
//! the model), is asked at most every `every_laps` laps, and its answer is
//! only ever a typed suggestion. Deterministic Rust rules own legality:
//! legal actions, pit feasibility, and fallback. Every exchange is recorded
//! — prompt summary, model, raw response, latency, application tick, and
//! outcome — so the benchmark can say when the model helps, ties, or hurts.
//!
//! Endpoint: stock Ollama speaks the OpenAI-compatible chat protocol, so
//! the default is `http://127.0.0.1:11434/v1/chat/completions`. The plan's
//! `/v1/systemone` path is not part of stock Ollama; the endpoint is
//! configurable so a dedicated Tev1 server can be plugged in unchanged.
//! The model defaults to the plan's `tev1:0.8b` (requires Ollama ≥ 0.35);
//! any local model works (`--ai-model qwen2.5:3b`).

use serde_json::json;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::time::Duration;

/// Advisor configuration (CLI: `--ai`, `--ai-model`, ...).
#[derive(Clone, Debug)]
pub struct AdvisorConfig {
    /// Full URL of an OpenAI-compatible chat-completions endpoint.
    pub endpoint: String,
    pub model: String,
    /// Ask every N completed laps of the advised car.
    pub every_laps: u32,
    /// Hard bound on one model call.
    pub timeout_ms: u64,
    /// Hard bound on generated tokens. Necessary, not optional: `tev1:0.8b`
    /// is a reasoning model and will happily spend thousands of tokens
    /// thinking aloud, so without a cap the answer never arrives inside the
    /// timeout and the advisor falls back on every single question.
    pub max_tokens: u32,
    /// Suggestions below this confidence are ignored (fallback keeps the
    /// deterministic plan). Confidence is concentration, not correctness —
    /// it gates only.
    pub min_confidence: f64,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        AdvisorConfig {
            endpoint: "http://127.0.0.1:11434/v1/chat/completions".into(),
            model: "tev1:0.8b".into(),
            every_laps: 2,
            timeout_ms: 5_000,
            max_tokens: 320,
            min_confidence: 0.5,
        }
    }
}

/// A typed suggestion for the advised car. `None` = the model did not give
/// a usable answer for that question (that question falls back).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrategyAdvice {
    pub pace: Option<(PaceChoice, f64)>,
    /// true = "pit this lap".
    pub pit: Option<(bool, f64)>,
    /// Requested compound for the (next) stop; `None` = keep.
    pub tire: Option<(Option<crate::race::TireCompound>, f64)>,
    /// True when at least one confidence was imputed rather than stated —
    /// a schema-lite answer that named the choices but not how sure it was.
    pub imputed: bool,
}

/// Confidence attributed to a choice the model stated without one. Small
/// local models routinely answer `{"pace":"hold"}` and skip the confidence
/// entirely; imputing exactly the default gate keeps them able to advise
/// while the `imputed` flag records that the number was not the model's.
pub const IMPUTED_CONFIDENCE: f64 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaceChoice {
    Push,
    Hold,
    Conserve,
}

impl PaceChoice {
    fn parse(s: &str) -> Option<PaceChoice> {
        match s.trim().to_ascii_lowercase().as_str() {
            "push" | "attack" => Some(PaceChoice::Push),
            "hold" | "maintain" => Some(PaceChoice::Hold),
            "conserve" | "lift" => Some(PaceChoice::Conserve),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            PaceChoice::Push => "push",
            PaceChoice::Hold => "hold",
            PaceChoice::Conserve => "conserve",
        }
    }
    /// The pace multiplier each choice maps to (legal band, clamped in
    /// Race before use).
    pub fn scale(self) -> f64 {
        match self {
            PaceChoice::Push => 1.04,
            PaceChoice::Hold => 1.0,
            PaceChoice::Conserve => 0.95,
        }
    }
}

#[derive(Clone, Debug)]
pub enum AiError {
    /// Could not reach the endpoint at all (service down, refused).
    Unavailable(String),
    /// No answer within the timeout.
    Timeout,
    /// HTTP-level failure (404 = model not loaded/unknown).
    Http(u16, String),
    /// Reached the model but the payload was unusable.
    BadResponse(String),
}

/// One recorded exchange — everything the plan requires on the record.
#[derive(Clone, Debug)]
pub struct AiExchange {
    pub request_tick: u64,
    pub applied_tick: Option<u64>,
    /// The compact race summary sent to the model (the prompt payload).
    pub prompt_summary: String,
    pub model: String,
    pub response: String,
    pub latency_ms: f64,
    /// applied | rejected | low_confidence | bad_answer | timeout |
    /// error | stale | skipped_disabled
    pub outcome: &'static str,
    pub detail: String,
}

// ---------------------------------------------------------------------------
// Raw HTTP client (std-only; no new dependencies)
// ---------------------------------------------------------------------------

/// One blocking chat call, bounded by `timeout`. Returns the assistant
/// message content.
pub fn chat(
    endpoint: &str,
    model: &str,
    system: &str,
    user: &str,
    timeout: Duration,
    max_tokens: u32,
) -> Result<String, AiError> {
    // endpoint = http://host:port/path
    let (hostport, path) = match endpoint.split_once("://") {
        Some((_, rest)) => match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        },
        None => return Err(AiError::BadResponse(format!("bad endpoint '{endpoint}'"))),
    };
    let addr: SocketAddr = format!(
        "{}:{}",
        hostport.rsplit_once(':').map(|(h, _)| h).unwrap_or(hostport),
        hostport.rsplit_once(':').map(|(_, p)| p).unwrap_or("80")
    )
    .parse()
    .map_err(|e| AiError::BadResponse(format!("bad endpoint host '{hostport}': {e}")))?;

    let body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": false,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        // Ollama reasoning models: keep the chain of thought off the wire so
        // the answer is the message content, and so the token budget is
        // spent on the answer rather than on deliberation.
        "think": false,
    })
    .to_string();

    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );

    let connect_deadline = std::time::Instant::now() + Duration::from_millis(1_500);
    let mut stream = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(1_500))
        .map_err(|e| AiError::Unavailable(format!("connect {addr}: {e}")))?;
    let _ = connect_deadline;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| AiError::Unavailable(format!("set timeout: {e}")))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| AiError::Unavailable(format!("set timeout: {e}")))?;
    stream
        .write_all(req.as_bytes())
        .map_err(|e| AiError::Unavailable(format!("write: {e}")))?;

    let started = std::time::Instant::now();
    let mut raw = Vec::new();
    match stream.read_to_end(&mut raw) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => {
            return Err(AiError::Timeout);
        }
        Err(e) => return Err(AiError::Unavailable(format!("read: {e}"))),
    }
    let _ = started;
    let text = String::from_utf8_lossy(&raw);
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| AiError::BadResponse("no HTTP body delimiter".into()))?;
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| AiError::BadResponse(format!("bad status line: {}", head.lines().next().unwrap_or(""))))?;
    if status != 200 {
        return Err(AiError::Http(status, body.chars().take(300).collect()));
    }
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| AiError::BadResponse(format!("json: {e}")))?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| AiError::BadResponse("missing choices[0].message.content".into()))?
        .to_string();
    Ok(content)
}

// ---------------------------------------------------------------------------
// Advice parsing + validation (typed questions, lenient extraction)
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "\
You are the race strategist of a racing driver. You will receive a compact \
race summary as JSON. Reply with ONLY one JSON object, no prose, no \
explanation. Example of a correct reply: \
{\"pace\":{\"choice\":\"push\",\"confidence\":0.9},\
\"pit\":{\"choice\":\"stay\",\"confidence\":0.8},\
\"tire\":{\"choice\":\"M\",\"confidence\":0.7}}. \
Rules: pace.choice is push, hold or conserve; pit.choice is now or stay; \
tire.choice is S, M, H or keep; each confidence is a number from 0.0 to \
1.0. pace = how hard to push this stint; pit = pit at the next \
opportunity; tire = compound to fit at the next stop (S=soft, M=medium, \
H=hard).";

/// Extract the first balanced JSON object from free-form model text.
fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0usize;
    for (i, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=start + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse the model's content into typed advice. Fails only when no JSON
/// object at all can be recovered (that is a `bad_answer`); individual
/// fields may be missing or malformed — those become `None` (fallback).
///
/// Accepts two shapes, because both are what local models actually emit:
///   nested  `{"pace":{"choice":"push","confidence":0.9}, ...}`  (asked for)
///   flat    `{"pace":"push","pit":"stay","tire":"M"}`            (common)
/// A *stated* confidence outside 0..1 is malformed → `None`. A *missing*
/// one is imputed at [`IMPUTED_CONFIDENCE`] and sets `imputed`.
pub fn parse_advice(content: &str) -> Result<StrategyAdvice, String> {
    let obj = extract_json_object(content).ok_or_else(|| {
        format!(
            "no JSON object in response: {}",
            content.chars().take(120).collect::<String>()
        )
    })?;
    let v: serde_json::Value =
        serde_json::from_str(obj).map_err(|e| format!("json: {e}"))?;
    let mut advice = StrategyAdvice::default();

    // Read one question's answer. Returns None for an unusable answer;
    // `imputed` is set when the model stated the choice but not a confidence.
    let read = |field: &str| -> Option<(String, Option<f64>, bool)> {
        let raw = v.get(field)?;
        // flat form: the value is the choice string itself
        if let Some(s) = raw.as_str() {
            return Some((s.trim().to_ascii_lowercase(), None, true));
        }
        let choice = raw
            .get("choice")
            .and_then(|c| c.as_str())?
            .trim()
            .to_ascii_lowercase();
        let stated = raw.get("confidence").and_then(|c| c.as_f64());
        match stated {
            // present but out of range = malformed, not missing
            Some(f) if !(0.0..=1.0).contains(&f) => None,
            Some(f) => Some((choice, Some(f), false)),
            None => Some((choice, None, true)),
        }
    };

    if let Some((c, conf, imputed)) = read("pace") {
        if let Some(p) = PaceChoice::parse(&c) {
            advice.pace = Some((p, conf.unwrap_or(IMPUTED_CONFIDENCE)));
            advice.imputed |= imputed;
        }
    }
    if let Some((c, conf, imputed)) = read("pit") {
        let choice = match c.as_str() {
            "now" | "yes" | "pit" => Some(true),
            "stay" | "no" => Some(false),
            _ => None,
        };
        if let Some(choice) = choice {
            advice.pit = Some((choice, conf.unwrap_or(IMPUTED_CONFIDENCE)));
            advice.imputed |= imputed;
        }
    }
    if let Some((c, conf, imputed)) = read("tire") {
        // "keep" means no change; anything else must name a real compound.
        // An unrecognised compound is not a fatal parse error — that one
        // question simply falls back to the deterministic plan.
        let compound: Option<Option<crate::race::TireCompound>> = match c.as_str() {
            "keep" | "none" | "same" => Some(None),
            other => crate::race::TireCompound::parse(other).map(Some),
        };
        if let Some(compound) = compound {
            advice.tire = Some((compound, conf.unwrap_or(IMPUTED_CONFIDENCE)));
            advice.imputed |= imputed;
        }
    }
    Ok(advice)
}

/// Build the compact user prompt from the race summary JSON.
pub fn build_prompt(summary_json: &str) -> String {
    format!(
        "Race summary:\n{summary_json}\n\nDecide pace, pit intent, and tire \
         choice. Pit is only legal before the final lap and only one stop \
         is planned. Answer with only the JSON object."
    )
}

// ---------------------------------------------------------------------------
// Advisor thread (the sim loop never blocks on the model)
// ---------------------------------------------------------------------------

pub struct AdviceRequest {
    pub id: u64,
    pub tick: u64,
    /// Compact race summary (embedded into the user prompt).
    pub summary: String,
}

pub struct AdviceReply {
    pub id: u64,
    pub result: Result<String, AiError>,
    pub latency_ms: f64,
}

pub struct AdvisorHandle {
    pub req_tx: std::sync::mpsc::Sender<AdviceRequest>,
    pub rep_rx: std::sync::mpsc::Receiver<AdviceReply>,
}

/// Spawn the advisor worker thread. One request in flight at a time; the
/// sim side never waits on it.
pub fn spawn_advisor(cfg: &AdvisorConfig) -> AdvisorHandle {
    let cfg = cfg.clone();
    let (req_tx, req_rx) = std::sync::mpsc::channel::<AdviceRequest>();
    let (rep_tx, rep_rx) = std::sync::mpsc::channel::<AdviceReply>();
    std::thread::Builder::new()
        .name("ai-advisor".into())
        .spawn(move || {
            while let Ok(req) = req_rx.recv() {
                let t0 = std::time::Instant::now();
                let result = chat(
                    &cfg.endpoint,
                    &cfg.model,
                    SYSTEM_PROMPT,
                    &build_prompt(&req.summary),
                    Duration::from_millis(cfg.timeout_ms),
                    cfg.max_tokens,
                );
                let latency_ms = t0.elapsed().as_secs_f64() * 1e3;
                if rep_tx.send(AdviceReply { id: req.id, result, latency_ms }).is_err() {
                    break; // sim side gone
                }
            }
        })
        .expect("spawn advisor thread");
    AdvisorHandle { req_tx, rep_rx }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wellformed_advice() {
        let a = parse_advice(
            "{\"pace\":{\"choice\":\"push\",\"confidence\":0.9},\
              \"pit\":{\"choice\":\"now\",\"confidence\":0.7},\
              \"tire\":{\"choice\":\"M\",\"confidence\":0.6}}",
        )
        .unwrap();
        assert_eq!(a.pace, Some((PaceChoice::Push, 0.9)));
        assert_eq!(a.pit, Some((true, 0.7)));
        assert_eq!(a.tire, Some((Some(crate::race::TireCompound::Medium), 0.6)));
    }

    #[test]
    fn parses_prose_wrapped_json_and_partial_fields() {
        let a = parse_advice("Sure! {\"pace\":{\"choice\":\"conserve\",\"confidence\":0.8}} hope that helps").unwrap();
        assert_eq!(a.pace, Some((PaceChoice::Conserve, 0.8)));
        assert_eq!(a.pit, None);
        assert_eq!(a.tire, None);
    }

    #[test]
    fn rejects_answer_without_json() {
        assert!(parse_advice("I would pit around lap 6.").is_err());
    }

    #[test]
    fn illegal_values_fall_back_to_none() {
        let a = parse_advice(
            "{\"pace\":{\"choice\":\"zoom\",\"confidence\":0.9},\
              \"pit\":{\"choice\":\"now\",\"confidence\":2.0},\
              \"tire\":{\"choice\":\"W\",\"confidence\":0.5}}",
        )
        .unwrap();
        assert_eq!(a.pace, None);
        assert_eq!(a.pit, None);
        assert_eq!(a.tire, None);
    }

    #[test]
    fn parses_flat_schema_lite_answers() {
        // What qwen2.5:3b emits before the prompt carries a worked example:
        // the choices, but no confidence at all.
        let a = parse_advice("{\"pace\":\"hold\",\"pit\":\"stay\",\"tire\":\"M\"}").unwrap();
        assert_eq!(a.pace, Some((PaceChoice::Hold, IMPUTED_CONFIDENCE)));
        assert_eq!(a.pit, Some((false, IMPUTED_CONFIDENCE)));
        assert_eq!(a.tire, Some((Some(crate::race::TireCompound::Medium), IMPUTED_CONFIDENCE)));
        assert!(a.imputed, "missing confidence must be flagged as imputed");
    }

    #[test]
    fn stated_confidence_is_not_imputed() {
        let a = parse_advice(
            "{\"pace\":{\"choice\":\"push\",\"confidence\":0.9},\"pit\":{\"choice\":\"now\"}}",
        )
        .unwrap();
        assert_eq!(a.pace, Some((PaceChoice::Push, 0.9)));
        assert_eq!(a.pit, Some((true, IMPUTED_CONFIDENCE)));
        assert!(a.imputed, "pit had no confidence");
    }

    #[test]
    fn prose_wrapped_flat_json_is_recovered() {
        let a = parse_advice("Strategy call: {\"pace\":\"conserve\",\"tire\":\"keep\"} done.")
            .unwrap();
        assert_eq!(a.pace, Some((PaceChoice::Conserve, IMPUTED_CONFIDENCE)));
        assert_eq!(a.tire, Some((None, IMPUTED_CONFIDENCE)));
        assert_eq!(a.pit, None);
    }

    #[test]
    fn prompt_contains_summary() {
        let p = build_prompt("{\"lap\":3}");
        assert!(p.contains("{\"lap\":3}"));
    }

    #[test]
    fn system_prompt_carries_a_worked_example() {
        // Probed against real local models: without a concrete example
        // qwen2.5:3b answers the flat schema with no confidences at all.
        assert!(SYSTEM_PROMPT.contains("\"pace\":{\"choice\":\"push\",\"confidence\":0.9}"));
        assert!(SYSTEM_PROMPT.contains("no prose"));
    }
}
