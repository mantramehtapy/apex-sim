//! Phase 6: structured logging.
//!
//! One JSON object per line (NDJSON) on stdout, optionally mirrored to a
//! log file. Every line carries the plan's required fields — run ID,
//! severity, and (when they exist) the session ID and simulation tick — so
//! a development session can be reconstructed and grepped after the fact.
//!
//! Logging happens at lifecycle events only (boot, ready, session changes,
//! finish, errors, shutdown), never per physics tick: the sim thread's
//! 250 Hz loop must stay predictable, and Phase 3 measured it that way.

use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

impl Level {
    pub fn label(self) -> &'static str {
        match self {
            Level::Debug => "debug",
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Error => "error",
        }
    }
    pub fn parse(s: &str) -> Option<Level> {
        match s {
            "debug" => Some(Level::Debug),
            "info" => Some(Level::Info),
            "warn" | "warning" => Some(Level::Warn),
            "error" => Some(Level::Error),
            _ => None,
        }
    }
}

/// One logger per run. Cheap to clone behind an `Arc`; `log` takes `&self`.
pub struct Logger {
    run_id: String,
    min: Level,
    file: Option<Mutex<File>>,
    start: Instant,
}

impl Logger {
    /// Create a logger. `file_path`, when given, is opened append (parents
    /// must exist); a failure to open is reported via `Err` — a log file
    /// that cannot be written is a visible problem, not a silent one.
    pub fn new(run_id: String, min: Level, file_path: Option<&std::path::Path>) -> std::io::Result<Logger> {
        let file = match file_path {
            Some(p) => Some(Mutex::new(
                std::fs::OpenOptions::new().create(true).append(true).open(p)?,
            )),
            None => None,
        };
        Ok(Logger { run_id, min, file, start: Instant::now() })
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Emit one structured line. `sid` and `tick` are the session id and
    /// simulation tick when the event belongs to a session; both optional.
    pub fn log(&self, level: Level, sid: Option<&str>, tick: Option<u64>, msg: &str) {
        if level < self.min {
            return;
        }
        let mut obj = json!({
            "ts_s": (self.start.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
            "level": level.label(),
            "run_id": self.run_id,
            "msg": msg,
        });
        if let Some(sid) = sid {
            obj["sid"] = json!(sid);
        }
        if let Some(tick) = tick {
            obj["tick"] = json!(tick);
        }
        let mut line = obj.to_string();
        line.push('\n');
        {
            use std::io::Write as _;
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(line.as_bytes());
            let _ = out.flush();
        }
        if let Some(f) = &self.file {
            if let Ok(mut f) = f.lock() {
                let _ = f.write_all(line.as_bytes());
            }
        }
    }

    pub fn info(&self, msg: &str) {
        self.log(Level::Info, None, None, msg);
    }
    pub fn warn(&self, msg: &str) {
        self.log(Level::Warn, None, None, msg);
    }
    pub fn error(&self, msg: &str) {
        self.log(Level::Error, None, None, msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_ordering_and_parse() {
        assert!(Level::Debug < Level::Info && Level::Info < Level::Warn && Level::Warn < Level::Error);
        assert_eq!(Level::parse("info"), Some(Level::Info));
        assert_eq!(Level::parse("warning"), Some(Level::Warn));
        assert_eq!(Level::parse("bogus"), None);
    }

    #[test]
    fn lines_carry_required_fields() {
        let dir = std::env::temp_dir().join(format!("apexsim_log_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run.log");
        let logger =
            Logger::new("run-test-1".into(), Level::Info, Some(&path)).unwrap();
        logger.log(Level::Info, Some("s42-1"), Some(1234), "session started");
        logger.log(Level::Error, None, None, "boom");
        logger.log(Level::Debug, None, None, "filtered out");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "debug line must be filtered by min level");
        let a: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(a["run_id"], "run-test-1");
        assert_eq!(a["level"], "info");
        assert_eq!(a["sid"], "s42-1");
        assert_eq!(a["tick"], 1234);
        assert_eq!(a["msg"], "session started");
        let b: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(b["level"], "error");
        assert_eq!(b["msg"], "boom");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
