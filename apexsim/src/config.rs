//! Phase 6: the documented development configuration file.
//!
//! `apexsim dev --init` writes `f1_dev.toml` (template below, comments
//! included); the launcher and `serve --config` read it back. The format
//! is deliberately tiny — `key = value` lines with `#` comments, parsed
//! here by hand so the engine keeps its dependency surface flat. Unknown
//! keys and out-of-range values are errors, not warnings: the plan's Phase
//! 6 goal is that failures are visible.

/// Keys, defaults, and meaning are all in this template — the file is the
/// documentation.
pub const CONFIG_TEMPLATE: &str = r#"# f1_dev.toml — development launcher configuration for apexsim.
# Start everything with:  apexsim dev
# Regenerate this file:   apexsim dev --init   (refuses to overwrite)
#
# Format: key = value lines, # comments. Unknown keys are errors.

# Dashboard port (the launcher waits for this to report ready).
port = 8080

# Field size (20 = the 2024-season grid; 1 = single-car session).
cars = 20

# Laps per session (serve uses this as the session length).
laps = 12

# Deterministic seed: same seed, same race.
seed = 42

# Driver-input noise amplitude (0 = silent, 0.1 = extreme).
noise = 0.002

# Session mode: race | qualifying | practice.
mode = race

# Launch straight into the formation/start sequence.
autostart = true

# How long the launcher waits for /api/health before declaring failure.
readiness_timeout_ms = 10000

# Minimum severity that gets logged: debug | info | warn | error.
log_level = info

# Optional structured-log mirror file (append). Leave unset for stdout only.
# log_file = logs/apexsim_dev.log
"#;

#[derive(Clone, Debug, PartialEq)]
pub struct DevConfig {
    pub port: u16,
    pub cars: u32,
    pub laps: u32,
    pub seed: u64,
    pub noise: f64,
    pub mode: String,
    pub autostart: bool,
    pub readiness_timeout_ms: u64,
    pub log_level: String,
    pub log_file: Option<String>,
}

impl Default for DevConfig {
    fn default() -> Self {
        DevConfig {
            port: 8080,
            cars: 20,
            laps: 12,
            seed: 42,
            noise: 0.002,
            mode: "race".into(),
            autostart: true,
            readiness_timeout_ms: 10_000,
            log_level: "info".into(),
            log_file: None,
        }
    }
}

impl DevConfig {
    /// Range validation with explicit messages — a bad value must name
    /// itself.
    pub fn validate(&self) -> Result<(), String> {
        if self.cars == 0 || self.cars > 30 {
            return Err(format!("cars must be 1..=30, got {}", self.cars));
        }
        if self.laps == 0 || self.laps > 1000 {
            return Err(format!("laps must be 1..=1000, got {}", self.laps));
        }
        if !(0.0..=0.1).contains(&self.noise) {
            return Err(format!("noise must be 0.0..=0.1, got {}", self.noise));
        }
        if crate::race::SessionMode::parse(&self.mode).is_none() {
            return Err(format!(
                "mode must be race | qualifying | practice, got '{}'",
                self.mode
            ));
        }
        if crate::logging::Level::parse(&self.log_level).is_none() {
            return Err(format!(
                "log_level must be debug | info | warn | error, got '{}'",
                self.log_level
            ));
        }
        if self.readiness_timeout_ms == 0 || self.readiness_timeout_ms > 120_000 {
            return Err(format!(
                "readiness_timeout_ms must be 1..=120000, got {}",
                self.readiness_timeout_ms
            ));
        }
        Ok(())
    }
}

/// Parse the key=value format. Every line must be blank, a comment, or a
/// known key with a well-typed value; anything else is an error naming the
/// line.
pub fn parse_config(text: &str) -> Result<DevConfig, String> {
    let mut cfg = DevConfig::default();
    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("config line {line_no}: expected 'key = value', got '{raw}'"))?;
        let key = key.trim();
        let value = value.trim();
        // Strip a trailing comment from the value ("port = 8080 # inline").
        let value = match value.split_once('#') {
            Some((v, _)) => v.trim(),
            None => value,
        };
        if value.is_empty() {
            return Err(format!("config line {line_no}: '{key}' has no value"));
        }
        match key {
            "port" => cfg.port = value.parse().map_err(|e| cfgerr(line_no, key, value, e))?,
            "cars" => cfg.cars = value.parse().map_err(|e| cfgerr(line_no, key, value, e))?,
            "laps" => cfg.laps = value.parse().map_err(|e| cfgerr(line_no, key, value, e))?,
            "seed" => cfg.seed = value.parse().map_err(|e| cfgerr(line_no, key, value, e))?,
            "noise" => cfg.noise = value.parse().map_err(|e| cfgerr(line_no, key, value, e))?,
            "mode" => cfg.mode = value.to_string(),
            "autostart" => {
                cfg.autostart = parse_bool(value)
                    .ok_or_else(|| format!("config line {line_no}: autostart must be true|false, got '{value}'"))?;
            }
            "readiness_timeout_ms" => {
                cfg.readiness_timeout_ms = value
                    .parse()
                    .map_err(|e| cfgerr(line_no, key, value, e))?
            }
            "log_level" => cfg.log_level = value.to_string(),
            "log_file" => cfg.log_file = Some(value.to_string()),
            other => {
                return Err(format!(
                    "config line {line_no}: unknown key '{other}' (see `apexsim dev --init` for the documented keys)"
                ))
            }
        }
    }
    cfg.validate()?;
    Ok(cfg)
}

fn cfgerr(line_no: usize, key: &str, value: &str, e: impl std::fmt::Display) -> String {
    format!("config line {line_no}: {key} = '{value}' is not a valid value ({e})")
}

fn parse_bool(v: &str) -> Option<bool> {
    match v {
        "true" | "yes" | "1" => Some(true),
        "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_parses_and_validates() {
        let cfg = parse_config(CONFIG_TEMPLATE).expect("documented template must parse");
        assert_eq!(cfg, DevConfig::default());
        // The only commented-out line must stay optional.
        assert_eq!(cfg.log_file, None);
    }

    #[test]
    fn unknown_keys_and_bad_values_are_errors() {
        let err = parse_config("bogus = 1\n").unwrap_err();
        assert!(err.contains("unknown key 'bogus'"), "{err}");
        let err = parse_config("cars = 99\n").unwrap_err();
        assert!(err.contains("cars must be 1..=30"), "{err}");
        let err = parse_config("noise = 7\n").unwrap_err();
        assert!(err.contains("noise must be 0.0..=0.1"), "{err}");
        let err = parse_config("mode = sprint\n").unwrap_err();
        assert!(err.contains("mode must be race"), "{err}");
        let err = parse_config("garbage line\n").unwrap_err();
        assert!(err.contains("expected 'key = value'"), "{err}");
    }

    #[test]
    fn overrides_and_comments_apply() {
        let cfg = parse_config(
            "# comment\nport = 9099 # inline\ncars = 4\nautostart = false\nlog_file = logs/dev.log\n",
        )
        .unwrap();
        assert_eq!(cfg.port, 9099);
        assert_eq!(cfg.cars, 4);
        assert!(!cfg.autostart);
        assert_eq!(cfg.log_file.as_deref(), Some("logs/dev.log"));
    }
}
