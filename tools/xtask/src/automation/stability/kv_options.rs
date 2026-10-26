//! closed KV certification inputs, separate from general stability options.
use crate::repository::check_args::{Grammar, ParsedArgs};
use std::{path::PathBuf, time::Duration};
use url::Url;
pub(super) const USAGE: &str = "automation stability kv-tool-loop [--base-url URL] [--models CSV] [--attempts N] [--pressure-turns N] [--overlap-requests N] [--timeout SECONDS] [--min-cached-tokens N] [--suffix-prefill-limit N] [--native-log PATH]... [--output-dir DIR] [--print-plan]";
const GRAMMAR: Grammar = Grammar {
    usage: USAGE,
    values: &[
        "--base-url",
        "--models",
        "--attempts",
        "--pressure-turns",
        "--overlap-requests",
        "--timeout",
        "--min-cached-tokens",
        "--suffix-prefill-limit",
        "--native-log",
        "--output-dir",
    ],
    flags: &["--print-plan"],
};
pub(super) struct Options {
    pub base: Url,
    pub models: Vec<String>,
    pub attempts: u32,
    pub pressure_turns: u32,
    pub overlap_requests: usize,
    pub timeout: Duration,
    pub minimum_cached: u64,
    pub suffix_limit: u64,
    pub native_logs: Vec<PathBuf>,
    pub output: PathBuf,
    pub plan: bool,
}
impl Options {
    pub fn parse(args: &[String]) -> Result<Self, String> {
        Self::with_environment(args, |name| std::env::var(name).ok())
    }
    fn with_environment(
        args: &[String],
        environment: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        let parsed = GRAMMAR
            .parse(args)
            .map_err(|_| "invalid KV stability arguments")?;
        if !parsed.positionals.is_empty() {
            return Err("KV stability accepts named inputs only".into());
        }
        let value = |flag: &str, suffix: &str, default: &str| {
            parsed
                .last(flag)
                .map(str::to_owned)
                .or_else(|| environment(&format!("MESH_KV_TOOL_LOOP_{suffix}")))
                .unwrap_or_else(|| default.into())
        };
        let base = parsed
            .last("--base-url")
            .map(str::to_owned)
            .or_else(|| environment("MESH_KV_TOOL_LOOP_BASE_URL").filter(|value| !value.is_empty()))
            .or_else(|| environment("MESH_CLIENT_API_BASE").filter(|value| !value.is_empty()))
            .unwrap_or_else(|| "http://127.0.0.1:9337/v1".into());
        let models = value("--models", "MODELS", "auto,mesh")
            .split(',')
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if models.is_empty()
            || models.len() > 256
            || models
                .iter()
                .any(|model| model.len() > 4096 || model.contains(['\0', '\r', '\n']))
        {
            return Err("KV stability requires 1..256 valid model ids".into());
        }
        let attempts = integer(&value("--attempts", "ATTEMPTS", "3"), "attempts")?;
        let pressure = integer(
            &value("--pressure-turns", "PRESSURE_TURNS", "6"),
            "pressure-turns",
        )?;
        let overlap = integer(
            &value("--overlap-requests", "OVERLAP_REQUESTS", "2"),
            "overlap-requests",
        )?;
        if !(1..=1000).contains(&attempts) || pressure > 128 || !(2..=64).contains(&overlap) {
            return Err("KV stability requires 1..1000 attempts, 0..128 pressure turns and 2..64 overlapping requests".into());
        }
        let cells = models.len() as u64 * attempts;
        // Each attempt includes the pressure loop and overlapping starts plus three tool follow-ups.
        // Both standalone cache probes use one warm and one measured request per model.
        let request_upper_bound = cells * (pressure + 4 * overlap + 3) + models.len() as u64 * 4;
        if cells > 10_000 || request_upper_bound > 100_000 {
            return Err("KV stability cohort exceeds its bounded request budget".into());
        }
        let output = value(
            "--output-dir",
            "OUTPUT_DIR",
            "target/kv-tool-loop-stability/latest",
        );
        if output.is_empty() {
            return Err("KV stability output path is empty".into());
        }
        let log_environment = environment("MESH_KV_TOOL_LOOP_NATIVE_LOGS");
        let home = environment("HOME").or_else(|| environment("USERPROFILE"));
        let native_logs = native_logs(&parsed, log_environment.as_deref(), home.as_deref())?;
        Ok(Self {
            base: super::options::normalize_base(&base)?,
            models,
            attempts: attempts as u32,
            pressure_turns: pressure as u32,
            overlap_requests: overlap as usize,
            timeout: seconds(&value("--timeout", "TIMEOUT", "180"))?,
            minimum_cached: integer(
                &value("--min-cached-tokens", "MIN_CACHED_TOKENS", "2048"),
                "min-cached-tokens",
            )?,
            suffix_limit: integer(
                &value("--suffix-prefill-limit", "SUFFIX_PREFILL_LIMIT", "256"),
                "suffix-prefill-limit",
            )?,
            native_logs,
            output: output.into(),
            plan: parsed.flag("--print-plan"),
        })
    }
}
fn integer(value: &str, name: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{name} must be a nonnegative integer"))
}
fn seconds(value: &str) -> Result<Duration, String> {
    let seconds = value
        .parse::<f64>()
        .map_err(|_| "KV timeout must be numeric")?;
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 3600.0 {
        return Err("KV timeout must be finite and within (0,3600] seconds".into());
    }
    let duration =
        Duration::try_from_secs_f64(seconds).map_err(|_| "KV timeout cannot be represented")?;
    if duration.is_zero() {
        return Err("KV timeout is below clock resolution".into());
    }
    Ok(duration)
}
fn native_logs(
    parsed: &ParsedArgs,
    environment: Option<&str>,
    home: Option<&str>,
) -> Result<Vec<PathBuf>, String> {
    let environment = environment.unwrap_or("");
    if environment.len() > 131072 {
        return Err("KV native log environment exceeds its bounded size".into());
    }
    let mut paths = Vec::new();
    let environment_paths = environment
        .split(',')
        .map(str::trim)
        .filter(|path| !path.is_empty());
    for path in environment_paths.chain(parsed.all("--native-log").iter().copied()) {
        if path.is_empty() || path.len() > 4096 || path.contains(['\0', '\r', '\n']) {
            return Err("KV native logs require bounded nonempty file paths".into());
        }
        let path = if let Some(relative) = path.strip_prefix("~/") {
            PathBuf::from(
                home.filter(|home| !home.is_empty())
                    .ok_or("KV native log home expansion requires HOME or USERPROFILE")?,
            )
            .join(relative)
        } else {
            PathBuf::from(path)
        };
        if !paths.contains(&path) {
            paths.push(path);
        }
        if paths.len() > 32 {
            return Err("KV native logs exceed 32 unique file paths".into());
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }
    #[test]
    fn kv_options_preserve_environment_defaults_and_explicit_cohort_overrides() {
        let options = Options::with_environment(
            &args(&[
                "--models",
                "fixture",
                "--pressure-turns",
                "0",
                "--native-log",
                "first.log",
                "--native-log",
                "second log",
                "--print-plan",
            ]),
            |name| match name {
                "MESH_KV_TOOL_LOOP_MODELS" => Some("ignored".into()),
                "MESH_KV_TOOL_LOOP_ATTEMPTS" => Some("2".into()),
                "MESH_CLIENT_API_BASE" => Some("https://fixture.invalid/tenant/v1".into()),
                "MESH_KV_TOOL_LOOP_NATIVE_LOGS" => Some("first.log, ~/native.log".into()),
                "HOME" => Some("/fixture-home".into()),
                _ => None,
            },
        )
        .unwrap();
        assert_eq!(options.models, ["fixture"]);
        assert_eq!(options.attempts, 2);
        assert_eq!(options.pressure_turns, 0);
        assert_eq!(options.overlap_requests, 2);
        assert_eq!(options.base.as_str(), "https://fixture.invalid/tenant/v1");
        assert_eq!(
            options.native_logs,
            [
                PathBuf::from("first.log"),
                PathBuf::from("/fixture-home/native.log"),
                PathBuf::from("second log")
            ]
        );
        assert!(options.plan);
    }
    #[test]
    fn kv_options_reject_unbounded_invalid_and_cross_mode_inputs() {
        for values in [
            &["--attempts", "0"][..],
            &["--pressure-turns", "129"],
            &["--overlap-requests", "1"],
            &["--overlap-requests", "65"],
            &["--timeout", "NaN"],
            &["--timeout", "inf"],
            &["--timeout", "0"],
            &["--min-cached-tokens", "-1"],
            &["--suffix-prefill-limit", "1.5"],
            &["--agent-smokes", "pi"],
            &["--skip-streaming"],
            &["--models", ""],
            &["--native-log", ""],
        ] {
            assert!(
                Options::with_environment(&args(values), |_| None).is_err(),
                "accepted invalid KV inputs: {values:?}"
            );
        }
        let models = (0..256)
            .map(|index| format!("model{index}"))
            .collect::<Vec<_>>()
            .join(",");
        assert!(
            Options::with_environment(&args(&["--models", &models, "--attempts", "1000"]), |_| {
                None
            })
            .is_err()
        );
    }
}
