use crate::repository::check_args::{Grammar, ParsedArgs};
use std::{path::PathBuf, time::Duration};
use url::Url;

pub(crate) const USAGE: &str = "automation stability {nightly|tool-call} [--base-url URL] [--models CSV] [--attempts N] [--timeout SECONDS] [--output PATH | --output-dir DIR] [--agent-smokes opencode,pi,goose] [--agent-timeout SECONDS] [--mesh-binary PATH --release-attestation-public-key-file PATH --release-attestation-expected-status valid|missing|invalid] [--skip-streaming] [--print-plan]\nNightly writes manifest.json, commands.jsonl, results.jsonl, release-attestation.json, summary.json, summary.md and logs. Tool-call writes its results.jsonl. Plans perform no HTTP or evidence writes.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Nightly,
    ToolCall,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Agent {
    Opencode,
    Pi,
    Goose,
}

impl Agent {
    pub fn name(self) -> &'static str {
        match self {
            Self::Opencode => "opencode",
            Self::Pi => "pi",
            Self::Goose => "goose",
        }
    }
}

pub(super) struct Options {
    pub mode: Mode,
    pub base: Url,
    pub models: Vec<String>,
    pub attempts: u32,
    pub timeout: Duration,
    pub output: PathBuf,
    pub agents: Vec<Agent>,
    pub agent_timeout: Duration,
    pub binary: Option<PathBuf>,
    pub public_key: Option<PathBuf>,
    pub expected_attestation: String,
    pub streaming: bool,
    pub plan: bool,
}

const G: Grammar = Grammar {
    usage: USAGE,
    values: &[
        "--base-url",
        "--models",
        "--attempts",
        "--timeout",
        "--output",
        "--output-dir",
        "--agent-smokes",
        "--agent-timeout",
        "--mesh-binary",
        "--release-attestation-public-key-file",
        "--release-attestation-expected-status",
    ],
    flags: &["--skip-streaming", "--print-plan"],
};

impl Options {
    pub fn parse(args: &[String]) -> Result<Self, String> {
        Self::with_environment(args, |name| std::env::var(name).ok())
    }

    fn with_environment(
        args: &[String],
        environment: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, String> {
        let (mode, args) = args.split_first().ok_or("missing stability mode")?;
        let mode = match mode.as_str() {
            "nightly" => Mode::Nightly,
            "tool-call" => Mode::ToolCall,
            _ => return Err("unknown stability mode".into()),
        };
        let parsed = G.parse(args).map_err(|_| "invalid stability arguments")?;
        if !parsed.positionals.is_empty() {
            return Err("stability accepts named inputs only".into());
        }
        validate_mode_flags(mode, &parsed)?;
        let prefix = if mode == Mode::Nightly {
            "MESH_STABILITY"
        } else {
            "MESH_AGENT_TOOL"
        };
        let value = |flag: &str, suffix: &str, default: &str| {
            parsed
                .last(flag)
                .map(str::to_owned)
                .or_else(|| environment(&format!("{prefix}_{suffix}")))
                .unwrap_or_else(|| default.into())
        };
        let base = parsed
            .last("--base-url")
            .map(str::to_owned)
            .unwrap_or_else(|| default_base(mode, &environment));
        let models = csv(&value("--models", "MODELS", "auto,mesh"));
        validate_models(&models)?;
        let attempts = value(
            "--attempts",
            "ATTEMPTS",
            if mode == Mode::Nightly { "3" } else { "1" },
        )
        .parse::<u32>()
        .map_err(|_| "attempts must be an integer")?;
        if !(1..=1000).contains(&attempts) || models.len() * attempts as usize > 10_000 {
            return Err(
                "stability requires 1..1000 attempts and at most 10000 model/attempt cells".into(),
            );
        }
        let output = match mode {
            Mode::Nightly => value(
                "--output-dir",
                "OUTPUT_DIR",
                "target/nightly-stability/latest",
            ),
            Mode::ToolCall => value(
                "--output",
                "OUTPUT",
                "target/agent-tool-call-reliability/results.jsonl",
            ),
        };
        if output.is_empty() {
            return Err("stability output path is empty".into());
        }
        let (agents, binary, key, expected_attestation, agent_timeout) = if mode == Mode::Nightly {
            (
                agents(&value("--agent-smokes", "AGENT_SMOKES", ""))?,
                value("--mesh-binary", "MESH_BINARY", ""),
                parsed
                    .last("--release-attestation-public-key-file")
                    .map(str::to_owned)
                    .or_else(|| environment("MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE")),
                value(
                    "--release-attestation-expected-status",
                    "RELEASE_ATTESTATION_EXPECTED_STATUS",
                    "valid",
                ),
                seconds(&value("--agent-timeout", "AGENT_TIMEOUT", "3600"), 86400.0)?,
            )
        } else {
            (
                Vec::new(),
                String::new(),
                None,
                "valid".into(),
                Duration::from_secs(3600),
            )
        };
        if !matches!(
            expected_attestation.as_str(),
            "valid" | "missing" | "invalid"
        ) {
            return Err("unsupported expected release attestation status".into());
        }
        Ok(Self {
            mode,
            base: normalize_base(&base)?,
            models,
            attempts,
            timeout: seconds(&value("--timeout", "TIMEOUT", "120"), 3600.0)?,
            output: output.into(),
            agents,
            agent_timeout,
            binary: (!binary.is_empty()).then(|| binary.into()),
            public_key: key.filter(|value| !value.is_empty()).map(PathBuf::from),
            expected_attestation,
            streaming: !parsed.flag("--skip-streaming"),
            plan: parsed.flag("--print-plan"),
        })
    }
}

fn validate_mode_flags(mode: Mode, parsed: &ParsedArgs) -> Result<(), String> {
    let denied = if mode == Mode::Nightly {
        &["--output"][..]
    } else {
        &[
            "--output-dir",
            "--agent-smokes",
            "--agent-timeout",
            "--mesh-binary",
            "--release-attestation-public-key-file",
            "--release-attestation-expected-status",
        ][..]
    };
    if denied.iter().any(|flag| parsed.last(flag).is_some()) {
        return Err("option belongs to the other stability mode".into());
    }
    Ok(())
}

fn default_base(mode: Mode, environment: &impl Fn(&str) -> Option<String>) -> String {
    let primary = if mode == Mode::Nightly {
        "MESH_STABILITY_BASE_URL"
    } else {
        "MESH_AGENT_TOOL_BASE_URL"
    };
    for name in [primary, "MESH_AGENT_BASE_URL", "MESH_OPENCODE_BASE_URL"] {
        if let Some(value) = environment(name).filter(|value| !value.is_empty()) {
            return value;
        }
    }
    environment("MESH_CLIENT_API_BASE")
        .filter(|value| !value.is_empty())
        .map_or_else(
            || "http://127.0.0.1:9337/v1".into(),
            |value| format!("{}/v1", value.trim_end_matches('/')),
        )
}

pub(super) fn normalize_base(value: &str) -> Result<Url, String> {
    let mut base = Url::parse(value.trim()).map_err(|_| "invalid stability base URL")?;
    if !matches!(base.scheme(), "http" | "https")
        || base.host_str().is_none()
        || !base.username().is_empty()
        || base.password().is_some()
        || base.query().is_some()
        || base.fragment().is_some()
    {
        return Err(
            "stability requires an HTTP(S) base without credentials, query or fragment".into(),
        );
    }
    let path = base.path().trim_end_matches('/');
    let path = if path.ends_with("/v1") {
        path.to_owned()
    } else {
        format!("{path}/v1")
    };
    base.set_path(&path);
    Ok(base)
}

fn csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn validate_models(models: &[String]) -> Result<(), String> {
    if models.is_empty()
        || models.len() > 256
        || models
            .iter()
            .any(|model| model.len() > 4096 || model.contains(['\0', '\r', '\n']))
    {
        return Err("at least one valid model is required, with at most 256 model ids".into());
    }
    Ok(())
}

fn agents(value: &str) -> Result<Vec<Agent>, String> {
    csv(value)
        .iter()
        .map(|value| match value.as_str() {
            "opencode" => Ok(Agent::Opencode),
            "pi" => Ok(Agent::Pi),
            "goose" => Ok(Agent::Goose),
            _ => Err("unknown agent smoke".into()),
        })
        .collect()
}

fn seconds(value: &str, maximum: f64) -> Result<Duration, String> {
    let seconds = value
        .parse::<f64>()
        .map_err(|_| "timeout must be numeric")?;
    if !seconds.is_finite() || seconds <= 0.0 || seconds > maximum {
        return Err("timeout must be finite and within the documented bound".into());
    }
    let duration = Duration::try_from_secs_f64(seconds).map_err(|_| "timeout out of range")?;
    if duration.is_zero() {
        return Err("timeout rounds to zero".into());
    }
    Ok(duration)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }

    #[test]
    fn environment_defaults_and_explicit_overrides_preserve_mode_semantics() {
        let options =
            Options::with_environment(&args(&["nightly", "--attempts=2"]), |name| match name {
                "MESH_STABILITY_ATTEMPTS" => Some("9".into()),
                "MESH_CLIENT_API_BASE" => Some("https://fixture.invalid/service".into()),
                "MESH_STABILITY_AGENT_SMOKES" => Some("opencode, pi".into()),
                _ => None,
            })
            .unwrap();
        assert_eq!(options.attempts, 2);
        assert_eq!(options.base.as_str(), "https://fixture.invalid/service/v1");
        assert_eq!(options.agents, [Agent::Opencode, Agent::Pi]);
        let tool = Options::with_environment(&args(&["tool-call"]), |_| None).unwrap();
        assert_eq!(tool.attempts, 1);
        assert_eq!(tool.models, ["auto", "mesh"]);
    }

    #[test]
    fn invalid_cohorts_endpoints_and_cross_mode_options_are_rejected_without_io() {
        for values in [
            vec!["nightly", "--attempts", "0"],
            vec!["nightly", "--timeout", "NaN"],
            vec!["tool-call", "--models", ""],
            vec!["tool-call", "--agent-smokes", "pi"],
            vec!["nightly", "--agent-smokes", "unknown"],
            vec!["nightly", "--base-url", "https://user:pass@fixture.invalid"],
            vec!["nightly", "--base-url", "https://fixture.invalid?q=1"],
            vec!["nightly", "--timeout", "1e-30"],
        ] {
            assert!(Options::with_environment(&args(&values), |_| None).is_err());
        }
    }

    #[test]
    fn tool_mode_ignores_nightly_only_environment() {
        let tool = Options::with_environment(&args(&["tool-call"]), |name| match name {
            "MESH_AGENT_TOOL_AGENT_SMOKES" => Some("unknown".into()),
            "MESH_AGENT_TOOL_AGENT_TIMEOUT" => Some("NaN".into()),
            "MESH_AGENT_TOOL_MESH_BINARY" => Some("unrelated-binary".into()),
            "MESH_AGENT_TOOL_RELEASE_ATTESTATION_EXPECTED_STATUS" => Some("unsupported".into()),
            "MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE" => Some("unrelated-key".into()),
            _ => None,
        })
        .unwrap();
        assert!(tool.agents.is_empty());
        assert!(tool.binary.is_none());
        assert!(tool.public_key.is_none());
        assert_eq!(tool.expected_attestation, "valid");
    }
}
