use super::integer::PositiveInteger;
use super::parameters::Parameters;
use crate::automation::codepoint_json::{parser, value::Value};
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

pub(super) enum Failure {
    Input(InputFailure),
    Policy(PolicyFailure),
}

pub(super) enum InputFailure {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Decode(String),
    RootShape,
}

pub(super) enum PolicyFailure {
    ReplayBlock,
    Mode(String),
    Positive(&'static str),
    Concurrency,
    Waves,
    Frameworks,
    Context,
    Window,
    Sampling,
    Backend,
}

impl From<PolicyFailure> for Failure {
    fn from(error: PolicyFailure) -> Self {
        Self::Policy(error)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(InputFailure::Read { path, source }) => write!(
                formatter,
                "replay matrix input: read {}: {source}",
                path.display()
            ),
            Self::Input(InputFailure::Decode(message)) => {
                write!(formatter, "replay matrix input: decode: {message}")
            }
            Self::Input(InputFailure::RootShape) => {
                formatter.write_str("replay matrix input: root must be an object")
            }
            Self::Policy(error) => error.fmt(formatter),
        }
    }
}

impl fmt::Display for PolicyFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mode(value) => write!(formatter, "replay mode must be all (got {value})"),
            Self::Positive(field) => write!(formatter, "{field} must be a positive integer"),
            Self::ReplayBlock => formatter.write_str("matrix replay block is missing"),
            Self::Concurrency => formatter
                .write_str("concurrency must be a non-empty list of unique positive integers"),
            Self::Waves => {
                formatter.write_str("session count does not cover the required worker waves")
            }
            Self::Frameworks => {
                formatter.write_str("session count must cover all three frameworks")
            }
            Self::Context => {
                formatter.write_str("nightly requires at least 128K effective context")
            }
            Self::Window => formatter.write_str("invalid selection window"),
            Self::Sampling => {
                formatter.write_str("replay sampling must be pinned to temperature 0 and seed 42")
            }
            Self::Backend => formatter.write_str("unsupported backend or selection algorithm"),
        }
    }
}

pub(super) struct LoadedReplay {
    pub(super) parameters: Parameters,
    pub(super) replay: Value,
}

pub(super) fn load_matrix(path: &Path) -> Result<Value, Failure> {
    let raw = std::fs::read(path).map_err(|source| {
        Failure::Input(InputFailure::Read {
            path: path.to_owned(),
            source,
        })
    })?;
    let matrix =
        parser::parse(&raw).map_err(|message| Failure::Input(InputFailure::Decode(message)))?;
    if !matches!(matrix, Value::Object(_)) {
        return Err(Failure::Input(InputFailure::RootShape));
    }
    Ok(matrix)
}

pub(super) fn load(path: &Path) -> Result<LoadedReplay, Failure> {
    let matrix = load_matrix(path)?;
    let Value::Object(mut entries) = matrix else {
        return Err(Failure::Input(InputFailure::RootShape));
    };
    let Some(index) = entries.iter().position(|(key, _)| key == "replay") else {
        return Err(PolicyFailure::ReplayBlock.into());
    };
    let (_, replay) = entries.swap_remove(index);
    let parameters = validate(&replay)?;
    Ok(LoadedReplay { parameters, replay })
}

pub(super) fn validate(replay: &Value) -> Result<Parameters, Failure> {
    if !matches!(replay, Value::Object(_)) {
        return Err(PolicyFailure::ReplayBlock.into());
    }
    let mode = replay.get("mode").unwrap_or(&Value::Null);
    if !matches!(mode, Value::Str(text) if text == "all") {
        return Err(PolicyFailure::Mode(mode.repr()).into());
    }
    let parameters = Parameters {
        sessions_per_concurrency: positive(replay, "sessions_per_concurrency")?,
        minimum_worker_waves: positive(replay, "minimum_worker_waves")?,
        minimum_context_tokens: positive(replay, "minimum_context_tokens")?,
        minimum_session_prompt_tokens: positive(replay, "minimum_session_prompt_tokens")?,
        min_isl: positive(replay, "min_isl")?,
        max_isl: positive(replay, "max_isl")?,
        min_turns: positive(replay, "min_turns")?,
        passes: positive(replay, "passes")?,
        warmup_turns: positive(replay, "warmup_turns")?,
        max_output_tokens: positive(replay, "max_output_tokens")?,
        concurrency: concurrency(replay.get("concurrency"))?,
    };
    check_relations(&parameters)?;
    if !temperature(replay.get("temperature")) || !seed(replay.get("seed")) {
        return Err(PolicyFailure::Sampling.into());
    }
    if !matches!(replay.get("backend"), Some(Value::Str(text)) if text == "metal")
        || !matches!(replay.get("selection_algorithm"), Some(Value::Str(text)) if text == "balanced-md5-v2")
    {
        return Err(PolicyFailure::Backend.into());
    }
    Ok(parameters)
}

fn positive(replay: &Value, field: &'static str) -> Result<PositiveInteger, PolicyFailure> {
    integer(replay.get(field)).ok_or(PolicyFailure::Positive(field))
}

fn integer(value: Option<&Value>) -> Option<PositiveInteger> {
    match value {
        Some(Value::Int(value)) => PositiveInteger::from_int(*value),
        Some(Value::BigInt(text)) => PositiveInteger::from_decimal(text),
        None
        | Some(
            Value::Null
            | Value::Bool(_)
            | Value::Float(_)
            | Value::Str(_)
            | Value::Array(_)
            | Value::Object(_),
        ) => None,
    }
}

fn concurrency(value: Option<&Value>) -> Result<Vec<PositiveInteger>, PolicyFailure> {
    let Some(Value::Array(values)) = value else {
        return Err(PolicyFailure::Concurrency);
    };
    let levels = values
        .iter()
        .map(|value| integer(Some(value)))
        .collect::<Option<Vec<_>>>()
        .ok_or(PolicyFailure::Concurrency)?;
    if levels.is_empty() || levels.iter().collect::<BTreeSet<_>>().len() != levels.len() {
        return Err(PolicyFailure::Concurrency);
    }
    Ok(levels)
}

fn check_relations(parameters: &Parameters) -> Result<(), PolicyFailure> {
    if parameters.concurrency.iter().any(|level| {
        parameters.sessions_per_concurrency < parameters.minimum_worker_waves.product(level)
    }) {
        return Err(PolicyFailure::Waves);
    }
    if parameters.sessions_per_concurrency.below(3) {
        return Err(PolicyFailure::Frameworks);
    }
    if parameters.minimum_context_tokens.below(131_072) {
        return Err(PolicyFailure::Context);
    }
    if parameters.max_isl <= parameters.min_isl {
        return Err(PolicyFailure::Window);
    }
    Ok(())
}

fn temperature(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Bool(false) | Value::Int(0)) => true,
        Some(Value::Float(number)) => *number == 0.0,
        None
        | Some(
            Value::Null
            | Value::Bool(true)
            | Value::Int(_)
            | Value::BigInt(_)
            | Value::Str(_)
            | Value::Array(_)
            | Value::Object(_),
        ) => false,
    }
}

fn seed(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Int(42)) => true,
        Some(Value::Float(number)) => *number == 42.0,
        None
        | Some(
            Value::Null
            | Value::Bool(_)
            | Value::Int(_)
            | Value::BigInt(_)
            | Value::Str(_)
            | Value::Array(_)
            | Value::Object(_),
        ) => false,
    }
}
