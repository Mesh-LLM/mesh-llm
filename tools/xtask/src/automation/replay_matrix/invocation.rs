use super::integer::PositiveInteger;
use super::parameters::Parameters;
use crate::automation::codepoint_json::value::Value;
use std::ffi::{OsStr, OsString};
use std::path::Path;

#[path = "invocation_strings.rs"]
mod strings;
use strings::ReplayString;
#[path = "invocation_encoding.rs"]
mod encoding;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum EncodingError {
    NonString { index: usize, kind: &'static str },
    UnencodableCodepoint { index: usize, codepoint: u32 },
    EmbeddedNul { index: usize },
}

#[cfg(unix)]
use encoding::EncodingError as OsEncodingError;
#[cfg(windows)]
type OsEncodingError = EncodingError;

#[cfg(test)]
#[path = "invocation_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "invocation_text_tests.rs"]
mod text_tests;

#[cfg(test)]
#[path = "invocation_argv_restoration_tests.rs"]
mod argv_restoration_tests;
#[cfg(test)]
#[path = "invocation_restoration_tests.rs"]
mod restoration_tests;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum SelectionError {
    FamilyCardinality,
    NativeContext,
    NativeContextType(&'static str),
    MissingField(&'static str),
    ModelsNotIterable(&'static str),
    RowMissingFamily { index: usize },
    RowType { index: usize, kind: &'static str },
}

pub(super) struct SelectedModel<'a> {
    pub(super) reference: ReplayString,
    sha256: &'a Value,
    recurrent: bool,
}

pub(super) enum Argument<'a> {
    Os(OsString),
    Text(ReplayString),
    Decoded(&'a Value),
}

impl From<OsString> for Argument<'_> {
    fn from(value: OsString) -> Self {
        Self::Os(value)
    }
}
impl From<&str> for Argument<'_> {
    fn from(value: &str) -> Self {
        Self::Os(value.into())
    }
}
impl From<String> for Argument<'_> {
    fn from(value: String) -> Self {
        Self::Os(value.into())
    }
}

pub(super) struct ReplayInvocation<'a> {
    pub(super) python: &'a OsStr,
    pub(super) script: &'a Path,
    pub(super) dataset: &'a Path,
    pub(super) output: &'a Path,
    pub(super) worktree_root: Option<&'a OsStr>,
    pub(super) refs: &'a [OsString],
}

impl Parameters {
    fn argv_fields(&self) -> [(&'static str, &PositiveInteger); 10] {
        [
            ("sessions-per-concurrency", &self.sessions_per_concurrency),
            ("minimum-worker-waves", &self.minimum_worker_waves),
            ("minimum-context-tokens", &self.minimum_context_tokens),
            (
                "minimum-session-prompt-tokens",
                &self.minimum_session_prompt_tokens,
            ),
            ("min-isl", &self.min_isl),
            ("max-isl", &self.max_isl),
            ("min-turns", &self.min_turns),
            ("passes", &self.passes),
            ("warmup-turns", &self.warmup_turns),
            ("max-output-tokens", &self.max_output_tokens),
        ]
    }
}

pub(super) fn select<'a>(
    models: Option<&'a Value>,
    parameters: &Parameters,
    family: &crate::automation::codepoint_json::strings::JsonString,
) -> Result<SelectedModel<'a>, SelectionError> {
    let models = match models {
        Some(Value::Array(models)) => models,
        Some(Value::Object(entries)) if entries.is_empty() => {
            return Err(SelectionError::FamilyCardinality);
        }
        Some(Value::Str(text)) if text.codepoints().next().is_none() => {
            return Err(SelectionError::FamilyCardinality);
        }
        Some(Value::Object(_) | Value::Str(_)) => {
            return Err(SelectionError::RowType {
                index: 0,
                kind: "str",
            });
        }
        Some(
            value @ (Value::Null
            | Value::Bool(_)
            | Value::Int(_)
            | Value::BigInt(_)
            | Value::Float(_)),
        ) => {
            return Err(SelectionError::ModelsNotIterable(kind(value)));
        }
        None => return Err(SelectionError::MissingField("models")),
    };
    let mut selected = None;
    let mut multiple = false;
    for (index, model) in models.iter().enumerate() {
        if !matches!(model, Value::Object(_)) {
            return Err(SelectionError::RowType {
                index,
                kind: kind(model),
            });
        }
        let row_family = model
            .get("family")
            .ok_or(SelectionError::RowMissingFamily { index })?;
        if matches!(row_family, Value::Str(value) if value == family) {
            multiple |= selected.replace(model).is_some();
        }
    }
    if multiple {
        return Err(SelectionError::FamilyCardinality);
    }
    let model = selected.ok_or(SelectionError::FamilyCardinality)?;
    if native_context_below(
        model.get("native_context_tokens"),
        &parameters.minimum_context_tokens,
    )? {
        return Err(SelectionError::NativeContext);
    }
    let repo = model_text(required(model, "repo")?);
    let revision = model_text(required(model, "revision")?);
    let file = model_text(required(model, "file")?);
    let reference = repo
        .codepoints()
        .chain([u32::from('@')])
        .chain(revision.codepoints())
        .chain([u32::from('/')])
        .chain(file.codepoints())
        .collect();
    let sha256 = required(model, "sha256")?;
    let recurrent =
        matches!(required(model, "class")?, Value::Str(value) if value == "hybrid-recurrent");
    Ok(SelectedModel {
        reference,
        sha256,
        recurrent,
    })
}

fn native_context_below(
    value: Option<&Value>,
    minimum: &PositiveInteger,
) -> Result<bool, SelectionError> {
    Ok(match value {
        Some(Value::Int(value)) => {
            PositiveInteger::from_int(*value).is_none_or(|value| &value < minimum)
        }
        Some(Value::BigInt(value)) => {
            PositiveInteger::from_decimal(value).is_none_or(|value| &value < minimum)
        }
        Some(Value::Float(value)) if value.is_nan() || *value == f64::INFINITY => false,
        Some(Value::Float(value)) => {
            PositiveInteger::from_decimal(&format!("{:.0}", value.floor()))
                .is_none_or(|value| &value < minimum)
        }
        None | Some(Value::Bool(_)) => true,
        Some(value @ (Value::Null | Value::Str(_) | Value::Array(_) | Value::Object(_))) => {
            return Err(SelectionError::NativeContextType(kind(value)));
        }
    })
}

fn required<'a>(model: &'a Value, name: &'static str) -> Result<&'a Value, SelectionError> {
    model.get(name).ok_or(SelectionError::MissingField(name))
}

fn model_text(value: &Value) -> ReplayString {
    match value {
        Value::Str(text) => text.into(),
        Value::Null
        | Value::Bool(_)
        | Value::Int(_)
        | Value::BigInt(_)
        | Value::Float(_)
        | Value::Array(_)
        | Value::Object(_) => strings::repr(value).as_str().into(),
    }
}

fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) | Value::BigInt(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

pub(super) fn argv<'a>(
    model: &SelectedModel<'a>,
    parameters: &Parameters,
    invocation: &ReplayInvocation<'_>,
) -> Vec<Argument<'a>> {
    let mut args = vec![
        invocation.python.to_owned().into(),
        invocation.script.as_os_str().to_owned().into(),
        "run".into(),
        "--model".into(),
        Argument::Text(model.reference.clone()),
        "--backend".into(),
        "metal".into(),
        "--replay-mode".into(),
        "all".into(),
        "--expected-model-sha256".into(),
        Argument::Decoded(model.sha256),
        "--dataset-file".into(),
        invocation.dataset.as_os_str().to_owned().into(),
        "--output".into(),
        invocation.output.as_os_str().to_owned().into(),
    ];
    if let Some(root) = invocation.worktree_root.filter(|root| !root.is_empty()) {
        args.extend(["--worktree-root".into(), root.to_owned().into()]);
    }
    for reference in invocation.refs {
        args.extend(["--ref".into(), reference.clone().into()]);
    }
    for (name, value) in parameters.argv_fields() {
        args.extend([format!("--{name}").into(), value.to_string().into()]);
    }
    for level in &parameters.concurrency {
        args.extend(["--concurrency".into(), level.to_string().into()]);
    }
    if model.recurrent {
        args.push("--require-recurrent-restores".into());
    }
    args
}

#[cfg(unix)]
pub(super) fn os_argv(arguments: &[Argument<'_>]) -> Result<Vec<OsString>, OsEncodingError> {
    encoding::unix_utf8_argv(arguments)
}

#[cfg(windows)]
pub(super) fn os_argv(arguments: &[Argument<'_>]) -> Result<Vec<OsString>, OsEncodingError> {
    encoding::windows_argv(arguments)
}
