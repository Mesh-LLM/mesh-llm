//! Body of `POST /api/runtime/models`: the model reference, an optional
//! `#profile` suffix and optional per-load `ctx_size` and `parallel`.
use crate::api::RuntimeLoadSizing;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RuntimeLoadRequest {
    pub(super) spec: String,
    pub(super) profile: String,
    pub(super) sizing: RuntimeLoadSizing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RuntimeLoadRequestParseError {
    InvalidJson,
    MissingModel,
    InvalidCtxSize,
    InvalidParallel,
}

impl RuntimeLoadRequestParseError {
    pub(super) fn message(self) -> &'static str {
        match self {
            Self::InvalidJson => "Invalid JSON body",
            Self::MissingModel => "Missing 'model' field",
            Self::InvalidCtxSize => "'ctx_size' must be a positive integer number of tokens",
            Self::InvalidParallel => "'parallel' must be a positive integer number of lanes",
        }
    }
}

pub(super) fn parse_runtime_load_request(
    body: &str,
) -> Result<RuntimeLoadRequest, RuntimeLoadRequestParseError> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|_| RuntimeLoadRequestParseError::InvalidJson)?;
    let model = value
        .get("model")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or(RuntimeLoadRequestParseError::MissingModel)?;
    let (model_ref, profile) = parse_model_with_profile(model);
    let ctx_size = positive_field(&value, "ctx_size")
        .map_err(|()| RuntimeLoadRequestParseError::InvalidCtxSize)?
        .map(u32::try_from)
        .transpose()
        .map_err(|_| RuntimeLoadRequestParseError::InvalidCtxSize)?;
    let parallel = positive_field(&value, "parallel")
        .map_err(|()| RuntimeLoadRequestParseError::InvalidParallel)?
        .map(usize::try_from)
        .transpose()
        .map_err(|_| RuntimeLoadRequestParseError::InvalidParallel)?;
    Ok(RuntimeLoadRequest {
        spec: model_ref.to_string(),
        profile: profile.to_string(),
        sizing: RuntimeLoadSizing { ctx_size, parallel },
    })
}

fn positive_field(value: &serde_json::Value, key: &str) -> Result<Option<u64>, ()> {
    match value.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(field) => field.as_u64().filter(|n| *n > 0).map(Some).ok_or(()),
    }
}

pub(super) fn parse_model_with_profile(model: &str) -> (&str, &str) {
    match model.rfind('#') {
        Some(hash_pos) => (&model[..hash_pos], &model[hash_pos + 1..]),
        None => (model, ""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(body: &str) -> RuntimeLoadRequest {
        parse_runtime_load_request(body).expect("valid load body")
    }

    #[test]
    fn model_and_profile_without_sizing_keep_the_plan() {
        assert_eq!(
            parsed(r#"{"model": "Qwen3-8B#low-ctx"}"#),
            RuntimeLoadRequest {
                spec: "Qwen3-8B".to_string(),
                profile: "low-ctx".to_string(),
                sizing: RuntimeLoadSizing::default(),
            }
        );
        assert_eq!(parsed(r#"{"model": "Qwen3-8B"}"#).profile, "");
        assert_eq!(
            parsed(r#"{"model": "Qwen3-8B", "ctx_size": null}"#).sizing,
            RuntimeLoadSizing::default()
        );
    }

    #[test]
    fn requested_context_and_lanes_are_carried() {
        assert_eq!(
            parsed(r#"{"model": "m", "ctx_size": 8192, "parallel": 1}"#).sizing,
            RuntimeLoadSizing {
                ctx_size: Some(8192),
                parallel: Some(1),
            }
        );
    }

    #[test]
    fn unusable_sizing_is_rejected() {
        for (body, expected) in [
            (
                r#"{"model": "m", "ctx_size": 0}"#,
                RuntimeLoadRequestParseError::InvalidCtxSize,
            ),
            (
                r#"{"model": "m", "ctx_size": 4294967296}"#,
                RuntimeLoadRequestParseError::InvalidCtxSize,
            ),
            (
                r#"{"model": "m", "ctx_size": "8192"}"#,
                RuntimeLoadRequestParseError::InvalidCtxSize,
            ),
            (
                r#"{"model": "m", "parallel": -1}"#,
                RuntimeLoadRequestParseError::InvalidParallel,
            ),
            (
                r#"{"model": "m", "parallel": 1.5}"#,
                RuntimeLoadRequestParseError::InvalidParallel,
            ),
        ] {
            assert_eq!(parse_runtime_load_request(body), Err(expected), "{body}");
        }
    }

    #[test]
    fn missing_model_and_invalid_json_are_rejected() {
        assert_eq!(
            parse_runtime_load_request(r#"{"foo": "bar"}"#),
            Err(RuntimeLoadRequestParseError::MissingModel)
        );
        assert_eq!(
            parse_runtime_load_request("invalid"),
            Err(RuntimeLoadRequestParseError::InvalidJson)
        );
    }
}
