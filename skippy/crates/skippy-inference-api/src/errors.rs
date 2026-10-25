use axum::{
    Json,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct InferenceError {
    status: StatusCode,
    message: String,
    error_type: String,
    param: Option<String>,
    code: Option<String>,
    retry_after_secs: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceErrorKind {
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    RateLimit,
    PayloadTooLarge,
    Timeout,
    Internal,
    ServiceUnavailable,
    ContextLengthExceeded,
    UnsupportedFeature,
    Cancelled,
}

impl InferenceError {
    pub fn from_kind(
        status: StatusCode,
        kind: InferenceErrorKind,
        message: impl Into<String>,
    ) -> Self {
        let (error_type, code) = kind_to_openai_fields(kind);
        Self {
            status,
            message: message.into(),
            error_type: error_type.to_string(),
            param: None,
            code: Some(code.to_string()),
            retry_after_secs: None,
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::BAD_REQUEST,
            InferenceErrorKind::InvalidRequest,
            message,
        )
    }

    pub fn model_not_found(model: impl Into<String>) -> Self {
        let model = model.into();
        Self::from_kind(
            StatusCode::NOT_FOUND,
            InferenceErrorKind::NotFound,
            format!("model not found: {model}"),
        )
    }

    pub fn backend(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::BAD_GATEWAY,
            InferenceErrorKind::ServiceUnavailable,
            message,
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::INTERNAL_SERVER_ERROR,
            InferenceErrorKind::Internal,
            message,
        )
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::BAD_REQUEST,
            InferenceErrorKind::UnsupportedFeature,
            message,
        )
    }

    pub fn route_not_found(path: impl std::fmt::Display) -> Self {
        Self::from_kind(
            StatusCode::NOT_FOUND,
            InferenceErrorKind::InvalidRequest,
            format!("route not found: {path}"),
        )
        .with_code("not_found")
    }

    pub fn method_not_allowed(method: impl std::fmt::Display) -> Self {
        Self::from_kind(
            StatusCode::METHOD_NOT_ALLOWED,
            InferenceErrorKind::InvalidRequest,
            format!("method not allowed: {method}"),
        )
        .with_code("method_not_allowed")
    }

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::PAYLOAD_TOO_LARGE,
            InferenceErrorKind::PayloadTooLarge,
            message,
        )
    }

    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::BAD_REQUEST,
            InferenceErrorKind::ContextLengthExceeded,
            message,
        )
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::from_kind(
            StatusCode::GATEWAY_TIMEOUT,
            InferenceErrorKind::Timeout,
            message,
        )
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::from_kind(
            crate::lifecycle::client_closed_request_status(),
            InferenceErrorKind::Cancelled,
            message,
        )
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// The human-readable error message carried by this error.
    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    pub fn with_retry_after_secs(mut self, retry_after_secs: u64) -> Self {
        self.retry_after_secs = Some(retry_after_secs);
        self
    }

    pub fn body(&self) -> ErrorResponse {
        ErrorResponse {
            error: ErrorBody {
                message: self.message.clone(),
                error_type: self.error_type.clone(),
                param: self.param.clone(),
                code: self.code.clone(),
            },
        }
    }
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for InferenceError {}

fn map_upstream_kind(status_code: u16, upstream_type: &str) -> InferenceErrorKind {
    match (status_code, upstream_type) {
        (400, "invalid_request_error") => InferenceErrorKind::InvalidRequest,
        (401, "authentication_error") => InferenceErrorKind::Authentication,
        (404, "not_found_error") => InferenceErrorKind::NotFound,
        (500, "server_error") => InferenceErrorKind::Internal,
        (403, "permission_error") => InferenceErrorKind::Permission,
        (501, "not_supported_error") => InferenceErrorKind::UnsupportedFeature,
        (503, "unavailable_error") => InferenceErrorKind::ServiceUnavailable,
        (400, "exceed_context_size_error") => InferenceErrorKind::ContextLengthExceeded,
        (400, _) => InferenceErrorKind::InvalidRequest,
        (401, _) => InferenceErrorKind::Authentication,
        (403, _) => InferenceErrorKind::Permission,
        (404, _) => InferenceErrorKind::NotFound,
        (429, _) => InferenceErrorKind::RateLimit,
        (502, _) => InferenceErrorKind::ServiceUnavailable,
        (503, _) => InferenceErrorKind::ServiceUnavailable,
        (504, _) => InferenceErrorKind::Timeout,
        _ => InferenceErrorKind::Internal,
    }
}

fn kind_to_openai_fields(kind: InferenceErrorKind) -> (&'static str, &'static str) {
    match kind {
        InferenceErrorKind::InvalidRequest => ("invalid_request_error", "invalid_value"),
        InferenceErrorKind::Authentication => ("authentication_error", "invalid_api_key"),
        InferenceErrorKind::Permission => ("permission_error", "insufficient_quota"),
        InferenceErrorKind::NotFound => ("invalid_request_error", "model_not_found"),
        InferenceErrorKind::RateLimit => ("rate_limit_error", "rate_limit_exceeded"),
        InferenceErrorKind::PayloadTooLarge => ("invalid_request_error", "payload_too_large"),
        InferenceErrorKind::Timeout => ("server_error", "timeout"),
        InferenceErrorKind::Internal => ("server_error", "internal_server_error"),
        InferenceErrorKind::ServiceUnavailable => ("server_error", "service_unavailable"),
        InferenceErrorKind::ContextLengthExceeded => {
            ("invalid_request_error", "context_length_exceeded")
        }
        InferenceErrorKind::UnsupportedFeature => {
            ("invalid_request_error", "unsupported_model_feature")
        }
        InferenceErrorKind::Cancelled => ("invalid_request_error", "request_cancelled"),
    }
}

fn extract_message(value: &Value) -> Option<String> {
    value
        .get("message")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            value
                .get("error")
                .and_then(Value::as_object)
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .or_else(|| {
            value
                .get("error")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
}

fn extract_upstream_type(value: &Value) -> Option<String> {
    value
        .get("type")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            value
                .get("error")
                .and_then(Value::as_object)
                .and_then(|error| error.get("type"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
}

pub fn already_openai_error(value: &Value) -> bool {
    value
        .get("error")
        .and_then(Value::as_object)
        .map(|error| {
            error.get("message").and_then(Value::as_str).is_some()
                && error.get("type").and_then(Value::as_str).is_some()
        })
        .unwrap_or(false)
}

pub fn map_upstream_error_body(status_code: u16, body: &[u8]) -> Option<Vec<u8>> {
    if status_code < 400 {
        return None;
    }

    let parsed = serde_json::from_slice::<Value>(body).ok();
    if let Some(value) = parsed.as_ref()
        && already_openai_error(value)
    {
        return None;
    }

    let message = parsed
        .as_ref()
        .and_then(extract_message)
        .or_else(|| {
            let text = String::from_utf8_lossy(body).trim().to_string();
            if text.is_empty() { None } else { Some(text) }
        })
        .unwrap_or_else(|| "Unknown error".to_string());

    let upstream_type = parsed
        .as_ref()
        .and_then(extract_upstream_type)
        .unwrap_or_default();
    let kind = map_upstream_kind(status_code, &upstream_type);
    let status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let error = InferenceError::from_kind(status, kind, message);
    Some(serde_json::to_vec(&error.body()).expect("serializing OpenAI error body should not fail"))
}

impl IntoResponse for InferenceError {
    fn into_response(self) -> Response {
        let retry_after_secs = self.retry_after_secs;
        let mut response = (self.status, Json(self.body())).into_response();
        if let Some(retry_after_secs) = retry_after_secs {
            let header_value = HeaderValue::from_str(&retry_after_secs.to_string())
                .expect("integer retry-after value should be a valid header");
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, header_value);
        }
        response
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}
