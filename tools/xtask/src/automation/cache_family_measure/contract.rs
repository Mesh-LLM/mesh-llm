use crate::command::DynResult;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(in crate::automation) enum Cohort {
    NativeSerial,
    NativeConcurrent,
    OpenaiConcurrent,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation) struct Input {
    pub schema_version: u64,
    pub cohort: Cohort,
    pub base_url: String,
    pub prompt: String,
    pub model_id: Option<String>,
    pub requests: usize,
    pub concurrency: usize,
    pub output_tokens: u64,
    pub request_timeout_ms: u64,
    pub execution_timeout_ms: u64,
}
impl Input {
    pub(in crate::automation) fn validate(&self) -> DynResult<()> {
        let uri: hyper::Uri = self.base_url.parse()?;
        let path = if self.cohort == Cohort::OpenaiConcurrent {
            "/v1"
        } else {
            "/"
        };
        if self.schema_version != 1
            || uri.scheme_str() != Some("http")
            || uri.host() != Some("127.0.0.1")
            || uri.port_u16().is_none_or(|port| port == 0)
            || uri.path() != path
            || uri.query().is_some()
            || uri
                .authority()
                .is_none_or(|authority| authority.as_str().contains('@'))
            || self.prompt.is_empty()
            || self.prompt.len() > 64 * 1024
            || !(1..=256).contains(&self.concurrency)
            || !(1..=4096).contains(&self.requests)
            || self.requests < self.concurrency
            || !(1..=600_000).contains(&self.request_timeout_ms)
            || !(1..=3_600_000).contains(&self.execution_timeout_ms)
        {
            return Err("invalid cache measurement URL/workload/budget".into());
        }
        match self.cohort {
            Cohort::NativeSerial
                if self.concurrency != 1 || self.output_tokens != 1 || self.model_id.is_some() =>
            {
                return Err(
                    "native serial requires one lane, >=1 repeat, one token and no model alias"
                        .into(),
                );
            }
            Cohort::NativeConcurrent if self.output_tokens != 128 || self.model_id.is_some() => {
                return Err(
                    "native concurrent requires original 128-token cohort and no model alias"
                        .into(),
                );
            }
            Cohort::OpenaiConcurrent
                if !(1..=4096).contains(&self.output_tokens)
                    || self.model_id.as_ref().is_none_or(|model| {
                        model.is_empty()
                            || model.len() > 4096
                            || model.chars().any(char::is_control)
                    }) =>
            {
                return Err(
                    "OpenAI concurrent requires bounded declared model/output tokens".into(),
                );
            }
            _ => {}
        }
        Ok(())
    }
}
