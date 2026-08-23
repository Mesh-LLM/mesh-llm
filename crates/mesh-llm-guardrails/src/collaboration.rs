use serde::Deserialize;
use serde_json::{Map, Value};

pub const MESH_COLLABORATION_FIELD: &str = "mesh_collaboration";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum CollaborationContract {
    ReportRequired(ReportRequired),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportRequired {
    pub version: u8,
    pub tool: String,
    pub body_argument: String,
    pub locked_arguments: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollaborationContractError {
    Invalid(String),
    UnknownTool(String),
    BodyArgumentLocked,
    ArgumentsDoNotMatchToolSchema(String),
}

impl std::fmt::Display for CollaborationContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => write!(
                f,
                "mesh_collaboration must be a valid report_required contract: {reason}"
            ),
            Self::UnknownTool(tool) => write!(
                f,
                "mesh_collaboration tool `{tool}` is not declared in tools"
            ),
            Self::BodyArgumentLocked => {
                f.write_str("mesh_collaboration body_argument must not also be locked")
            }
            Self::ArgumentsDoNotMatchToolSchema(reason) => write!(
                f,
                "mesh_collaboration arguments do not match the declared tool schema: {reason}"
            ),
        }
    }
}

impl std::error::Error for CollaborationContractError {}

impl CollaborationContract {
    pub fn parse(value: &Value, tools: Option<&Value>) -> Result<Self, CollaborationContractError> {
        let contract: Self = serde_json::from_value(value.clone())
            .map_err(|error| CollaborationContractError::Invalid(error.to_string()))?;
        let report = contract.report_required();
        if report.version != 1 {
            return Err(CollaborationContractError::Invalid(
                "version must be 1".into(),
            ));
        }
        if report.tool.trim().is_empty() || report.body_argument.trim().is_empty() {
            return Err(CollaborationContractError::Invalid(
                "tool and body_argument must be non-empty".into(),
            ));
        }
        if report.locked_arguments.contains_key(&report.body_argument) {
            return Err(CollaborationContractError::BodyArgumentLocked);
        }
        if !tool_is_declared(tools, &report.tool) {
            return Err(CollaborationContractError::UnknownTool(report.tool.clone()));
        }
        let tool = declared_tools(tools, &report.tool)
            .next()
            .expect("unique tool checked");
        if tool
            .pointer(&format!(
                "/function/parameters/properties/{}/type",
                report.body_argument
            ))
            .and_then(Value::as_str)
            != Some("string")
        {
            return Err(CollaborationContractError::ArgumentsDoNotMatchToolSchema(
                "body_argument must name a top-level string property".into(),
            ));
        }
        let mut candidate = report.locked_arguments.clone();
        candidate.insert(report.body_argument.clone(), Value::String("report".into()));
        let sanitized = crate::tools::sanitize_tool_arguments_for_tool(
            &report.tool,
            &Value::Object(candidate.clone()),
            tools,
        )
        .map_err(|error| {
            CollaborationContractError::ArgumentsDoNotMatchToolSchema(error.to_string())
        })?;
        if sanitized != Value::Object(candidate) {
            return Err(CollaborationContractError::ArgumentsDoNotMatchToolSchema(
                "an argument was rejected by the schema".into(),
            ));
        }
        Ok(contract)
    }

    pub fn report_required(&self) -> &ReportRequired {
        match self {
            Self::ReportRequired(report) => report,
        }
    }

    pub fn final_arguments(
        &self,
        content: Option<&str>,
        tool_calls: Option<&Value>,
    ) -> Result<Map<String, Value>, CollaborationOutputError> {
        let report = self.report_required();
        let native_arguments = match tool_calls {
            Some(calls) => arguments_from_single_call(calls, &report.tool)?,
            None => Map::new(),
        };
        let body = native_arguments
            .get(&report.body_argument)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                content
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(str::to_owned)
            })
            .ok_or(CollaborationOutputError::MissingReportBody)?;
        let mut arguments = report.locked_arguments.clone();
        arguments.insert(report.body_argument.clone(), Value::String(body));
        Ok(arguments)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollaborationOutputError {
    InvalidChoiceCount,
    InvalidToolCallCount,
    WrongTool { expected: String, actual: String },
    InvalidArguments,
    MissingReportBody,
}

impl std::fmt::Display for CollaborationOutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidChoiceCount => {
                f.write_str("report_required output must contain exactly one choice")
            }
            Self::InvalidToolCallCount => {
                f.write_str("report_required output must contain exactly one tool call")
            }
            Self::WrongTool { expected, actual } => write!(
                f,
                "report_required output called `{actual}` instead of `{expected}`"
            ),
            Self::InvalidArguments => {
                f.write_str("report_required tool arguments must be a JSON object")
            }
            Self::MissingReportBody => {
                f.write_str("report_required output did not contain a report body")
            }
        }
    }
}

impl std::error::Error for CollaborationOutputError {}

fn tool_is_declared(tools: Option<&Value>, expected: &str) -> bool {
    declared_tools(tools, expected).count() == 1
}

fn declared_tools<'a>(
    tools: Option<&'a Value>,
    expected: &'a str,
) -> impl Iterator<Item = &'a Value> {
    tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(move |tool| {
            tool.pointer("/function/name").and_then(Value::as_str) == Some(expected)
        })
}

pub fn collaboration_calls_output_tool(calls: &Value, contract: &CollaborationContract) -> bool {
    calls.as_array().is_some_and(|calls| {
        calls.iter().any(|call| {
            call.pointer("/function/name").and_then(Value::as_str)
                == Some(contract.report_required().tool.as_str())
        })
    })
}

fn arguments_from_single_call(
    calls: &Value,
    expected_tool: &str,
) -> Result<Map<String, Value>, CollaborationOutputError> {
    let calls = calls
        .as_array()
        .filter(|calls| calls.len() == 1)
        .ok_or(CollaborationOutputError::InvalidToolCallCount)?;
    let call = &calls[0];
    let actual = call
        .pointer("/function/name")
        .and_then(Value::as_str)
        .ok_or(CollaborationOutputError::InvalidArguments)?;
    if actual != expected_tool {
        return Err(CollaborationOutputError::WrongTool {
            expected: expected_tool.to_owned(),
            actual: actual.to_owned(),
        });
    }
    let arguments = call
        .pointer("/function/arguments")
        .ok_or(CollaborationOutputError::InvalidArguments)?;
    match arguments {
        Value::Object(arguments) => Ok(arguments.clone()),
        Value::String(arguments) => serde_json::from_str::<Map<String, Value>>(arguments)
            .map_err(|_| CollaborationOutputError::InvalidArguments),
        _ => Err(CollaborationOutputError::InvalidArguments),
    }
}

pub fn finalize_openai_response_value(
    response: &mut Value,
    contract: &CollaborationContract,
) -> Result<bool, CollaborationOutputError> {
    let choices = response
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .filter(|choices| choices.len() == 1)
        .ok_or(CollaborationOutputError::InvalidChoiceCount)?;
    let choice = choices
        .first_mut()
        .and_then(Value::as_object_mut)
        .ok_or(CollaborationOutputError::InvalidChoiceCount)?;
    let message = choice
        .get_mut("message")
        .and_then(Value::as_object_mut)
        .ok_or(CollaborationOutputError::MissingReportBody)?;
    let tool_calls = message.get("tool_calls").cloned();
    if let Some(calls) = tool_calls.as_ref()
        && !collaboration_calls_output_tool(calls, contract)
    {
        return Ok(false);
    }
    let content = message.get("content").and_then(Value::as_str);
    let arguments = contract.final_arguments(content, tool_calls.as_ref())?;
    message.insert("content".into(), Value::Null);
    message.insert(
        "tool_calls".into(),
        serde_json::json!([{
            "id":"call_mesh_collaboration",
            "type":"function",
            "function":{
                "name":contract.report_required().tool,
                "arguments":Value::Object(arguments).to_string()
            }
        }]),
    );
    choice.insert("finish_reason".into(), Value::String("tool_calls".into()));
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Value {
        json!([{"type":"function","function":{"name":"buzz_send","parameters":{"type":"object","properties":{"content":{"type":"string"},"channel":{"type":"string"},"reply_to":{"type":"string"}},"required":["content"]}}}])
    }

    #[test]
    fn prose_becomes_report_arguments_with_locked_values() {
        let contract = CollaborationContract::parse(
            &json!({
                "mode":"report_required", "version":1, "tool":"buzz_send", "body_argument":"content",
                "locked_arguments":{"channel":"locked", "reply_to":"root"}
            }),
            Some(&tools()),
        )
        .expect("valid contract");
        let arguments = contract
            .final_arguments(Some("Done."), None)
            .expect("report");
        assert_eq!(
            arguments,
            json!({"content":"Done.","channel":"locked","reply_to":"root"})
                .as_object()
                .unwrap()
                .clone()
        );
    }

    #[test]
    fn native_call_body_survives_but_locked_values_win() {
        let contract = CollaborationContract::parse(
            &json!({
                "mode":"report_required", "version":1, "tool":"buzz_send", "body_argument":"content",
                "locked_arguments":{"channel":"locked"}
            }),
            Some(&tools()),
        )
        .expect("valid contract");
        let calls = json!([{"function":{"name":"buzz_send","arguments":"{\"content\":\"Done.\",\"channel\":\"wrong\"}"}}]);
        let arguments = contract
            .final_arguments(None, Some(&calls))
            .expect("report");
        assert_eq!(arguments["content"], "Done.");
        assert_eq!(arguments["channel"], "locked");
    }

    #[test]
    fn contract_rejects_undeclared_tool() {
        let error = CollaborationContract::parse(
            &json!({
                "mode":"report_required", "version":1, "tool":"missing", "body_argument":"content",
                "locked_arguments":{}
            }),
            Some(&tools()),
        )
        .expect_err("unknown tool");
        assert!(matches!(error, CollaborationContractError::UnknownTool(_)));
    }
}
