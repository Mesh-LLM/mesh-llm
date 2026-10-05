use serde_json::Value;

use crate::structured::StructuredOutputSpec;

pub const MESH_GUARDRAILS_FIELD: &str = "mesh_guardrails";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailRequestContract {
    pub tools: RawToolSpec,
    pub tool_choice: RawToolChoice,
    pub parallel_tool_calls: ParallelToolCalls,
    pub response_format: RawResponseFormat,
    pub mesh_guardrails: MeshGuardrailsOverride,
}

impl GuardrailRequestContract {
    pub fn from_parts(
        tools: Option<&Value>,
        tool_choice: Option<&Value>,
        parallel_tool_calls: Option<bool>,
        response_format: Option<&Value>,
        mesh_guardrails: Option<&Value>,
    ) -> Self {
        Self {
            tools: RawToolSpec::from_value(tools),
            tool_choice: RawToolChoice::from_value(tool_choice),
            parallel_tool_calls: ParallelToolCalls::from_option(parallel_tool_calls),
            response_format: RawResponseFormat::from_value(response_format),
            mesh_guardrails: MeshGuardrailsOverride::from_value(mesh_guardrails),
        }
    }

    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        let names: &[RawToolDefinition] = match &self.tools {
            RawToolSpec::Entries(entries) => entries.as_slice(),
            RawToolSpec::Absent | RawToolSpec::InvalidType => &[],
        };
        names.iter().filter_map(|tool| tool.name.as_deref())
    }

    pub fn forced_tool_name(&self) -> Option<&str> {
        match &self.tool_choice {
            RawToolChoice::ForcedName(name) => Some(name.as_str()),
            RawToolChoice::Absent
            | RawToolChoice::Auto
            | RawToolChoice::None
            | RawToolChoice::Required
            | RawToolChoice::OtherString(_)
            | RawToolChoice::InvalidType => None,
        }
    }

    pub fn has_real_tools(&self) -> bool {
        matches!(&self.tools, RawToolSpec::Entries(entries) if !entries.is_empty())
    }

    pub fn requests_structured_output(&self) -> bool {
        matches!(self.response_format, RawResponseFormat::Structured(_))
    }

    pub fn structured_output_spec(&self) -> Option<&StructuredOutputSpec> {
        match &self.response_format {
            RawResponseFormat::Structured(StructuredResponseFormat::Supported(spec)) => Some(spec),
            RawResponseFormat::Absent
            | RawResponseFormat::Text
            | RawResponseFormat::Structured(StructuredResponseFormat::Unsupported { .. })
            | RawResponseFormat::InvalidType => None,
        }
    }

    pub fn has_supported_structured_output(&self) -> bool {
        self.structured_output_spec().is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawToolSpec {
    Absent,
    InvalidType,
    Entries(Vec<RawToolDefinition>),
}

impl RawToolSpec {
    fn from_value(value: Option<&Value>) -> Self {
        match value {
            None => Self::Absent,
            Some(Value::Array(entries)) => {
                Self::Entries(entries.iter().map(RawToolDefinition::from_value).collect())
            }
            Some(_) => Self::InvalidType,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawToolDefinition {
    pub name: Option<String>,
}

impl RawToolDefinition {
    fn from_value(value: &Value) -> Self {
        let name = value
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .map(ToString::to_string);
        Self { name }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawToolChoice {
    Absent,
    Auto,
    None,
    Required,
    ForcedName(String),
    OtherString(String),
    InvalidType,
}

impl RawToolChoice {
    fn from_value(value: Option<&Value>) -> Self {
        match value {
            None => Self::Absent,
            Some(Value::String(choice)) => match choice.as_str() {
                "auto" => Self::Auto,
                "none" => Self::None,
                "required" => Self::Required,
                other => Self::OtherString(other.to_string()),
            },
            Some(Value::Object(object)) => object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(|name| Self::ForcedName(name.to_string()))
                .unwrap_or(Self::InvalidType),
            Some(_) => Self::InvalidType,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelToolCalls {
    Absent,
    Enabled,
    Disabled,
}

impl ParallelToolCalls {
    fn from_option(value: Option<bool>) -> Self {
        match value {
            None => Self::Absent,
            Some(true) => Self::Enabled,
            Some(false) => Self::Disabled,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawResponseFormat {
    Absent,
    Text,
    Structured(StructuredResponseFormat),
    InvalidType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredResponseFormat {
    Supported(StructuredOutputSpec),
    Unsupported { format_type: String },
}

impl RawResponseFormat {
    fn from_value(value: Option<&Value>) -> Self {
        match value {
            None => Self::Absent,
            Some(Value::Object(object)) => match object.get("type").and_then(Value::as_str) {
                Some("text") => Self::Text,
                Some(format_type) => StructuredOutputSpec::from_response_format_object(object)
                    .map(StructuredResponseFormat::Supported)
                    .unwrap_or_else(|_| StructuredResponseFormat::Unsupported {
                        format_type: format_type.to_string(),
                    })
                    .into(),
                None => Self::InvalidType,
            },
            Some(_) => Self::InvalidType,
        }
    }
}

impl From<StructuredResponseFormat> for RawResponseFormat {
    fn from(value: StructuredResponseFormat) -> Self {
        Self::Structured(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshGuardrailsOverride {
    Unset,
    Enabled,
    Disabled,
    InvalidType,
}

impl MeshGuardrailsOverride {
    fn from_value(value: Option<&Value>) -> Self {
        match value {
            None => Self::Unset,
            Some(Value::Bool(true)) => Self::Enabled,
            Some(Value::Bool(false)) => Self::Disabled,
            Some(_) => Self::InvalidType,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn guardrail_request_contract_parses_tools_tool_choice_response_format_and_override() {
        let tools = json!([
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function"}
        ]);
        let tool_choice = json!({"type": "function", "function": {"name": "read_file"}});
        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        });
        let contract = GuardrailRequestContract::from_parts(
            Some(&tools),
            Some(&tool_choice),
            Some(false),
            Some(&response_format),
            Some(&json!(true)),
        );

        assert_eq!(contract.tool_names().collect::<Vec<_>>(), vec!["read_file"]);
        assert_eq!(contract.forced_tool_name(), Some("read_file"));
        assert!(contract.has_real_tools());
        assert_eq!(contract.parallel_tool_calls, ParallelToolCalls::Disabled);
        assert_eq!(contract.mesh_guardrails, MeshGuardrailsOverride::Enabled);
        assert!(contract.requests_structured_output());
        assert!(contract.has_supported_structured_output());
    }
}
