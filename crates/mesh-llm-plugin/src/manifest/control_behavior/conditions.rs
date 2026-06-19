use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::proto;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct PackagedPluginControlCondition {
    key: String,
    operator: PackagedPluginConditionOperator,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    values: Vec<PackagedPluginConditionValue>,
}

impl TryFrom<&proto::PluginConfigControlCondition> for PackagedPluginControlCondition {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigControlCondition) -> Result<Self> {
        let values = value
            .values
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                PackagedPluginConditionValue::try_from(candidate)
                    .with_context(|| format!("invalid condition value #{}", index + 1))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            key: value.key.clone(),
            operator: PackagedPluginConditionOperator::try_from_i32(value.operator)?,
            values,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PackagedPluginConditionOperator {
    Equals,
    NotEquals,
    In,
    NotIn,
    Present,
    Absent,
    Truthy,
    Falsy,
    Range,
}

impl PackagedPluginConditionOperator {
    fn try_from_i32(value: i32) -> Result<Self> {
        let operator = match proto::PluginConfigConditionOperator::try_from(value)
            .map_err(|_| anyhow!("unknown plugin config condition operator `{value}`"))?
        {
            proto::PluginConfigConditionOperator::Equals => Self::Equals,
            proto::PluginConfigConditionOperator::NotEquals => Self::NotEquals,
            proto::PluginConfigConditionOperator::In => Self::In,
            proto::PluginConfigConditionOperator::NotIn => Self::NotIn,
            proto::PluginConfigConditionOperator::Present => Self::Present,
            proto::PluginConfigConditionOperator::Absent => Self::Absent,
            proto::PluginConfigConditionOperator::Truthy => Self::Truthy,
            proto::PluginConfigConditionOperator::Falsy => Self::Falsy,
            proto::PluginConfigConditionOperator::Range => Self::Range,
            proto::PluginConfigConditionOperator::Unspecified => {
                return Err(anyhow!("plugin config condition operator is unspecified"));
            }
        };
        Ok(operator)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum PackagedPluginConditionValue {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

impl TryFrom<&proto::PluginConfigConditionValue> for PackagedPluginConditionValue {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigConditionValue) -> Result<Self> {
        match value
            .value
            .as_ref()
            .ok_or_else(|| anyhow!("plugin config condition value is empty"))?
        {
            proto::plugin_config_condition_value::Value::BoolValue(value) => Ok(Self::Bool(*value)),
            proto::plugin_config_condition_value::Value::IntegerValue(value) => {
                Ok(Self::Integer(*value))
            }
            proto::plugin_config_condition_value::Value::FloatValue(value) => {
                Ok(Self::Float(*value))
            }
            proto::plugin_config_condition_value::Value::StringValue(value) => {
                Ok(Self::String(value.clone()))
            }
        }
    }
}
