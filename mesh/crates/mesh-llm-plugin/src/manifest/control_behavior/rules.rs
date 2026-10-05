use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::proto;

use super::conditions::PackagedPluginControlCondition;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct PackagedPluginControlAvailability {
    pub(crate) enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) note: Option<String>,
    pub(crate) source: PackagedPluginControlAvailabilitySource,
}

impl TryFrom<&proto::PluginConfigControlAvailability> for PackagedPluginControlAvailability {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigControlAvailability) -> Result<Self> {
        Ok(Self {
            enabled: value.enabled,
            reason: value.reason.clone(),
            note: value.note.clone(),
            source: PackagedPluginControlAvailabilitySource::try_from_i32(value.source)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PackagedPluginControlAvailabilitySource {
    Static,
    Runtime,
    Dependency,
    Conflict,
}

impl PackagedPluginControlAvailabilitySource {
    fn try_from_i32(value: i32) -> Result<Self> {
        match proto::PluginConfigControlAvailabilitySource::try_from(value)
            .map_err(|_| anyhow!("unknown plugin control availability source `{value}`"))?
        {
            proto::PluginConfigControlAvailabilitySource::Static => Ok(Self::Static),
            proto::PluginConfigControlAvailabilitySource::Runtime => Ok(Self::Runtime),
            proto::PluginConfigControlAvailabilitySource::Dependency => Ok(Self::Dependency),
            proto::PluginConfigControlAvailabilitySource::Conflict => Ok(Self::Conflict),
            proto::PluginConfigControlAvailabilitySource::Unspecified => {
                Err(anyhow!("plugin control availability source is unspecified"))
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct PackagedPluginConditionalDisable {
    condition: PackagedPluginControlCondition,
    reason: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    note: Option<String>,
    write_policy: PackagedPluginDisabledWritePolicy,
}

impl TryFrom<&proto::PluginConfigConditionalDisable> for PackagedPluginConditionalDisable {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigConditionalDisable) -> Result<Self> {
        let condition = value
            .condition
            .as_ref()
            .ok_or_else(|| anyhow!("plugin config conditional disable is missing condition"))?;

        Ok(Self {
            condition: PackagedPluginControlCondition::try_from(condition)?,
            reason: value.reason.clone(),
            note: value.note.clone(),
            write_policy: PackagedPluginDisabledWritePolicy::try_from_i32(value.write_policy)?,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct PackagedPluginConflictRule {
    group: String,
    condition: PackagedPluginControlCondition,
    reason: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    preferred_key: Option<String>,
}

impl TryFrom<&proto::PluginConfigConflictRule> for PackagedPluginConflictRule {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigConflictRule) -> Result<Self> {
        let condition = value
            .condition
            .as_ref()
            .ok_or_else(|| anyhow!("plugin config conflict rule is missing condition"))?;

        Ok(Self {
            group: value.group.clone(),
            condition: PackagedPluginControlCondition::try_from(condition)?,
            reason: value.reason.clone(),
            preferred_key: value.preferred_key.clone(),
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PackagedPluginDisabledWritePolicy {
    PreserveExisting,
    OmitWhenDisabled,
    RejectWhenDisabled,
}

impl PackagedPluginDisabledWritePolicy {
    pub(crate) fn try_from_i32(value: i32) -> Result<Self> {
        match proto::PluginConfigDisabledWritePolicy::try_from(value)
            .map_err(|_| anyhow!("unknown plugin config disabled write policy `{value}`"))?
        {
            proto::PluginConfigDisabledWritePolicy::PreserveExisting => Ok(Self::PreserveExisting),
            proto::PluginConfigDisabledWritePolicy::OmitWhenDisabled => Ok(Self::OmitWhenDisabled),
            proto::PluginConfigDisabledWritePolicy::RejectWhenDisabled => {
                Ok(Self::RejectWhenDisabled)
            }
            proto::PluginConfigDisabledWritePolicy::Unspecified => Err(anyhow!(
                "plugin config disabled write policy is unspecified"
            )),
        }
    }
}
