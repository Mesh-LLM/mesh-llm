use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::proto;

use super::conditions::PackagedPluginControlCondition;
use super::rules::{
    PackagedPluginConditionalDisable, PackagedPluginConflictRule,
    PackagedPluginControlAvailability, PackagedPluginDisabledWritePolicy,
};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PackagedPluginControlBehavior {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) numeric: Option<PackagedPluginNumericControl>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) text_format: Option<PackagedPluginTextFormat>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) options_source: Option<PackagedPluginOptionsSource>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) availability: Option<PackagedPluginControlAvailability>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) enable_when: Vec<PackagedPluginControlCondition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) disable_when: Vec<PackagedPluginConditionalDisable>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) conflicts: Vec<PackagedPluginConflictRule>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub(crate) write_policy: Option<PackagedPluginDisabledWritePolicy>,
}

impl TryFrom<&proto::PluginConfigControlBehavior> for PackagedPluginControlBehavior {
    type Error = anyhow::Error;

    fn try_from(value: &proto::PluginConfigControlBehavior) -> Result<Self> {
        let enable_when = value
            .enable_when
            .iter()
            .map(PackagedPluginControlCondition::try_from)
            .collect::<Result<Vec<_>>>()?;
        let disable_when = value
            .disable_when
            .iter()
            .enumerate()
            .map(|(index, disable)| {
                PackagedPluginConditionalDisable::try_from(disable)
                    .with_context(|| format!("invalid conditional disable #{}", index + 1))
            })
            .collect::<Result<Vec<_>>>()?;
        let conflicts = value
            .conflicts
            .iter()
            .enumerate()
            .map(|(index, conflict)| {
                PackagedPluginConflictRule::try_from(conflict)
                    .with_context(|| format!("invalid conflict rule #{}", index + 1))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            numeric: value
                .numeric
                .as_ref()
                .map(PackagedPluginNumericControl::from),
            text_format: value
                .text_format
                .map(PackagedPluginTextFormat::try_from_i32)
                .transpose()?,
            options_source: value
                .options_source
                .map(PackagedPluginOptionsSource::try_from_i32)
                .transpose()?,
            availability: value
                .availability
                .as_ref()
                .map(PackagedPluginControlAvailability::try_from)
                .transpose()?,
            enable_when,
            disable_when,
            conflicts,
            write_policy: value
                .write_policy
                .map(PackagedPluginDisabledWritePolicy::try_from_i32)
                .transpose()?,
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct PackagedPluginNumericControl {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    step: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    soft_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    soft_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    unit: Option<String>,
}

impl From<&proto::PluginConfigNumericControl> for PackagedPluginNumericControl {
    fn from(value: &proto::PluginConfigNumericControl) -> Self {
        Self {
            min: value.min,
            max: value.max,
            step: value.step,
            soft_min: value.soft_min,
            soft_max: value.soft_max,
            unit: value.unit.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PackagedPluginTextFormat {
    Plain,
    Path,
    Url,
    SocketAddr,
    Semver,
    Ed25519Key,
    CsvPositiveInts,
}

impl PackagedPluginTextFormat {
    fn try_from_i32(value: i32) -> Result<Self> {
        match proto::PluginConfigTextFormat::try_from(value)
            .map_err(|_| anyhow!("unknown plugin config text format `{value}`"))?
        {
            proto::PluginConfigTextFormat::Plain => Ok(Self::Plain),
            proto::PluginConfigTextFormat::Path => Ok(Self::Path),
            proto::PluginConfigTextFormat::Url => Ok(Self::Url),
            proto::PluginConfigTextFormat::SocketAddr => Ok(Self::SocketAddr),
            proto::PluginConfigTextFormat::Semver => Ok(Self::Semver),
            proto::PluginConfigTextFormat::Ed25519Key => Ok(Self::Ed25519Key),
            proto::PluginConfigTextFormat::CsvPositiveInts => Ok(Self::CsvPositiveInts),
            proto::PluginConfigTextFormat::Unspecified => {
                Err(anyhow!("plugin config text format is unspecified"))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PackagedPluginOptionsSource {
    Static,
    RuntimeGpus,
    RuntimeNativeBackends,
    RuntimeLocalModels,
    RuntimeInstalledPlugins,
    RuntimeMeshPeers,
}

impl PackagedPluginOptionsSource {
    fn try_from_i32(value: i32) -> Result<Self> {
        match proto::PluginConfigOptionsSource::try_from(value)
            .map_err(|_| anyhow!("unknown plugin config options source `{value}`"))?
        {
            proto::PluginConfigOptionsSource::Static => Ok(Self::Static),
            proto::PluginConfigOptionsSource::RuntimeGpus => Ok(Self::RuntimeGpus),
            proto::PluginConfigOptionsSource::RuntimeNativeBackends => {
                Ok(Self::RuntimeNativeBackends)
            }
            proto::PluginConfigOptionsSource::RuntimeLocalModels => Ok(Self::RuntimeLocalModels),
            proto::PluginConfigOptionsSource::RuntimeInstalledPlugins => {
                Ok(Self::RuntimeInstalledPlugins)
            }
            proto::PluginConfigOptionsSource::RuntimeMeshPeers => Ok(Self::RuntimeMeshPeers),
            proto::PluginConfigOptionsSource::Unspecified => {
                Err(anyhow!("plugin config options source is unspecified"))
            }
        }
    }
}
