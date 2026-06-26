use anyhow::Result;
use serde::Serialize;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use crate::models::ModelCapabilities;

pub type ServingFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DevicePolicy {
    #[default]
    Auto,
    Cpu,
    Gpu {
        device_ids: Vec<String>,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct LoadModelRequest {
    pub model_ref: String,
    pub device_policy: DevicePolicy,
    #[serde(default)]
    #[serde(skip_serializing_if = "String::is_empty")]
    pub profile: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnloadModelRequest {
    pub target: UnloadTarget,
    pub options: UnloadOptions,
}

impl Default for UnloadModelRequest {
    fn default() -> Self {
        Self {
            target: UnloadTarget::Model(String::new()),
            options: UnloadOptions::default(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnloadTarget {
    Model(String),
    Instance(String),
}

impl UnloadTarget {
    pub fn as_runtime_target(&self) -> &str {
        match self {
            Self::Model(value) | Self::Instance(value) => value,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnloadOptions {
    pub drain_timeout: Duration,
    pub force: bool,
}

impl Default for UnloadOptions {
    fn default() -> Self {
        Self {
            drain_timeout: Duration::from_secs(30),
            force: false,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingModelState {
    Loading,
    #[default]
    Ready,
    Failed,
    Unloading,
    Stopped,
    Unknown(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ServedModel {
    pub model_ref: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "String::is_empty")]
    pub profile: String,
    pub model_id: String,
    pub instance_id: Option<String>,
    pub state: ServingModelState,
    pub backend: Option<String>,
    pub capabilities: ModelCapabilities,
    pub context_length: Option<u32>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ServingStatus {
    pub enabled: bool,
    pub models: Vec<ServedModel>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServingError {
    ModelNotFound {
        model_ref: String,
    },
    DownloadRequired {
        model_ref: String,
    },
    LoadFailed {
        model_ref: String,
        message: String,
    },
    UnloadFailed {
        target: UnloadTarget,
        message: String,
    },
    UnsupportedDevicePolicy {
        policy: DevicePolicy,
    },
    RuntimeUnavailable {
        message: String,
    },
}

impl std::fmt::Display for ServingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotFound { model_ref } => write!(formatter, "model not found: {model_ref}"),
            Self::DownloadRequired { model_ref } => {
                write!(
                    formatter,
                    "model must be downloaded before serving: {model_ref}"
                )
            }
            Self::LoadFailed { model_ref, message } => {
                write!(formatter, "failed to load {model_ref}: {message}")
            }
            Self::UnloadFailed { target, message } => {
                write!(formatter, "failed to unload {target}: {message}")
            }
            Self::UnsupportedDevicePolicy { policy } => {
                write!(formatter, "unsupported device policy: {policy:?}")
            }
            Self::RuntimeUnavailable { message } => {
                write!(formatter, "runtime unavailable: {message}")
            }
        }
    }
}

impl std::fmt::Display for UnloadTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(value) => write!(formatter, "model {value}"),
            Self::Instance(value) => write!(formatter, "instance {value}"),
        }
    }
}

impl std::error::Error for ServingError {}

pub trait ServingController: Send + Sync {
    fn load<'a>(&'a self, request: LoadModelRequest) -> ServingFuture<'a, ServedModel>;

    fn unload<'a>(&'a self, request: UnloadModelRequest) -> ServingFuture<'a, ()>;

    fn served_models<'a>(&'a self) -> ServingFuture<'a, Vec<ServedModel>>;

    fn status<'a>(&'a self) -> ServingFuture<'a, ServingStatus>;

    fn set_device_policy<'a>(&'a self, policy: DevicePolicy) -> ServingFuture<'a, ()>;
}
