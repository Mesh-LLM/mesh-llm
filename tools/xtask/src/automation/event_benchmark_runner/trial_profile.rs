//! Inherit explicit benchmark tuning while private state owns discovery and storage.
use super::{plan::Mode, trial_environment};
use crate::process::Value;
use serde::Serialize;
use std::{collections::BTreeMap, ffi::OsString};

const DEVICES_AND_THREADS: &[&str] = &[
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
];
const OWNED: &[&str] = &[
    "MESH_LLM_CONFIG",
    "MESH_LLM_RUNTIME_ROOT",
    "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
    "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
    trial_environment::GATE,
    trial_environment::SELECTOR,
];
const PRIVATE_NAMES: &[&str] = &[
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
    "URL",
    "PATH",
];

#[derive(Default, Serialize)]
pub(super) struct Inheritance {
    /// Presence only, including values overwritten by owned private state.
    pub dropped_inherited_settings: Vec<String>,
    /// The legacy manifest only snapshots MESH_LLM settings. These additional
    /// public values bind inherited device/thread/backend settings explicitly.
    pub device_and_backend_settings: BTreeMap<String, String>,
}

fn candidate(name: &str) -> bool {
    name.starts_with("MESH_LLM_")
        || name.starts_with("GGML_")
        || DEVICES_AND_THREADS.contains(&name)
}
fn sensitive(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    PRIVATE_NAMES.iter().any(|part| upper.contains(part))
}

pub(super) fn apply(
    mut isolated: BTreeMap<OsString, Value>,
    inherited: &BTreeMap<OsString, OsString>,
    mode: Mode,
) -> (BTreeMap<OsString, Value>, Inheritance) {
    let mut evidence = Inheritance::default();
    for (name, value) in inherited {
        let Some(label) = name.to_str() else {
            continue;
        };
        if !candidate(label) {
            continue;
        }
        if sensitive(label)
            || OWNED.contains(&label)
            || value.to_str().is_none()
            || value.len() > 16 * 1024
        {
            evidence.dropped_inherited_settings.push(label.into());
            continue;
        }
        isolated.insert(name.clone(), Value::Public(value.clone()));
        if !label.starts_with("MESH_LLM_") {
            evidence
                .device_and_backend_settings
                .insert(label.into(), value.to_string_lossy().into_owned());
        }
    }
    isolated.insert(trial_environment::GATE.into(), Value::Public("1".into()));
    isolated.insert(
        trial_environment::SELECTOR.into(),
        Value::Public(mode.label().into()),
    );
    (isolated, evidence)
}

#[cfg(test)]
#[path = "trial_profile_tests.rs"]
mod tests;
