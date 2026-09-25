//! The identity half of the native SDK artifact contract: target triple to
//! platform/os/arch, backend to flavor, and the derived artifact id and
//! directory name. Checks run in the legacy order.

use super::{Checked, Rejected};
use crate::ci_plan::document::Json;
use std::path::Path;

/// `(target_triple, platform, os, arch)`.
const TARGETS: [(&str, &str, &str, &str); 8] = [
    ("aarch64-apple-darwin", "darwin-aarch64", "macos", "aarch64"),
    ("x86_64-apple-darwin", "darwin-x86_64", "macos", "x86_64"),
    (
        "x86_64-unknown-linux-gnu",
        "linux-x86_64",
        "linux",
        "x86_64",
    ),
    (
        "aarch64-unknown-linux-gnu",
        "linux-aarch64",
        "linux",
        "aarch64",
    ),
    (
        "aarch64-linux-android",
        "android-arm64-v8a",
        "linux",
        "aarch64",
    ),
    (
        "armv7-linux-androideabi",
        "android-armeabi-v7a",
        "linux",
        "arm",
    ),
    ("x86_64-linux-android", "android-x86_64", "linux", "x86_64"),
    (
        "x86_64-pc-windows-msvc",
        "windows-x86_64",
        "windows",
        "x86_64",
    ),
];

/// Backend to SDK flavor; `hip` is the ROCm alias.
const FLAVORS: [(&str, &str); 7] = [
    ("cpu", "cpu"),
    ("metal", "metal"),
    ("cuda", "cuda"),
    ("cuda-blackwell", "cuda-blackwell"),
    ("rocm", "rocm"),
    ("hip", "rocm"),
    ("vulkan", "vulkan"),
];

/// A manifest whose required string fields have been checked.
pub(super) struct Manifest<'a>(pub(super) &'a Json);

impl Manifest<'_> {
    pub(super) fn text(&self, field: &str) -> &str {
        self.0.get(field).and_then(Json::as_str).unwrap_or_default()
    }
}

pub(super) fn reject<T>(message: String) -> Checked<T> {
    Err(Rejected(message))
}

pub(super) fn check_identity(manifest: &Manifest<'_>, artifact_dir: &str) -> Checked<()> {
    let field = |name| manifest.text(name);
    let expected_id = format!("meshllm-native-{}-{}", field("platform"), field("flavor"));
    let pairs = [
        (
            "artifact_id does not match platform/flavor",
            field("artifact_id"),
            expected_id.as_str(),
        ),
        (
            "native_runtime_id must match artifact_id",
            field("native_runtime_id"),
            field("artifact_id"),
        ),
        (
            "mesh_version must match sdk_version",
            field("mesh_version"),
            field("sdk_version"),
        ),
    ];
    for (message, actual, expected) in pairs {
        if actual != expected {
            return reject(format!("{message}: {actual} != {expected}"));
        }
    }
    let triple = field("target_triple");
    let Some((_, platform, os, arch)) = TARGETS.into_iter().find(|row| row.0 == triple) else {
        return reject(format!("unsupported target_triple: {triple}"));
    };
    for (name, expected) in [("platform", platform), ("os", os), ("arch", arch)] {
        if field(name) != expected {
            return reject(format!(
                "{name} does not match target_triple: {} != {expected}",
                field(name)
            ));
        }
    }
    let backend = field("backend");
    let Some((_, flavor)) = FLAVORS.into_iter().find(|row| row.0 == backend) else {
        return reject(format!("unsupported native SDK backend: {backend}"));
    };
    if field("flavor") != flavor {
        return reject(format!(
            "flavor does not match backend: {} != {flavor}",
            field("flavor")
        ));
    }
    let dir_name = Path::new(artifact_dir)
        .file_name()
        .map(|name| name.to_string_lossy());
    let dir_name = dir_name.as_deref().unwrap_or_default();
    if dir_name != field("artifact_id") {
        return reject(format!(
            "artifact directory name does not match artifact_id: {dir_name} != {}",
            field("artifact_id")
        ));
    }
    Ok(())
}
