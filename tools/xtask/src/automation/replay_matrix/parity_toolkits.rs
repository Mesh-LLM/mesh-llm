//! Explicit nonstandard toolkit directory observations, not library/build attestation.
use crate::{command::DynResult, process::Value as Argument};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    #[serde(rename = "CUDA_PATH")]
    cuda: Option<PathBuf>,
    #[serde(rename = "HIP_PATH")]
    hip: Option<PathBuf>,
    #[serde(rename = "ROCM_PATH")]
    rocm: Option<PathBuf>,
    #[serde(rename = "LLVMInstallDir")]
    llvm: Option<PathBuf>,
    #[serde(rename = "VULKAN_SDK")]
    vulkan: Option<PathBuf>,
}
impl Input {
    fn entries(&self) -> [(&'static str, Option<&Path>); 5] {
        [
            ("CUDA_PATH", self.cuda.as_deref()),
            ("HIP_PATH", self.hip.as_deref()),
            ("ROCM_PATH", self.rocm.as_deref()),
            ("LLVMInstallDir", self.llvm.as_deref()),
            ("VULKAN_SDK", self.vulkan.as_deref()),
        ]
    }
}
pub(super) struct Admitted {
    pub environment: BTreeMap<OsString, Argument>,
    pub observation: Value,
}
fn metadata(path: &Path) -> DynResult<Value> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() {
        return Err(format!(
            "toolkit observation must be regular directory: {}",
            path.display()
        )
        .into());
    }
    let value = json!({"modified":metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string(),"created":metadata.created().ok().and_then(|t|t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d|d.as_nanos().to_string())});
    #[cfg(unix)]
    let value = {
        let mut value = value;
        use std::os::unix::fs::MetadataExt as _;
        value["device"] = json!(metadata.dev());
        value["inode"] = json!(metadata.ino());
        value
    };
    Ok(value)
}
fn search_directories(name: &str) -> &'static [&'static str] {
    match name {
        "CUDA_PATH" => &["lib/x64"],
        "HIP_PATH" | "ROCM_PATH" => &["lib", "hip/lib", "llvm/lib"],
        "LLVMInstallDir" => &["lib", "llvm/lib"],
        "VULKAN_SDK" => &["Lib"],
        _ => &[],
    }
}
fn directory(
    name: &str,
    requested: &Path,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> DynResult<(PathBuf, Value)> {
    guard()?;
    if !requested.is_absolute()
        || requested
            .to_str()
            .is_none_or(|s| s.is_empty() || s.len() > 4096 || s.contains(['\0', '\n', '\r']))
    {
        return Err(format!("toolkit_dirs.{name} requires an existing absolute directory").into());
    }
    let canonical = requested
        .canonicalize()
        .map_err(|e| format!("toolkit_dirs.{name} directory admission failed: {e}"))?;
    let root = metadata(&canonical)
        .map_err(|e| format!("toolkit_dirs.{name} directory admission failed: {e}"))?;
    let mut searches = serde_json::Map::new();
    for suffix in search_directories(name) {
        guard()?;
        let path = canonical.join(suffix);
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                let actual = path.canonicalize()?;
                searches.insert(
                    (*suffix).into(),
                    json!({"canonical":actual,"directory":metadata(&actual)?}),
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                searches.insert((*suffix).into(), Value::Null);
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok((
        canonical.clone(),
        json!({"requested":requested,"canonical":canonical,"directory":root,"link_search_directories":searches}),
    ))
}
pub(super) fn admit(
    input: &Input,
    guard: &mut impl FnMut() -> std::io::Result<()>,
) -> DynResult<Admitted> {
    let mut environment = BTreeMap::new();
    let mut observations = serde_json::Map::new();
    for (name, requested) in input.entries() {
        guard()?;
        let ambient = std::env::var_os(name);
        let Some(requested) = requested else {
            if ambient.is_some() {
                return Err(format!("ambient {name} requires explicit toolkit_dirs.{name}; arbitrary toolchain paths are not forwarded").into());
            }
            continue;
        };
        let (canonical, observation) = directory(name, requested, guard)?;
        if let Some(ambient) = ambient {
            let ambient = PathBuf::from(ambient);
            if !ambient.is_absolute() || ambient.canonicalize().ok().as_ref() != Some(&canonical) {
                return Err(
                    format!("ambient {name} conflicts with explicit toolkit_dirs.{name}").into(),
                );
            }
        }
        environment.insert(name.into(), Argument::Public(canonical.into_os_string()));
        observations.insert(name.into(), observation);
    }
    Ok(Admitted {
        environment,
        observation: json!({"scope":"canonical_toolkit_directory_and_selected_link_search_metadata_observation_not_library_bytes_or_build_custody","directories":observations}),
    })
}
