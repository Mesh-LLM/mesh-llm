//! Explicit nonsecret cache producer profile shared by correctness and serving.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Toolkit {
    pub path: PathBuf,
    pub sha256: String,
}
pub(super) fn scalar(name: &str, value: &str) -> bool {
    let names = [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "LLAMA_STAGE_BACKEND",
        "SKIPPY_LLAMA_BACKEND",
        "LLAMA_STAGE_LINK_MODE",
        "SKIPPY_LLAMA_LINK_MODE",
        "LLAMA_STAGE_CUDA_ARCHITECTURES",
        "SKIPPY_CUDA_ARCHITECTURES",
        "LLAMA_STAGE_AMDGPU_TARGETS",
        "SKIPPY_AMDGPU_TARGETS",
        "LLAMA_STAGE_GGML_NATIVE",
        "SKIPPY_GGML_NATIVE",
    ];
    let denied = [
        "AUTH",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "PATH",
        "DIR",
        "ROOT",
        "KEY",
    ];
    (names.contains(&name) || name.starts_with("GGML_"))
        && !denied.iter().any(|p| name.contains(p))
        && !value.is_empty()
        && value.len() <= 256
        && !value.chars().any(char::is_control)
}
pub(super) fn validate(
    settings: &BTreeMap<String, String>,
    toolkits: &BTreeMap<String, Toolkit>,
) -> DynResult<()> {
    if settings.len() > 64
        || settings.iter().any(|(k, v)| !scalar(k, v))
        || toolkits.len() > 5
        || toolkits.iter().any(|(k, v)| {
            ![
                "CUDA_PATH",
                "HIP_PATH",
                "ROCM_PATH",
                "LLVMInstallDir",
                "VULKAN_SDK",
            ]
            .contains(&k.as_str())
                || !v.path.is_absolute()
                || !(v.sha256.len() == 64
                    && v.sha256
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
        })
    {
        return Err(
            "cache profile requires explicit nonsecret scalars and pinned toolkit directories"
                .into(),
        );
    }
    for (primary, alias) in [
        ("LLAMA_STAGE_BACKEND", "SKIPPY_LLAMA_BACKEND"),
        ("LLAMA_STAGE_LINK_MODE", "SKIPPY_LLAMA_LINK_MODE"),
        (
            "LLAMA_STAGE_CUDA_ARCHITECTURES",
            "SKIPPY_CUDA_ARCHITECTURES",
        ),
        ("LLAMA_STAGE_AMDGPU_TARGETS", "SKIPPY_AMDGPU_TARGETS"),
        ("LLAMA_STAGE_GGML_NATIVE", "SKIPPY_GGML_NATIVE"),
    ] {
        if let (Some(a), Some(b)) = (settings.get(primary), settings.get(alias))
            && a != b
        {
            return Err("conflicting cache backend/build aliases".into());
        }
    }
    Ok(())
}
pub(super) fn observe(toolkits: &mut BTreeMap<String, Toolkit>) -> DynResult<()> {
    for toolkit in toolkits.values_mut() {
        toolkit.path = toolkit.path.canonicalize()?;
        crate::automation::native_artifact_identity::verify(&toolkit.path, &toolkit.sha256)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_profile_preserves_consumed_nonsecret_settings_and_refuses_conflicting_aliases() {
        let mut settings = BTreeMap::from([
            ("LLAMA_STAGE_BACKEND".into(), "cuda".into()),
            ("CUDA_VISIBLE_DEVICES".into(), "2,3".into()),
            ("GGML_CUDA_NO_VMM".into(), "1".into()),
        ]);
        validate(&settings, &BTreeMap::new()).unwrap();
        settings.insert("SKIPPY_LLAMA_BACKEND".into(), "metal".into());
        assert!(validate(&settings, &BTreeMap::new()).is_err());
        for key in ["HF_TOKEN", "GGML_AUTH_TOKEN", "CUDA_PATH", "MESH_UNOWNED"] {
            assert!(!scalar(key, "private"));
        }
        assert!(!scalar("CUDA_VISIBLE_DEVICES", "1\n2"));
    }
    #[test]
    fn cache_profile_toolkit_custody_refuses_wrong_pin_and_missing_directory() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().canonicalize().unwrap();
        std::fs::write(path.join("toolkit-version"), b"inert toolkit").unwrap();
        let mut toolkits = BTreeMap::from([(
            "CUDA_PATH".into(),
            Toolkit {
                path: path.clone(),
                sha256: "0".repeat(64),
            },
        )]);
        validate(&BTreeMap::new(), &toolkits).unwrap();
        assert!(observe(&mut toolkits).is_err());
        toolkits.get_mut("CUDA_PATH").unwrap().path = path.join("absent");
        assert!(observe(&mut toolkits).is_err());
        toolkits.get_mut("CUDA_PATH").unwrap().path = "relative".into();
        assert!(validate(&BTreeMap::new(), &toolkits).is_err());
    }
}
