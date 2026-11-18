//! Shared Windows ABI cache identity for restorers and retention planning.

use crate::repository::check_report::CheckReport;
use serde::Serialize;

pub(super) const PREFIX: &str = "mesh-llm-windows-2022-skippy-abi-";

pub(super) struct Inputs<'a> {
    pub backend: &'a str,
    pub architectures: &'a str,
    pub epoch: &'a str,
    pub input_hash: &'a str,
    pub cuda: &'a str,
    pub vulkan: &'a str,
    pub rocm: &'a str,
}

#[derive(Serialize)]
pub(super) struct Identity {
    pub key: String,
    pub family: String,
}

pub(super) fn segment(value: &str, name: &str) -> Result<String, String> {
    if value.contains(['\r', '\n']) {
        return Err(format!("{name} must be a single-line cache-key segment"));
    }
    let mut normalized = String::new();
    let mut separator = false;
    for ch in value.trim().chars() {
        if ch.is_whitespace() || ",;:/\\".contains(ch) {
            separator = !normalized.is_empty();
        } else {
            if separator {
                normalized.push('_');
                separator = false;
            }
            normalized.push(ch);
        }
    }
    let normalized = normalized.trim_matches('_');
    if normalized.is_empty()
        || !normalized.starts_with(|ch: char| ch.is_ascii_alphanumeric())
        || !normalized
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "._+-".contains(ch))
    {
        return Err(format!("{name} contains invalid cache-key characters"));
    }
    Ok(normalized.to_owned())
}

pub(super) fn identity(input: &Inputs<'_>) -> Result<Identity, String> {
    let epoch = segment(input.epoch, "toolchain_epoch")?;
    let architectures = if input.architectures.trim().is_empty() {
        String::new()
    } else {
        segment(input.architectures, "architecture_set")?
    };
    let toolkit = match input.backend {
        "cpu" => "cpu".to_owned(),
        "cuda" => format!(
            "cuda-{}-Jimver-v0.2.36",
            segment(input.cuda, "cuda_toolchain_version")?
        ),
        "vulkan" => format!(
            "vulkan-{}-jakoch-v1.5.2",
            segment(input.vulkan, "vulkan_toolchain_version")?
        ),
        "rocm" => format!("rocm-{}", segment(input.rocm, "rocm_toolchain_version")?),
        _ => return Err("unsupported Windows ABI cache backend".into()),
    };
    if matches!(input.backend, "cuda" | "rocm") && architectures.is_empty() {
        return Err("architecture_set is required for accelerator cache identity".into());
    }
    let architecture_valid = match input.backend {
        "cpu" | "vulkan" => architectures.is_empty(),
        "cuda" => architectures
            .split('_')
            .all(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())),
        "rocm" => architectures.split('_').all(|value| {
            value.strip_prefix("gfx").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        }),
        _ => false,
    };
    if !architecture_valid {
        return Err("invalid architecture_set for Windows ABI backend".into());
    }
    if matches!(input.backend, "cuda" | "vulkan") {
        let version = if input.backend == "cuda" {
            input.cuda
        } else {
            input.vulkan
        };
        if !version
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err("Windows toolkit version must contain dotted decimal components".into());
        }
    }
    if input.input_hash.len() != 64
        || !input
            .input_hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("Windows ABI cache input hash is missing or invalid".into());
    }
    let family = format!(
        "{PREFIX}{}-{architectures}-{toolkit}-{epoch}",
        input.backend
    );
    let key = format!("{family}-{}", input.input_hash);
    if key.len() > 512 {
        return Err("Windows ABI cache key exceeds 512 bytes".into());
    }
    super::windows_cache_retention::family(&key).map_err(|error| error.to_string())?;
    Ok(Identity { key, family })
}

pub(super) fn run(args: &[String]) -> CheckReport {
    if args == ["--help"] {
        return CheckReport::success(
            "usage: ci-ops windows-cache-identity (reads INPUT_* and CACHE_INPUT_HASH)\n".into(),
        );
    }
    let result = (|| {
        if !args.is_empty() {
            return Err("windows-cache-identity accepts no arguments".to_owned());
        }
        let value = |name| std::env::var(name).unwrap_or_default();
        let epoch = segment(&value("INPUT_TOOLCHAIN_EPOCH"), "toolchain_epoch")?;
        if epoch
            != segment(
                &value("MESH_LLM_LLAMA_TOOLCHAIN_EPOCH"),
                "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
            )?
        {
            return Err("toolchain_epoch must match MESH_LLM_LLAMA_TOOLCHAIN_EPOCH".into());
        }
        let result = identity(&Inputs {
            backend: &value("INPUT_BACKEND").trim().to_ascii_lowercase(),
            architectures: &value("INPUT_ARCHITECTURE_SET"),
            epoch: &epoch,
            input_hash: &value("CACHE_INPUT_HASH"),
            cuda: &value("INPUT_CUDA_TOOLCHAIN_VERSION"),
            vulkan: &value("INPUT_VULKAN_TOOLCHAIN_VERSION"),
            rocm: &value("INPUT_ROCM_TOOLCHAIN_VERSION"),
        })?;
        serde_json::to_string(&result).map_err(|error| error.to_string())
    })();
    match result {
        Ok(json) => CheckReport::success(format!("{json}\n")),
        Err(error) => CheckReport::failure(String::new(), format!("{error}\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda<'a>(hash: &'a str) -> Inputs<'a> {
        Inputs {
            backend: "cuda",
            architectures: "75;86",
            epoch: "github-win22-20261115-X64",
            input_hash: hash,
            cuda: "12.6.3",
            vulkan: "",
            rocm: "",
        }
    }

    #[test]
    fn compatibility_dimensions_separate_keys_and_input_revisions_share_family() {
        let hash = "a".repeat(64);
        let baseline = identity(&cuda(&hash)).unwrap();
        assert_eq!(
            baseline.key,
            format!(
                "{PREFIX}cuda-75_86-cuda-12.6.3-Jimver-v0.2.36-github-win22-20261115-X64-{hash}"
            )
        );
        let changed_hash = "b".repeat(64);
        let changed = identity(&cuda(&changed_hash)).unwrap();
        assert_eq!(baseline.family, changed.family);
        assert_ne!(baseline.key, changed.key);
        for input in [
            Inputs {
                cuda: "13.1.2",
                ..cuda(&hash)
            },
            Inputs {
                architectures: "75;86;120",
                ..cuda(&hash)
            },
            Inputs {
                epoch: "github-win22-20261116-X64",
                ..cuda(&hash)
            },
            Inputs {
                backend: "rocm",
                architectures: "gfx1100",
                rocm: "AMD-25.Q3.exe",
                ..cuda(&hash)
            },
        ] {
            assert_ne!(baseline.family, identity(&input).unwrap().family);
        }
    }

    #[test]
    fn malformed_identity_never_produces_a_key() {
        let hash = "a".repeat(64);
        for input in [
            Inputs {
                backend: "metal",
                ..cuda(&hash)
            },
            Inputs {
                architectures: "",
                ..cuda(&hash)
            },
            Inputs {
                epoch: "image\nspoof",
                ..cuda(&hash)
            },
            Inputs {
                cuda: "",
                ..cuda(&hash)
            },
            Inputs {
                input_hash: "bad",
                ..cuda(&hash)
            },
        ] {
            assert!(identity(&input).is_err());
        }
    }
}
