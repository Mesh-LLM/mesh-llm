use crate::command::DynResult;
use std::{path::PathBuf, time::Duration};

pub(super) struct Options {
    pub artifact: Option<PathBuf>,
    pub binary: PathBuf,
    pub model: PathBuf,
    pub native: PathBuf,
    pub cache: PathBuf,
    pub command: PathBuf,
    pub arguments: Vec<String>,
    pub api: u16,
    pub console: u16,
    pub context: u32,
    pub wait: Duration,
    pub consumer_deadline: Duration,
}

impl Options {
    pub fn parse(args: &[String]) -> DynResult<Self> {
        let [binary, _, model, separator, command, arguments @ ..] = args else {
            return Err("usage: automation sdk-fixture BINARY BIN_DIR MODEL -- COMMAND...".into());
        };
        if separator != "--" {
            return Err("SDK command requires -- separator".into());
        }
        let binary = PathBuf::from(binary).canonicalize()?;
        let model = PathBuf::from(model).canonicalize()?;
        if !binary.is_file() || !model.is_file() {
            return Err("SDK binary and model must be files".into());
        }
        let artifact = std::env::var_os("MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let native = artifact
            .as_ref()
            .map(|path| path.as_os_str().to_owned())
            .or_else(|| std::env::var_os("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"))
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                binary
                    .parent()
                    .unwrap_or(std::path::Path::new("/"))
                    .join("native-runtimes")
            })
            .canonicalize()?;
        let cache = std::env::var_os("MESH_SDK_NATIVE_RUNTIME_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("RUNNER_TEMP")
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir)
                    .join("mesh-llm-sdk-native-runtimes")
            });
        let command = resolve(command)?;
        let number = |key: &str, default: u64| -> DynResult<u64> {
            let value = std::env::var(key).map_or(Ok(default), |value| value.parse())?;
            if value == 0 || value > 86400 {
                return Err("SDK budgets and sizes must be in 1..=86400".into());
            }
            Ok(value)
        };
        Ok(Self {
            artifact,
            binary,
            model,
            native,
            cache,
            command,
            arguments: arguments.to_vec(),
            api: u16::try_from(number("MESH_SDK_API_PORT", 9347)?)?,
            console: u16::try_from(number("MESH_SDK_CONSOLE_PORT", 3141)?)?,
            context: u32::try_from(number("MESH_SDK_CTX_SIZE", 256)?)?,
            wait: Duration::from_secs(number("MESH_SDK_MAX_WAIT", 180)?),
            consumer_deadline: Duration::from_secs(number("MESH_SDK_COMMAND_MAX_WAIT", 1800)?),
        })
    }
}

pub(super) fn resolve(command: &str) -> DynResult<PathBuf> {
    let path = PathBuf::from(command);
    if path.components().count() > 1 || path.is_absolute() {
        return Ok(std::path::absolute(path)?);
    }
    // Keep the PATH entry: rustup proxies are symlinks whose target dispatches on argv0.
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        for suffix in ["", std::env::consts::EXE_SUFFIX] {
            let path = directory.join(format!("{command}{suffix}"));
            if path.is_file() {
                return Ok(path);
            }
        }
    }
    Err("SDK executable not found on PATH".into())
}
