use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

const SKIPPY_FEATURE_MODEL_INTROSPECTION: u64 = 1 << 3;
const SKIPPY_FEATURE_GGUF_SLICE_WRITE: u64 = 1 << 4;

#[derive(Debug, Parser)]
pub struct BackendArgs {
    #[arg(long = "skippy-runtime-library", value_name = "PATH")]
    skippy_runtime_libraries: Vec<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
pub struct BackendCapabilities {
    pub native_rust: NativeRustCapabilities,
    pub llama_api: LlamaApiCapabilities,
    pub skippy_abi: SkippyAbiCapabilities,
}

#[derive(Debug, Serialize)]
pub struct NativeRustCapabilities {
    pub convert_hf_to_gguf: bool,
    pub llama_quantize: bool,
    pub resumable_windows: bool,
    pub low_residency_streaming: bool,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct LlamaApiCapabilities {
    pub convert_hf_to_gguf: bool,
    pub llama_quantize: bool,
    pub runtime_loaded: bool,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct SkippyAbiCapabilities {
    pub convert_hf_to_gguf: bool,
    pub llama_quantize: bool,
    pub runtime_loaded: bool,
    pub feature_mask: Option<u64>,
    pub model_introspection: bool,
    pub gguf_slice_write: bool,
    pub load_error: Option<String>,
    pub reason: String,
}

pub fn capabilities(skippy_runtime_libraries: &[PathBuf]) -> BackendCapabilities {
    let skippy_abi = skippy_abi_capabilities(skippy_runtime_libraries);
    let llama_api = LlamaApiCapabilities {
        convert_hf_to_gguf: false,
        llama_quantize: skippy_ffi::native_runtime_loaded(),
        runtime_loaded: skippy_ffi::native_runtime_loaded(),
        reason: if skippy_ffi::native_runtime_loaded() {
            "loaded native runtime exposes llama_model_quantize".to_string()
        } else {
            "no native runtime library was loaded for llama API probing".to_string()
        },
    };
    BackendCapabilities {
        native_rust: NativeRustCapabilities {
            convert_hf_to_gguf: true,
            llama_quantize: false,
            resumable_windows: true,
            low_residency_streaming: true,
            reason: "Rust SafeTensors-to-GGUF writer streams tensor payloads and materializes one split window per run".to_string(),
        },
        llama_api,
        skippy_abi,
    }
}

pub fn run_backends(args: BackendArgs) -> Result<()> {
    let capabilities = capabilities(&args.skippy_runtime_libraries);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&capabilities)?);
    } else {
        println!(
            "native-rust: convert_hf_to_gguf={} llama_quantize={} resumable_windows={} low_residency_streaming={} reason={}",
            capabilities.native_rust.convert_hf_to_gguf,
            capabilities.native_rust.llama_quantize,
            capabilities.native_rust.resumable_windows,
            capabilities.native_rust.low_residency_streaming,
            capabilities.native_rust.reason
        );
        println!(
            "llama-api: convert_hf_to_gguf={} llama_quantize={} runtime_loaded={} reason={}",
            capabilities.llama_api.convert_hf_to_gguf,
            capabilities.llama_api.llama_quantize,
            capabilities.llama_api.runtime_loaded,
            capabilities.llama_api.reason
        );
        println!(
            "skippy-abi: convert_hf_to_gguf={} llama_quantize={} runtime_loaded={} feature_mask={} model_introspection={} gguf_slice_write={} reason={}",
            capabilities.skippy_abi.convert_hf_to_gguf,
            capabilities.skippy_abi.llama_quantize,
            capabilities.skippy_abi.runtime_loaded,
            capabilities
                .skippy_abi
                .feature_mask
                .map_or_else(|| "unknown".to_string(), |value| format!("{value:#x}")),
            capabilities.skippy_abi.model_introspection,
            capabilities.skippy_abi.gguf_slice_write,
            capabilities.skippy_abi.reason
        );
        if let Some(load_error) = capabilities.skippy_abi.load_error.as_deref() {
            println!("skippy-abi-load-error: {load_error}");
        }
    }
    Ok(())
}

fn skippy_abi_capabilities(skippy_runtime_libraries: &[PathBuf]) -> SkippyAbiCapabilities {
    let load_error = load_skippy_runtime_for_probe(skippy_runtime_libraries);
    let runtime_loaded = skippy_ffi::native_runtime_loaded();
    let feature_mask = if runtime_loaded {
        std::panic::catch_unwind(skippy_ffi::skippy_abi_features).ok()
    } else {
        None
    };
    let model_introspection = feature_mask.is_some_and(|mask| {
        mask & SKIPPY_FEATURE_MODEL_INTROSPECTION == SKIPPY_FEATURE_MODEL_INTROSPECTION
    });
    let gguf_slice_write = feature_mask.is_some_and(|mask| {
        mask & SKIPPY_FEATURE_GGUF_SLICE_WRITE == SKIPPY_FEATURE_GGUF_SLICE_WRITE
    });
    SkippyAbiCapabilities {
        convert_hf_to_gguf: false,
        llama_quantize: runtime_loaded,
        runtime_loaded,
        feature_mask,
        model_introspection,
        gguf_slice_write,
        load_error,
        reason: skippy_abi_reason(runtime_loaded, feature_mask, gguf_slice_write),
    }
}

fn load_skippy_runtime_for_probe(skippy_runtime_libraries: &[PathBuf]) -> Option<String> {
    if skippy_runtime_libraries.is_empty() || skippy_ffi::native_runtime_loaded() {
        return None;
    }
    // The caller explicitly supplied these native runtime libraries for probing.
    // Loading arbitrary libraries would be unsafe, so the command never guesses.
    let result = unsafe { skippy_ffi::load_native_runtime_libraries(skippy_runtime_libraries) };
    result.err().map(|err| err.to_string())
}

fn skippy_abi_reason(
    runtime_loaded: bool,
    feature_mask: Option<u64>,
    gguf_slice_write: bool,
) -> String {
    if !runtime_loaded {
        return "no Skippy native runtime library was loaded for ABI probing".to_string();
    }
    if feature_mask.is_none() {
        return "loaded Skippy runtime does not expose skippy_abi_features".to_string();
    }
    if gguf_slice_write {
        return "loaded Skippy ABI exposes GGUF slice writing and the linked llama symbols can be used for GGUF quantization, but not HF checkpoint conversion".to_string();
    }
    "loaded Skippy ABI exposes staged inference/runtime entry points and linked llama symbols can be used for GGUF quantization, but not HF checkpoint conversion".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendKind {
    NativeRust,
    LlamaApi,
    SkippyAbi,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NativeRust => "native-rust",
            Self::LlamaApi => "llama-api",
            Self::SkippyAbi => "skippy-abi",
        }
    }
}

pub fn ensure_convert_backend(kind: BackendKind) -> Result<()> {
    ensure!(
        matches!(kind, BackendKind::NativeRust),
        "backend {} cannot convert HF checkpoints yet: {}",
        kind.as_str(),
        capabilities(&[]).skippy_abi.reason
    );
    Ok(())
}

pub fn ensure_quant_backend(kind: BackendKind) -> Result<()> {
    ensure!(
        matches!(kind, BackendKind::LlamaApi | BackendKind::SkippyAbi),
        "backend {} cannot quantize GGUFs yet: {}",
        kind.as_str(),
        capabilities(&[]).skippy_abi.reason
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendRunStatus {
    pub status_code: Option<i32>,
    pub success: bool,
}

impl BackendRunStatus {
    pub fn from_code(status_code: i32) -> Self {
        Self {
            status_code: Some(status_code),
            success: status_code == 0,
        }
    }
}

pub fn ensure_success(status: BackendRunStatus, command: &[String]) -> Result<()> {
    ensure!(
        status.success,
        "command failed with status_code {:?}: {}",
        status.status_code,
        shell_words(command)
    );
    Ok(())
}

fn shell_words(command: &[String]) -> String {
    command.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_current_backend_capabilities() {
        let capabilities = capabilities(&[]);
        assert!(capabilities.native_rust.convert_hf_to_gguf);
        assert!(!capabilities.native_rust.llama_quantize);
        assert!(!capabilities.llama_api.convert_hf_to_gguf);
        assert!(!capabilities.llama_api.llama_quantize);
        assert!(!capabilities.llama_api.runtime_loaded);
        assert!(!capabilities.skippy_abi.convert_hf_to_gguf);
        assert!(!capabilities.skippy_abi.llama_quantize);
        assert!(!capabilities.skippy_abi.runtime_loaded);
        assert_eq!(capabilities.skippy_abi.feature_mask, None);
        assert!(!capabilities.skippy_abi.model_introspection);
        assert!(!capabilities.skippy_abi.gguf_slice_write);
        assert_eq!(capabilities.skippy_abi.load_error, None);
        assert!(capabilities.skippy_abi.reason.contains("Skippy"));
    }

    #[test]
    fn rejects_skippy_abi_conversion_backend_until_supported() {
        assert!(ensure_convert_backend(BackendKind::SkippyAbi).is_err());
    }

    #[test]
    fn accepts_skippy_abi_quant_backend() {
        assert!(ensure_quant_backend(BackendKind::SkippyAbi).is_ok());
    }

    #[test]
    fn validates_backend_run_status() {
        let ok = BackendRunStatus {
            status_code: Some(0),
            success: true,
        };
        let failed = BackendRunStatus {
            status_code: Some(2),
            success: false,
        };
        assert!(ensure_success(ok, &["tool".to_string()]).is_ok());
        assert!(ensure_success(failed, &["tool".to_string()]).is_err());
    }
}
