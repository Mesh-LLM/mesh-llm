use super::{BackendControlFuture, BackendLaunchFuture, BackendOps};
use crate::launch::{
    reqwest_health_check, InferenceServerHandle, InferenceServerProcess, ModelLaunchSpec,
};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::process::Command;

pub(super) struct VllmBackend;

#[derive(Clone, Debug)]
struct ResolvedVllmRuntime {
    executable: PathBuf,
    flavor: &'static str,
}

#[derive(Debug, Default, PartialEq)]
struct VllmLaunchOptions {
    dtype: Option<String>,
    trust_remote_code: bool,
    gpu_memory_utilization: Option<f32>,
    max_num_batched_tokens: Option<u32>,
    extra_args: Vec<String>,
}

impl BackendOps for VllmBackend {
    fn as_str(&self) -> &'static str {
        "vllm"
    }

    fn process_label(&self) -> &'static str {
        "vllm"
    }

    fn resolve_runtime_label(&self) -> Result<Option<String>> {
        Ok(Some(resolve_vllm_runtime()?.flavor.to_string()))
    }

    fn validate_model<'a>(&self, model: &'a Path) -> Result<()> {
        validate_vllm_model(model)
    }

    fn start_server<'a>(
        &self,
        _bin_dir: &'a Path,
        _binary_flavor: Option<crate::launch::BinaryFlavor>,
        spec: ModelLaunchSpec<'a>,
    ) -> BackendLaunchFuture<'a> {
        Box::pin(start_vllm_server(spec))
    }

    fn kill_server_processes<'a>(&'a self) -> BackendControlFuture<'a> {
        Box::pin(kill_vllm_server())
    }
}

pub(super) static VLLM_BACKEND: VllmBackend = VllmBackend;

async fn kill_vllm_server() {
    let _ = std::process::Command::new("pkill")
        .args(["-f", "vllm serve"])
        .status();
    let _ = std::process::Command::new("pkill")
        .args(["-f", "vllm.entrypoints.openai.api_server"])
        .status();
}

fn search_path(exe: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(exe);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn detect_vllm_runtime_flavor(path: &Path) -> &'static str {
    let path_text = path.to_string_lossy().to_ascii_lowercase();
    if path_text.contains("vllm-metal") {
        "metal"
    } else {
        "standard"
    }
}

fn resolve_vllm_runtime() -> Result<ResolvedVllmRuntime> {
    if let Some(path) = std::env::var_os("MESH_LLM_VLLM_BIN") {
        let path = PathBuf::from(path);
        anyhow::ensure!(
            path.exists(),
            "MESH_LLM_VLLM_BIN points to missing path {}",
            path.display()
        );
        return Ok(ResolvedVllmRuntime {
            flavor: detect_vllm_runtime_flavor(&path),
            executable: path,
        });
    }

    if let Some(home) = dirs::home_dir() {
        let default = home.join(".venv-vllm-metal").join("bin").join("vllm");
        if default.exists() {
            return Ok(ResolvedVllmRuntime {
                flavor: "metal",
                executable: default,
            });
        }
    }

    if let Some(path) = search_path("vllm") {
        return Ok(ResolvedVllmRuntime {
            flavor: detect_vllm_runtime_flavor(&path),
            executable: path,
        });
    }

    anyhow::bail!(
        "vllm executable not found. Install vllm/vllm-metal, pass --vllm-bin, or set MESH_LLM_VLLM_BIN"
    );
}

fn read_trimmed_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_bool_env(name: &str) -> Result<Option<bool>> {
    let Some(value) = read_trimmed_env(name) else {
        return Ok(None);
    };
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => anyhow::bail!(
            "{name} must be one of true/false/1/0/yes/no/on/off, got '{}'",
            value
        ),
    }
}

fn parse_extra_args(raw: &str) -> Result<Vec<String>> {
    if raw.contains('\n') {
        return Ok(raw
            .lines()
            .map(str::trim)
            .filter(|arg| !arg.is_empty())
            .map(str::to_string)
            .collect());
    }

    shlex::split(raw).ok_or_else(|| anyhow::anyhow!("Failed to parse MESH_LLM_VLLM_ARGS"))
}

fn load_vllm_launch_options() -> Result<VllmLaunchOptions> {
    let dtype = read_trimmed_env("MESH_LLM_VLLM_DTYPE");
    let trust_remote_code = parse_bool_env("MESH_LLM_VLLM_TRUST_REMOTE_CODE")?.unwrap_or(false);
    let gpu_memory_utilization = read_trimmed_env("MESH_LLM_VLLM_GPU_MEMORY_UTILIZATION")
        .map(|value| {
            let parsed: f32 = value.parse().with_context(|| {
                format!(
                    "Failed to parse MESH_LLM_VLLM_GPU_MEMORY_UTILIZATION='{}'",
                    value
                )
            })?;
            anyhow::ensure!(
                (0.0..=1.0).contains(&parsed),
                "MESH_LLM_VLLM_GPU_MEMORY_UTILIZATION must be between 0.0 and 1.0"
            );
            Ok(parsed)
        })
        .transpose()?;
    let max_num_batched_tokens = read_trimmed_env("MESH_LLM_VLLM_MAX_NUM_BATCHED_TOKENS")
        .map(|value| {
            value.parse::<u32>().with_context(|| {
                format!(
                    "Failed to parse MESH_LLM_VLLM_MAX_NUM_BATCHED_TOKENS='{}'",
                    value
                )
            })
        })
        .transpose()?;
    let extra_args = read_trimmed_env("MESH_LLM_VLLM_ARGS")
        .map(|value| parse_extra_args(&value))
        .transpose()?
        .unwrap_or_default();

    Ok(VllmLaunchOptions {
        dtype,
        trust_remote_code,
        gpu_memory_utilization,
        max_num_batched_tokens,
        extra_args,
    })
}

fn validate_vllm_model(model: &Path) -> Result<()> {
    anyhow::ensure!(model.exists(), "Model not found at {}", model.display());
    anyhow::ensure!(
        model.is_dir(),
        "vllm backend expects a HuggingFace-style model directory, got {}",
        model.display()
    );

    let config_path = model.join("config.json");
    anyhow::ensure!(
        config_path.exists(),
        "vllm backend requires config.json in {}",
        model.display()
    );
    anyhow::ensure!(
        model.join("tokenizer.json").exists() || model.join("tokenizer_config.json").exists(),
        "vllm backend requires tokenizer metadata in {}",
        model.display()
    );
    anyhow::ensure!(
        model.join("model.safetensors").exists()
            || model.join("model.safetensors.index.json").exists()
            || model.join("pytorch_model.bin").exists()
            || model.join("pytorch_model.bin.index.json").exists(),
        "vllm backend requires safetensors or pytorch weight files in {}",
        model.display()
    );

    let config_text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_text)
        .with_context(|| format!("Failed to parse {}", config_path.display()))?;
    anyhow::ensure!(
        config_json.get("quantization").is_none(),
        "vllm backend does not yet support HuggingFace directories with config.quantization in {}. Use bf16/fp16 weights or a GGUF model instead.",
        model.display()
    );

    Ok(())
}

async fn start_vllm_server(spec: ModelLaunchSpec<'_>) -> Result<InferenceServerProcess> {
    anyhow::ensure!(
        spec.tunnel_ports.is_empty(),
        "vllm backend does not support rpc split workers yet"
    );
    anyhow::ensure!(
        spec.tensor_split.is_none(),
        "vllm backend does not support tensor split yet"
    );
    anyhow::ensure!(
        spec.draft.is_none(),
        "vllm backend does not support speculative draft models yet"
    );
    anyhow::ensure!(
        spec.mmproj.is_none(),
        "vllm backend does not support llama.cpp mmproj launch args"
    );
    validate_vllm_model(spec.model)?;

    let runtime = resolve_vllm_runtime()?;
    let options = load_vllm_launch_options()?;
    let log_path = std::env::temp_dir().join("mesh-llm-vllm.log");
    let log_file = std::fs::File::create(&log_path)
        .with_context(|| format!("Failed to create vllm log file {}", log_path.display()))?;
    let log_file2 = log_file.try_clone()?;

    let mut args = vec![
        "serve".to_string(),
        spec.model.to_string_lossy().to_string(),
        "--host".to_string(),
        "0.0.0.0".to_string(),
        "--port".to_string(),
        spec.http_port.to_string(),
        "--served-model-name".to_string(),
        spec.served_model_name.to_string(),
    ];
    if let Some(ctx_size) = spec.ctx_size_override {
        args.push("--max-model-len".to_string());
        args.push(ctx_size.to_string());
    }
    if let Some(dtype) = &options.dtype {
        args.push("--dtype".to_string());
        args.push(dtype.clone());
    }
    if options.trust_remote_code {
        args.push("--trust-remote-code".to_string());
    }
    if let Some(utilization) = options.gpu_memory_utilization {
        args.push("--gpu-memory-utilization".to_string());
        args.push(utilization.to_string());
    }
    if let Some(tokens) = options.max_num_batched_tokens {
        args.push("--max-num-batched-tokens".to_string());
        args.push(tokens.to_string());
    }
    args.extend(options.extra_args.iter().cloned());

    tracing::info!(
        "Starting vllm ({}) on :{} with model {}",
        runtime.flavor,
        spec.http_port,
        spec.model.display()
    );

    let mut child = Command::new(&runtime.executable)
        .args(&args)
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file2))
        .spawn()
        .with_context(|| format!("Failed to start vllm at {}", runtime.executable.display()))?;

    let url = format!("http://localhost:{}/health", spec.http_port);
    for _ in 0..600 {
        if reqwest_health_check(&url).await {
            let pid = child
                .id()
                .context("vllm started but did not expose a PID")?;
            let expected_exit = Arc::new(AtomicBool::new(false));
            let handle = InferenceServerHandle::new(pid, expected_exit.clone());
            let (death_tx, death_rx) = tokio::sync::oneshot::channel();
            tokio::spawn(async move {
                let _ = child.wait().await;
                if !expected_exit.load(Ordering::Relaxed) {
                    eprintln!("⚠️  vllm process exited unexpectedly");
                }
                let _ = death_tx.send(());
            });
            return Ok(InferenceServerProcess {
                handle,
                death_rx,
                backend_runtime: Some(runtime.flavor.to_string()),
            });
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    anyhow::bail!(
        "vllm failed to become healthy within 600s. See {}",
        log_path.display()
    );
}

#[cfg(test)]
mod tests {
    use super::{
        load_vllm_launch_options, parse_extra_args, validate_vllm_model, VllmLaunchOptions,
    };
    use serial_test::serial;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn validate_vllm_model_rejects_quantized_config_dirs() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("mesh-llm-vllm-quantized-{unique}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"model_type":"qwen2","quantization":{"bits":4}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"").unwrap();

        let err = validate_vllm_model(&dir).unwrap_err().to_string();
        assert!(err.contains("config.quantization"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn parse_extra_args_supports_shell_words() {
        let args = parse_extra_args("--enforce-eager --chat-template 'foo bar'").unwrap();
        assert_eq!(
            args,
            vec![
                "--enforce-eager".to_string(),
                "--chat-template".to_string(),
                "foo bar".to_string()
            ]
        );
    }

    #[test]
    #[serial]
    fn load_vllm_launch_options_reads_env() {
        std::env::set_var("MESH_LLM_VLLM_DTYPE", "bfloat16");
        std::env::set_var("MESH_LLM_VLLM_TRUST_REMOTE_CODE", "true");
        std::env::set_var("MESH_LLM_VLLM_GPU_MEMORY_UTILIZATION", "0.85");
        std::env::set_var("MESH_LLM_VLLM_MAX_NUM_BATCHED_TOKENS", "4096");
        std::env::set_var(
            "MESH_LLM_VLLM_ARGS",
            "--enforce-eager\n--chat-template-content-format\nstring",
        );

        let options = load_vllm_launch_options().unwrap();
        assert_eq!(
            options,
            VllmLaunchOptions {
                dtype: Some("bfloat16".to_string()),
                trust_remote_code: true,
                gpu_memory_utilization: Some(0.85),
                max_num_batched_tokens: Some(4096),
                extra_args: vec![
                    "--enforce-eager".to_string(),
                    "--chat-template-content-format".to_string(),
                    "string".to_string()
                ],
            }
        );

        std::env::remove_var("MESH_LLM_VLLM_DTYPE");
        std::env::remove_var("MESH_LLM_VLLM_TRUST_REMOTE_CODE");
        std::env::remove_var("MESH_LLM_VLLM_GPU_MEMORY_UTILIZATION");
        std::env::remove_var("MESH_LLM_VLLM_MAX_NUM_BATCHED_TOKENS");
        std::env::remove_var("MESH_LLM_VLLM_ARGS");
    }
}
