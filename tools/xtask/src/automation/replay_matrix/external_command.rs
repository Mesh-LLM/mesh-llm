use super::external_config::{Arm, Engine};
use crate::command::DynResult;
use std::path::{Path, PathBuf};

pub(super) fn server(arm: &Arm, executable: &Path, port: u16) -> DynResult<Vec<String>> {
    if port == 0 {
        return Err("external server requires a nonzero port".into());
    }
    let served = arm.served_model()?;
    let mut args: Vec<String> = match arm.engine {
        Engine::Llama => vec![
            "--model",
            &arm.model,
            "--alias",
            served,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--ctx-size",
            &arm.context_size.to_string(),
            "--parallel",
            &arm.max_concurrency.to_string(),
            "--batch-size",
            &arm.batch_size.to_string(),
            "--ubatch-size",
            &arm.ubatch_size.to_string(),
            "--n-gpu-layers",
            "all",
            "--cont-batching",
            "--kv-unified",
            "--no-context-shift",
            "--metrics",
            "--no-webui",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        Engine::Vllm => vec![
            "serve",
            &arm.model,
            "--served-model-name",
            served,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--max-model-len",
            &arm.context_size.to_string(),
            "--max-num-seqs",
            &arm.max_concurrency.to_string(),
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        Engine::Sglang => vec![
            "-m",
            "sglang.launch_server",
            "--model-path",
            &arm.model,
            "--served-model-name",
            served,
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--context-length",
            &arm.context_size.to_string(),
            "--max-running-requests",
            &arm.max_concurrency.to_string(),
            "--disable-prefill-cuda-graph",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
    };
    options(arm, &mut args);
    args.extend(arm.extra_args.clone());
    let mut command = vec![
        executable
            .to_str()
            .ok_or("non-Unicode external executable")?
            .into(),
    ];
    command.extend(args);
    Ok(command)
}

fn options(arm: &Arm, args: &mut Vec<String>) {
    match arm.engine {
        Engine::Llama => {
            if !arm.prefix_cache {
                args.push("--no-cache-prompt".into());
            }
        }
        Engine::Vllm => {
            if let Some(tokenizer) = &arm.tokenizer {
                args.extend(["--tokenizer".into(), tokenizer.clone()]);
            }
            if let Some(config) = &arm.hf_config {
                args.extend(["--hf-config-path".into(), config.clone()]);
            }
            gguf(arm, args);
            args.push(
                if arm.prefix_cache {
                    "--enable-prefix-caching"
                } else {
                    "--no-enable-prefix-caching"
                }
                .into(),
            );
        }
        Engine::Sglang => {
            if let Some(tokenizer) = &arm.tokenizer {
                args.extend(["--tokenizer-path".into(), tokenizer.clone()]);
            }
            gguf(arm, args);
            if !arm.prefix_cache {
                args.push("--disable-radix-cache".into());
            }
        }
    }
}
fn gguf(arm: &Arm, args: &mut Vec<String>) {
    if arm.model.to_ascii_lowercase().ends_with(".gguf") {
        args.extend(["--load-format", "gguf", "--quantization", "gguf"].map(str::to_owned));
    }
}

pub(super) fn version(arm: &Arm) -> Vec<String> {
    match arm.engine {
        Engine::Sglang => vec![
            "-c".into(),
            "import importlib.metadata; print(importlib.metadata.version('sglang'))".into(),
        ],
        Engine::Llama | Engine::Vllm => vec!["--version".into()],
    }
}

pub(super) fn executable(arm: &Arm) -> DynResult<PathBuf> {
    if !arm.cwd.is_dir() {
        return Err("external engine cwd does not exist".into());
    }
    let path = super::external_config::expand_home(Path::new(&arm.executable))?;
    let has_separator = arm.executable.contains(std::path::MAIN_SEPARATOR)
        || cfg!(windows) && arm.executable.contains('/');
    let candidates = if path.is_absolute() || has_separator {
        vec![if path.is_absolute() {
            path
        } else {
            arm.cwd.join(path)
        }]
    } else {
        std::env::split_paths(&std::env::var_os("PATH").ok_or("missing PATH")?)
            .map(|directory| {
                if directory.is_absolute() {
                    directory
                } else {
                    arm.cwd.join(directory)
                }
            })
            .map(|directory| directory.join(&path))
            .collect()
    };
    for candidate in candidates {
        if runnable(&candidate) {
            return Ok(std::path::absolute(candidate)?);
        }
        #[cfg(windows)]
        if candidate.extension().is_none() {
            for extension in std::env::var("PATHEXT")
                .unwrap_or(".COM;.EXE;.BAT;.CMD".into())
                .split(';')
            {
                let extended = candidate.with_extension(extension.trim_start_matches('.'));
                if runnable(&extended) {
                    return Ok(std::path::absolute(extended)?);
                }
            }
        }
    }
    Err(format!("external executable not found: {}", arm.executable).into())
}

fn runnable(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}
