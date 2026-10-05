use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Serialize;

use crate::output::{print_json_pretty, print_success};
use crate::tool_paths::resolve_llama_cli;

#[derive(Debug, Parser)]
pub(crate) struct ValidateLlamaLoadArgs {
    #[arg(long)]
    llama_cli: Option<PathBuf>,
    #[arg(long)]
    check_tensors: bool,
    #[arg(long)]
    json: bool,
    model: PathBuf,
}

#[derive(Debug, Serialize)]
pub(crate) struct LlamaLoadReport {
    pub(crate) model: PathBuf,
    pub(crate) llama_cli: PathBuf,
    pub(crate) command: Vec<String>,
    pub(crate) status_code: Option<i32>,
    pub(crate) success: bool,
    pub(crate) stdout_tail: String,
    pub(crate) stderr_tail: String,
}

pub(crate) fn run_validate_llama_load(args: ValidateLlamaLoadArgs) -> Result<()> {
    let report = validate_llama_load(
        &args.model,
        args.llama_cli.as_deref(),
        LlamaLoadOptions {
            check_tensors: args.check_tensors,
        },
    )?;
    if args.json {
        print_json_pretty(&report)?;
    } else {
        print_success(format!(
            "llama load valid: model={} llama_cli={} status={}",
            report.model.display(),
            report.llama_cli.display(),
            report
                .status_code
                .map_or_else(|| "signal".to_string(), |code| code.to_string())
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LlamaLoadOptions {
    pub(crate) check_tensors: bool,
}

pub(crate) fn validate_llama_load(
    model: &Path,
    llama_cli: Option<&Path>,
    options: LlamaLoadOptions,
) -> Result<LlamaLoadReport> {
    ensure!(model.is_file(), "model does not exist: {}", model.display());
    let llama_cli = resolve_llama_cli(llama_cli).with_context(
        || "llama-cli was not found; pass --llama-cli or set SKIPPY_QUANTIZE_LLAMA_CLI",
    )?;
    ensure!(
        llama_cli.is_file(),
        "llama-cli does not exist: {}",
        llama_cli.display()
    );
    let command = build_llama_load_command(&llama_cli, model, options)?;
    let output = Command::new(&command[0])
        .args(&command[1..])
        .output()
        .with_context(|| format!("run {}", command.join(" ")))?;
    let report = LlamaLoadReport {
        model: model.to_path_buf(),
        llama_cli,
        command,
        status_code: output.status.code(),
        success: output.status.success(),
        stdout_tail: tail_lossy(&output.stdout, 16 * 1024),
        stderr_tail: tail_lossy(&output.stderr, 16 * 1024),
    };
    ensure!(
        report.success,
        "llama-cli failed to load model {} status={:?}\nstderr_tail:\n{}",
        model.display(),
        report.status_code,
        report.stderr_tail
    );
    Ok(report)
}

pub(crate) fn build_llama_load_command(
    llama_cli: &Path,
    model: &Path,
    options: LlamaLoadOptions,
) -> Result<Vec<String>> {
    if is_llama_simple(llama_cli) {
        ensure!(
            !options.check_tensors,
            "--check-tensors requires llama-cli; llama-simple only supports a load smoke"
        );
        return Ok(vec![
            llama_cli.display().to_string(),
            "-m".to_string(),
            model.display().to_string(),
            "-n".to_string(),
            "1".to_string(),
            " ".to_string(),
        ]);
    }
    let mut command = vec![
        llama_cli.display().to_string(),
        "--model".to_string(),
        model.display().to_string(),
        "--n-predict".to_string(),
        "0".to_string(),
        "--prompt".to_string(),
        String::new(),
        "--no-conversation".to_string(),
    ];
    if options.check_tensors {
        command.push("--check-tensors".to_string());
    }
    Ok(command)
}

fn is_llama_simple(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "llama-simple")
}

fn tail_lossy(bytes: &[u8], max_bytes: usize) -> String {
    let start = bytes.len().saturating_sub(max_bytes);
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_load_command_with_optional_tensor_check() {
        let command = build_llama_load_command(
            Path::new("/bin/llama-cli"),
            Path::new("/models/model.gguf"),
            LlamaLoadOptions {
                check_tensors: true,
            },
        );

        assert_eq!(
            command.unwrap(),
            vec![
                "/bin/llama-cli",
                "--model",
                "/models/model.gguf",
                "--n-predict",
                "0",
                "--prompt",
                "",
                "--no-conversation",
                "--check-tensors",
            ]
        );
    }

    #[test]
    fn builds_simple_load_command_for_older_llama_builds() {
        let command = build_llama_load_command(
            Path::new("/bin/llama-simple"),
            Path::new("/models/model.gguf"),
            LlamaLoadOptions {
                check_tensors: false,
            },
        )
        .unwrap();

        assert_eq!(
            command,
            vec![
                "/bin/llama-simple",
                "-m",
                "/models/model.gguf",
                "-n",
                "1",
                " "
            ]
        );
    }

    #[test]
    fn rejects_tensor_check_with_simple_loader() {
        let error = build_llama_load_command(
            Path::new("/bin/llama-simple"),
            Path::new("/models/model.gguf"),
            LlamaLoadOptions {
                check_tensors: true,
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("--check-tensors requires llama-cli"));
    }

    #[test]
    fn keeps_bounded_output_tail() {
        assert_eq!(tail_lossy(b"abcdef", 3), "def");
        assert_eq!(tail_lossy(b"abc", 16), "abc");
    }
}
