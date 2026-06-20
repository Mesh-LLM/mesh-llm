use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;

use crate::backend::{ExternalProcessOptions, run_backend_command};
use crate::hf_checkpoint::resolve_auto_output_type;
use crate::locking::with_manifest_lock;
use crate::manifest::ensure_manifest;
use crate::memory_budget::{MemoryPolicy, MemorySize};
use crate::preflight::run_job_preflight;
use crate::splits::parse_split_file_name;
use crate::types::ConvertOutputType;
use crate::verify::print_verify_on_complete;
use crate::{
    BackendKind, ConvertRunnerArgs, InitConvertArgs, RunConvertArgs, RunConvertWindowArgs,
    VerifyLoadArgs, convert_manifest_from_args, prepare_convert_runner, run_convert_unlocked,
};

#[derive(Debug, Parser)]
pub(crate) struct DirectConvertArgs {
    #[command(flatten)]
    runner: ConvertRunnerArgs,
    #[arg(long)]
    target_prefix: Option<String>,
    #[arg(long)]
    output_basename: Option<String>,
    #[arg(long, alias = "outtype", value_enum, default_value_t = ConvertOutputType::Auto)]
    output_type: ConvertOutputType,
    #[arg(short = 'o', long)]
    outfile: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    expected_splits: u32,
    #[arg(long, default_value_t = 1)]
    window_size: u32,
    #[arg(long)]
    max_windows: Option<u32>,
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long = "no-verify-on-complete", action = clap::ArgAction::SetFalse, default_value_t = true)]
    verify_on_complete: bool,
    #[command(flatten)]
    verify_load: VerifyLoadArgs,
    #[arg(long)]
    preflight_only: bool,
    #[arg(long)]
    json: bool,
    source: Option<PathBuf>,
    output: Option<PathBuf>,
}

pub(crate) fn run_direct_convert(args: DirectConvertArgs) -> Result<()> {
    let runner = prepare_convert_runner(args.runner.clone())?;
    if runner.print_supported_models && args.source.is_none() && args.output.is_none() {
        ensure!(
            args.outfile.is_none(),
            "--print-supported-models does not accept --outfile"
        );
        return run_print_supported_models(&runner);
    }
    let source = args
        .source
        .clone()
        .context("missing source path: provide MODEL or use --print-supported-models")?;
    let output_type = direct_output_type(&runner, &args, &source)?;
    let output =
        if let Some(output) = resolved_output(args.output.as_deref(), args.outfile.as_deref())? {
            output.to_path_buf()
        } else if should_passthrough_without_output(&runner) {
            return run_passthrough(&runner, &args, &source, None);
        } else {
            default_output_path(&source, output_type)?
        };
    if runner.has_upstream_shard_controls() || is_templated_output_path(&output) {
        return run_passthrough(&runner, &args, &source, Some(&output));
    }
    let target = derive_output(
        &output,
        args.target_prefix.as_deref(),
        args.output_basename.as_deref(),
        args.expected_splits,
    )?;
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| default_manifest_path(&target, output_type));
    let manifest_args = InitConvertArgs {
        source,
        target: target.root,
        target_prefix: target.prefix,
        output_basename: target.output_basename,
        output_type,
        expected_splits: args.expected_splits,
        window_size: args.window_size,
        manifest: manifest_path.clone(),
    };
    let manifest = convert_manifest_from_args(&manifest_args)?;
    if args.preflight_only {
        return run_job_preflight(
            &manifest_path,
            &manifest,
            None,
            None,
            runner.backend,
            runner.converter.as_deref(),
            args.json,
        );
    }
    with_manifest_lock(&manifest_path, || {
        ensure_manifest(&manifest_path, &manifest)?;
        run_convert_unlocked(RunConvertArgs {
            window: RunConvertWindowArgs {
                manifest: manifest_path.clone(),
                runner,
            },
            max_windows: args.max_windows,
        })?;
        print_verify_on_complete(
            &manifest_path,
            args.verify_load.options(args.verify_on_complete),
        )
    })
}

fn should_passthrough_without_output(runner: &ConvertRunnerArgs) -> bool {
    runner.print_supported_models || runner.remote || runner.has_upstream_shard_controls()
}

fn direct_output_type(
    runner: &ConvertRunnerArgs,
    args: &DirectConvertArgs,
    source: &Path,
) -> Result<ConvertOutputType> {
    if runner.backend == BackendKind::NativeRust || should_resolve_source_only_auto(runner, args) {
        return resolve_auto_output_type(source, args.output_type);
    }
    Ok(args.output_type)
}

fn should_resolve_source_only_auto(runner: &ConvertRunnerArgs, args: &DirectConvertArgs) -> bool {
    runner.backend == BackendKind::ExternalProcess
        && args.output_type == ConvertOutputType::Auto
        && args.output.is_none()
        && args.outfile.is_none()
        && !should_passthrough_without_output(runner)
}

fn run_print_supported_models(runner: &ConvertRunnerArgs) -> Result<()> {
    let converter = runner
        .converter
        .as_deref()
        .context("--converter is required for --print-supported-models")?;
    let command = vec![
        runner.python.clone(),
        converter.display().to_string(),
        "--print-supported-models".to_string(),
    ];
    let status = run_backend_command(
        runner.backend,
        &command,
        &ExternalProcessOptions {
            watchdog_seconds: runner.watchdog_seconds,
            max_memory_bytes: runner.max_memory.map(MemorySize::bytes),
            memory_policy: MemoryPolicy::Advisory,
        },
    )?;
    ensure!(status.success, "converter exited unsuccessfully");
    Ok(())
}

fn run_passthrough(
    runner: &ConvertRunnerArgs,
    args: &DirectConvertArgs,
    source: &Path,
    output: Option<&Path>,
) -> Result<()> {
    ensure!(
        runner.backend == crate::BackendKind::ExternalProcess,
        "direct convert passthrough currently requires --backend external-process"
    );
    ensure!(
        !args.preflight_only,
        "--preflight-only requires the resumable convert manifest path"
    );
    ensure!(
        !args.verify_load.llama_load && !args.verify_load.check_tensors,
        "load verification requires the resumable convert manifest path"
    );
    let converter = runner
        .converter
        .as_deref()
        .context("--converter is required for external conversion backend")?;
    let command = build_passthrough_command(runner, args.output_type, converter, source, output);
    println!(
        "convert_passthrough={}",
        serde_json::to_string(&serde_json::json!({ "command": command }))?
    );
    if runner.print_only {
        return Ok(());
    }
    let status = run_backend_command(
        runner.backend,
        &command,
        &ExternalProcessOptions {
            watchdog_seconds: runner.watchdog_seconds,
            max_memory_bytes: runner.max_memory.map(MemorySize::bytes),
            memory_policy: runner.memory_policy,
        },
    )?;
    ensure!(status.success, "converter exited unsuccessfully");
    Ok(())
}

fn build_passthrough_command(
    runner: &ConvertRunnerArgs,
    output_type: ConvertOutputType,
    converter: &Path,
    source: &Path,
    output: Option<&Path>,
) -> Vec<String> {
    let mut command = vec![
        runner.python.clone(),
        converter.display().to_string(),
        "--outtype".to_string(),
        output_type.as_arg().to_string(),
        "--split-max-size".to_string(),
        runner.split_max_size.clone(),
    ];
    push_passthrough_path(&mut command, "--outfile", output);
    push_passthrough_flag(&mut command, runner.vocab_only, "--vocab-only");
    push_passthrough_flag(&mut command, runner.bigendian, "--bigendian");
    push_passthrough_flag(&mut command, runner.use_temp_file, "--use-temp-file");
    push_passthrough_flag(&mut command, runner.no_lazy, "--no-lazy");
    push_passthrough_value(&mut command, "--model-name", runner.model_name.as_deref());
    push_passthrough_flag(&mut command, runner.verbose, "--verbose");
    if let Some(split_max_tensors) = runner.split_max_tensors {
        command.push("--split-max-tensors".to_string());
        command.push(split_max_tensors.to_string());
    }
    if let Some(skip_output_shards_before) = runner.skip_output_shards_before {
        command.push("--skip-output-shards-before".to_string());
        command.push(skip_output_shards_before.to_string());
    }
    if let Some(stop_output_shards_after) = runner.stop_output_shards_after {
        command.push("--stop-output-shards-after".to_string());
        command.push(stop_output_shards_after.to_string());
    }
    push_passthrough_flag(
        &mut command,
        runner.no_tensor_first_split,
        "--no-tensor-first-split",
    );
    push_passthrough_path(&mut command, "--metadata", runner.metadata.as_deref());
    push_passthrough_flag(
        &mut command,
        runner.print_supported_models,
        "--print-supported-models",
    );
    push_passthrough_flag(&mut command, runner.dry_run, "--dry-run");
    push_passthrough_flag(&mut command, runner.remote, "--remote");
    push_passthrough_flag(&mut command, runner.mmproj, "--mmproj");
    push_passthrough_flag(&mut command, runner.mtp, "--mtp");
    push_passthrough_flag(&mut command, runner.no_mtp, "--no-mtp");
    push_passthrough_flag(&mut command, runner.mistral_format, "--mistral-format");
    push_passthrough_flag(
        &mut command,
        runner.disable_mistral_community_chat_template,
        "--disable-mistral-community-chat-template",
    );
    push_passthrough_flag(
        &mut command,
        runner.sentence_transformers_dense_modules,
        "--sentence-transformers-dense-modules",
    );
    push_passthrough_flag(
        &mut command,
        runner.fuse_gate_up_exps,
        "--fuse-gate-up-exps",
    );
    push_passthrough_flag(&mut command, runner.fp8_as_q8, "--fp8-as-q8");
    push_passthrough_value(
        &mut command,
        "--target-model-dir",
        runner.target_model_dir.as_deref(),
    );
    command.push(source.display().to_string());
    command
}

fn push_passthrough_flag(command: &mut Vec<String>, enabled: bool, flag: &str) {
    if enabled {
        command.push(flag.to_string());
    }
}

fn push_passthrough_path(command: &mut Vec<String>, flag: &str, value: Option<&Path>) {
    if let Some(value) = value {
        command.push(flag.to_string());
        command.push(value.display().to_string());
    }
}

fn push_passthrough_value(command: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        command.push(flag.to_string());
        command.push(value.to_string());
    }
}

fn resolved_output<'a>(
    positional: Option<&'a Path>,
    outfile: Option<&'a Path>,
) -> Result<Option<&'a Path>> {
    match (positional, outfile) {
        (Some(_), Some(_)) => {
            anyhow::bail!("provide either positional OUTPUT or --outfile, not both")
        }
        (Some(output), None) | (None, Some(output)) => Ok(Some(output)),
        (None, None) => Ok(None),
    }
}

fn default_output_path(source: &Path, output_type: ConvertOutputType) -> Result<PathBuf> {
    let model_name = source
        .file_name()
        .and_then(|value| value.to_str())
        .with_context(|| {
            format!(
                "cannot derive default output name from {}",
                source.display()
            )
        })?;
    let parent = source.parent().unwrap_or_else(|| Path::new("."));
    Ok(parent
        .join(model_name)
        .join(format!("{model_name}-{}.gguf", output_type.as_arg())))
}

fn is_templated_output_path(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    ["{}", "{ftype}", "{outtype}", "{FTYPE}", "{OUTTYPE}"]
        .iter()
        .any(|marker| name.contains(marker))
}

#[derive(Debug)]
struct OutputLocation {
    root: PathBuf,
    prefix: String,
    output_basename: String,
}

fn derive_output(
    path: &Path,
    prefix_override: Option<&str>,
    basename_override: Option<&str>,
    expected_splits: u32,
) -> Result<OutputLocation> {
    let (root, prefix) = derive_root_and_prefix(path, prefix_override)?;
    let output_basename = match basename_override {
        Some(value) => value.to_string(),
        None => output_basename(path, expected_splits)?,
    };
    Ok(OutputLocation {
        root,
        prefix,
        output_basename,
    })
}

fn derive_root_and_prefix(path: &Path, prefix_override: Option<&str>) -> Result<(PathBuf, String)> {
    let parent = path
        .parent()
        .with_context(|| format!("path has no parent directory: {}", path.display()))?;
    if parent.as_os_str().is_empty() || parent == Path::new(".") {
        return Ok((
            PathBuf::from("."),
            prefix_override.unwrap_or("").to_string(),
        ));
    }
    let prefix = match prefix_override {
        Some(value) => value.to_string(),
        None => parent
            .file_name()
            .and_then(|value| value.to_str())
            .with_context(|| format!("cannot derive prefix from {}", path.display()))?
            .to_string(),
    };
    let root = if prefix.is_empty() {
        parent.to_path_buf()
    } else {
        parent
            .parent()
            .with_context(|| format!("path has no root above prefix: {}", path.display()))?
            .to_path_buf()
    };
    Ok((root, prefix))
}

fn output_basename(path: &Path, expected_splits: u32) -> Result<String> {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .with_context(|| format!("invalid output file name: {}", path.display()))?;
    let stem = file_name
        .strip_suffix(".gguf")
        .with_context(|| format!("output must be a GGUF path: {}", path.display()))?;
    if let Some((_, total)) = parse_split_file_name(file_name) {
        ensure!(
            total == expected_splits,
            "output split total {total} does not match --expected-splits {expected_splits}"
        );
        let (before_total, _) = stem.rsplit_once("-of-").with_context(|| {
            format!(
                "invalid split output file name after parse: {}",
                path.display()
            )
        })?;
        let (base, _) = before_total.rsplit_once('-').with_context(|| {
            format!(
                "invalid split output file name after parse: {}",
                path.display()
            )
        })?;
        return Ok(base.to_string());
    }
    Ok(stem.to_string())
}

fn default_manifest_path(target: &OutputLocation, output_type: ConvertOutputType) -> PathBuf {
    target.root.join(&target.prefix).join(format!(
        ".{}.{}.skippy-convert.json",
        target.output_basename,
        output_type.as_arg()
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn parses_short_outfile_and_upstream_auto_default() {
        let args = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "-o",
            "/repo/auto/model.gguf",
            "/models/source",
        ])
        .unwrap();

        assert_eq!(args.outfile, Some(PathBuf::from("/repo/auto/model.gguf")));
        assert_eq!(args.source, Some(PathBuf::from("/models/source")));
        assert_eq!(args.output_type, ConvertOutputType::Auto);
    }

    #[test]
    fn parses_print_supported_models_without_source_or_output() {
        let args = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--print-supported-models",
        ])
        .unwrap();

        assert!(args.runner.print_supported_models);
        assert!(args.source.is_none());
        assert!(args.output.is_none());
    }

    #[test]
    fn parses_source_without_output_for_upstream_default_filename_shape() {
        let args =
            DirectConvertArgs::try_parse_from(["convert_hf_to_gguf.py", "/models/source"]).unwrap();

        assert_eq!(args.source, Some(PathBuf::from("/models/source")));
        assert!(args.output.is_none());
        assert!(args.outfile.is_none());
        assert_eq!(args.runner.split_max_size, "0");
    }

    #[test]
    fn native_convert_rejects_unsupported_python_converter_flags() {
        let args = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--backend",
            "native-rust",
            "--vocab-only",
            "/models/source",
            "/repo/BF16/model.gguf",
        ])
        .unwrap();
        let error = prepare_convert_runner(args.runner.clone()).unwrap_err();

        assert!(error.to_string().contains("--vocab-only"));
    }

    #[test]
    fn native_convert_accepts_supported_runner_flags() {
        let args = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--backend",
            "native-rust",
            "--mtp",
            "--max-memory",
            "32G",
            "--stream-buffer-bytes",
            "1024",
            "/models/source",
            "/repo/BF16/model.gguf",
        ])
        .unwrap();

        assert!(prepare_convert_runner(args.runner.clone()).is_ok());
    }

    #[test]
    fn native_convert_rejects_conflicting_mtp_flags() {
        let args = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--backend",
            "native-rust",
            "--mtp",
            "--no-mtp",
            "/models/source",
            "/repo/BF16/model.gguf",
        ])
        .unwrap();
        let error = prepare_convert_runner(args.runner.clone()).unwrap_err();

        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn builds_passthrough_command_with_upstream_shard_controls() {
        let args = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--converter",
            "/llama.cpp/convert_hf_to_gguf.py",
            "--skip-output-shards-before",
            "2",
            "--stop-output-shards-after",
            "3",
            "--outfile",
            "/repo/BF16/model.gguf",
            "/models/source",
        ])
        .unwrap();
        let command = build_passthrough_command(
            &args.runner,
            args.output_type,
            Path::new("/llama.cpp/convert_hf_to_gguf.py"),
            args.source.as_deref().unwrap(),
            args.outfile.as_deref(),
        );

        assert!(args.runner.has_upstream_shard_controls());
        assert_eq!(
            command,
            vec![
                "python3",
                "/llama.cpp/convert_hf_to_gguf.py",
                "--outtype",
                "auto",
                "--split-max-size",
                "0",
                "--outfile",
                "/repo/BF16/model.gguf",
                "--skip-output-shards-before",
                "2",
                "--stop-output-shards-after",
                "3",
                "/models/source"
            ]
        );
    }

    #[test]
    fn passthrough_preserves_print_supported_models_with_source() {
        let args = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--converter",
            "/llama.cpp/convert_hf_to_gguf.py",
            "--print-supported-models",
            "/models/source",
        ])
        .unwrap();
        let command = build_passthrough_command(
            &args.runner,
            args.output_type,
            Path::new("/llama.cpp/convert_hf_to_gguf.py"),
            args.source.as_deref().unwrap(),
            None,
        );

        assert_eq!(
            command,
            vec![
                "python3",
                "/llama.cpp/convert_hf_to_gguf.py",
                "--outtype",
                "auto",
                "--split-max-size",
                "0",
                "--print-supported-models",
                "/models/source"
            ]
        );
    }

    #[test]
    fn direct_converter_covers_pinned_python_options() {
        let pinned = pinned_converter_options()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let local = local_converter_options();
        let missing = pinned.difference(&local).collect::<Vec<_>>();

        assert!(
            missing.is_empty(),
            "direct converter is missing pinned convert_hf_to_gguf.py options: {missing:?}"
        );
    }

    #[test]
    fn detects_templated_output_paths() {
        assert!(is_templated_output_path(Path::new(
            "/repo/model-{ftype}.gguf"
        )));
        assert!(is_templated_output_path(Path::new(
            "/repo/model-{OUTTYPE}.gguf"
        )));
        assert!(is_templated_output_path(Path::new("/repo/model-{}.gguf")));
        assert!(!is_templated_output_path(Path::new(
            "/repo/model-bf16.gguf"
        )));
    }

    #[test]
    fn builds_passthrough_command_for_templated_outfile() {
        let args = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--converter",
            "/llama.cpp/convert_hf_to_gguf.py",
            "--outtype",
            "bf16",
            "--outfile",
            "/repo/model-{ftype}.gguf",
            "/models/source",
        ])
        .unwrap();
        let command = build_passthrough_command(
            &args.runner,
            args.output_type,
            Path::new("/llama.cpp/convert_hf_to_gguf.py"),
            args.source.as_deref().unwrap(),
            args.outfile.as_deref(),
        );

        assert!(is_templated_output_path(args.outfile.as_deref().unwrap()));
        assert_eq!(
            command,
            vec![
                "python3",
                "/llama.cpp/convert_hf_to_gguf.py",
                "--outtype",
                "bf16",
                "--split-max-size",
                "0",
                "--outfile",
                "/repo/model-{ftype}.gguf",
                "/models/source"
            ]
        );
    }

    #[test]
    fn derives_output_basename_from_unsplit_path() {
        let output = Path::new("/repo/BF16/model-bf16.gguf");
        let location = derive_output(output, None, None, 3).unwrap();

        assert_eq!(location.root, PathBuf::from("/repo"));
        assert_eq!(location.prefix, "BF16");
        assert_eq!(location.output_basename, "model-bf16");
    }

    #[test]
    fn derives_current_directory_output_location() {
        let location = derive_output(Path::new("model-bf16.gguf"), None, None, 1).unwrap();

        assert_eq!(location.root, PathBuf::from("."));
        assert_eq!(location.prefix, "");
        assert_eq!(location.output_basename, "model-bf16");
    }

    #[test]
    fn derives_output_basename_from_split_path() {
        let output = Path::new("/repo/BF16/model-bf16-00001-of-00003.gguf");
        let location = derive_output(output, None, None, 3).unwrap();

        assert_eq!(location.output_basename, "model-bf16");
    }

    #[test]
    fn rejects_output_with_wrong_split_total() {
        let output = Path::new("/repo/BF16/model-bf16-00001-of-00002.gguf");
        assert!(derive_output(output, None, None, 3).is_err());
    }

    #[test]
    fn resolves_outfile_without_positional_output() {
        let outfile = Path::new("/repo/BF16/model.gguf");
        assert_eq!(resolved_output(None, Some(outfile)).unwrap(), Some(outfile));
    }

    #[test]
    fn resolves_missing_output_as_passthrough_default() {
        assert_eq!(resolved_output(None, None).unwrap(), None);
    }

    #[test]
    fn derives_default_output_path_for_source_only_resumable_convert() {
        assert_eq!(
            default_output_path(Path::new("/models/source"), ConvertOutputType::Bf16).unwrap(),
            PathBuf::from("/models/source/source-bf16.gguf")
        );
        assert_eq!(
            default_output_path(Path::new("source"), ConvertOutputType::Auto).unwrap(),
            PathBuf::from("source/source-auto.gguf")
        );
    }

    #[test]
    fn keeps_python_only_source_only_shapes_on_passthrough() {
        let remote = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--remote",
            "zai-org/GLM-5.2",
        ])
        .unwrap();
        let print_models = DirectConvertArgs::try_parse_from([
            "convert_hf_to_gguf.py",
            "--print-supported-models",
            "/models/source",
        ])
        .unwrap();

        assert!(should_passthrough_without_output(&remote.runner));
        assert!(should_passthrough_without_output(&print_models.runner));
    }

    #[test]
    fn rejects_conflicting_output_forms() {
        assert!(
            resolved_output(
                Some(Path::new("/repo/BF16/a.gguf")),
                Some(Path::new("/repo/BF16/b.gguf")),
            )
            .is_err()
        );
    }

    fn local_converter_options() -> BTreeSet<String> {
        [
            "--bigendian",
            "--disable-mistral-community-chat-template",
            "--dry-run",
            "--fp8-as-q8",
            "--fuse-gate-up-exps",
            "--metadata",
            "--mistral-format",
            "--mmproj",
            "--model-name",
            "--mtp",
            "--no-lazy",
            "--no-mtp",
            "--no-tensor-first-split",
            "--outtype",
            "--outfile",
            "--print-supported-models",
            "--remote",
            "--sentence-transformers-dense-modules",
            "--skip-output-shards-before",
            "--split-max-size",
            "--split-max-tensors",
            "--stop-output-shards-after",
            "--target-model-dir",
            "--use-temp-file",
            "--verbose",
            "--vocab-only",
        ]
        .into_iter()
        .map(ToString::to_string)
        .collect()
    }

    fn pinned_converter_options() -> Vec<String> {
        let converter = repo_root().join(".deps/llama.cpp/convert_hf_to_gguf.py");
        let source = std::fs::read_to_string(&converter)
            .unwrap_or_else(|err| panic!("read {}: {err}", converter.display()));
        source
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                line.strip_prefix("\"--")
                    .or_else(|| line.strip_prefix("'--"))
                    .and_then(|rest| rest.split_once(['"', '\'']).map(|(flag, _)| flag))
                    .map(|flag| format!("--{flag}"))
            })
            .collect()
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("crate lives under crates/skippy-quantize")
            .to_path_buf()
    }
}
