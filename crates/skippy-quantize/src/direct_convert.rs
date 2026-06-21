use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;

use crate::hf_checkpoint::resolve_auto_output_type;
use crate::locking::with_manifest_lock;
use crate::manifest::ensure_manifest;
use crate::preflight::run_job_preflight;
use crate::splits::parse_split_file_name;
use crate::types::ConvertOutputType;
use crate::verify::print_verify_on_complete;
use crate::{
    ConvertRunnerArgs, InitConvertArgs, RunConvertArgs, RunConvertWindowArgs, VerifyLoadArgs,
    convert_manifest_from_args, prepare_convert_runner, run_convert_unlocked,
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
    let source = args
        .source
        .clone()
        .context("missing source path: provide MODEL")?;
    let output_type = direct_output_type(&runner, &args, &source)?;
    let output =
        if let Some(output) = resolved_output(args.output.as_deref(), args.outfile.as_deref())? {
            output.to_path_buf()
        } else {
            default_output_path(&source, output_type)?
        };
    ensure!(
        !runner.has_upstream_shard_controls(),
        "--skip-output-shards-before/--stop-output-shards-after are not accepted by direct native conversion; use convert-job/run-convert-window windowing"
    );
    ensure!(
        !is_templated_output_path(&output),
        "templated output paths are not supported by the native converter"
    );
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
            None,
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

fn direct_output_type(
    _runner: &ConvertRunnerArgs,
    args: &DirectConvertArgs,
    source: &Path,
) -> Result<ConvertOutputType> {
    resolve_auto_output_type(source, args.output_type)
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
    use super::*;

    #[test]
    fn parses_short_outfile_and_upstream_auto_default() {
        let args = DirectConvertArgs::try_parse_from([
            "skippy-quantize convert",
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
    fn parses_source_without_output_for_default_filename_shape() {
        let args = DirectConvertArgs::try_parse_from(["skippy-quantize convert", "/models/source"])
            .unwrap();

        assert_eq!(args.source, Some(PathBuf::from("/models/source")));
        assert!(args.output.is_none());
        assert!(args.outfile.is_none());
        assert_eq!(args.runner.split_max_size, "0");
    }

    #[test]
    fn native_convert_rejects_unsupported_python_converter_flags() {
        let args = DirectConvertArgs::try_parse_from([
            "skippy-quantize convert",
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
            "skippy-quantize convert",
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
            "skippy-quantize convert",
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
    fn rejects_conflicting_output_forms() {
        assert!(
            resolved_output(
                Some(Path::new("/repo/BF16/a.gguf")),
                Some(Path::new("/repo/BF16/b.gguf")),
            )
            .is_err()
        );
    }
}
