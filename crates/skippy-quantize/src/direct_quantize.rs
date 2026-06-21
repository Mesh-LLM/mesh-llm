use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;

use crate::locking::with_manifest_lock;
use crate::manifest::ensure_manifest;
use crate::preflight::run_job_preflight;
use crate::splits::{SplitWindow, parse_split_file_name, validate_split_window};
use crate::types::QuantSpec;
use crate::verify::print_verify_on_complete;
use crate::{
    InitQuantArgs, QuantRunnerArgs, RunQuantArgs, RunQuantWindowArgs, VerifyLoadArgs,
    prepare_quant_runner, quant_backend_path, quant_manifest_from_args, run_quant_unlocked,
};

#[derive(Debug, Parser)]
pub(crate) struct DirectQuantizeArgs {
    #[command(flatten)]
    runner: QuantRunnerArgs,
    #[arg(long)]
    source_prefix: Option<String>,
    #[arg(long)]
    target_prefix: Option<String>,
    #[arg(long)]
    output_basename: Option<String>,
    #[arg(long)]
    tensor_type_file: Option<PathBuf>,
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
    #[arg(long)]
    keep_split: bool,
    #[arg(long)]
    first_split: Option<u32>,
    #[arg(long)]
    last_split: Option<u32>,
    input: PathBuf,
    #[arg(value_name = "OUTPUT_OR_QUANT", num_args = 1..=3)]
    positional: Vec<String>,
}

pub(crate) fn run_direct_quantize(args: DirectQuantizeArgs) -> Result<()> {
    let mut runner = prepare_quant_runner(args.runner.clone())?;
    let positional = parse_direct_quantize_positionals(&args.positional)?;
    positional
        .quant
        .validate_recipe_requirements(args.tensor_type_file.is_some())
        .map_err(anyhow::Error::msg)?;
    apply_positional_nthreads(&mut runner, positional.nthreads)?;
    let source = derive_input(&args.input, args.source_prefix.as_deref())?;
    let window_override = args.manual_split_window(source.expected_splits)?;
    let output = if let Some(output) = positional.output.as_deref() {
        output.to_path_buf()
    } else {
        default_output_path(&args.input, &positional.quant)?
    };
    let target = derive_output(
        &output,
        args.target_prefix.as_deref(),
        args.output_basename.as_deref(),
        source.expected_splits,
    )?;
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| default_manifest_path(&target, &positional.quant));
    let manifest_args = InitQuantArgs {
        source: source.root,
        source_prefix: source.prefix,
        target: target.root,
        target_prefix: target.prefix,
        output_basename: target.output_basename,
        quant: positional.quant.clone(),
        tensor_type_file: args.tensor_type_file,
        window_size: args.window_size,
        manifest: manifest_path.clone(),
    };
    let manifest = quant_manifest_from_args(&manifest_args)?;
    if args.preflight_only {
        return run_job_preflight(
            &manifest_path,
            &manifest,
            Some((&manifest_args.source, &manifest_args.source_prefix)),
            window_override,
            runner.backend,
            quant_backend_path(&runner),
            args.json,
        );
    }
    with_manifest_lock(&manifest_path, || {
        ensure_manifest(&manifest_path, &manifest)?;
        run_quant_unlocked(RunQuantArgs {
            window: RunQuantWindowArgs {
                manifest: manifest_path.clone(),
                runner,
                json: args.json,
            },
            window_override,
            max_windows: args.max_windows,
        })?;
        print_verify_on_complete(
            &manifest_path,
            args.verify_load.options(args.verify_on_complete),
        )
    })
}

impl DirectQuantizeArgs {
    fn has_upstream_split_controls(&self) -> bool {
        self.keep_split || self.first_split.is_some() || self.last_split.is_some()
    }

    fn manual_split_window(&self, expected_splits: u32) -> Result<Option<SplitWindow>> {
        if !self.has_upstream_split_controls() {
            return Ok(None);
        }
        ensure!(
            self.keep_split,
            "--first-split and --last-split require --keep-split"
        );
        let window = SplitWindow {
            first_split: self.first_split.unwrap_or(1),
            last_split: self.last_split.unwrap_or(expected_splits),
        };
        validate_split_window(window, expected_splits)?;
        Ok(Some(window))
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DirectQuantizePositionals {
    output: Option<PathBuf>,
    quant: QuantSpec,
    nthreads: Option<u32>,
}

fn parse_direct_quantize_positionals(tokens: &[String]) -> Result<DirectQuantizePositionals> {
    match tokens {
        [quant] => Ok(DirectQuantizePositionals {
            output: None,
            quant: parse_quant_type(quant)?,
            nthreads: None,
        }),
        [first, second] => {
            if let Ok(quant) = first.parse() {
                return Ok(DirectQuantizePositionals {
                    output: None,
                    quant,
                    nthreads: Some(parse_nthreads(second)?),
                });
            }
            Ok(DirectQuantizePositionals {
                output: Some(PathBuf::from(first)),
                quant: parse_quant_type(second)?,
                nthreads: None,
            })
        }
        [output, quant, nthreads] => Ok(DirectQuantizePositionals {
            output: Some(PathBuf::from(output)),
            quant: parse_quant_type(quant)?,
            nthreads: Some(parse_nthreads(nthreads)?),
        }),
        _ => {
            anyhow::bail!("expected QUANT, QUANT NTHREADS, OUTPUT QUANT, or OUTPUT QUANT NTHREADS")
        }
    }
}

fn parse_quant_type(raw: &str) -> Result<QuantSpec> {
    raw.parse::<QuantSpec>()
        .map_err(|error| anyhow::anyhow!(error))
}

fn parse_nthreads(raw: &str) -> Result<u32> {
    raw.parse::<u32>()
        .with_context(|| format!("invalid nthreads {raw:?}"))
}

#[derive(Debug)]
struct InputLocation {
    root: PathBuf,
    prefix: String,
    expected_splits: u32,
}

#[derive(Debug)]
struct OutputLocation {
    root: PathBuf,
    prefix: String,
    output_basename: String,
}

fn derive_input(path: &Path, prefix_override: Option<&str>) -> Result<InputLocation> {
    let file_name = file_name(path)?;
    let expected_splits = if let Some((index, expected_splits)) = parse_split_file_name(file_name) {
        ensure!(
            index == 1,
            "direct quantize requires the first split shard, got shard {index}: {}",
            path.display()
        );
        expected_splits
    } else {
        ensure!(
            file_name.ends_with(".gguf"),
            "input must be a GGUF file: {}",
            path.display()
        );
        1
    };
    let (root, prefix) = derive_root_and_prefix(path, prefix_override)?;
    Ok(InputLocation {
        root,
        prefix,
        expected_splits,
    })
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
    let file_name = file_name(path)?;
    let stem = file_name
        .strip_suffix(".gguf")
        .with_context(|| format!("output must be a GGUF path: {}", path.display()))?;
    if let Some((_, total)) = parse_split_file_name(file_name) {
        ensure!(
            total == expected_splits,
            "output split total {total} does not match input split total {expected_splits}"
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

fn default_manifest_path(target: &OutputLocation, quant: &QuantSpec) -> PathBuf {
    target.root.join(&target.prefix).join(format!(
        ".{}.{}.skippy-quantize.json",
        target.output_basename,
        quant.output_name()
    ))
}

fn default_output_path(input: &Path, quant: &QuantSpec) -> Result<PathBuf> {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    Ok(parent.join(format!("ggml-model-{}.gguf", quant.output_name())))
}

fn file_name(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|value| value.to_str())
        .with_context(|| format!("invalid path file name: {}", path.display()))
}

fn apply_positional_nthreads(
    runner: &mut QuantRunnerArgs,
    positional_nthreads: Option<u32>,
) -> Result<()> {
    if let Some(positional_nthreads) = positional_nthreads {
        if let Some(flag_nthreads) = runner.nthreads {
            ensure!(
                flag_nthreads == positional_nthreads,
                "positional nthreads {positional_nthreads} conflicts with --nthreads {flag_nthreads}"
            );
        }
        runner.nthreads = Some(positional_nthreads);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::types::QuantType;

    use super::*;

    #[test]
    fn parses_quant_without_output_for_upstream_dry_run_shape() {
        let parsed = parse_direct_quantize_positionals(&["Q4_K".to_string()]).unwrap();

        assert_eq!(
            parsed,
            DirectQuantizePositionals {
                output: None,
                quant: QuantType::Q4K.into(),
                nthreads: None,
            }
        );
    }

    #[test]
    fn parses_quant_without_output_for_upstream_default_output_shape() {
        let args = DirectQuantizeArgs::try_parse_from([
            "skippy-quantize quantize",
            "/repo/model.gguf",
            "Q4_K",
        ])
        .unwrap();

        assert_eq!(args.input, PathBuf::from("/repo/model.gguf"));
        assert_eq!(
            parse_direct_quantize_positionals(&args.positional).unwrap(),
            DirectQuantizePositionals {
                output: None,
                quant: QuantType::Q4K.into(),
                nthreads: None,
            }
        );
        assert!(!args.runner.dry_run);
        assert!(!args.runner.leave_output_tensor);
    }

    #[test]
    fn parses_numeric_ftype_like_upstream_llama_quantize() {
        let args = DirectQuantizeArgs::try_parse_from([
            "skippy-quantize quantize",
            "/repo/model.gguf",
            "15",
        ])
        .unwrap();

        assert_eq!(
            parse_direct_quantize_positionals(&args.positional).unwrap(),
            DirectQuantizePositionals {
                output: None,
                quant: QuantType::Q4K.into(),
                nthreads: None,
            }
        );
    }

    #[test]
    fn parses_recipe_quant_label_for_direct_quantize() {
        let parsed = parse_direct_quantize_positionals(&["UD-Q3_K_S".to_string()]).unwrap();

        assert_eq!(parsed.output, None);
        assert_eq!(parsed.quant.base_quant(), QuantType::Q3KS);
        assert_eq!(parsed.quant.output_name(), "UD-Q3_K_S");
    }

    #[test]
    fn manual_split_window_matches_upstream_keep_split_defaults() {
        let args = DirectQuantizeArgs::try_parse_from([
            "skippy-quantize quantize",
            "--backend",
            "llama-api",
            "--keep-split",
            "--first-split",
            "3",
            "/repo/model-00001-of-00005.gguf",
            "/repo/q4/model-q4.gguf",
            "Q4_K",
        ])
        .unwrap();

        let window = args.manual_split_window(5).unwrap().unwrap();

        assert_eq!(window.first_split, 3);
        assert_eq!(window.last_split, 5);
    }

    #[test]
    fn manual_split_window_rejects_first_or_last_without_keep_split() {
        let args = DirectQuantizeArgs::try_parse_from([
            "skippy-quantize quantize",
            "--backend",
            "llama-api",
            "--last-split",
            "3",
            "/repo/model-00001-of-00005.gguf",
            "/repo/q4/model-q4.gguf",
            "Q4_K",
        ])
        .unwrap();

        let error = args.manual_split_window(5).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("--first-split and --last-split require --keep-split")
        );
    }

    #[test]
    fn manual_split_window_rejects_out_of_range_bounds() {
        let args = DirectQuantizeArgs::try_parse_from([
            "skippy-quantize quantize",
            "--backend",
            "llama-api",
            "--keep-split",
            "--first-split",
            "4",
            "--last-split",
            "6",
            "/repo/model-00001-of-00005.gguf",
            "/repo/q4/model-q4.gguf",
            "Q4_K",
        ])
        .unwrap();

        assert!(args.manual_split_window(5).is_err());
    }

    #[test]
    fn parses_quant_and_threads_without_output() {
        let parsed =
            parse_direct_quantize_positionals(&["Q4_K".to_string(), "8".to_string()]).unwrap();

        assert_eq!(
            parsed,
            DirectQuantizePositionals {
                output: None,
                quant: QuantType::Q4K.into(),
                nthreads: Some(8),
            }
        );
    }

    #[test]
    fn parses_output_quant_and_threads() {
        let parsed = parse_direct_quantize_positionals(&[
            "/repo/Q4/model.gguf".to_string(),
            "Q4_K".to_string(),
            "8".to_string(),
        ])
        .unwrap();

        assert_eq!(
            parsed,
            DirectQuantizePositionals {
                output: Some(PathBuf::from("/repo/Q4/model.gguf")),
                quant: QuantType::Q4K.into(),
                nthreads: Some(8),
            }
        );
    }

    #[test]
    fn derives_split_input_location() {
        let input = Path::new("/repo/BF16/model-00001-of-00003.gguf");
        let location = derive_input(input, None).unwrap();

        assert_eq!(location.root, PathBuf::from("/repo"));
        assert_eq!(location.prefix, "BF16");
        assert_eq!(location.expected_splits, 3);
    }

    #[test]
    fn derives_unsplit_input_location_as_one_shard() {
        let input = Path::new("/repo/BF16/model.gguf");
        let location = derive_input(input, None).unwrap();

        assert_eq!(location.root, PathBuf::from("/repo"));
        assert_eq!(location.prefix, "BF16");
        assert_eq!(location.expected_splits, 1);
    }

    #[test]
    fn derives_current_directory_input_and_output_locations() {
        let input = derive_input(Path::new("model.gguf"), None).unwrap();
        let output = derive_output(Path::new("ggml-model-Q4_K.gguf"), None, None, 1).unwrap();

        assert_eq!(input.root, PathBuf::from("."));
        assert_eq!(input.prefix, "");
        assert_eq!(input.expected_splits, 1);
        assert_eq!(output.root, PathBuf::from("."));
        assert_eq!(output.prefix, "");
        assert_eq!(output.output_basename, "ggml-model-Q4_K");
    }

    #[test]
    fn derives_default_output_path_for_no_output_quantize_shape() {
        assert_eq!(
            default_output_path(Path::new("/repo/BF16/model.gguf"), &QuantType::Q4K.into())
                .unwrap(),
            PathBuf::from("/repo/BF16/ggml-model-Q4_K.gguf")
        );
        assert_eq!(
            default_output_path(Path::new("model.gguf"), &QuantType::Q2KS.into()).unwrap(),
            PathBuf::from("ggml-model-Q2_K_S.gguf")
        );
        assert_eq!(
            default_output_path(
                Path::new("/repo/BF16/model.gguf"),
                &"UD-Q3_K_S".parse::<QuantSpec>().unwrap()
            )
            .unwrap(),
            PathBuf::from("/repo/BF16/ggml-model-UD-Q3_K_S.gguf")
        );
    }

    #[test]
    fn derives_output_basename_from_unsplit_output_path() {
        let output = Path::new("/repo/Q2_K/model-q2.gguf");
        let location = derive_output(output, None, None, 3).unwrap();

        assert_eq!(location.root, PathBuf::from("/repo"));
        assert_eq!(location.prefix, "Q2_K");
        assert_eq!(location.output_basename, "model-q2");
    }

    #[test]
    fn derives_output_basename_from_split_output_path() {
        let output = Path::new("/repo/Q2_K/model-q2-00001-of-00003.gguf");
        let location = derive_output(output, None, None, 3).unwrap();

        assert_eq!(location.output_basename, "model-q2");
    }

    #[test]
    fn rejects_non_first_input_shard() {
        let input = Path::new("/repo/BF16/model-00002-of-00003.gguf");
        assert!(derive_input(input, None).is_err());
    }
}
