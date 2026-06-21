use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};

mod artifacts;
mod backend;
mod command_reports;
mod direct_convert;
mod direct_quantize;
mod float_convert;
mod gguf_template;
mod gguf_writer;
mod hf_checkpoint;
mod imatrix;
mod llama_load;
mod locking;
mod manifest;
mod memory_budget;
mod native_convert;
mod native_quantize;
mod plan_convert;
mod preflight;
mod quantize;
mod records;
mod residency;
mod splits;
mod tensor_map;
mod tokenizer_metadata;
mod tool_paths;
mod type_catalog;
mod types;
mod validation_commands;
mod verify;
mod verify_command;
mod window_loop;

use artifacts::{clean_spooled_window, execution_root, publish_spooled_window};
use backend::{
    BackendArgs, BackendKind, ensure_convert_backend, ensure_quant_backend, ensure_success,
};
use command_reports::{ConvertWindowPlan, QuantWindowPlan};
use direct_convert::{DirectConvertArgs, run_direct_convert};
use direct_quantize::{DirectQuantizeArgs, run_direct_quantize};
use hf_checkpoint::resolve_auto_output_type;
use llama_load::{ValidateLlamaLoadArgs, run_validate_llama_load};
use locking::with_manifest_lock;
use manifest::{
    MANIFEST_VERSION, Manifest, ensure_manifest, manifest_progress, read_manifest, write_manifest,
};
use memory_budget::{
    MemoryBudgetPlanInput, MemoryPolicy, MemorySize, effective_stream_buffer_bytes,
    native_convert_stream_working_set_bytes, print_memory_budget_plan,
};
use native_convert::{build_native_convert_command, run_native_convert};
use native_quantize::{build_native_quantize_command, run_native_quantize};
use plan_convert::{PlanConvertArgs, run_plan_convert};
use preflight::run_job_preflight;
use records::{WindowRunRecordInput, unix_timestamp_ms, write_window_record};
use residency::remove_dir_if_exists;
use splits::{
    SplitWindow, find_first_shard, next_missing_window_in_range, split_status, stage_source_window,
    validate_split_window,
};
use type_catalog::{TypeCatalogArgs, list_quants, list_tensor_types};
use types::{ConvertOutputType, JobKind, QuantSpec};
use validation_commands::{
    run_next_window as run_next_window_command, run_status as run_status_command,
    validate_splits_command, validate_tensor_types, validate_tensor_types_command,
};
use verify::{VerifyOnCompleteOptions, print_verify_on_complete};
use verify_command::verify_job as run_verify_job;
use window_loop::run_window_loop;

#[derive(Debug, Parser)]
#[command(name = "skippy-quantize")]
#[command(about = "Resumable GGUF conversion and quantization for Skippy workflows")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Backends(BackendArgs),
    ListQuants(TypeCatalogArgs),
    ListTensorTypes(TypeCatalogArgs),
    InitQuant(InitQuantArgs),
    InitConvert(InitConvertArgs),
    Convert(DirectConvertArgs),
    PlanConvert(PlanConvertArgs),
    Quantize(DirectQuantizeArgs),
    ConvertJob(ConvertJobArgs),
    QuantJob(QuantJobArgs),
    Status(StatusArgs),
    NextWindow(NextWindowArgs),
    RunConvert(RunConvertArgs),
    RunConvertWindow(RunConvertWindowArgs),
    RunQuant(RunQuantArgs),
    RunQuantWindow(RunQuantWindowArgs),
    VerifyJob(VerifyJobArgs),
    ValidateLlamaLoad(ValidateLlamaLoadArgs),
    ValidateTensorTypes(ValidateTensorTypesArgs),
    ValidateSplits(ValidateSplitsArgs),
}

#[derive(Debug, Parser)]
struct InitQuantArgs {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    source_prefix: String,
    #[arg(long)]
    target: PathBuf,
    #[arg(long)]
    target_prefix: String,
    #[arg(long)]
    output_basename: String,
    #[arg(long)]
    quant: QuantSpec,
    #[arg(long)]
    tensor_type_file: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    window_size: u32,
    #[arg(long)]
    manifest: PathBuf,
}

#[derive(Debug, Parser)]
struct InitConvertArgs {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    target: PathBuf,
    #[arg(long)]
    target_prefix: String,
    #[arg(long)]
    output_basename: String,
    #[arg(long, value_enum, default_value_t = ConvertOutputType::Bf16)]
    output_type: ConvertOutputType,
    #[arg(long)]
    expected_splits: u32,
    #[arg(long, default_value_t = 1)]
    window_size: u32,
    #[arg(long)]
    manifest: PathBuf,
}

#[derive(Debug, Parser)]
struct ConvertJobArgs {
    #[command(flatten)]
    init: InitConvertArgs,
    #[command(flatten)]
    run: ConvertJobRunArgs,
}

#[derive(Debug, Parser)]
struct ConvertJobRunArgs {
    #[command(flatten)]
    runner: ConvertRunnerArgs,
    #[arg(long)]
    max_windows: Option<u32>,
    #[arg(long)]
    preflight_only: bool,
    #[arg(long = "no-verify-on-complete", action = clap::ArgAction::SetFalse, default_value_t = true)]
    verify_on_complete: bool,
    #[command(flatten)]
    verify_load: VerifyLoadArgs,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser)]
struct QuantJobArgs {
    #[command(flatten)]
    init: InitQuantArgs,
    #[command(flatten)]
    run: QuantJobRunArgs,
}

#[derive(Debug, Parser)]
struct QuantJobRunArgs {
    #[command(flatten)]
    runner: QuantRunnerArgs,
    #[arg(long)]
    max_windows: Option<u32>,
    #[arg(long)]
    preflight_only: bool,
    #[arg(long = "no-verify-on-complete", action = clap::ArgAction::SetFalse, default_value_t = true)]
    verify_on_complete: bool,
    #[command(flatten)]
    verify_load: VerifyLoadArgs,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser, Clone)]
pub(crate) struct VerifyLoadArgs {
    #[arg(long = "verify-llama-load")]
    llama_load: bool,
    #[arg(long = "verify-llama-cli")]
    llama_cli: Option<PathBuf>,
    #[arg(long = "verify-check-tensors")]
    check_tensors: bool,
}

impl VerifyLoadArgs {
    pub(crate) fn options(&self, enabled: bool) -> VerifyOnCompleteOptions<'_> {
        VerifyOnCompleteOptions {
            enabled,
            llama_load: self.llama_load,
            llama_cli: self.llama_cli.as_deref(),
            check_tensors: self.check_tensors,
        }
    }
}

#[derive(Debug, Parser, Clone)]
struct ConvertRunnerArgs {
    #[arg(long, value_enum, default_value_t = BackendKind::NativeRust)]
    backend: BackendKind,
    #[arg(long, default_value = "0")]
    split_max_size: String,
    #[arg(long)]
    split_max_tensors: Option<u32>,
    #[arg(long)]
    skip_output_shards_before: Option<u32>,
    #[arg(long)]
    stop_output_shards_after: Option<u32>,
    #[arg(long)]
    remote: bool,
    #[arg(long)]
    vocab_only: bool,
    #[arg(long)]
    bigendian: bool,
    #[arg(long)]
    verbose: bool,
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    use_temp_file: bool,
    #[arg(long)]
    no_lazy: bool,
    #[arg(long)]
    model_name: Option<String>,
    #[arg(long)]
    no_tensor_first_split: bool,
    #[arg(long)]
    metadata: Option<PathBuf>,
    #[arg(long)]
    print_supported_models: bool,
    #[arg(long)]
    mmproj: bool,
    #[arg(long)]
    mtp: bool,
    #[arg(long)]
    no_mtp: bool,
    #[arg(long)]
    mistral_format: bool,
    #[arg(long)]
    disable_mistral_community_chat_template: bool,
    #[arg(long)]
    sentence_transformers_dense_modules: bool,
    #[arg(long)]
    fuse_gate_up_exps: bool,
    #[arg(long)]
    fp8_as_q8: bool,
    #[arg(long)]
    target_model_dir: Option<String>,
    #[arg(long)]
    spool_dir: Option<PathBuf>,
    #[arg(long)]
    keep_spool: bool,
    #[arg(long)]
    watchdog_seconds: Option<u64>,
    #[arg(long)]
    max_memory: Option<MemorySize>,
    #[arg(long, value_enum, default_value_t = MemoryPolicy::Hard)]
    memory_policy: MemoryPolicy,
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    stream_buffer_bytes: usize,
    #[arg(long)]
    print_only: bool,
    #[arg(long)]
    record_dir: Option<PathBuf>,
}

#[derive(Debug, Parser)]
struct StatusArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser)]
struct NextWindowArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser, Clone)]
struct RunConvertWindowArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[command(flatten)]
    runner: ConvertRunnerArgs,
}

#[derive(Debug, Parser)]
struct RunConvertArgs {
    #[command(flatten)]
    window: RunConvertWindowArgs,
    #[arg(long)]
    max_windows: Option<u32>,
}

#[derive(Debug, Parser, Clone)]
struct RunQuantWindowArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[command(flatten)]
    runner: QuantRunnerArgs,
}

#[derive(Debug, Parser, Clone)]
struct QuantRunnerArgs {
    #[arg(long, value_enum, default_value_t = BackendKind::LlamaApi)]
    backend: BackendKind,
    #[arg(long = "native-runtime-library", value_name = "PATH")]
    native_runtime_libraries: Vec<PathBuf>,
    #[arg(long, default_value = "/tmp/skippy-quantize-work")]
    work_dir: PathBuf,
    #[arg(long)]
    print_only: bool,
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    allow_requantize: bool,
    #[arg(long)]
    pure: bool,
    #[arg(long)]
    imatrix: Option<PathBuf>,
    #[arg(long)]
    include_weights: Vec<String>,
    #[arg(long)]
    exclude_weights: Vec<String>,
    #[arg(long)]
    output_tensor_type: Option<String>,
    #[arg(long)]
    token_embedding_type: Option<String>,
    #[arg(long)]
    tensor_type: Vec<String>,
    #[arg(long)]
    prune_layers: Option<String>,
    #[arg(long)]
    override_kv: Vec<String>,
    #[arg(long)]
    nthreads: Option<u32>,
    #[arg(long)]
    leave_output_tensor: bool,
    #[arg(long)]
    no_stage_source: bool,
    #[arg(long)]
    keep_staged_source: bool,
    #[arg(long)]
    spool_dir: Option<PathBuf>,
    #[arg(long)]
    keep_spool: bool,
    #[arg(long)]
    watchdog_seconds: Option<u64>,
    #[arg(long)]
    max_memory: Option<MemorySize>,
    #[arg(long, value_enum, default_value_t = MemoryPolicy::Hard)]
    memory_policy: MemoryPolicy,
    #[arg(long)]
    record_dir: Option<PathBuf>,
}

#[derive(Debug, Parser)]
struct RunQuantArgs {
    #[command(flatten)]
    window: RunQuantWindowArgs,
    #[arg(skip)]
    window_override: Option<SplitWindow>,
    #[arg(long)]
    max_windows: Option<u32>,
}

#[derive(Debug, Parser)]
struct ValidateTensorTypesArgs {
    file: PathBuf,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser)]
struct VerifyJobArgs {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    llama_load: bool,
    #[arg(long)]
    llama_cli: Option<PathBuf>,
    #[arg(long)]
    check_tensors: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Parser)]
struct ValidateSplitsArgs {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    prefix: String,
    #[arg(long)]
    expected_splits: Option<u32>,
    #[arg(long)]
    basename: Option<String>,
    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Backends(args) => backend::run_backends(args),
        Command::ListQuants(args) => list_quants(args),
        Command::ListTensorTypes(args) => list_tensor_types(args),
        Command::InitQuant(args) => init_quant(args),
        Command::InitConvert(args) => init_convert(args),
        Command::Convert(args) => run_direct_convert(args),
        Command::PlanConvert(args) => run_plan_convert(args),
        Command::Quantize(args) => run_direct_quantize(args),
        Command::ConvertJob(args) => convert_job(args),
        Command::QuantJob(args) => quant_job(args),
        Command::Status(args) => run_status_command(&args.manifest, args.json),
        Command::NextWindow(args) => run_next_window_command(&args.manifest, args.json),
        Command::RunConvert(args) => run_convert(args),
        Command::RunConvertWindow(args) => run_convert_window(args),
        Command::RunQuant(args) => run_quant(args),
        Command::RunQuantWindow(args) => run_quant_window(args),
        Command::VerifyJob(args) => run_verify_job(
            &args.manifest,
            args.llama_load,
            args.llama_cli.as_deref(),
            args.check_tensors,
            args.json,
        ),
        Command::ValidateLlamaLoad(args) => run_validate_llama_load(args),
        Command::ValidateTensorTypes(args) => validate_tensor_types_command(&args.file, args.json),
        Command::ValidateSplits(args) => validate_splits_command(
            &args.root,
            &args.prefix,
            args.expected_splits,
            args.basename.as_deref(),
            args.json,
        ),
    }
}

pub(crate) fn prepare_convert_runner(runner: ConvertRunnerArgs) -> Result<ConvertRunnerArgs> {
    ensure_convert_backend(runner.backend)?;
    ensure!(
        !(runner.mtp && runner.no_mtp),
        "--mtp and --no-mtp are mutually exclusive"
    );
    ensure!(
        runner.stream_buffer_bytes > 0,
        "--stream-buffer-bytes must be greater than zero"
    );
    if runner.backend == BackendKind::NativeRust {
        ensure_native_convert_runner_supported(&runner)?;
    }
    Ok(runner)
}

fn ensure_native_convert_runner_supported(runner: &ConvertRunnerArgs) -> Result<()> {
    ensure!(
        !runner.remote,
        "--remote is not supported by the native converter"
    );
    ensure!(
        !runner.vocab_only,
        "--vocab-only is not supported by the native converter"
    );
    ensure!(
        !runner.bigendian,
        "--bigendian is not supported by the native converter"
    );
    ensure!(
        !runner.use_temp_file,
        "--use-temp-file is not supported by the native converter"
    );
    ensure!(
        !runner.no_lazy,
        "--no-lazy is not supported by the native converter"
    );
    ensure!(
        runner.model_name.is_none(),
        "--model-name is not supported by the native converter"
    );
    ensure!(
        !runner.no_tensor_first_split,
        "--no-tensor-first-split is not supported by the native converter"
    );
    ensure!(
        runner.metadata.is_none(),
        "--metadata is not supported by the native converter"
    );
    ensure!(
        !runner.print_supported_models,
        "--print-supported-models is not supported by the native converter"
    );
    ensure!(
        !runner.mmproj,
        "--mmproj is not supported by the native converter"
    );
    ensure!(
        !runner.mistral_format,
        "--mistral-format is not supported by the native converter"
    );
    ensure!(
        !runner.disable_mistral_community_chat_template,
        "--disable-mistral-community-chat-template is not supported by the native converter"
    );
    ensure!(
        !runner.sentence_transformers_dense_modules,
        "--sentence-transformers-dense-modules is not supported by the native converter"
    );
    ensure!(
        !runner.fuse_gate_up_exps,
        "--fuse-gate-up-exps is not supported by the native converter"
    );
    ensure!(
        !runner.fp8_as_q8,
        "--fp8-as-q8 is not supported by the native converter"
    );
    ensure!(
        runner.target_model_dir.is_none(),
        "--target-model-dir is not supported by the native converter"
    );
    Ok(())
}

impl ConvertRunnerArgs {
    fn has_upstream_shard_controls(&self) -> bool {
        self.skip_output_shards_before.is_some() || self.stop_output_shards_after.is_some()
    }
}

pub(crate) fn prepare_quant_runner(runner: QuantRunnerArgs) -> Result<QuantRunnerArgs> {
    ensure_quant_backend(runner.backend)?;
    Ok(runner)
}

pub(crate) fn quant_backend_path(runner: &QuantRunnerArgs) -> Option<&Path> {
    match runner.backend {
        BackendKind::LlamaApi => runner
            .native_runtime_libraries
            .first()
            .map(PathBuf::as_path),
        BackendKind::NativeRust => None,
        BackendKind::SkippyAbi => runner
            .native_runtime_libraries
            .first()
            .map(PathBuf::as_path),
    }
}

fn init_quant(args: InitQuantArgs) -> Result<()> {
    let manifest = quant_manifest_from_args(&args)?;
    write_manifest(&args.manifest, &manifest)
}

fn quant_job(args: QuantJobArgs) -> Result<()> {
    let manifest = quant_manifest_from_args(&args.init)?;
    let manifest_path = args.init.manifest.clone();
    let runner = prepare_quant_runner(args.run.runner)?;
    if args.run.preflight_only {
        return run_job_preflight(
            &manifest_path,
            &manifest,
            Some((&args.init.source, &args.init.source_prefix)),
            None,
            runner.backend,
            quant_backend_path(&runner),
            args.run.json,
        );
    }
    let verify_options = args.run.verify_load.options(args.run.verify_on_complete);
    with_manifest_lock(&manifest_path, || {
        ensure_manifest(&manifest_path, &manifest)?;
        run_quant_unlocked(RunQuantArgs {
            window: RunQuantWindowArgs {
                manifest: manifest_path.clone(),
                runner,
            },
            window_override: None,
            max_windows: args.run.max_windows,
        })?;
        print_verify_on_complete(&manifest_path, verify_options)
    })
}

fn quant_manifest_from_args(args: &InitQuantArgs) -> Result<Manifest> {
    ensure!(
        args.window_size > 0,
        "--window-size must be greater than zero"
    );
    if let Some(path) = args.tensor_type_file.as_deref() {
        validate_tensor_types(path)?;
    }
    args.quant
        .validate_recipe_requirements(args.tensor_type_file.is_some())
        .map_err(anyhow::Error::msg)?;

    let source_status = split_status(&args.source, &args.source_prefix, None)
        .with_context(|| format!("scan source {}", args.source.display()))?;
    ensure!(
        source_status.expected_splits > 0,
        "source contains no split GGUF shards under prefix {:?}",
        args.source_prefix
    );
    ensure!(
        source_status.complete,
        "source split is incomplete: {}/{} shards present missing_ranges={:?}",
        source_status.completed_count,
        source_status.expected_splits,
        source_status.missing_ranges
    );

    let manifest = Manifest {
        schema_version: MANIFEST_VERSION,
        kind: JobKind::QuantizeGguf,
        source: args.source.clone(),
        source_prefix: Some(args.source_prefix.clone()),
        target: args.target.clone(),
        target_prefix: args.target_prefix.clone(),
        output_basename: args.output_basename.clone(),
        expected_splits: source_status.expected_splits,
        window_size: args.window_size,
        quant: Some(args.quant.base_quant().as_llama_name().to_string()),
        output_type: None,
        tensor_type_file: args.tensor_type_file.clone(),
    };
    Ok(manifest)
}

fn init_convert(args: InitConvertArgs) -> Result<()> {
    let manifest = convert_manifest_from_args(&args)?;
    write_manifest(&args.manifest, &manifest)
}

fn convert_job(args: ConvertJobArgs) -> Result<()> {
    let manifest = convert_manifest_from_args(&args.init)?;
    let manifest_path = args.init.manifest.clone();
    let runner = prepare_convert_runner(args.run.runner)?;
    if args.run.preflight_only {
        return run_job_preflight(
            &manifest_path,
            &manifest,
            None,
            None,
            runner.backend,
            None,
            args.run.json,
        );
    }
    let verify_options = args.run.verify_load.options(args.run.verify_on_complete);
    with_manifest_lock(&manifest_path, || {
        ensure_manifest(&manifest_path, &manifest)?;
        run_convert_unlocked(RunConvertArgs {
            window: RunConvertWindowArgs {
                manifest: manifest_path.clone(),
                runner,
            },
            max_windows: args.run.max_windows,
        })?;
        print_verify_on_complete(&manifest_path, verify_options)
    })
}

fn convert_manifest_from_args(args: &InitConvertArgs) -> Result<Manifest> {
    ensure!(
        args.expected_splits > 0,
        "--expected-splits must be greater than zero"
    );
    ensure!(
        args.window_size > 0,
        "--window-size must be greater than zero"
    );
    let manifest = Manifest {
        schema_version: MANIFEST_VERSION,
        kind: JobKind::ConvertHf,
        source: args.source.clone(),
        source_prefix: None,
        target: args.target.clone(),
        target_prefix: args.target_prefix.clone(),
        output_basename: args.output_basename.clone(),
        expected_splits: args.expected_splits,
        window_size: args.window_size,
        quant: None,
        output_type: Some(args.output_type),
        tensor_type_file: None,
    };
    Ok(manifest)
}

fn run_convert(args: RunConvertArgs) -> Result<()> {
    let manifest_path = args.window.manifest.clone();
    with_manifest_lock(&manifest_path, || run_convert_unlocked(args))
}

pub(crate) fn run_convert_unlocked(args: RunConvertArgs) -> Result<()> {
    ensure!(
        !args.window.runner.print_only,
        "run-convert does not support --print-only; use run-convert-window"
    );
    run_window_loop("convert", args.max_windows, || {
        run_convert_window_once(&args.window)
    })
}

fn run_convert_window(args: RunConvertWindowArgs) -> Result<()> {
    with_manifest_lock(&args.manifest, || {
        run_convert_window_once(&args).map(|_| ())
    })
}

fn run_convert_window_once(args: &RunConvertWindowArgs) -> Result<bool> {
    let manifest = read_manifest(&args.manifest)?;
    ensure!(
        manifest.kind == JobKind::ConvertHf,
        "run-convert-window requires a convert manifest"
    );
    let runner = prepare_convert_runner(args.runner.clone())?;
    ensure!(
        !runner.has_upstream_shard_controls(),
        "run-convert-window owns shard selection; use direct convert passthrough for --skip-output-shards-before/--stop-output-shards-after"
    );

    let progress = manifest_progress(&manifest)?;
    let Some(window) = progress.next_window else {
        println!("convert_windows_complete=true");
        return Ok(false);
    };

    let output_root = execution_root(
        &manifest.target,
        &manifest.target_prefix,
        runner.spool_dir.as_deref(),
    );
    fs::create_dir_all(&output_root)
        .with_context(|| format!("create {}", output_root.display()))?;
    let output_prefix = output_root.join(format!("{}.gguf", manifest.output_basename));
    let command = match runner.backend {
        BackendKind::NativeRust => {
            build_native_convert_command(&runner, &manifest, &output_prefix, window)
        }
        BackendKind::LlamaApi | BackendKind::SkippyAbi => {
            unreachable!("unsupported convert backend checked earlier")
        }
    };
    let plan = ConvertWindowPlan {
        first_split: window.first_split,
        last_split: window.last_split,
        output_prefix,
        command,
    };
    println!("convert_window={}", serde_json::to_string(&plan)?);
    let stream_buffer_bytes = (runner.backend == BackendKind::NativeRust)
        .then(|| effective_stream_buffer_bytes(runner.stream_buffer_bytes, runner.max_memory))
        .transpose()?;
    let native_output_type = if runner.backend == BackendKind::NativeRust {
        manifest
            .output_type
            .map(|output_type| resolve_auto_output_type(&manifest.source, output_type))
            .transpose()?
    } else {
        manifest.output_type
    };
    let estimated_stream_working_set_bytes = stream_buffer_bytes
        .map(|buffer_size| native_convert_stream_working_set_bytes(buffer_size, native_output_type))
        .transpose()?;
    print_memory_budget_plan(MemoryBudgetPlanInput {
        kind: "convert",
        backend: runner.backend.as_str(),
        max_memory: runner.max_memory,
        memory_policy: runner.memory_policy,
        watchdog_seconds: runner.watchdog_seconds,
        window,
        stream_buffer_bytes,
        estimated_stream_working_set_bytes,
        llama_quantize_env_bytes: None,
    })?;
    if runner.print_only {
        return Ok(true);
    }
    clean_spooled_window(
        runner.spool_dir.as_deref(),
        &manifest.target_prefix,
        &manifest.output_basename,
        manifest.expected_splits,
        window,
    )?;
    let started_unix_ms = unix_timestamp_ms();
    let started = Instant::now();
    let status = match runner.backend {
        BackendKind::NativeRust => {
            run_native_convert(&runner, &manifest, window, &plan.output_prefix)?
        }
        BackendKind::LlamaApi | BackendKind::SkippyAbi => {
            unreachable!("unsupported convert backend checked earlier")
        }
    };
    let duration_ms = started.elapsed().as_millis();
    write_window_record(
        runner.record_dir.as_deref(),
        WindowRunRecordInput {
            schema_version: MANIFEST_VERSION,
            kind: manifest.kind,
            command: &plan.command,
            output_prefix: &plan.output_prefix,
            window,
            status,
            duration_ms,
            started_unix_ms,
        },
    )?;
    ensure_success(status, &plan.command)?;
    if !runner.dry_run {
        publish_spooled_window(
            runner.spool_dir.as_deref(),
            &manifest.target,
            &manifest.target_prefix,
            &manifest.output_basename,
            manifest.expected_splits,
            window,
            args.runner.keep_spool,
        )?;
    }
    Ok(true)
}

fn run_quant(args: RunQuantArgs) -> Result<()> {
    let manifest_path = args.window.manifest.clone();
    with_manifest_lock(&manifest_path, || run_quant_unlocked(args))
}

pub(crate) fn run_quant_unlocked(args: RunQuantArgs) -> Result<()> {
    ensure!(
        !args.window.runner.print_only,
        "run-quant does not support --print-only; use run-quant-window"
    );
    run_window_loop("quant", args.max_windows, || {
        run_quant_window_once(&args.window, args.window_override)
    })
}

fn run_quant_window(args: RunQuantWindowArgs) -> Result<()> {
    with_manifest_lock(&args.manifest, || {
        run_quant_window_once(&args, None).map(|_| ())
    })
}

fn run_quant_window_once(
    args: &RunQuantWindowArgs,
    window_override: Option<SplitWindow>,
) -> Result<bool> {
    let manifest = read_manifest(&args.manifest)?;
    ensure!(
        manifest.kind == JobKind::QuantizeGguf,
        "run-quant-window requires a quantize manifest"
    );
    let runner = prepare_quant_runner(args.runner.clone())?;

    let progress = manifest_progress(&manifest)?;
    let window = if let Some(requested) = window_override {
        validate_split_window(requested, manifest.expected_splits)?;
        let Some(window) = next_missing_window_in_range(&progress.missing_ranges, requested) else {
            println!(
                "quant_requested_window_complete={}",
                serde_json::to_string(&requested)?
            );
            return Ok(false);
        };
        window
    } else if let Some(window) = progress.next_window {
        window
    } else {
        println!("quant_windows_complete=true");
        return Ok(false);
    };

    let source_prefix = manifest
        .source_prefix
        .as_deref()
        .context("quantize manifest is missing source_prefix")?;
    let first_source_shard = find_first_shard(&manifest.source, source_prefix)?;
    let stage_path = runner.work_dir.join("source-window");
    let staged_first_shard = if runner.no_stage_source {
        first_source_shard
    } else {
        stage_source_window(
            &manifest.source,
            source_prefix,
            &first_source_shard,
            &stage_path,
            window,
            manifest.expected_splits,
        )?
    };

    let output_root = execution_root(
        &manifest.target,
        &manifest.target_prefix,
        runner.spool_dir.as_deref(),
    );
    fs::create_dir_all(&output_root)
        .with_context(|| format!("create {}", output_root.display()))?;
    let output_prefix = output_root.join(format!("{}.gguf", manifest.output_basename));
    let command = match runner.backend {
        BackendKind::LlamaApi | BackendKind::SkippyAbi => build_native_quantize_command(
            &runner,
            &manifest,
            &staged_first_shard,
            &output_prefix,
            window,
        )?,
        BackendKind::NativeRust => {
            unreachable!("unsupported quant backend checked earlier")
        }
    };
    let plan = QuantWindowPlan {
        first_split: window.first_split,
        last_split: window.last_split,
        staged_first_shard,
        output_prefix,
        command,
    };
    println!("quant_window={}", serde_json::to_string(&plan)?);
    print_memory_budget_plan(MemoryBudgetPlanInput {
        kind: "quant",
        backend: runner.backend.as_str(),
        max_memory: runner.max_memory,
        memory_policy: runner.memory_policy,
        watchdog_seconds: runner.watchdog_seconds,
        window,
        stream_buffer_bytes: None,
        estimated_stream_working_set_bytes: None,
        llama_quantize_env_bytes: runner.max_memory.map(MemorySize::bytes),
    })?;

    if runner.print_only {
        return Ok(true);
    }
    clean_spooled_window(
        runner.spool_dir.as_deref(),
        &manifest.target_prefix,
        &manifest.output_basename,
        manifest.expected_splits,
        window,
    )?;
    let started_unix_ms = unix_timestamp_ms();
    let started = Instant::now();
    let status = match runner.backend {
        BackendKind::LlamaApi | BackendKind::SkippyAbi => run_native_quantize(
            &runner,
            &manifest,
            &plan.staged_first_shard,
            &plan.output_prefix,
            window,
        )?,
        BackendKind::NativeRust => {
            unreachable!("unsupported quant backend checked earlier")
        }
    };
    let duration_ms = started.elapsed().as_millis();
    write_window_record(
        runner.record_dir.as_deref(),
        WindowRunRecordInput {
            schema_version: MANIFEST_VERSION,
            kind: manifest.kind,
            command: &plan.command,
            output_prefix: &plan.output_prefix,
            window,
            status,
            duration_ms,
            started_unix_ms,
        },
    )?;
    ensure_success(status, &plan.command)?;
    if !runner.dry_run {
        publish_spooled_window(
            runner.spool_dir.as_deref(),
            &manifest.target,
            &manifest.target_prefix,
            &manifest.output_basename,
            manifest.expected_splits,
            window,
            args.runner.keep_spool,
        )?;
    }
    if !runner.no_stage_source && !runner.keep_staged_source {
        remove_dir_if_exists(&stage_path)?;
        println!("stage_source_cleanup path={}", stage_path.display());
    }
    Ok(true)
}
