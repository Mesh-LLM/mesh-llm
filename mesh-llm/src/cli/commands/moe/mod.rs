mod formatters;
mod formatters_console;
mod formatters_json;
mod hf_jobs;

use anyhow::{bail, Context, Result};
use hf_hub::{
    FileProgress, FileStatus, Progress, ProgressEvent, ProgressHandler, RepoUploadFolderParams,
    UploadEvent, UploadPhase,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::moe::{HfJobArgs, MoeAnalyzeCommand, MoeCommand};
use crate::cli::terminal_progress::{clear_stderr_line, start_spinner, SpinnerHandle};
use crate::cli::Cli;
use crate::inference::moe;
use crate::models;
use crate::system::moe_planner::{self, MoePlanArgs};

use formatters::moe_plan_formatter;

const MICRO_PROMPTS: &[&str] = &[
    "Write a concise explanation of how a rainbow forms.",
    "Summarize the causes and effects of inflation in a paragraph.",
    "Explain why distributed systems are hard to debug.",
    "Give three practical tips for writing reliable shell scripts.",
    "Describe the water cycle for a middle school student.",
    "Compare TCP and QUIC in two short paragraphs.",
    "Explain the difference between RAM and disk storage.",
    "Write a short answer on why model evaluation matters.",
];

struct TempRootGuard(PathBuf);

impl Drop for TempRootGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct ShareUploadFileState {
    bytes_completed: u64,
    total_bytes: u64,
}

struct ShareUploadProgressState {
    spinner: Option<SpinnerHandle>,
    phase: Option<UploadPhase>,
    total_files: usize,
    total_bytes: u64,
    bytes_completed: u64,
    bytes_per_sec: Option<f64>,
    transfer_bytes_completed: u64,
    transfer_bytes: u64,
    transfer_bytes_per_sec: Option<f64>,
    completed_files: BTreeSet<String>,
    active_files: BTreeMap<String, ShareUploadFileState>,
    last_draw: Option<std::time::Instant>,
}

struct ShareUploadProgress {
    state: Mutex<ShareUploadProgressState>,
}

impl ShareUploadProgress {
    fn new() -> Self {
        Self {
            state: Mutex::new(ShareUploadProgressState {
                spinner: None,
                phase: None,
                total_files: 0,
                total_bytes: 0,
                bytes_completed: 0,
                bytes_per_sec: None,
                transfer_bytes_completed: 0,
                transfer_bytes: 0,
                transfer_bytes_per_sec: None,
                completed_files: BTreeSet::new(),
                active_files: BTreeMap::new(),
                last_draw: None,
            }),
        }
    }

    fn transition_phase(state: &mut ShareUploadProgressState, phase: &UploadPhase) {
        if state.phase.as_ref() == Some(phase) {
            return;
        }
        if let Some(mut spinner) = state.spinner.take() {
            spinner.finish();
        }
        state.phase = Some(phase.clone());
        match phase {
            UploadPhase::Preparing => {
                state.spinner = Some(start_spinner("Preparing files for upload"));
            }
            UploadPhase::CheckingUploadMode => {
                state.spinner = Some(start_spinner("Checking upload mode"));
            }
            UploadPhase::Uploading => {
                let _ = clear_stderr_line();
                eprintln!("⬆️ Uploading staged files...");
            }
            UploadPhase::Committing => {
                let done = state.completed_files.len().min(state.total_files);
                state.spinner = Some(start_spinner(&format!(
                    "Creating contribution PR ({done}/{})",
                    state.total_files
                )));
            }
        }
    }

    fn apply_file_progress(state: &mut ShareUploadProgressState, file: &FileProgress) {
        match file.status {
            FileStatus::Started | FileStatus::InProgress => {
                state.active_files.insert(
                    file.filename.clone(),
                    ShareUploadFileState {
                        bytes_completed: file.bytes_completed,
                        total_bytes: file.total_bytes,
                    },
                );
            }
            FileStatus::Complete => {
                state.completed_files.insert(file.filename.clone());
                state.active_files.remove(&file.filename);
            }
        }
    }

    fn draw(state: &mut ShareUploadProgressState, force: bool) {
        let now = std::time::Instant::now();
        if !force
            && state.last_draw.is_some_and(|last| {
                now.duration_since(last) < std::time::Duration::from_millis(700)
            })
        {
            return;
        }
        state.last_draw = Some(now);

        let done = state.completed_files.len().min(state.total_files);
        let percent = if state.total_files == 0 {
            0.0
        } else {
            (done as f64 / state.total_files as f64) * 100.0
        };
        let processing = if state.total_bytes > 0 {
            format!(
                " processed {}/{}",
                format_share_bytes(state.bytes_completed),
                format_share_bytes(state.total_bytes)
            )
        } else {
            String::new()
        };
        let transfer = if state.transfer_bytes > 0 {
            format!(
                ", uploading {}/{}",
                format_share_bytes(state.transfer_bytes_completed),
                format_share_bytes(state.transfer_bytes)
            )
        } else {
            String::new()
        };
        let speed = state
            .transfer_bytes_per_sec
            .or(state.bytes_per_sec)
            .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
            .map(|bytes_per_sec| format!(" at {}/s", format_share_bytes(bytes_per_sec as u64)))
            .unwrap_or_default();

        let _ = clear_stderr_line();
        eprintln!(
            "⬆️ Uploading {:>5.1}% [{}/{} files]{}{}{}",
            percent, done, state.total_files, processing, transfer, speed
        );

        let mut active_files: Vec<(&String, &ShareUploadFileState)> =
            state.active_files.iter().collect();
        active_files.sort_by(|(left_name, left_file), (right_name, right_file)| {
            right_file
                .bytes_completed
                .cmp(&left_file.bytes_completed)
                .then_with(|| right_file.total_bytes.cmp(&left_file.total_bytes))
                .then_with(|| left_name.cmp(right_name))
        });

        let active_count = active_files.len();
        for (name, file) in active_files.into_iter().take(8) {
            let file_percent = if file.total_bytes == 0 {
                0.0
            } else {
                (file.bytes_completed as f64 / file.total_bytes as f64) * 100.0
            };
            eprintln!(
                "   {} {:>5.1}% ({}/{})",
                display_upload_filename(name),
                file_percent,
                format_share_bytes(file.bytes_completed),
                format_share_bytes(file.total_bytes)
            );
        }
        if active_count > 8 {
            eprintln!("   … {} more tracked file(s)", active_count - 8);
        }
        let _ = std::io::stderr().flush();
    }
}

impl ProgressHandler for ShareUploadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Upload(event) = event else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            UploadEvent::Start {
                total_files,
                total_bytes,
            } => {
                state.total_files = *total_files;
                state.total_bytes = *total_bytes;
            }
            UploadEvent::Progress {
                phase,
                bytes_completed,
                total_bytes,
                bytes_per_sec,
                transfer_bytes_completed,
                transfer_bytes,
                transfer_bytes_per_sec,
                files,
            } => {
                Self::transition_phase(&mut state, phase);
                state.bytes_completed = *bytes_completed;
                if *total_bytes > 0 {
                    state.total_bytes = *total_bytes;
                }
                state.bytes_per_sec = *bytes_per_sec;
                state.transfer_bytes_completed = *transfer_bytes_completed;
                if *transfer_bytes > 0 {
                    state.transfer_bytes = *transfer_bytes;
                }
                state.transfer_bytes_per_sec = *transfer_bytes_per_sec;
                for file in files {
                    Self::apply_file_progress(&mut state, file);
                }
                if *phase == UploadPhase::Uploading {
                    Self::draw(&mut state, false);
                }
            }
            UploadEvent::FileComplete { files, phase } => {
                Self::transition_phase(&mut state, phase);
                for name in files {
                    state.completed_files.insert(name.clone());
                    state.active_files.remove(name);
                }
                if *phase == UploadPhase::Uploading {
                    Self::draw(&mut state, true);
                }
            }
            UploadEvent::Complete => {
                if let Some(mut spinner) = state.spinner.take() {
                    spinner.finish();
                }
                if state.total_files > 0 {
                    let remaining: Vec<String> = state.active_files.keys().cloned().collect();
                    state.completed_files.extend(remaining);
                    state.active_files.clear();
                    Self::draw(&mut state, true);
                }
            }
        }
    }
}

impl Drop for ShareUploadProgress {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(mut spinner) = state.spinner.take() {
                spinner.finish();
            }
        }
    }
}

fn display_upload_filename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn format_share_bytes(bytes: u64) -> String {
    const KB: f64 = 1_000.0;
    const MB: f64 = 1_000_000.0;
    const GB: f64 = 1_000_000_000.0;

    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / GB)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / MB)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / KB)
    } else {
        format!("{bytes}B")
    }
}

pub(crate) async fn dispatch_moe_command(command: &MoeCommand, cli: &Cli) -> Result<()> {
    match command {
        MoeCommand::Plan {
            model,
            ranking_file,
            json,
            max_vram,
            nodes,
            dataset_repo,
        } => {
            run_plan(
                model,
                ranking_file.as_deref(),
                *json,
                max_vram.or(cli.max_vram),
                *nodes,
                dataset_repo,
            )
            .await
        }
        MoeCommand::Analyze { command } => match command {
            MoeAnalyzeCommand::Full {
                model,
                context_size,
                n_gpu_layers,
                hf_job,
            } => run_analyze_full(model, *context_size, *n_gpu_layers, hf_job).await,
            MoeAnalyzeCommand::Micro {
                model,
                prompt_count,
                token_count,
                context_size,
                n_gpu_layers,
                hf_job,
            } => {
                run_analyze_micro(
                    model,
                    *prompt_count,
                    *token_count,
                    *context_size,
                    *n_gpu_layers,
                    hf_job,
                )
                .await
            }
        },
        MoeCommand::Share {
            model,
            ranking_file,
            dataset_repo,
            with_experts,
        } => run_share(model, ranking_file.as_deref(), dataset_repo, *with_experts).await,
    }
}

async fn run_plan(
    model: &str,
    ranking_file: Option<&Path>,
    json_output: bool,
    max_vram: Option<f64>,
    nodes: Option<usize>,
    dataset_repo: &str,
) -> Result<()> {
    if !json_output {
        eprintln!("📍 Resolving MoE model: {model}");
        if let Some(path) = ranking_file {
            eprintln!("📦 Using explicit ranking override: {}", path.display());
        } else {
            eprintln!("📦 Checking local MoE ranking cache...");
            eprintln!("☁️ Checking {dataset_repo} for published rankings...");
        }
    }
    let report = moe_planner::plan_moe(MoePlanArgs {
        model: model.to_string(),
        ranking_file: ranking_file.map(Path::to_path_buf),
        max_vram_gb: max_vram,
        nodes,
        dataset_repo: dataset_repo.to_string(),
        progress: !json_output,
    })
    .await?;
    moe_plan_formatter(json_output).render(&report)
}

async fn run_analyze_full(
    model: &str,
    context_size: u32,
    n_gpu_layers: u32,
    hf_job: &HfJobArgs,
) -> Result<()> {
    if hf_job.hf_job {
        return hf_jobs::submit_full_analyze_job(model, context_size, n_gpu_layers, hf_job).await;
    }
    let resolved = moe_planner::resolve_model_context(model).await?;
    let output_path = moe::ranking_cache_path(&resolved.path);
    let log_path = log_path_for(&resolved.path, "full-v1");
    let binary = resolve_analyze_binary()?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }

    eprintln!("📍 Model: {}", resolved.display_name);
    eprintln!("🧠 Running full-v1 MoE analysis");
    let command = vec![
        binary.to_string_lossy().to_string(),
        "-m".to_string(),
        resolved.path.display().to_string(),
        "--all-layers".to_string(),
        "--export-ranking".to_string(),
        output_path.display().to_string(),
        "-n".to_string(),
        "32".to_string(),
        "-c".to_string(),
        context_size.to_string(),
        "-ngl".to_string(),
        n_gpu_layers.to_string(),
    ];
    run_analyzer_command(&command, &log_path, "full-v1")?;
    let analysis_path = moe_planner::write_analysis_json(&resolved, &output_path, "full-v1")?;
    println!("✅ Full MoE analysis complete");
    println!("  Ranking: {}", output_path.display());
    println!("  Analysis: {}", analysis_path.display());
    println!("  Log: {}", log_path.display());
    print_submit_suggestion(&resolved.path);
    Ok(())
}

async fn run_analyze_micro(
    model: &str,
    prompt_count: usize,
    token_count: u32,
    context_size: u32,
    n_gpu_layers: u32,
    hf_job: &HfJobArgs,
) -> Result<()> {
    if hf_job.hf_job {
        return hf_jobs::submit_micro_analyze_job(
            model,
            prompt_count,
            token_count,
            context_size,
            n_gpu_layers,
            hf_job,
        )
        .await;
    }
    let resolved = moe_planner::resolve_model_context(model).await?;
    let prompt_count = prompt_count.clamp(1, MICRO_PROMPTS.len());
    let log_path = log_path_for(&resolved.path, "micro-v1");
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let binary = resolve_analyze_binary()?;
    let temp_root = std::env::temp_dir().join(format!(
        "mesh-llm-moe-micro-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir_all(&temp_root)?;
    let _temp_root_guard = TempRootGuard(temp_root.clone());

    eprintln!("📍 Model: {}", resolved.display_name);
    eprintln!(
        "🧠 Running micro-v1 MoE analysis with {} prompt(s), {} token(s)",
        prompt_count, token_count
    );
    let mut spinner = start_spinner("Running micro-v1 prompts");
    let mut logs = String::new();
    let mut totals: BTreeMap<u32, (f64, u64)> = BTreeMap::new();
    for (index, prompt) in MICRO_PROMPTS.iter().take(prompt_count).enumerate() {
        spinner.set_message(format!(
            "Running micro-v1 prompt {}/{}",
            index + 1,
            prompt_count
        ));
        let partial = temp_root.join(format!("prompt-{}.csv", index + 1));
        let command = vec![
            binary.to_string_lossy().to_string(),
            "-m".to_string(),
            resolved.path.display().to_string(),
            "--export-ranking".to_string(),
            partial.display().to_string(),
            "-n".to_string(),
            token_count.to_string(),
            "-c".to_string(),
            context_size.to_string(),
            "-ngl".to_string(),
            n_gpu_layers.to_string(),
            "--all-layers".to_string(),
            "-p".to_string(),
            (*prompt).to_string(),
        ];
        let output = Command::new(&binary)
            .args(&command[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| {
                format!(
                    "Run micro-v1 prompt {} for {}",
                    index + 1,
                    resolved.path.display()
                )
            })?;
        writeln!(&mut logs, "$ {}", shell_join(&command)).ok();
        writeln!(&mut logs, "[prompt {}]\n{}\n", index + 1, prompt).ok();
        writeln!(
            &mut logs,
            "[stdout]\n{}\n[stderr]\n{}\n",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .ok();
        if !output.status.success() {
            spinner.finish();
            fs::write(&log_path, logs)?;
            bail!("MoE micro analysis failed. Log: {}", log_path.display());
        }
        for row in read_analyze_rows(&partial)? {
            let entry = totals.entry(row.expert_id).or_insert((0.0, 0));
            entry.0 += row.gate_mass;
            entry.1 += row.selection_count;
        }
    }
    spinner.finish();
    fs::write(&log_path, logs)?;
    let artifact = moe::SharedRankingArtifact {
        kind: moe::SharedRankingKind::MicroAnalyze,
        origin: moe::SharedRankingOrigin::LocalMicroAnalyze,
        ranking: totals.keys().copied().collect::<Vec<_>>(),
        micro_prompt_count: Some(prompt_count),
        micro_tokens: Some(token_count),
        micro_layer_scope: Some(moe::MoeMicroLayerScope::All),
    };
    let mut ranking = totals.into_iter().collect::<Vec<_>>();
    ranking.sort_by(|a, b| {
        b.1 .0
            .partial_cmp(&a.1 .0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let artifact = moe::SharedRankingArtifact {
        ranking: ranking.iter().map(|(expert_id, _)| *expert_id).collect(),
        ..artifact
    };
    let wrote_cache = moe::cache_shared_ranking_if_stronger(&resolved.path, &artifact)?;
    let cache_path = moe::shared_ranking_cache_path(&resolved.path, &artifact);
    write_canonical_micro_ranking(
        &cache_path,
        &artifact,
        &ranking,
        ranking.iter().map(|(_, values)| values.0).sum::<f64>(),
    )?;
    let analysis_path = moe_planner::write_analysis_json(&resolved, &cache_path, "micro-v1")?;
    println!("✅ Micro MoE analysis complete");
    println!("  Ranking: {}", cache_path.display());
    println!("  Analysis: {}", analysis_path.display());
    if !wrote_cache {
        println!(
            "  Note: A stronger or equivalent shared ranking already exists, so this micro-v1 result was not promoted as the preferred shared artifact."
        );
    }
    println!("  Log: {}", log_path.display());
    print_submit_suggestion(&resolved.path);
    Ok(())
}

fn write_canonical_micro_ranking(
    path: &Path,
    artifact: &moe::SharedRankingArtifact,
    ranking: &[(u32, (f64, u64))],
    total_mass_sum: f64,
) -> Result<()> {
    let mut output = String::new();
    writeln!(&mut output, "# mesh-llm-moe-ranking=v1").ok();
    writeln!(&mut output, "# ranking_kind={}", artifact.kind.label()).ok();
    writeln!(&mut output, "# ranking_origin={}", artifact.origin.label()).ok();
    if let Some(prompt_count) = artifact.micro_prompt_count {
        writeln!(&mut output, "# micro_prompt_count={prompt_count}").ok();
    }
    if let Some(tokens) = artifact.micro_tokens {
        writeln!(&mut output, "# micro_tokens={tokens}").ok();
    }
    if let Some(layer_scope) = artifact.micro_layer_scope {
        let scope = match layer_scope {
            moe::MoeMicroLayerScope::First => "first",
            moe::MoeMicroLayerScope::All => "all",
        };
        writeln!(&mut output, "# micro_layer_scope={scope}").ok();
    }
    writeln!(
        &mut output,
        "expert_id,total_mass,mass_fraction,selection_count"
    )
    .ok();
    for (expert_id, (gate_mass, selection_count)) in ranking {
        let mass_fraction = if total_mass_sum > 0.0 {
            gate_mass / total_mass_sum
        } else {
            0.0
        };
        writeln!(
            &mut output,
            "{expert_id},{gate_mass:.12},{mass_fraction:.12},{selection_count}"
        )
        .ok();
    }
    fs::write(path, output).with_context(|| format!("Write {}", path.display()))?;
    Ok(())
}

fn print_submit_suggestion(model_path: &Path) {
    let Some(identity) = models::huggingface_identity_for_path(model_path) else {
        return;
    };
    println!("📤 Contribute this ranking to mesh-llm so other users can reuse it:");
    println!("  mesh-llm moe share '{}'", identity.distribution_ref());
}

async fn run_share(
    model: &str,
    ranking_file: Option<&Path>,
    dataset_repo: &str,
    with_experts: bool,
) -> Result<()> {
    let share_error = |title: &str, detail: &str| -> anyhow::Error {
        eprintln!("❌ {title}");
        eprintln!("   {detail}");
        anyhow::anyhow!("{title}: {detail}")
    };

    let resolved = moe_planner::resolve_model_context(model).await?;
    let ranking = moe_planner::local_submit_ranking(&resolved, ranking_file)?;
    moe_planner::validate_ranking(&resolved, &ranking).with_context(|| {
        format!(
            "Validate ranking {} against model {}",
            ranking.path.display(),
            resolved.display_name
        )
    })?;
    let log_path = log_path_for(&resolved.path, &ranking.analyzer_id);
    let bundle = moe_planner::build_submit_bundle(&resolved, &ranking, Some(log_path.as_path()))?;
    models::hf_token_override().ok_or_else(|| {
        share_error(
            "Missing Hugging Face token",
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running `mesh-llm moe share`.",
        )
    })?;
    let api =
        models::build_hf_tokio_api(false).context("Build Hugging Face client for MoE share")?;
    let (owner, name) = parse_dataset_repo(dataset_repo)?;
    let dataset = api.dataset(owner, name);
    let info = dataset
        .info(
            &hf_hub::RepoInfoParams::builder()
                .revision("main".to_string())
                .build(),
        )
        .await
        .with_context(|| format!("Fetch dataset info for {}", dataset_repo))?;
    let hf_hub::RepoInfo::Dataset(info) = info else {
        anyhow::bail!("Expected dataset repo info for {}", dataset_repo);
    };
    let siblings = info.siblings.as_deref().unwrap_or(&[]);
    let existing = bundle
        .dataset_paths
        .iter()
        .filter(|path| siblings.iter().any(|entry| &entry.rfilename == *path))
        .cloned()
        .collect::<Vec<_>>();

    println!("📤 MoE ranking share");
    println!("📦 {}", resolved.display_name);
    println!("   ranking: {}", ranking.path.display());
    println!("   source: {}", ranking.source.label());
    println!("☁️ Dataset contribution");
    println!("   repo: {dataset_repo}");
    println!("   prefix: {}", bundle.dataset_prefix);
    let ranking_state = classify_share_prefix(&bundle.dataset_paths, &existing);

    let temp_root = std::env::temp_dir().join(format!(
        "mesh-llm-moe-share-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir_all(&temp_root)?;
    let _temp_root_guard = TempRootGuard(temp_root.clone());

    let mut staged_files = Vec::new();
    let mut ranking_uploaded = false;
    match ranking_state {
        SharePrefixState::AlreadyPublished(existing) => {
            println!("✅ Ranking already published");
            for path in existing {
                println!("   existing: {path}");
            }
        }
        SharePrefixState::PartiallyPopulated(existing) => {
            return Err(share_error(
                "Remote ranking prefix is partially populated",
                &format!("{} already contains: {}", dataset_repo, existing.join(", ")),
            ));
        }
        SharePrefixState::New => {
            ranking_uploaded = true;
            stage_share_file(
                &temp_root,
                &format!("{}/ranking.csv", bundle.dataset_prefix),
                &bundle.ranking_path,
            )?;
            stage_share_text(
                &temp_root,
                &format!("{}/metadata.json", bundle.dataset_prefix),
                &bundle.metadata_content,
            )?;
            stage_share_text(
                &temp_root,
                &format!("{}/analysis.json", bundle.dataset_prefix),
                &bundle.analysis_content,
            )?;
            staged_files.push(format!("{}/ranking.csv", bundle.dataset_prefix));
            staged_files.push(format!("{}/metadata.json", bundle.dataset_prefix));
            staged_files.push(format!("{}/analysis.json", bundle.dataset_prefix));
            if let Some(log_path) = bundle.log_path.as_ref() {
                stage_share_file(
                    &temp_root,
                    &format!("{}/run.log", bundle.dataset_prefix),
                    log_path,
                )?;
                staged_files.push(format!("{}/run.log", bundle.dataset_prefix));
            }
        }
    }

    let mut experts_uploaded = false;
    if with_experts {
        let expert_prefix = moe_planner::canonical_expert_components_prefix_for_model(
            &resolved,
            &ranking.analyzer_id,
        )?;
        let expected_paths = expected_expert_component_paths(&expert_prefix, resolved.expert_count);
        let existing = existing_dataset_paths(&expected_paths, siblings);
        println!("🧩 Expert components");
        println!("   prefix: {expert_prefix}");
        match classify_share_prefix(&expected_paths, &existing) {
            SharePrefixState::AlreadyPublished(existing) => {
                println!("   already published");
                for path in existing {
                    println!("   existing: {path}");
                }
            }
            SharePrefixState::PartiallyPopulated(existing) => {
                return Err(share_error(
                    "Remote expert component prefix is partially populated",
                    &format!("{} already contains: {}", dataset_repo, existing.join(", ")),
                ));
            }
            SharePrefixState::New => {
                let current_exe =
                    std::env::current_exe().context("Failed to determine own binary path")?;
                let bin_dir = current_exe
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("Current executable has no parent directory"))?;
                stage_expert_components(&temp_root, &resolved, &ranking, &expert_prefix, bin_dir)?;
                staged_files.extend(expected_paths);
                experts_uploaded = true;
            }
        }
    }

    if staged_files.is_empty() {
        println!("✅ Nothing new to publish");
        return Ok(());
    }

    let (commit_message, commit_description) = build_share_commit_metadata(
        &bundle,
        &resolved,
        &ranking,
        ranking_uploaded,
        experts_uploaded,
    );

    println!("⬆️ Opening contribution PR...");
    let upload_progress: Progress = Some(Arc::new(ShareUploadProgress::new()));
    let commit = dataset
        .upload_folder(
            &RepoUploadFolderParams::builder()
                .folder_path(temp_root.clone())
                .revision("main".to_string())
                .commit_message(commit_message.clone())
                .commit_description(commit_description.clone())
                .create_pr(true)
                .progress(upload_progress)
                .build(),
        )
        .await
        .map_err(|err| {
            share_error(
                "Dataset contribution failed",
                &format!("Upload staged files to {}: {}", dataset_repo, err),
            )
        })?;
    println!("✅ Opened MoE dataset contribution");
    if let Some(commit_oid) = commit.commit_oid.as_deref() {
        println!("   commit: {commit_oid}");
    }
    if let Some(commit_url) = commit.commit_url.as_deref() {
        println!("   url: {commit_url}");
    }
    if let Some(pr_url) = commit.pr_url.as_deref() {
        println!("   pr: {pr_url}");
    }
    Ok(())
}

fn build_share_commit_metadata(
    bundle: &moe_planner::MoeSubmitBundle,
    model: &moe_planner::MoeModelContext,
    ranking: &moe_planner::ResolvedRanking,
    ranking_uploaded: bool,
    experts_uploaded: bool,
) -> (String, String) {
    match (ranking_uploaded, experts_uploaded) {
        (true, false) => (bundle.commit_message.clone(), bundle.commit_description.clone()),
        (false, true) => (
            format!(
                "Add {} {} expert components",
                model.distribution_id, ranking.analyzer_id
            ),
            format!(
                "Publish topology-independent {} expert components for {} ({})",
                ranking.analyzer_id, model.display_name, model.input
            ),
        ),
        (true, true) => (
            format!(
                "Add {} {} ranking and expert components",
                model.distribution_id, ranking.analyzer_id
            ),
            format!(
                "Publish {} ranking artifacts and topology-independent expert components for {} ({})",
                ranking.analyzer_id, model.display_name, model.input
            ),
        ),
        (false, false) => (bundle.commit_message.clone(), bundle.commit_description.clone()),
    }
}

fn expected_expert_component_paths(prefix: &str, expert_count: u32) -> Vec<String> {
    let mut paths = vec![
        format!("{prefix}/manifest.json"),
        format!("{prefix}/trunk.gguf"),
    ];
    for expert_id in 0..expert_count {
        paths.push(format!(
            "{prefix}/{}",
            moe::expert_component_filename(expert_id, expert_count)
        ));
    }
    paths
}

fn existing_dataset_paths(
    expected_paths: &[String],
    siblings: &[hf_hub::RepoSibling],
) -> Vec<String> {
    expected_paths
        .iter()
        .filter(|path| siblings.iter().any(|entry| &entry.rfilename == *path))
        .cloned()
        .collect()
}

fn stage_expert_components(
    temp_root: &Path,
    model: &moe_planner::MoeModelContext,
    ranking: &moe_planner::ResolvedRanking,
    expert_prefix: &str,
    bin_dir: &Path,
) -> Result<()> {
    println!("   extracting trunk");
    let trunk_repo_path = format!("{expert_prefix}/trunk.gguf");
    let trunk_path = temp_root.join(&trunk_repo_path);
    moe::run_extract_trunk(bin_dir, &model.path, &trunk_path)?;

    let mut spinner = start_spinner("Extracting expert components");
    let mut expert_files = Vec::with_capacity(model.expert_count as usize);
    for expert_id in 0..model.expert_count {
        spinner.set_message(format!(
            "Extracting expert {}/{}",
            expert_id + 1,
            model.expert_count
        ));
        let filename = moe::expert_component_filename(expert_id, model.expert_count);
        let repo_path = format!("{expert_prefix}/{filename}");
        let output_path = temp_root.join(&repo_path);
        moe::run_extract_expert(bin_dir, &model.path, expert_id, &output_path)?;
        expert_files.push(moe::ExpertComponentFile {
            path: filename,
            sha256: moe_planner::sha256_file(&output_path)?,
            expert_id: Some(expert_id),
        });
    }
    spinner.finish();

    let manifest = moe::ExpertComponentsManifest {
        schema_version: 1,
        source_repo: model.source_repo.clone(),
        source_revision: model.source_revision.clone(),
        distribution_id: model.distribution_id.clone(),
        analyzer_id: ranking.analyzer_id.clone(),
        ranking_sha256: moe_planner::sha256_file(&ranking.path)?,
        format: "gguf-moe-components".to_string(),
        expert_count: model.expert_count,
        expert_used_count: model.used_expert_count,
        trunk: moe::ExpertComponentFile {
            path: "trunk.gguf".to_string(),
            sha256: moe_planner::sha256_file(&trunk_path)?,
            expert_id: None,
        },
        experts: expert_files,
    };
    stage_share_text(
        temp_root,
        &format!("{expert_prefix}/manifest.json"),
        &(serde_json::to_string_pretty(&manifest)? + "\n"),
    )?;
    Ok(())
}

fn parse_dataset_repo(dataset_repo: &str) -> Result<(&str, &str)> {
    dataset_repo.split_once('/').ok_or_else(|| {
        anyhow::anyhow!("Dataset repo must look like `owner/name`, got {dataset_repo}")
    })
}

fn stage_share_text(temp_root: &Path, relative_path: &str, content: &str) -> Result<()> {
    let target = temp_root.join(relative_path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(target, content).with_context(|| format!("Write staged {}", relative_path))?;
    Ok(())
}

fn stage_share_file(temp_root: &Path, relative_path: &str, source: &Path) -> Result<()> {
    let target = temp_root.join(relative_path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    if target.exists() {
        fs::remove_file(&target).with_context(|| format!("Remove staged {}", target.display()))?;
    }
    match fs::hard_link(source, &target) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, &target)
                .with_context(|| format!("Copy {} to {}", source.display(), target.display()))?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_expert_component_paths_include_manifest_trunk_and_experts() {
        assert_eq!(
            expected_expert_component_paths("prefix/experts", 3),
            vec![
                "prefix/experts/manifest.json".to_string(),
                "prefix/experts/trunk.gguf".to_string(),
                "prefix/experts/expert-000.gguf".to_string(),
                "prefix/experts/expert-001.gguf".to_string(),
                "prefix/experts/expert-002.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn build_share_commit_metadata_mentions_expert_components() {
        let bundle = moe_planner::MoeSubmitBundle {
            dataset_prefix: "data/demo/full-v1".to_string(),
            dataset_paths: vec![],
            ranking_path: PathBuf::from("/tmp/ranking.csv"),
            metadata_content: "{}".to_string(),
            analysis_content: "{}".to_string(),
            log_path: None,
            commit_message: "ranking".to_string(),
            commit_description: "ranking-desc".to_string(),
        };
        let model = moe_planner::MoeModelContext {
            input: "demo".to_string(),
            path: PathBuf::from("/tmp/model.gguf"),
            display_name: "Demo".to_string(),
            source_repo: Some("org/repo".to_string()),
            source_revision: Some("abc".to_string()),
            distribution_id: "demo.gguf".to_string(),
            expert_count: 8,
            used_expert_count: 2,
            min_experts_per_node: 4,
            total_model_bytes: 1,
        };
        let ranking = moe_planner::ResolvedRanking {
            path: PathBuf::from("/tmp/ranking.csv"),
            metadata_path: None,
            analysis_path: None,
            analyzer_id: "full-v1".to_string(),
            source: moe_planner::RankingSource::LocalCache,
            reason: "test".to_string(),
        };

        let (message, description) =
            build_share_commit_metadata(&bundle, &model, &ranking, false, true);
        assert!(message.contains("expert components"));
        assert!(description.contains("topology-independent"));
    }

    #[test]
    fn classify_share_prefix_distinguishes_new_existing_and_partial() {
        let all = vec![
            "a/ranking.csv".to_string(),
            "a/metadata.json".to_string(),
            "a/analysis.json".to_string(),
            "a/run.log".to_string(),
        ];
        assert_eq!(classify_share_prefix(&all, &[]), SharePrefixState::New);
        assert_eq!(
            classify_share_prefix(&all, &all),
            SharePrefixState::AlreadyPublished(all.clone())
        );
        assert_eq!(
            classify_share_prefix(&all, &all[..1]),
            SharePrefixState::PartiallyPopulated(vec!["a/ranking.csv".to_string()])
        );
    }

    #[test]
    fn parse_dataset_repo_requires_owner_and_name() {
        assert!(parse_dataset_repo("meshllm/moe-rankings").is_ok());
        assert!(parse_dataset_repo("invalid").is_err());
    }
}

fn resolve_analyze_binary() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Current executable has no parent directory"))?;
    let candidates = [
        bin_dir.join("llama-moe-analyze"),
        bin_dir.join("../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-moe-analyze"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }
    bail!(
        "llama-moe-analyze not found next to {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    )
}

fn log_path_for(model_path: &Path, analyzer_id: &str) -> PathBuf {
    let stem = model_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    models::mesh_llm_cache_dir()
        .join("moe")
        .join("logs")
        .join(format!("{stem}.{analyzer_id}.log"))
}

fn run_analyzer_command(command: &[String], log_path: &Path, label: &str) -> Result<()> {
    let mut spinner = start_spinner(&format!("Running {label}"));
    let output = Command::new(&command[0])
        .args(&command[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("Run {}", command[0]))?;
    spinner.finish();
    fs::write(
        log_path,
        format!(
            "$ {}\n\n[stdout]\n{}\n[stderr]\n{}",
            shell_join(command),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )?;
    if !output.status.success() {
        bail!(
            "MoE analysis failed. Log: {}. Cause: llama-moe-analyze exited with {}",
            log_path.display(),
            output.status
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AnalyzeRow {
    expert_id: u32,
    gate_mass: f64,
    selection_count: u64,
}

fn read_analyze_rows(path: &Path) -> Result<Vec<AnalyzeRow>> {
    let content = fs::read_to_string(path).with_context(|| format!("Read {}", path.display()))?;
    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 4 {
            continue;
        }
        rows.push(AnalyzeRow {
            expert_id: parts[0].parse()?,
            gate_mass: parts[1].parse()?,
            selection_count: parts[3].parse()?,
        });
    }
    Ok(rows)
}

#[derive(Debug, PartialEq, Eq)]
enum SharePrefixState {
    New,
    AlreadyPublished(Vec<String>),
    PartiallyPopulated(Vec<String>),
}

fn classify_share_prefix(dataset_paths: &[String], existing: &[String]) -> SharePrefixState {
    if existing.len() == dataset_paths.len() {
        SharePrefixState::AlreadyPublished(existing.to_vec())
    } else if !existing.is_empty() {
        SharePrefixState::PartiallyPopulated(existing.to_vec())
    } else {
        SharePrefixState::New
    }
}

fn shell_join(command: &[String]) -> String {
    command
        .iter()
        .map(|part| {
            if part.contains([' ', '\n', '\t', '"', '\'']) {
                format!("{:?}", part)
            } else {
                part.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
