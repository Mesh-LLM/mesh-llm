use super::publish::ensure_package_calibration_benchmark;
use super::{
    log_path_for, read_analyze_rows, resolve_analyze_binary, run_analyzer_command, shell_join,
    TempRootGuard, FULL_ANALYZE_TOKEN_COUNT, MICRO_PROMPTS,
};
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::terminal_progress::start_spinner;
use crate::inference::moe;
use crate::models;
use crate::system::moe_planner;

pub(super) async fn run_analyze_full(
    model: &str,
    context_size: u32,
    n_gpu_layers: u32,
) -> Result<()> {
    let resolved = moe_planner::resolve_model_context(model).await?;
    eprintln!("📍 Model: {}", resolved.display_name);
    eprintln!("🧠 Running full-v1 MoE analysis");
    let artifacts = run_local_full_analysis(&resolved, context_size, n_gpu_layers)?;
    ensure_package_calibration_benchmark(&resolved, &artifacts.ranking).await?;
    println!("✅ Full MoE analysis complete");
    println!("  Ranking: {}", artifacts.ranking_path.display());
    println!("  Analysis: {}", artifacts.analysis_path.display());
    println!("  Log: {}", artifacts.log_path.display());
    println!(
        "  Package cache: {}",
        moe::package_cache_root_dir(&resolved.path).display()
    );
    println!(
        "  Variant cache: {}",
        moe::package_cache_variant_dir(&resolved.path).display()
    );
    print_submit_suggestion(&resolved.path);
    Ok(())
}

pub(super) async fn run_analyze_micro(
    model: &str,
    prompt_count: usize,
    token_count: u32,
    context_size: u32,
    n_gpu_layers: u32,
) -> Result<()> {
    let resolved = moe_planner::resolve_model_context(model).await?;
    let prompt_count = prompt_count.clamp(1, MICRO_PROMPTS.len());
    let log_path = log_path_for(&resolved.path, "micro-v1");
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let binary = resolve_analyze_binary()?;
    let temp_root_base = moe::package_cache_temp_root(&resolved.path);
    fs::create_dir_all(&temp_root_base)?;
    let temp_root = temp_root_base.join(format!(
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
    let ranking = moe_planner::ResolvedRanking {
        path: cache_path.clone(),
        metadata_path: None,
        analysis_path: Some(analysis_path.clone()),
        analyzer_id: "micro-v1".to_string(),
        source: moe_planner::RankingSource::LocalCache,
        reason: "local analysis artifact".to_string(),
    };
    sync_local_package_cache(&resolved, &ranking, Some(&log_path))?;
    println!("✅ Micro MoE analysis complete");
    println!("  Ranking: {}", cache_path.display());
    println!("  Analysis: {}", analysis_path.display());
    if !wrote_cache {
        println!(
            "  Note: A stronger or equivalent shared ranking already exists, so this micro-v1 result was not promoted as the preferred shared artifact."
        );
    }
    println!("  Log: {}", log_path.display());
    println!(
        "  Package cache: {}",
        moe::package_cache_root_dir(&resolved.path).display()
    );
    println!(
        "  Variant cache: {}",
        moe::package_cache_variant_dir(&resolved.path).display()
    );
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
    println!("  mesh-llm moe publish '{}'", identity.distribution_ref());
}

#[derive(Clone)]
pub(super) struct FullAnalyzeArtifacts {
    pub(super) ranking: moe_planner::ResolvedRanking,
    pub(super) ranking_path: PathBuf,
    pub(super) analysis_path: PathBuf,
    pub(super) log_path: PathBuf,
}

pub(super) fn full_analyze_artifacts(model: &moe_planner::MoeModelContext) -> FullAnalyzeArtifacts {
    let ranking_path = moe::package_cache_ranking_path(&model.path);
    let analysis_path = moe::package_cache_analysis_path(&model.path);
    let log_path = moe::package_cache_run_log_path(&model.path);
    FullAnalyzeArtifacts {
        ranking: moe_planner::ResolvedRanking {
            path: ranking_path.clone(),
            metadata_path: None,
            analysis_path: Some(analysis_path.clone()),
            analyzer_id: "full-v1".to_string(),
            source: moe_planner::RankingSource::LocalCache,
            reason: "local full analysis artifact".to_string(),
        },
        ranking_path,
        analysis_path,
        log_path,
    }
}

pub(super) fn has_complete_full_analyze_artifacts(model: &moe_planner::MoeModelContext) -> bool {
    let artifacts = full_analyze_artifacts(model);
    artifacts.ranking_path.exists()
        && artifacts.analysis_path.exists()
        && artifacts.log_path.exists()
}

pub(super) fn run_local_full_analysis(
    model: &moe_planner::MoeModelContext,
    context_size: u32,
    n_gpu_layers: u32,
) -> Result<FullAnalyzeArtifacts> {
    let artifacts = full_analyze_artifacts(model);
    let binary = resolve_analyze_binary()?;
    if let Some(parent) = artifacts.ranking_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = artifacts.log_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let command = vec![
        binary.to_string_lossy().to_string(),
        "-m".to_string(),
        model.path.display().to_string(),
        "--all-layers".to_string(),
        "--export-ranking".to_string(),
        artifacts.ranking_path.display().to_string(),
        "-n".to_string(),
        FULL_ANALYZE_TOKEN_COUNT.to_string(),
        "-c".to_string(),
        context_size.to_string(),
        "-ngl".to_string(),
        n_gpu_layers.to_string(),
    ];
    run_analyzer_command(&command, &artifacts.log_path, "full-v1")?;
    let analysis_path =
        moe_planner::write_analysis_json(model, &artifacts.ranking_path, "full-v1")?;
    let ranking = moe_planner::ResolvedRanking {
        analysis_path: Some(analysis_path.clone()),
        ..artifacts.ranking.clone()
    };
    sync_local_package_cache(model, &ranking, Some(&artifacts.log_path))?;
    Ok(FullAnalyzeArtifacts {
        ranking,
        ranking_path: artifacts.ranking_path,
        analysis_path,
        log_path: artifacts.log_path,
    })
}

pub(super) fn sync_local_package_cache(
    model: &moe_planner::MoeModelContext,
    ranking: &moe_planner::ResolvedRanking,
    log_path: Option<&Path>,
) -> Result<moe_planner::MoeSubmitBundle> {
    let bundle = moe_planner::build_submit_bundle(model, ranking, log_path)?;

    let meshllm_path = moe::package_cache_meshllm_path(&model.path);
    let existing_meshllm = if meshllm_path.exists() {
        Some(
            fs::read_to_string(&meshllm_path)
                .with_context(|| format!("Read {}", meshllm_path.display()))?,
        )
    } else {
        None
    };
    let meshllm = moe_planner::build_meshllm_descriptor(
        existing_meshllm.as_deref(),
        model,
        &bundle.manifest_repo_path,
    )?;
    if let Some(parent) = meshllm_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    fs::write(
        &meshllm_path,
        serde_json::to_string_pretty(&meshllm)? + "\n",
    )
    .with_context(|| format!("Write {}", meshllm_path.display()))?;

    let package_ranking_path = moe::package_cache_ranking_path(&model.path);
    if let Some(parent) = package_ranking_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    if ranking.path != package_ranking_path {
        fs::copy(&ranking.path, &package_ranking_path).with_context(|| {
            format!(
                "Copy ranking {} to {}",
                ranking.path.display(),
                package_ranking_path.display()
            )
        })?;
    }

    let package_analysis_path = moe::package_cache_analysis_path(&model.path);
    moe_planner::write_package_analysis_json(model, ranking, &package_analysis_path)?;

    let package_log_path = moe::package_cache_run_log_path(&model.path);
    if let Some(log_path) = log_path.filter(|path| path.exists()) {
        if let Some(parent) = package_log_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
        }
        if log_path != package_log_path {
            fs::copy(log_path, &package_log_path).with_context(|| {
                format!(
                    "Copy run log {} to {}",
                    log_path.display(),
                    package_log_path.display()
                )
            })?;
        }
    } else if package_log_path.exists() {
        fs::remove_file(&package_log_path)
            .with_context(|| format!("Remove stale {}", package_log_path.display()))?;
    }

    Ok(bundle)
}
