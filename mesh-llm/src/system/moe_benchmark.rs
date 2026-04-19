use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::Digest;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;
use tokio::process::{Child, Command as TokioCommand};
use tokio::task::JoinSet;

use crate::inference::moe;
use crate::models::{self, catalog};
use crate::network::router;
use crate::system::benchmark_prompts::{
    self, PromptCorpusEntry, PromptCorpusSummary, PromptImportSource,
};
use crate::system::moe_planner;

const PACKAGE_BENCHMARK_VERSION: u32 = 3;
const PACKAGE_BENCHMARK_MAX_TOKENS: u32 = 128;
const PACKAGE_BENCHMARK_CONTEXT_SIZE: u32 = 4096;
const PACKAGE_BENCHMARK_GPU_LAYERS: u32 = 99;
const PACKAGE_BENCHMARK_QUALITY_FLOOR: f64 = 0.95;
const PACKAGE_BENCHMARK_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
const PACKAGE_BENCHMARK_HEALTH_TIMEOUT: Duration = Duration::from_secs(600);
const PACKAGE_BENCHMARK_ASSEMBLY_CONCURRENCY_CAP: usize = 4;
const PACKAGE_BENCHMARK_REQUEST_CONCURRENCY_CAP: usize = 4;
const PACKAGE_BENCHMARK_MT_BENCH_LIMIT: usize = 80;
const PACKAGE_BENCHMARK_IFEVAL_LIMIT: usize = 128;
const PACKAGE_BENCHMARK_GSM8K_LIMIT: usize = 128;
const PACKAGE_BENCHMARK_HUMANEVAL_LIMIT: usize = 164;

#[derive(Clone, Debug)]
pub(crate) struct MoeRankingBenchmarkArgs {
    pub model: String,
    pub nodes: usize,
    pub overlap: usize,
    pub min_experts: Option<u32>,
    pub variants: Vec<BenchmarkVariant>,
    pub analyze_ranking: Option<PathBuf>,
    pub prompts: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeMicroAnalyzeBenchmarkArgs {
    pub model: String,
    pub min_experts: Option<u32>,
    pub analyze_ranking: Option<PathBuf>,
    pub prompts: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeGroupingBenchmarkArgs {
    pub model: String,
    pub nodes: usize,
    pub overlap: usize,
    pub min_experts: Option<u32>,
    pub analyze_ranking: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeModelMatrixBenchmarkArgs {
    pub models: Vec<String>,
    pub nodes: usize,
    pub overlap: usize,
    pub min_experts: Option<u32>,
    pub prompts: Option<PathBuf>,
    pub analyze_ranking_dir: Option<PathBuf>,
    pub output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BenchmarkVariant {
    Analyze,
}

#[derive(Clone)]
struct ResolvedBenchmarkModel {
    input: String,
    path: PathBuf,
    name: String,
    architecture: String,
    info: moe::GgufMoeInfo,
    min_experts: u32,
}

#[derive(Clone, Debug)]
struct AnalyzeExpertMass {
    expert_id: u32,
    gate_mass: f64,
    mass_pct: f64,
    selection_count: u64,
}

#[derive(Clone, Debug)]
struct AnalyzeMassProfile {
    entries: Vec<AnalyzeExpertMass>,
    mass_by_expert: HashMap<u32, f64>,
    total_mass: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkModelInfo {
    input: String,
    resolved_path: String,
    name: String,
    architecture: String,
    expert_count: u32,
    expert_used_count: u32,
}

#[derive(Debug, Serialize)]
struct BenchmarkConfig {
    nodes: usize,
    overlap: usize,
    min_experts: u32,
}

#[derive(Debug, Serialize)]
struct AssignmentReport {
    node: usize,
    expert_count: usize,
    shared: usize,
    unique: usize,
    experts: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct VariantReport {
    name: &'static str,
    ranking_source: String,
    ranking_len: usize,
    assignments: Vec<AssignmentReport>,
}

#[derive(Debug, Serialize)]
struct MoeRankingBenchmarkReport {
    benchmark: &'static str,
    model: BenchmarkModelInfo,
    config: BenchmarkConfig,
    prompt_corpus: Option<PromptCorpusSummary>,
    variants: Vec<VariantReport>,
}

#[derive(Debug, Serialize)]
struct GroupingStrategyReport {
    name: &'static str,
    ranking_source: String,
    grouping_mode: &'static str,
    replicated_experts: usize,
    shared_mass_pct: f64,
    mean_node_mass_pct: f64,
    min_node_mass_pct: f64,
    max_node_mass_pct: f64,
    node_mass_imbalance_pct: f64,
    assignments: Vec<AssignmentReport>,
}

#[derive(Debug, Serialize)]
struct MoeGroupingBenchmarkReport {
    benchmark: &'static str,
    model: BenchmarkModelInfo,
    config: BenchmarkConfig,
    analyze_ranking_source: String,
    strategies: Vec<GroupingStrategyReport>,
}

#[derive(Debug, Serialize)]
struct MicroAnalyzeExperimentReport {
    name: String,
    prompt_count: usize,
    tokens: u32,
    all_layers: bool,
    runtime_seconds: f64,
    spearman_rank_correlation: f64,
    recall_at_min_experts: f64,
    weighted_recall_at_min_experts: f64,
    captures_top_truth_expert: bool,
    ranking_preview: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct MoeMicroAnalyzeBenchmarkReport {
    benchmark: &'static str,
    model: BenchmarkModelInfo,
    min_experts: u32,
    analyze_ranking_source: String,
    prompt_corpus: Option<PromptCorpusSummary>,
    experiments: Vec<MicroAnalyzeExperimentReport>,
}

#[derive(Debug, Serialize)]
struct MoeModelMatrixModelReport {
    model: BenchmarkModelInfo,
    ranking: MoeRankingBenchmarkReport,
    grouping: MoeGroupingBenchmarkReport,
    micro_analyze: MoeMicroAnalyzeBenchmarkReport,
}

#[derive(Debug, Serialize)]
struct MoeModelMatrixReport {
    benchmark: &'static str,
    prompt_corpus: Option<PromptCorpusSummary>,
    models: Vec<MoeModelMatrixModelReport>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PackageCalibrationBenchmarkReport {
    version: u32,
    corpora: Vec<PackageCalibrationCorpus>,
    baseline: &'static str,
    metric: &'static str,
    quality_floor: f64,
    candidates: Vec<PackageCalibrationCandidateReport>,
    recommended_min_experts_per_node: u32,
}

#[derive(Debug, Serialize)]
struct PackageCalibrationCorpus {
    source: &'static str,
    dataset: &'static str,
    prompt_count: usize,
    max_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct PackageCalibrationBaselineCache {
    version: u32,
    dataset: String,
    prompt_hash: String,
    max_tokens: u32,
    context_size: u32,
    n_gpu_layers: u32,
    outputs: Vec<String>,
}

struct LoadedCalibrationCorpus {
    source: PromptImportSource,
    prompts: Vec<PromptCorpusEntry>,
    baseline_outputs: Vec<String>,
}

struct LocalServerRunner {
    base_url: String,
    model_path: PathBuf,
    log_path: PathBuf,
    child: Child,
}

#[derive(Clone, Debug, Serialize)]
struct PackageCalibrationCandidateReport {
    min_experts_per_node: u32,
    node_count: usize,
    mean_score: f64,
    worst_node_score: f64,
    node_scores: Vec<f64>,
    corpora: Vec<PackageCalibrationCandidateCorpusReport>,
}

#[derive(Clone, Debug, Serialize)]
struct PackageCalibrationCandidateCorpusReport {
    source: &'static str,
    dataset: &'static str,
    prompt_count: usize,
    mean_score: f64,
    worst_node_score: f64,
    node_scores: Vec<f64>,
}

#[derive(Clone, Copy)]
struct PackageCalibrationCorpusSpec {
    source: PromptImportSource,
    limit: usize,
}

struct CleanupDir {
    path: PathBuf,
}

impl CleanupDir {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for CleanupDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn create_package_temp_dir(model_path: &Path, prefix: &str) -> Result<CleanupDir> {
    let temp_root = moe::package_cache_temp_root(model_path);
    fs::create_dir_all(&temp_root)
        .with_context(|| format!("Create benchmark temp root {}", temp_root.display()))?;
    let path = temp_root.join(format!(
        "{prefix}{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir_all(&path).with_context(|| format!("Create temp dir {}", path.display()))?;
    Ok(CleanupDir { path })
}

fn package_benchmark_corpus_specs() -> &'static [PackageCalibrationCorpusSpec] {
    &[
        PackageCalibrationCorpusSpec {
            source: PromptImportSource::MtBench,
            limit: PACKAGE_BENCHMARK_MT_BENCH_LIMIT,
        },
        PackageCalibrationCorpusSpec {
            source: PromptImportSource::IfEval,
            limit: PACKAGE_BENCHMARK_IFEVAL_LIMIT,
        },
        PackageCalibrationCorpusSpec {
            source: PromptImportSource::Gsm8k,
            limit: PACKAGE_BENCHMARK_GSM8K_LIMIT,
        },
        PackageCalibrationCorpusSpec {
            source: PromptImportSource::Humaneval,
            limit: PACKAGE_BENCHMARK_HUMANEVAL_LIMIT,
        },
    ]
}

pub(crate) async fn run_moe_ranking_benchmark(args: MoeRankingBenchmarkArgs) -> Result<()> {
    validate_nodes(args.nodes)?;
    if args.variants.is_empty() {
        bail!("--variants must include at least one ranking source");
    }

    let model = resolve_benchmark_model(&args.model, args.min_experts).await?;
    let prompt_corpus = load_prompt_summary(args.prompts.as_deref())?;
    let report = build_ranking_report(
        &model,
        args.nodes,
        args.overlap,
        &args.variants,
        args.analyze_ranking.as_deref(),
        prompt_corpus,
    )?;
    write_json_report(&report, args.output.as_deref(), "MoE ranking benchmark")?;
    Ok(())
}

pub(crate) async fn run_moe_micro_analyze_benchmark(
    args: MoeMicroAnalyzeBenchmarkArgs,
) -> Result<()> {
    let model = resolve_benchmark_model(&args.model, args.min_experts).await?;
    let prompt_corpus = load_prompt_summary(args.prompts.as_deref())?;
    let report =
        build_micro_analyze_report(&model, args.analyze_ranking.as_deref(), prompt_corpus)?;
    write_json_report(
        &report,
        args.output.as_deref(),
        "MoE micro-analyze benchmark",
    )?;
    Ok(())
}

pub(crate) async fn run_moe_grouping_benchmark(args: MoeGroupingBenchmarkArgs) -> Result<()> {
    validate_nodes(args.nodes)?;
    let model = resolve_benchmark_model(&args.model, args.min_experts).await?;
    let report = build_grouping_report(
        &model,
        args.nodes,
        args.overlap,
        args.analyze_ranking.as_deref(),
    )?;
    write_json_report(&report, args.output.as_deref(), "MoE grouping benchmark")?;
    Ok(())
}

pub(crate) async fn run_moe_model_matrix_benchmark(
    args: MoeModelMatrixBenchmarkArgs,
) -> Result<()> {
    validate_nodes(args.nodes)?;
    if args.models.is_empty() {
        bail!("--model must be provided at least once");
    }

    let prompt_corpus = load_prompt_summary(args.prompts.as_deref())?;
    let mut reports = Vec::with_capacity(args.models.len());
    for model_spec in &args.models {
        let model = resolve_benchmark_model(model_spec, args.min_experts).await?;
        let explicit_analyze = explicit_analyze_path(&model, args.analyze_ranking_dir.as_deref());
        let ensured_analyze = ensure_full_analyze_ranking(&model, explicit_analyze.as_deref())?;
        let ranking = build_ranking_report(
            &model,
            args.nodes,
            args.overlap,
            &[BenchmarkVariant::Analyze],
            Some(ensured_analyze.as_path()),
            prompt_corpus.clone(),
        )?;
        let grouping = build_grouping_report(
            &model,
            args.nodes,
            args.overlap,
            Some(ensured_analyze.as_path()),
        )?;
        let micro_analyze = build_micro_analyze_report(
            &model,
            Some(ensured_analyze.as_path()),
            prompt_corpus.clone(),
        )?;
        reports.push(MoeModelMatrixModelReport {
            model: benchmark_model_info(&model),
            ranking,
            grouping,
            micro_analyze,
        });
    }

    let report = MoeModelMatrixReport {
        benchmark: "moe-model-matrix",
        prompt_corpus,
        models: reports,
    };
    write_json_report(
        &report,
        args.output.as_deref(),
        "MoE model matrix benchmark",
    )?;
    Ok(())
}

pub(crate) async fn run_local_package_calibration_benchmark(
    model: &moe_planner::MoeModelContext,
    ranking: &moe_planner::ResolvedRanking,
    bin_dir: &Path,
) -> Result<PackageCalibrationBenchmarkReport> {
    let benchmark_started = Instant::now();
    let ranking_vec = moe::load_cached_ranking(&ranking.path).ok_or_else(|| {
        anyhow!(
            "cached ranking not found for calibration benchmark: {}",
            ranking.path.display()
        )
    })?;
    let server_bin = resolve_llama_server_binary()?;
    let backend_label = if PACKAGE_BENCHMARK_GPU_LAYERS == 0 {
        "cpu"
    } else if cfg!(target_os = "macos") {
        "metal"
    } else {
        "gpu"
    };
    let baseline_request_concurrency = benchmark_request_concurrency(model.total_model_bytes);
    eprintln!("🧪 Package calibration benchmark");
    eprintln!("   phase: benchmark calibration");
    let corpora =
        load_package_benchmark_corpora(&server_bin, model, baseline_request_concurrency).await?;
    if corpora.is_empty() {
        bail!("package calibration suite was empty");
    }
    eprintln!(
        "   corpora: {}",
        corpora
            .iter()
            .map(|corpus| format!("{}({})", corpus.source.short_name(), corpus.prompts.len()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "   runner: warm llama-server via {} ({backend_label}, -ngl {})",
        server_bin.display(),
        PACKAGE_BENCHMARK_GPU_LAYERS
    );
    eprintln!("   baseline request workers: {baseline_request_concurrency}");
    eprintln!("   model: {}", model.path.display());
    eprintln!("   ranking: {}", ranking.path.display());
    eprintln!("   max tokens: {}", PACKAGE_BENCHMARK_MAX_TOKENS);
    eprintln!("   request mode: deterministic raw completions (temperature 0, seed 0)");
    let search_floor = package_benchmark_search_floor(model);
    let search_ceiling = package_benchmark_search_ceiling(model);
    eprintln!(
        "   search: coarse halving from {} down to {}, then bisection",
        search_ceiling, search_floor
    );
    let temp_root = create_package_temp_dir(&model.path, "benchmark-calibration-")?;

    let mut evaluated = BTreeMap::new();
    let mut evaluation_order = Vec::new();
    let coarse_candidates = package_benchmark_coarse_candidates(search_floor, search_ceiling);
    let mut passing_bound = None;
    let mut failing_bound = None;

    for min_experts in coarse_candidates {
        let report = evaluate_package_benchmark_candidate(
            evaluated.len() + 1,
            min_experts,
            model,
            &ranking_vec,
            &corpora,
            bin_dir,
            &server_bin,
            temp_root.path(),
        )
        .await?;
        let passes = package_candidate_passes_quality_floor(&report);
        evaluated.insert(min_experts, report);
        if !evaluation_order.contains(&min_experts) {
            evaluation_order.push(min_experts);
        }
        if passes {
            passing_bound = Some(min_experts);
        } else if passing_bound.is_some() {
            failing_bound = Some(min_experts);
            break;
        } else {
            break;
        }
    }

    if let (Some(mut passing), Some(mut failing)) = (passing_bound, failing_bound) {
        while passing > failing + 1 {
            let mid = failing + ((passing - failing) / 2);
            if !evaluated.contains_key(&mid) {
                let report = evaluate_package_benchmark_candidate(
                    evaluated.len() + 1,
                    mid,
                    model,
                    &ranking_vec,
                    &corpora,
                    bin_dir,
                    &server_bin,
                    temp_root.path(),
                )
                .await?;
                evaluated.insert(mid, report);
                evaluation_order.push(mid);
            }
            let passes = evaluated
                .get(&mid)
                .map(package_candidate_passes_quality_floor)
                .unwrap_or(false);
            if passes {
                passing = mid;
            } else {
                failing = mid;
            }
        }
    }

    let candidates = evaluation_order
        .iter()
        .filter_map(|min_experts| evaluated.get(min_experts).cloned())
        .collect::<Vec<_>>();

    let recommended_min_experts_per_node = candidates
        .iter()
        .filter(|candidate| {
            candidate
                .corpora
                .iter()
                .all(|corpus| corpus.worst_node_score >= PACKAGE_BENCHMARK_QUALITY_FLOOR)
        })
        .map(|candidate| candidate.min_experts_per_node)
        .min();
    let Some(recommended_min_experts_per_node) = recommended_min_experts_per_node else {
        bail!(
            "No benchmark candidate met the quality floor. The assembled full-expert shard did not match the full-model baseline closely enough, so the calibration result is not trustworthy."
        );
    };
    eprintln!(
        "🧪 Recommended min_experts_per_node: {}",
        recommended_min_experts_per_node
    );
    eprintln!(
        "🧪 Benchmark calibration completed in {:.1}s",
        benchmark_started.elapsed().as_secs_f64()
    );

    Ok(PackageCalibrationBenchmarkReport {
        version: PACKAGE_BENCHMARK_VERSION,
        corpora: corpora
            .iter()
            .map(|corpus| PackageCalibrationCorpus {
                source: corpus.source.short_name(),
                dataset: corpus.source.dataset_name(),
                prompt_count: corpus.prompts.len(),
                max_tokens: PACKAGE_BENCHMARK_MAX_TOKENS,
            })
            .collect(),
        baseline: "full-model",
        metric: "token_dice_similarity",
        quality_floor: PACKAGE_BENCHMARK_QUALITY_FLOOR,
        candidates,
        recommended_min_experts_per_node,
    })
}

async fn evaluate_package_benchmark_candidate(
    candidate_index: usize,
    min_experts: u32,
    model: &moe_planner::MoeModelContext,
    ranking_vec: &[u32],
    corpora: &[LoadedCalibrationCorpus],
    bin_dir: &Path,
    server_bin: &Path,
    temp_root: &Path,
) -> Result<PackageCalibrationCandidateReport> {
    let candidate_started = Instant::now();
    let node_count = ((model.expert_count as f64) / (min_experts as f64))
        .ceil()
        .max(1.0) as usize;
    eprintln!(
        "   candidate {}: min_experts_per_node={} -> {} node(s)",
        candidate_index, min_experts, node_count
    );
    let assignments =
        moe::compute_assignments_with_overlap(ranking_vec, node_count, min_experts, 1);
    let identity_candidate = assignments.len() == 1 && min_experts >= model.expert_count;
    if identity_candidate {
        eprintln!("     identity candidate: using original model without assembly");
    }
    let assembly_concurrency =
        benchmark_assembly_concurrency(temp_root, model.total_model_bytes, assignments.len());
    let mut node_scores = vec![0.0; assignments.len()];
    let mut corpus_node_scores = corpora
        .iter()
        .map(|_| vec![0.0; assignments.len()])
        .collect::<Vec<_>>();
    if identity_candidate {
        evaluate_package_benchmark_node(
            model.path.clone(),
            0,
            assignments.len(),
            min_experts,
            corpora,
            server_bin,
            temp_root,
            &mut node_scores,
            &mut corpus_node_scores,
        )
        .await?;
    } else {
        eprintln!(
            "     assembly workers: {} for {} node shard(s)",
            assembly_concurrency,
            assignments.len()
        );
        let mut next_index = 0usize;
        let mut join_set = JoinSet::new();

        while next_index < assignments.len() && join_set.len() < assembly_concurrency {
            queue_candidate_shard_assembly(
                &mut join_set,
                bin_dir,
                model,
                temp_root,
                min_experts,
                &assignments,
                next_index,
            );
            next_index += 1;
        }

        while let Some(joined) = join_set.join_next().await {
            let (node_index, shard_path) = joined.context("Join shard assembly task")??;
            eprintln!(
                "     assembled node {}/{} -> {}",
                node_index + 1,
                assignments.len(),
                shard_path.display()
            );

            while next_index < assignments.len() && join_set.len() < assembly_concurrency {
                queue_candidate_shard_assembly(
                    &mut join_set,
                    bin_dir,
                    model,
                    temp_root,
                    min_experts,
                    &assignments,
                    next_index,
                );
                next_index += 1;
            }

            evaluate_package_benchmark_node(
                shard_path,
                node_index,
                assignments.len(),
                min_experts,
                corpora,
                server_bin,
                temp_root,
                &mut node_scores,
                &mut corpus_node_scores,
            )
            .await?;
        }
    }

    let candidate_mean_score = mean_score(&node_scores);
    let worst_node_score = node_scores.iter().copied().fold(1.0_f64, f64::min);
    let corpus_reports = corpora
        .iter()
        .enumerate()
        .map(|(corpus_index, corpus)| {
            let node_scores = corpus_node_scores[corpus_index].clone();
            let corpus_mean_score = mean_score(&node_scores);
            let worst_node_score = node_scores.iter().copied().fold(1.0_f64, f64::min);
            PackageCalibrationCandidateCorpusReport {
                source: corpus.source.short_name(),
                dataset: corpus.source.dataset_name(),
                prompt_count: corpus.prompts.len(),
                mean_score: corpus_mean_score,
                worst_node_score,
                node_scores,
            }
        })
        .collect::<Vec<_>>();
    eprintln!(
        "   result: min_experts_per_node={} mean={:.3} worst={:.3} elapsed={:.1}s",
        min_experts,
        candidate_mean_score,
        worst_node_score,
        candidate_started.elapsed().as_secs_f64()
    );

    Ok(PackageCalibrationCandidateReport {
        min_experts_per_node: min_experts,
        node_count,
        mean_score: candidate_mean_score,
        worst_node_score,
        node_scores,
        corpora: corpus_reports,
    })
}

async fn evaluate_package_benchmark_node(
    shard_path: PathBuf,
    node_index: usize,
    node_count: usize,
    min_experts: u32,
    corpora: &[LoadedCalibrationCorpus],
    server_bin: &Path,
    temp_root: &Path,
    node_scores: &mut [f64],
    corpus_node_scores: &mut [Vec<f64>],
) -> Result<()> {
    let node_started = Instant::now();
    let runner_label = format!(
        "candidate min_experts={} node {}/{}",
        min_experts,
        node_index + 1,
        node_count
    );
    let request_concurrency =
        benchmark_request_concurrency(fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0));
    eprintln!(
        "     request workers: {} for node {}/{}",
        request_concurrency,
        node_index + 1,
        node_count
    );
    let runner = start_local_server_runner(
        server_bin,
        &shard_path,
        PACKAGE_BENCHMARK_CONTEXT_SIZE,
        PACKAGE_BENCHMARK_GPU_LAYERS,
        &runner_label,
        temp_root,
    )
    .await?;
    let mut all_prompt_scores = Vec::new();
    let mut node_result = Ok(());
    for (corpus_index, corpus) in corpora.iter().enumerate() {
        let corpus_started = Instant::now();
        eprintln!(
            "       corpus {}/{}: {} ({})",
            corpus_index + 1,
            corpora.len(),
            corpus.source.dataset_name(),
            corpus.prompts.len()
        );
        match evaluate_prompts_with_server(
            &runner,
            &corpus.prompts,
            &format!("{runner_label} [{}]", corpus.source.short_name()),
            request_concurrency,
        )
        .await
        {
            Ok(candidate_outputs) => {
                let prompt_scores = candidate_outputs
                    .iter()
                    .zip(&corpus.baseline_outputs)
                    .map(|(candidate, baseline)| token_dice_similarity(baseline, candidate))
                    .collect::<Vec<_>>();
                let corpus_mean = mean_score(&prompt_scores);
                corpus_node_scores[corpus_index][node_index] = corpus_mean;
                all_prompt_scores.extend(prompt_scores);
                eprintln!(
                    "       corpus result: {} mean={:.3} elapsed={:.1}s",
                    corpus.source.short_name(),
                    corpus_mean,
                    corpus_started.elapsed().as_secs_f64()
                );
            }
            Err(err) => {
                node_result = Err(err);
                break;
            }
        }
    }
    runner.shutdown().await;
    node_result?;
    node_scores[node_index] = mean_score(&all_prompt_scores);
    eprintln!(
        "     node result: {}/{} mean={:.3} elapsed={:.1}s",
        node_index + 1,
        node_count,
        node_scores[node_index],
        node_started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn package_candidate_passes_quality_floor(candidate: &PackageCalibrationCandidateReport) -> bool {
    candidate
        .corpora
        .iter()
        .all(|corpus| corpus.worst_node_score >= PACKAGE_BENCHMARK_QUALITY_FLOOR)
}

fn package_benchmark_search_floor(model: &moe_planner::MoeModelContext) -> u32 {
    model.used_expert_count.max(1)
}

fn package_benchmark_search_ceiling(model: &moe_planner::MoeModelContext) -> u32 {
    model
        .expert_count
        .max(package_benchmark_search_floor(model))
}

fn package_benchmark_coarse_candidates(floor: u32, ceiling: u32) -> Vec<u32> {
    let mut candidates = Vec::new();
    let mut current = ceiling.max(floor);
    candidates.push(current);
    while current > floor {
        let next = (current / 2).max(floor);
        if next == current {
            break;
        }
        candidates.push(next);
        current = next;
    }
    candidates
}

fn build_ranking_report(
    model: &ResolvedBenchmarkModel,
    nodes: usize,
    overlap: usize,
    variants: &[BenchmarkVariant],
    analyze_ranking: Option<&Path>,
    prompt_corpus: Option<PromptCorpusSummary>,
) -> Result<MoeRankingBenchmarkReport> {
    let mut reports = Vec::with_capacity(variants.len());
    for &variant in variants {
        let ranking = resolve_variant_ranking(variant, model, analyze_ranking)?;
        let assignments =
            moe::compute_assignments_with_overlap(&ranking, nodes, model.min_experts, overlap);
        reports.push(VariantReport {
            name: variant_name(variant),
            ranking_source: variant_source_label(variant, model, analyze_ranking),
            ranking_len: ranking.len(),
            assignments: assignment_reports(assignments),
        });
    }

    Ok(MoeRankingBenchmarkReport {
        benchmark: "moe-ranking",
        model: benchmark_model_info(model),
        config: BenchmarkConfig {
            nodes,
            overlap,
            min_experts: model.min_experts,
        },
        prompt_corpus,
        variants: reports,
    })
}

fn build_grouping_report(
    model: &ResolvedBenchmarkModel,
    nodes: usize,
    overlap: usize,
    analyze_ranking: Option<&Path>,
) -> Result<MoeGroupingBenchmarkReport> {
    let analyze_path = ensure_full_analyze_ranking(model, analyze_ranking)?;
    let profile = load_analyze_mass_profile(&analyze_path)?;
    let analyze_ranking_vec = profile.ranking().to_vec();
    let sequential: Vec<u32> = (0..model.info.expert_count).collect();
    let replicate = model.min_experts as usize;

    let strategies = vec![
        (
            "current-sequential",
            "shared-core-overlap",
            "sequential-fallback".to_string(),
            sequential.clone(),
            moe::compute_assignments_with_overlap(&sequential, nodes, model.min_experts, overlap),
            replicate,
        ),
        (
            "current-analyze",
            "shared-core-overlap",
            analyze_path.display().to_string(),
            analyze_ranking_vec.clone(),
            moe::compute_assignments_with_overlap(
                &analyze_ranking_vec,
                nodes,
                model.min_experts,
                overlap,
            ),
            replicate,
        ),
        (
            "snake-analyze-replicated",
            "snake-draft",
            analyze_path.display().to_string(),
            analyze_ranking_vec.clone(),
            moe::compute_snake_draft_assignments(&analyze_ranking_vec, nodes, replicate),
            replicate,
        ),
    ];

    let reports = strategies
        .into_iter()
        .map(
            |(name, grouping_mode, ranking_source, ranking, assignments, replicated_experts)| {
                let node_mass_pct = assignments
                    .iter()
                    .map(|assignment| mass_pct_for_experts(&assignment.experts, &profile))
                    .collect::<Vec<_>>();
                let shared_mass_pct = mass_pct_for_experts(
                    &ranking[..replicated_experts.min(ranking.len())],
                    &profile,
                );
                let mean_node_mass_pct =
                    node_mass_pct.iter().sum::<f64>() / node_mass_pct.len().max(1) as f64;
                let min_node_mass_pct = node_mass_pct.iter().copied().fold(f64::INFINITY, f64::min);
                let max_node_mass_pct = node_mass_pct
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                GroupingStrategyReport {
                    name,
                    ranking_source,
                    grouping_mode,
                    replicated_experts,
                    shared_mass_pct,
                    mean_node_mass_pct,
                    min_node_mass_pct,
                    max_node_mass_pct,
                    node_mass_imbalance_pct: max_node_mass_pct - min_node_mass_pct,
                    assignments: assignment_reports(assignments),
                }
            },
        )
        .collect();

    Ok(MoeGroupingBenchmarkReport {
        benchmark: "moe-grouping",
        model: benchmark_model_info(model),
        config: BenchmarkConfig {
            nodes,
            overlap,
            min_experts: model.min_experts,
        },
        analyze_ranking_source: analyze_path.display().to_string(),
        strategies: reports,
    })
}

fn build_micro_analyze_report(
    model: &ResolvedBenchmarkModel,
    analyze_ranking: Option<&Path>,
    prompt_corpus: Option<PromptCorpusSummary>,
) -> Result<MoeMicroAnalyzeBenchmarkReport> {
    let analyze_path = ensure_full_analyze_ranking(model, analyze_ranking)?;
    let profile = load_analyze_mass_profile(&analyze_path)?;
    let prompts = load_or_default_prompts(prompt_corpus.as_ref())?;
    let prompt_count = prompts.len();
    let experiments = micro_experiment_configs(prompt_count)
        .into_iter()
        .map(|config| run_micro_experiment(model, &profile, &prompts, config))
        .collect::<Result<Vec<_>>>()?;

    Ok(MoeMicroAnalyzeBenchmarkReport {
        benchmark: "moe-micro-analyze",
        model: benchmark_model_info(model),
        min_experts: model.min_experts,
        analyze_ranking_source: analyze_path.display().to_string(),
        prompt_corpus,
        experiments,
    })
}

async fn resolve_benchmark_model(
    model_spec: &str,
    min_experts_override: Option<u32>,
) -> Result<ResolvedBenchmarkModel> {
    let path = models::resolve_model_spec(Path::new(model_spec)).await?;
    let info = moe::detect_moe(&path).with_context(|| {
        format!(
            "Model is not auto-detected as MoE from GGUF header: {}",
            path.display()
        )
    })?;
    let name = model_display_name(&path);
    let bundled = bundled_moe_config(&name);
    let min_experts = min_experts_override
        .or_else(|| bundled.as_ref().map(|cfg| cfg.min_experts_per_node))
        .unwrap_or_else(|| ((info.expert_count as f64) * 0.5).ceil() as u32);
    let architecture = moe::scan_gguf_compact_meta(&path)
        .map(|meta| meta.architecture)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    Ok(ResolvedBenchmarkModel {
        input: model_spec.to_string(),
        path,
        name,
        architecture,
        info,
        min_experts,
    })
}

fn explicit_analyze_path(
    model: &ResolvedBenchmarkModel,
    analyze_ranking_dir: Option<&Path>,
) -> Option<PathBuf> {
    let dir = analyze_ranking_dir?;
    let path = dir.join(format!("{}.csv", model.path.file_stem()?.to_string_lossy()));
    path.exists().then_some(path)
}

fn benchmark_model_info(model: &ResolvedBenchmarkModel) -> BenchmarkModelInfo {
    BenchmarkModelInfo {
        input: model.input.clone(),
        resolved_path: model.path.display().to_string(),
        name: model.name.clone(),
        architecture: model.architecture.clone(),
        expert_count: model.info.expert_count,
        expert_used_count: model.info.expert_used_count,
    }
}

fn model_display_name(model_path: &Path) -> String {
    model_path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(router::strip_split_suffix_owned)
        .unwrap_or_else(|| model_path.display().to_string())
}

fn bundled_moe_config(model_name: &str) -> Option<catalog::MoeConfig> {
    let q = model_name.to_lowercase();
    catalog::MODEL_CATALOG
        .iter()
        .find(|m| m.name.to_lowercase() == q || m.file.to_lowercase().contains(&q))
        .and_then(|m| m.moe.clone())
}

fn validate_nodes(nodes: usize) -> Result<()> {
    if nodes == 0 {
        bail!("--nodes must be at least 1");
    }
    Ok(())
}

fn load_prompt_summary(path: Option<&Path>) -> Result<Option<PromptCorpusSummary>> {
    path.map(benchmark_prompts::summarize_prompt_corpus)
        .transpose()
}

fn load_or_default_prompts(prompt_corpus: Option<&PromptCorpusSummary>) -> Result<Vec<String>> {
    let Some(summary) = prompt_corpus else {
        return Ok(vec![
            "User: Explain how MoE expert routing works in a large language model.\nAssistant:"
                .to_string(),
        ]);
    };

    let prompts = benchmark_prompts::load_prompt_corpus(Path::new(&summary.path))?;
    Ok(prompts.into_iter().map(render_prompt).collect())
}

pub(crate) fn package_benchmark_is_current(analysis_path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(analysis_path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) else {
        return false;
    };
    value
        .get("benchmark")
        .and_then(|benchmark| benchmark.get("version"))
        .and_then(|version| version.as_u64())
        == Some(PACKAGE_BENCHMARK_VERSION as u64)
}

fn render_prompt(entry: PromptCorpusEntry) -> String {
    let mut rendered = String::new();
    for message in entry.messages {
        let _ = writeln!(
            rendered,
            "{}: {}\n",
            capitalize_role(&message.role),
            message.content.trim()
        );
    }
    rendered.trim().to_string()
}

fn baseline_cache_path(model_path: &Path, source: PromptImportSource) -> PathBuf {
    moe::package_cache_variant_dir(model_path)
        .join("benchmark-baselines")
        .join(format!("{}.json", source.short_name()))
}

fn prompt_corpus_hash(prompts: &[PromptCorpusEntry]) -> Result<String> {
    let bytes =
        serde_json::to_vec(prompts).context("Serialize prompt corpus for benchmark hash")?;
    Ok(hex::encode(sha2::Sha256::digest(bytes)))
}

fn load_baseline_cache(
    model_path: &Path,
    source: PromptImportSource,
    dataset: &str,
    prompts: &[PromptCorpusEntry],
) -> Result<Option<PackageCalibrationBaselineCache>> {
    let cache_path = baseline_cache_path(model_path, source);
    let Ok(content) = fs::read_to_string(&cache_path) else {
        return Ok(None);
    };
    let cache: PackageCalibrationBaselineCache = serde_json::from_str(&content)
        .with_context(|| format!("Parse {}", cache_path.display()))?;
    if cache.version != PACKAGE_BENCHMARK_VERSION
        || cache.dataset != dataset
        || cache.prompt_hash != prompt_corpus_hash(prompts)?
        || cache.max_tokens != PACKAGE_BENCHMARK_MAX_TOKENS
        || cache.context_size != PACKAGE_BENCHMARK_CONTEXT_SIZE
        || cache.n_gpu_layers != PACKAGE_BENCHMARK_GPU_LAYERS
        || cache.outputs.len() != prompts.len()
    {
        return Ok(None);
    }
    Ok(Some(cache))
}

fn save_baseline_cache(
    model_path: &Path,
    source: PromptImportSource,
    dataset: &str,
    prompts: &[PromptCorpusEntry],
    outputs: &[String],
) -> Result<()> {
    let cache = PackageCalibrationBaselineCache {
        version: PACKAGE_BENCHMARK_VERSION,
        dataset: dataset.to_string(),
        prompt_hash: prompt_corpus_hash(prompts)?,
        max_tokens: PACKAGE_BENCHMARK_MAX_TOKENS,
        context_size: PACKAGE_BENCHMARK_CONTEXT_SIZE,
        n_gpu_layers: PACKAGE_BENCHMARK_GPU_LAYERS,
        outputs: outputs.to_vec(),
    };
    let path = baseline_cache_path(model_path, source);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_string_pretty(&cache)? + "\n")
        .with_context(|| format!("Write {}", path.display()))?;
    Ok(())
}

async fn load_or_create_baseline_outputs(
    server_bin: &Path,
    model: &moe_planner::MoeModelContext,
    source: PromptImportSource,
    prompts: &[PromptCorpusEntry],
    request_concurrency: usize,
) -> Result<Vec<String>> {
    let baseline_started = Instant::now();
    let dataset = source.dataset_name();
    if let Some(cache) = load_baseline_cache(&model.path, source, dataset, prompts)? {
        eprintln!(
            "   baseline cache ({}): reusing {} outputs from {} ({:.1}s)",
            source.short_name(),
            cache.outputs.len(),
            baseline_cache_path(&model.path, source).display(),
            baseline_started.elapsed().as_secs_f64()
        );
        return Ok(cache.outputs);
    }

    eprintln!("   baseline cache ({}): none", source.short_name());
    let temp_root = create_package_temp_dir(&model.path, "benchmark-baseline-")?;
    let runner = start_local_server_runner(
        server_bin,
        &model.path,
        PACKAGE_BENCHMARK_CONTEXT_SIZE,
        PACKAGE_BENCHMARK_GPU_LAYERS,
        "baseline",
        temp_root.path(),
    )
    .await?;
    let outputs =
        evaluate_prompts_with_server(&runner, prompts, "baseline", request_concurrency).await;
    runner.shutdown().await;
    let outputs = outputs?;
    save_baseline_cache(&model.path, source, dataset, prompts, &outputs)?;
    eprintln!(
        "   baseline cache ({}): wrote {} ({:.1}s)",
        source.short_name(),
        baseline_cache_path(&model.path, source).display(),
        baseline_started.elapsed().as_secs_f64()
    );
    Ok(outputs)
}

async fn load_package_benchmark_corpora(
    server_bin: &Path,
    model: &moe_planner::MoeModelContext,
    request_concurrency: usize,
) -> Result<Vec<LoadedCalibrationCorpus>> {
    let mut corpora = Vec::new();
    for spec in package_benchmark_corpus_specs() {
        let corpus_started = Instant::now();
        let prompts = benchmark_prompts::fetch_prompt_corpus(
            spec.source,
            spec.limit,
            Some(PACKAGE_BENCHMARK_MAX_TOKENS),
        )
        .await
        .with_context(|| format!("Fetch prompt corpus {}", spec.source.dataset_name()))?;
        eprintln!(
            "   corpus: {} ({}) prompts={}",
            spec.source.dataset_name(),
            spec.source.short_name(),
            prompts.len()
        );
        let baseline_outputs = load_or_create_baseline_outputs(
            server_bin,
            model,
            spec.source,
            &prompts,
            request_concurrency,
        )
        .await?;
        eprintln!(
            "   corpus ready: {} ({}) baseline={} elapsed={:.1}s",
            spec.source.dataset_name(),
            spec.source.short_name(),
            baseline_outputs.len(),
            corpus_started.elapsed().as_secs_f64()
        );
        corpora.push(LoadedCalibrationCorpus {
            source: spec.source,
            prompts,
            baseline_outputs,
        });
    }
    Ok(corpora)
}

fn queue_candidate_shard_assembly(
    join_set: &mut JoinSet<Result<(usize, PathBuf)>>,
    bin_dir: &Path,
    model: &moe_planner::MoeModelContext,
    temp_root: &Path,
    min_experts: u32,
    assignments: &[moe::NodeAssignment],
    node_index: usize,
) {
    let experts = assignments[node_index].experts.clone();
    let bin_dir = bin_dir.to_path_buf();
    let model_path = model.path.clone();
    let expert_count = model.expert_count;
    let temp_root = temp_root.to_path_buf();
    let shard_path = temp_root.join(format!("min-{}-node-{}.gguf", min_experts, node_index));
    eprintln!(
        "     queue assemble node {}/{} -> {}",
        node_index + 1,
        assignments.len(),
        shard_path.display()
    );
    join_set.spawn_blocking(move || -> Result<(usize, PathBuf)> {
        let (trunk_path, expert_paths) =
            moe::ensure_local_component_cache(&bin_dir, &model_path, &experts, expert_count)?;
        moe::run_assemble_from_components(&bin_dir, &trunk_path, &expert_paths, &shard_path)?;
        Ok((node_index, shard_path))
    });
}

fn benchmark_prompt_label(prompt: &PromptCorpusEntry) -> String {
    let line = prompt
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .find(|content| !content.trim().is_empty())
        .unwrap_or(prompt.id.as_str())
        .trim();
    let compact = line.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let preview = chars.by_ref().take(72).collect::<String>();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

fn resolve_llama_server_binary() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = exe
        .parent()
        .ok_or_else(|| anyhow!("Current executable has no parent directory"))?;
    let candidates = [
        bin_dir.join("llama-server"),
        bin_dir.join("../llama.cpp/build/bin/llama-server"),
        bin_dir.join("../../llama.cpp/build/bin/llama-server"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-server"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }
    bail!(
        "llama-server not found next to {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    )
}

async fn start_local_server_runner(
    server_bin: &Path,
    model_path: &Path,
    context_size: u32,
    n_gpu_layers: u32,
    label: &str,
    temp_root: &Path,
) -> Result<LocalServerRunner> {
    let temp_id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let log_root = temp_root.join("runner-logs");
    fs::create_dir_all(&log_root)
        .with_context(|| format!("Create benchmark runner log dir {}", log_root.display()))?;
    let stdout_path = log_root.join(format!("mesh-llm-benchmark-server-{temp_id}.stdout"));
    let stderr_path = log_root.join(format!("mesh-llm-benchmark-server-{temp_id}.stderr"));
    let stdout_file =
        File::create(&stdout_path).with_context(|| format!("Create {}", stdout_path.display()))?;
    let stderr_file =
        File::create(&stderr_path).with_context(|| format!("Create {}", stderr_path.display()))?;
    let model_label = model_path
        .file_name()
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| model_path.to_string_lossy().into_owned());
    let port = find_free_port().await?;
    let base_url = format!("http://127.0.0.1:{port}");
    eprintln!("       ▶ start {label} runner on {base_url} using {model_label}");
    let args = vec![
        "-m".to_string(),
        model_path.to_string_lossy().to_string(),
        "-ngl".to_string(),
        n_gpu_layers.to_string(),
        "--no-jinja".to_string(),
        "--reasoning-format".to_string(),
        "none".to_string(),
        "--reasoning".to_string(),
        "off".to_string(),
        "-fa".to_string(),
        "on".to_string(),
        "-fit".to_string(),
        "off".to_string(),
        "--no-mmap".to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "-c".to_string(),
        context_size.to_string(),
    ];
    let mut child = TokioCommand::new(server_bin)
        .args(&args)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .kill_on_drop(true)
        .spawn()
        .with_context(|| {
            format!(
                "Start {} for {}",
                server_bin.display(),
                model_path.display()
            )
        })?;

    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().with_context(|| {
            format!("Poll {} for {}", server_bin.display(), model_path.display())
        })? {
            let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
            if !status.success() {
                bail!(
                    "llama-server failed for {} during {}: {}",
                    model_path.display(),
                    label,
                    stderr
                );
            }
            bail!(
                "llama-server exited before becoming healthy for {} during {}",
                model_path.display(),
                label
            );
        }

        if reqwest_health_check(&format!("{base_url}/health")).await {
            eprintln!(
                "       ✓ {label} runner ready in {:.1}s",
                started.elapsed().as_secs_f64()
            );
            return Ok(LocalServerRunner {
                base_url,
                model_path: model_path.to_path_buf(),
                log_path: stderr_path,
                child,
            });
        }
        if started.elapsed() >= PACKAGE_BENCHMARK_HEALTH_TIMEOUT {
            bail!(
                "llama-server health check timed out for {} during {} (log: {})",
                model_path.display(),
                label,
                stderr_path.display()
            );
        }
        tokio::time::sleep(PACKAGE_BENCHMARK_HEARTBEAT_INTERVAL).await;
        eprintln!(
            "       … waiting for {label} runner ({:.0}s elapsed)",
            started.elapsed().as_secs_f64()
        );
    }
}

impl LocalServerRunner {
    async fn shutdown(mut self) {
        let _ = self.child.start_kill();
        let _ = self.child.wait().await;
    }
}

async fn evaluate_prompts_with_server(
    runner: &LocalServerRunner,
    prompts: &[PromptCorpusEntry],
    phase_label: &str,
    request_concurrency: usize,
) -> Result<Vec<String>> {
    let phase_started = Instant::now();
    let mut outputs: Vec<Option<String>> = vec![None; prompts.len()];
    let mut next_index = 0usize;
    let mut completed = 0usize;
    let mut join_set = JoinSet::new();

    while completed < prompts.len() {
        while next_index < prompts.len() && join_set.len() < request_concurrency {
            let prompt = prompts[next_index].clone();
            let request_label = format!(
                "{} prompt {}/{}",
                phase_label,
                next_index + 1,
                prompts.len()
            );
            eprintln!(
                "       ▶ {request_label}: {}",
                benchmark_prompt_label(&prompt)
            );
            let base_url = runner.base_url.clone();
            let model_path = runner.model_path.clone();
            join_set.spawn(async move {
                let prompt_started = Instant::now();
                let output = request_server_completion(&base_url, &prompt, &request_label).await;
                (
                    next_index,
                    output,
                    request_label,
                    model_path,
                    prompt_started.elapsed(),
                )
            });
            next_index += 1;
        }

        let Some(result) = join_set.join_next().await else {
            break;
        };
        let (index, output, request_label, model_path, elapsed) =
            result.context("Join benchmark request task")?;
        let output = output.with_context(|| {
            format!(
                "Benchmark request failed for {} (log: {})",
                model_path.display(),
                runner.log_path.display()
            )
        })?;
        eprintln!(
            "       ✓ {request_label} completed in {:.1}s",
            elapsed.as_secs_f64()
        );
        outputs[index] = Some(output);
        completed += 1;
        eprintln!("       progress: {completed}/{} prompt(s)", prompts.len());
    }

    let outputs = outputs
        .into_iter()
        .map(|value| value.ok_or_else(|| anyhow!("missing benchmark output")))
        .collect::<Result<Vec<_>>>()?;
    eprintln!(
        "       ✓ {phase_label} completed in {:.1}s",
        phase_started.elapsed().as_secs_f64()
    );
    Ok(outputs)
}

async fn request_server_completion(
    base_url: &str,
    prompt: &PromptCorpusEntry,
    step_label: &str,
) -> Result<String> {
    let client = reqwest::Client::new();
    let start = Instant::now();
    let rendered_prompt = render_benchmark_completion_prompt(prompt);
    let response = client
        .post(format!("{base_url}/completion"))
        .json(&json!({
            "prompt": rendered_prompt,
            "n_predict": PACKAGE_BENCHMARK_MAX_TOKENS,
            "chat_format": 0,
            "reasoning_format": "none",
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "stream": false
        }))
        .send()
        .await
        .with_context(|| format!("Send benchmark request for {step_label}"))?;

    let status = response.status();
    let body: serde_json::Value = response
        .json()
        .await
        .with_context(|| format!("Parse benchmark response for {step_label}"))?;
    if !status.is_success() {
        if benchmark_response_is_parser_failure(&body) {
            eprintln!(
                "       ⚠ {step_label} triggered llama-server parse failure after {:.1}s; scoring as empty output",
                start.elapsed().as_secs_f64()
            );
            return Ok(String::new());
        }
        bail!("server returned {} for {}: {}", status, step_label, body);
    }
    let content = extract_benchmark_response_text(&body);
    if content.is_empty() && !benchmark_response_has_completion_payload(&body) {
        bail!(
            "empty completion for {} after {:.1}s: {}",
            step_label,
            start.elapsed().as_secs_f64(),
            body
        );
    }
    if content.is_empty() {
        eprintln!(
            "       ⚠ {step_label} returned only whitespace after {:.1}s; scoring as empty output",
            start.elapsed().as_secs_f64()
        );
    }
    Ok(content)
}

fn extract_benchmark_response_text(body: &serde_json::Value) -> String {
    let mut parts = Vec::new();

    if let Some(text) = body.get("content").and_then(|value| value.as_str()) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    if let Some(text) = body
        .pointer("/choices/0/text")
        .and_then(|value| value.as_str())
    {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    if let Some(content) = body.pointer("/choices/0/message/content") {
        match content {
            serde_json::Value::String(text) => {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    parts.push(trimmed.to_string());
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    if let Some(text) = item.get("text").and_then(|value| value.as_str()) {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            parts.push(trimmed.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(reasoning) = body
        .pointer("/choices/0/message/reasoning_content")
        .and_then(|value| value.as_str())
    {
        let trimmed = reasoning.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    parts.join("\n\n")
}

fn benchmark_response_has_completion_payload(body: &serde_json::Value) -> bool {
    body.get("content").is_some()
        || body.pointer("/choices/0/text").is_some()
        || body.pointer("/choices/0/message/content").is_some()
        || body
            .pointer("/choices/0/message/reasoning_content")
            .is_some()
}

fn benchmark_response_is_parser_failure(body: &serde_json::Value) -> bool {
    body.pointer("/error/message")
        .and_then(|value| value.as_str())
        .map(|message| message.contains("Failed to parse input"))
        .unwrap_or(false)
}

fn render_benchmark_completion_prompt(prompt: &PromptCorpusEntry) -> String {
    let mut rendered = String::new();
    for message in &prompt.messages {
        let _ = writeln!(
            rendered,
            "{}: {}\n",
            capitalize_role(&message.role),
            message.content.trim()
        );
    }
    rendered.push_str("Assistant:");
    rendered
}

async fn find_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    Ok(listener.local_addr()?.port())
}

fn benchmark_assembly_concurrency(temp_root: &Path, model_bytes: u64, shard_count: usize) -> usize {
    if let Ok(raw) = std::env::var("MESH_LLM_MOE_ASSEMBLY_CONCURRENCY") {
        if let Ok(parsed) = raw.trim().parse::<usize>() {
            return parsed.max(1).min(shard_count.max(1));
        }
    }

    let cpu_slots = match std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4)
    {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 3,
        _ => 4,
    };

    let per_assembly_budget = model_bytes.saturating_mul(2).max(8 * 1024 * 1024 * 1024);
    let disk_slots = free_disk_space(temp_root)
        .map(|free| (free / per_assembly_budget) as usize)
        .unwrap_or(PACKAGE_BENCHMARK_ASSEMBLY_CONCURRENCY_CAP)
        .max(1);

    cpu_slots
        .min(disk_slots)
        .min(PACKAGE_BENCHMARK_ASSEMBLY_CONCURRENCY_CAP)
        .min(shard_count.max(1))
        .max(1)
}

fn benchmark_request_concurrency(model_bytes: u64) -> usize {
    if let Ok(raw) = std::env::var("MESH_LLM_MOE_REQUEST_CONCURRENCY") {
        if let Ok(parsed) = raw.trim().parse::<usize>() {
            return parsed.max(1).min(PACKAGE_BENCHMARK_REQUEST_CONCURRENCY_CAP);
        }
    }

    let cpu_slots = match std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4)
    {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 3,
        _ => 4,
    };

    let model_slots = if model_bytes >= 40 * 1024 * 1024 * 1024 {
        1
    } else if model_bytes >= 20 * 1024 * 1024 * 1024 {
        2
    } else if model_bytes >= 10 * 1024 * 1024 * 1024 {
        3
    } else {
        4
    };

    cpu_slots
        .min(model_slots)
        .min(PACKAGE_BENCHMARK_REQUEST_CONCURRENCY_CAP)
        .max(1)
}

fn free_disk_space(path: &Path) -> Option<u64> {
    let mut check = path.to_path_buf();
    loop {
        if check.exists() {
            break;
        }
        if !check.pop() {
            return None;
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(check.as_os_str().as_bytes()).ok()?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
        if ret == 0 {
            Some(stat.f_bavail as u64 * stat.f_frsize)
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

async fn reqwest_health_check(url: &str) -> bool {
    reqwest::Client::new()
        .get(url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn tokenize_similarity_text(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn token_dice_similarity(left: &str, right: &str) -> f64 {
    let left_tokens = tokenize_similarity_text(left);
    let right_tokens = tokenize_similarity_text(right);
    if left_tokens.is_empty() && right_tokens.is_empty() {
        return 1.0;
    }
    let mut left_counts = HashMap::new();
    let mut right_counts = HashMap::new();
    for token in left_tokens {
        *left_counts.entry(token).or_insert(0usize) += 1;
    }
    for token in right_tokens {
        *right_counts.entry(token).or_insert(0usize) += 1;
    }
    let overlap = left_counts
        .iter()
        .map(|(token, left_count)| {
            right_counts
                .get(token)
                .map(|right_count| (*left_count).min(*right_count))
                .unwrap_or(0)
        })
        .sum::<usize>();
    let total = left_counts.values().sum::<usize>() + right_counts.values().sum::<usize>();
    if total == 0 {
        1.0
    } else {
        (2.0 * overlap as f64) / total as f64
    }
}

fn mean_score(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn capitalize_role(role: &str) -> String {
    let mut chars = role.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
        None => "User".to_string(),
    }
}

fn assignment_reports(assignments: Vec<moe::NodeAssignment>) -> Vec<AssignmentReport> {
    assignments
        .into_iter()
        .enumerate()
        .map(|(node, assignment)| AssignmentReport {
            node,
            expert_count: assignment.experts.len(),
            shared: assignment.n_shared,
            unique: assignment.n_unique,
            experts: assignment.experts,
        })
        .collect()
}

fn resolve_variant_ranking(
    variant: BenchmarkVariant,
    model: &ResolvedBenchmarkModel,
    analyze_ranking: Option<&Path>,
) -> Result<Vec<u32>> {
    match variant {
        BenchmarkVariant::Analyze => {
            if let Some(path) = analyze_ranking {
                return moe::load_cached_ranking(path)
                    .with_context(|| format!("Load moe-analyze ranking from {}", path.display()));
            }

            let cached_path = moe::ranking_cache_path(&model.path);
            if let Some(ranking) = moe::load_cached_ranking(&cached_path) {
                return Ok(ranking);
            }

            bail!(
                "No moe-analyze ranking found for {}. Provide --analyze-ranking or cache a ranking at {}",
                model.name,
                cached_path.display()
            )
        }
    }
}

fn variant_name(variant: BenchmarkVariant) -> &'static str {
    match variant {
        BenchmarkVariant::Analyze => "analyze",
    }
}

fn variant_source_label(
    variant: BenchmarkVariant,
    model: &ResolvedBenchmarkModel,
    analyze_ranking: Option<&Path>,
) -> String {
    match variant {
        BenchmarkVariant::Analyze => {
            if let Some(path) = analyze_ranking {
                return path.display().to_string();
            }
            let cached_path = moe::ranking_cache_path(&model.path);
            if cached_path.exists() {
                cached_path.display().to_string()
            } else {
                "missing".to_string()
            }
        }
    }
}

fn ensure_full_analyze_ranking(
    model: &ResolvedBenchmarkModel,
    explicit: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        bail!("Explicit analyze ranking not found: {}", path.display());
    }

    let cached = moe::ranking_cache_path(&model.path);
    if cached.exists() {
        return Ok(cached);
    }

    if let Some(parent) = cached.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Create analyze ranking directory {}", parent.display()))?;
    }

    let analyze_bin = resolve_analyze_binary()?;
    let output = Command::new(&analyze_bin)
        .args([
            "-m",
            &model.path.to_string_lossy(),
            "--all-layers",
            "--export-ranking",
            &cached.to_string_lossy(),
            "-n",
            "32",
            "-c",
            "4096",
            "-ngl",
            "99",
        ])
        .output()
        .with_context(|| format!("Run {} for {}", analyze_bin.display(), model.path.display()))?;

    if !output.status.success() {
        bail!(
            "llama-moe-analyze failed for {}: {}",
            model.path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(cached)
}

fn resolve_analyze_binary() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = exe
        .parent()
        .ok_or_else(|| anyhow!("Current executable has no parent directory"))?;
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

fn load_analyze_mass_profile(path: &Path) -> Result<AnalyzeMassProfile> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Read analyze ranking {}", path.display()))?;
    let mut entries = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 4 {
            continue;
        }
        entries.push(AnalyzeExpertMass {
            expert_id: parts[0].parse().with_context(|| {
                format!("Parse expert id from {} in {}", trimmed, path.display())
            })?,
            gate_mass: parts[1].parse().with_context(|| {
                format!("Parse gate mass from {} in {}", trimmed, path.display())
            })?,
            mass_pct: parts[2].parse().with_context(|| {
                format!("Parse mass pct from {} in {}", trimmed, path.display())
            })?,
            selection_count: parts[3].parse().with_context(|| {
                format!(
                    "Parse selection count from {} in {}",
                    trimmed,
                    path.display()
                )
            })?,
        });
    }

    if entries.is_empty() {
        bail!(
            "Analyze ranking was empty or unreadable: {}",
            path.display()
        );
    }

    let mut mass_by_expert = HashMap::new();
    let total_mass = entries.iter().map(|entry| entry.gate_mass).sum::<f64>();
    for entry in &entries {
        mass_by_expert.insert(entry.expert_id, entry.gate_mass);
    }

    Ok(AnalyzeMassProfile {
        entries,
        mass_by_expert,
        total_mass,
    })
}

impl AnalyzeMassProfile {
    fn ranking(&self) -> Vec<u32> {
        self.entries.iter().map(|entry| entry.expert_id).collect()
    }
}

fn recall_at_top_n(candidate: &[u32], truth: &[u32], n: usize) -> f64 {
    let n = n.min(candidate.len()).min(truth.len());
    if n == 0 {
        return 0.0;
    }
    let candidate_set: BTreeSet<u32> = candidate.iter().take(n).copied().collect();
    let truth_set: BTreeSet<u32> = truth.iter().take(n).copied().collect();
    candidate_set.intersection(&truth_set).count() as f64 / n as f64
}

fn weighted_recall_at_top_n(candidate: &[u32], truth: &AnalyzeMassProfile, n: usize) -> f64 {
    let truth_top = truth.entries.iter().take(n).collect::<Vec<_>>();
    if truth_top.is_empty() {
        return 0.0;
    }
    let denominator = truth_top.iter().map(|entry| entry.gate_mass).sum::<f64>();
    if denominator <= f64::EPSILON {
        return 0.0;
    }
    let candidate_set: BTreeSet<u32> = candidate.iter().take(n).copied().collect();
    let numerator = truth_top
        .iter()
        .filter(|entry| candidate_set.contains(&entry.expert_id))
        .map(|entry| entry.gate_mass)
        .sum::<f64>();
    numerator / denominator
}

fn spearman_rank_correlation(candidate: &[u32], truth: &AnalyzeMassProfile) -> f64 {
    let n = candidate.len().min(truth.entries.len());
    if n < 2 {
        return 1.0;
    }
    let mut candidate_rank = HashMap::new();
    for (idx, expert) in candidate.iter().enumerate() {
        candidate_rank.insert(*expert, idx as f64);
    }
    let sum_d2 = truth
        .entries
        .iter()
        .take(n)
        .enumerate()
        .filter_map(|(idx, entry)| {
            candidate_rank
                .get(&entry.expert_id)
                .map(|cand| (*cand - idx as f64).powi(2))
        })
        .sum::<f64>();
    let n = n as f64;
    1.0 - (6.0 * sum_d2) / (n * (n * n - 1.0))
}

fn mass_pct_for_experts(experts: &[u32], profile: &AnalyzeMassProfile) -> f64 {
    if profile.total_mass <= f64::EPSILON {
        return 0.0;
    }
    let numerator = experts
        .iter()
        .filter_map(|expert| profile.mass_by_expert.get(expert).copied())
        .sum::<f64>();
    100.0 * numerator / profile.total_mass
}

#[derive(Clone, Copy)]
struct MicroAnalyzeConfig {
    name: &'static str,
    prompt_count: usize,
    tokens: u32,
    all_layers: bool,
}

fn micro_experiment_configs(prompt_count: usize) -> Vec<MicroAnalyzeConfig> {
    let mut configs = vec![
        MicroAnalyzeConfig {
            name: "micro-1p-8t-first-layer",
            prompt_count: 1,
            tokens: 8,
            all_layers: false,
        },
        MicroAnalyzeConfig {
            name: "micro-1p-8t-all-layers",
            prompt_count: 1,
            tokens: 8,
            all_layers: true,
        },
    ];
    if prompt_count >= 4 {
        configs.push(MicroAnalyzeConfig {
            name: "micro-4p-8t-all-layers",
            prompt_count: 4,
            tokens: 8,
            all_layers: true,
        });
        configs.push(MicroAnalyzeConfig {
            name: "micro-4p-32t-all-layers",
            prompt_count: 4,
            tokens: 32,
            all_layers: true,
        });
    } else if prompt_count >= 2 {
        configs.push(MicroAnalyzeConfig {
            name: "micro-2p-32t-all-layers",
            prompt_count: 2,
            tokens: 32,
            all_layers: true,
        });
    }
    configs
}

fn run_micro_experiment(
    model: &ResolvedBenchmarkModel,
    truth: &AnalyzeMassProfile,
    prompts: &[String],
    config: MicroAnalyzeConfig,
) -> Result<MicroAnalyzeExperimentReport> {
    let selected_prompts = prompts.iter().take(config.prompt_count).collect::<Vec<_>>();
    let start = Instant::now();
    let temp_root = create_package_temp_dir(&model.path, "micro-analyze-")?;

    let mut mass_by_expert: BTreeMap<u32, f64> = BTreeMap::new();
    let mut selection_count_by_expert: BTreeMap<u32, u64> = BTreeMap::new();

    for (idx, prompt) in selected_prompts.iter().enumerate() {
        let output_path = temp_root.path().join(format!("prompt-{idx}.csv"));
        run_micro_analyze_export(
            model,
            prompt,
            &output_path,
            config.tokens,
            config.all_layers,
        )?;
        let partial = load_analyze_mass_profile(&output_path)?;
        for entry in partial.entries {
            *mass_by_expert.entry(entry.expert_id).or_insert(0.0) += entry.gate_mass;
            *selection_count_by_expert
                .entry(entry.expert_id)
                .or_insert(0) += entry.selection_count;
        }
    }

    let mut entries = mass_by_expert
        .into_iter()
        .map(|(expert_id, gate_mass)| AnalyzeExpertMass {
            expert_id,
            gate_mass,
            mass_pct: 0.0,
            selection_count: selection_count_by_expert
                .get(&expert_id)
                .copied()
                .unwrap_or(0),
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| {
        b.gate_mass
            .partial_cmp(&a.gate_mass)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.expert_id.cmp(&b.expert_id))
    });
    let total_mass = entries.iter().map(|entry| entry.gate_mass).sum::<f64>();
    for entry in &mut entries {
        entry.mass_pct = if total_mass <= f64::EPSILON {
            0.0
        } else {
            100.0 * entry.gate_mass / total_mass
        };
    }
    let ranking = entries
        .iter()
        .map(|entry| entry.expert_id)
        .collect::<Vec<_>>();
    let elapsed = start.elapsed();

    Ok(MicroAnalyzeExperimentReport {
        name: config.name.to_string(),
        prompt_count: selected_prompts.len(),
        tokens: config.tokens,
        all_layers: config.all_layers,
        runtime_seconds: elapsed.as_secs_f64(),
        spearman_rank_correlation: spearman_rank_correlation(&ranking, truth),
        recall_at_min_experts: recall_at_top_n(
            &ranking,
            &truth.ranking(),
            model.min_experts as usize,
        ),
        weighted_recall_at_min_experts: weighted_recall_at_top_n(
            &ranking,
            truth,
            model.min_experts as usize,
        ),
        captures_top_truth_expert: truth
            .entries
            .first()
            .map(|entry| {
                ranking
                    .iter()
                    .take(model.min_experts as usize)
                    .any(|expert| *expert == entry.expert_id)
            })
            .unwrap_or(false),
        ranking_preview: ranking.iter().take(16).copied().collect(),
    })
}

fn run_micro_analyze_export(
    model: &ResolvedBenchmarkModel,
    prompt: &str,
    output_path: &Path,
    tokens: u32,
    all_layers: bool,
) -> Result<()> {
    let analyze_bin = resolve_analyze_binary()?;
    let mut command = Command::new(&analyze_bin);
    command.args([
        "-m",
        &model.path.to_string_lossy(),
        "--export-ranking",
        &output_path.to_string_lossy(),
        "-n",
        &tokens.to_string(),
        "-c",
        "4096",
        "-ngl",
        "99",
        "-p",
        prompt,
    ]);
    if all_layers {
        command.arg("--all-layers");
    }

    let output = command
        .output()
        .with_context(|| format!("Run micro analyze for {}", model.path.display()))?;
    if !output.status.success() {
        bail!(
            "llama-moe-analyze micro run failed for {}: {}",
            model.path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn write_json_report<T: Serialize>(report: &T, output: Option<&Path>, label: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Create benchmark output directory {}", parent.display())
            })?;
        }
        std::fs::write(path, json)
            .with_context(|| format!("Write benchmark report to {}", path.display()))?;
        eprintln!("📝 Wrote {label} report to {}", path.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_profile() -> AnalyzeMassProfile {
        let entries = vec![
            AnalyzeExpertMass {
                expert_id: 0,
                gate_mass: 50.0,
                mass_pct: 50.0,
                selection_count: 10,
            },
            AnalyzeExpertMass {
                expert_id: 1,
                gate_mass: 30.0,
                mass_pct: 30.0,
                selection_count: 7,
            },
            AnalyzeExpertMass {
                expert_id: 2,
                gate_mass: 20.0,
                mass_pct: 20.0,
                selection_count: 5,
            },
        ];
        let mut mass_by_expert = HashMap::new();
        for entry in &entries {
            mass_by_expert.insert(entry.expert_id, entry.gate_mass);
        }
        AnalyzeMassProfile {
            entries,
            mass_by_expert,
            total_mass: 100.0,
        }
    }

    #[test]
    fn weighted_recall_prefers_hot_experts() {
        let profile = fixture_profile();
        let candidate = vec![0, 2, 1];
        assert!((weighted_recall_at_top_n(&candidate, &profile, 2) - 0.625).abs() < 1e-9);
    }

    #[test]
    fn spearman_is_one_for_identical_ranking() {
        let profile = fixture_profile();
        assert!((spearman_rank_correlation(&[0, 1, 2], &profile) - 1.0).abs() < 1e-9);
    }
}
