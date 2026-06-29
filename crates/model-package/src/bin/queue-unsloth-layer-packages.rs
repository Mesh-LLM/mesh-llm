use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use hf_hub::{
    HFClient, HFError, HFRepository, RepoTypeModel,
    repository::{AddSource, ModelInfo},
};
use model_package::jobs::{CpuJobPlan, HfJobsClient, JobInfo, JobSpec, JobStage, JobVolume};
use model_package::prepare::{self, DiscoveredQuant};
use model_package::script;
use serde::Serialize;
use serde_json::Value;

const DEFAULT_QUANT_PREFERENCE: &[&str] = &["UD-Q4_K_XL", "UD-Q4_K_M", "Q4_K_XL", "Q4_K_M"];
const DEFAULT_SPLIT_CANDIDATE_VRAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const RUNTIME_MODEL_FIT_HEADROOM_NUMERATOR: u64 = 11;
const RUNTIME_MODEL_FIT_HEADROOM_DENOMINATOR: u64 = 10;

#[derive(Debug, Clone)]
struct Args {
    author: String,
    search: String,
    recent_limit: usize,
    popular_limit: usize,
    max_jobs: usize,
    max_per_family: usize,
    target_namespace: String,
    job_namespace: String,
    flavor: String,
    timeout_seconds: u64,
    mesh_llm_ref: String,
    retry_queued_after: Duration,
    split_candidate_vram_bytes: u64,
    quant_preference: Vec<String>,
    wait_for_jobs: bool,
    job_poll_interval: Duration,
    catalog_direct: bool,
    confirm: bool,
    dry_run: bool,
}

#[derive(Debug, Clone)]
struct RankedModel {
    repo_id: String,
    downloads: u64,
    likes: u64,
    recent_rank: Option<usize>,
    popular_rank: Option<usize>,
}

#[derive(Debug, Clone)]
struct Candidate {
    model: RankedModel,
    quant: DiscoveredQuant,
    target_repo: String,
    model_layer_repos: Vec<String>,
    model_id: String,
    family: String,
}

#[derive(Debug, Clone)]
struct SubmittedJob {
    candidate: Candidate,
    info: JobInfo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SplitCompatibility {
    Compatible,
    Incompatible(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QueueStatus {
    Missing,
    Published { repo: String },
    Cataloged { repo: String },
    Queued { repo: String },
    Failed { repo: String },
    StaleQueued,
}

#[derive(Debug, Serialize)]
struct QueueMarker<'a> {
    schema_version: u32,
    queued_at: String,
    source_repo: &'a str,
    source_file: &'a str,
    quant: &'a str,
    target_repo: &'a str,
    model_id: &'a str,
    mesh_llm_ref: &'a str,
    github_run_url: Option<String>,
    recent_rank: Option<usize>,
    popular_rank: Option<usize>,
}

#[derive(Debug, Serialize)]
struct QueueFailureMarker<'a> {
    schema_version: u32,
    failed_at: String,
    source_repo: &'a str,
    source_file: &'a str,
    quant: &'a str,
    target_repo: &'a str,
    model_id: &'a str,
    mesh_llm_ref: &'a str,
    github_run_url: Option<String>,
    hf_job_id: &'a str,
    stage: String,
    message: Option<&'a str>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    if args.max_jobs == 0 {
        bail!("--max-jobs must be at least 1");
    }

    let hf_client = model_package::build_hf_client()?;
    let submitting = args.confirm && !args.dry_run;
    let jobs_client = if submitting {
        Some(HfJobsClient::from_env()?)
    } else {
        None
    };
    if submitting {
        ensure_bucket_script_current(&hf_client).await?;
    }

    let recent = list_ranked_models(
        &hf_client,
        &args.author,
        &args.search,
        "lastModified",
        args.recent_limit,
    )
    .await?;
    let popular = list_ranked_models(
        &hf_client,
        &args.author,
        &args.search,
        "downloads",
        args.popular_limit,
    )
    .await?;
    let ranked = merge_recent_and_popular(recent, popular);

    println!(
        "Compared {} candidate {} models.",
        ranked.len(),
        args.author
    );
    println!(
        "Preferred 4-bit quants: {}",
        args.quant_preference.join(", ")
    );
    println!(
        "Split candidates: selected quant requires more than {} with 10% runtime headroom.",
        prepare::format_size(args.split_candidate_vram_bytes)
    );
    println!(
        "Family diversity: at most {} queued model(s) per family.",
        args.max_per_family
    );
    println!(
        "Mode: {}.",
        if submitting {
            "confirmed submit"
        } else {
            "dry run (use --confirm to submit HF Jobs)"
        }
    );

    let mut candidates = Vec::new();
    for model in ranked {
        if let Some(candidate) = build_candidate(&hf_client, model, &args).await? {
            candidates.push(candidate);
        }
    }
    candidates.sort_by_key(|candidate| {
        (
            candidate.quant.total_bytes,
            candidate.model.downloads,
            candidate.model.likes,
        )
    });
    candidates.reverse();
    println!(
        "Queue order: {} eligible split candidates by selected quant size descending, then family-capped.",
        candidates.len(),
    );

    let hardware = model_package::jobs::fetch_hardware(&model_package::jobs::hf_endpoint()).await?;
    let mut submitted = 0usize;
    let mut total_max_cost_usd = 0.0f64;
    let mut submitted_jobs = Vec::new();
    let mut submitted_by_family: HashMap<String, usize> = HashMap::new();
    for candidate in candidates {
        if submitted >= args.max_jobs {
            break;
        }

        let family_count = submitted_by_family
            .get(&candidate.family)
            .copied()
            .unwrap_or_default();
        if family_count >= args.max_per_family {
            println!(
                "skip {}: family {} already selected {} time(s)",
                candidate.model_id, candidate.family, args.max_per_family
            );
            continue;
        }
        let status = candidate_status(&hf_client, &candidate, args.retry_queued_after).await?;
        match status {
            QueueStatus::Published { repo } => {
                println!(
                    "skip {}: already has published layer package {}",
                    candidate.model.repo_id, repo
                );
                continue;
            }
            QueueStatus::Cataloged { repo } => {
                println!(
                    "skip {}: already has layer package {} in meshllm/catalog",
                    candidate.model.repo_id, repo
                );
                continue;
            }
            QueueStatus::Queued { repo } => {
                println!(
                    "skip {}: already recently queued layer package {}",
                    candidate.model.repo_id, repo
                );
                continue;
            }
            QueueStatus::Failed { repo } => {
                println!(
                    "skip {}: layer package job previously failed for {}",
                    candidate.model.repo_id, repo
                );
                continue;
            }
            QueueStatus::Missing | QueueStatus::StaleQueued => {}
        }

        let job_plan = model_package::jobs::plan_cpu_job_from_hardware(
            &hardware,
            &args.flavor,
            args.timeout_seconds,
            candidate.quant.total_bytes,
        )?;
        total_max_cost_usd += job_plan.max_cost_usd;

        let action = if submitting {
            "queueing"
        } else {
            "would queue"
        };
        println!(
            "{} {} -> {} ({}, source={}, bucket_estimate={}, {}, {}, family={}, target={}, hardware={} {}, timeout={}, max_cost={})",
            action,
            candidate.model_id,
            candidate.target_repo,
            candidate.quant.name,
            prepare::format_size(candidate.quant.total_bytes),
            prepare::format_size(estimated_bucket_workspace_bytes(
                candidate.quant.total_bytes
            )),
            shard_label(candidate.quant.shard_count),
            rank_label(&candidate.model),
            candidate.family,
            status_label(status),
            job_plan.flavor,
            hardware_label(&job_plan),
            format_duration(job_plan.timeout_seconds),
            format_cost(job_plan.max_cost_usd),
        );

        if !submitting {
            submitted += 1;
            *submitted_by_family
                .entry(candidate.family.clone())
                .or_default() += 1;
            continue;
        }

        write_queue_marker(&hf_client, &candidate, &args).await?;
        let jobs_client = jobs_client.as_ref().expect("jobs client initialized");
        let info = jobs_client
            .submit(
                &args.job_namespace,
                &job_spec(&candidate, &args, jobs_client, &job_plan)?,
            )
            .await?;
        println!(
            "submitted HF job {}: https://huggingface.co/jobs/{}/{}",
            info.id, args.job_namespace, info.id
        );
        submitted_jobs.push(SubmittedJob {
            candidate: candidate.clone(),
            info,
        });
        submitted += 1;
        *submitted_by_family
            .entry(candidate.family.clone())
            .or_default() += 1;
    }

    println!(
        "{} {} job(s).",
        if submitting {
            "Submitted"
        } else {
            "Would submit"
        },
        submitted
    );
    println!(
        "Maximum HF Jobs cost for this selection: {}",
        format_cost(total_max_cost_usd)
    );

    if args.wait_for_jobs && !submitted_jobs.is_empty() {
        let jobs_client = jobs_client.as_ref().expect("jobs client initialized");
        wait_for_submitted_jobs(&hf_client, jobs_client, &args, submitted_jobs).await?;
    }
    Ok(())
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self {
            author: "unsloth".to_string(),
            search: "GGUF".to_string(),
            recent_limit: 80,
            popular_limit: 80,
            max_jobs: 5,
            max_per_family: 1,
            target_namespace: "meshllm".to_string(),
            job_namespace: "meshllm".to_string(),
            flavor: "auto".to_string(),
            timeout_seconds: parse_duration_seconds("1h")?,
            mesh_llm_ref: "main".to_string(),
            retry_queued_after: Duration::from_secs(30 * 60 * 60),
            split_candidate_vram_bytes: DEFAULT_SPLIT_CANDIDATE_VRAM_BYTES,
            quant_preference: DEFAULT_QUANT_PREFERENCE
                .iter()
                .map(|value| value.to_string())
                .collect(),
            wait_for_jobs: false,
            job_poll_interval: Duration::from_secs(60),
            catalog_direct: true,
            confirm: false,
            dry_run: true,
        };

        let mut iter = std::env::args().skip(1);
        while let Some(flag) = iter.next() {
            match flag.as_str() {
                "--author" => args.author = next_value(&mut iter, &flag)?,
                "--search" => args.search = next_value(&mut iter, &flag)?,
                "--recent-limit" => args.recent_limit = parse_next(&mut iter, &flag)?,
                "--popular-limit" => args.popular_limit = parse_next(&mut iter, &flag)?,
                "--max-jobs" => args.max_jobs = parse_next(&mut iter, &flag)?,
                "--max-per-family" => args.max_per_family = parse_next(&mut iter, &flag)?,
                "--target-namespace" => args.target_namespace = next_value(&mut iter, &flag)?,
                "--job-namespace" => args.job_namespace = next_value(&mut iter, &flag)?,
                "--flavor" => args.flavor = next_value(&mut iter, &flag)?,
                "--timeout" => {
                    args.timeout_seconds = parse_duration_seconds(&next_value(&mut iter, &flag)?)?
                }
                "--mesh-llm-ref" => args.mesh_llm_ref = next_value(&mut iter, &flag)?,
                "--retry-queued-after-hours" => {
                    let hours: u64 = parse_next(&mut iter, &flag)?;
                    args.retry_queued_after = Duration::from_secs(hours * 60 * 60);
                }
                "--split-candidate-vram-gib" => {
                    let gib: f64 = parse_next(&mut iter, &flag)?;
                    if gib <= 0.0 {
                        bail!("--split-candidate-vram-gib must be greater than zero");
                    }
                    args.split_candidate_vram_bytes =
                        (gib * 1024.0 * 1024.0 * 1024.0).round() as u64;
                }
                "--quant-preference" => {
                    args.quant_preference = next_value(&mut iter, &flag)?
                        .split(',')
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                        .collect();
                }
                "--wait-for-jobs" => args.wait_for_jobs = true,
                "--job-poll-seconds" => {
                    let seconds: u64 = parse_next(&mut iter, &flag)?;
                    if seconds == 0 {
                        bail!("--job-poll-seconds must be at least 1");
                    }
                    args.job_poll_interval = Duration::from_secs(seconds);
                }
                "--catalog-direct" => args.catalog_direct = true,
                "--no-catalog-direct" => args.catalog_direct = false,
                "--confirm" => {
                    args.confirm = true;
                    args.dry_run = false;
                }
                "--dry-run" => {
                    args.confirm = false;
                    args.dry_run = true;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown argument: {other}"),
            }
        }

        if args.quant_preference.is_empty() {
            bail!("--quant-preference must include at least one quant");
        }
        if args.max_per_family == 0 {
            bail!("--max-per-family must be at least 1");
        }
        Ok(args)
    }
}

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    iter.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn parse_next<T>(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    next_value(iter, flag)?
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid value for {flag}: {err}"))
}

fn print_help() {
    println!(
        "queue-unsloth-layer-packages\n\n\
           Options:\n\
           --author HF_AUTHOR_OR_ORG\n\
           --search QUERY\n\
           --max-jobs N\n\
           --max-per-family N\n\
           --confirm\n\
           --dry-run\n\
           --mesh-llm-ref REF\n\
           --quant-preference CSV\n\
           --recent-limit N\n\
           --popular-limit N\n\
           --target-namespace NAME\n\
           --job-namespace NAME\n\
           --flavor HF_JOB_FLAVOR (default: auto CPU)\n\
           --timeout DURATION (requested; raised by size-based minimum)\n\
           --wait-for-jobs\n\
           --job-poll-seconds N\n\
           --split-candidate-vram-gib GiB\n\
           --no-catalog-direct"
    );
}

async fn ensure_bucket_script_current(client: &HFClient) -> Result<()> {
    match script::check_bucket_script(client).await {
        Ok(freshness) if freshness.is_current => Ok(()),
        Ok(freshness) => {
            eprintln!(
                "Bucket script is out of date ({}); updating it now...",
                freshness
                    .mismatch_reason
                    .as_deref()
                    .unwrap_or("embedded script differs from bucket script")
            );
            script::update_bucket_script(client).await?;
            eprintln!("Bucket script updated.");
            Ok(())
        }
        Err(err) => {
            eprintln!(
                "Could not check bucket script freshness ({err:#}); uploading current script..."
            );
            script::update_bucket_script(client).await?;
            eprintln!("Bucket script updated.");
            Ok(())
        }
    }
}

async fn wait_for_submitted_jobs(
    hf_client: &HFClient,
    jobs_client: &HfJobsClient,
    args: &Args,
    submitted_jobs: Vec<SubmittedJob>,
) -> Result<()> {
    let mut pending = submitted_jobs
        .into_iter()
        .map(|job| (job.info.id.clone(), job))
        .collect::<HashMap<_, _>>();
    let mut failures = Vec::new();

    println!(
        "Monitoring {} HF job(s) every {}s.",
        pending.len(),
        args.job_poll_interval.as_secs()
    );

    while !pending.is_empty() {
        let mut finished = Vec::new();

        for (job_id, submitted) in &pending {
            let info = jobs_client
                .inspect(&args.job_namespace, job_id)
                .await
                .with_context(|| format!("inspect HF job {job_id}"))?;
            let stage = info.status.stage;
            println!(
                "HF job {} for {}: {}{}",
                job_id,
                submitted.candidate.model_id,
                stage,
                status_message_suffix(info.status.message.as_deref())
            );

            if stage.is_success() {
                finished.push(job_id.clone());
                continue;
            }
            if stage.is_terminal() {
                let failure = format!(
                    "HF job {} for {} finished unsuccessfully: {}{}",
                    job_id,
                    submitted.candidate.model_id,
                    stage,
                    status_message_suffix(info.status.message.as_deref())
                );
                eprintln!("{failure}");
                if let Err(err) =
                    write_queue_failure_marker(hf_client, submitted, args, &info).await
                {
                    eprintln!(
                        "failed to write queue failure marker for {}: {err:#}",
                        submitted.candidate.target_repo
                    );
                }
                failures.push(failure);
                finished.push(job_id.clone());
            }
            if stage == JobStage::Unknown {
                println!(
                    "HF job {} reported an unknown non-terminal stage; continuing to poll.",
                    job_id
                );
            }
        }

        for job_id in finished {
            pending.remove(&job_id);
        }
        if pending.is_empty() {
            break;
        }

        tokio::time::sleep(args.job_poll_interval).await;
    }

    if !failures.is_empty() {
        bail!(
            "{} HF job(s) finished unsuccessfully:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    println!("All submitted HF jobs completed successfully.");
    Ok(())
}

fn status_message_suffix(message: Option<&str>) -> String {
    message
        .filter(|message| !message.trim().is_empty())
        .map(|message| format!(" ({message})"))
        .unwrap_or_default()
}

async fn list_ranked_models(
    client: &HFClient,
    author: &str,
    search: &str,
    sort: &str,
    limit: usize,
) -> Result<Vec<RankedModel>> {
    let stream = client
        .list_models()
        .author(author.to_string())
        .search(search.to_string())
        .sort(sort.to_string())
        .full(false)
        .limit(limit)
        .send()
        .with_context(|| format!("list {author} models sorted by {sort}"))?;
    tokio::pin!(stream);

    let mut models = Vec::new();
    while let Some(info) = stream.next().await {
        let info = info?;
        if !info.id.ends_with("-GGUF") {
            continue;
        }
        models.push(RankedModel {
            repo_id: info.id,
            downloads: info.downloads.unwrap_or_default(),
            likes: info.likes.unwrap_or_default(),
            recent_rank: None,
            popular_rank: None,
        });
    }
    Ok(models)
}

fn merge_recent_and_popular(
    recent: Vec<RankedModel>,
    popular: Vec<RankedModel>,
) -> Vec<RankedModel> {
    let mut merged: HashMap<String, RankedModel> = HashMap::new();

    for (index, mut model) in recent.into_iter().enumerate() {
        model.recent_rank = Some(index + 1);
        merged.insert(model.repo_id.clone(), model);
    }

    for (index, model) in popular.into_iter().enumerate() {
        match merged.get_mut(&model.repo_id) {
            Some(existing) => {
                existing.downloads = existing.downloads.max(model.downloads);
                existing.likes = existing.likes.max(model.likes);
                existing.popular_rank = Some(index + 1);
            }
            None => {
                let mut model = model;
                model.popular_rank = Some(index + 1);
                merged.insert(model.repo_id.clone(), model);
            }
        }
    }

    let mut models = merged.into_values().collect::<Vec<_>>();
    models.sort_by_key(|model| {
        let recent_score = model
            .recent_rank
            .map(|rank| 10_000usize.saturating_sub(rank))
            .unwrap_or_default();
        let popular_score = model
            .popular_rank
            .map(|rank| 10_000usize.saturating_sub(rank))
            .unwrap_or_default();
        (recent_score + popular_score, model.downloads, model.likes)
    });
    models.reverse();
    models
}

async fn build_candidate(
    client: &HFClient,
    model: RankedModel,
    args: &Args,
) -> Result<Option<Candidate>> {
    let Some(source_info) = model_repo_info(client, &model.repo_id).await? else {
        eprintln!("skip {}: source repo no longer exists", model.repo_id);
        return Ok(None);
    };
    if let SplitCompatibility::Incompatible(reason) = model_split_compatibility(&source_info) {
        eprintln!("skip {}: {reason}", model.repo_id);
        return Ok(None);
    }

    let quants = match prepare::list_quants(client, &model.repo_id).await {
        Ok(quants) => quants,
        Err(err) => {
            eprintln!(
                "skip {}: failed to list GGUF quants: {err:#}",
                model.repo_id
            );
            return Ok(None);
        }
    };

    let Some(quant) = select_preferred_quant(&quants, &args.quant_preference) else {
        let available = quants
            .iter()
            .map(|quant| quant.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "skip {}: no preferred 4-bit quant ({}); available: {}",
            model.repo_id,
            args.quant_preference.join(", "),
            if available.is_empty() {
                "none"
            } else {
                &available
            }
        );
        return Ok(None);
    };

    let required_bytes = runtime_model_required_bytes(quant.total_bytes);
    if required_bytes <= args.split_candidate_vram_bytes {
        eprintln!(
            "skip {}:{}: fits {} comfortably (requires {}, threshold {})",
            model.repo_id,
            quant.name,
            prepare::format_size(args.split_candidate_vram_bytes),
            prepare::format_size(required_bytes),
            prepare::format_size(args.split_candidate_vram_bytes)
        );
        return Ok(None);
    }

    let target_repo = layer_target_repo(&quant, &args.target_namespace);
    let model_layer_repos = model_layer_repos(&quants, &args.target_namespace);
    let model_id =
        model_ref::format_gguf_selection_ref(&model.repo_id, &quant.first_file, &quant.name);

    let family = model_family_key(&model.repo_id);

    Ok(Some(Candidate {
        model,
        quant,
        target_repo,
        model_layer_repos,
        model_id,
        family,
    }))
}

fn model_layer_repos(quants: &[DiscoveredQuant], target_namespace: &str) -> Vec<String> {
    quants
        .iter()
        .map(|quant| layer_target_repo(quant, target_namespace))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn layer_target_repo(quant: &DiscoveredQuant, target_namespace: &str) -> String {
    let distribution_id =
        model_ref::normalize_gguf_distribution_id(&quant.first_file).unwrap_or(quant.name.clone());
    format!("{target_namespace}/{distribution_id}-layers")
}

fn select_preferred_quant(
    quants: &[DiscoveredQuant],
    quant_preference: &[String],
) -> Option<DiscoveredQuant> {
    quant_preference.iter().find_map(|preferred| {
        quants
            .iter()
            .find(|quant| quant.name.eq_ignore_ascii_case(preferred))
            .cloned()
    })
}

async fn candidate_status(
    client: &HFClient,
    candidate: &Candidate,
    retry_queued_after: Duration,
) -> Result<QueueStatus> {
    if let Some(repo) = catalog_layer_package_repo(client, candidate).await? {
        return Ok(QueueStatus::Cataloged { repo });
    }

    let mut exact_status = QueueStatus::Missing;
    for repo in &candidate.model_layer_repos {
        let Some(repo_info) = model_repo_info(client, repo).await? else {
            continue;
        };

        let siblings = repo_info.siblings.unwrap_or_default();
        if siblings
            .iter()
            .any(|sibling| sibling.rfilename == "model-package.json")
        {
            return Ok(QueueStatus::Published { repo: repo.clone() });
        }

        let has_queue_marker = siblings
            .iter()
            .any(|sibling| sibling.rfilename == "automation/queue.json");
        let has_failure_marker = siblings
            .iter()
            .any(|sibling| sibling.rfilename == "automation/failure.json");
        if has_failure_marker && repo == &candidate.target_repo {
            return Ok(QueueStatus::Failed { repo: repo.clone() });
        }
        if has_queue_marker {
            let queued_recently = repo_info
                .last_modified
                .as_deref()
                .and_then(parse_hf_datetime)
                .map(|last_modified| {
                    Utc::now()
                        .signed_duration_since(last_modified)
                        .to_std()
                        .unwrap_or_default()
                        < retry_queued_after
                })
                .unwrap_or(true);
            if queued_recently {
                return Ok(QueueStatus::Queued { repo: repo.clone() });
            }
            if repo == &candidate.target_repo {
                exact_status = QueueStatus::StaleQueued;
            }
        }
    }

    Ok(exact_status)
}

fn model_split_compatibility(info: &ModelInfo) -> SplitCompatibility {
    if let Some(tag) = first_incompatible_media_tag(info) {
        return SplitCompatibility::Incompatible(format!(
            "unsupported media-generation tag '{tag}'; layer packages currently target text-generation GGUFs"
        ));
    }

    let Some(pipeline_tag) = model_pipeline_tag(info) else {
        return SplitCompatibility::Incompatible(
            "missing text-generation pipeline_tag".to_string(),
        );
    };
    if pipeline_tag.eq_ignore_ascii_case("text-generation") {
        return SplitCompatibility::Compatible;
    }

    SplitCompatibility::Incompatible(format!(
        "unsupported pipeline_tag '{pipeline_tag}'; layer packages currently require text-generation"
    ))
}

fn model_pipeline_tag(info: &ModelInfo) -> Option<String> {
    info.pipeline_tag
        .as_deref()
        .or_else(|| {
            info.transformers_info
                .as_ref()
                .and_then(|transformers| transformers.pipeline_tag.as_deref())
        })
        .map(str::to_string)
        .or_else(|| card_string_value(info.card_data.as_ref(), "pipeline_tag"))
}

fn first_incompatible_media_tag(info: &ModelInfo) -> Option<String> {
    collect_model_tags(info)
        .into_iter()
        .find(|tag| is_incompatible_media_tag(tag))
}

fn collect_model_tags(info: &ModelInfo) -> Vec<String> {
    let mut tags = info.tags.clone().unwrap_or_default();
    if let Some(card_tags) = info.card_data.as_ref().and_then(|card| card.get("tags")) {
        match card_tags {
            Value::Array(values) => {
                tags.extend(values.iter().filter_map(Value::as_str).map(str::to_string));
            }
            Value::String(tag) => tags.push(tag.clone()),
            _ => {}
        }
    }
    tags
}

fn card_string_value(card_data: Option<&Value>, key: &str) -> Option<String> {
    card_data?
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn is_incompatible_media_tag(tag: &str) -> bool {
    let tag = tag.to_ascii_lowercase();
    matches!(
        tag.as_str(),
        "image-to-video"
            | "text-to-video"
            | "video-to-video"
            | "image-text-to-video"
            | "audio-to-video"
            | "text-to-audio"
            | "video-to-audio"
            | "audio-to-audio"
            | "text-to-audio-video"
            | "image-to-audio-video"
            | "image-text-to-audio-video"
            | "image-to-image"
            | "text-to-image"
            | "image-generation"
    ) || tag.contains("diffusion")
}

async fn model_repo_info(client: &HFClient, repo_id: &str) -> Result<Option<ModelInfo>> {
    let (owner, name) = parse_repo(repo_id)?;
    let repo = client.model(owner, name);
    match repo.info().revision("main".to_string()).send().await {
        Ok(info) => Ok(Some(info)),
        Err(HFError::RepoNotFound { .. }) | Err(HFError::RevisionNotFound { .. }) => Ok(None),
        Err(HFError::Http { context }) if context.status.as_u16() == 404 => Ok(None),
        Err(err) => Err(err).with_context(|| format!("fetch target repo info for {repo_id}")),
    }
}

async fn catalog_layer_package_repo(
    client: &HFClient,
    candidate: &Candidate,
) -> Result<Option<String>> {
    let entry_path = catalog_entry_path(&candidate.model.repo_id)?;
    let dataset = client.dataset("meshllm", "catalog");
    let bytes = match dataset
        .download_file_to_bytes()
        .filename(entry_path.clone())
        .revision("main".to_string())
        .send()
        .await
    {
        Ok(bytes) => bytes,
        Err(HFError::EntryNotFound { .. }) | Err(HFError::RepoNotFound { .. }) => {
            return Ok(None);
        }
        Err(HFError::Http { context }) if context.status.as_u16() == 404 => return Ok(None),
        Err(err) => {
            return Err(err).with_context(|| format!("download catalog entry {entry_path}"));
        }
    };

    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse catalog entry {entry_path}"))?;
    let target_namespace = candidate
        .target_repo
        .split_once('/')
        .map(|(namespace, _)| namespace)
        .unwrap_or_default();
    Ok(json_layer_package_repo(&value, target_namespace))
}

fn json_layer_package_repo(value: &Value, target_namespace: &str) -> Option<String> {
    match value {
        Value::Object(map) => {
            if let Some(repo) = map.get("repo").and_then(Value::as_str).filter(|repo| {
                repo.strip_prefix(target_namespace)
                    .is_some_and(|suffix| suffix.starts_with('/'))
                    && repo.ends_with("-layers")
            }) {
                return Some(repo.to_string());
            }
            map.values()
                .find_map(|value| json_layer_package_repo(value, target_namespace))
        }
        Value::Array(items) => items
            .iter()
            .find_map(|value| json_layer_package_repo(value, target_namespace)),
        _ => None,
    }
}

async fn write_queue_marker(client: &HFClient, candidate: &Candidate, args: &Args) -> Result<()> {
    client
        .create_repository()
        .repo_id(candidate.target_repo.clone())
        .repo_type(RepoTypeModel)
        .exist_ok(true)
        .send()
        .await
        .with_context(|| format!("create target repo {}", candidate.target_repo))?;

    let marker = QueueMarker {
        schema_version: 1,
        queued_at: Utc::now().to_rfc3339(),
        source_repo: &candidate.model.repo_id,
        source_file: &candidate.quant.first_file,
        quant: &candidate.quant.name,
        target_repo: &candidate.target_repo,
        model_id: &candidate.model_id,
        mesh_llm_ref: &args.mesh_llm_ref,
        github_run_url: std::env::var("GITHUB_RUN_URL").ok(),
        recent_rank: candidate.model.recent_rank,
        popular_rank: candidate.model.popular_rank,
    };
    let bytes = serde_json::to_vec_pretty(&marker)?;
    let repo = model_repo(client, &candidate.target_repo)?;
    repo.upload_file()
        .source(AddSource::bytes(bytes))
        .path_in_repo("automation/queue.json")
        .commit_message(format!("Queue layer package for {}", candidate.model_id))
        .send()
        .await
        .with_context(|| format!("upload queue marker to {}", candidate.target_repo))?;
    Ok(())
}

async fn write_queue_failure_marker(
    client: &HFClient,
    submitted: &SubmittedJob,
    args: &Args,
    info: &JobInfo,
) -> Result<()> {
    let marker = QueueFailureMarker {
        schema_version: 1,
        failed_at: Utc::now().to_rfc3339(),
        source_repo: &submitted.candidate.model.repo_id,
        source_file: &submitted.candidate.quant.first_file,
        quant: &submitted.candidate.quant.name,
        target_repo: &submitted.candidate.target_repo,
        model_id: &submitted.candidate.model_id,
        mesh_llm_ref: &args.mesh_llm_ref,
        github_run_url: std::env::var("GITHUB_RUN_URL").ok(),
        hf_job_id: &info.id,
        stage: info.status.stage.to_string(),
        message: info.status.message.as_deref(),
    };
    let bytes = serde_json::to_vec_pretty(&marker)?;
    let repo = model_repo(client, &submitted.candidate.target_repo)?;
    repo.upload_file()
        .source(AddSource::bytes(bytes))
        .path_in_repo("automation/failure.json")
        .commit_message(format!(
            "Record failed layer package job for {}",
            submitted.candidate.model_id
        ))
        .send()
        .await
        .with_context(|| {
            format!(
                "upload queue failure marker to {}",
                submitted.candidate.target_repo
            )
        })?;
    Ok(())
}

fn job_spec(
    candidate: &Candidate,
    args: &Args,
    jobs_client: &HfJobsClient,
    job_plan: &CpuJobPlan,
) -> Result<JobSpec> {
    job_spec_with_token(candidate, args, jobs_client.token(), job_plan)
}

fn job_spec_with_token(
    candidate: &Candidate,
    args: &Args,
    hf_token: &str,
    job_plan: &CpuJobPlan,
) -> Result<JobSpec> {
    let mut environment = HashMap::new();
    environment.insert("SOURCE_REPO".into(), candidate.model.repo_id.clone());
    environment.insert("SOURCE_FILE".into(), candidate.quant.first_file.clone());
    environment.insert("SOURCE_QUANT".into(), candidate.quant.name.clone());
    environment.insert(
        "SOURCE_TOTAL_BYTES".into(),
        candidate.quant.total_bytes.to_string(),
    );
    environment.insert("TARGET_REPO".into(), candidate.target_repo.clone());
    environment.insert("MODEL_ID".into(), candidate.model_id.clone());
    environment.insert("SOURCE_REVISION".into(), "main".into());
    environment.insert("MESH_LLM_REF".into(), args.mesh_llm_ref.clone());
    environment.insert(
        "CATALOG_CREATE_PR".into(),
        if args.catalog_direct { "false" } else { "true" }.into(),
    );

    let mut secrets = HashMap::new();
    secrets.insert("HF_TOKEN".into(), hf_token.to_string());

    Ok(JobSpec {
        docker_image: "ubuntu:22.04".into(),
        command: vec!["bash".into(), "/bucket/split-model-job.sh".into()],
        arguments: vec![],
        environment,
        secrets,
        flavor: job_plan.flavor.clone(),
        timeout_seconds: job_plan.timeout_seconds,
        volumes: vec![
            JobVolume {
                volume_type: "bucket".into(),
                source: "meshllm/layer-split-output".into(),
                mount_path: "/bucket".into(),
                read_only: None,
            },
            JobVolume {
                volume_type: "model".into(),
                source: candidate.model.repo_id.clone(),
                mount_path: "/source".into(),
                read_only: Some(true),
            },
        ],
    })
}

fn hardware_label(plan: &CpuJobPlan) -> String {
    match (plan.cpu.as_deref(), plan.ram.as_deref()) {
        (Some(cpu), Some(ram)) => format!("({cpu}, {ram})"),
        (Some(cpu), None) => format!("({cpu})"),
        (None, Some(ram)) => format!("({ram})"),
        (None, None) => String::new(),
    }
}

fn format_cost(value: f64) -> String {
    format!("${value:.2} USD")
}

fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    if hours > 0 && minutes > 0 {
        format!("{hours}h{minutes}m")
    } else if hours > 0 {
        format!("{hours}h")
    } else {
        format!("{}m", seconds / 60)
    }
}

fn parse_repo(repo_id: &str) -> Result<(&str, &str)> {
    repo_id
        .split_once('/')
        .filter(|(owner, name)| !owner.is_empty() && !name.is_empty())
        .with_context(|| format!("invalid Hugging Face repo id: {repo_id}"))
}

fn model_repo(client: &HFClient, repo_id: &str) -> Result<HFRepository<RepoTypeModel>> {
    let (owner, name) = parse_repo(repo_id)?;
    Ok(client.model(owner, name))
}

fn catalog_entry_path(source_repo: &str) -> Result<String> {
    let (owner, name) = parse_repo(source_repo)?;
    Ok(format!("entries/{owner}/{name}.json"))
}

fn rank_label(model: &RankedModel) -> String {
    let mut parts = Vec::new();
    if let Some(rank) = model.recent_rank {
        parts.push(format!("recent #{rank}"));
    }
    if let Some(rank) = model.popular_rank {
        parts.push(format!("popular #{rank}"));
    }
    if parts.is_empty() {
        "unranked".to_string()
    } else {
        parts.join(", ")
    }
}

fn model_family_key(repo_id: &str) -> String {
    let repo_name = repo_id.rsplit('/').next().unwrap_or(repo_id);
    let base_name = repo_name
        .strip_suffix("-GGUF")
        .or_else(|| repo_name.strip_suffix("-gguf"))
        .unwrap_or(repo_name);
    let lower = base_name.to_ascii_lowercase();
    let tokens = lower
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();

    if tokens.contains(&"kimi") {
        return "kimi".to_string();
    }
    if tokens.contains(&"deepseek") || tokens.windows(2).any(|window| window == ["deep", "seek"]) {
        return "deepseek".to_string();
    }
    if tokens
        .iter()
        .any(|token| token.starts_with("qwen") || *token == "qwq")
    {
        return "qwen".to_string();
    }
    if tokens.contains(&"llama") {
        return "llama".to_string();
    }
    if tokens.iter().any(|token| token.starts_with("gemma")) {
        return "gemma".to_string();
    }
    if tokens.iter().any(|token| token.starts_with("mistral")) {
        return "mistral".to_string();
    }
    if tokens.iter().any(|token| token.starts_with("mixtral")) {
        return "mixtral".to_string();
    }
    if tokens.contains(&"glm") {
        return "glm".to_string();
    }
    if tokens.contains(&"phi") {
        return "phi".to_string();
    }
    if tokens.contains(&"nemotron") {
        return "nemotron".to_string();
    }
    if tokens.windows(2).any(|window| window == ["gpt", "oss"]) {
        return "gpt-oss".to_string();
    }
    if tokens.windows(2).any(|window| window == ["seed", "oss"]) {
        return "seed-oss".to_string();
    }

    tokens
        .iter()
        .find(|token| !is_versionish_family_token(token))
        .copied()
        .unwrap_or(base_name)
        .to_string()
}

fn is_versionish_family_token(token: &str) -> bool {
    token
        .chars()
        .all(|ch| ch.is_ascii_digit() || matches!(ch, 'b' | 'm' | 'x' | 'v'))
}

fn shard_label(shard_count: usize) -> String {
    if shard_count == 1 {
        "1 file".to_string()
    } else {
        format!("{shard_count} shards")
    }
}

fn runtime_model_required_bytes(model_bytes: u64) -> u64 {
    model_bytes
        .saturating_mul(RUNTIME_MODEL_FIT_HEADROOM_NUMERATOR)
        .div_ceil(RUNTIME_MODEL_FIT_HEADROOM_DENOMINATOR)
}

fn estimated_bucket_workspace_bytes(source_bytes: u64) -> u64 {
    source_bytes
        .saturating_mul(9)
        .div_ceil(4)
        .saturating_add(32 * 1024 * 1024 * 1024)
}

fn status_label(status: QueueStatus) -> &'static str {
    match status {
        QueueStatus::Missing => "missing",
        QueueStatus::Published { .. } => "published",
        QueueStatus::Cataloged { .. } => "cataloged",
        QueueStatus::Queued { .. } => "queued",
        QueueStatus::Failed { .. } => "failed",
        QueueStatus::StaleQueued => "stale-queued",
    }
}

fn parse_hf_datetime(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|time| time.with_timezone(&Utc))
        .ok()
}

fn parse_duration_seconds(input: &str) -> Result<u64> {
    let input = input.trim();
    if input.is_empty() {
        bail!("duration cannot be empty");
    }
    if let Ok(seconds) = input.parse::<u64>() {
        return Ok(seconds);
    }

    let mut total = 0u64;
    let mut current = String::new();
    for ch in input.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
            continue;
        }
        let value = current
            .parse::<u64>()
            .with_context(|| format!("invalid duration: {input}"))?;
        current.clear();
        match ch {
            'd' | 'D' => total += value * 24 * 60 * 60,
            'h' | 'H' => total += value * 60 * 60,
            'm' | 'M' => total += value * 60,
            's' | 'S' => total += value,
            _ => bail!("invalid duration unit '{ch}' in {input}"),
        }
    }
    if !current.is_empty() {
        total += current.parse::<u64>()?;
    }
    if total == 0 {
        bail!("duration must be greater than zero: {input}");
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use model_package::jobs::CpuJobPlan;

    use hf_hub::repository::ModelInfo;

    use super::{
        Args, Candidate, DiscoveredQuant, RankedModel, SplitCompatibility,
        estimated_bucket_workspace_bytes, job_spec_with_token, json_layer_package_repo,
        model_family_key, model_layer_repos, model_split_compatibility,
    };

    fn model_info(value: serde_json::Value) -> ModelInfo {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn model_family_key_collapses_common_unsloth_families() {
        assert_eq!(model_family_key("unsloth/Kimi-K2-Instruct-GGUF"), "kimi");
        assert_eq!(
            model_family_key("unsloth/DeepSeek-V3.1-Terminus-GGUF"),
            "deepseek"
        );
        assert_eq!(
            model_family_key("unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF"),
            "deepseek"
        );
        assert_eq!(
            model_family_key("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"),
            "qwen"
        );
        assert_eq!(
            model_family_key("unsloth/Meta-Llama-3.1-70B-Instruct-GGUF"),
            "llama"
        );
        assert_eq!(model_family_key("unsloth/gpt-oss-120b-GGUF"), "gpt-oss");
        assert_eq!(
            model_family_key("unsloth/NVIDIA-Nemotron-3-Super-120B-GGUF"),
            "nemotron"
        );
    }

    #[test]
    fn split_compatibility_accepts_text_generation_models() {
        let info = model_info(serde_json::json!({
            "id": "unsloth/Qwen-AgentWorld-35B-A3B-GGUF",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "qwen", "text-generation"]
        }));

        assert_eq!(
            model_split_compatibility(&info),
            SplitCompatibility::Compatible
        );
    }

    #[test]
    fn split_compatibility_rejects_media_generation_pipeline() {
        let info = model_info(serde_json::json!({
            "id": "unsloth/LTX-2-GGUF",
            "pipeline_tag": "image-to-video",
            "tags": ["gguf", "image-to-video", "text-to-video"]
        }));

        let SplitCompatibility::Incompatible(reason) = model_split_compatibility(&info) else {
            panic!("LTX media model should not be split queued");
        };
        assert!(reason.contains("image-to-video"));
    }

    #[test]
    fn split_compatibility_rejects_media_generation_card_tags() {
        let info = model_info(serde_json::json!({
            "id": "unsloth/Qwen-Image-Edit-2511-GGUF",
            "pipeline_tag": "text-generation",
            "cardData": {
                "tags": ["gguf", "image-to-image"]
            }
        }));

        let SplitCompatibility::Incompatible(reason) = model_split_compatibility(&info) else {
            panic!("image generation model should not be split queued");
        };
        assert!(reason.contains("image-to-image"));
    }

    #[test]
    fn split_compatibility_uses_model_card_pipeline_tag() {
        let info = model_info(serde_json::json!({
            "id": "unsloth/Text-Only-GGUF",
            "tags": ["gguf"],
            "cardData": {
                "pipeline_tag": "text-generation"
            }
        }));

        assert_eq!(
            model_split_compatibility(&info),
            SplitCompatibility::Compatible
        );
    }

    #[test]
    fn job_spec_uses_bucket_cache_without_model_volume() {
        let candidate = Candidate {
            model: RankedModel {
                repo_id: "unsloth/GLM-5-GGUF".to_string(),
                downloads: 0,
                likes: 0,
                recent_rank: None,
                popular_rank: None,
            },
            quant: DiscoveredQuant {
                name: "UD-Q4_K_XL".to_string(),
                shard_count: 10,
                total_bytes: 401,
                first_file: "UD-Q4_K_XL/GLM-5-UD-Q4_K_XL-00001-of-00010.gguf".to_string(),
            },
            target_repo: "meshllm/GLM-5-UD-Q4_K_XL-layers".to_string(),
            model_layer_repos: vec!["meshllm/GLM-5-UD-Q4_K_XL-layers".to_string()],
            model_id: "unsloth/GLM-5-GGUF:UD-Q4_K_XL".to_string(),
            family: "glm".to_string(),
        };
        let args = Args {
            author: "unsloth".to_string(),
            search: "GGUF".to_string(),
            recent_limit: 1,
            popular_limit: 1,
            max_jobs: 1,
            max_per_family: 1,
            target_namespace: "meshllm".to_string(),
            job_namespace: "meshllm".to_string(),
            flavor: "cpu-upgrade".to_string(),
            timeout_seconds: 43_200,
            mesh_llm_ref: "main".to_string(),
            retry_queued_after: Duration::from_secs(1),
            split_candidate_vram_bytes: 8,
            quant_preference: vec!["UD-Q4_K_XL".to_string()],
            wait_for_jobs: true,
            job_poll_interval: Duration::from_secs(60),
            catalog_direct: true,
            confirm: true,
            dry_run: false,
        };
        let job_plan = CpuJobPlan {
            flavor: "cpu-upgrade".to_string(),
            pretty_name: "cpu-upgrade".to_string(),
            cpu: Some("8 vCPU".to_string()),
            ram: Some("32 GB".to_string()),
            unit_cost_usd: 0.0005,
            unit_label: "minute".to_string(),
            max_cost_usd: 0.36,
            timeout_seconds: 43_200,
            minimum_timeout_seconds: 43_200,
            requested_timeout_seconds: 43_200,
            timeout_bumped_to_minimum: false,
            auto_selected_hardware: true,
            selection_reason: "test".to_string(),
            model_size_bytes: 401,
        };

        let spec = job_spec_with_token(&candidate, &args, "hf_test", &job_plan).unwrap();

        assert_eq!(
            spec.environment.get("SOURCE_QUANT").map(String::as_str),
            Some("UD-Q4_K_XL")
        );
        assert_eq!(
            spec.environment
                .get("SOURCE_TOTAL_BYTES")
                .map(String::as_str),
            Some("401")
        );
        assert_eq!(spec.volumes.len(), 2);
        assert_eq!(spec.volumes[0].volume_type, "bucket");
        assert_eq!(spec.volumes[0].mount_path, "/bucket");
        assert_eq!(spec.volumes[1].volume_type, "model");
        assert_eq!(spec.volumes[1].source, candidate.model.repo_id);
        assert_eq!(spec.volumes[1].mount_path, "/source");
        assert_eq!(spec.volumes[1].read_only, Some(true));
    }

    #[test]
    fn bucket_workspace_estimate_scales_with_source_size() {
        assert_eq!(
            estimated_bucket_workspace_bytes(600 * 1024 * 1024 * 1024),
            1382 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn model_layer_repos_include_every_quant_distribution() {
        let repos = model_layer_repos(
            &[
                DiscoveredQuant {
                    name: "UD-Q4_K_XL".to_string(),
                    shard_count: 2,
                    total_bytes: 200,
                    first_file: "UD-Q4_K_XL/Kimi-K2-Instruct-UD-Q4_K_XL-00001-of-00002.gguf"
                        .to_string(),
                },
                DiscoveredQuant {
                    name: "Q4_K_M".to_string(),
                    shard_count: 1,
                    total_bytes: 100,
                    first_file: "Kimi-K2-Instruct-Q4_K_M.gguf".to_string(),
                },
            ],
            "meshllm",
        );

        assert_eq!(
            repos,
            vec![
                "meshllm/Kimi-K2-Instruct-Q4_K_M-layers",
                "meshllm/Kimi-K2-Instruct-UD-Q4_K_XL-layers"
            ]
        );
    }

    #[test]
    fn catalog_lookup_treats_any_layer_repo_as_model_coverage() {
        let catalog = serde_json::json!({
            "models": [{
                "variants": [{
                    "packages": [{
                        "type": "layer-package",
                        "repo": "meshllm/Kimi-K2-Instruct-Q4_K_M-layers"
                    }]
                }]
            }]
        });

        assert_eq!(
            json_layer_package_repo(&catalog, "meshllm"),
            Some("meshllm/Kimi-K2-Instruct-Q4_K_M-layers".to_string())
        );
    }
}
