use std::collections::HashMap;

use anyhow::{Context, Result};
use futures::StreamExt;
use hf_hub::HFClient;
use hf_hub::repository::RepoTreeEntry;
use model_ref::{
    gguf_matches_quant_selector, normalize_gguf_distribution_id, quant_selector_from_gguf_file,
    split_gguf_shard_info,
};
use serde::Serialize;

use crate::jobs::{CpuJobPlan, JobSpec, JobVolume};
use crate::permissions::PermissionCheck;

/// Parameters for a model-package job.
pub struct PrepareParams {
    pub source_repo: String,
    pub source_revision: Option<String>,
    pub quant: Option<String>,
    pub target: Option<String>,
    pub model_id: Option<String>,
    pub flavor: String,
    pub timeout_seconds: u64,
    pub mesh_llm_ref: String,
    pub experimental: bool,
    pub hf_token: Option<String>,
}

/// A fully resolved model-package job, ready to submit.
pub struct PrepareJob {
    pub source_repo: String,
    pub source_revision: String,
    pub source_file: String,
    pub projectors: Vec<DiscoveredProjector>,
    pub target_repo: String,
    pub model_id: String,
    pub namespace: String,
    pub catalog_create_pr: bool,
    pub experimental: bool,
    pub job_plan: CpuJobPlan,
    pub spec: JobSpec,
}

/// A discovered quant variant in a HF model repo.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredQuant {
    /// The quant selector name (e.g. "Q4_K_M", "UD-Q4_K_XL").
    pub name: String,
    /// Number of GGUF files (shards) for this quant.
    pub shard_count: usize,
    /// Total size in bytes across all shards.
    pub total_bytes: u64,
    /// The first shard file path (or single file path).
    pub first_file: String,
}

/// A multimodal projector sidecar discovered in a HF model repo.
#[derive(Debug, Clone, Serialize, Eq, PartialEq)]
pub struct DiscoveredProjector {
    /// Repo-relative projector path.
    pub path: String,
    /// Projector size in bytes.
    pub total_bytes: u64,
}

/// GGUF artifacts discovered in a HF model repo, separated by runtime role.
#[derive(Debug, Clone, Default)]
pub struct RepoGgufInventory {
    pub quants: Vec<DiscoveredQuant>,
    pub projectors: Vec<DiscoveredProjector>,
}

/// List all available GGUF quant variants in a HF model repo.
pub async fn list_quants(client: &HFClient, repo: &str) -> Result<Vec<DiscoveredQuant>> {
    Ok(list_inventory(client, repo, None).await?.quants)
}

/// List model GGUF quants and multimodal projectors at an optional repo revision.
pub async fn list_inventory(
    client: &HFClient,
    repo: &str,
    revision: Option<&str>,
) -> Result<RepoGgufInventory> {
    let (owner, name) = parse_repo(repo)?;
    let hf_repo = client.model(&owner, &name);

    let stream = hf_repo
        .list_tree()
        .maybe_revision(revision.map(str::to_string))
        .recursive(true)
        .send()
        .context("list repo tree")?;

    futures::pin_mut!(stream);

    // Collect all GGUF files with their sizes.
    let mut gguf_files: Vec<(String, u64)> = Vec::new();
    while let Some(entry) = stream.next().await {
        let entry = entry.context("read repo tree entry")?;
        if let RepoTreeEntry::File { path, size, .. } = entry
            && path.ends_with(".gguf")
        {
            gguf_files.push((path, size));
        }
    }

    Ok(discover_inventory_from_gguf_files(gguf_files))
}

/// Separate model GGUF distributions from multimodal projector sidecars.
pub fn discover_inventory_from_gguf_files(gguf_files: Vec<(String, u64)>) -> RepoGgufInventory {
    let (projector_files, model_files): (Vec<_>, Vec<_>) = gguf_files
        .into_iter()
        .partition(|(path, _)| is_projector_path(path));
    let mut projectors = projector_files
        .into_iter()
        .map(|(path, total_bytes)| DiscoveredProjector { path, total_bytes })
        .collect::<Vec<_>>();
    projectors.sort_by(|a, b| a.path.cmp(&b.path));
    RepoGgufInventory {
        quants: discover_quants_from_gguf_files(model_files),
        projectors,
    }
}

fn is_projector_path(path: &str) -> bool {
    std::path::Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            let name = name.to_ascii_lowercase();
            name.starts_with("mmproj") && name.ends_with(".gguf")
        })
}

/// Group GGUF files into quant variants.
pub fn discover_quants_from_gguf_files(gguf_files: Vec<(String, u64)>) -> Vec<DiscoveredQuant> {
    let mut quant_map: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    for (path, size) in gguf_files {
        if let Some(selector) = quant_selector_from_gguf_file(&path) {
            quant_map.entry(selector).or_default().push((path, size));
        } else if let Some(dist_id) = normalize_gguf_distribution_id(&path) {
            // Fallback: use the distribution ID as the selector.
            quant_map.entry(dist_id).or_default().push((path, size));
        }
    }

    let mut quants: Vec<DiscoveredQuant> = quant_map
        .into_iter()
        .map(|(name, mut files)| {
            // Sort so shard -00001 comes first.
            files.sort_by(|a, b| a.0.cmp(&b.0));
            let total_bytes = files.iter().map(|(_, size)| size).sum();
            let shard_count = files.len();
            let first_file = files[0].0.clone();
            DiscoveredQuant {
                name,
                shard_count,
                total_bytes,
                first_file,
            }
        })
        .collect();

    // Sort by name for stable output.
    quants.sort_by(|a, b| a.name.cmp(&b.name));
    quants
}

/// Resolve source files, permissions, target repo, and build the job spec.
pub async fn resolve(
    client: &HFClient,
    params: PrepareParams,
    permissions: &PermissionCheck,
) -> Result<PrepareJob> {
    let quant = params
        .quant
        .as_deref()
        .context("--quant is required when submitting a job")?;

    let (owner, name) = parse_repo(&params.source_repo)?;
    let hf_repo = client.model(&owner, &name);
    let requested_revision = params.source_revision.as_deref().unwrap_or("main");
    let source_info = hf_repo
        .info()
        .revision(requested_revision.to_string())
        .send()
        .await
        .with_context(|| {
            format!(
                "resolve source revision {}@{requested_revision}",
                params.source_repo
            )
        })?;
    let source_revision = source_info
        .sha
        .context("source repo info did not include a commit SHA")?;
    let source_pipeline_tag = source_info
        .pipeline_tag
        .unwrap_or_else(|| "text-generation".to_string());
    let inventory = list_inventory(client, &params.source_repo, Some(&source_revision)).await?;
    let quants = inventory.quants;

    if quants.is_empty() {
        anyhow::bail!("No GGUF files found in {}", params.source_repo);
    }

    // Find matching quant.
    let matched = quants
        .iter()
        .find(|q| q.name.eq_ignore_ascii_case(quant))
        .or_else(|| {
            // Fall back to gguf_matches_quant_selector on the first file.
            quants
                .iter()
                .find(|q| gguf_matches_quant_selector(&q.first_file, quant))
        })
        .with_context(|| {
            let available: Vec<&str> = quants.iter().map(|q| q.name.as_str()).collect();
            format!(
                "No quant matching '{}' in {}.\nAvailable: {}",
                quant,
                params.source_repo,
                available.join(", ")
            )
        })?;

    // For sharded models, ensure we have the first shard.
    let source_file = if let Some(shard) = split_gguf_shard_info(&matched.first_file) {
        // Verify it's shard 00001.
        if shard.part != "00001" {
            // Reconstruct the -00001- path.
            matched
                .first_file
                .replace(&format!("-{}-of-", shard.part), "-00001-of-")
        } else {
            matched.first_file.clone()
        }
    } else {
        matched.first_file.clone()
    };

    // Derive distribution ID and target repo.
    let dist_id =
        normalize_gguf_distribution_id(&source_file).unwrap_or_else(|| matched.name.clone());

    let target_repo = params
        .target
        .unwrap_or_else(|| format!("{}/{}-layers", permissions.namespace, dist_id));

    let model_id = resolve_model_id(
        params.model_id,
        &params.source_repo,
        &source_file,
        &matched.name,
    )?;

    let projector_bytes = inventory
        .projectors
        .iter()
        .try_fold(0u64, |total, projector| {
            total.checked_add(projector.total_bytes)
        })
        .context("source GGUF and projector sizes overflowed u64")?;
    let source_total_bytes = matched
        .total_bytes
        .checked_add(projector_bytes)
        .context("source GGUF and projector sizes overflowed u64")?;
    let job_plan = crate::jobs::plan_cpu_job(
        &crate::jobs::hf_endpoint(),
        &params.flavor,
        params.timeout_seconds,
        source_total_bytes,
    )
    .await?;

    // Build environment variables.
    let mut environment = HashMap::new();
    environment.insert("SOURCE_REPO".into(), params.source_repo.clone());
    environment.insert("SOURCE_FILE".into(), source_file.clone());
    environment.insert("SOURCE_QUANT".into(), matched.name.clone());
    environment.insert("SOURCE_TOTAL_BYTES".into(), source_total_bytes.to_string());
    environment.insert("TARGET_REPO".into(), target_repo.clone());
    environment.insert("MODEL_ID".into(), model_id.clone());
    environment.insert("SOURCE_REVISION".into(), source_revision.clone());
    environment.insert("SOURCE_PIPELINE_TAG".into(), source_pipeline_tag);
    if !inventory.projectors.is_empty() {
        environment.insert(
            "SOURCE_PROJECTOR_FILES".into(),
            inventory
                .projectors
                .iter()
                .map(|projector| projector.path.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    environment.insert("MESH_LLM_REF".into(), params.mesh_llm_ref.clone());
    let catalog_create_pr = params.experimental || permissions.catalog_create_pr;
    environment.insert(
        "CATALOG_CREATE_PR".into(),
        if catalog_create_pr { "true" } else { "false" }.into(),
    );
    environment.insert(
        "PACKAGE_EXPERIMENTAL".into(),
        if params.experimental { "true" } else { "false" }.into(),
    );

    // The HF Jobs API passes secrets as env vars inside the container.
    // Dry runs intentionally omit secrets so users can inspect cost/spec first.
    let mut secrets = HashMap::new();
    if let Some(hf_token) = params.hf_token {
        secrets.insert("HF_TOKEN".into(), hf_token);
    }

    let volumes = vec![
        JobVolume {
            volume_type: "bucket".into(),
            source: "meshllm/layer-split-output".into(),
            mount_path: "/bucket".into(),
            read_only: None,
            revision: None,
        },
        JobVolume {
            volume_type: "model".into(),
            source: params.source_repo.clone(),
            mount_path: "/source".into(),
            read_only: Some(true),
            revision: Some(source_revision.clone()),
        },
    ];

    let spec = JobSpec {
        docker_image: "ubuntu:22.04".into(),
        command: vec!["bash".into(), "/bucket/split-model-job.sh".into()],
        arguments: vec![],
        environment,
        secrets,
        flavor: job_plan.flavor.clone(),
        timeout_seconds: job_plan.timeout_seconds,
        volumes,
    };

    Ok(PrepareJob {
        source_repo: params.source_repo,
        source_revision,
        source_file,
        projectors: inventory.projectors,
        target_repo,
        model_id,
        namespace: permissions.namespace.clone(),
        catalog_create_pr,
        experimental: params.experimental,
        job_plan,
        spec,
    })
}

/// Format a byte count as a human-readable size.
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn parse_repo(repo: &str) -> Result<(String, String)> {
    let parts: Vec<&str> = repo.splitn(2, '/').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid repo format: '{}'. Expected 'owner/name'.", repo);
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

fn resolve_model_id(
    explicit_model_id: Option<String>,
    source_repo: &str,
    source_file: &str,
    quant_name: &str,
) -> Result<String> {
    if let Some(model_id) = explicit_model_id {
        model_ref::ModelRef::parse(&model_id)
            .with_context(|| format!("invalid --model-id {model_id:?}"))?;
        return Ok(model_id);
    }

    Ok(model_ref::format_gguf_selection_ref(
        source_repo,
        source_file,
        quant_name,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_sharded_quant_files() {
        let quants = discover_quants_from_gguf_files(vec![
            (
                "UD-Q4_K_XL/Qwen3-32B-UD-Q4_K_XL-00002-of-00002.gguf".to_string(),
                20,
            ),
            (
                "UD-Q4_K_XL/Qwen3-32B-UD-Q4_K_XL-00001-of-00002.gguf".to_string(),
                10,
            ),
        ]);

        assert_eq!(quants.len(), 1);
        assert_eq!(quants[0].name, "UD-Q4_K_XL");
        assert_eq!(quants[0].shard_count, 2);
        assert_eq!(quants[0].total_bytes, 30);
        assert!(quants[0].first_file.ends_with("00001-of-00002.gguf"));
    }

    #[test]
    fn groups_root_quant_files_by_selector() {
        let quants = discover_quants_from_gguf_files(vec![
            ("Qwen3-8B-Q4_K_M.gguf".to_string(), 5),
            ("Qwen3-8B-Q8_0.gguf".to_string(), 9),
        ]);

        let names = quants
            .iter()
            .map(|quant| quant.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["Q4_K_M", "Q8_0"]);
    }

    #[test]
    fn separates_multimodal_projectors_from_model_quants() {
        let inventory = discover_inventory_from_gguf_files(vec![
            (
                "UD-Q2_K_XL/Inkling-UD-Q2_K_XL-00001-of-00008.gguf".to_string(),
                317,
            ),
            ("mmproj-BF16.gguf".to_string(), 183),
        ]);

        assert_eq!(inventory.quants.len(), 1);
        assert_eq!(inventory.quants[0].name, "UD-Q2_K_XL");
        assert_eq!(
            inventory.projectors,
            vec![DiscoveredProjector {
                path: "mmproj-BF16.gguf".to_string(),
                total_bytes: 183,
            }]
        );
    }

    #[test]
    fn accepts_explicit_model_id_coordinate() {
        let model_id = resolve_model_id(
            Some("unsloth/gemma-4-E4B-it-GGUF:Q4_K_M".to_string()),
            "unsloth/gemma-4-E4B-it-GGUF",
            "gemma-4-E4B-it-Q4_K_M.gguf",
            "Q4_K_M",
        )
        .expect("valid model id");

        assert_eq!(model_id, "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M");
    }

    #[test]
    fn rejects_explicit_model_id_without_repo_coordinate() {
        let error = resolve_model_id(
            Some("gemma-4-E4B-it-Q4_K_M".to_string()),
            "unsloth/gemma-4-E4B-it-GGUF",
            "gemma-4-E4B-it-Q4_K_M.gguf",
            "Q4_K_M",
        )
        .expect_err("invalid model id should fail");

        assert!(
            error.to_string().contains("invalid --model-id"),
            "{error:#}"
        );
    }

    #[test]
    fn derives_model_id_from_selected_quant() {
        let model_id = resolve_model_id(
            None,
            "unsloth/gemma-4-E4B-it-GGUF",
            "gemma-4-E4B-it-Q4_K_M.gguf",
            "Q4_K_M",
        )
        .expect("derived model id");

        assert_eq!(model_id, "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M");
    }
}
