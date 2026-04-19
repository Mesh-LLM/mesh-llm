use super::analyze::{
    full_analyze_artifacts, has_complete_full_analyze_artifacts, run_local_full_analysis,
    sync_local_package_cache,
};
use super::hf_jobs;
use super::model_card::{build_package_readme, load_source_model_card_metadata};
use super::upload::{
    batch_commit_description, batch_commit_message, build_upload_batches, contribute_catalog_entry,
    dataset_branch_head, download_repo_text, ensure_repo_ready, load_share_branch_state,
    parse_repo_id, repo_info_siblings_and_sha, resolve_publish_target, share_branch_name,
    stage_share_file, stage_variant_components, staged_upload_file, upload_share_batch_with_retry,
    ShareUploadBatch, ShareUploadProgress,
};
use super::{FULL_ANALYZE_CONTEXT_SIZE, FULL_ANALYZE_GPU_LAYERS};
use anyhow::{Context, Result};
use hf_hub::{CommitInfo, HFError, RepoCreateBranchParams, RepoInfoParams, RepoType};
use std::fs;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::moe::HfJobArgs;
use crate::inference::moe;
use crate::models;
use crate::system::{moe_benchmark, moe_planner};

pub(super) async fn run_publish(
    model: &str,
    catalog_repo: &str,
    namespace: Option<&str>,
    hf_job: &HfJobArgs,
) -> Result<()> {
    if hf_job.hf_job {
        return hf_jobs::submit_publish_job(model, catalog_repo, namespace, hf_job).await;
    }
    let share_error = |title: &str, detail: &str| -> anyhow::Error {
        eprintln!("❌ {title}");
        eprintln!("   {detail}");
        anyhow::anyhow!("{title}: {detail}")
    };

    let resolved = moe_planner::resolve_model_context(model).await?;
    let ranking = if has_complete_full_analyze_artifacts(&resolved) {
        eprintln!(
            "🧠 Reusing full-v1 MoE analysis from {}",
            full_analyze_artifacts(&resolved).ranking_path.display()
        );
        let artifacts = full_analyze_artifacts(&resolved);
        sync_local_package_cache(&resolved, &artifacts.ranking, Some(&artifacts.log_path))?;
        artifacts.ranking
    } else {
        eprintln!("🧠 Running full-v1 MoE analysis before publish");
        run_local_full_analysis(
            &resolved,
            FULL_ANALYZE_CONTEXT_SIZE,
            FULL_ANALYZE_GPU_LAYERS,
        )?
        .ranking
    };
    moe_planner::validate_ranking(&resolved, &ranking).with_context(|| {
        format!(
            "Validate ranking {} against model {}",
            ranking.path.display(),
            resolved.display_name
        )
    })?;
    let log_path = moe::package_cache_run_log_path(&resolved.path);
    let bundle = sync_local_package_cache(&resolved, &ranking, Some(log_path.as_path()))?;
    ensure_package_calibration_benchmark(&resolved, &ranking).await?;
    let package_ranking_path = moe::package_cache_ranking_path(&resolved.path);
    models::hf_token_override().ok_or_else(|| {
        share_error(
            "Missing Hugging Face token",
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running `mesh-llm moe publish`.",
        )
    })?;
    let api =
        models::build_hf_tokio_api(false).context("Build Hugging Face client for MoE publish")?;
    let publish_target = resolve_publish_target(&api, &resolved, namespace)
        .await
        .map_err(|err| {
            share_error(
                "Failed to resolve package repository target",
                &err.to_string(),
            )
        })?;
    ensure_repo_ready(&api, &publish_target.package_repo, RepoType::Model)
        .await
        .map_err(|err| {
            share_error(
                "Failed to prepare package repository",
                &format!(
                    "Ensure {} exists and is ready: {}",
                    publish_target.package_repo, err
                ),
            )
        })?;
    let (package_owner, package_name) = parse_repo_id(&publish_target.package_repo)?;
    let package_repo = api.model(package_owner, package_name);
    let package_info = match package_repo
        .info(
            &RepoInfoParams::builder()
                .revision("main".to_string())
                .build(),
        )
        .await
    {
        Ok(info) => Some(info),
        Err(HFError::RevisionNotFound { .. }) => None,
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "Fetch package repo info for {}",
                    publish_target.package_repo
                )
            });
        }
    };
    let (package_siblings, package_main_head) = if let Some(info) = package_info.as_ref() {
        repo_info_siblings_and_sha(info)?
    } else {
        (Vec::new(), None)
    };

    println!("📤 MoE package publish");
    println!("📦 {}", resolved.display_name);
    println!("   ranking: {}", package_ranking_path.display());
    println!("   source: {}", ranking.source.label());
    println!("📚 Catalog");
    println!("   repo: {catalog_repo}");
    println!("📦 Package repo");
    println!("   repo: {}", publish_target.package_repo);
    println!("   model_ref: {}", bundle.model_ref);
    println!("   trust: {}", publish_target.trust);
    println!("   variant: {}", bundle.variant);

    let temp_root_base = moe::package_cache_temp_root(&resolved.path);
    fs::create_dir_all(&temp_root_base)?;
    let temp_root = temp_root_base.join(format!(
        "mesh-llm-moe-publish-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir_all(&temp_root)?;
    let _temp_root_guard = super::TempRootGuard(temp_root.clone());

    let existing_meshllm = if package_siblings
        .iter()
        .any(|entry| entry.rfilename == "meshllm.json")
    {
        Some(
            download_repo_text(&package_repo, "meshllm.json", "main")
                .await
                .with_context(|| {
                    format!(
                        "Download existing meshllm.json from {}",
                        publish_target.package_repo
                    )
                })?,
        )
    } else {
        None
    };
    let meshllm = moe_planner::build_meshllm_descriptor(
        existing_meshllm.as_deref(),
        &resolved,
        &bundle.manifest_repo_path,
    )?;
    let meshllm_cache_path = moe::package_cache_meshllm_path(&resolved.path);
    if let Some(parent) = meshllm_cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &meshllm_cache_path,
        serde_json::to_string_pretty(&meshllm)? + "\n",
    )
    .with_context(|| format!("Write {}", meshllm_cache_path.display()))?;
    let model_card_metadata =
        load_source_model_card_metadata(&api, &meshllm.source.repo, &meshllm.source.revision)
            .await
            .with_context(|| {
                format!(
                    "Load source model-card metadata from {}@{}",
                    meshllm.source.repo, meshllm.source.revision
                )
            })?;
    let readme = build_package_readme(&meshllm, &publish_target.package_repo, &model_card_metadata);
    let readme_cache_path = moe::package_cache_readme_path(&resolved.path);
    fs::write(&readme_cache_path, &readme)
        .with_context(|| format!("Write {}", readme_cache_path.display()))?;

    let mut staged_files = Vec::new();
    stage_share_file(&temp_root, "README.md", &readme_cache_path)?;
    staged_files.push(staged_upload_file(&temp_root, "README.md")?);
    stage_share_file(&temp_root, "meshllm.json", &meshllm_cache_path)?;
    staged_files.push(staged_upload_file(&temp_root, "meshllm.json")?);
    stage_share_file(
        &temp_root,
        &bundle.ranking_repo_path,
        &moe::package_cache_ranking_path(&resolved.path),
    )?;
    staged_files.push(staged_upload_file(&temp_root, &bundle.ranking_repo_path)?);
    stage_share_file(
        &temp_root,
        &bundle.analysis_repo_path,
        &moe::package_cache_analysis_path(&resolved.path),
    )?;
    staged_files.push(staged_upload_file(&temp_root, &bundle.analysis_repo_path)?);
    if let Some(log_repo_path) = bundle.log_repo_path.as_ref() {
        let package_log_path = moe::package_cache_run_log_path(&resolved.path);
        if package_log_path.exists() {
            stage_share_file(&temp_root, log_repo_path, &package_log_path)?;
            staged_files.push(staged_upload_file(&temp_root, log_repo_path)?);
        }
    }

    let current_exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = current_exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Current executable has no parent directory"))?;
    for path in stage_variant_components(&temp_root, &resolved, &ranking, &bundle, bin_dir)? {
        staged_files.push(staged_upload_file(&temp_root, &path)?);
    }

    let branch_name = share_branch_name(&format!(
        "{}/{}",
        publish_target.package_repo, bundle.variant_root
    ));
    let mut upload_branch = branch_name.clone();
    let mut use_direct_main = false;
    let branch_state = if package_main_head.is_some() {
        load_share_branch_state(&package_repo, &branch_name, &staged_files)
            .await
            .with_context(|| {
                format!(
                    "Inspect contribution branch {} for {}",
                    branch_name, publish_target.package_repo
                )
            })?
    } else {
        None
    };
    let mut branch_head = if let Some(state) = &branch_state {
        println!("🌿 Resuming contribution branch");
        println!("   branch: {branch_name}");
        println!(
            "   staged remotely: {}/{}",
            state.uploaded_paths.len(),
            staged_files.len()
        );
        state.head_commit.clone()
    } else if let Some(main_head) = package_main_head.clone() {
        println!("🌿 Creating contribution branch");
        println!("   branch: {branch_name}");
        package_repo
            .create_branch(
                &RepoCreateBranchParams::builder()
                    .branch(branch_name.clone())
                    .revision(main_head.clone())
                    .build(),
            )
            .await
            .map_err(|err| {
                share_error(
                    "Failed to create contribution branch",
                    &format!(
                        "Create branch {branch_name} on {}: {}",
                        publish_target.package_repo, err
                    ),
                )
            })?;
        main_head
    } else {
        use_direct_main = true;
        upload_branch = "main".to_string();
        println!("🌱 Initializing empty package repo");
        println!("   repo: {}", publish_target.package_repo);
        "main".to_string()
    };

    let mut completed_paths = branch_state
        .as_ref()
        .map(|state| state.uploaded_paths.clone())
        .unwrap_or_default();
    let mut completed_bytes = staged_files
        .iter()
        .filter(|file| completed_paths.contains(&file.repo_path))
        .map(|file| file.size_bytes)
        .sum::<u64>();

    if use_direct_main {
        let bootstrap_files = staged_files
            .iter()
            .filter(|file| matches!(file.repo_path.as_str(), "README.md" | "meshllm.json"))
            .cloned()
            .collect::<Vec<_>>();
        if !bootstrap_files.is_empty() {
            let bootstrap_batch = ShareUploadBatch {
                total_bytes: bootstrap_files.iter().map(|file| file.size_bytes).sum(),
                files: bootstrap_files,
            };
            let progress = Arc::new(ShareUploadProgress::new(
                staged_files.len(),
                staged_files.iter().map(|file| file.size_bytes).sum(),
            ));
            progress.begin_batch(
                1,
                1,
                completed_paths.len(),
                completed_bytes,
                &bootstrap_batch,
            );
            let bootstrap_commit = upload_share_batch_with_retry(
                &package_repo,
                None,
                None,
                &bootstrap_batch,
                &progress,
                format!("Initialize {}", publish_target.package_repo),
                format!(
                    "Initialize `{}` with the first Mesh-LLM package metadata.",
                    publish_target.package_repo
                ),
                false,
            )
            .await
            .map_err(|err| {
                share_error(
                    "Failed to initialize empty package repository",
                    &format!(
                        "Create the first commit in {}: {}",
                        publish_target.package_repo, err
                    ),
                )
            })?;
            for file in &bootstrap_batch.files {
                completed_paths.insert(file.repo_path.clone());
            }
            completed_bytes += bootstrap_batch.total_bytes;
            if let Some(commit_oid) = bootstrap_commit.commit_oid {
                branch_head = commit_oid;
            }
            println!("✅ Initialized empty package repo");
            println!("   branch: main");
        }
    }

    let pending_files = staged_files
        .iter()
        .filter(|file| !completed_paths.contains(&file.repo_path))
        .cloned()
        .collect::<Vec<_>>();

    if pending_files.is_empty() {
        println!("✅ Contribution branch already contains all staged files");
        println!("   branch: {upload_branch}");
        return Ok(());
    }

    let batches = build_upload_batches(&pending_files);
    let progress = Arc::new(ShareUploadProgress::new(
        staged_files.len(),
        staged_files.iter().map(|file| file.size_bytes).sum(),
    ));
    let mut final_commit: Option<CommitInfo> = None;
    if use_direct_main {
        println!(
            "⬆️ Uploading new package in {} remaining batch(es)...",
            batches.len()
        );
    } else {
        println!(
            "⬆️ Opening contribution PR in {} upload batch(es)...",
            batches.len()
        );
    }
    for (index, batch) in batches.iter().enumerate() {
        progress.begin_batch(
            index + 1,
            batches.len(),
            completed_paths.len(),
            completed_bytes,
            batch,
        );
        let batch_commit = upload_share_batch_with_retry(
            &package_repo,
            if use_direct_main && index == 0 {
                None
            } else {
                Some(upload_branch.as_str())
            },
            if use_direct_main && index == 0 {
                None
            } else {
                Some(branch_head.clone())
            },
            batch,
            &progress,
            batch_commit_message(&bundle.commit_message, index + 1, batches.len()),
            batch_commit_description(&bundle.commit_description, index + 1, batches.len()),
            !use_direct_main && index == 0,
        )
        .await
        .map_err(|err| {
            share_error(
                "Package upload failed",
                &format!(
                    "Upload staged files to {}: {}",
                    publish_target.package_repo, err
                ),
            )
        })?;
        for file in &batch.files {
            completed_paths.insert(file.repo_path.clone());
        }
        completed_bytes += batch.total_bytes;
        if !use_direct_main {
            if let Some(head_commit) = dataset_branch_head(&package_repo, &upload_branch)
                .await
                .with_context(|| format!("Refresh branch head for {}", upload_branch))?
            {
                branch_head = head_commit;
            } else if let Some(commit_oid) = batch_commit.commit_oid.clone() {
                branch_head = commit_oid;
            }
        } else if let Some(commit_oid) = batch_commit.commit_oid.clone() {
            branch_head = commit_oid;
        }
        final_commit = Some(match (final_commit.take(), batch_commit.pr_url.clone()) {
            (None, _) => batch_commit,
            (Some(mut current), Some(pr_url)) => {
                current.pr_url = Some(pr_url);
                if current.pr_num.is_none() {
                    current.pr_num = batch_commit.pr_num;
                }
                if batch_commit.commit_oid.is_some() {
                    current.commit_oid = batch_commit.commit_oid;
                }
                if batch_commit.commit_url.is_some() {
                    current.commit_url = batch_commit.commit_url;
                }
                current
            }
            (Some(mut current), None) => {
                if batch_commit.commit_oid.is_some() {
                    current.commit_oid = batch_commit.commit_oid;
                }
                if batch_commit.commit_url.is_some() {
                    current.commit_url = batch_commit.commit_url;
                }
                current
            }
        });
    }

    let commit = final_commit.unwrap_or(CommitInfo {
        commit_url: None,
        commit_message: Some(bundle.commit_message.clone()),
        commit_description: Some(bundle.commit_description.clone()),
        commit_oid: Some(branch_head.clone()),
        pr_url: None,
        pr_num: None,
    });
    if use_direct_main {
        println!("✅ Published MoE package");
        println!("   branch: main");
    } else {
        println!("✅ Opened MoE package contribution");
        println!("   branch: {branch_name}");
    }
    if let Some(commit_oid) = commit.commit_oid.as_deref() {
        println!("   commit: {commit_oid}");
    }
    if let Some(commit_url) = commit.commit_url.as_deref() {
        println!("   url: {commit_url}");
    }
    if let Some(pr_url) = commit.pr_url.as_deref() {
        println!("   pr: {pr_url}");
    }

    let package_revision = commit
        .commit_oid
        .clone()
        .unwrap_or_else(|| branch_head.clone());
    let catalog_path = moe_planner::catalog_entry_path_for_source_repo(
        resolved.source_repo.as_deref().ok_or_else(|| {
            anyhow::anyhow!("Resolved model is missing a source repo for catalog publication")
        })?,
    )?;
    let source_file = resolved
        .path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow::anyhow!("Resolved model path has no file name"))?;
    let catalog_source = crate::models::catalog::CatalogSource {
        repo: resolved
            .source_repo
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Resolved model is missing a source repo"))?,
        revision: resolved
            .source_revision
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Resolved model is missing a source revision"))?,
        file: source_file.to_string(),
    };
    let package_pointer = crate::models::catalog::CatalogPackagePointer {
        package_repo: publish_target.package_repo.clone(),
        package_revision: package_revision.clone(),
        publisher: publish_target.publisher.clone(),
        trust: publish_target.trust.to_string(),
    };
    let catalog_commit = contribute_catalog_entry(
        &api,
        catalog_repo,
        &catalog_path,
        &bundle.variant,
        catalog_source,
        package_pointer,
    )
    .await
    .map_err(|err| {
        share_error(
            "Catalog contribution failed",
            &format!("Update {}: {}", catalog_repo, err),
        )
    })?;
    println!("✅ Opened catalog contribution");
    println!("   repo: {catalog_repo}");
    println!("   entry: {catalog_path}");
    if let Some(pr_url) = catalog_commit.pr_url.as_deref() {
        println!("   pr: {pr_url}");
    } else if let Some(commit_url) = catalog_commit.commit_url.as_deref() {
        println!("   url: {commit_url}");
    }
    Ok(())
}

pub(super) async fn ensure_package_calibration_benchmark(
    model: &moe_planner::MoeModelContext,
    ranking: &moe_planner::ResolvedRanking,
) -> Result<()> {
    let analysis_path = moe::package_cache_analysis_path(&model.path);
    if moe_benchmark::package_benchmark_is_current(&analysis_path) {
        eprintln!(
            "🧪 Reusing package calibration benchmark from {}",
            analysis_path.display()
        );
        return Ok(());
    }

    eprintln!("🧪 Running local package calibration benchmark");
    eprintln!("   phase: benchmark");
    eprintln!("   output: {}", analysis_path.display());
    let current_exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = current_exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Current executable has no parent directory"))?;
    let benchmark = moe_benchmark::run_local_package_calibration_benchmark(model, ranking, bin_dir)
        .await
        .context("Run local package calibration benchmark")?;
    let benchmark_value = serde_json::to_value(&benchmark)?;
    moe_planner::write_package_benchmark_json(&analysis_path, &benchmark_value)?;
    println!(
        "  Benchmark: local package calibration written to {}",
        analysis_path.display()
    );
    Ok(())
}
