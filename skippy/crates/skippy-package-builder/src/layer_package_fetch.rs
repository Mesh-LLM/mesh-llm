//! Commit-bound WAN package acquisition through the existing native Hub client.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
#[path = "layer_package_fetch/worker_lifecycle.rs"]
mod worker_lifecycle;
use skippy_model_artifact::ModelRepository as _;
use skippy_model_hf::HfModelRepository;
use skippy_model_ref::package_reference::PackageReference;
use skippy_runtime::package::{
    PackageIntegrityOptions, PackageStageRequest, artifact_plan, verify_layer_package_integrity,
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FetchReport {
    pub repo: String,
    pub requested_revision: String,
    pub commit: String,
    pub snapshot_path: PathBuf,
    pub model_id: String,
    pub layer_count: u32,
    pub activation_width: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub artifacts_verified: usize,
    #[serde(skip)]
    manifest_sha256: String,
}
pub(crate) struct Input<'a> {
    pub reference: &'a str,
    pub cache_root: &'a Path,
    pub stage: Option<(u32, u32)>,
    pub timeout: Duration,
    pub expected_layers: Option<u32>,
    pub expected_width: Option<u32>,
}
pub(super) fn fetch_worker(input: Input<'_>) -> Result<FetchReport> {
    ensure!(
        !input.timeout.is_zero(),
        "acquisition timeout must be positive"
    );
    let reference = PackageReference::parse(input.reference)?;
    let root = prepare_cache(input.cache_root, reference.repo())?;
    let client = HfModelRepository::builder()
        .cache_dir(&root)
        .request_timeout(input.timeout)
        .build()?;
    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let report = runtime.block_on(async {
        tokio::time::timeout(input.timeout, acquire(&input, &reference, &root, &client))
            .await
            .map_err(|_| anyhow::anyhow!("package acquisition deadline expired"))?
    })?;
    // All SDK/native workers and verification stay in this child process.
    // Its parent owns the hard deadline, including this runtime shutdown.
    drop(runtime);
    let request = PackageStageRequest {
        model_id: report.model_id.clone(),
        topology_id: "wan-download".into(),
        package_ref: report.snapshot_path.to_string_lossy().into_owned(),
        stage_id: "download".into(),
        layer_start: report.layer_start,
        layer_end: report.layer_end,
        source_stage: input.stage.is_none_or(|(index, _)| index == 0),
        terminal_stage: input.stage.is_none_or(|(index, count)| index + 1 == count),
    };
    let integrity =
        verify_layer_package_integrity(&request, &PackageIntegrityOptions::verify_without_cache())?;
    ensure!(
        integrity.manifest_sha256 == report.manifest_sha256,
        "package manifest changed after artifact planning"
    );
    ensure!(
        started.elapsed() < input.timeout,
        "package verification completed after deadline"
    );
    Ok(FetchReport {
        artifacts_verified: integrity.artifacts,
        ..report
    })
}
pub(crate) fn fetch(input: Input<'_>) -> Result<FetchReport> {
    ensure!(
        !input.timeout.is_zero(),
        "acquisition timeout must be positive"
    );
    let until = Instant::now()
        .checked_add(input.timeout)
        .ok_or_else(|| anyhow::anyhow!("acquisition deadline overflow"))?;
    let reference = PackageReference::parse(input.reference)?;
    let mut command = std::process::Command::new(std::env::current_exe()?);
    command
        .args([
            "__fetch-layer-package-worker",
            "--reference",
            input.reference,
            "--cache-root",
        ])
        .arg(input.cache_root)
        .arg("--timeout-millis")
        .arg(input.timeout.as_millis().to_string());
    if let Some((index, count)) = input.stage {
        command.args([
            "--stage-index",
            &index.to_string(),
            "--stage-count",
            &count.to_string(),
        ]);
    }
    if let Some(value) = input.expected_layers {
        command.arg("--expected-layer-count").arg(value.to_string());
    }
    if let Some(value) = input.expected_width {
        command
            .arg("--expected-activation-width")
            .arg(value.to_string());
    }
    let report: FetchReport = serde_json::from_slice(&worker_lifecycle::run(
        command,
        until.saturating_duration_since(Instant::now()),
    )?)?;
    ensure!(
        report.repo == reference.repo()
            && report.requested_revision == reference.revision()
            && is_commit(&report.commit),
        "acquisition worker receipt identity differs"
    );
    if is_commit(reference.revision()) {
        ensure!(
            report.commit.eq_ignore_ascii_case(reference.revision()),
            "worker receipt substituted requested commit"
        );
    }
    ensure!(
        input
            .expected_layers
            .is_none_or(|n| n == report.layer_count)
            && input
                .expected_width
                .is_none_or(|n| n == report.activation_width),
        "worker receipt geometry differs"
    );
    let (index, count) = input.stage.unwrap_or((0, 1));
    let (start, end) = artifact_plan::even_stage_range(index, count, report.layer_count)?;
    ensure!(
        (report.layer_start, report.layer_end) == (start, end) && report.artifacts_verified > 0,
        "worker receipt selection differs"
    );
    ensure!(
        Instant::now() < until,
        "acquisition receipt admission exceeded overall deadline"
    );
    Ok(report)
}

async fn acquire(
    input: &Input<'_>,
    reference: &PackageReference,
    root: &Path,
    client: &HfModelRepository,
) -> Result<FetchReport> {
    let requested = reference.revision();
    let commit = if is_commit(requested) {
        requested.to_owned()
    } else {
        client
            .resolve_revision(reference.repo(), Some(requested))
            .await?
    };
    ensure!(
        is_commit(&commit),
        "requested package revision did not resolve to a valid commit"
    );
    let repo_root = repo_root(root, reference.repo())?;
    let snapshot = repo_root.join("snapshots").join(&commit);
    directory(&snapshot)?;
    let manifest = download(
        client,
        reference.repo(),
        &commit,
        "model-package.json",
        &snapshot,
        &repo_root,
    )
    .await?;
    let bytes = artifact_plan::read_manifest_declaration(&manifest)?;
    use sha2::Digest as _;
    let manifest_sha256 = crate::hash::hex_lower(&sha2::Sha256::digest(&bytes));
    let (model_id, layer_count, activation_width) = artifact_plan::declared_geometry(&bytes)?;
    ensure!(
        input.expected_layers.is_none_or(|n| n == layer_count),
        "package layer count differs from requested geometry"
    );
    ensure!(
        input.expected_width.is_none_or(|n| n == activation_width),
        "package activation width differs from requested geometry"
    );
    let (index, count) = input.stage.unwrap_or((0, 1));
    let (start, end) = artifact_plan::even_stage_range(index, count, layer_count)?;
    let parts = artifact_plan::declared_stage_parts(&bytes, index, count, start, end)?;
    for part in parts {
        let filename = part
            .path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("artifact name is not UTF-8"))?;
        download(
            client,
            reference.repo(),
            &commit,
            filename,
            &snapshot,
            &repo_root,
        )
        .await?;
    }
    Ok(FetchReport {
        repo: reference.repo().into(),
        requested_revision: requested.into(),
        commit,
        snapshot_path: snapshot,
        model_id,
        layer_count,
        activation_width,
        layer_start: start,
        layer_end: end,
        artifacts_verified: 0,
        manifest_sha256,
    })
}
async fn download(
    client: &HfModelRepository,
    repo: &str,
    commit: &str,
    filename: &str,
    snapshot: &Path,
    repo_root: &Path,
) -> Result<PathBuf> {
    let expected = snapshot.join(filename);
    let mut parent = snapshot.to_path_buf();
    for component in Path::new(filename)
        .parent()
        .unwrap_or(Path::new(""))
        .components()
    {
        parent.push(component);
        directory(&parent)?;
    }
    if expected.exists() {
        file_custody(&expected, snapshot, repo_root)?;
    }
    let returned = client.download_file(repo, commit, filename).await?;
    ensure!(
        returned == expected,
        "Hub returned a foreign package artifact path"
    );
    file_custody(&returned, snapshot, repo_root)?;
    Ok(returned)
}
fn file_custody(path: &Path, snapshot: &Path, repo_root: &Path) -> Result<()> {
    let target = path.canonicalize()?;
    ensure!(
        target.starts_with(snapshot) || target.starts_with(repo_root.join("blobs")),
        "package artifact escaped exact repository cache"
    );
    ensure!(
        fs::metadata(&target)?.is_file(),
        "package artifact is not regular"
    );
    Ok(())
}
fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn directory(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "cache directory redirects package custody"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => fs::create_dir(path)?,
        Err(error) => return Err(error.into()),
    }
    Ok(())
}
fn repo_root(root: &Path, repo: &str) -> Result<PathBuf> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("package repo lacks namespace"))?;
    ensure!(
        !owner.contains("--") && !name.contains("--"),
        "package repo aliases another cache name"
    );
    Ok(root.join(format!("models--{owner}--{name}")))
}
fn prepare_cache(path: &Path, repo: &str) -> Result<PathBuf> {
    fs::create_dir_all(path)?;
    let root = path.canonicalize()?;
    let repository = repo_root(&root, repo)?;
    directory(&repository)?;
    for name in ["blobs", "snapshots", "refs"] {
        directory(&repository.join(name))?;
    }
    Ok(root)
}
