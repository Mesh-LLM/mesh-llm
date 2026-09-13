//! Source-complete v2 creation. Shards are physical containers, not stage owners.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use skippy_model::gguf_catalog::read_gguf_metadata_catalog;
use skippy_model::package_carrier::resolve_package_carrier;
use skippy_package_format::{
    Artifact, ArtifactCatalog, PACKAGE_SCHEMA_VERSION, PackageManifest, Sidecar, SidecarKind,
    SourceModel, Tensor, TensorCatalog,
};
use skippy_runtime::{ModelInfo, TensorInfo, write_gguf_metadata_from_parts};

use crate::hash::file_sha256;
use crate::package::{
    ArtifactHook, ExplicitSourceIdentity, PackageInput, resolve_package_input, run_artifact_hook,
};
use crate::plan::StagePlan;
use crate::progress::{PackageProgress, format_bytes};
use crate::source_inventory::{SourceInventory, inspect, normalized_model_metadata};
use crate::tensor_payload::{TensorLocation, compare_tensor_payload};
use crate::write::{ModelSource, create_parent_dir, write_json_file, write_stage_artifact};

mod layout;

use layout::{PlannedArtifact, PlannedArtifactKind, plan_artifacts_with_budget};

#[allow(clippy::too_many_arguments)]
pub(crate) fn write_package(
    model: String,
    out_dir: PathBuf,
    projectors: Vec<PathBuf>,
    artifact_hook: ArtifactHook,
    artifact_transform: ArtifactHook,
    explicit: ExplicitSourceIdentity,
    resume_existing_artifacts: bool,
    max_artifact_bytes: Option<u64>,
) -> Result<()> {
    ensure!(
        artifact_transform.command.is_none(),
        "v2 creation preserves source bytes; transform the independent source before packaging, not package artifacts"
    );
    let input = resolve_package_input(model, explicit)?;
    let inventory = SourceInventory::read(&input)?;
    let source = ModelSource::open(&input.model_path)?;
    ensure_native_inventory_matches(&inventory, &source)?;
    let budget = max_artifact_bytes.unwrap_or(layout::DEFAULT_MAX_ARTIFACT_BYTES);
    let planned = plan_artifacts_with_budget(&source.tensors, budget)?;
    let mut manifest = manifest_from_source(&input, &inventory)?;
    fs::create_dir_all(&out_dir)?;
    ensure!(
        !out_dir.join("model-package.json").exists(),
        "output already contains model-package.json; use a new directory for v2 creation"
    );
    let mut progress = PackageProgress::new(planned.len() + projectors.len() + 2);
    let source_tensors = source_tensors_by_name(&inventory)?;
    let common_names = planned
        .iter()
        .find(|artifact| artifact.kind == PlannedArtifactKind::Common)
        .map(|artifact| artifact.tensor_names.iter().cloned().collect())
        .unwrap_or_default();
    let mut catalog = Vec::with_capacity(source_tensors.len());
    let no_hook = ArtifactHook { command: None };
    // Payload artifacts may be uploaded and locally deleted by their hook as
    // soon as they are verified, so the metadata carrier cannot depend on the
    // full parts surviving until the end. Each artifact's header — everything
    // before its aligned data start — fully determines its descriptor table and
    // tensor offsets, so a header-only stub of each part carries the same
    // locators while occupying kilobytes instead of the full payload.
    let headers_dir = out_dir.join(".headers");
    fs::create_dir_all(&headers_dir)?;
    let mut header_stubs = Vec::with_capacity(planned.len());
    for (stage_index, artifact_plan) in planned.iter().enumerate() {
        progress.start_step(&artifact_plan.path)?;
        let (artifact, mut tensors) = emit_payload_artifact(
            &source,
            &source_tensors,
            &inventory,
            artifact_plan,
            &common_names,
            inventory.layer_count,
            stage_index,
            &out_dir,
            &no_hook,
            resume_existing_artifacts,
        )?;
        let path = out_dir.join(&artifact.path);
        // Capture the header stub before the upload hook can delete the part.
        let header = header_stub_path(&headers_dir, &artifact.id);
        write_header_stub(&path, &header, artifact.byte_size)?;
        header_stubs.push(header);
        run_artifact_hook(&artifact_hook, &path, &artifact.path)?;
        verify_hook_result(&artifact, &path, &artifact_hook)?;
        progress.finish_step(&format!(
            "{} {}",
            artifact.path,
            format_bytes(artifact.byte_size)
        ))?;
        manifest.artifact_catalog.entries.push(artifact);
        catalog.append(&mut tensors);
    }
    catalog.sort_by(|left, right| left.id.cmp(&right.id));
    ensure!(
        catalog
            .iter()
            .map(|tensor| tensor.id.as_str())
            .collect::<BTreeSet<_>>()
            == source_tensors
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
        "emitted payload tensor catalog differs from independent source inventory"
    );
    manifest.tensor_catalog = TensorCatalog { entries: catalog };
    progress.start_step("shared/metadata.gguf")?;
    let metadata_artifact = emit_metadata_artifact(
        &source,
        &inventory,
        &manifest,
        &out_dir,
        &header_stubs,
        resume_existing_artifacts,
    )?;
    progress.finish_step(&format!(
        "{} {}",
        metadata_artifact.path,
        format_bytes(metadata_artifact.byte_size)
    ))?;
    manifest
        .artifact_catalog
        .entries
        .insert(0, metadata_artifact);
    manifest.package_id = manifest.computed_package_id()?;
    let resolved = resolve_package_carrier(manifest.clone(), out_dir.join("shared/metadata.gguf"))?;
    ensure!(
        resolved.model_metadata == manifest.model_metadata,
        "metadata carrier model metadata differs from the independent source"
    );
    ensure!(
        resolved.tensor_catalog == manifest.tensor_catalog,
        "metadata carrier tensor inventory differs from the independently verified payloads"
    );
    // The carrier is fully verified against the manifest while it is still on
    // disk; only then does the optional artifact hook run.
    let carrier_path = out_dir.join("shared/metadata.gguf");
    run_artifact_hook(&artifact_hook, &carrier_path, "shared/metadata.gguf")?;
    verify_hook_result(
        manifest
            .artifact_catalog
            .entries
            .first()
            .expect("carrier entry"),
        &carrier_path,
        &artifact_hook,
    )?;
    // Header stubs are an internal working set; the published package root
    // contains only the manifest and its catalogued artifacts.
    if !header_stubs.is_empty() {
        let _ = fs::remove_dir_all(&headers_dir);
    }
    for (index, projector) in projectors.iter().enumerate() {
        let artifact = copy_projector(projector, index, &out_dir, resume_existing_artifacts)?;
        progress.start_step(&artifact.path)?;
        run_artifact_hook(
            &artifact_hook,
            &out_dir.join(&artifact.path),
            &artifact.path,
        )?;
        if artifact_hook.command.is_some() && out_dir.join(&artifact.path).exists() {
            ensure!(
                artifact_unchanged_on_disk(&artifact, &out_dir.join(&artifact.path))?,
                "projector changed after artifact hook"
            );
        }
        progress.finish_step(&format!(
            "{} {}",
            artifact.path,
            format_bytes(artifact.byte_size)
        ))?;
        manifest.sidecars.push(Sidecar {
            kind: SidecarKind::Mmproj,
            artifact_id: artifact.id.clone(),
            name: Some(artifact.id.clone()),
        });
        manifest.artifact_catalog.entries.push(artifact);
    }
    manifest.package_id = manifest.computed_package_id()?;
    manifest.validate()?;
    progress.start_step("model-package.json")?;
    // The completion marker is published only after all exact-coverage checks.
    let temporary = out_dir.join(".model-package-v2.json.tmp");
    ensure!(
        !temporary.exists(),
        "stale v2 manifest temporary file exists"
    );
    write_json_file(&temporary, &manifest)?;
    fs::rename(&temporary, out_dir.join("model-package.json"))?;
    progress.finish_step("model-package.json")?;
    progress.finish()?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn manifest_from_source(
    input: &PackageInput,
    inventory: &SourceInventory,
) -> Result<PackageManifest> {
    let identity = &input.source_identity;
    let primary = identity
        .primary_file
        .as_ref()
        .context("missing primary source identity")?;
    let primary_shard = inventory
        .shards
        .iter()
        .find(|s| &s.source_file.path == primary)
        .context("primary source absent from independent inventory")?;
    let model_metadata = normalized_model_metadata(&inventory.shards[0]);
    Ok(PackageManifest {
        schema_version: PACKAGE_SCHEMA_VERSION,
        package_id: String::new(),
        model_id: input.model_id.clone(),
        source_model: SourceModel {
            sha256: primary_shard.source_file.sha256.clone(),
            metadata_artifact_id: "metadata".to_string(),
            repo: identity.repo.clone(),
            revision: identity.revision.clone(),
            primary_file: identity.primary_file.clone(),
            canonical_ref: identity.canonical_ref.clone(),
            distribution_id: identity.distribution_id.clone(),
            files: inventory
                .shards
                .iter()
                .map(|s| s.source_file.clone())
                .collect(),
        },
        format: "gguf".to_string(),
        layer_count: inventory.layer_count,
        model_metadata,
        artifact_catalog: ArtifactCatalog {
            entries: Vec::new(),
        },
        tensor_catalog: TensorCatalog {
            entries: Vec::new(),
        },
        sidecars: Vec::new(),
        generation: None,
        native_abi_version: format!(
            "{}.{}.{}",
            skippy_ffi::ABI_VERSION_MAJOR,
            skippy_ffi::ABI_VERSION_MINOR,
            skippy_ffi::ABI_VERSION_PATCH
        ),
        generator_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_secs(),
    })
}

fn ensure_native_inventory_matches(
    inventory: &SourceInventory,
    source: &ModelSource,
) -> Result<()> {
    let expected = inventory
        .shards
        .iter()
        .flat_map(|shard| {
            shard
                .tensors
                .entries
                .iter()
                .map(|tensor| tensor.name.as_str())
        })
        .collect::<BTreeSet<_>>();
    let actual = source
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        actual.len() == source.tensors.len() && actual == expected,
        "native tensor inventory differs from independent source inventory"
    );
    Ok(())
}

fn source_tensors_by_name(inventory: &SourceInventory) -> Result<BTreeMap<String, TensorLocation>> {
    let mut tensors = BTreeMap::new();
    for shard in &inventory.shards {
        for tensor in &shard.tensors.entries {
            ensure!(
                tensors
                    .insert(
                        tensor.name.clone(),
                        TensorLocation {
                            path: shard.path.clone(),
                            tensor: tensor.clone(),
                        },
                    )
                    .is_none(),
                "duplicate source tensor {:?}",
                tensor.name
            );
        }
    }
    Ok(tensors)
}

fn emit_metadata_artifact(
    source: &ModelSource,
    inventory: &SourceInventory,
    _manifest: &PackageManifest,
    out_dir: &Path,
    header_stubs: &[PathBuf],
    resume: bool,
) -> Result<Artifact> {
    let relative = "shared/metadata.gguf";
    let path = out_dir.join(relative);
    create_parent_dir(&path)?;
    ensure_not_source_file(source, &path)?;
    if !path.exists() {
        // Payload parts may already be uploaded and deleted; the header-only
        // stubs carry the exact descriptor tables and locators of the full
        // parts, so the carrier is built from them instead. Stub order must
        // match the id-sorted payload artifacts the carrier locators index.
        let mut payload_paths: Vec<&Path> = header_stubs.iter().map(PathBuf::as_path).collect();
        payload_paths.sort();
        write_gguf_metadata_from_parts(&payload_paths, &path)
            .with_context(|| format!("write GGUF metadata carrier {}", path.display()))?;
    } else {
        ensure!(
            resume,
            "artifact {} already exists; use --resume-existing-artifacts to verify it",
            path.display()
        );
    }
    let info = ModelInfo::open(&path)
        .with_context(|| format!("open metadata carrier {}", path.display()))?;
    let tensors = info.tensors()?;
    let expected = inventory
        .shards
        .iter()
        .flat_map(|shard| shard.tensors.entries.iter())
        .map(|tensor| {
            let element_count = tensor
                .dimensions
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .context("source tensor element count overflow")?;
            Ok((tensor.name.clone(), (tensor.ggml_type, element_count)))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    ensure!(
        metadata_descriptors_match(&tensors, &expected),
        "metadata carrier descriptor table differs from independent source inventory"
    );
    let artifact = artifact_record("metadata", relative, &path)?;
    Ok(artifact)
}

fn metadata_descriptors_match(
    tensors: &[TensorInfo],
    expected: &BTreeMap<String, (u32, u64)>,
) -> bool {
    let actual = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    actual.len() == tensors.len()
        && actual.len() == expected.len()
        && actual.iter().all(|(name, tensor)| {
            expected.get(*name).is_some_and(|(ggml_type, elements)| {
                tensor.ggml_type == *ggml_type && tensor.element_count == *elements
            })
        })
}

#[allow(clippy::too_many_arguments)]
fn emit_payload_artifact(
    source: &ModelSource,
    source_tensors: &BTreeMap<String, TensorLocation>,
    inventory: &SourceInventory,
    planned: &PlannedArtifact,
    common_names: &BTreeSet<String>,
    layer_count: u32,
    stage_index: usize,
    out_dir: &Path,
    artifact_hook: &ArtifactHook,
    resume: bool,
) -> Result<(Artifact, Vec<Tensor>)> {
    let path = out_dir.join(&planned.path);
    ensure_not_source_file(source, &path)?;
    if !path.exists() {
        if planned.is_part() {
            crate::part_writer::write_part(inventory, &planned.tensor_names, &path)?;
        } else {
            let stage = stage_plan(planned, stage_index, layer_count);
            write_stage_artifact(source, &stage, &path)?;
        }
    } else {
        ensure!(
            resume,
            "artifact {} already exists; use --resume-existing-artifacts to verify it",
            path.display()
        );
    }
    let (_, emitted) = inspect(&path, &planned.id)?;
    let emitted_by_name = emitted
        .entries
        .into_iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect::<BTreeMap<_, _>>();
    let emitted_locations = emitted_by_name
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                TensorLocation {
                    path: path.clone(),
                    tensor: tensor.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let expected_physical = if planned.is_part() {
        // Part artifacts are written by the Rust part writer and hold exactly
        // their planned tensors; unlike native stage slices they carry no
        // duplicated common tensors.
        planned
            .tensor_names
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
    } else {
        planned
            .tensor_names
            .iter()
            .chain(common_names)
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
    };
    ensure!(
        emitted_by_name
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            == expected_physical,
        "written artifact {:?} differs from its native role plan",
        planned.id
    );
    let layer_ordinal = match planned.kind {
        PlannedArtifactKind::Layer { ordinal } => Some(ordinal),
        PlannedArtifactKind::Common
        | PlannedArtifactKind::Embeddings
        | PlannedArtifactKind::Output => None,
    };
    let mut bound = Vec::with_capacity(planned.tensor_names.len());
    for name in &planned.tensor_names {
        let mut tensor = emitted_by_name
            .get(name)
            .with_context(|| format!("written artifact {:?} omitted tensor {name:?}", planned.id))?
            .clone();
        let expected = source_tensors
            .get(name)
            .context("planned tensor is absent from independent source inventory")?;
        ensure!(
            tensor.name == expected.tensor.name
                && tensor.ggml_type == expected.tensor.ggml_type
                && tensor.dimensions == expected.tensor.dimensions,
            "written tensor {name:?} metadata differs from independent source inventory"
        );
        compare_tensor_payload(name, source_tensors, &emitted_locations)?;
        tensor.layer_ordinal = layer_ordinal;
        bound.push(tensor);
    }
    let artifact = artifact_record(&planned.id, &planned.path, &path)?;
    run_artifact_hook(artifact_hook, &path, &planned.path)?;
    verify_hook_result(&artifact, &path, artifact_hook)?;
    Ok((artifact, bound))
}

/// Path of the header-only stub kept for artifact `id` while its full payload
/// may be uploaded and deleted by its hook.
fn header_stub_path(headers_dir: &Path, artifact_id: &str) -> PathBuf {
    let safe_id = artifact_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == '.')
        .collect::<String>();
    headers_dir.join(format!("{safe_id}.gguf"))
}

/// Copy the header of the verified artifact at `path` — every byte before its
/// aligned tensor data start — to `header`. The stub re-opens as a legal
/// descriptor-only GGUF whose tensor offsets equal the full artifact's, which
/// is what the metadata carrier records.
fn write_header_stub(path: &Path, header: &Path, byte_size: u64) -> Result<()> {
    let catalog = read_gguf_metadata_catalog(path)?;
    ensure!(
        catalog.data_start <= byte_size && catalog.data_start <= catalog.artifact_bytes,
        "artifact {} header extends beyond its recorded size",
        path.display()
    );
    let length = usize::try_from(catalog.data_start)
        .with_context(|| format!("artifact {} header length overflows usize", path.display()))?;
    let input =
        File::open(path).with_context(|| format!("open verified artifact {}", path.display()))?;
    let mut output =
        File::create(header).with_context(|| format!("create header stub {}", header.display()))?;
    io::copy(
        &mut input.take(u64::try_from(length).context("header length to u64")?),
        &mut output,
    )
    .with_context(|| format!("copy header stub {}", header.display()))?;
    output.sync_all()?;
    Ok(())
}

fn stage_plan(planned: &PlannedArtifact, stage_index: usize, layer_count: u32) -> StagePlan {
    let (layer_start, layer_end, includes_embeddings, includes_output) = match planned.kind {
        PlannedArtifactKind::Common => (0, 0, false, false),
        PlannedArtifactKind::Embeddings => (0, 0, true, false),
        PlannedArtifactKind::Output => (layer_count, layer_count, false, true),
        PlannedArtifactKind::Layer { ordinal } => (ordinal, ordinal + 1, false, false),
    };
    StagePlan {
        stage_index,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        includes_per_layer_token_embd: planned
            .tensor_names
            .iter()
            .any(|name| name == crate::plan::PER_LAYER_TOKEN_EMBD),
        tensor_count: planned.tensor_names.len(),
        tensor_bytes: 0,
    }
}

fn artifact_record(id: &str, relative: &str, path: &Path) -> Result<Artifact> {
    Ok(Artifact {
        id: id.to_string(),
        path: relative.to_string(),
        byte_size: fs::metadata(path)?.len(),
        sha256: file_sha256(path)?,
    })
}

fn ensure_not_source_file(source: &ModelSource, path: &Path) -> Result<()> {
    if path.exists() {
        for source_path in &source.paths {
            ensure!(
                !same_file::is_same_file(source_path, path)?,
                "package artifact must not be the independent source file"
            );
        }
    }
    Ok(())
}

fn verify_hook_result(artifact: &Artifact, path: &Path, hook: &ArtifactHook) -> Result<()> {
    if hook.command.is_some() && path.exists() {
        ensure!(
            artifact_unchanged_on_disk(artifact, path)?,
            "artifact {:?} changed after artifact hook",
            artifact.id
        );
    }
    Ok(())
}

/// Whether the on-disk artifact still matches its record.
///
/// A hook may retain or remove the artifact. A FUSE bucket mount can keep
/// reporting a freshly unlinked file as present via a stale attr cache, so a
/// file that can no longer be opened counts as unchanged rather than corrupted.
/// Only a file that opens but differs (the hook mutated it) fails.
fn artifact_unchanged_on_disk(artifact: &Artifact, path: &Path) -> Result<bool> {
    let probe =
        || -> Result<bool> {
            Ok(fs::metadata(path)?.len() == artifact.byte_size
                && file_sha256(path)? == artifact.sha256)
        };
    match probe() {
        Ok(same) => Ok(same),
        Err(err) if is_not_found(&err) => Ok(true),
        Err(err) => Err(err),
    }
}

fn is_not_found(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
    })
}

fn copy_artifact(source: &Path, output: &Path, resume: bool) -> Result<()> {
    if resume && output.is_file() {
        ensure!(
            !same_file::is_same_file(source, output)?,
            "package artifact must not be the independent source file"
        );
        return Ok(());
    }
    fs::create_dir_all(
        output
            .parent()
            .context("artifact has no parent directory")?,
    )?;
    let mut input = File::open(source)?;
    let mut destination = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .with_context(|| {
            format!(
                "create artifact {}; use --resume-existing-artifacts to verify an existing copy",
                output.display()
            )
        })?;
    io::copy(&mut input, &mut destination)?;
    destination.sync_all()?;
    Ok(())
}

fn copy_projector(source: &Path, index: usize, out_dir: &Path, resume: bool) -> Result<Artifact> {
    let id = format!("projector-{index:05}");
    let (directory, tensors) = inspect(source, &id)?;
    let sha256 = file_sha256(source)?;
    let relative = format!("projectors/{id}.gguf");
    let path = out_dir.join(&relative);
    copy_artifact(source, &path, resume)?;
    let (written_directory, written_tensors) = inspect(&path, &id)?;
    ensure!(
        written_directory == directory
            && written_tensors == tensors
            && file_sha256(&path)? == sha256,
        "written projector differs from source"
    );
    Ok(Artifact {
        id,
        path: relative,
        byte_size: directory.artifact_bytes,
        sha256,
    })
}

#[cfg(test)]
mod tests;
