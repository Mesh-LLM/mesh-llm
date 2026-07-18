use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use model_artifact::ModelArtifactFile;
use model_ref::split_gguf_shard_info;
use serde::Serialize;
use skippy_runtime::{ModelInfo, TensorInfo, write_gguf_from_parts};

use crate::hash::file_sha256;
use crate::plan::{
    StagePlan, build_plan_from_tensors, layer_count, parse_layer_range, stage_plan_from_tensors,
};

#[derive(Debug, Serialize)]
pub(crate) struct SliceManifest {
    pub(crate) schema_version: u32,
    pub(crate) source_model: String,
    pub(crate) source_sha256: String,
    pub(crate) stage_count: usize,
    pub(crate) layer_count: u32,
    pub(crate) stages: Vec<SliceManifestStage>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SliceManifestStage {
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
    path: String,
    tensor_count: usize,
    tensor_bytes: u64,
    artifact_bytes: u64,
    sha256: String,
}

pub(crate) struct ModelSource {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) infos: Vec<ModelInfo>,
    pub(crate) tensors: Vec<TensorInfo>,
}

pub(crate) fn write_one(
    model: PathBuf,
    layers: String,
    out: PathBuf,
    stage_index: Option<u32>,
    include_embeddings: bool,
    include_output: bool,
    manifest: Option<PathBuf>,
) -> Result<()> {
    let source = ModelSource::open(&model)?;
    let tensors = &source.tensors;
    let layer_count = layer_count(tensors)?;
    let (layer_start, layer_end) = parse_layer_range(&layers)?;
    if layer_end > layer_count {
        bail!("layer range end exceeds model layer count {layer_count}");
    }

    let stage_index = stage_index.unwrap_or(0);
    let includes_embeddings = include_embeddings || layer_start == 0;
    let includes_output = include_output || layer_end == layer_count;
    let stage = stage_plan_from_tensors(
        stage_index as usize,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        tensors,
    );

    write_stage_artifact(&source, &stage, &out)?;

    if let Some(path) = manifest {
        let manifest = build_manifest(&model, layer_count, vec![(stage, out)])?;
        write_json_file(&path, &manifest)?;
    }
    Ok(())
}

pub(crate) fn write_stages(model: PathBuf, stages: usize, out_dir: PathBuf) -> Result<()> {
    if stages == 0 {
        bail!("--stages must be greater than zero");
    }
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output directory {}", out_dir.display()))?;

    let source = ModelSource::open(&model)?;
    let tensors = &source.tensors;
    let plan = build_plan_from_tensors(stages, tensors)?;
    let mut written = Vec::new();
    for stage in plan.stages {
        let path = out_dir.join(format!("stage-{:03}.gguf", stage.stage_index));
        write_stage_artifact(&source, &stage, &path)?;
        written.push((stage, path));
    }

    let manifest = build_manifest(&model, plan.layer_count, written)?;
    let manifest_path = out_dir.join("slice-manifest.json");
    write_json_file(&manifest_path, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

pub(crate) fn write_stage_artifact(
    source: &ModelSource,
    stage: &StagePlan,
    out: &Path,
) -> Result<()> {
    create_parent_dir(out)?;

    if source.infos.len() == 1 {
        write_single_source_stage_artifact(&source.infos[0], stage, out)
    } else {
        write_sharded_stage_artifact(source, stage, out)
    }
}

fn build_manifest(
    model: &Path,
    layer_count: u32,
    written: Vec<(StagePlan, PathBuf)>,
) -> Result<SliceManifest> {
    let mut stages = Vec::new();
    for (stage, path) in written {
        let metadata = fs::metadata(&path)
            .with_context(|| format!("read artifact metadata {}", path.display()))?;
        stages.push(SliceManifestStage {
            stage_index: stage.stage_index,
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            includes_embeddings: stage.includes_embeddings,
            includes_output: stage.includes_output,
            path: path.display().to_string(),
            tensor_count: stage.tensor_count,
            tensor_bytes: stage.tensor_bytes,
            artifact_bytes: metadata.len(),
            sha256: file_sha256(&path)?,
        });
    }

    Ok(SliceManifest {
        schema_version: 1,
        source_model: model.display().to_string(),
        source_sha256: file_sha256(model)?,
        stage_count: stages.len(),
        layer_count,
        stages,
    })
}

pub(crate) fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    create_parent_dir(path)?;
    let json = serde_json::to_vec_pretty(value)?;
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    file.write_all(&json)
        .with_context(|| format!("write {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub(crate) fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    Ok(())
}

impl ModelSource {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let paths = resolve_gguf_shard_paths(path)?;
        let mut infos = Vec::with_capacity(paths.len());
        let mut tensors = Vec::new();
        for path in &paths {
            let info = ModelInfo::open(path)
                .with_context(|| format!("open GGUF metadata {}", path.display()))?;
            tensors.extend(
                info.tensors()
                    .with_context(|| format!("read GGUF tensors {}", path.display()))?,
            );
            infos.push(info);
        }
        Ok(Self {
            paths,
            infos,
            tensors,
        })
    }
}

pub(crate) fn resolve_gguf_shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some(shard) = split_gguf_shard_info(file_name) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let total = shard
        .total
        .parse::<usize>()
        .with_context(|| format!("parse GGUF shard total from {file_name}"))?;
    if total <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }

    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let mut paths = Vec::with_capacity(total);
    for part in 1..=total {
        let shard_name = format!("{}-{part:05}-of-{}.gguf", shard.prefix, shard.total);
        let shard_path = parent.join(shard_name);
        if !shard_path.exists() {
            bail!(
                "split GGUF shard {} is missing sibling {}",
                path.display(),
                shard_path.display()
            );
        }
        paths.push(shard_path);
    }
    Ok(paths)
}

pub(crate) fn local_artifact_files(
    model_path: &Path,
    primary_file: &str,
) -> Result<Vec<ModelArtifactFile>> {
    let shard_paths = resolve_gguf_shard_paths(model_path)?;
    if shard_paths.len() <= 1 {
        return Ok(vec![ModelArtifactFile::new(primary_file.to_string())]);
    }

    let primary_path = Path::new(primary_file);
    let primary_parent = primary_path.parent();
    let files = shard_paths
        .into_iter()
        .map(|path| {
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("split GGUF shard path has no file name")?;
            let relative = primary_parent
                .map(|parent| parent.join(file_name))
                .unwrap_or_else(|| PathBuf::from(file_name));
            Ok(ModelArtifactFile::new(
                relative.to_string_lossy().replace('\\', "/"),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(files)
}

fn write_single_source_stage_artifact(
    info: &ModelInfo,
    stage: &StagePlan,
    out: &Path,
) -> Result<()> {
    let mut plan = info.create_slice_plan()?;
    plan.add_layer_range(
        stage.stage_index as u32,
        stage.layer_start,
        stage.layer_end,
        stage.includes_embeddings,
        stage.includes_output,
    )?;
    info.write_slice_gguf(&plan, stage.stage_index as u32, out)
        .with_context(|| format!("write GGUF slice {}", out.display()))
}

fn write_sharded_stage_artifact(source: &ModelSource, stage: &StagePlan, out: &Path) -> Result<()> {
    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    let stem = out
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("stage");
    let pid = std::process::id();
    let scratch = parent.join(format!(".{stem}.shard-parts-{pid}"));
    if scratch.exists() {
        fs::remove_dir_all(&scratch)
            .with_context(|| format!("remove stale shard scratch {}", scratch.display()))?;
    }
    fs::create_dir_all(&scratch)
        .with_context(|| format!("create shard scratch {}", scratch.display()))?;

    let result = (|| {
        let mut parts = Vec::with_capacity(source.infos.len());
        for (index, info) in source.infos.iter().enumerate() {
            let part_path = scratch.join(format!("part-{index:05}.gguf"));
            write_single_source_stage_artifact(info, stage, &part_path).with_context(|| {
                format!(
                    "write shard-local GGUF slice from {}",
                    source.paths[index].display()
                )
            })?;
            parts.push(part_path);
        }
        write_gguf_from_parts(&parts, out)
            .with_context(|| format!("merge split-GGUF shard slices into {}", out.display()))
    })();

    let cleanup = fs::remove_dir_all(&scratch)
        .with_context(|| format!("remove shard scratch {}", scratch.display()));
    result.and(cleanup)
}
