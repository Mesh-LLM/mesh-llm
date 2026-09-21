use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use model_artifact::ModelArtifactFile;
use model_ref::split_gguf_shard_info;
use serde::Serialize;
use skippy_runtime::{ModelInfo, TensorInfo};

pub(crate) struct ModelSource {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) tensors: Vec<TensorInfo>,
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
        let mut tensors = Vec::new();
        for path in &paths {
            let info = ModelInfo::open(path)
                .with_context(|| format!("open GGUF metadata {}", path.display()))?;
            tensors.extend(
                info.tensors()
                    .with_context(|| format!("read GGUF tensors {}", path.display()))?,
            );
        }
        Ok(Self { paths, tensors })
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

    let primary_parent = Path::new(primary_file).parent();
    shard_paths
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
        .collect()
}
