//! SafeTensors checkpoint controls shared by single-stage model loading.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use skippy_runtime::CheckpointQuantization;

use crate::SingleStageOptions;

pub(super) struct PreparedCheckpoint {
    pub(super) quantization: CheckpointQuantization,
    pub(super) imatrix: Option<String>,
    pub(super) imatrix_sha256: Option<String>,
}

pub(super) fn prepare(options: &SingleStageOptions) -> Result<PreparedCheckpoint> {
    let quantization = options
        .checkpoint_quantization
        .as_deref()
        .unwrap_or("preserve")
        .parse::<CheckpointQuantization>()
        .map_err(anyhow::Error::msg)?;
    let (imatrix, imatrix_sha256) = match options.checkpoint_imatrix.as_deref() {
        Some(configured_path) => {
            let configured_path = PathBuf::from(configured_path);
            let resolved = resolve_imatrix_path(&options.model_path, configured_path);
            let canonical = resolved.canonicalize().with_context(|| {
                format!(
                    "resolve checkpoint importance matrix {}",
                    resolved.display()
                )
            })?;
            let sha256 = sha256_file(&canonical)?;
            (Some(canonical.to_string_lossy().into_owned()), Some(sha256))
        }
        None => (None, None),
    };
    Ok(PreparedCheckpoint {
        quantization,
        imatrix,
        imatrix_sha256,
    })
}

fn resolve_imatrix_path(model_path: &Path, configured_path: PathBuf) -> PathBuf {
    if configured_path.is_absolute() {
        return configured_path;
    }
    if model_path.is_dir() {
        model_path.join(configured_path)
    } else {
        model_path
            .parent()
            .unwrap_or(model_path)
            .join(configured_path)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(
        File::open(path)
            .with_context(|| format!("open checkpoint importance matrix {}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash checkpoint importance matrix {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}
