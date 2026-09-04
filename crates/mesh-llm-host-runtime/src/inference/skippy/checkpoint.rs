//! SafeTensors checkpoint controls shared by single-stage model loading.

use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};
use sha2::{Digest, Sha256};
use skippy_runtime::CheckpointQuantization;

use super::SkippyModelLoadOptions;

pub(super) struct PreparedCheckpoint {
    pub(super) quantization: CheckpointQuantization,
    pub(super) imatrix: Option<String>,
    pub(super) imatrix_sha256: Option<String>,
}

pub(super) fn prepare(options: &SkippyModelLoadOptions) -> Result<PreparedCheckpoint> {
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

pub(super) fn emit_load_notice(
    model_path: &Path,
    quantization: CheckpointQuantization,
    has_imatrix: bool,
) {
    if !skippy_runtime::is_safetensors_checkpoint(model_path) {
        return;
    }
    let imatrix_status = if has_imatrix { "configured" } else { "none" };
    let message = format!(
        "SafeTensors native loader: quantization={} imatrix={imatrix_status}. Set with `mesh-llm serve --quant <RECIPE>`; see `mesh-llm serve --help-advanced` for valid recipes.",
        quantization.canonical_name()
    );
    let event = if quantization == CheckpointQuantization::Preserve {
        OutputEvent::Info {
            message,
            context: None,
        }
    } else {
        OutputEvent::Warning {
            message,
            context: None,
        }
    };
    let _ = emit_event(event);
}
