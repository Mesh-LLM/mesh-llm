//! Identity-bound reusable cache for derived MLX stage artifacts.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use model_hf::safetensors_stage::{SafetensorsStageMaterializer, SafetensorsStageRequest};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{
    DERIVED_STAGE_SCHEMA_VERSION, MlxDerivationControl, MlxDerivedStageConfig,
    MlxDerivedStageReport, REPORT_FILE, artifact_file_bytes, derive_quantized_stage, open_locked,
    prepare_derivation_recipe,
};
use crate::stage::MlxWeightQuantization;

const DERIVED_CACHE_ROOT_ENV: &str = "MESH_MLX_DERIVED_CACHE_DIR";

/// Returns the managed MLX derived-stage cache root for this process.
pub fn mlx_derived_stage_cache_root() -> PathBuf {
    std::env::var_os(DERIVED_CACHE_ROOT_ENV).map_or_else(
        || model_hf::store::mesh_llm_cache_dir().join("mlx-derived-stages"),
        PathBuf::from,
    )
}

/// Configuration for an identity-bound, reusable derived-stage cache entry.
#[derive(Clone, Debug)]
pub struct MlxDerivedStageCacheConfig {
    pub source: SafetensorsStageRequest,
    pub cache_root: PathBuf,
    pub quantization: MlxWeightQuantization,
    pub control: MlxDerivationControl,
    /// Soft output bundle target. A single packed tensor may exceed this size.
    pub shard_size_bytes: usize,
}

/// Result of a managed derived-stage lookup or build.
#[derive(Clone, Debug, Serialize)]
pub struct MlxDerivedStageCacheResult {
    pub cache_hit: bool,
    /// Tensor payload range requests made by this invocation.
    pub source_range_request_count: usize,
    pub output_dir: PathBuf,
    pub report: MlxDerivedStageReport,
}

/// Loads a verified derived stage from cache or builds it from exact tensor ranges.
pub fn derive_quantized_stage_cached(
    materializer: &SafetensorsStageMaterializer,
    config: &MlxDerivedStageCacheConfig,
) -> Result<MlxDerivedStageCacheResult> {
    resolve_quantized_stage_cache(materializer, config, true)
}

/// Loads a verified stage that was already produced by the prepare lifecycle.
///
/// This performs metadata planning and full cache validation, but never fetches
/// tensor payloads or derives a replacement on a miss.
pub fn load_prepared_quantized_stage(
    materializer: &SafetensorsStageMaterializer,
    config: &MlxDerivedStageCacheConfig,
) -> Result<MlxDerivedStageCacheResult> {
    resolve_quantized_stage_cache(materializer, config, false)
}

fn resolve_quantized_stage_cache(
    materializer: &SafetensorsStageMaterializer,
    config: &MlxDerivedStageCacheConfig,
    build_on_miss: bool,
) -> Result<MlxDerivedStageCacheResult> {
    config.control.ensure_active()?;
    ensure!(
        config.shard_size_bytes > 0,
        "derived shard size must be non-zero"
    );
    let recipe = prepare_derivation_recipe(
        materializer,
        config.source.clone(),
        config.quantization,
        config.shard_size_bytes,
        &config.control,
    )?;
    fs::create_dir_all(&config.cache_root).with_context(|| {
        format!(
            "create MLX derived stage cache {}",
            config.cache_root.display()
        )
    })?;
    let output_dir = config.cache_root.join(&recipe);
    let lock_path = config.cache_root.join(format!(".{recipe}.lock"));
    // Keep this pathname stable across invocations. Removing an advisory-lock
    // file after unlock can split waiters between the unlinked and new inodes.
    let _lock = open_cache_lock(&lock_path, &config.control)?;
    config.control.ensure_active()?;

    if let Some(report) = load_cached(&output_dir, &recipe, &config.control)? {
        config.control.ensure_active()?;
        return Ok(MlxDerivedStageCacheResult {
            cache_hit: true,
            source_range_request_count: 0,
            output_dir,
            report,
        });
    }
    if !build_on_miss {
        bail!(
            "prepared MLX derived stage cache entry {recipe} is missing or invalid; run StagePrepare before StageLoad"
        );
    }
    config.control.ensure_active()?;
    remove_invalid_cache_entry(&output_dir)?;
    let report = derive_quantized_stage(
        materializer,
        &MlxDerivedStageConfig {
            source: config.source.clone(),
            output_dir: output_dir.clone(),
            quantization: config.quantization,
            control: config.control.clone(),
            shard_size_bytes: config.shard_size_bytes,
        },
    )?;
    Ok(MlxDerivedStageCacheResult {
        cache_hit: false,
        source_range_request_count: report.source_range_request_count,
        output_dir,
        report,
    })
}

fn open_cache_lock(path: &Path, control: &MlxDerivationControl) -> Result<File> {
    loop {
        control.ensure_active()?;
        if let Some(lock) = open_locked(path, true)? {
            return Ok(lock);
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn load_cached(
    output_dir: &Path,
    recipe: &str,
    control: &MlxDerivationControl,
) -> Result<Option<MlxDerivedStageReport>> {
    if !output_dir.is_dir() {
        return Ok(None);
    }
    if !contains_only_regular_files(output_dir)? {
        return Ok(None);
    }
    let bytes = match fs::read(output_dir.join(REPORT_FILE)) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(None),
    };
    let mut report = match serde_json::from_slice::<MlxDerivedStageReport>(&bytes) {
        Ok(report) => report,
        Err(_) => return Ok(None),
    };
    if report.schema_version != DERIVED_STAGE_SCHEMA_VERSION
        || report.derivation_recipe_sha256 != recipe
        || report.artifact_file_bytes != artifact_file_bytes(output_dir)?
        || !validate_cache_files(output_dir, &report, control)?
    {
        return Ok(None);
    }
    // `output_dir` is diagnostic, not artifact identity. Refresh it so a
    // relocated cache remains reusable without exposing its stale build path.
    report.output_dir = output_dir.to_path_buf();
    Ok(Some(report))
}

fn contains_only_regular_files(output_dir: &Path) -> Result<bool> {
    for entry in fs::read_dir(output_dir)? {
        if !entry?.file_type()?.is_file() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn validate_cache_files(
    output_dir: &Path,
    report: &MlxDerivedStageReport,
    control: &MlxDerivationControl,
) -> Result<bool> {
    let mut expected_shards = BTreeMap::new();
    for shard in &report.shards {
        let relative = Path::new(&shard.file);
        if relative.components().count() != 1
            || !matches!(relative.components().next(), Some(Component::Normal(_)))
        {
            return Ok(false);
        }
        if expected_shards.insert(shard.file.as_str(), shard).is_some() {
            return Ok(false);
        }
    }
    if expected_shards.is_empty() {
        return Ok(false);
    }

    let mut files = fs::read_dir(output_dir)?
        .filter_map(|entry| match entry {
            Ok(entry) if entry.file_name() != REPORT_FILE => Some(Ok(entry)),
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort_by_key(|entry| entry.file_name());

    let mut aggregate = Sha256::new();
    aggregate.update(b"mesh-mlx-derived-output-v1");
    for entry in files {
        control.ensure_active()?;
        if !entry.file_type()?.is_file() {
            return Ok(false);
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let file_bytes = entry.metadata()?.len();
        aggregate.update(u64::try_from(name.len())?.to_le_bytes());
        aggregate.update(name.as_bytes());
        aggregate.update(file_bytes.to_le_bytes());

        let expected_shard = expected_shards.remove(name.as_ref());
        if expected_shard.is_some_and(|shard| shard.file_bytes != file_bytes) {
            return Ok(false);
        }
        let mut shard_hasher = expected_shard.map(|_| Sha256::new());
        let mut reader = BufReader::new(File::open(entry.path())?);
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            control.ensure_active()?;
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            aggregate.update(&buffer[..read]);
            if let Some(hasher) = &mut shard_hasher {
                hasher.update(&buffer[..read]);
            }
        }
        if let (Some(expected), Some(hasher)) = (expected_shard, shard_hasher)
            && format!("{:x}", hasher.finalize()) != expected.sha256
        {
            return Ok(false);
        }
    }
    control.ensure_active()?;
    Ok(expected_shards.is_empty()
        && format!("{:x}", aggregate.finalize()) == report.output_content_sha256)
}

fn remove_invalid_cache_entry(path: &Path) -> Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path)
            .with_context(|| format!("remove invalid derived stage cache {}", path.display()))?;
    } else if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("remove invalid derived stage cache {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use serde_json::json;

    use super::*;
    use crate::derived::{MlxDerivedStageShard, output_content_sha256, sha256_file, write_json};

    #[test]
    fn cancelled_cache_waiter_stops_without_acquiring_the_recipe_lock() {
        let directory = tempfile::tempdir().unwrap();
        let lock_path = directory.path().join("recipe.lock");
        let held = open_locked(&lock_path, false).unwrap().unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let control = MlxDerivationControl::new(None, Some(Arc::clone(&cancelled)));
        let waiter_path = lock_path.clone();
        let waiter = std::thread::spawn(move || open_cache_lock(&waiter_path, &control));

        std::thread::sleep(Duration::from_millis(75));
        cancelled.store(true, Ordering::Release);
        let error = waiter.join().unwrap().unwrap_err();

        assert!(error.to_string().contains("derivation cancelled"));
        drop(held);
    }

    fn write_test_cache(output_dir: &Path, recipe: &str) -> MlxDerivedStageReport {
        fs::create_dir_all(output_dir).unwrap();
        fs::write(output_dir.join("config.json"), b"{}\n").unwrap();
        fs::write(output_dir.join("model.safetensors"), b"packed").unwrap();
        let shard_path = output_dir.join("model.safetensors");
        let mut report = MlxDerivedStageReport {
            schema_version: DERIVED_STAGE_SCHEMA_VERSION,
            derivation_recipe_sha256: recipe.to_string(),
            output_content_sha256: output_content_sha256(output_dir).unwrap(),
            checkpoint_sha256: "checkpoint".to_string(),
            plan_sha256: "plan".to_string(),
            repo: "owner/model".to_string(),
            revision: "0".repeat(40),
            layer_start: 0,
            layer_end: 1,
            quantization: json!({"mode":"affine","bits":4,"group_size":64}),
            quantization_label: "affine-4bit-g64".to_string(),
            safemlx_revision: "revision".to_string(),
            output_dir: output_dir.to_path_buf(),
            source_tensor_count: 1,
            source_tensor_bytes: 16,
            source_range_request_count: 1,
            source_temporary_file_peak_bytes: 16,
            quantized_tensor_count: 1,
            copied_tensor_count: 0,
            output_tensor_bytes: 6,
            artifact_file_bytes: artifact_file_bytes(output_dir).unwrap(),
            working_disk_peak_bytes: 22,
            mlx_active_memory_bytes: 0,
            mlx_cache_memory_bytes: 0,
            mlx_peak_memory_bytes: 0,
            shards: vec![MlxDerivedStageShard {
                file: "model.safetensors".to_string(),
                file_bytes: 6,
                sha256: sha256_file(&shard_path).unwrap(),
            }],
        };
        write_json(output_dir.join(REPORT_FILE), &report).unwrap();
        report.output_content_sha256 = output_content_sha256(output_dir).unwrap();
        report
    }

    #[test]
    fn validates_cache_content_and_rejects_tampering() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("recipe");
        let expected = write_test_cache(&output, "recipe");

        let cached = load_cached(&output, "recipe", &MlxDerivationControl::default())
            .unwrap()
            .unwrap();
        assert_eq!(cached.output_content_sha256, expected.output_content_sha256);

        fs::write(output.join("model.safetensors"), b"broken").unwrap();
        assert!(
            load_cached(&output, "recipe", &MlxDerivationControl::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn rejects_traversal_in_cached_shard_name() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("recipe");
        let mut report = write_test_cache(&output, "recipe");
        report.shards[0].file = "../model.safetensors".to_string();
        assert!(!validate_cache_files(&output, &report, &MlxDerivationControl::default()).unwrap());
    }

    #[test]
    fn accepts_relocated_cache_and_refreshes_diagnostic_path() {
        let directory = tempfile::tempdir().unwrap();
        let original_root = directory.path().join("original");
        let original = original_root.join("recipe");
        write_test_cache(&original, "recipe");
        let relocated_root = directory.path().join("relocated");
        fs::rename(&original_root, &relocated_root).unwrap();
        let relocated = relocated_root.join("recipe");

        let cached = load_cached(&relocated, "recipe", &MlxDerivationControl::default())
            .unwrap()
            .unwrap();

        assert_eq!(cached.output_dir, relocated);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_non_regular_cache_entries() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("recipe");
        write_test_cache(&output, "recipe");
        symlink("config.json", output.join("unexpected-index.json")).unwrap();

        assert!(
            load_cached(&output, "recipe", &MlxDerivationControl::default())
                .unwrap()
                .is_none()
        );
    }
}
