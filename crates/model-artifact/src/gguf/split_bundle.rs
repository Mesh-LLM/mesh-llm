//! Parameter counting across a split-GGUF shard set.
//!
//! A large model is often published as `name-00001-of-000NN.gguf` … and every
//! shard carries its own tensor-info table holding only *its* slice of the
//! weights. Summing one shard therefore reports a fraction of the model, so a
//! model size must be summed across the whole bundle.

use std::path::{Path, PathBuf};

use model_ref::split_gguf_shard_info;

/// Total stored parameter count for the model at `path`, summed across every
/// shard when `path` is one part of a split GGUF.
///
/// Single-file models delegate straight to [`super::scan_gguf_total_parameters`].
///
/// Returns `None` when the shard set is incomplete or any shard fails to parse.
/// An incomplete bundle cannot be loaded for inference anyway, and reporting a
/// partial sum would understate the model — the same defect this function
/// exists to fix. Callers treat `None` as *unknown size*, which is safe; a
/// confident undercount is not.
pub fn scan_gguf_bundle_total_parameters(path: &Path) -> Option<u64> {
    let Some(shards) = split_gguf_shard_paths(path) else {
        // Not a split shard at all — one file holds the whole model.
        return super::scan_gguf_total_parameters(path);
    };
    // A declared-but-incomplete set yields an empty vec, which must read as
    // unknown rather than falling back to a single-shard undercount.
    if shards.is_empty() {
        return None;
    }
    let mut total: u64 = 0;
    for shard in shards {
        total = total.checked_add(super::scan_gguf_total_parameters(&shard)?)?;
    }
    Some(total)
}

/// Every shard path of the split GGUF that `path` belongs to, in part order.
///
/// * `None` — `path` is not a split shard; the caller should scan it directly.
/// * `Some(vec![])` — `path` declares a shard set but at least one part is
///   missing on disk. Distinct from `None` so the caller reports *unknown*
///   instead of silently summing the shards it happens to have.
fn split_gguf_shard_paths(path: &Path) -> Option<Vec<PathBuf>> {
    let file_name = path.file_name()?.to_str()?;
    let shard = split_gguf_shard_info(file_name)?;
    let total: u32 = shard.total.parse().ok()?;
    if total == 0 {
        return None;
    }
    let directory = path.parent()?;
    let width = shard.part.len();
    let prefix = shard.prefix;
    let total_label = shard.total;

    let mut paths = Vec::with_capacity(total as usize);
    for part in 1..=total {
        let candidate = directory.join(format!(
            "{prefix}-{part:0width$}-of-{total_label}.gguf",
            part = part
        ));
        if !candidate.exists() {
            return Some(Vec::new());
        }
        paths.push(candidate);
    }
    Some(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid GGUF: header + `n_kv` = 0 + one tensor of `elements`
    /// weights. Enough for `scan_gguf_total_parameters` to sum, which is all
    /// these tests need — the real scanner is exercised against real artifacts.
    fn write_gguf_with_parameters(path: &Path, elements: u64) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        bytes.extend_from_slice(&0u64.to_le_bytes()); // n_kv
        let name = b"weights";
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        bytes.extend_from_slice(&elements.to_le_bytes()); // dim 0
        bytes.extend_from_slice(&0u32.to_le_bytes()); // ggml type
        bytes.extend_from_slice(&0u64.to_le_bytes()); // offset
        std::fs::write(path, bytes).expect("write test gguf");
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mesh-llm-split-bundle-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn sums_every_shard_of_a_split_bundle() {
        let dir = temp_dir("sums");
        for (part, elements) in [(1u32, 10u64), (2, 20), (3, 30)] {
            write_gguf_with_parameters(
                &dir.join(format!("Model-Q4_K_M-{part:05}-of-00003.gguf")),
                elements,
            );
        }

        // Any shard resolves the whole bundle, not just its own slice.
        for part in 1..=3 {
            let shard = dir.join(format!("Model-Q4_K_M-{part:05}-of-00003.gguf"));
            assert_eq!(
                scan_gguf_bundle_total_parameters(&shard),
                Some(60),
                "shard {part} must report the bundle total, not its own tensors"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn incomplete_bundle_reads_as_unknown_rather_than_undercounting() {
        let dir = temp_dir("incomplete");
        write_gguf_with_parameters(&dir.join("Model-Q4_K_M-00001-of-00003.gguf"), 10);
        write_gguf_with_parameters(&dir.join("Model-Q4_K_M-00002-of-00003.gguf"), 20);
        // Shard 3 was never downloaded.

        assert_eq!(
            scan_gguf_bundle_total_parameters(&dir.join("Model-Q4_K_M-00001-of-00003.gguf")),
            None,
            "a missing shard must read as unknown size, never as a partial sum"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_file_model_delegates_to_the_plain_scan() {
        let dir = temp_dir("single");
        let path = dir.join("Model-Q4_K_M.gguf");
        write_gguf_with_parameters(&path, 42);

        assert_eq!(scan_gguf_bundle_total_parameters(&path), Some(42));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
