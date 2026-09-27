use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod affinity;
pub mod cache_aware;
pub mod cache_inventory;

/// Calculate total model size, summing all split files if present.
/// Split files follow the pattern: name-00001-of-00004.gguf.
pub fn total_model_bytes(model: &Path) -> u64 {
    let name = model.to_string_lossy();
    if let Some(pos) = name.find("-00001-of-") {
        let of_pos = pos + 10;
        if let Some(ext_pos) = name[of_pos..].find(".gguf")
            && let Ok(n_split) = name[of_pos..of_pos + ext_pos].parse::<u32>()
        {
            let prefix = &name[..pos + 1];
            let suffix = &name[of_pos + ext_pos..];
            let mut total: u64 = 0;
            for i in 1..=n_split {
                let split_name = format!("{}{:05}-of-{:05}{}", prefix, i, n_split, suffix);
                total += std::fs::metadata(&split_name).map(|m| m.len()).unwrap_or(0);
            }
            return total;
        }
    }
    match std::fs::metadata(model) {
        Ok(metadata) if metadata.is_dir() => {
            // A SafeTensors checkpoint directory (model.safetensors +
            // tokenizer/config siblings) reports the directory inode's size
            // (~4 KiB) as its length. Sum the checkpoint files instead so
            // memory planning charges the real weight bytes; a directory of
            // plain GGUFs is not a loadable single model, but summing its
            // files is still the least-wrong size estimate for routing.
            dir_file_bytes(model)
        }
        Ok(metadata) => metadata.len(),
        Err(_) => 0,
    }
}

/// Sum the regular-file sizes directly inside `dir` (non-recursive; model
/// checkpoint directories are flat).
fn dir_file_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                // Follow symlinks: HuggingFace-style snapshot entries are
                // symlinks into a blobs/ store, and DirEntry::metadata is
                // lstat — it would report the link itself (is_file() false,
                // len 0), zeroing the total. fs::metadata follows the link
                // and reports the target file.
                .filter_map(|entry| std::fs::metadata(entry.path()).ok())
                .filter(|metadata| metadata.is_file())
                .map(|metadata| metadata.len())
                .sum()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_model_reports_summed_file_bytes() {
        // A SafeTensors checkpoint dir must report the summed weight-file
        // bytes, not the directory inode's ~4 KiB st_size — memory planning
        // charges this quantity against the KV budget.
        let dir =
            std::env::temp_dir().join(format!("mesh-routing-total-bytes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), vec![0u8; 4096]).unwrap();
        std::fs::write(dir.join("config.json"), vec![0u8; 128]).unwrap();
        std::fs::create_dir(dir.join("nested")).unwrap();
        std::fs::write(dir.join("nested").join("ignored.bin"), vec![0u8; 999_999]).unwrap();
        let total = total_model_bytes(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(total, 4096 + 128);
    }

    #[test]
    fn file_model_reports_its_own_bytes() {
        let file = std::env::temp_dir().join(format!(
            "mesh-routing-total-bytes-file-{}",
            std::process::id()
        ));
        std::fs::write(&file, vec![0u8; 2048]).unwrap();
        let total = total_model_bytes(&file);
        let _ = std::fs::remove_file(&file);
        assert_eq!(total, 2048);
    }

    #[test]
    fn symlinked_snapshot_entries_report_target_bytes() {
        // HuggingFace-style snapshot dirs symlink weight files into a blobs/
        // store. DirEntry::metadata is lstat: it sees the link itself
        // (is_file() false, len 0) and would total zero — worse than the
        // directory-inode bug this replaced. The reader must follow links.
        let root = std::env::temp_dir().join(format!(
            "mesh-routing-total-bytes-symlink-{}",
            std::process::id()
        ));
        let blobs = root.join("blobs");
        let snapshot = root.join("snapshots").join("abc123");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(blobs.join("blob-weights"), vec![0u8; 8192]).unwrap();
        std::fs::write(blobs.join("blob-config"), vec![0u8; 256]).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            "../../blobs/blob-weights",
            snapshot.join("model.safetensors"),
        )
        .unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("../../blobs/blob-config", snapshot.join("config.json"))
            .unwrap();
        let total = total_model_bytes(&snapshot);
        let _ = std::fs::remove_dir_all(&root);
        #[cfg(unix)]
        assert_eq!(total, 8192 + 256);
    }

    #[test]
    fn missing_model_reports_zero() {
        assert_eq!(
            total_model_bytes(Path::new("/nonexistent/mesh-routing-missing-model")),
            0
        );
    }
}

/// The current inference target selected by runtime planning.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InferenceTarget {
    /// No backend running anywhere.
    None,
    /// This node serves the model on the given local HTTP port.
    Local(u16),
    /// Another node serves the model; proxy via QUIC to this peer.
    Remote(iroh::EndpointId),
}

/// Per-model routing table.
#[derive(Clone, Debug, Default)]
pub struct ModelTargets {
    /// model_name -> list of inference targets.
    pub targets: HashMap<String, Vec<InferenceTarget>>,
    /// Shared round-robin counter across clones.
    counter: Arc<AtomicU64>,
}

impl ModelTargets {
    /// Get target for a specific model. Round-robins across multiple hosts.
    pub fn get(&self, model: &str) -> InferenceTarget {
        match self.targets.get(model) {
            Some(targets) if !targets.is_empty() => {
                let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize % targets.len();
                targets[idx].clone()
            }
            _ => InferenceTarget::None,
        }
    }

    /// All candidate targets for a model, preserving their current order.
    pub fn candidates(&self, model: &str) -> Vec<InferenceTarget> {
        self.targets.get(model).cloned().unwrap_or_default()
    }

    /// Round-robin pick from a caller-supplied candidate slice.
    pub fn pick_from(&self, candidates: &[InferenceTarget]) -> InferenceTarget {
        if candidates.is_empty() {
            InferenceTarget::None
        } else {
            let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize % candidates.len();
            candidates[idx].clone()
        }
    }

    /// Sticky pick from a caller-supplied candidate slice.
    pub fn pick_sticky_from(candidates: &[InferenceTarget], sticky_key: u64) -> InferenceTarget {
        if candidates.is_empty() {
            InferenceTarget::None
        } else {
            let idx = sticky_key as usize % candidates.len();
            candidates[idx].clone()
        }
    }
}
