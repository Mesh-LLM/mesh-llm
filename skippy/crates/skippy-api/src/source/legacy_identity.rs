use sha2::{Digest, Sha256};

use super::{SkippyPackageSourceFile, hex_lower};

/// Preserve the pre-content-addressed aggregate byte-for-byte. Legacy direct
/// GGUF manifests intentionally remain path-bound for mixed-version cache
/// compatibility; only the explicit content-addressed mode is relocatable.
pub(super) fn aggregate_source_sha256(source_files: &[SkippyPackageSourceFile]) -> String {
    if source_files.len() == 1 {
        return source_files[0].sha256.clone();
    }
    let mut hasher = Sha256::new();
    for file in source_files {
        hasher.update(file.path.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(file.bytes.to_le_bytes());
        hasher.update([0]);
        hasher.update(file.sha256.as_bytes());
        hasher.update([0]);
    }
    hex_lower(&hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn split_digest_remains_path_bound_and_stable() {
        let first = vec![
            SkippyPackageSourceFile {
                path: PathBuf::from("/models/model-00001-of-00002.gguf"),
                bytes: 10,
                sha256: "a".repeat(64),
            },
            SkippyPackageSourceFile {
                path: PathBuf::from("/models/model-00002-of-00002.gguf"),
                bytes: 20,
                sha256: "b".repeat(64),
            },
        ];
        let relocated = first
            .iter()
            .enumerate()
            .map(|(index, file)| SkippyPackageSourceFile {
                path: PathBuf::from(format!("/relocated/shard-{index}.gguf")),
                ..file.clone()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            aggregate_source_sha256(&first),
            "493ed077339e02c9b8b8a00561acfa26cbc2de61c603ad749c91013e58077ed2"
        );
        assert_ne!(
            aggregate_source_sha256(&first),
            aggregate_source_sha256(&relocated)
        );
    }
}
