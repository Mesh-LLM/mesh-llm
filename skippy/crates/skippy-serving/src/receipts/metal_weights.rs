//! Conservative Metal admission from the same local artifacts the loader uses.
//! All matrices must be Q8_0, including embeddings and the output head.

use std::path::Path;

use anyhow::{Result, bail, ensure};
use skippy_protocol::StageConfig;
use skippy_receipts::weights::{LocalFile, MetaValue, WeightFile};
use skippy_runtime::GGML_TYPE_Q8_0;

pub(super) fn unaudited(config: &StageConfig) -> Option<String> {
    check(config)
        .err()
        .map(|error| format!("Metal tensor coverage is unqualified: {error:#}"))
}

fn check(config: &StageConfig) -> Result<()> {
    // Check loaded artifacts, never the operator's claimed audit model.
    let paths: Vec<&str> = if config.model_part_paths.is_empty() {
        config.model_path.as_deref().into_iter().collect()
    } else {
        config.model_part_paths.iter().map(String::as_str).collect()
    };
    ensure!(!paths.is_empty(), "no loaded GGUF artifacts");
    let mut matrices = 0;
    for path in paths {
        let weights = WeightFile::open(Box::new(LocalFile::open(Path::new(path))?))?;
        // The single-path native loader discovers shards that this header cannot cover.
        ensure!(
            !config.model_part_paths.is_empty()
                || matches!(
                    weights.metadata.get("split.count"),
                    None | Some(MetaValue::Int(0 | 1))
                ),
            "implicit GGUF shards are unqualified; enumerate every loaded model part"
        );
        for tensor in weights.tensors.values().filter(|tensor| tensor.is_matrix()) {
            check_matrix(tensor)?;
            matrices += 1;
        }
    }
    ensure!(matrices > 0, "no matrix tensor metadata");
    Ok(())
}

fn check_matrix(tensor: &skippy_receipts::weights::TensorInfo) -> Result<()> {
    if tensor.ggml_type != GGML_TYPE_Q8_0 {
        bail!("{} has type {}, not Q8_0", tensor.name, tensor.ggml_type);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_matrix_types_and_unqualified_heads_are_rejected() {
        for name in ["blk.0.attn_q.weight", "output.weight", "token_embd.weight"] {
            for (ggml_type, covered) in
                [(GGML_TYPE_Q8_0, true), (0, false), (12, false), (14, false)]
            {
                let tensor = skippy_receipts::weights::TensorInfo {
                    name: name.into(),
                    dims: vec![32, 4],
                    ggml_type,
                    offset: 0,
                };
                assert!(tensor.is_matrix());
                assert_eq!(
                    check_matrix(&tensor).is_ok(),
                    covered,
                    "{name} type {ggml_type}"
                );
            }
        }
    }

    // Minimal GGUF with a Q8 layer, a selectable head, and an F32 norm.
    fn artifact(head_type: u32) -> tempfile::NamedTempFile {
        artifact_with_split(head_type, None)
    }

    fn artifact_with_split(head_type: u32, split_count: Option<u16>) -> tempfile::NamedTempFile {
        fn string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(3u64.to_le_bytes());
        bytes.extend(u64::from(split_count.is_some()).to_le_bytes());
        if let Some(count) = split_count {
            string(&mut bytes, "split.count");
            bytes.extend(2u32.to_le_bytes()); // GGUF UINT16
            bytes.extend(count.to_le_bytes());
        }
        for (name, dims, kind, offset) in [
            ("blk.0.attn_q.weight", vec![32u64, 4], GGML_TYPE_Q8_0, 0u64),
            ("output.weight", vec![32, 4], head_type, 160),
            (
                "output_norm.weight",
                vec![32],
                0,
                if head_type == GGML_TYPE_Q8_0 {
                    320
                } else {
                    672
                },
            ),
        ] {
            string(&mut bytes, name);
            bytes.extend((dims.len() as u32).to_le_bytes());
            for dim in dims {
                bytes.extend(dim.to_le_bytes());
            }
            bytes.extend(kind.to_le_bytes());
            bytes.extend(offset.to_le_bytes());
        }
        bytes.resize(bytes.len().div_ceil(32) * 32 + 1024, 0);
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), bytes).unwrap();
        file
    }

    #[test]
    fn loaded_artifacts_include_the_head_but_not_scalar_norms() {
        let q8 = artifact(GGML_TYPE_Q8_0);
        let mixed = artifact(0);
        let mut config = StageConfig {
            model_path: Some(q8.path().to_str().unwrap().into()),
            ..StageConfig::default()
        };
        assert!(unaudited(&config).is_none(), "{:?}", unaudited(&config));
        // Parts are what the loader reads, even when model_path claims Q8.
        config.model_part_paths = vec![mixed.path().to_str().unwrap().into()];
        assert!(unaudited(&config).unwrap().contains("output.weight"));
        config
            .model_part_paths
            .push(q8.path().to_str().unwrap().into());
        assert!(unaudited(&config).is_some());
        config.model_part_paths = vec![q8.path().to_str().unwrap().into()];
        config.model_path = Some("not-the-loaded-artifact.gguf".into());
        assert!(unaudited(&config).is_none());
    }

    #[test]
    fn implicit_shards_cannot_hide_an_unqualified_later_matrix() {
        let first = artifact_with_split(GGML_TYPE_Q8_0, Some(2));
        let later = artifact(0);
        let mut config = StageConfig {
            model_path: Some(first.path().to_str().unwrap().into()),
            ..StageConfig::default()
        };
        assert!(unaudited(&config).unwrap().contains("implicit GGUF shards"));
        // Explicit parts do not auto-discover; every supplied header is checked.
        config.model_part_paths = vec![
            first.path().to_str().unwrap().into(),
            later.path().to_str().unwrap().into(),
        ];
        assert!(unaudited(&config).unwrap().contains("output.weight"));
        let q8_later = artifact(GGML_TYPE_Q8_0);
        config.model_part_paths[1] = q8_later.path().to_str().unwrap().into();
        assert!(unaudited(&config).is_none());
    }

    #[test]
    fn missing_or_unreadable_loaded_metadata_keeps_receipts_off() {
        assert!(unaudited(&StageConfig::default()).is_some());
        let config = StageConfig {
            model_path: Some("/nonexistent-metal-receipt-model.gguf".into()),
            ..StageConfig::default()
        };
        assert!(unaudited(&config).is_some());
    }
}
