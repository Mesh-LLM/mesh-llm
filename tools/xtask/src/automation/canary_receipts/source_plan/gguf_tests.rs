use super::gguf::{AdmissionError, Expected, verify};
use crate::automation::replay_matrix::model_preflight::dimensions::Dimensions;

pub(super) fn fixture(architecture: &str, fields: &[(&str, u64)]) -> Vec<u8> {
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend(u64::try_from(value.len()).unwrap().to_le_bytes());
        bytes.extend(value.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(u64::try_from(fields.len() + 1).unwrap().to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, architecture);
    for (key, value) in fields {
        string(&mut bytes, key);
        bytes.extend(10_u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

fn target() -> Dimensions {
    Dimensions {
        architecture: "llama".into(),
        block_count: 32,
        activation_width: 2048,
        mtp_layers: 0,
    }
}

#[test]
fn rejection_when_target_architecture_count_width_or_mtp_disagrees() {
    for expected in [
        Dimensions {
            architecture: "rwkv6".into(),
            ..target()
        },
        Dimensions {
            block_count: 31,
            ..target()
        },
        Dimensions {
            activation_width: 1024,
            ..target()
        },
        Dimensions {
            mtp_layers: 1,
            ..target()
        },
    ] {
        let state = tempfile::tempdir().unwrap();
        let path = state.path().join("target.gguf");
        std::fs::write(
            &path,
            fixture(
                "llama",
                &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
            ),
        )
        .unwrap();

        let result = verify(&[path], Expected::Target(&expected));

        assert!(matches!(result, Err(AdmissionError::Mismatch { .. })));
    }
}

#[test]
fn admission_when_only_one_target_shard_carries_dimensions() {
    let state = tempfile::tempdir().unwrap();
    let paths = [state.path().join("one.gguf"), state.path().join("two.gguf")];
    std::fs::write(&paths[0], fixture("llama", &[])).unwrap();
    std::fs::write(
        &paths[1],
        fixture(
            "llama",
            &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
        ),
    )
    .unwrap();

    let result = verify(&paths, Expected::Target(&target()));

    assert!(result.is_ok());
}

#[test]
fn rejection_when_later_metadata_bearing_target_shard_disagrees() {
    let state = tempfile::tempdir().unwrap();
    let paths = [state.path().join("one.gguf"), state.path().join("two.gguf")];
    std::fs::write(
        &paths[0],
        fixture(
            "llama",
            &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
        ),
    )
    .unwrap();
    std::fs::write(
        &paths[1],
        fixture(
            "llama",
            &[("llama.block_count", 32), ("llama.embedding_length", 4096)],
        ),
    )
    .unwrap();

    let result = verify(&paths, Expected::Target(&target()));

    assert!(matches!(result, Err(AdmissionError::Mismatch { .. })));
}

#[test]
fn rejection_when_target_or_draft_has_no_metadata_bearing_shard() {
    let planned = target();
    for expected in [Expected::Target(&planned), Expected::Draft] {
        let state = tempfile::tempdir().unwrap();
        let path = state.path().join("shard.gguf");
        std::fs::write(&path, fixture("llama", &[])).unwrap();

        let result = verify(&[path], expected);

        assert!(matches!(result, Err(AdmissionError::MissingDimensions)));
    }
}

#[test]
fn draft_when_valid_architecture_and_dimensions_differ_from_target() {
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("draft.gguf");
    std::fs::write(
        &path,
        fixture(
            "dflash",
            &[("dflash.block_count", 4), ("dflash.embedding_length", 512)],
        ),
    )
    .unwrap();

    let result = verify(&[path], Expected::Draft);

    assert!(result.is_ok());
}
