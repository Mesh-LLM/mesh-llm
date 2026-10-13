use super::dimensions::{DimensionsError, inspect};

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend(u64::try_from(value.len()).unwrap().to_le_bytes());
    bytes.extend(value.as_bytes());
}

pub(crate) fn fixture(architecture: &str, fields: &[(&str, i64)]) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(u64::try_from(fields.len() + 1).unwrap().to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, architecture);
    for (key, value) in fields {
        string(&mut bytes, key);
        bytes.extend(11_u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

fn read(bytes: &[u8]) -> Result<Option<super::dimensions::Dimensions>, DimensionsError> {
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("fixture.gguf");
    std::fs::write(&path, bytes).unwrap();
    inspect(&path)
}

#[test]
fn dimensions_when_dense_or_recurrent_metadata_is_valid() {
    for architecture in ["llama", "rwkv6"] {
        let fields = [
            (format!("{architecture}.block_count"), 32),
            (format!("{architecture}.embedding_length"), 2048),
        ];
        let fields: Vec<_> = fields
            .iter()
            .map(|(key, value)| (key.as_str(), *value))
            .collect();
        let bytes = fixture(architecture, &fields);

        let result = read(&bytes).unwrap().unwrap();

        assert_eq!(
            (
                result.block_count,
                result.activation_width,
                result.mtp_layers
            ),
            (32, 2048, 0)
        );
    }
}

#[test]
fn activation_width_when_qwen4_or_dflash_is_hyper_connected() {
    for architecture in ["qwen4exp", "dflash"] {
        let fields: Vec<_> = [
            ("block_count", 33),
            ("embedding_length", 2048),
            ("hyper_connection.count", 4),
            ("embedding_length_out", 8192),
            ("nextn_predict_layers", 1),
        ]
        .into_iter()
        .map(|(key, value)| (format!("{architecture}.{key}"), value))
        .collect();
        let fields: Vec<_> = fields
            .iter()
            .map(|(key, value)| (key.as_str(), *value))
            .collect();
        let bytes = fixture(architecture, &fields);

        let result = read(&bytes).unwrap().unwrap();

        assert_eq!(
            (
                result.block_count,
                result.activation_width,
                result.mtp_layers
            ),
            (33, 8192, 1)
        );
    }
}

#[test]
fn rejection_when_dimensions_are_duplicate_partial_or_wrongly_qualified() {
    for fields in [
        vec![("llama.block_count", 32)],
        vec![("other.block_count", 32), ("llama.embedding_length", 2048)],
        vec![
            ("llama.block_count", 32),
            ("llama.block_count", 64),
            ("llama.embedding_length", 2048),
        ],
        vec![
            ("llama.block_count", 32),
            ("llama.embedding_length", 2048),
            ("other.embedding_length", 2048),
        ],
        vec![("llama.block_count", 0), ("llama.embedding_length", 2048)],
    ] {
        let bytes = fixture("llama", &fields);

        let result = read(&bytes);

        assert!(matches!(result, Err(DimensionsError::Field { .. })));
    }
}

#[test]
fn rejection_when_native_mtp_count_is_negative_duplicate_foreign_or_out_of_range() {
    for extra in [
        vec![("llama.nextn_predict_layers", -1)],
        vec![("llama.nextn_predict_layers", 32)],
        vec![("other.nextn_predict_layers", 1)],
        vec![
            ("llama.nextn_predict_layers", 1),
            ("llama.nextn_predict_layers", 1),
        ],
    ] {
        let mut fields = vec![("llama.block_count", 32), ("llama.embedding_length", 2048)];
        fields.extend(extra);
        let bytes = fixture("llama", &fields);

        let result = read(&bytes);

        assert!(matches!(result, Err(DimensionsError::Field { .. })));
    }
}

#[test]
fn rejection_when_hyper_connection_width_overflows_or_output_disagrees() {
    for (embedding, count, output) in [
        (2048, 4, 2048),
        (i64::MAX, i64::MAX, 1),
        (i64::from(i32::MAX), 2, 1),
        (2048, 0, 1),
    ] {
        let bytes = fixture(
            "qwen4exp",
            &[
                ("qwen4exp.block_count", 32),
                ("qwen4exp.embedding_length", embedding),
                ("qwen4exp.hyper_connection.count", count),
                ("qwen4exp.embedding_length_out", output),
            ],
        );

        let result = read(&bytes);

        assert!(result.is_err());
    }
}

#[test]
fn rejection_when_any_metadata_prefix_is_truncated() {
    let bytes = fixture(
        "llama",
        &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
    );
    for length in 0..bytes.len() {
        let result = read(&bytes[..length]);

        assert!(result.is_err(), "accepted truncated prefix {length}");
    }
}

#[test]
fn arrays_when_fixed_or_variable_values_are_skipped() {
    let mut bytes = fixture(
        "llama",
        &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
    );
    bytes[16..24].copy_from_slice(&5_u64.to_le_bytes());
    string(&mut bytes, "tokenizer.ids");
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(4_u32.to_le_bytes());
    bytes.extend(1024_u64.to_le_bytes());
    bytes.extend([0; 4096]);
    string(&mut bytes, "tokenizer.tokens");
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(8_u32.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    string(&mut bytes, "one");
    string(&mut bytes, "two");

    let result = read(&bytes).unwrap().unwrap();

    assert_eq!(result.activation_width, 2048);
}

#[test]
fn rejection_when_array_byte_size_overflows() {
    let mut bytes = fixture("llama", &[]);
    bytes[16..24].copy_from_slice(&2_u64.to_le_bytes());
    string(&mut bytes, "tokenizer.ids");
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(10_u32.to_le_bytes());
    bytes.extend(u64::MAX.to_le_bytes());

    let result = read(&bytes);

    assert!(matches!(result, Err(DimensionsError::Metadata(_))));
}

#[test]
fn rejection_when_qwen4_requires_missing_hyper_connection_count() {
    let bytes = fixture(
        "qwen4exp",
        &[
            ("qwen4exp.block_count", 32),
            ("qwen4exp.embedding_length", 2048),
        ],
    );

    let result = read(&bytes);

    assert!(matches!(
        result,
        Err(DimensionsError::Field {
            field: "hyper_connection.count",
            ..
        })
    ));
}

#[test]
fn rejection_when_architecture_is_conflicting() {
    let mut bytes = fixture(
        "llama",
        &[("llama.block_count", 32), ("llama.embedding_length", 2048)],
    );
    bytes[16..24].copy_from_slice(&4_u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, "rwkv6");

    let result = read(&bytes);

    assert!(matches!(
        result,
        Err(DimensionsError::Field {
            field: "architecture",
            ..
        })
    ));
}

#[test]
fn rejection_when_dimension_is_boolean_instead_of_integer() {
    let mut bytes = fixture("llama", &[("llama.block_count", 32)]);
    bytes[16..24].copy_from_slice(&3_u64.to_le_bytes());
    string(&mut bytes, "llama.embedding_length");
    bytes.extend(7_u32.to_le_bytes());
    bytes.push(1);

    let result = read(&bytes);

    assert!(matches!(
        result,
        Err(DimensionsError::Field {
            field: "embedding_length",
            ..
        })
    ));
}
