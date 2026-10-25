use super::*;
fn fixture(dimensions: &[u64], kind: u32) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(6_u64.to_le_bytes());
    bytes.extend(b"weight");
    bytes.extend((dimensions.len() as u32).to_le_bytes());
    for dimension in dimensions {
        bytes.extend(dimension.to_le_bytes());
    }
    bytes.extend(kind.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes
}
fn check(bytes: &[u8]) -> DynResult<u64> {
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("shard.gguf");
    std::fs::write(&path, bytes).unwrap();
    inspect(&path, &[(2, (32, 18))].into_iter().collect())
}
#[test]
fn descriptor_scan_counts_quantized_bytes_without_reading_payload() {
    assert_eq!(check(&fixture(&[64], 2)).unwrap(), 36);
    assert_eq!(check(&fixture(&[64, 3], 2)).unwrap(), 108);
}
#[test]
fn descriptor_scan_refuses_rank_zero_dimension_unknown_type_alignment_and_overflow() {
    for (dimensions, kind, diagnostic) in [
        (vec![], 2, "rank"),
        (vec![32; 9], 2, "rank"),
        (vec![0], 2, "positive"),
        (vec![32, 0], 2, "positive"),
        (vec![64], 999, "unknown GGML tensor type"),
        (vec![33], 2, "unaligned"),
        (vec![64, u64::MAX], 2, "overflow"),
    ] {
        let error = check(&fixture(&dimensions, kind)).unwrap_err().to_string();
        assert!(
            error.contains("shard.gguf") && error.contains("weight") && error.contains(diagnostic),
            "{error}"
        );
    }
}
#[test]
fn descriptor_scan_rejects_truncation_and_header_count_bombs() {
    let bytes = fixture(&[64], 2);
    for length in 0..bytes.len() {
        assert!(check(&bytes[..length]).is_err(), "{length}");
    }
    let mut bomb = bytes;
    bomb[8..16].copy_from_slice(&1_000_001_u64.to_le_bytes());
    assert!(
        check(&bomb)
            .unwrap_err()
            .to_string()
            .contains("count exceeds bound")
    );
}

#[test]
fn descriptor_scan_refuses_overflow_across_individually_valid_tensors() {
    let one = fixture(&[u64::MAX / 32 * 32], 2);
    assert!(check(&one).is_ok());
    let mut two = one.clone();
    two[8..16].copy_from_slice(&2_u64.to_le_bytes());
    two.extend(&one[24..]);
    assert!(
        check(&two)
            .unwrap_err()
            .to_string()
            .contains("sum overflow")
    );
}
