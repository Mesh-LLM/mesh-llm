use super::*;
const DATA: &str = "class GGMLQuantizationType(IntEnum):\n    F32 = 0\n    Q4_0 = 2\n    Q2_K = 10\n\nQK_K = 256\nGGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {\n    GGMLQuantizationType.F32: (1, 4),\n    GGMLQuantizationType.Q4_0: (32, 2 + 16),\n    GGMLQuantizationType.Q2_K: (256, 2 + 2 + QK_K // 16 + QK_K // 4),\n}\n";
#[test]
fn prepared_layout_data_resolves_current_table_arithmetic() {
    let layouts = parse(DATA).unwrap();
    assert_eq!(layouts[&0], (1, 4));
    assert_eq!(layouts[&2], (32, 18));
    assert_eq!(layouts[&10], (256, 84));
    assert_eq!(expression("2 + 4 * 13", 256).unwrap(), 54);
}
#[test]
fn layout_data_refuses_executable_expressions_duplicates_unknowns_and_overflow() {
    for (before, after) in [
        ("2 + 16", "__import__('os').system('false')"),
        ("2 + 16", "2 ** 4"),
        ("2 + 16", "18446744073709551615 + 1"),
        ("2 + 16", "18446744073709551615 * 2"),
        ("QK_K // 16", "QK_K // 0"),
        ("QK_K // 16", "QK_K * 3 // 2"),
        ("F32 = 0", "F32 = 2"),
        ("F32: (1, 4)", "UNKNOWN: (1, 4)"),
        ("QK_K = 256", "QK_K = 256\nQK_K = 256"),
        ("Q4_0 = 2", "Q4_0 = 2\n    Q4_0 = 2"),
        ("(32, 2 + 16)", "(0, 18)"),
        ("(32, 2 + 16)", "(32, 0)"),
    ] {
        assert!(parse(&DATA.replace(before, after)).is_err(), "{after}");
    }
}
#[test]
fn bounded_layout_read_rejects_oversize_source() {
    let state = tempfile::tempdir().unwrap();
    let path = state.path().join("constants.py");
    std::fs::write(&path, vec![b' '; MAX_SOURCE + 1]).unwrap();
    assert!(read(&path).is_err());
}

#[test]
fn layout_table_requires_its_complete_finite_declaration() {
    assert!(parse(DATA.trim_end().strip_suffix('}').unwrap()).is_err());
    assert!(
        parse(&DATA.replace(
            "class GGMLQuantizationType(IntEnum):",
            "class GGMLQuantizationType(IntEnum):\nclass GGMLQuantizationType(IntEnum):"
        ))
        .is_err()
    );
}
