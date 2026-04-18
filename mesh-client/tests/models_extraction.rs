use mesh_client::models::capabilities::ModelCapabilities;
use mesh_client::models::catalog::MODEL_CATALOG;
use mesh_client::models::gguf::scan_gguf_compact_meta;

#[test]
fn catalog_has_entries() {
    assert!(MODEL_CATALOG.iter().count() > 0);
}

#[test]
fn capabilities_default_is_none() {
    let caps = ModelCapabilities::default();
    assert!(!caps.multimodal);
    assert!(!caps.moe);
}

#[test]
fn gguf_parse_minimal_fixture() {
    use std::io::Write;

    // Minimal valid GGUF: magic + version=3 + n_tensors=0 + n_kv=0
    let mut fixture = Vec::<u8>::new();
    fixture.extend_from_slice(b"GGUF"); // magic
    fixture.extend_from_slice(&3u32.to_le_bytes()); // version
    fixture.extend_from_slice(&0i64.to_le_bytes()); // n_tensors
    fixture.extend_from_slice(&0i64.to_le_bytes()); // n_kv

    let tmp = std::env::temp_dir().join("mesh-client-models-extraction.gguf");
    std::fs::File::create(&tmp)
        .unwrap()
        .write_all(&fixture)
        .unwrap();
    let meta = scan_gguf_compact_meta(&tmp);
    assert!(meta.is_some(), "should parse minimal GGUF fixture");
    let meta = meta.unwrap();
    assert_eq!(meta.context_length, 0);
    assert_eq!(meta.expert_count, 0);
    let _ = std::fs::remove_file(&tmp);
}
