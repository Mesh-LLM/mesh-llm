use super::*;
use sha2::{Digest, Sha256};
fn gguf(context: u32) -> Vec<u8> {
    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, "fixture");
    string(&mut bytes, "fixture.context_length");
    bytes.extend(4_u32.to_le_bytes());
    bytes.extend(context.to_le_bytes());
    bytes
}
fn fixture() -> (tempfile::TempDir, Input) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    fs::write(root.join("binary"), b"fixture executable bytes").unwrap();
    fs::create_dir(root.join("native-runtimes")).unwrap();
    fs::write(root.join("model.gguf"), gguf(4096)).unwrap();
    let input = Input {
        schema_version: 1,
        binaries: [root.join("binary"), root.join("binary")],
        model: root.join("model.gguf"),
        minimum_context_tokens: 2048,
    };
    (dir, input)
}
#[test]
fn local_file_digests_and_gguf_metadata_reuse_owning_verifier() {
    let (_dir, input) = fixture();
    let evidence = execute(&input, &Cancellation::default()).unwrap();
    assert_eq!(
        evidence.binaries[0].file.sha256,
        hex::encode(Sha256::digest(b"fixture executable bytes"))
    );
    assert_eq!(
        evidence.model.sha256,
        hex::encode(Sha256::digest(gguf(4096)))
    );
    assert_eq!(evidence.model_metadata.architecture, "fixture");
    assert_eq!(evidence.model_metadata.native_context_tokens, 4096);
    assert_eq!(evidence.model_metadata.sha256, evidence.model.sha256);
    assert_eq!(
        evidence.binaries[0].runtime_validation,
        "pending_owning_native_package_and_loader_policy"
    );
}
#[test]
fn preflight_refuses_nonlocal_empty_missing_and_directory_inputs() {
    let (dir, mut input) = fixture();
    input.model = PathBuf::from("relative.gguf");
    assert!(execute(&input, &Cancellation::default()).is_err());
    input.model = dir.path().join("missing");
    assert!(execute(&input, &Cancellation::default()).is_err());
    input.model = dir.path().join("native-runtimes");
    assert!(execute(&input, &Cancellation::default()).is_err());
    input.model = dir.path().join("empty");
    fs::write(&input.model, []).unwrap();
    assert!(execute(&input, &Cancellation::default()).is_err());
}
#[test]
fn gguf_format_and_native_context_failure_prevent_identity_admission() {
    let (_dir, mut input) = fixture();
    input.minimum_context_tokens = 4097;
    assert!(execute(&input, &Cancellation::default()).is_err());
    input.minimum_context_tokens = 1;
    fs::write(&input.model, b"not a model").unwrap();
    assert!(execute(&input, &Cancellation::default()).is_err());
}
#[test]
fn no_adjacent_runtime_root_never_uses_environment_or_download_fallback() {
    let (_dir, input) = fixture();
    fs::remove_dir(input.binaries[0].parent().unwrap().join("native-runtimes")).unwrap();
    assert!(execute(&input, &Cancellation::default()).is_err());
}
#[test]
fn cancellation_and_unknown_schema_refuse_before_publication() {
    let (_dir, mut input) = fixture();
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(execute(&input, &cancel).is_err());
    input.schema_version = 2;
    assert!(execute(&input, &Cancellation::default()).is_err());
}
#[test]
fn atomic_receipt_preserves_existing_bytes_and_emits_bound_schema() {
    let (dir, input) = fixture();
    let output = dir.path().canonicalize().unwrap().join("receipt.json");
    write_receipt(&input, &output, &Cancellation::default()).unwrap();
    let bytes = fs::read(&output).unwrap();
    let evidence: Evidence = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(evidence.schema_version, 1);
    assert!(write_receipt(&input, &output, &Cancellation::default()).is_err());
    assert_eq!(fs::read(&output).unwrap(), bytes);
}
#[cfg(unix)]
#[test]
fn canonical_binary_alias_resolves_real_adjacent_root_and_escaping_root_symlink_is_refused() {
    use std::os::unix::fs::symlink;
    let (dir, mut input) = fixture();
    let alias = dir.path().join("alias");
    symlink(&input.binaries[0], &alias).unwrap();
    input.binaries = [alias.clone(), alias];
    let evidence = execute(&input, &Cancellation::default()).unwrap();
    assert!(evidence.binaries[0].file.path.ends_with("binary"));
    let outside = tempfile::tempdir().unwrap();
    let root = input.binaries[0].parent().unwrap().join("native-runtimes");
    fs::remove_dir(&root).unwrap();
    symlink(outside.path(), &root).unwrap();
    assert!(execute(&input, &Cancellation::default()).is_err());
}
