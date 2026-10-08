use serde_json::{Value, json};
use std::{fs, path::Path};
fn artifact(path: &str, bytes: &[u8]) -> Value {
    use sha2::{Digest as _, Sha256};
    let sha256: String = Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    json!({"path":path,"artifact_bytes":bytes.len(),"tensor_count":1,"tensor_bytes":1,"sha256":sha256})
}
pub(super) fn fixture(root: &Path) -> Value {
    let mut layers = Vec::new();
    for index in 0..2 {
        let path = format!("layers/{index}.gguf");
        fs::create_dir_all(root.join("layers")).unwrap();
        fs::write(root.join(&path), b"layer").unwrap();
        let mut a = artifact(&path, b"layer");
        a["layer_index"] = json!(index);
        layers.push(a);
    }
    for (path, bytes) in [
        ("metadata.gguf", b"metadata".as_slice()),
        ("embeddings.gguf", b"embeddings".as_slice()),
        ("output.gguf", b"output".as_slice()),
        ("projector.gguf", b"projector".as_slice()),
    ] {
        fs::write(root.join(path), bytes).unwrap();
    }
    let mut projector = artifact("projector.gguf", b"projector");
    projector["kind"] = json!("mmproj");
    let manifest = json!({"schema_version":1,"format":"layer-package","model_id":"fixture/model","layer_count":2,"activation_width":4096,
        "source_model":{"path":"source.gguf","sha256":"a".repeat(64)},
        "shared":{"metadata":artifact("metadata.gguf",b"metadata"),"embeddings":artifact("embeddings.gguf",b"embeddings"),"output":artifact("output.gguf",b"output")},
        "layers":layers,"projectors":[projector],
        "skippy_abi_version":format!("{}.{}.{}",skippy_ffi::ABI_VERSION_MAJOR,skippy_ffi::ABI_VERSION_MINOR,skippy_ffi::ABI_VERSION_PATCH)});
    save(root, &manifest);
    manifest
}
pub(super) fn save(root: &Path, manifest: &Value) {
    fs::write(
        root.join("model-package.json"),
        serde_json::to_vec(manifest).unwrap(),
    )
    .unwrap();
}
