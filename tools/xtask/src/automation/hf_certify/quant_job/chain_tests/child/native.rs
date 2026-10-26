use super::*;
use std::os::unix::fs::symlink;
pub(super) fn window(c: &Context) {
    let ordinal: u32 = c.get("--first-split").parse().unwrap();
    assert_eq!(c.get("--last-split"), ordinal.to_string());
    assert_eq!(c.get("--max-windows"), "1");
    assert_eq!(c.get("--max-memory"), "32G");
    assert_eq!(c.get("--memory-policy"), "hard");
    assert!(c.args.iter().any(|s| s == "--keep-staged-source"));
    c.event(&format!("quant-{ordinal}"));
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(c.get("--manifest")).unwrap()).unwrap();
    let work = PathBuf::from(c.get("--work-dir"));
    let stage = work.join("source-window/BF16");
    std::fs::create_dir_all(&stage).unwrap();
    let source = PathBuf::from(manifest["source"].as_str().unwrap()).join("BF16");
    for i in 1..=2 {
        let name = format!("model-{i:05}-of-00002.gguf");
        if i == ordinal {
            std::fs::copy(source.join(&name), stage.join(&name)).unwrap();
        } else {
            symlink(source.join(&name), stage.join(&name)).unwrap();
        }
    }
    let target = PathBuf::from(manifest["target"].as_str().unwrap()).join("Q4");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::copy(
        source.join(format!("model-{ordinal:05}-of-00002.gguf")),
        target.join(format!("model-{ordinal:05}-of-00002.gguf")),
    )
    .unwrap();
}
pub(super) fn verify(c: &Context) {
    c.event("native-verify");
    assert!(c.args.iter().any(|s| s == "--check-tensors"));
    assert!(c.args.iter().any(|s| s == "--llama-load"));
    let m: Value = serde_json::from_slice(&std::fs::read(c.get("--manifest")).unwrap()).unwrap();
    let target = PathBuf::from(m["target"].as_str().unwrap());
    let prefix = m["target_prefix"].as_str().unwrap();
    let first = target.join(prefix).join("model-00001-of-00002.gguf");
    assert!(first.is_file());
    assert!(
        target
            .join(prefix)
            .join("model-00002-of-00002.gguf")
            .is_file()
    );
    let success = c.mode != "native";
    c.json(&json!({"artifact":{"complete":true,"expected_splits":2,"completed_count":2,"root":target,"prefix":prefix,"basename":"model"},
      "llama_load":{"success":success,"status_code":if success{0}else{47},"llama_cli":c.get("--llama-cli"),"model":first}}));
}
pub(super) fn package(c: &Context) {
    c.event("package-write");
    let source = PathBuf::from(&c.args[1]);
    let root = PathBuf::from(c.get("--out-dir"));
    std::fs::create_dir_all(root.join("layers")).unwrap();
    let artifact = root.join("layers/model.bin");
    std::fs::copy(&source, &artifact).unwrap();
    let files = (1..=2)
        .map(|i| {
            let name = format!("model-{i:05}-of-00002.gguf");
            let id = identity(&source.parent().unwrap().join(&name));
            json!({"path":format!("Q4/{name}"),"sha256":id["sha256"],"byte_size":id["byte_size"]})
        })
        .collect::<Vec<_>>();
    let id = identity(&artifact);
    write(
        &root.join("model-package.json"),
        &json!({"schema_version":2,"format":"gguf","package_id":"fixture-package",
      "model_id":c.get("--model-id"),"source_model":{"repo":c.get("--source-repo"),"revision":c.get("--source-revision"),
      "primary_file":c.get("--source-file"),"files":files},"artifact_catalog":{"entries":[{"id":"layer-0","path":"layers/model.bin","sha256":id["sha256"],"byte_size":id["byte_size"]}]}}),
    );
}
pub(super) fn package_verify(c: &Context) {
    c.event("package-verify");
    let root = PathBuf::from(&c.args[1]);
    let source = PathBuf::from(c.get("--source"));
    let m: Value =
        serde_json::from_slice(&std::fs::read(root.join("model-package.json")).unwrap()).unwrap();
    for a in m["source_model"]["files"].as_array().unwrap() {
        let name = Path::new(a["path"].as_str().unwrap()).file_name().unwrap();
        let id = identity(&source.parent().unwrap().join(name));
        assert_eq!(id["sha256"], a["sha256"]);
        assert_eq!(id["byte_size"], a["byte_size"]);
    }
    c.json(&json!({"package_id":m["package_id"],"source_completeness_verified":c.mode!="package","checked_source_files":2,"checked_artifacts":1,"checked_tensors":1,"checked_projectors":0}));
}
