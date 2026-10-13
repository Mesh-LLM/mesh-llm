use super::*;
use std::os::unix::fs::PermissionsExt as _;
fn pin(path: PathBuf, bytes: &[u8]) -> admission::Artifact {
    std::fs::write(&path, bytes).unwrap();
    admission::Artifact {
        path,
        sha256: admission::digest(bytes),
    }
}
fn script(base: &Path, role: &str) -> admission::Artifact {
    let executable = std::env::current_exe().unwrap();
    let source = format!(
        "#!/bin/bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" > '{}/argv'\nexport QUANT_JOB_CHAIN_ROOT='{}'\nexport QUANT_JOB_CHAIN_ROLE='{}'\n'{}' --ignored --exact automation::hf_certify::quant_job::chain_tests::quant_job_owned_chain_child --nocapture | sed -n 's/^QUANT_JOB_JSON://p'\n",
        base.display(),
        base.display(),
        role,
        executable.display()
    );
    let a = pin(base.join(role), source.as_bytes());
    std::fs::set_permissions(&a.path, std::fs::Permissions::from_mode(0o700)).unwrap();
    a
}
pub(in crate::automation::hf_certify) fn new(
    mode: &str,
    combined: bool,
) -> (tempfile::TempDir, Input, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let b = temp.path().canonicalize().unwrap();
    std::fs::create_dir_all(b.join("source/BF16")).unwrap();
    std::fs::create_dir(b.join("evidence")).unwrap();
    let bytes = b"GGUF\x03\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    let parts = (1..=2)
        .map(|i| {
            pin(
                b.join(format!("source/BF16/model-{i:05}-of-00002.gguf")),
                bytes,
            )
        })
        .collect();
    let recipe = pin(b.join("recipe"), b"tensor fixture");
    let tool = script(&b, "tool");
    let helper = script(&b, "helper");
    let manifest = json!({"schema_version":1,"kind":"QUANTIZE_GGUF","source":b.join("source"),"source_prefix":"BF16",
      "target":b.join("target"),"target_prefix":"Q4","output_basename":"model","expected_splits":2,"window_size":1,"quant":"Q4_0","tensor_type_file":recipe.path});
    let w = window::contract::Input {
        schema_version: 1,
        tool_kind: "supplied-window-quantizer".into(),
        profile_version: "inert-chain-v1".into(),
        tool,
        tool_source: pin(b.join("tool-source"), b"inert tool source"),
        runtime: pin(b.join("runtime"), b"inert runtime"),
        manifest: pin(b.join("manifest"), &serde_json::to_vec(&manifest).unwrap()),
        recipe,
        helper,
        helper_source: pin(b.join("helper-source"), b"inert helper source"),
        source_repo: "fixture/source".into(),
        source_revision: "a".repeat(40),
        source_root: b.join("source"),
        source_prefix: "BF16".into(),
        source_parts: parts,
        target_repo: "fixture/quant".into(),
        target_root: b.join("target"),
        target_prefix: "Q4".into(),
        basename: "model".into(),
        quant: "Q4_0".into(),
        expected_splits: 2,
        ordinal: 1,
        work_root: b.join("work"),
        credential_file: b.join("credential"),
        publication_confirmed: true,
        timeout_seconds: 120,
        resume: None,
    };
    std::fs::write(&w.credential_file, b"fixture-only-token").unwrap();
    std::fs::set_permissions(&w.credential_file, std::fs::Permissions::from_mode(0o600)).unwrap();
    let package = combined.then(|| contract::Package {
        writer: script(&b, "writer"),
        writer_source: pin(b.join("writer-source"), b"inert writer source"),
        generation_defaults: pin(b.join("defaults"), b"{}"),
        target_repo: "fixture/package".into(),
        max_artifact_bytes: 1024,
    });
    let input = Input {
        schema_version: 1,
        workflow: if combined {
            contract::Workflow::QuantizationAndPackage
        } else {
            contract::Workflow::Quantization
        },
        timeout_seconds: 120,
        window_template: w,
        resumes: vec![],
        loader: pin(b.join("loader"), b"inert loader"),
        package,
    };
    std::fs::write(b.join("mode"), mode).unwrap();
    (temp, input, b.join("evidence"))
}
