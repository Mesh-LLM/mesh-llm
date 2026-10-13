//! Actual source-admitted CLI target publication and strict stale-row refusal.
use super::*;

fn add_gap(root: &Path, authority: &mut Value, status: &str) {
    let manifest = root.join("docs/skippy/llama-parity-candidates.json");
    let mut document: Value = serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    document["candidates"]
        .as_array_mut()
        .unwrap()
        .push(json!({"llama_model":"gap","family":"gap_family","status":status}));
    std::fs::write(&manifest, serde_json::to_vec(&document).unwrap()).unwrap();
    local_git(root, &["add", "docs/skippy/llama-parity-candidates.json"]);
    local_git(
        root,
        &["commit", "--quiet", "-m", "classified boundary gap"],
    );
    let revision = local_git(root, &["rev-parse", "HEAD"]);
    authority["controller"]["revision"] = revision.clone().into();
    authority["base"] = revision.into();
    let native = root.join(".deps/llama.cpp");
    std::fs::write(native.join("src/models/gap.cpp"), "void gap_model() {}\n").unwrap();
    local_git(&native, &["add", "src/models/gap.cpp"]);
    local_git(&native, &["commit", "--quiet", "-m", "native boundary gap"]);
    std::fs::write(
        native.join(".mesh-llm-patched-sha"),
        format!("{}\n", local_git(&native, &["rev-parse", "HEAD"])),
    )
    .unwrap();
}

#[test]
fn parity_frontdoor_publishes_source_bound_next_gap_and_refuses_stale_runnable_gap() {
    for (status, accepted) in [("needs_boundary_registration", true), ("candidate", false)] {
        let root = tempfile::tempdir().unwrap();
        let mut authority = source_authority(root.path(), &gguf());
        add_gap(root.path(), &mut authority, status);
        std::fs::create_dir(root.path().join("cache")).unwrap();
        let request = json!({"authority":authority,"cache_root":root.path().join("cache"),"mode":"inventory","statuses":[],"families":[],"llama_models":[],"priorities":[],"limit":null,"missing_only":false,"local_only":false,"policy":{},"run":null,"admission_seconds":20});
        let (ok, stdout, stderr) = run_verb(root.path(), &request, "parity-local");
        assert_eq!(ok, accepted, "{status}: {stderr}");
        if accepted {
            let output: Value = serde_json::from_str(&stdout).unwrap();
            assert_eq!(output["admission"]["status"], "parity_inventory_admitted");
            assert_eq!(
                output["admission"]["next_boundary_target"],
                json!({"llama_model":"gap","family":"gap_family","status":"needs_boundary_registration","source_file":root.path().canonicalize().unwrap().join(".deps/llama.cpp/src/models/gap.cpp")})
            );
        } else {
            assert!(
                stdout.is_empty(),
                "refused inventory published output: {stdout}"
            );
            assert!(
                stderr.contains("pending_reclassification: [\"gap\"]"),
                "{stderr}"
            );
        }
    }
}
