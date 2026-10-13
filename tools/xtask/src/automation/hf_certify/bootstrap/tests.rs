use super::*;
fn input() -> contract::Input {
    let root = std::env::current_dir().unwrap();
    contract::Input {
        schema_version: 1,
        mesh_commit: "a".repeat(40),
        git_tree: "b".repeat(40),
        llama_commit: "c".repeat(40),
        upstream_file_sha256: "d".repeat(64),
        image: format!("ghcr.io/owner/prepared@sha256:{}", "e".repeat(64)),
        native_profile: "standalone-static-skippy-quantize-cpu".into(),
        tools: [
            "git", "just", "cargo", "rustc", "cmake", "c++", "ld.lld", "curl",
        ]
        .map(|name| contract::Tool {
            name: name.into(),
            path: root.join(name),
            sha256: "f".repeat(64),
        })
        .into(),
        path_directories: vec![root],
        timeout_seconds: 300,
        cpu_plan_receipt_sha256: "1".repeat(64),
        declared_estimate_usd: 1.0,
        max_cost_usd: 2.0,
    }
}
#[test]
fn bootstrap_requires_immutable_source_image_and_exact_tools() {
    let base = input();
    base.validate().unwrap();
    for field in ["mesh", "tree", "llama", "image", "profile", "tools"] {
        let mut value = base.clone();
        match field {
            "mesh" => value.mesh_commit = "main".into(),
            "tree" => value.git_tree = "main".into(),
            "llama" => value.llama_commit = "latest".into(),
            "image" => value.image = "ubuntu:latest".into(),
            "profile" => value.native_profile = "cuda".into(),
            _ => value.tools.pop().map(|_| ()).unwrap(),
        };
        assert!(value.validate().is_err());
    }
}
#[test]
fn bootstrap_cost_declaration_never_accepts_nan_infinity_or_over_cap() {
    for value in [f64::NAN, f64::INFINITY, -1.0, 3.0] {
        let mut request = input();
        request.declared_estimate_usd = value;
        assert!(request.validate().is_err());
    }
    let mut value = input();
    value.tools[0] = value.tools[1].clone();
    assert!(value.validate().is_err());
}
#[test]
fn bootstrap_precancel_and_expired_budget_refuse_before_any_tool_or_checkout() {
    let request = input();
    for cancelled in [true, false] {
        let cancellation = crate::process::Cancellation::default();
        if cancelled {
            cancellation.cancel();
        }
        let mut rows = Vec::new();
        assert!(
            execution::execute(
                &request,
                &std::env::current_dir().unwrap(),
                Instant::now(),
                &cancellation,
                &mut rows
            )
            .is_err()
        );
        assert!(rows.is_empty());
    }
}

#[test]
fn bootstrap_terminal_decision_refuses_cancelled_or_expired_completion() {
    let cancel = crate::process::Cancellation::default();
    terminal_decision(Instant::now() + Duration::from_secs(1), &cancel).unwrap();
    assert!(terminal_decision(Instant::now(), &cancel).is_err());
    cancel.cancel();
    assert!(terminal_decision(Instant::now() + Duration::from_secs(1), &cancel).is_err());
}
