//! Parse actual workflow/action trees, then mutate the owned cache obligations.
use super::{
    cache_evidence, cache_identity,
    support::{action, root},
    workflow_yaml::{self, Node},
};
use std::{collections::BTreeMap, fs};
fn workflows() -> BTreeMap<String, Node> {
    fs::read_dir(root().join(".github/workflows"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("yml" | "yaml")
            )
        })
        .map(|path| {
            (
                path.file_name().unwrap().to_str().unwrap().to_owned(),
                workflow_yaml::parse(&fs::read_to_string(path).unwrap()).unwrap(),
            )
        })
        .collect()
}
fn changed(name: &str, old: &str, new: &str) -> BTreeMap<String, Node> {
    let mut documents = workflows();
    let source = fs::read_to_string(root().join(".github/workflows").join(name)).unwrap();
    assert!(
        source.contains(old),
        "mutation preimage missing: {name}: {old}"
    );
    documents.insert(
        name.into(),
        workflow_yaml::parse(&source.replacen(old, new, 1)).unwrap(),
    );
    documents
}
#[test]
fn current_cache_capture_callers_and_compatibility_identities_are_owned() {
    let documents = workflows();
    cache_evidence::capture_action(&action("capture-sccache-stats")).unwrap();
    cache_evidence::configure_action(&action("configure-sccache-gha")).unwrap();
    cache_evidence::check(&documents).unwrap();
    cache_identity::check(&documents).unwrap();
}
#[test]
fn evidence_upload_binding_retention_and_enforcement_mutations_are_rejected() {
    let source =
        fs::read_to_string(root().join(".github/actions/capture-sccache-stats/action.yml"))
            .unwrap();
    for (old, new) in [
        ("retention-days: 14", "retention-days: 1"),
        ("if-no-files-found: error", "if-no-files-found: ignore"),
        (
            "path: ${{ steps.capture.outputs.stats_file }}",
            "path: /fixture/raw-sccache",
        ),
        ("name: ${{ inputs.artifact_name }}", "name: fixed-evidence"),
        ("steps.capture.outputs.cache_passed != 'true'", "false"),
        (
            "SCCACHE_CACHE_EXPECTATION: ${{ inputs.cache_expectation }}",
            "SCCACHE_CACHE_EXPECTATION: cold",
        ),
        (
            "SCCACHE_MINIMUM_HIT_RATE: ${{ inputs.minimum_hit_rate }}",
            "SCCACHE_MINIMUM_HIT_RATE: 0",
        ),
        ("id: capture", "id: raw"),
    ] {
        assert!(source.contains(old));
        let mutated = workflow_yaml::parse(&source.replacen(old, new, 1)).unwrap();
        assert!(
            cache_evidence::capture_action(&mutated).is_err(),
            "accepted mutation {old}"
        );
    }
    let mut reordered = action("capture-sccache-stats");
    let Node::Map(fields) = &mut reordered else {
        panic!("action");
    };
    let Node::Map(runs) = &mut fields.iter_mut().find(|(k, _)| k == "runs").unwrap().1 else {
        panic!("runs");
    };
    let Node::Seq(steps) = &mut runs.iter_mut().find(|(k, _)| k == "steps").unwrap().1 else {
        panic!("steps");
    };
    steps.swap(0, 1);
    assert!(cache_evidence::capture_action(&reordered).is_err());
}
#[test]
fn evidence_attempt_identity_and_safetensors_order_cannot_be_lost() {
    let documents = changed(
        "ci-linux-host-slice.yml",
        "artifact_name: sccache-ci-host-linux-${{ matrix.host.architecture }}-${{ github.run_attempt }}",
        "artifact_name: sccache-ci-host-linux-${{ matrix.host.architecture }}",
    );
    assert!(cache_evidence::check(&documents).is_err());
    let documents = changed(
        "ci-rust-tests-slice.yml",
        "id: safetensors_smoke_test",
        "id: unbound_build",
    );
    assert!(cache_evidence::check(&documents).is_err());
    let documents = changed(
        "ci-rust-tests-slice.yml",
        "      - name: Capture SafeTensors smoke cache evidence",
        "      - run: echo delaying-capture\n      - name: Capture SafeTensors smoke cache evidence",
    );
    assert!(cache_evidence::check(&documents).is_err());
    let source =
        fs::read_to_string(root().join(".github/workflows/swift-sdk-artifact.yml")).unwrap();
    let documents = changed(
        "swift-sdk-artifact.yml",
        "artifact_name: sccache-swift-sdk-${{ inputs.mode }}-${{ github.run_attempt }}",
        "artifact_name: sccache-swift-sdk-${{ matrix.target }}-${{ github.run_attempt }}",
    );
    assert!(source.contains("sccache-swift-sdk-${{ matrix.target }}"));
    assert!(cache_evidence::check(&documents).is_err());
}
#[test]
fn direct_sccache_users_cannot_bypass_immediate_policy_or_omit_authority_inputs() {
    let mut documents = workflows();
    // A parsed real caller tree supplies the installation; a inserted unrelated
    // step proves adjacency without depending on surrounding comments or names.
    let mut found = false;
    for document in documents.values_mut() {
        let Node::Map(fields) = document else {
            continue;
        };
        let Some((_, Node::Map(jobs))) = fields.iter_mut().find(|(key, _)| key == "jobs") else {
            continue;
        };
        for (_, job) in jobs {
            let Node::Map(fields) = job else {
                continue;
            };
            let Some((_, Node::Seq(steps))) = fields.iter_mut().find(|(key, _)| key == "steps")
            else {
                continue;
            };
            if let Some(index) = steps.iter().position(|s| {
                cache_evidence::text(s, "uses").starts_with("mozilla-actions/sccache-action@")
            }) {
                steps.insert(
                    index + 1,
                    Node::Map(vec![("run".into(), Node::Scalar("echo bypass".into()))]),
                );
                found = true;
                break;
            }
        }
        if found {
            break;
        }
    }
    assert!(found, "current direct caller fixture missing");
    assert!(cache_evidence::check(&documents).is_err());
    let documents = changed(
        "cache-warm-sccache.yml",
        "allow_native_github_cache: \"false\"",
        "unused_native_policy: \"false\"",
    );
    assert!(cache_evidence::check(&documents).is_err());
}
#[test]
fn sdk_dependency_cache_identity_and_hash_boundaries_reject_each_lost_dimension() {
    for (old, new) in [
        (
            "shared-key: swift-sdk-${{ matrix.target }}",
            "shared-key: generic",
        ),
        (
            "key: ${{ steps.native_toolchain.outputs.epoch }}",
            "key: fixed",
        ),
        ("add-job-id-key: \"false\"", "add-job-id-key: \"true\""),
        (
            "github.event_name == 'push' && github.ref == 'refs/heads/main'",
            "github.ref == 'refs/heads/main'",
        ),
    ] {
        assert!(
            cache_identity::check(&changed("swift-sdk-artifact.yml", old, new)).is_err(),
            "{old}"
        );
    }
    for path in [
        "Cargo.lock",
        ".github/cache-version.txt",
        ".cargo/config.toml",
        "scripts/cargo-linker",
        "scripts/cargo-linker-linux-*",
        "scripts/lib/lld.sh",
        "**/Cargo.toml",
        "scripts/ci-rust-sdk-smoke.sh",
        "scripts/ci-sdk-fixture.sh",
        "scripts/ci-prepare-native-runtime.sh",
        "scripts/package-sdk-console-assets.sh",
        "scripts/check-sdk-contract.sh",
        "scripts/verify-sdk-console-assets.sh",
        ".github/workflows/sdk-smoke.yml",
    ] {
        assert!(
            cache_identity::check(&changed(
                "sdk-smoke.yml",
                &format!("'{path}'"),
                "'unrelated-input'"
            ))
            .is_err(),
            "{path}"
        );
    }
    for dimension in [
        "env.SDK_RUST_TARGET",
        "env.SDK_RUST_IMAGE_DIGEST",
        "env.SDK_RUST_TOOLCHAIN_EPOCH",
        "env.SDK_RUST_PROFILE_LINKER",
    ] {
        assert!(
            cache_identity::check(&changed("sdk-smoke.yml", dimension, "env.UNRELATED")).is_err(),
            "{dimension}"
        );
    }
    for (old, new) in [
        ("shared-key: ci-sdk-smoke-rust", "shared-key: generic"),
        ("cache-bin: \"false\"", "cache-bin: \"true\""),
    ] {
        assert!(
            cache_identity::check(&changed("sdk-smoke.yml", old, new)).is_err(),
            "{old}"
        );
    }
}
#[test]
fn seed_identity_restore_save_and_runtime_opt_out_mutations_are_rejected() {
    for name in [
        "ci-quality-slice.yml",
        "ci-rust-tests-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-linux-runtime-slice.yml",
    ] {
        assert!(
            cache_identity::check(&changed(
                name,
                "cache_key: mesh-llm-sccache-seed-linux-x86_64-img-",
                "cache_key: unrelated-linux-x86_64-img-"
            ))
            .is_err(),
            "{name}"
        );
    }
    for (old, new) in [
        ("key: ${{ steps.seed.outputs.key }}", "key: stale-seed"),
        ("run: just ci-sccache-seed-build", "run: just unrelated"),
        ("'Justfile'", "'unrelated'"),
        ("'scripts/cargo-linker'", "'unrelated'"),
    ] {
        assert!(
            cache_identity::check(&changed("cache-warm-sccache.yml", old, new)).is_err(),
            "{old}"
        );
    }
    assert!(
        cache_identity::check(&changed(
            "ci-linux-runtime-slice.yml",
            "allow_trusted_seed: \"false\"",
            "allow_trusted_seed: \"true\""
        ))
        .is_err()
    );
    assert!(
        cache_identity::check(&changed(
            "ci-linux-runtime-slice.yml",
            "uses: ./.github/actions/restore-sccache-seed",
            "uses: ./.github/actions/unrelated"
        ))
        .is_err()
    );
}

#[test]
fn configure_default_denial_and_original_event_bindings_cannot_be_dropped() {
    let source =
        fs::read_to_string(root().join(".github/actions/configure-sccache-gha/action.yml"))
            .unwrap();
    for (old, new) in [
        ("default: \"false\"", "default: \"true\""),
        (
            "INPUT_ALLOW_NATIVE_GITHUB_CACHE: ${{ inputs.allow_native_github_cache }}",
            "INPUT_ALLOW_NATIVE_GITHUB_CACHE: true",
        ),
        (
            "INPUT_ALLOW_DEPOT_REMOTE_CACHE: ${{ inputs.allow_depot_remote_cache }}",
            "INPUT_ALLOW_DEPOT_REMOTE_CACHE: true",
        ),
        (
            "DISPATCH_ORIGINAL_EVENT_NAME: ${{ github.event.inputs.original_event_name || '' }}",
            "DISPATCH_ORIGINAL_EVENT_NAME: push",
        ),
    ] {
        assert!(source.contains(old));
        assert!(
            cache_evidence::configure_action(
                &workflow_yaml::parse(&source.replacen(old, new, 1)).unwrap()
            )
            .is_err(),
            "{old}"
        );
    }
}

#[test]
fn trusted_caller_exceptions_preserve_explicit_native_cache_decisions() {
    for (name, old, new) in [
        (
            "cache-warm-sccache.yml",
            "allow_native_github_cache: \"false\"",
            "allow_native_github_cache: \"true\"",
        ),
        (
            "depot-canary.yml",
            "allow_native_github_cache: \"false\"",
            "allow_native_github_cache: \"true\"",
        ),
        (
            "hf-download-smoke.yml",
            "allow_native_github_cache: \"true\"",
            "allow_native_github_cache: \"false\"",
        ),
        (
            "node-sdk-addon-artifact.yml",
            "allow_native_github_cache: \"true\"",
            "allow_native_github_cache: \"false\"",
        ),
        (
            "release.yml",
            "allow_native_github_cache: \"false\"",
            "allow_native_github_cache: \"true\"",
        ),
        (
            "release.yml",
            "allow_native_github_cache: ${{ startsWith(needs.metadata.outputs.runner_16, 'depot-') && 'false' || 'true' }}",
            "allow_native_github_cache: \"true\"",
        ),
    ] {
        assert!(
            cache_evidence::check(&changed(name, old, new)).is_err(),
            "{name}"
        );
    }
}

#[test]
fn current_seed_recipe_compiles_host_clippy_and_cli_test_dependencies() {
    let source = fs::read_to_string(root().join("just/ci.just")).unwrap();
    let body = source.split_once("\nci-sccache-seed-build:\n").unwrap().1;
    let commands: Vec<_> = body
        .lines()
        .take_while(|line| line.starts_with("    "))
        .map(str::trim)
        .collect();
    assert_eq!(
        commands,
        [
            "cargo clippy --locked -p mesh-llm --all-targets -- -D warnings",
            "cargo build --release --locked -p mesh-llm --bin mesh-llm --no-default-features --features web-ui,dynamic-native-runtime",
            "cargo test --locked -p mesh-llm-cli --no-run",
        ]
    );
}

fn action_steps_mut(action: &mut Node) -> &mut Vec<Node> {
    let Node::Map(fields) = action else {
        panic!("action map");
    };
    let Node::Map(runs) = &mut fields.iter_mut().find(|(key, _)| key == "runs").unwrap().1 else {
        panic!("runs map");
    };
    let Node::Seq(steps) = &mut runs.iter_mut().find(|(key, _)| key == "steps").unwrap().1 else {
        panic!("steps sequence");
    };
    steps
}
fn literal_log(run: &str) -> Node {
    Node::Map(vec![
        ("name".into(), Node::Scalar("diagnostic".into())),
        ("shell".into(), Node::Scalar("bash".into())),
        ("run".into(), Node::Scalar(run.into())),
    ])
}
#[test]
fn cache_actions_allow_literal_logging_around_required_nodes() {
    for name in ["capture-sccache-stats", "configure-sccache-gha"] {
        let mut document = action(name);
        let steps = action_steps_mut(&mut document);
        let original = std::mem::take(steps);
        steps.push(literal_log("echo before cache evidence"));
        for step in original {
            steps.push(step);
            steps.push(literal_log("echo after owned step"));
        }
        if name == "capture-sccache-stats" {
            cache_evidence::capture_action(&document).unwrap();
        } else {
            cache_evidence::configure_action(&document).unwrap();
        }
    }
}
#[test]
fn cache_actions_reject_duplicate_required_nodes_and_ancillary_cache_mutation() {
    for name in ["capture-sccache-stats", "configure-sccache-gha"] {
        for run in [
            "sccache --zero-stats",
            "echo harmless; sccache --stop-server",
            "echo $(sccache --zero-stats)",
            "echo harmless > cache.json",
        ] {
            let mut document = action(name);
            action_steps_mut(&mut document).push(literal_log(run));
            let result = if name == "capture-sccache-stats" {
                cache_evidence::capture_action(&document)
            } else {
                cache_evidence::configure_action(&document)
            };
            assert!(result.is_err(), "{name}: {run}");
        }
        let mut document = action(name);
        let steps = action_steps_mut(&mut document);
        steps.push(steps[0].clone());
        let result = if name == "capture-sccache-stats" {
            cache_evidence::capture_action(&document)
        } else {
            cache_evidence::configure_action(&document)
        };
        assert!(result.is_err(), "duplicate required {name}");
    }
}

#[test]
fn optional_direct_installation_is_outside_ci_policy_but_explicit_callers_keep_authority() {
    let mut documents = workflows();
    let optional = "jobs:\n  manual:\n    steps:\n      - uses: mozilla-actions/sccache-action@fixture\n      - run: echo manual model check\n";
    documents.insert(
        "manual-fixture.yml".into(),
        workflow_yaml::parse(optional).unwrap(),
    );
    cache_evidence::check(&documents).unwrap();
    // The same parsed installation still requires adjacent policy in owned CI.
    documents.insert(
        "ci-fixture.yml".into(),
        workflow_yaml::parse(optional).unwrap(),
    );
    assert!(cache_evidence::check(&documents).is_err());
    documents.remove("ci-fixture.yml");
    let explicit = format!(
        "{optional}      - uses: ./.github/actions/configure-sccache-gha\n        with:\n          allow_depot_remote_cache: false\n"
    );
    documents.insert(
        "manual-fixture.yml".into(),
        workflow_yaml::parse(&explicit).unwrap(),
    );
    assert!(cache_evidence::check(&documents).is_err());
}
