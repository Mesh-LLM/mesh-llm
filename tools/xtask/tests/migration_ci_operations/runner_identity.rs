//! `ci-ops runner-identity` parity for each subcommand's happy path, the
//! argparse surface, and catalog/workflow drift that must fail closed.

use super::support::{Stage, TestResult, check_case, raw_case};

const HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[test]
fn every_read_only_subcommand_matches_legacy_on_the_checkout() -> TestResult {
    let stage = Stage::checkout("happy")?;
    let cases: [(&str, &[&str]); 7] = [
        ("validate", &["validate"]),
        ("check", &["check"]),
        ("diagnose", &["diagnose"]),
        ("lookup_image", &["lookup", "rust-clippy"]),
        (
            "lookup_reference",
            &["lookup", "rust-clippy", "--field", "reference"],
        ),
        (
            "lookup_receipt_abbrev",
            &["lookup", "release-ui-artifact", "--f=receipt"],
        ),
        ("seed_key", &["seed-key", "--recipe-hash", HASH]),
    ];
    for (name, args) in cases {
        let outcome = check_case(name, &stage, args)?;
        assert_eq!(outcome.code, 0, "{name}: {}", outcome.stderr);
    }
    Ok(())
}

#[test]
fn argparse_usage_errors_exit_two_with_legacy_wording() -> TestResult {
    let stage = Stage::empty("usage")?;
    let cases: [(&str, &[&str]); 14] = [
        ("usage_missing_command", &[]),
        ("usage_invalid_choice", &["bogus"]),
        ("usage_lookup_missing_role", &["lookup"]),
        ("usage_bind_missing_all", &["bind"]),
        ("usage_bind_missing_some", &["bind", "--image-id", "x"]),
        ("usage_seed_key_missing", &["seed-key"]),
        (
            "usage_seed_key_option_value",
            &["seed-key", "--recipe-hash", "-x"],
        ),
        ("usage_field_choice", &["lookup", "x", "--field", "nope"]),
        ("usage_field_missing_value", &["lookup", "--field"]),
        ("usage_root_missing_value", &["--root"]),
        ("usage_unrecognized", &["validate", "extra", "--x=1"]),
        ("usage_subcommand_root", &["validate", "--root", "/tmp"]),
        ("usage_double_dash_extra", &["validate", "--", "x"]),
        ("usage_catalog_missing_value", &["--root", "x", "--catalog"]),
    ];
    for (name, args) in cases {
        let outcome = raw_case(name, &stage, args)?;
        assert_eq!(outcome.code, 2, "{name}");
        assert!(outcome.stdout.is_empty(), "{name}");
    }
    Ok(())
}

#[test]
fn help_is_printed_to_stdout_with_status_zero() -> TestResult {
    let stage = Stage::empty("help")?;
    for (name, args) in [
        ("help_top", &["-h"][..]),
        ("help_lookup", &["lookup", "--help"]),
        ("help_bind", &["bind", "-h"]),
        ("help_seed_key", &["seed-key", "-h"]),
        ("help_check_ignores_rest", &["check", "-h", "extra"]),
    ] {
        let outcome = raw_case(name, &stage, args)?;
        assert_eq!(outcome.code, 0, "{name}");
    }
    Ok(())
}

#[test]
fn command_failures_exit_one_without_stdout() -> TestResult {
    let stage = Stage::checkout("failures")?;
    for (name, args) in [
        ("lookup_unknown_role", &["lookup", "missing-role"][..]),
        ("lookup_after_double_dash", &["lookup", "--", "--x"]),
        (
            "seed_key_rejects_non_sha",
            &["seed-key", "--recipe-hash", "../../bad"],
        ),
        (
            "seed_key_rejects_negative_number",
            &["seed-key", "--recipe-hash", "-5"],
        ),
    ] {
        let outcome = check_case(name, &stage, args)?;
        assert_eq!((outcome.code, outcome.stdout.as_str()), (1, ""), "{name}");
    }
    Ok(())
}

#[test]
fn malformed_catalog_input_fails_with_decoder_wording() -> TestResult {
    let deep_bounded = format!("{}0{}", "[".repeat(9998), "]".repeat(9998));
    let deep_limit = format!("{}0{}", "[".repeat(9999), "]".repeat(9999));
    let cases: [(&str, &[u8]); 8] = [
        (
            "malformed_duplicate_key",
            br#"{"schema_version": 1, "schema_version": 1}"#,
        ),
        ("malformed_trailing_comma", b"{\"a\": 1,\n}"),
        ("malformed_not_object", b"[1]"),
        ("malformed_float", b"{\"schema_version\": 1.0}"),
        ("malformed_invalid_utf8", b"{\"a\": \"\xff\"}"),
        ("malformed_deep_nesting", &[b'['; 2000]),
        ("malformed_nesting_within_scanner", deep_bounded.as_bytes()),
        ("malformed_nesting_beyond_scanner", deep_limit.as_bytes()),
    ];
    for (name, bytes) in cases {
        let stage = Stage::checkout(name)?;
        stage.write("ci/runner-images.json", bytes)?;
        let outcome = check_case(name, &stage, &["validate"])?;
        assert_eq!((outcome.code, outcome.stdout.as_str()), (1, ""), "{name}");
    }
    let stage = Stage::empty("missing-catalog")?;
    let root = stage.root_arg();
    let outcome = raw_case(
        "missing_catalog",
        &stage,
        &["--root", &root, "--catalog", "absent.json", "validate"],
    )?;
    assert_eq!(outcome.code, 1);
    Ok(())
}

type Mutation = fn(&Stage) -> TestResult;

fn run_drift(cases: &[(&str, Mutation)], command: &str) -> TestResult {
    for (name, mutate) in cases {
        let stage = Stage::checkout(name)?;
        mutate(&stage)?;
        let outcome = check_case(name, &stage, &[command])?;
        assert_eq!((outcome.code, outcome.stdout.as_str()), (1, ""), "{name}");
    }
    Ok(())
}

#[test]
fn catalog_contract_violations_fail_closed() -> TestResult {
    let cases: [(&str, Mutation); 8] = [
        ("catalog_mutable_reference", |s| {
            s.edit_catalog(|c| {
                c["images"]["public-cpu"]["reference"] =
                    "ghcr.io/mesh-llm/mesh-llm-cuda-runner:latest".into()
            })
        }),
        ("catalog_epoch_mismatch", |s| {
            s.edit_catalog(|c| {
                c["images"]["public-cpu"]["native_toolchain_epoch"] = "different-epoch".into()
            })
        }),
        ("catalog_unknown_image", |s| {
            s.edit_catalog(|c| c["consumer_roles"]["ui-quality"]["image_id"] = "missing".into())
        }),
        ("catalog_duplicate_binding", |s| {
            s.edit_catalog(|c| {
                c["consumer_roles"]["extra-role"] = c["consumer_roles"]["ui-quality"].clone()
            })
        }),
        ("catalog_workflow_escape", |s| {
            s.edit_catalog(|c| {
                c["consumer_roles"]["ui-quality"]["bindings"][0]["workflow"] =
                    "../private.yml".into()
            })
        }),
        ("catalog_seed_coverage_claimed", |s| {
            s.edit_catalog(|c| c["compiler_seed"]["workload_coverage"] = "qualified".into())
        }),
        ("catalog_release_pair_incomplete", |s| {
            s.edit_catalog(|c| {
                c["consumer_roles"]
                    .as_object_mut()
                    .map(|roles| roles.remove("release-ui-artifact"));
            })
        }),
        ("catalog_type_error_is_reported", |s| {
            s.edit_catalog(|c| c["images"]["public-cpu"]["receipt"] = "text".into())
        }),
    ];
    run_drift(&cases, "validate")
}

#[test]
fn workflow_and_planner_drift_fail_check() -> TestResult {
    let cases: [(&str, Mutation); 12] = [
        ("drift_ordinary_image", |s| {
            s.replace(
                ".github/workflows/ci-quality-slice.yml",
                &s.image("public-cpu")?,
                &s.image("public-web")?,
            )
        }),
        ("drift_matrix_epoch", |s| {
            s.replace(
                ".github/workflows/release.yml",
                "pinned_epoch: ${{ matrix.toolchain_epoch }}",
                "pinned_epoch: ignored",
            )
        }),
        ("drift_unregistered_consumer", |s| {
            let image = s.image("public-web")?;
            s.append(
                ".github/workflows/ci-web-slice.yml",
                &format!("\n  extra_ui:\n    container:\n      image: {image}\n"),
            )
        }),
        ("drift_flow_container", |s| {
            let image = s.image("public-web")?;
            s.append(
                ".github/workflows/ci-web-slice.yml",
                &format!("\n  extra_ui:\n    container: {{image: {image}}}\n"),
            )
        }),
        ("drift_yaml_extension", |s| {
            let image = s.image("public-web")?;
            s.write(
                ".github/workflows/extra-ui.yaml",
                format!("name: x\njobs:\n  extra_ui:\n    container:\n      image: {image}\n")
                    .as_bytes(),
            )
        }),
        ("drift_duplicate_image_field", |s| {
            let line = format!("      image: {}", s.image("public-web")?);
            s.replace(
                ".github/workflows/ci-web-slice.yml",
                &line,
                &format!("{line}\n{line}"),
            )
        }),
        ("drift_planner_image", |s| {
            s.replace(
                "ci/slices.yml",
                &s.image("public-rocm-ci")?,
                &s.image("public-rocm-release")?,
            )
        }),
        ("drift_runtime_seed_reenabled", |s| {
            s.replace(
                ".github/workflows/ci-linux-runtime-slice.yml",
                "allow_trusted_seed: \"false\"",
                "allow_trusted_seed: \"true\"",
            )
        }),
        ("drift_restore_census", |s| {
            s.replace(
                ".github/workflows/ci-linux-host-slice.yml",
                "uses: ./.github/actions/restore-sccache-seed",
                "uses: ./.github/actions/another-action",
            )
        }),
        ("drift_publisher_save_key", |s| {
            let path = ".github/workflows/cache-warm-sccache.yml";
            let text = s.read(path)?;
            let (head, tail) = text
                .rsplit_once("key: ${{ steps.seed.outputs.key }}")
                .ok_or("no publisher key")?;
            s.write(
                path,
                format!("{head}key: unrelated-cache-key{tail}").as_bytes(),
            )
        }),
        ("drift_sdk_action", |s| {
            let action = s.catalog()?["sdk_rust"]["rust_action_ref"]
                .as_str()
                .ok_or("action")?
                .to_owned();
            s.replace(
                ".github/workflows/sdk-smoke.yml",
                &action,
                &format!("dtolnay/rust-toolchain@{}", "a".repeat(40)),
            )
        }),
        ("drift_seed_guard_outside_job", |s| {
            s.append(
                ".github/workflows/ci-linux-runtime-slice.yml",
                "\nenv:\n          allow_trusted_seed: ghcr.io/mesh-llm/mesh-llm-cuda-runner:x\n",
            )
        }),
    ];
    run_drift(&cases, "check")
}

#[test]
fn diagnose_refuses_reenabled_runtime_seed_restore() -> TestResult {
    let cases: [(&str, Mutation); 1] = [("diagnose_seed_reenabled", |s| {
        s.replace(
            ".github/workflows/ci-linux-runtime-slice.yml",
            "allow_trusted_seed: \"false\"",
            "allow_trusted_seed: ${{ needs.runner_policy.outputs.allow_trusted_sccache_seed }}",
        )
    })];
    run_drift(&cases, "diagnose")
}

#[test]
fn runner_check_and_diagnose_use_rust_planner_without_python_input() -> TestResult {
    let stage = Stage::checkout("rust-planner-only")?;
    for command in ["check", "diagnose"] {
        let outcome = super::support::xtask_in(
            stage.path(),
            &[
                "ci-ops".into(),
                "runner-identity".into(),
                "--root".into(),
                stage.root_arg(),
                command.into(),
            ],
        )?;
        assert_eq!(outcome.code, 0, "{}", outcome.stderr);
    }
    Ok(())
}

#[test]
fn additional_platform_seed_and_conditional_drift_is_rejected() -> TestResult {
    let cases: &[(&str, Mutation)] = &[
        ("rocm-release", |s| {
            s.replace(
                ".github/workflows/release.yml",
                &s.image("public-rocm-release")?,
                &s.image("public-rocm-ci")?,
            )
        }),
        ("runtime-image", |s| {
            s.replace(
                ".github/workflows/ci-linux-runtime-slice.yml",
                "image: ${{ matrix.runtime.container_image }}",
                "image: ignored",
            )
        }),
        ("runtime-epoch", |s| {
            s.replace(
                "ci/slices.yml",
                s.catalog()?["images"]["public-cpu"]["native_toolchain_epoch"]
                    .as_str()
                    .ok_or("epoch")?,
                "changed-native-epoch",
            )
        }),
        ("sdk-optout", |s| {
            s.replace(
                ".github/workflows/sdk-smoke.yml",
                "inputs.sdk_kind != 'swift' &&",
                "inputs.sdk_kind == 'swift' &&",
            )
        }),
        ("sdk-cache", |s| {
            s.replace(
                ".github/workflows/sdk-smoke.yml",
                "env.SDK_RUST_TOOLCHAIN_EPOCH,",
                "'ignored-epoch',",
            )
        }),
        ("publisher-recipe", |s| {
            s.replace(
                ".github/workflows/cache-warm-sccache.yml",
                "'Justfile', 'just/**'",
                "'Justfile', 'just/ci.just'",
            )
        }),
        ("missing-runtime-guard", |s| {
            s.replace(
                ".github/workflows/ci-linux-runtime-slice.yml",
                "          allow_trusted_seed: \"false\"\n",
                "",
            )
        }),
        ("release-selector", |s| {
            s.replace(
                ".github/workflows/ci-ui-artifact-slice.yml",
                "inputs.release_tag != '' &&",
                "inputs.source_sha != '' &&",
            )
        }),
        ("short-container", |s| {
            s.append(
                ".github/workflows/ci-web-slice.yml",
                &format!("\n  extra:\n    container: {}\n", s.image("public-web")?),
            )
        }),
        ("unregistered-mutable", |s| {
            s.write(".github/workflows/unregistered.yml", b"jobs:\n  extra:\n    container:\n      image: ghcr.io/mesh-llm/mesh-llm-cuda-runner:latest\n")
        }),
        ("canary-override", |s| {
            s.replace(
                ".github/workflows/depot-canary.yml",
                "ci-ops runtime-seed preflight runtime-seed-evidence",
                "ci-ops runtime-seed preflight runtime-seed-evidence; echo key=other",
            )
        }),
        ("canary-missing", |s| {
            s.replace(
                ".github/workflows/depot-canary.yml",
                "id: seed_key",
                "id: other_key",
            )
        }),
        ("canary-unused", |s| {
            s.replace(
                ".github/workflows/depot-canary.yml",
                "CANARY_KEY:",
                "UNUSED_KEY:",
            )
        }),
        ("canary-duplicate", |s| {
            s.replace(".github/workflows/depot-canary.yml", "  runtime_seed_summary:", "      - name: duplicate key\n        id: seed_key\n        run: true\n\n  runtime_seed_summary:")
        }),
    ];
    for (name, mutate) in cases {
        let stage = Stage::checkout(name)?;
        mutate(&stage)?;
        let outcome = super::support::xtask_in(
            stage.path(),
            &[
                "ci-ops".into(),
                "runner-identity".into(),
                "--root".into(),
                stage.root_arg(),
                "check".into(),
            ],
        )?;
        assert_ne!(outcome.code, 0, "{name}");
        assert!(outcome.stdout.is_empty(), "{name}");
    }
    Ok(())
}

#[test]
fn receipt_role_binding_and_registered_yaml_contracts_are_checked() -> TestResult {
    for mutation in [
        "provenance",
        "receipt",
        "image",
        "publisher",
        "sdk",
        "artifact",
    ] {
        let stage = Stage::checkout(mutation)?;
        stage.edit_catalog(|catalog| match mutation {
            "provenance" => catalog["images"]["public-cpu"]["provenance"] = serde_json::Value::Null,
            "receipt" => {
                catalog["images"]["public-cpu"]["receipt"]["index_candidate_key"] =
                    "invented".into()
            }
            "image" => {
                catalog["images"]["public-ui"]["receipt"]["index_candidate_key"] =
                    "candidate-index-public-browser".into()
            }
            "publisher" | "sdk" => {
                let role = catalog[if mutation == "publisher" {
                    "compiler_seed"
                } else {
                    "sdk_rust"
                }][if mutation == "publisher" {
                    "publisher_role"
                } else {
                    "role"
                }]
                .as_str()
                .expect("role")
                .to_owned();
                let mut extra = catalog["consumer_roles"][&role]["bindings"][0].clone();
                extra["job"] = "additional".into();
                catalog["consumer_roles"][&role]["bindings"]
                    .as_array_mut()
                    .expect("bindings")
                    .push(extra);
            }
            "artifact" => {
                catalog["consumer_roles"]["ui-artifact"]["bindings"][0]["job"] = "ui_quality".into()
            }
            _ => unreachable!(),
        })?;
        let outcome = super::support::xtask_in(
            stage.path(),
            &[
                "ci-ops".into(),
                "runner-identity".into(),
                "--root".into(),
                stage.root_arg(),
                "validate".into(),
            ],
        )?;
        assert_ne!(outcome.code, 0, "{mutation}");
    }
    let stage = Stage::checkout("registered-yaml")?;
    std::fs::rename(
        stage.path().join(".github/workflows/ci-web-slice.yml"),
        stage.path().join(".github/workflows/ci-web-slice.yaml"),
    )?;
    stage.edit_catalog(|catalog| {
        for role in catalog["consumer_roles"]
            .as_object_mut()
            .expect("roles")
            .values_mut()
        {
            for binding in role["bindings"].as_array_mut().expect("bindings") {
                if binding["workflow"] == "ci-web-slice.yml" {
                    binding["workflow"] = "ci-web-slice.yaml".into();
                }
            }
        }
    })?;
    let args = [
        "ci-ops".into(),
        "runner-identity".into(),
        "--root".into(),
        stage.root_arg(),
        "check".into(),
    ];
    assert_eq!(super::support::xtask_in(stage.path(), &args)?.code, 0);
    stage.replace(
        ".github/workflows/ci-web-slice.yaml",
        &stage.image("public-web")?,
        &stage.image("public-cpu")?,
    )?;
    assert_ne!(super::support::xtask_in(stage.path(), &args)?.code, 0);
    Ok(())
}
