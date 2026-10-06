use super::fixture::Fixture;
use std::fs;

#[test]
fn actual_local_certification_projects_full_plan_verify_cache_and_skip_build_battery() {
    for mode in ["repair", "verify"] {
        for failure in ["none", "inspection", "plan", "battery"] {
            let fixture = Fixture::new();
            fixture.tool(
                "scripts/skippy-workload-oracles-build.sh",
                "workload",
                "printf '%s\\n' 'INERT_ORACLE=exact value' 'INERT_SECOND=present'",
            );
            fixture.tool("scripts/inert-inspection.sh", "inspection", "");
            fixture.tool("scripts/inert-plan-step.sh", "plan", "");
            fixture.tool("scripts/skippy-family-battery.sh", "battery", r#"printf '%s\0' "$FAMILY_BATTERY_RUN_ID" "$INERT_ORACLE" "$INERT_SECOND" > "$INERT_ROOT/battery.environment""#);
            let report = fixture.run(
                mode,
                &[
                    "run_for",
                    "remaining_verification_seconds",
                    "run_verification_logged",
                    "repair_family_plan",
                    "run_certification",
                ],
                r#"
HF_CACHE="$ROOT/immutable-cache"
repair_source_inspection() { scripts/inert-inspection.sh "$@"; }
verification_source_inspection() { scripts/inert-inspection.sh "$@"; }
repair_family_plan_step() { scripts/inert-plan-step.sh "$@"; }
"#,
                "run_certification",
                &[("INERT_FAIL", failure.into())],
            );
            let expected = match failure {
                "none" => 0,
                "battery" => 17,
                _ => 1,
            };
            assert_eq!(
                report.process.status.unwrap().code(),
                Some(expected),
                "mode={mode}, failure={failure}: {}",
                fixture.diagnostic(&report)
            );
            assert_eq!(
                fixture.args("workload"),
                [
                    "--print-env".to_owned(),
                    format!("{}-workloads", fixture.root.join("native").display())
                ]
            );
            let inspected = fixture.args("inspection");
            assert_eq!(
                inspected[0],
                if mode == "repair" {
                    "local-parity-inventory"
                } else {
                    "verification-parity-inventory"
                }
            );
            if failure == "inspection" {
                assert_eq!(fixture.order(), ["workload", "inspection"]);
            } else {
                let first = fixture.args("plan-1");
                assert_eq!(
                    first[0],
                    fixture.root.join("certify.log").display().to_string()
                );
                assert_eq!(first[1], env!("CARGO_BIN_EXE_xtask"));
                assert_eq!(
                    &first[2..5],
                    ["--repo-root", fixture.root.to_str().unwrap(), "ci"]
                );
                assert_eq!(first[5], "family-plan");
                assert_eq!(
                    &first[6..],
                    [
                        "--manifest".to_owned(),
                        fixture
                            .root
                            .join("ci/llama-canary/family-certified.json")
                            .display()
                            .to_string(),
                        "--shard-count".into(),
                        "1".into(),
                        "--output".into(),
                        fixture.root.join("plan.json").display().to_string()
                    ]
                );
                if failure == "plan" {
                    assert_eq!(fixture.order(), ["workload", "inspection", "plan"]);
                } else {
                    let second = fixture.args("plan-2");
                    assert_eq!(
                        &second[6..],
                        [
                            "--manifest".to_owned(),
                            fixture
                                .root
                                .join("ci/llama-canary/family-certified.json")
                                .display()
                                .to_string(),
                            "--verify-plan".into(),
                            fixture.root.join("plan.json").display().to_string()
                        ]
                    );
                    let third = fixture.args("plan-3");
                    assert_eq!(
                        &third[2..],
                        [
                            "automation".to_owned(),
                            "family-battery-policy".into(),
                            "--cache".into(),
                            fixture.root.display().to_string(),
                            fixture
                                .root
                                .join("ci/llama-canary/family-certified.json")
                                .display()
                                .to_string(),
                            fixture.root.join("plan.json").display().to_string(),
                            fixture.root.join("immutable-cache").display().to_string()
                        ]
                    );
                    assert_eq!(
                        fixture.order(),
                        ["workload", "inspection", "plan", "plan", "plan", "battery"]
                    );
                    assert_eq!(
                        fixture.args("battery"),
                        [
                            "--skip-build".to_owned(),
                            "--plan".into(),
                            fixture.root.join("plan.json").display().to_string()
                        ]
                    );
                    if failure == "none" {
                        assert_eq!(
                            fixture.values("battery.environment"),
                            ["fixture", "exact value", "present"]
                        );
                    }
                }
            }
            fixture.finish();
        }
    }
    let fixture = Fixture::new();
    let report = fixture.run(
        "verify-build",
        &["run_certification"],
        "",
        "run_certification",
        &[],
    );
    assert_eq!(report.process.status.unwrap().code(), Some(2));
    assert!(!fixture.root.join("order").exists());
    fixture.finish();
}

#[test]
fn actual_verifier_materialization_invalidates_only_fixture_native_and_smoke_products() {
    let fixture = Fixture::new();
    fixture.tool("tools/git", "git", r#"
if [[ "$*" == *'worktree add --detach'* ]]; then
  mkdir -p "$INERT_ROOT/verification/third_party/llama.cpp"
  printf '%s\n' "$INERT_VERIFIED_SHA" > "$INERT_ROOT/verification/third_party/llama.cpp/upstream.txt"
fi
"#);
    let relative = [
        "native-verification-fixture",
        "verification/.deps/llama.cpp",
        "verification/target/family-battery/fixture-verification",
        "verification/target/skippy-stage-rewriter-check",
        "verification/target/skippy-system-one-smoke",
    ];
    for entry in relative {
        fs::create_dir_all(fixture.root.join(entry)).unwrap();
        fs::write(fixture.root.join(entry).join("stale"), b"prior product").unwrap();
    }
    let sentinel = fixture.root.join("native/src/libllama.a");
    let report = fixture.run("verify", &["materialize_verification_tree", "verify_repair_pin"], r#"
VERIFY_ROOT="$ROOT/verification"
RUN_KEY=fixture
CERTIFIED_SHA=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
cleanup_verification_worktree() { :; }
"#, r#"
materialize_verification_tree
printf '%s\0' "$ROOT" "$PIN_FILE" "$FAMILY_BATTERY_RUN_ID" "$PLAN_PATH" "$LLAMA_STAGE_BUILD_DIR" "$LLAMA_BUILD_DIR" "$SYSTEMONE_SMOKE_DIR" "$PWD" > "$INERT_ROOT/verification.environment"
"#, &[("INERT_VERIFIED_SHA", "a".repeat(40))]);
    assert_eq!(report.process.status.unwrap().code(), Some(0));
    assert_eq!(fixture.order(), ["git", "git"]);
    assert_eq!(
        fixture.args("git-1"),
        [
            "-c".to_owned(),
            "core.hooksPath=/dev/null".into(),
            "-C".into(),
            fixture.root.display().to_string(),
            "worktree".into(),
            "prune".into()
        ]
    );
    let verify = fixture.root.join("verification");
    assert_eq!(
        fixture.args("git-2"),
        [
            "-c".to_owned(),
            "core.hooksPath=/dev/null".into(),
            "-C".into(),
            fixture.root.display().to_string(),
            "worktree".into(),
            "add".into(),
            "--detach".into(),
            verify.display().to_string(),
            "b".repeat(40)
        ]
    );
    for entry in relative {
        assert!(!fixture.root.join(entry).join("stale").exists());
    }
    assert_eq!(fs::read(sentinel).unwrap(), b"inert archive bytes");
    let stage = fixture
        .root
        .join("native-verification-fixture")
        .display()
        .to_string();
    assert_eq!(
        fixture.values("verification.environment"),
        [
            verify.display().to_string(),
            verify
                .join("third_party/llama.cpp/upstream.txt")
                .display()
                .to_string(),
            "fixture-verification".into(),
            verify
                .join("target/family-battery/fixture-verification/policy-plan.json")
                .display()
                .to_string(),
            stage.clone(),
            stage,
            verify
                .join("target/skippy-system-one-smoke")
                .display()
                .to_string(),
            verify.display().to_string()
        ]
    );
    fixture.finish();
}
