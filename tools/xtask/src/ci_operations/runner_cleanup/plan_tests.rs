use super::{Options, Profile, boundary, fixtures::Fixture};

#[test]
fn preflight_evidence_is_retained_until_uploaded_and_other_outputs_survive() {
    let fixture = Fixture::new();
    let evidence = fixture.temporary.join("llama-canary-preflight");
    let sentinel = fixture.temporary.join("independent-evidence");
    fixture.seed(&evidence);
    fixture.seed(&sentinel);
    let retained = fixture.plan(Profile::CanaryPreflight, false);
    assert!(retained.targets.is_empty());
    fixture.delete(&retained);
    assert!(evidence.join("payload").exists());
    let uploaded = fixture.plan(Profile::CanaryPreflight, true);
    assert_eq!(uploaded.targets.len(), 1);
    assert_eq!(uploaded.targets[0].base, fixture.temporary);
    assert_eq!(uploaded.targets[0].path, evidence);
    assert!(uploaded.replay.is_none());
    fixture.delete(&uploaded);
    assert!(!evidence.exists());
    assert!(sentinel.join("payload").exists());
    assert!(Options::parse("canary-preflight", "true", None).is_ok());
}

#[test]
fn roster_order_when_each_profile_is_selected() {
    let fixture = Fixture::new();
    let cases = [
        (
            Profile::Build,
            vec![
                (false, "target/debug"),
                (false, ".deps/llama.cpp"),
                (false, ".deps/llama-123-2-repair-1"),
                (false, ".deps/llama-123-2-repair-1-workloads"),
                (false, ".deps/llama-123-2-repair-1-verification-123-2"),
                (
                    false,
                    ".deps/llama-123-2-repair-1-verification-123-2-workloads",
                ),
                (true, "canary-previous-repair-1"),
                (true, "canary-feedback-repair-1"),
                (true, "canary-export-123-2-repair-1"),
                (false, ".deps/llama-canary-state-123-2-repair-1"),
            ],
        ),
        (
            Profile::Family,
            vec![
                (false, "target/debug"),
                (false, ".deps/llama.cpp"),
                (false, ".deps/canary-input-123-2-repair-1-3"),
                (false, ".deps/canary-workload-oracles"),
                (false, "ci/canary-python/.venv"),
                (false, "target/canary-evidence-123-2/repair-1-3"),
            ],
        ),
        (
            Profile::Replay,
            vec![
                (false, "ci/agentic-replay-nightly/.venv"),
                (true, "agentic-replay-history"),
                (true, "agentic-replay-worktrees"),
                (true, "agentic-replay-artifacts"),
            ],
        ),
        (
            Profile::CudaRelease,
            vec![
                (false, "target"),
                (false, ".deps/llama.cpp"),
                (false, ".deps/llama-build"),
                (false, "dist/native-runtimes"),
            ],
        ),
        (
            Profile::Smoke,
            vec![
                (false, "ci-artifacts/linux"),
                (false, "target/release/mesh-llm"),
                (false, "target/release/native-runtimes"),
            ],
        ),
        (Profile::RunnerContract, vec![(false, "target")]),
    ];
    for (profile, names) in cases {
        let expected: Vec<_> = names
            .into_iter()
            .map(|(temporary, name)| {
                if temporary {
                    fixture.temporary.join(name)
                } else {
                    fixture.workspace.join(name)
                }
            })
            .collect();
        let actual = fixture.plan(profile, true);
        assert_eq!(
            actual
                .targets
                .iter()
                .map(|target| target.path.clone())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn recovery_outputs_when_uploads_fail() {
    let fixture = Fixture::new();
    for (profile, retained) in [
        (Profile::Build, 2),
        (Profile::Family, 1),
        (Profile::Replay, 1),
        (Profile::CudaRelease, 1),
    ] {
        let full = fixture.plan(profile, true);
        let safe = fixture.plan(profile, false);
        assert_eq!(full.targets.len() - safe.targets.len(), retained);
        for target in &full.targets {
            fixture.seed(&target.path);
        }
        let mut files = safe;
        files.replay = None;
        fixture.delete(&files);
        for target in full
            .targets
            .iter()
            .filter(|target| !files.targets.contains(target))
        {
            assert_eq!(
                std::fs::read(target.path.join("payload")).unwrap(),
                b"sentinel"
            );
        }
    }
}

#[test]
fn package_and_evidence_are_independent_when_one_upload_fails() {
    let fixture = Fixture::new();
    for (evidence, package, retained) in [
        (false, true, ".deps/llama-canary-state-123-2-repair-1"),
        (true, false, "canary-export-123-2-repair-1"),
    ] {
        let plan = boundary::from_environment(
            &Options {
                profile: Profile::Build,
                evidence_uploaded: evidence,
                package_uploaded: package,
            },
            &fixture.env,
        )
        .unwrap();
        assert_eq!(plan.targets.len(), 9);
        assert!(
            !plan
                .targets
                .iter()
                .any(|target| target.path.ends_with(retained))
        );
    }
}

#[test]
fn selected_source_when_family_uses_a_separate_checkout() {
    let mut fixture = Fixture::new();
    let selected = fixture.workspace.join("canary-source");
    fixture.env.insert(
        "CANARY_SOURCE_ROOT".into(),
        selected.clone().into_os_string(),
    );
    let plan = fixture.plan(Profile::Family, true);
    assert_eq!(plan.targets[0].path, selected.join("target/debug"));
    assert_eq!(
        plan.targets[3].path,
        selected.join(".deps/canary-workload-oracles")
    );
    assert_eq!(
        plan.targets[4].path,
        fixture.workspace.join("ci/canary-python/.venv")
    );
}

#[test]
fn identities_rejected_when_ascii_or_pass_contract_is_violated() {
    let fixture = Fixture::new();
    for (key, value) in [
        ("GITHUB_RUN_ID", ""),
        ("GITHUB_RUN_ID", "١"),
        ("GITHUB_RUN_ATTEMPT", "../2"),
        ("CANARY_PASS_ID", "repair-4"),
        ("CANARY_PASS_ID", "verify-0"),
        ("CANARY_SHARD_INDEX", "../3"),
        ("CANARY_SOURCE_ROOT", "/outside"),
    ] {
        let mut env = fixture.env.clone();
        env.insert(key.into(), value.into());
        assert!(
            boundary::from_environment(&Options::parse("family", "true", None).unwrap(), &env)
                .is_err()
        );
    }
    for pass in [
        "repair-1", "repair-2", "repair-3", "verify-1", "verify-2", "verify-3",
    ] {
        let mut env = fixture.env.clone();
        env.insert("CANARY_PASS_ID".into(), pass.into());
        assert!(
            boundary::from_environment(&Options::parse("family", "true", None).unwrap(), &env)
                .is_ok()
        );
    }
}

#[test]
fn required_environment_and_exact_booleans_when_boundary_is_parsed() {
    let fixture = Fixture::new();
    for value in ["True", "1", "", " true", "false\n"] {
        assert!(Options::parse("build", value, None).is_err());
        assert!(Options::parse("build", "true", Some(value)).is_err());
    }
    for key in [
        "GITHUB_WORKSPACE",
        "RUNNER_TEMP",
        "CANARY_SOURCE_ROOT",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "CANARY_PASS_ID",
        "CANARY_SHARD_INDEX",
    ] {
        let mut env = fixture.env.clone();
        env.remove(std::ffi::OsStr::new(key));
        assert!(
            boundary::from_environment(&Options::parse("family", "false", None).unwrap(), &env)
                .is_err()
        );
    }
    assert!(Options::parse("unknown", "true", None).is_err());
}
