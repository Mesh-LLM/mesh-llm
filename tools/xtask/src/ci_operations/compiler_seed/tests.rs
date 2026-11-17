use super::super::compiler_seed_process;
use super::*;
use serde_json::json;

fn argv(pair: &str) -> Vec<String> {
    ["run", "--pair", pair, "--output", "evidence"]
        .map(String::from)
        .to_vec()
}

#[test]
fn compiler_seed_bounds_and_order_are_closed() {
    for pair in ["1", "2", "3"] {
        assert_eq!(parse(&argv(pair)).unwrap().0.to_string(), pair);
    }
    for pair in ["0", "4", "-1", "256", "anything"] {
        assert!(parse(&argv(pair)).is_err());
    }
    assert!(parse(&["run".into()]).is_err());
    assert_eq!(order(1), ["current", "extended"]);
    assert_eq!(order(2), ["extended", "current"]);
    assert_eq!(order(3), order(1));
}

#[test]
fn compiler_seed_environment_is_disk_only_and_targets_repeat_within_arm() {
    let temporary = tempfile::tempdir().unwrap();
    let first = temporary.path().join("current");
    let second = temporary.path().join("extended");
    let env = environment(&first, &first.join("cache"), &first.join("target"));
    let recorded = super::super::compiler_seed_process::recorded_environment(&env).unwrap();
    assert_eq!(
        serde_json::to_value(recorded).unwrap()["SCCACHE_CACHE_SIZE"],
        "2G"
    );
    let other = environment(&second, &second.join("cache"), &second.join("target"));
    assert_eq!(env[std::ffi::OsStr::new("SCCACHE_GHA_ENABLED")], "false");
    assert_eq!(
        env[std::ffi::OsStr::new("SCCACHE_MULTILEVEL_CHAIN")],
        "disk"
    );
    assert_eq!(env[std::ffi::OsStr::new("SCCACHE_CACHE_SIZE")], "2G");
    for key in [
        "SCCACHE_DIR",
        "SCCACHE_SERVER_UDS",
        "CARGO_TARGET_DIR",
        "CARGO_BUILD_BUILD_DIR",
    ] {
        assert_ne!(
            env[std::ffi::OsStr::new(key)],
            other[std::ffi::OsStr::new(key)]
        );
    }
    assert!(!env.keys().any(|name| {
        let name = name.to_string_lossy();
        name.contains("TOKEN")
            || name.starts_with("AWS_")
            || name.starts_with("ACTIONS_")
            || name.starts_with("SCCACHE_WEBDAV")
    }));
}

#[test]
fn compiler_seed_target_reset_preserves_cache_and_unrelated_outputs() {
    let temporary = tempfile::tempdir().unwrap();
    let work = temporary.path().canonicalize().unwrap().join("owned");
    fs::create_dir(&work).unwrap();
    fs::create_dir(work.join("target")).unwrap();
    fs::write(work.join("target/stale"), "cargo fingerprint").unwrap();
    fs::create_dir(work.join("cache")).unwrap();
    fs::write(work.join("cache/object"), "seed").unwrap();
    fs::write(temporary.path().join("user-file"), "untouched").unwrap();
    reset_target(&work).unwrap();
    assert!(fs::read_dir(work.join("target")).unwrap().next().is_none());
    assert_eq!(
        fs::read_to_string(work.join("cache/object")).unwrap(),
        "seed"
    );
    assert_eq!(
        fs::read_to_string(temporary.path().join("user-file")).unwrap(),
        "untouched"
    );
    #[cfg(unix)]
    {
        fs::remove_dir(work.join("target")).unwrap();
        std::os::unix::fs::symlink(temporary.path(), work.join("target")).unwrap();
        assert!(reset_target(&work).is_err());
        assert!(temporary.path().join("user-file").exists());
    }
}

#[test]
fn compiler_seed_object_evidence_identifies_eviction_bytes() {
    let temporary = tempfile::tempdir().unwrap();
    fs::write(temporary.path().join("old-key"), b"12345").unwrap();
    let before = evidence::inventory(temporary.path()).unwrap();
    fs::remove_file(temporary.path().join("old-key")).unwrap();
    fs::write(temporary.path().join("new-key"), b"123").unwrap();
    let after = evidence::inventory(temporary.path()).unwrap();
    let delta = evidence::change(&before, &after);
    assert_eq!(delta["removed_bytes"], 5);
    assert_eq!(delta["added_bytes"], 3);
    assert_eq!(delta["removed_objects"]["old-key"], 5);
    assert_eq!(after.bytes, 3);
}

fn stats(hits: u64, misses: u64) -> serde_json::Value {
    let mut stats = json!({"cache_hits":{"counts":{"Rust":hits},"adv_counts":{}},"cache_misses":{"counts":{"Rust":misses},"adv_counts":{}},"cache_errors":{"counts":{},"adv_counts":{}},"not_cached":{}});
    for name in [
        "compile_requests",
        "requests_unsupported_compiler",
        "requests_not_compile",
        "requests_not_cacheable",
        "requests_executed",
        "cache_timeouts",
        "cache_read_errors",
        "non_cacheable_compilations",
        "forced_recaches",
        "cache_write_errors",
        "cache_writes",
        "compilations",
        "compile_fails",
        "dist_errors",
    ] {
        stats[name] = json!(if name == "compile_requests" { 100 } else { 0 });
    }
    json!({"stats":stats})
}

#[test]
fn compiler_seed_measurement_uses_cacheable_denominator_and_rejects_errors() {
    let mut raw = stats(3, 1);
    assert_eq!(
        evidence::measurement(&serde_json::to_vec(&raw).unwrap()).unwrap()["hit_rate"],
        0.75
    );
    assert!(evidence::measurement(&serde_json::to_vec(&stats(0, 0)).unwrap()).is_err());
    for name in [
        "compile_fails",
        "cache_read_errors",
        "cache_write_errors",
        "cache_timeouts",
    ] {
        raw["stats"][name] = json!(1);
        assert!(evidence::measurement(&serde_json::to_vec(&raw).unwrap()).is_err());
        raw["stats"][name] = json!(0);
    }
    assert!(evidence::measurement(b"{}").is_err());
}

#[test]
fn compiler_seed_existing_evidence_is_never_overwritten() {
    let temporary = tempfile::tempdir().unwrap();
    fs::write(temporary.path().join("result.json"), "user evidence").unwrap();
    let mut args = argv("1");
    args[4] = temporary.path().to_string_lossy().into_owned();
    assert!(run(temporary.path(), &args).is_err());
    assert_eq!(
        fs::read_to_string(temporary.path().join("result.json")).unwrap(),
        "user evidence"
    );
}

#[test]
fn compiler_seed_failed_preflight_retains_inconclusive_receipt() {
    let temporary = tempfile::tempdir().unwrap();
    let output = temporary.path().join("evidence");
    let mut args = argv("1");
    args[4] = output.to_string_lossy().into_owned();
    // A nonexistent source fails without invoking Cargo or touching other paths.
    assert!(run(&temporary.path().join("absent-source"), &args).is_err());
    let result: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["classification"], "inconclusive");
    assert_eq!(result["cache_published"], false);
    assert!(result["error"].is_string());
}

#[test]
fn compiler_seed_supervisor_preserves_nonzero_exit_and_failure_receipt() {
    let temporary = tempfile::tempdir().unwrap();
    let executable = std::env::current_exe().unwrap();
    let mut env = compiler_seed_process::Environment::new();
    env.insert(
        "PATH".into(),
        executable.parent().unwrap().as_os_str().into(),
    );
    let runner = Runner {
        root: temporary.path().canonicalize().unwrap(),
        environment: env,
        cancellation: Cancellation::default(),
        deadline: Instant::now() + Duration::from_secs(20),
    };
    let evidence = temporary.path().join("fixture");
    assert!(
        runner
            .command(
                executable.file_name().unwrap().to_str().unwrap(),
                &["--compiler-seed-invalid-fixture-argument"],
                &evidence
            )
            .is_err()
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(evidence.with_extension("process.json")).unwrap())
            .unwrap();
    assert_ne!(receipt["exit_code"], 0);
    assert_eq!(receipt["cleanup_complete"], true);
    assert!(evidence.with_extension("failure.json").is_file());
    assert!(evidence.with_extension("stderr.log").is_file());
}

#[test]
fn compiler_seed_git_admits_only_its_checkout_with_foreign_ownership() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let mut environment = environment(&root, &root.join("cache"), &root.join("target"));
    environment.insert("GIT_TEST_ASSUME_DIFFERENT_OWNER".into(), "1".into());
    let runner = Runner {
        root: root.clone(),
        environment,
        cancellation: Cancellation::default(),
        deadline: Instant::now() + Duration::from_secs(30),
    };
    runner
        .command("git", &["init", "--quiet"], &root.join("init"))
        .unwrap();
    let result = runner
        .command(
            "git",
            &["rev-parse", "--is-inside-work-tree"],
            &root.join("admission"),
        )
        .unwrap();
    assert_eq!(String::from_utf8(result).unwrap().trim(), "true");
    let config = runner
        .command(
            "git",
            &["config", "--get", "safe.directory"],
            &root.join("config"),
        )
        .unwrap();
    assert_eq!(
        String::from_utf8(config).unwrap().trim(),
        root.to_str().unwrap()
    );
}

#[cfg(unix)]
#[test]
fn compiler_seed_fixture_phases_retain_failures_and_run_both_fresh_consumers() {
    use std::os::unix::fs::PermissionsExt;
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let bin = root.join("bin");
    fs::create_dir(&bin).unwrap();
    let mut initial = stats(0, 0);
    initial["stats"]["compile_requests"] = json!(0);
    let initial = serde_json::to_string(&initial).unwrap();
    let measured = serde_json::to_string(&stats(2, 2)).unwrap();
    let cache_fixture = format!(
        "#!/usr/bin/env sh\ncase \"$1\" in\n--zero-stats) printf '%s' '{initial}' > \"$SCCACHE_CACHED_CONF\";;\n--show-stats) IFS= read -r payload < \"$SCCACHE_CACHED_CONF\" || :; printf '%s' \"$payload\";;\nesac\n"
    );
    let build_fixture = format!(
        "#!/usr/bin/env sh\nif [ \"$1\" != ci-sccache-seed-build ] && [ \"$1\" != skippy-cli-release-build ]; then [ ! -f \"$CARGO_TARGET_DIR/marker\" ] || exit 89; fi\nprintf '%s' marker > \"$CARGO_TARGET_DIR/marker\"\nprintf '%s' '{measured}' > \"$SCCACHE_CACHED_CONF\"\nprintf '%s' object > \"$SCCACHE_DIR/$1\"\nif [ \"$1\" = ci-sccache-seed-build ]; then exit 23; fi\n"
    );
    for (name, text) in [("sccache", cache_fixture), ("just", build_fixture)] {
        let path = bin.join(name);
        fs::write(&path, text).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    for arm in ["current", "extended"] {
        let work = root.join(arm);
        let output = root.join(format!("evidence-{arm}"));
        fs::create_dir(&work).unwrap();
        fs::create_dir(&output).unwrap();
        for name in ["cache", "target", "tmp"] {
            fs::create_dir(work.join(name)).unwrap();
        }
        let mut environment = environment(&work, &work.join("cache"), &work.join("target"));
        let mut paths = vec![bin.clone()];
        paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap()));
        environment.insert("PATH".into(), std::env::join_paths(paths).unwrap());
        let runner = Runner {
            root: root.clone(),
            environment,
            cancellation: Cancellation::default(),
            deadline: Instant::now() + Duration::from_secs(30),
        };
        assert!(phases(&runner, arm, &work, &output).is_err());
        for consumer in ["consumer-standalone", "consumer-mesh"] {
            let receipt: serde_json::Value = serde_json::from_slice(
                &fs::read(output.join(consumer).join("phase.json")).unwrap(),
            )
            .unwrap();
            assert!(receipt["build_error"].is_null());
            assert_eq!(receipt["measurement"]["hit_rate"], 0.5);
            assert_eq!(receipt["target"], work.join("target").to_str().unwrap());
        }
        let failed: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("seed/result.json")).unwrap()).unwrap();
        assert_eq!(failed["classification"], "inconclusive");
        assert!(output.join("seed/build.failure.json").is_file());
        assert_eq!(
            output.join("additional-seed/phase.json").is_file(),
            arm == "extended"
        );
    }
}
