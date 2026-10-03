//! Finite copied-real normal verify content and logged timeout qualification.
use super::*;
use std::time::Instant;

fn policy_source(root: &Path) {
    for dir in [
        "third_party/llama.cpp/patches/model_support",
        "third_party/llama.cpp/patches/generated",
        "ci/llama-canary",
        "docs/skippy",
        "crates/skippy-ffi/src",
        "crates/mesh-llm-host-runtime/src/inference/skippy",
    ] {
        fs::create_dir_all(root.join(dir)).unwrap();
    }
    fs::write(root.join(".gitignore"), ".deps/\ntarget/\n").unwrap();
    fs::write(
        root.join("third_party/llama.cpp/patches/0001-fixture.patch"),
        "fixture",
    )
    .unwrap();
    for (lane, name) in [
        ("model_support", "0001-model.patch"),
        ("generated", "0001-family-fixture.patch"),
    ] {
        fs::write(
            root.join(format!("third_party/llama.cpp/patches/{lane}/series")),
            format!("{name}\n"),
        )
        .unwrap();
        fs::write(
            root.join(format!("third_party/llama.cpp/patches/{lane}/{name}")),
            "fixture",
        )
        .unwrap();
    }
    fs::write(
        root.join("third_party/llama.cpp/upstream.txt"),
        format!("{}\n", "a".repeat(40)),
    )
    .unwrap();
    fs::write(root.join("crates/skippy-ffi/src/lib.rs"), "pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_PATCH: u32 = 3;\n").unwrap();
    let family = json!({"policy":{"profiles":{"full":{"status":"certified","required_lanes":["single-step","chain","state-handoff"]}}},"models":[{"family":"fixture","class":"causal_generation","architecture":"fixture","profile":"full","resources":{"estimated_model_bytes":1}}]});
    fs::write(
        root.join("ci/llama-canary/family-certified.json"),
        serde_json::to_vec(&family).unwrap(),
    )
    .unwrap();
    fs::write(root.join("docs/skippy/llama-parity-candidates.json"), serde_json::to_vec(&json!({"candidates":[{"llama_model":"fixture","family":"fixture","status":"needs_candidate"}]})).unwrap()).unwrap();
}
fn prepared_source(root: &Path) {
    let native = root.join(".deps/llama.cpp");
    fs::create_dir_all(native.join("src/models")).unwrap();
    fs::write(
        native.join("src/models/fixture.cpp"),
        "begin_block(layer); end_block(layer);\n",
    )
    .unwrap();
    git(&native, &["init", "--quiet"]);
    let head = commit(&native);
    // The finite preparation witness covers exactly the fixture's three literal patch members.
    let mut digest = Sha256::new();
    for name in [
        "0001-fixture.patch",
        "model_support/0001-model.patch",
        "generated/0001-family-fixture.patch",
    ] {
        digest.update(format!("{name}\n{}\n", hex::encode(Sha256::digest(b"fixture"))).as_bytes());
    }
    for (name, value) in [
        (".mesh-llm-upstream-sha", "a".repeat(40)),
        (".mesh-llm-patched-sha", head),
        (".mesh-llm-prepare-schema", "5".into()),
        (".mesh-llm-patch-digest", hex::encode(digest.finalize())),
    ] {
        fs::write(native.join(name), format!("{value}\n")).unwrap();
    }
}
impl Fixture {
    fn content_new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let work = temp.path().canonicalize().unwrap();
        let controller = work.join("controller");
        policy_source(&controller);
        git(&controller, &["init", "--quiet"]);
        let base = commit(&controller);
        let family_path = controller.join("ci/llama-canary/family-certified.json");
        let mut family: Json = serde_json::from_slice(&fs::read(&family_path).unwrap()).unwrap();
        family["models"][0]["resources"]["estimated_model_bytes"] = json!(2);
        fs::write(&family_path, serde_json::to_vec(&family).unwrap()).unwrap();
        let digest = hex::encode(Sha256::digest(
            fs::read(env!("CARGO_BIN_EXE_xtask")).unwrap(),
        ));
        let input = work.join("input.json");
        // Produce the expected roster through the existing real local owner before freezing candidate.
        fs::write(&input, serde_json::to_vec(&json!({"authority":{"controller":{"root":controller,"revision":base,"executable_sha256":digest},"root":controller,"base":base},"check":false})).unwrap()).unwrap();
        let roster = run(
            Path::new(env!("CARGO_BIN_EXE_xtask")),
            vec![
                "automation".into(),
                "canary-receipts".into(),
                "local-split-roster".into(),
                "--input".into(),
                input.to_str().unwrap().into(),
            ],
            &work,
        );
        assert!(roster.process.success(), "{:?}", roster.process);
        let head = commit(&controller);
        let tree = git(&controller, &["rev-parse", "HEAD^{tree}"]);
        let candidate = work.join("candidate");
        git(
            &controller,
            &[
                "worktree",
                "add",
                "--quiet",
                "--detach",
                candidate.to_str().unwrap(),
                &head,
            ],
        );
        git(&controller, &["checkout", "--quiet", "--detach", &base]);
        fs::write(
            controller.join("trusted-note.txt"),
            "advanced independent verifier\n",
        )
        .unwrap();
        let revision = commit(&controller);
        prepared_source(&candidate);
        let document = json!({"authority":{"controller":{"root":controller,"revision":revision,"executable_sha256":digest},"root":candidate,"base":base,"candidate":head,"tree":tree}});
        Self {
            _temp: temp,
            work,
            input,
            document,
        }
    }
    fn logged_script(&self, action: &str) -> String {
        let source = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/llama-canary-agent-repair.sh"),
        )
        .unwrap();
        let selection = source
            .split_once("# Legacy workload automation selection begins.\n")
            .unwrap()
            .1
            .split_once("# Legacy workload automation selection ends.")
            .unwrap()
            .0;
        assert!(
            source
                .find("# Legacy workload automation selection begins.")
                .unwrap()
                < source.find("\nload_candidate_bundle\n").unwrap()
        );
        let helpers = [
            "verification_source_inspection",
            "verification_candidate_unchanged",
            "repair_family_plan_step",
            "run_for",
            "remaining_verification_seconds",
            "run_verification_logged",
        ]
        .map(|name| caller_function(&source, name))
        .join("\n");
        format!(
            "set -euo pipefail\nHARNESS_MODE=verify\nMESH_LLM_AUTOMATION_BIN=$1\nTRUSTED_ROOT=$2\nBASE_HEAD=$3\nROOT=$4\nCANDIDATE_BASE_HEAD=$5\nCERTIFIED_SHA=$6\nVERIFICATION_TREE=$7\nRUNNER_TEMP=$8\nPATH=\"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin\"\nexport PATH\n{selection}\n{helpers}\n{action}\n"
        )
    }
    fn logged_spec(&self, action: &str) -> ProcessSpec {
        let binary = self.work.join("frozen-controller");
        if !binary.exists() {
            fs::copy(env!("CARGO_BIN_EXE_xtask"), &binary).unwrap();
        }
        let a = &self.document["authority"];
        let c = &a["controller"];
        ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: self.work.clone(),
            environment: BTreeMap::from([
                ("PATH".into(), Value::Public("/usr/bin:/bin".into())),
                (
                    "VERIFY_FIXTURE_ENV".into(),
                    Value::Public("kept value".into()),
                ),
                ("GIT_CONFIG_NOSYSTEM".into(), Value::Public("1".into())),
                (
                    "GIT_CONFIG_GLOBAL".into(),
                    Value::Public("/dev/null".into()),
                ),
                ("GIT_ALLOW_PROTOCOL".into(), Value::Public("file".into())),
            ]),
            arguments: vec![
                "-c".into(),
                self.logged_script(action),
                "real-verify-wrapper".into(),
                binary.display().to_string(),
                c["root"].as_str().unwrap().into(),
                c["revision"].as_str().unwrap().into(),
                a["root"].as_str().unwrap().into(),
                a["base"].as_str().unwrap().into(),
                a["candidate"].as_str().unwrap().into(),
                a["tree"].as_str().unwrap().into(),
                self.work.display().to_string(),
            ]
            .into_iter()
            .map(|arg: String| Value::Public(arg.into()))
            .collect(),
        }
    }
    fn logged(&self, action: &str) -> process::RawProcessReport {
        logged_raw(&self.logged_spec(action))
    }
}
fn logged_raw(spec: &ProcessSpec) -> process::RawProcessReport {
    process::supervise_raw(
        spec,
        &Limits {
            execution: Duration::from_secs(45),
            graceful_shutdown: Duration::from_secs(12),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}
#[test]
fn copied_real_verify_content_wrappers_admit_all_three_with_advanced_controller_and_exact_prepared_candidate()
 {
    let f = Fixture::content_new();
    assert_ne!(
        f.document["authority"]["base"],
        f.document["authority"]["controller"]["revision"]
    );
    let roster = f
        .work
        .join("candidate/crates/mesh-llm-host-runtime/src/inference/skippy/split-certified.json");
    let before = fs::read(&roster).unwrap();
    for (verb, status) in [
        (
            "verification-manifest-policy",
            "agent_manifest_policy_admitted",
        ),
        ("verification-parity-inventory", "parity_inventory_admitted"),
        ("verification-split-roster-check", "split_roster_admitted"),
    ] {
        let log = f.work.join(format!("{verb}.log"));
        let action = format!(
            "VERIFICATION_DEADLINE_AT=$(( $(date +%s) + 20 ))\nverification_source_inspection {verb} '{}'",
            log.display()
        );
        let report = f.logged(&action);
        assert!(
            report.process.success(),
            "{verb}: {:?}; {}",
            report.process,
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
        );
        let result: Json = serde_json::from_slice(report.stdout.unwrap().as_bytes()).unwrap();
        assert_eq!(result["status"], status);
        if verb == "verification-parity-inventory" {
            assert_eq!(result["model_sources"], 1);
        }
        assert_eq!(
            serde_json::from_slice::<Json>(&fs::read(log).unwrap()).unwrap(),
            result
        );
        assert!(report.process.cleanup.complete);
        f.no_transport_temps();
    }
    assert_eq!(fs::read(roster).unwrap(), before);
    assert_eq!(
        git(&f.work.join("candidate"), &["rev-parse", "HEAD"]),
        f.document["authority"]["candidate"].as_str().unwrap()
    );
}

#[test]
fn copied_real_logged_verify_preserves_child_status_literal_argv_environment_and_tee() {
    let f = Fixture::new();
    let action="VERIFICATION_DEADLINE_AT=$(( $(date +%s) + 10 ))
if run_verification_logged 'finite status' verify.log /bin/bash -c 'test \"$VERIFY_FIXTURE_ENV\" = \"kept value\" || exit 97; printf \"%s\\0\" \"$@\" > child.args; printf \"live stdout\\n\"; printf \"live stderr\\n\" >&2; exit 23' child '' 'space value' '*.txt'; then exit 96; else exit \"$?\"; fi";
    let report = f.logged(action);
    assert_eq!(report.process.status.unwrap().code(), Some(23));
    assert_eq!(
        fs::read(f.work.join("child.args")).unwrap(),
        b"\0space value\0*.txt\0"
    );
    let stdout = String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes());
    let log = fs::read_to_string(f.work.join("verify.log")).unwrap();
    for text in ["live stdout", "live stderr"] {
        assert!(stdout.contains(text));
        assert!(log.contains(text));
    }
    assert!(report.process.cleanup.complete);
    f.no_transport_temps();
}
#[test]
fn copied_real_logged_verify_uses_remaining_budget_and_exhaustion_refuses_spawn() {
    let f = Fixture::new();
    let exhausted=f.logged("VERIFICATION_DEADLINE_AT=$(( $(date +%s) - 1 ))\nif run_verification_logged exhausted exhausted.log /usr/bin/touch should-not-exist; then exit 96; else exit \"$?\"; fi");
    assert_eq!(exhausted.process.status.unwrap().code(), Some(124));
    assert!(!f.work.join("should-not-exist").exists());
    assert!(
        fs::read_to_string(f.work.join("exhausted.log"))
            .unwrap()
            .contains("final verification budget exhausted")
    );
    f.no_transport_temps();
    let started = Instant::now();
    let deadline=f.logged("VERIFICATION_DEADLINE_AT=$(( $(date +%s) + 3 ))\n/bin/sleep 1\nif run_verification_logged remaining remaining.log /bin/sleep 20; then exit 96; else exit \"$?\"; fi");
    assert_eq!(deadline.process.status.unwrap().code(), Some(124));
    assert!(started.elapsed() < Duration::from_secs(15));
    let log = fs::read_to_string(f.work.join("remaining.log")).unwrap();
    assert!(
        log.contains("timed out after 2s") || log.contains("timed out after 1s"),
        "{log}"
    );
    assert!(deadline.process.cleanup.complete);
    f.no_transport_temps();
}
#[test]
fn copied_real_logged_verify_frozen_owner_checks_before_and_after_child() {
    let before = Fixture::new();
    let report=before.logged("printf changed >> \"$repair_workload_controller\"\nVERIFICATION_DEADLINE_AT=$(( $(date +%s) + 10 ))\nif run_verification_logged changed changed.log /usr/bin/touch not-launched; then exit 96; else exit \"$?\"; fi");
    assert_eq!(report.process.status.unwrap().code(), Some(125));
    assert!(!before.work.join("not-launched").exists());
    before.no_transport_temps();
    let after = Fixture::new();
    let report=after.logged("VERIFICATION_DEADLINE_AT=$(( $(date +%s) + 10 ))\nif run_verification_logged changed changed.log /bin/bash -c 'cp /usr/bin/true \"$1.new\"; mv \"$1.new\" \"$1\"' child \"$repair_workload_controller\"; then exit 96; else exit \"$?\"; fi");
    assert_eq!(report.process.status.unwrap().code(), Some(125));
    assert!(
        fs::read_to_string(after.work.join("changed.log"))
            .unwrap()
            .contains("frozen workload automation controller changed")
    );
    after.no_transport_temps();
}
fn process_gone(pid: i32) -> bool {
    let until = Instant::now() + Duration::from_secs(2);
    loop {
        if unsafe { libc::kill(pid, 0) } == -1
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
        {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
struct UnrelatedSentinel(std::process::Child);
impl Drop for UnrelatedSentinel {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
fn copied_real_logged_verify_cancellation_cleans_owned_descendant_and_request_directory() {
    let f = Fixture::new();
    let mut sentinel = UnrelatedSentinel(
        std::process::Command::new("/bin/sleep")
            .arg("40")
            .spawn()
            .unwrap(),
    );
    let owner = f.work.join("timeout-owner.pid");
    let signal_file = owner.clone();
    let signal = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(8);
        while !signal_file.exists() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(10));
        }
        let pid = fs::read_to_string(signal_file)
            .unwrap()
            .trim()
            .parse::<i32>()
            .unwrap();
        assert!(pid > 1);
        assert_eq!(unsafe { libc::kill(pid, libc::SIGTERM) }, 0);
    });
    let started = Instant::now();
    let report=f.logged("VERIFICATION_DEADLINE_AT=$(( $(date +%s) + 30 ))\nif run_verification_logged cancelled cancelled.log /bin/bash -c 'trap \"wait; exit 143\" TERM; /bin/sleep 30 & printf \"%s\\n\" \"$!\" > descendant.pid; printf \"ready\\n\"; printf \"%s\\n\" \"$PPID\" > timeout-owner.pid; wait'; then exit 96; else exit \"$?\"; fi");
    signal.join().unwrap();
    let sentinel_alive = sentinel.0.try_wait().unwrap().is_none();
    drop(sentinel);
    assert!(
        sentinel_alive,
        "unrelated process must survive owned cancellation"
    );
    assert_eq!(report.process.outcome, process::Outcome::Exited);
    assert_eq!(report.process.status.unwrap().code(), Some(143));
    assert!(report.process.cleanup.complete && !report.process.cleanup.forced);
    assert!(started.elapsed() < Duration::from_secs(25));
    let descendant = fs::read_to_string(f.work.join("descendant.pid"))
        .unwrap()
        .trim()
        .parse::<i32>()
        .unwrap();
    assert!(
        process_gone(descendant),
        "owned descendant remained after typed cancellation"
    );
    let log = fs::read_to_string(f.work.join("cancelled.log")).unwrap();
    assert!(
        log.contains("ready") && log.contains("received signal 15"),
        "{log}"
    );
    f.no_transport_temps();
}

#[test]
fn copied_real_final_bundle_uses_candidate_base_when_controller_contains_candidate() {
    let mut fixture = Fixture::content_new();
    let candidate = fixture.document["authority"]["candidate"]
        .as_str()
        .unwrap()
        .to_owned();
    let controller = PathBuf::from(
        fixture.document["authority"]["controller"]["root"]
            .as_str()
            .unwrap(),
    );
    // The verifier is legitimately newer and already includes this candidate.
    git(
        &controller,
        &["checkout", "--quiet", "--detach", &candidate],
    );
    fs::write(
        controller.join("trusted-note.txt"),
        "later trusted controller revision\n",
    )
    .unwrap();
    fixture.document["authority"]["controller"]["revision"] = json!(commit(&controller));
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/llama-canary-agent-repair.sh"),
    )
    .unwrap();
    let finalization = caller_function(&source, "finalize_certified_tree");
    let pin = caller_function(&source, "verify_repair_pin");
    let action = format!(
        "cd \"$ROOT\"\nPIN_FILE=\"$ROOT/third_party/llama.cpp/upstream.txt\"\nUPSTREAM_SHA={}\nBRANCH=llama-canary/repair-fixture\nBUNDLE=\"$RUNNER_TEMP/final.bundle\"\nPR_BODY=\"$RUNNER_TEMP/body.md\"\nGITHUB_OUTPUT=\"$RUNNER_TEMP/final.outputs\"\nwrite_pr_body() {{ printf 'finite body\\n' > \"$PR_BODY\"; }}\n{pin}\n{finalization}\nfinalize_certified_tree\n",
        "a".repeat(40)
    );
    let report = fixture.logged(&action);
    assert!(
        report.process.success(),
        "{:?} {:?}",
        report.process,
        report.stderr
    );
    assert!(report.process.cleanup.complete);
    let bundle = fixture.work.join("final.bundle");
    let heads = git(
        &controller,
        &[
            "bundle",
            "list-heads",
            bundle.to_str().unwrap(),
            "refs/heads/llama-canary/repair-fixture",
        ],
    );
    assert_eq!(
        heads,
        format!("{candidate} refs/heads/llama-canary/repair-fixture")
    );
    let outputs = fs::read_to_string(fixture.work.join("final.outputs")).unwrap();
    assert!(
        outputs
            .lines()
            .any(|line| line == format!("head={candidate}"))
    );
    assert_eq!(
        git(&controller, &["rev-parse", "HEAD"]),
        fixture.document["authority"]["controller"]["revision"]
            .as_str()
            .unwrap()
    );
}
