use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, FileTimes},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::{Duration, UNIX_EPOCH},
};
fn command(executable: &str, root: &Path, args: Vec<String>) -> process::ProcessReport {
    let spec = ProcessSpec {
        executable: executable.into(),
        cwd: root.into(),
        arguments: args.into_iter().map(|v| Value::Public(v.into())).collect(),
        environment: BTreeMap::from([
            (
                "PATH".into(),
                Value::Public("/usr/bin:/bin:/opt/homebrew/bin".into()),
            ),
            ("GIT_MASTER".into(), Value::Public("1".into())),
            (
                "GIT_CONFIG_GLOBAL".into(),
                Value::Public("/dev/null".into()),
            ),
            ("GIT_CONFIG_NOSYSTEM".into(), Value::Public("1".into())),
        ]),
    };
    let limits = Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let result = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        result.cleanup.complete && !result.stdout.truncated && !result.stderr.truncated,
        "{result:?}"
    );
    result
}
fn git(root: &Path, args: &[&str]) -> String {
    let r = command(
        "/usr/bin/git",
        root,
        args.iter().map(|v| v.to_string()).collect(),
    );
    assert!(r.success(), "{r:?}");
    String::from_utf8(r.stdout.bytes_retained)
        .unwrap()
        .trim()
        .into()
}
fn commit(root: &Path) -> String {
    git(root, &["add", "-A"]);
    git(
        root,
        &[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "authored fixture",
        ],
    );
    git(root, &["rev-parse", "HEAD"])
}
fn write(path: &Path, bytes: &[u8], seconds: u64) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(seconds)))
        .unwrap();
}
struct Fixture {
    temp: tempfile::TempDir,
    root: PathBuf,
    closure: PathBuf,
    test: PathBuf,
    snapshot: PathBuf,
    native: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("selected source");
        fs::create_dir(&root).unwrap();
        git(&root, &["init", "-q"]);
        fs::write(root.join(".gitignore"), ".deps/\ntarget/\n").unwrap();
        fs::write(root.join("source.rs"), "original\n").unwrap();
        let patches = root.join("third_party/llama.cpp/patches");
        fs::create_dir_all(&patches).unwrap();
        fs::write(patches.join("0001-core.patch"), b"fixture recipe").unwrap();
        let upstream = "a".repeat(40);
        fs::write(root.join("third_party/llama.cpp/upstream.txt"), &upstream).unwrap();
        commit(&root);
        let prepared = root.join(".deps/llama.cpp");
        fs::create_dir_all(&prepared).unwrap();
        git(&prepared, &["init", "-q"]);
        fs::write(prepared.join("source.cpp"), "fixture\n").unwrap();
        let native = commit(&prepared);
        let patch_hash = hex::encode(Sha256::digest(b"fixture recipe"));
        let recipe = format!("0001-core.patch\n{patch_hash}\n");
        let recipe_hash = hex::encode(Sha256::digest(recipe.as_bytes()));
        for (name, value) in [
            (".mesh-llm-upstream-sha", upstream),
            (".mesh-llm-patched-sha", native.clone()),
            (".mesh-llm-patch-digest", recipe_hash),
            (".mesh-llm-prepare-schema", "5".into()),
        ] {
            fs::write(prepared.join(name), format!("{value}\n")).unwrap();
        }
        let closure = temp.path().join("producer output");
        fs::create_dir(&closure).unwrap();
        let stamp =
            format!("stamp-version=3\npatched-sha={native}\nbackend=cpu\nlink-mode=static\n");
        write(
            &closure.join("native/.mesh-llm-build-stamp"),
            stamp.as_bytes(),
            100,
        );
        for name in [
            "cargo/debug/skippy-server",
            "cargo/debug/skippy-model-package",
            "cargo/debug/skippy-correctness",
            "cargo/debug/skippy-topology-plan",
            "native/bin/llama-server",
            "native/bin/llama-completion",
            "native/bin/llama-tts",
            "cargo/debug/deps/skippy_server-fixture",
        ] {
            write(
                &closure.join(name),
                b"platform-neutral authored executable data\n",
                200,
            );
        }
        let test = closure.join("cargo/debug/deps/skippy_server-fixture");
        let snapshot = closure.join("source.json");
        Self {
            temp,
            root,
            closure,
            test,
            snapshot,
            native,
        }
    }
    fn run(&self, mode: &str, paths: &[&Path]) -> process::ProcessReport {
        let mut args = vec![
            "automation".into(),
            "canary-receipts".into(),
            "workload-manifest".into(),
            mode.into(),
            self.root.display().to_string(),
        ];
        args.extend(paths.iter().map(|p| p.display().to_string()));
        command(env!("CARGO_BIN_EXE_xtask"), self.temp.path(), args)
    }
    fn snapshot(&self) {
        let r = self.run("snapshot", &[&self.snapshot]);
        assert!(r.success(), "{r:?}");
    }
    fn produce(&self) -> process::ProcessReport {
        self.run("produce", &[&self.closure, &self.test, &self.snapshot])
    }
    fn verify(&self) -> process::ProcessReport {
        self.run(
            "verify",
            &[
                &self.closure.join("cargo/debug/skippy-server"),
                &self.closure.join("native"),
                &self.closure.join("producer.json"),
            ],
        )
    }
}
#[test]
fn actual_dirty_source_producer_verifies_relocated_files_and_rejects_source_and_digest_changes() {
    let mut f = Fixture::new();
    fs::write(f.root.join("source.rs"), "dirty tracked\n").unwrap();
    fs::write(f.root.join("new.rs"), "new source\n").unwrap();
    f.snapshot();
    assert!(f.produce().success());
    assert!(f.verify().success());
    let moved = f.temp.path().join("relocated closure");
    fs::rename(&f.closure, &moved).unwrap();
    f.closure = moved;
    assert!(f.verify().success());
    let original = fs::read(f.closure.join("producer.json")).unwrap();
    fs::write(f.root.join("new.rs"), "changed source\n").unwrap();
    assert!(!f.verify().success());
    fs::write(f.root.join("new.rs"), "new source\n").unwrap();
    let binary = f.closure.join("cargo/debug/skippy-server");
    let mut bytes = fs::read(&binary).unwrap();
    bytes[0] = b'X';
    fs::write(binary, bytes).unwrap();
    assert!(!f.verify().success());
    assert_eq!(fs::read(f.closure.join("producer.json")).unwrap(), original);
}
#[test]
fn actual_producer_rejects_changed_prebuild_snapshot_stale_test_wrong_stamp_and_escape() {
    let f = Fixture::new();
    f.snapshot();
    fs::write(f.root.join("source.rs"), "changed\n").unwrap();
    assert!(!f.produce().success());
    assert!(!f.closure.join("producer.json").exists());
    let f = Fixture::new();
    f.snapshot();
    write(&f.test, b"test", 100);
    assert!(!f.produce().success());
    let f = Fixture::new();
    f.snapshot();
    let stamp = f.closure.join("native/.mesh-llm-build-stamp");
    write(
        &stamp,
        format!(
            "stamp-version=3\npatched-sha={}\nbackend=metal\nlink-mode=static\n",
            f.native
        )
        .as_bytes(),
        100,
    );
    assert!(!f.produce().success());
    let f = Fixture::new();
    f.snapshot();
    let binary = f.closure.join("cargo/debug/skippy-server");
    fs::remove_file(&binary).unwrap();
    std::os::unix::fs::symlink(&f.test, binary).unwrap();
    assert!(!f.produce().success());
}
#[test]
fn actual_local_freshness_preserves_platform_neutral_executable_and_strict_native_mtime() {
    let f = Fixture::new();
    let binary = f.closure.join("cargo/debug/skippy-server");
    let native = f.closure.join("native");
    assert!(f.run("fresh", &[&binary, &native]).success());
    write(&binary, b"freshness equal", 100);
    assert!(!f.run("fresh", &[&binary, &native]).success());
    write(&binary, b"freshness restored", 200);
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(!f.run("fresh", &[&binary, &native]).success());
}

#[test]
fn actual_producer_and_certify_caller_commands_admit_without_executing_native_builds() {
    let f = Fixture::new();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let caller =
        fs::read_to_string(repository.join("scripts/skippy-workload-oracles-build.sh")).unwrap();
    let snapshot = caller
        .lines()
        .find(|line| line.contains("workload-manifest snapshot"))
        .unwrap();
    let produce = caller
        .split("\"${workload_automation[@]}\" automation canary-receipts workload-manifest produce")
        .nth(1)
        .unwrap();
    let certify =
        fs::read_to_string(repository.join("scripts/skippy-workload-certify.sh")).unwrap();
    let commands = certify.lines().collect::<Vec<_>>();
    let mut admission = String::new();
    for mode in ["verify", "fresh"] {
        let index = commands
            .iter()
            .position(|line| line.contains(&format!("workload-manifest {mode} ")))
            .unwrap();
        admission.push_str(commands[index]);
        admission.push('\n');
        admission.push_str(commands[index + 1]);
        admission.push('\n');
    }
    let script = f.temp.path().join("real caller commands.sh");
    fs::write(&script,format!("set -euo pipefail\nworkload_automation=(\"$OWNER\")\n{snapshot}\n\"${{workload_automation[@]}}\" automation canary-receipts workload-manifest produce{produce}\n{admission}")).unwrap();
    let mut args = vec![script.display().to_string()];
    // Environment values are passed as argv to env, with no shell interpolation.
    let mut environment = vec![
        format!("OWNER={}", env!("CARGO_BIN_EXE_xtask")),
        format!("ROOT={}", f.root.display()),
        format!("BUILD_ROOT={}", f.closure.display()),
        format!("test_binary={}", f.test.display()),
        format!(
            "CANDIDATE_BIN_DIR={}",
            f.closure.join("cargo/debug").display()
        ),
        format!("CANDIDATE_BUILD_DIR={}", f.closure.join("native").display()),
        format!(
            "PRODUCER_MANIFEST={}",
            f.closure.join("producer.json").display()
        ),
        "/bin/bash".into(),
    ];
    environment.append(&mut args);
    let result = command("/usr/bin/env", f.temp.path(), environment);
    assert!(result.success(), "{result:?}");
    assert!(f.verify().success());
}

#[test]
fn actual_producer_rejects_every_admitted_member_mutation_and_missing_file_without_republishing() {
    let f = Fixture::new();
    f.snapshot();
    assert!(f.produce().success());
    let manifest_path = f.closure.join("producer.json");
    let admitted = fs::read(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&admitted).unwrap();
    let files = manifest["files"].as_object().unwrap();
    assert_eq!(
        files.len(),
        9,
        "candidate/test/tools/native stamp and three oracles"
    );
    for (name, record) in files {
        let path = f.closure.join(record["path"].as_str().unwrap());
        let original = fs::read(&path).unwrap();
        let metadata = fs::metadata(&path).unwrap();
        let mut changed = original.clone();
        changed.push(b'X');
        fs::write(&path, changed).unwrap();
        assert!(!f.verify().success(), "accepted changed member {name}");
        assert_eq!(fs::read(&manifest_path).unwrap(), admitted);
        fs::remove_file(&path).unwrap();
        assert!(!f.verify().success(), "accepted missing member {name}");
        assert_eq!(fs::read(&manifest_path).unwrap(), admitted);
        fs::write(&path, original).unwrap();
        fs::set_permissions(&path, metadata.permissions()).unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(FileTimes::new().set_modified(metadata.modified().unwrap()))
            .unwrap();
        assert!(f.verify().success(), "restored member {name} remains valid");
    }
}

#[test]
fn actual_freshness_rejects_older_missing_inputs_and_wrong_cpu_static_native_identity() {
    for seconds in [99, 100] {
        let f = Fixture::new();
        let binary = f.closure.join("cargo/debug/skippy-server");
        let native = f.closure.join("native");
        write(&binary, b"existing executable", seconds);
        assert!(
            !f.run("fresh", &[&binary, &native]).success(),
            "accepted mtime={seconds}"
        );
    }
    for missing in ["cargo/debug/skippy-server", "native/.mesh-llm-build-stamp"] {
        let f = Fixture::new();
        fs::remove_file(f.closure.join(missing)).unwrap();
        assert!(
            !f.run(
                "fresh",
                &[
                    &f.closure.join("cargo/debug/skippy-server"),
                    &f.closure.join("native")
                ]
            )
            .success(),
            "accepted missing {missing}"
        );
    }
    for replacement in ["version", "head", "backend", "link", "duplicate", "missing"] {
        let f = Fixture::new();
        let stamp = f.closure.join("native/.mesh-llm-build-stamp");
        let original = fs::read_to_string(&stamp).unwrap();
        let changed = match replacement {
            "version" => original.replace("stamp-version=3", "stamp-version=2"),
            "head" => original.replace(&f.native, &"f".repeat(40)),
            "backend" => original.replace("backend=cpu", "backend=metal"),
            "link" => original.replace("link-mode=static", "link-mode=dynamic"),
            "duplicate" => format!("{original}backend=cpu\n"),
            "missing" => original
                .lines()
                .filter(|line| !line.starts_with("patched-sha="))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => unreachable!(),
        };
        assert_ne!(changed, original);
        write(&stamp, changed.as_bytes(), 100);
        assert!(
            !f.run(
                "fresh",
                &[
                    &f.closure.join("cargo/debug/skippy-server"),
                    &f.closure.join("native")
                ]
            )
            .success(),
            "accepted stamp {replacement}"
        );
    }
}

fn legacy_repair_fixture() -> Fixture {
    let mut fixture = Fixture::new();
    let closure = fixture.temp.path().join("cpu native-workloads");
    fs::rename(&fixture.closure, &closure).unwrap();
    fixture.closure = closure;
    fixture.test = fixture
        .closure
        .join("cargo/debug/deps/skippy_server-fixture");
    fixture.snapshot = fixture.closure.join("source.json");
    fs::write(
        fixture.root.join("source.rs"),
        "admitted dirty tracked source\n",
    )
    .unwrap();
    fs::write(
        fixture.root.join("new source.rs"),
        "admitted untracked source\n",
    )
    .unwrap();
    fixture.snapshot();
    let output = fixture.produce();
    assert!(output.success(), "{output:?}");
    fixture
}
struct RepairCaller {
    script: PathBuf,
    bin: PathBuf,
    trusted: PathBuf,
    trace: PathBuf,
    receipt: PathBuf,
    just_trace: PathBuf,
    controller_mutation: bool,
    harness_mode: &'static str,
}
impl RepairCaller {
    fn new(fixture: &Fixture) -> Self {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source =
            fs::read_to_string(repository.join("scripts/llama-canary-agent-repair.sh")).unwrap();
        let selector = source
            .split("# Legacy workload automation selection begins.")
            .nth(1)
            .unwrap()
            .split("# Legacy workload automation selection ends.")
            .next()
            .unwrap();
        let function = source
            .split("snapshot_candidate_tree() {")
            .nth(1)
            .unwrap()
            .split("\nwrite_candidate_bundle() {")
            .next()
            .unwrap();
        let trusted = fixture.temp.path().join("trusted controller");
        let bin = fixture.temp.path().join("selector tools");
        fs::create_dir(&trusted).unwrap();
        fs::create_dir(&bin).unwrap();
        fs::write(trusted.join("Justfile"), "# intercepted normal facade\n").unwrap();
        let just = bin.join("just");
        fs::write(&just, "#!/bin/bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" >> \"$JUST_TRACE\"\n[[ \"$1\" == --justfile && \"$2\" == \"$TRUSTED_ROOT/Justfile\" && \"$3\" == automation-bootstrap && \"$#\" == 3 ]] || exit 94\nprintf 'binary_path=%s\\ntarget_directory=%s\\nhost=fixture\\n' \"$OWNER\" \"$TRUSTED_ROOT/target\"\n").unwrap();
        fs::set_permissions(&just, fs::Permissions::from_mode(0o700)).unwrap();
        let script = fixture.temp.path().join("production repair snapshot.sh");
        fs::write(
            &script,
            format!(
                r#"set -euo pipefail
{selector}
if [[ "$HARNESS_MODE" != repair ]]; then printf 'repair-snapshot-not-applicable\n'; exit 0; fi
if [[ "${{FREEZE_MUTATION:-}}" == yes ]]; then
  MESH_LLM_AUTOMATION_BIN=relative-after-selection
  OWNER="$TRUSTED_ROOT/would-be-rebuilt-owner"
  printf '#!/bin/sh\nexit 96\n' > "$OWNER"
  chmod +x "$OWNER"
fi
if [[ "${{CONTROLLER_MUTATION:-}}" == yes ]]; then printf '#!/bin/sh\nexit 97\n' > "$repair_workload_controller"; fi
assert_agent_control_unchanged() {{ return 0; }}
verify_repair_pin() {{ return 0; }}
validate_agent_manifest_changes() {{ return 0; }}
controller_producer_receipt() {{ printf controller > "$CONTROLLER_MARKER"; }}
git() {{
  printf '%s\n' "$@" >> "$REPAIR_FIXTURE_GIT_LOG"
  if [[ "$1" == commit-tree ]]; then /bin/cat >/dev/null; printf '%s\n' "$FAKE_COMMIT"; return; fi
  /usr/bin/git "$@"
}}
snapshot_candidate_tree() {{{function}
cd "$ROOT"
snapshot_candidate_tree || exit "$?"
/usr/bin/printenv CANARY_VERIFIED_WORKLOAD_PRODUCER > "$RECEIPT"
printf '%s\n' "$VERIFICATION_TREE" "$CERTIFIED_SHA" > "$TREE_RECEIPT"
"#
            ),
        )
        .unwrap();
        Self {
            script,
            bin,
            trusted,
            trace: fixture.temp.path().join("git.trace"),
            receipt: fixture.temp.path().join("admitted.sha"),
            just_trace: fixture.temp.path().join("just.argv"),
            controller_mutation: false,
            harness_mode: "repair",
        }
    }
    fn invoke(
        &self,
        fixture: &Fixture,
        selected: Option<&str>,
        freeze_mutation: bool,
        cancellation: &Cancellation,
    ) -> process::ProcessReport {
        let native_base = fixture.temp.path().join("cpu native");
        let mut environment: BTreeMap<_, _> = [
            ("PATH", format!("{}:/usr/bin:/bin", self.bin.display())),
            ("ROOT", fixture.root.display().to_string()),
            ("TRUSTED_ROOT", self.trusted.display().to_string()),
            ("OWNER", env!("CARGO_BIN_EXE_xtask").into()),
            ("JUST_TRACE", self.just_trace.display().to_string()),
            ("REPAIR_FIXTURE_GIT_LOG", self.trace.display().to_string()),
            ("RECEIPT", self.receipt.display().to_string()),
            (
                "TREE_RECEIPT",
                fixture
                    .temp
                    .path()
                    .join("tree.receipt")
                    .display()
                    .to_string(),
            ),
            (
                "CONTROLLER_MARKER",
                fixture
                    .temp
                    .path()
                    .join("controller.marker")
                    .display()
                    .to_string(),
            ),
            ("FAKE_COMMIT", "b".repeat(40)),
            ("BASE_HEAD", git(&fixture.root, &["rev-parse", "HEAD"])),
            ("UPSTREAM_SHA", "a".repeat(40)),
            ("HARNESS_MODE", self.harness_mode.into()),
            ("LLAMA_STAGE_BUILD_DIR", native_base.display().to_string()),
            (
                "FREEZE_MUTATION",
                if freeze_mutation { "yes" } else { "no" }.into(),
            ),
            (
                "CONTROLLER_MUTATION",
                if self.controller_mutation {
                    "yes"
                } else {
                    "no"
                }
                .into(),
            ),
            ("GIT_MASTER", "1".into()),
            ("GIT_CONFIG_GLOBAL", "/dev/null".into()),
            ("GIT_CONFIG_NOSYSTEM", "1".into()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        if let Some(selected) = selected {
            environment.insert(
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(selected.into()),
            );
        }
        let result = process::supervise(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: fixture.temp.path().into(),
                arguments: vec![Value::Public(self.script.clone().into())],
                environment,
            },
            &Limits {
                execution: Duration::from_secs(8),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            cancellation,
            OutputFiles::default(),
        )
        .unwrap();
        assert!(result.cleanup.complete, "{result:?}");
        result
    }
}
#[test]
fn legacy_repair_snapshot_admits_exact_dirty_producer_before_staging_with_frozen_owner_or_just() {
    for configured in [true, false] {
        let fixture = legacy_repair_fixture();
        let caller = RepairCaller::new(&fixture);
        let manifest = fixture.closure.join("producer.json");
        let bytes = fs::read(&manifest).unwrap();
        let output = caller.invoke(
            &fixture,
            configured.then_some(env!("CARGO_BIN_EXE_xtask")),
            true,
            &Cancellation::default(),
        );
        assert!(output.success(), "{output:?}");
        assert_eq!(
            fs::read_to_string(&caller.receipt).unwrap(),
            format!("{}\n", hex::encode(Sha256::digest(&bytes)))
        );
        assert_eq!(fs::read(&manifest).unwrap(), bytes);
        let trace = fs::read_to_string(&caller.trace).unwrap();
        assert!(
            trace.starts_with("add\n-A\ndiff\n--cached\n--quiet\nwrite-tree\ncommit-tree\n"),
            "{trace}"
        );
        assert!(!fixture.temp.path().join("controller.marker").exists());
        assert_eq!(caller.just_trace.exists(), !configured);
        if !configured {
            let args = fs::read_to_string(&caller.just_trace).unwrap();
            assert_eq!(
                args,
                format!(
                    "--justfile\n{}\nautomation-bootstrap\n",
                    caller.trusted.join("Justfile").display()
                )
            );
        }
        // Untracked source is now staged; the old source snapshot cannot be reused.
        assert!(!fixture.verify().success());
    }
}
#[test]
fn legacy_repair_snapshot_rejects_mutated_source_closure_and_freshness_before_git_add_or_hash_export()
 {
    for mutation in [
        "source",
        "native-head",
        "binary",
        "test-mtime",
        "missing-manifest",
    ] {
        let fixture = legacy_repair_fixture();
        let caller = RepairCaller::new(&fixture);
        match mutation {
            "source" => fs::write(
                fixture.root.join("new source.rs"),
                "changed after producer\n",
            )
            .unwrap(),
            "native-head" => {
                let stamp = fixture.closure.join("native/.mesh-llm-build-stamp");
                let bytes = fs::read_to_string(&stamp)
                    .unwrap()
                    .replace(&fixture.native, &"f".repeat(40));
                fs::write(stamp, bytes).unwrap();
            }
            "binary" => fs::write(
                fixture.closure.join("cargo/debug/skippy-server"),
                "replacement executable\n",
            )
            .unwrap(),
            "test-mtime" => {
                File::options()
                    .write(true)
                    .open(&fixture.test)
                    .unwrap()
                    .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(100)))
                    .unwrap();
            }
            _ => fs::remove_file(fixture.closure.join("producer.json")).unwrap(),
        }
        let output = caller.invoke(
            &fixture,
            Some(env!("CARGO_BIN_EXE_xtask")),
            false,
            &Cancellation::default(),
        );
        assert!(!output.success(), "{mutation}: {output:?}");
        assert!(
            !caller.trace.exists(),
            "failed admission reached Git staging"
        );
        assert!(!caller.receipt.exists());
        assert!(!caller.just_trace.exists());
        assert!(git(&fixture.root, &["diff", "--cached", "--name-only"]).is_empty());
    }
}
#[test]
fn legacy_repair_configured_invalid_controller_never_falls_back_or_reaches_snapshot() {
    for kind in ["empty", "relative", "missing", "directory", "nonexec"] {
        let fixture = legacy_repair_fixture();
        let caller = RepairCaller::new(&fixture);
        let path = fixture.temp.path().join("configured owner");
        let selected = match kind {
            "empty" => String::new(),
            "relative" => "owner-relative".into(),
            "directory" => {
                fs::create_dir(&path).unwrap();
                path.display().to_string()
            }
            "nonexec" => {
                fs::write(&path, "not executable").unwrap();
                fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
                path.display().to_string()
            }
            _ => path.display().to_string(),
        };
        let output = caller.invoke(&fixture, Some(&selected), false, &Cancellation::default());
        assert!(!output.success(), "{kind}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr.bytes_retained).contains("absolute executable")
        );
        assert!(!caller.just_trace.exists());
        assert!(!caller.trace.exists());
        assert!(!caller.receipt.exists());
    }
}
#[test]
fn cancelled_legacy_repair_owner_does_not_stage_or_export_pass_and_cleans_owned_fixture_tree() {
    let fixture = legacy_repair_fixture();
    let caller = RepairCaller::new(&fixture);
    let delayed = fixture.temp.path().join("delayed configured owner");
    fs::write(&delayed, "#!/bin/bash\nset -euo pipefail\n/bin/sleep 30 & child=$!\ntrap 'kill \"$child\" 2>/dev/null; wait \"$child\" 2>/dev/null; exit 143' TERM INT\nprintf started > \"$JUST_TRACE\"\nwait \"$child\"\n").unwrap();
    fs::set_permissions(&delayed, fs::Permissions::from_mode(0o700)).unwrap();
    let cancellation = Cancellation::default();
    let trigger = cancellation.clone();
    let output = std::thread::scope(|scope| {
        scope.spawn(|| {
            let deadline = std::time::Instant::now() + Duration::from_secs(4);
            while !caller.just_trace.exists() {
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(10));
            }
            trigger.cancel();
        });
        caller.invoke(
            &fixture,
            Some(delayed.to_str().unwrap()),
            false,
            &cancellation,
        )
    });
    assert_eq!(output.outcome, process::Outcome::Cancelled, "{output:?}");
    assert!(!caller.trace.exists());
    assert!(!caller.receipt.exists());
    assert!(git(&fixture.root, &["diff", "--cached", "--name-only"]).is_empty());
}

#[test]
fn legacy_repair_valid_controller_failure_under_conditional_call_stops_before_staging() {
    let fixture = legacy_repair_fixture();
    let caller = RepairCaller::new(&fixture);
    let failed_owner = fixture.temp.path().join("failed configured owner");
    fs::write(&failed_owner, "#!/bin/sh\nexit 7\n").unwrap();
    fs::set_permissions(&failed_owner, fs::Permissions::from_mode(0o700)).unwrap();
    let output = caller.invoke(
        &fixture,
        Some(failed_owner.to_str().unwrap()),
        false,
        &Cancellation::default(),
    );
    assert!(!output.success(), "{output:?}");
    assert!(!caller.just_trace.exists());
    assert!(
        !caller.trace.exists(),
        "failure under Bash conditional reached git add"
    );
    assert!(!caller.receipt.exists());
    assert!(git(&fixture.root, &["diff", "--cached", "--name-only"]).is_empty());
}

#[test]
fn legacy_repair_changed_frozen_controller_is_rejected_before_admission_and_staging() {
    let fixture = legacy_repair_fixture();
    let mut caller = RepairCaller::new(&fixture);
    caller.controller_mutation = true;
    let owner = fixture.temp.path().join("original admitted controller");
    fs::write(&owner, "#!/bin/sh\nexit 0\n").unwrap();
    fs::set_permissions(&owner, fs::Permissions::from_mode(0o700)).unwrap();
    let output = caller.invoke(
        &fixture,
        Some(owner.to_str().unwrap()),
        false,
        &Cancellation::default(),
    );
    assert!(!output.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr.bytes_retained)
            .contains("frozen workload automation controller changed after admission")
    );
    assert!(!caller.trace.exists());
    assert!(!caller.receipt.exists());
    assert!(!caller.just_trace.exists());
    assert!(git(&fixture.root, &["diff", "--cached", "--name-only"]).is_empty());
}

#[test]
fn legacy_repair_controller_initialization_is_not_applicable_to_three_build_modes() {
    for mode in ["repair-build", "verify-build", "pinned-build"] {
        let fixture = legacy_repair_fixture();
        let mut caller = RepairCaller::new(&fixture);
        caller.harness_mode = mode;
        // An unused empty configured selector must not impose a new admission
        // requirement on the protected-receipt or selected-source paths.
        let output = caller.invoke(&fixture, Some(""), false, &Cancellation::default());
        assert!(output.success(), "{mode}: {output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout.bytes_retained)
                .contains("repair-snapshot-not-applicable")
        );
        assert!(!caller.just_trace.exists());
        assert!(!caller.trace.exists());
        assert!(!caller.receipt.exists());
        assert!(!fixture.temp.path().join("controller.marker").exists());
    }
}

#[test]
fn normal_verify_controller_selection_requires_actual_owner_or_trusted_bootstrap() {
    for configured in [Some(""), Some(env!("CARGO_BIN_EXE_xtask")), None] {
        let fixture = legacy_repair_fixture();
        let mut caller = RepairCaller::new(&fixture);
        caller.harness_mode = "verify";
        let output = caller.invoke(&fixture, configured, false, &Cancellation::default());
        assert_eq!(
            output.success(),
            configured != Some(""),
            "{configured:?}: {output:?}"
        );
        if configured == Some("") {
            assert!(
                String::from_utf8_lossy(&output.stderr.bytes_retained)
                    .contains("must be an absolute executable")
            );
        } else {
            assert!(
                String::from_utf8_lossy(&output.stdout.bytes_retained)
                    .contains("repair-snapshot-not-applicable")
            );
        }
        assert_eq!(caller.just_trace.exists(), configured.is_none());
        assert!(!caller.trace.exists());
        assert!(!caller.receipt.exists());
        assert!(!fixture.temp.path().join("controller.marker").exists());
    }
}
