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
