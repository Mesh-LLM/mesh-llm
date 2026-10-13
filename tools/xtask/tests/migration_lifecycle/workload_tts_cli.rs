//! Finite local source repositories and fake native producers; never real native/model execution.
#![cfg(unix)]
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, FileTimes},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::{Duration, UNIX_EPOCH},
};
fn run(
    cwd: &Path,
    executable: &Path,
    args: &[String],
    environment: BTreeMap<std::ffi::OsString, Value>,
) -> process::ProcessReport {
    let limits = Limits {
        execution: Duration::from_secs(10),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 131072,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let spec = ProcessSpec {
        executable: executable.into(),
        arguments: args.iter().map(|s| Value::Public(s.into())).collect(),
        cwd: cwd.into(),
        environment,
    };
    let output = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(output.cleanup.complete, "{output:?}");
    output
}
fn git(cwd: &Path, args: &[&str]) -> String {
    let output = run(
        cwd,
        Path::new("/usr/bin/git"),
        &args.iter().map(|s| (*s).into()).collect::<Vec<_>>(),
        [
            ("GIT_MASTER", "1"),
            ("GIT_CONFIG_NOSYSTEM", "1"),
            ("GIT_CONFIG_GLOBAL", "/dev/null"),
            ("GIT_AUTHOR_NAME", "fixture"),
            ("GIT_AUTHOR_EMAIL", "fixture@example.invalid"),
            ("GIT_COMMITTER_NAME", "fixture"),
            ("GIT_COMMITTER_EMAIL", "fixture@example.invalid"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect(),
    );
    assert!(output.success(), "{output:?}");
    String::from_utf8(output.stdout.bytes_retained)
        .unwrap()
        .trim()
        .into()
}
fn executable(path: &Path, body: &str) {
    fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
fn sha(path: &Path) -> String {
    hex::encode(Sha256::digest(fs::read(path).unwrap()))
}
fn write_prepared_patch_recipe(root: &Path) -> String {
    let patches = root.join("third_party/llama.cpp/patches");
    for (name, bytes) in [
        ("0001-base.patch", "base\n"),
        ("model_support/0001-test-support.patch", "support\n"),
        ("model_support/series", "0001-test-support.patch\r\n"),
        ("generated/0001-family-test.patch", "generated\n"),
        ("generated/series", "0001-family-test.patch\n"),
    ] {
        fs::write(patches.join(name), bytes).unwrap();
    }
    // Independent known transcript for this finite core/support/generated recipe.
    "3800406abecfae8bd783a773345793e24506e649d42731e3c1da6c6d2d113d31".into()
}

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    head: String,
    native_head: String,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let canonical = directory.path().canonicalize().unwrap();
        let root = canonical.as_path();
        for relative in [
            ".deps/llama.cpp",
            "third_party/llama.cpp/patches/model_support",
            "third_party/llama.cpp/patches/generated",
            "closure/native/bin",
            "closure/cargo/debug/deps",
            "bin",
            "work",
        ] {
            fs::create_dir_all(root.join(relative)).unwrap();
        }
        git(&root.join(".deps/llama.cpp"), &["init", "-q"]);
        git(
            &root.join(".deps/llama.cpp"),
            &["commit", "-q", "--allow-empty", "-m", "native fixture"],
        );
        let native_head = git(&root.join(".deps/llama.cpp"), &["rev-parse", "HEAD"]);
        fs::write(
            root.join("third_party/llama.cpp/upstream.txt"),
            "a".repeat(40),
        )
        .unwrap();
        let patch_digest = write_prepared_patch_recipe(root);
        for (name, value) in [
            (".mesh-llm-patched-sha", native_head.as_str()),
            (".mesh-llm-upstream-sha", &"a".repeat(40)),
            (".mesh-llm-prepare-schema", "5"),
            (".mesh-llm-patch-digest", patch_digest.as_str()),
        ] {
            fs::write(root.join(".deps/llama.cpp").join(name), value).unwrap();
        }
        fs::write(
            root.join(".gitignore"),
            ".deps/\nclosure/\nwork/\nbin/\nfixture.wav\nmodel.gguf\nprojector.gguf\n",
        )
        .unwrap();
        fs::write(
            root.join("Justfile"),
            "# finite stub intercepted by fake Just\n",
        )
        .unwrap();
        fs::write(root.join("Cargo.toml"), "# not built\n").unwrap();
        git(root, &["init", "-q"]);
        git(root, &["add", "."]);
        git(root, &["commit", "-q", "-m", "fixture source"]);
        let head = git(root, &["rev-parse", "HEAD"]);
        let fixture = Self {
            _directory: directory,
            root: canonical,
            head,
            native_head,
        };
        fixture.producers();
        fixture
    }
    fn root(&self) -> &Path {
        &self.root
    }
    fn producers(&self) {
        let root = self.root();
        let mut wav = b"RIFF".to_vec();
        wav.extend(44_u32.to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16_u32.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(1_u16.to_le_bytes());
        wav.extend(8000_u32.to_le_bytes());
        wav.extend(16000_u32.to_le_bytes());
        wav.extend(2_u16.to_le_bytes());
        wav.extend(16_u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend(8_u32.to_le_bytes());
        wav.extend(
            [1000_i16, -1000, 1000, -1000]
                .into_iter()
                .flat_map(i16::to_le_bytes),
        );
        fs::write(root.join("fixture.wav"), wav).unwrap();
        for name in ["model.gguf", "projector.gguf"] {
            fs::write(root.join(name), b"GGUF finite model").unwrap();
        }
        let candidate = "if [ \"${FIXTURE_FAIL:-}\" = cancel ]; then /bin/sleep 30 & child=$!; trap 'kill \"$child\" 2>/dev/null; wait \"$child\" 2>/dev/null; exit 143' TERM INT; printf '%s\n' \"$child\" > \"$FIXTURE_ROOT/work/descendant.pid\"; wait \"$child\"; exit 0; fi\nprintf '%s\\n' \"$@\" > \"$FIXTURE_ROOT/work/candidate.argv\"\nprintf '%s\\n' \"$SKIPPY_TTS_ORACLE_PROMPT\" \"$SKIPPY_TTS_ORACLE_SEED\" \"$SKIPPY_TTS_ORACLE_TOP_K\" \"$SKIPPY_TTS_ORACLE_TOP_P\" \"$SKIPPY_TTS_ORACLE_MAX_FRAMES\" \"$LLAMA_STAGE_BACKEND\" > \"$FIXTURE_ROOT/work/candidate.env\"\nif [ \"${FIXTURE_FAIL:-}\" = candidate ]; then exit 7; fi\nif [ \"${FIXTURE_FAIL:-}\" = oversized ]; then exec /usr/bin/head -c 8388609 /dev/zero; fi\nif [ \"${FIXTURE_FAIL:-}\" != filtered ]; then /bin/cp \"$FIXTURE_ROOT/fixture.wav\" \"$SKIPPY_TTS_ORACLE_CANDIDATE_WAV\"; fi\nprintf '%s\\n' 'token authorization candidate raw'";
        executable(&root.join("bin/just"), candidate);
        executable(
            &root.join("closure/cargo/debug/deps/skippy_serving-fixture"),
            candidate,
        );
        for name in [
            "skippy",
            "skippy-package-builder",
            "skippy-correctness",
            "skippy-topology-plan",
        ] {
            executable(&root.join("closure/cargo/debug").join(name), "exit 0");
        }
        for name in ["llama-server", "llama-completion"] {
            executable(&root.join("closure/native/bin").join(name), "exit 0");
        }
        executable(
            &root.join("closure/native/bin/llama-tts"),
            "printf '%s\\n' \"$@\" > \"$FIXTURE_ROOT/work/oracle.argv\"\nif [ \"${FIXTURE_FAIL:-}\" = oracle ]; then exit 8; fi\nwhile [ $# -gt 0 ]; do if [ \"$1\" = --output ]; then shift; case \"${FIXTURE_FAIL:-}\" in oracle-fifo) /usr/bin/mkfifo \"$1\" ;; oracle-directory) /bin/mkdir \"$1\" ;; *) /bin/cp \"$FIXTURE_ROOT/fixture.wav\" \"$1\" ;; esac; fi; shift; done\nprintf '%s\\n' 'token authorization oracle raw'",
        );
        let stamp = format!(
            "stamp-version=3\npatched-sha={}\nbackend=cpu\nlink-mode=static\nggml-native=OFF\ncmake-arg=-DGGML_NATIVE=OFF\ncmake-arg=-DGGML_METAL=OFF\ncmake-arg=-DLLAMA_BUILD_TOOLS=ON\n",
            self.native_head
        );
        fs::write(root.join("closure/native/.mesh-llm-build-stamp"), stamp).unwrap();
        let paths = [
            ("candidate", "cargo/debug/skippy"),
            ("model_package", "cargo/debug/skippy-package-builder"),
            ("correctness", "cargo/debug/skippy-correctness"),
            ("topology_plan", "cargo/debug/skippy-topology-plan"),
            ("native_stamp", "native/.mesh-llm-build-stamp"),
            ("llama-server", "native/bin/llama-server"),
            ("llama-completion", "native/bin/llama-completion"),
            ("llama-tts", "native/bin/llama-tts"),
            ("test_binary", "cargo/debug/deps/skippy_serving-fixture"),
        ];
        let mut files = serde_json::Map::new();
        for (key, path) in paths {
            let actual = root.join("closure").join(path);
            let seconds = if key == "native_stamp" { 100 } else { 200 };
            fs::File::options()
                .write(true)
                .open(&actual)
                .unwrap()
                .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(seconds)))
                .unwrap();
            files.insert(key.into(), json!({"path":path,"sha256":sha(&actual)}));
        }
        fs::write(root.join("closure/producer.json"),serde_json::to_vec(&json!({"schema_version":1,"source":{"head":self.head,"worktree_sha256":hex::encode(Sha256::digest(b""))},"files":files})).unwrap()).unwrap();
    }
    fn invoke(&self, prebuilt: bool, fail: &str) -> process::ProcessReport {
        self.invoke_cancel(prebuilt, fail, &Cancellation::default())
    }
    fn invoke_cancel(
        &self,
        prebuilt: bool,
        fail: &str,
        cancellation: &Cancellation,
    ) -> process::ProcessReport {
        let root = self.root();
        let mut env: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", root.join("bin").display()),
            ),
            ("FIXTURE_ROOT", root.to_string_lossy().into_owned()),
            ("FIXTURE_FAIL", fail.into()),
            (
                "LLAMA_STAGE_BUILD_DIR",
                root.join("closure/native").to_string_lossy().into_owned(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        if prebuilt {
            for (key, path) in [
                ("SKIPPY_WORKLOAD_PRODUCER_MANIFEST", "closure/producer.json"),
                ("SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR", "closure/cargo/debug"),
                ("SKIPPY_WORKLOAD_NATIVE_BUILD_DIR", "closure/native"),
            ] {
                env.insert(key.into(), Value::Public(root.join(path).into()));
            }
        }
        if fail == "partial" {
            env.insert(
                "SKIPPY_WORKLOAD_NATIVE_BUILD_DIR".into(),
                Value::Public(root.join("closure/native").into()),
            );
        }
        if fail == "empty-manifest" {
            env.insert(
                "SKIPPY_WORKLOAD_PRODUCER_MANIFEST".into(),
                Value::Public("".into()),
            );
        }
        let args = vec![
            "automation".into(),
            "workload-tts-oracle".into(),
            "--root".into(),
            root.into(),
            "--oracle-cli".into(),
            root.join("closure/native/bin/llama-tts").into(),
            "--model-path".into(),
            root.join("model.gguf").into(),
            "--projector-path".into(),
            root.join("projector.gguf").into(),
            "--model".into(),
            "tts-fixture".into(),
            "--layer-end".into(),
            "28".into(),
            "--work-dir".into(),
            root.join("work").into(),
        ];
        // Preserve path bytes while building the existing ProcessSpec argv.
        let args: Vec<std::ffi::OsString> = args;
        let limits = Limits {
            execution: Duration::from_secs(15),
            graceful_shutdown: Duration::from_secs(4),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 131072,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let output = process::supervise(
            &ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                arguments: args.into_iter().map(Value::Public).collect(),
                cwd: root.into(),
                environment: env,
            },
            &limits,
            cancellation,
            OutputFiles::default(),
        )
        .unwrap();
        assert!(output.cleanup.complete, "{output:?}");
        output
    }
}
#[test]
fn deterministic_tts_cli_uses_verified_prebuilt_or_normal_just_and_binds_full_pcm_receipt() {
    for prebuilt in [false, true] {
        let fixture = Fixture::new();
        let output = fixture.invoke(prebuilt, "");
        assert!(output.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stdout.bytes_retained)
                .starts_with("speech_synthesis local-monolithic oracle passed: ")
        );
        let root = fixture.root();
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("work/tts-oracle-result.json")).unwrap())
                .unwrap();
        assert_eq!(result["status"], "pass");
        assert_eq!(result["pinned_patch_sha"], fixture.native_head);
        assert_eq!(result["max_frames"], 512);
        assert_eq!(result["metrics"]["sample_count"], 4);
        assert_eq!(result["metrics"]["relative_rms_error"], 0.0);
        assert_eq!(result["metrics"]["waveform_cosine"], 1.0);
        assert_eq!(
            result["candidate_wav_sha256"],
            sha(&root.join("work/tts-candidate.wav"))
        );
        assert_eq!(
            result["oracle_wav_sha256"],
            sha(&root.join("work/tts-monolithic-oracle.wav"))
        );
        assert_eq!(
            fs::read_to_string(root.join("work/candidate.env")).unwrap(),
            "The mesh is ready.\n7\n20\n0.8\n512\ncpu\n"
        );
        let args = fs::read_to_string(root.join("work/candidate.argv")).unwrap();
        let candidate_arguments: Vec<_> = args.lines().collect();
        for required in [
            "frontend::tests::tts_oracle::deterministic_tts_candidate_when_fixture_is_set",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ] {
            assert_eq!(
                candidate_arguments
                    .iter()
                    .filter(|argument| **argument == required)
                    .count(),
                1,
                "candidate must select the exact deterministic test: {args}"
            );
        }
        assert_eq!(args.contains("cargo\ntest"), !prebuilt);
        if !prebuilt {
            assert!(args.contains("-p\nskippy-serving\n--lib\n"), "{args}");
            assert!(args.starts_with(&format!(
                "--justfile\n{}\nwith-lld\ncargo\ntest\n",
                root.join("Justfile").display()
            )));
        }
        let args = fs::read_to_string(root.join("work/oracle.argv")).unwrap();
        assert!(args.contains("--seed\n7\n--top-k\n20\n--top-p\n0.8\n--temp\n1\n--min-p\n0\n--repeat-penalty\n1\n--no-repack\n-c\n2048\n-b\n2048\n-ub\n2048\n-ngl\n0\n"));
        assert!(args.contains("-n\n512\n"));
        assert_eq!(
            fs::read_to_string(root.join("work/tts-candidate-test.log")).unwrap(),
            "token authorization candidate raw\n"
        );
        assert_eq!(
            fs::read_to_string(root.join("work/tts-monolithic-oracle.log")).unwrap(),
            "token authorization oracle raw\n"
        );
    }
}
#[test]
fn failed_or_filtered_tts_producer_cannot_inherit_old_wav_or_pass_record() {
    for fail in ["candidate", "oracle", "filtered", "oversized"] {
        let fixture = Fixture::new();
        let work = fixture.root().join("work");
        for name in [
            "tts-candidate.wav",
            "tts-monolithic-oracle.wav",
            "tts-oracle-result.json",
            "unrelated",
        ] {
            fs::write(work.join(name), "stale pass").unwrap();
        }
        let output = fixture.invoke(true, fail);
        assert!(!output.success(), "{output:?}");
        assert!(output.stdout.bytes_retained.is_empty());
        assert!(!work.join("tts-oracle-result.json").exists());
        assert_eq!(
            fs::read_to_string(work.join("unrelated")).unwrap(),
            "stale pass"
        );
    }
}
#[test]
fn source_and_producer_mutations_fail_before_candidate_or_oracle_execution() {
    for attack in [
        "source",
        "candidate-digest",
        "retired-candidate",
        "native-policy",
        "oracle-policy",
        "test-freshness",
    ] {
        let fixture = Fixture::new();
        let root = fixture.root();
        match attack {
            "source" => fs::write(root.join("Cargo.toml"), "changed source").unwrap(),
            "candidate-digest" => {
                fs::write(root.join("closure/cargo/debug/skippy"), "changed candidate").unwrap()
            }
            "retired-candidate" => fs::rename(
                root.join("closure/cargo/debug/skippy"),
                root.join("closure/cargo/debug/skippy-server"),
            )
            .unwrap(),
            "native-policy" | "oracle-policy" => {
                let stamp = root.join("closure/native/.mesh-llm-build-stamp");
                let text = fs::read_to_string(&stamp).unwrap();
                let text = if attack == "oracle-policy" {
                    text.replace("cmake-arg=-DLLAMA_BUILD_TOOLS=ON\n", "")
                } else {
                    text.replace("ggml-native=OFF", "ggml-native=ON")
                };
                fs::write(stamp, text).unwrap();
            }
            _ => {
                fs::File::options()
                    .write(true)
                    .open(root.join("closure/cargo/debug/deps/skippy_serving-fixture"))
                    .unwrap()
                    .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(100)))
                    .unwrap();
            }
        }
        let output = fixture.invoke(true, "");
        assert!(!output.success(), "{attack}: {output:?}");
        assert!(!root.join("work/candidate.argv").exists());
        assert!(!root.join("work/oracle.argv").exists());
        assert!(!root.join("work/tts-oracle-result.json").exists());
    }
}

#[test]
fn partial_or_empty_prebuilt_configuration_never_falls_back_to_just() {
    for fail in ["partial", "empty-manifest"] {
        let fixture = Fixture::new();
        let output = fixture.invoke(false, fail);
        assert!(!output.success(), "{output:?}");
        assert!(!fixture.root().join("work/candidate.argv").exists());
        assert!(!fixture.root().join("work/oracle.argv").exists());
    }
}

#[test]
fn special_oracle_wav_outputs_reject_promptly_without_receipt_or_success_prefix() {
    for mode in ["oracle-fifo", "oracle-directory"] {
        let fixture = Fixture::new();
        let before = std::time::Instant::now();
        let output = fixture.invoke(true, mode);
        assert!(!output.success(), "{mode}: {output:?}");
        assert_eq!(output.outcome, process::Outcome::Exited, "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr.bytes_retained).contains("regular file"),
            "{output:?}"
        );
        assert!(before.elapsed() < Duration::from_secs(5));
        assert!(output.stdout.bytes_retained.is_empty());
        assert!(!fixture.root().join("work/tts-oracle-result.json").exists());
    }
}
#[test]
fn cancelled_tts_child_reaps_owned_descendant_and_preserves_unrelated_files() {
    let fixture = Fixture::new();
    let marker = fixture.root().join("work/descendant.pid");
    fs::write(fixture.root().join("work/unrelated"), "sentinel").unwrap();
    let cancellation = Cancellation::default();
    let trigger = cancellation.clone();
    let output = std::thread::scope(|scope| {
        scope.spawn(|| {
            let deadline = std::time::Instant::now() + Duration::from_secs(4);
            while !marker.exists() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "candidate never started"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            trigger.cancel();
        });
        fixture.invoke_cancel(true, "cancel", &cancellation)
    });
    assert_eq!(output.outcome, process::Outcome::Cancelled, "{output:?}");
    assert!(output.cleanup.complete, "{output:?}");
    let pid: i32 = fs::read_to_string(marker).unwrap().trim().parse().unwrap();
    // SAFETY: signal zero only observes this fixture's recorded descendant.
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
    assert!(!fixture.root().join("work/tts-oracle-result.json").exists());
    assert!(!fixture.root().join("work/oracle.argv").exists());
    assert_eq!(
        fs::read_to_string(fixture.root().join("work/unrelated")).unwrap(),
        "sentinel"
    );
}

#[test]
fn special_native_stamp_inputs_fail_before_producer_launch_without_waiting_for_fifo() {
    use std::os::unix::ffi::OsStrExt as _;
    for mode in ["fifo", "directory"] {
        let fixture = Fixture::new();
        let stamp = fixture.root().join("closure/native/.mesh-llm-build-stamp");
        fs::remove_file(&stamp).unwrap();
        if mode == "fifo" {
            let name = std::ffi::CString::new(stamp.as_os_str().as_bytes()).unwrap();
            // SAFETY: valid NUL-terminated fixture path; creates only our temporary FIFO.
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        } else {
            fs::create_dir(&stamp).unwrap();
        }
        let before = std::time::Instant::now();
        let output = fixture.invoke(true, "");
        assert!(!output.success(), "{mode}: {output:?}");
        assert_eq!(output.outcome, process::Outcome::Exited, "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr.bytes_retained).contains("regular file"),
            "{output:?}"
        );
        assert!(before.elapsed() < Duration::from_secs(5));
        assert!(!fixture.root().join("work/candidate.argv").exists());
        assert!(!fixture.root().join("work/oracle.argv").exists());
        assert!(!fixture.root().join("work/tts-oracle-result.json").exists());
    }
}

#[test]
fn missing_stale_or_metal_native_stamp_rejects_before_tts_producers_or_receipt() {
    for mode in ["missing", "stale-patched-sha", "metal"] {
        let fixture = Fixture::new();
        let stamp = fixture.root().join("closure/native/.mesh-llm-build-stamp");
        let admitted = fs::read_to_string(&stamp).unwrap();
        match mode {
            "missing" => fs::remove_file(&stamp).unwrap(),
            "stale-patched-sha" => fs::write(
                &stamp,
                admitted.replace(
                    &format!("patched-sha={}", fixture.native_head),
                    &format!("patched-sha={}", "f".repeat(40)),
                ),
            )
            .unwrap(),
            _ => fs::write(&stamp, admitted.replace("backend=cpu", "backend=metal")).unwrap(),
        }
        let before = std::time::Instant::now();
        let output = fixture.invoke(true, "");
        assert!(!output.success(), "{mode}: {output:?}");
        assert_eq!(output.outcome, process::Outcome::Exited, "{output:?}");
        assert!(before.elapsed() < Duration::from_secs(5));
        assert!(output.stdout.bytes_retained.is_empty());
        assert!(!fixture.root().join("work/candidate.argv").exists());
        assert!(!fixture.root().join("work/oracle.argv").exists());
        assert!(!fixture.root().join("work/tts-oracle-result.json").exists());
        if mode != "missing" {
            assert!(
                String::from_utf8_lossy(&output.stderr.bytes_retained)
                    .contains("pinned static CPU policy"),
                "{output:?}"
            );
        }
    }
}
