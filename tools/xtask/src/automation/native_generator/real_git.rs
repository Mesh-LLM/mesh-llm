use super::*;

struct OwnedTempDir(tempfile::TempDir);

struct GitFixture {
    root: OwnedTempDir,
    executable: PathBuf,
    environment: BTreeMap<OsString, Value>,
}

fn executable_file(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(windows)]
    {
        path.extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("exe"))
    }
}

impl GitFixture {
    fn environment(&self) -> BTreeMap<OsString, Value> {
        self.environment
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    Value::Public(value) => Value::Public(value.clone()),
                    Value::Secret(value) => Value::Secret(value.clone()),
                };
                (key.clone(), value)
            })
            .collect()
    }
    fn new() -> Self {
        let root = OwnedTempDir(tempfile::tempdir().unwrap());
        let executable = match std::env::var_os("MIGRATION_TEST_GIT") {
            Some(path) => {
                let path = PathBuf::from(path);
                assert!(
                    path.is_absolute() && executable_file(&path),
                    "MIGRATION_TEST_GIT must name an absolute Git executable"
                );
                path.canonicalize().unwrap()
            }
            None => {
                let path = std::env::var_os("PATH")
                    .expect("Git discovery requires PATH or MIGRATION_TEST_GIT");
                std::env::split_paths(&path)
                    .map(|directory| directory.join(format!("git{}", std::env::consts::EXE_SUFFIX)))
                    .find(|candidate| executable_file(candidate))
                    .expect("install Git or set MIGRATION_TEST_GIT to its absolute executable")
                    .canonicalize()
                    .unwrap()
            }
        };
        let mut environment: BTreeMap<OsString, Value> =
            ["PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"]
                .into_iter()
                .filter_map(|key| {
                    std::env::var_os(key).map(|value| (key.into(), Value::Public(value)))
                })
                .collect();
        for (key, value) in [
            ("HOME", root.0.path().as_os_str().to_owned()),
            ("USERPROFILE", root.0.path().as_os_str().to_owned()),
            ("GIT_CONFIG_NOSYSTEM", "1".into()),
            (
                "GIT_CONFIG_GLOBAL",
                root.0.path().join("empty-config").into_os_string(),
            ),
            ("GIT_TERMINAL_PROMPT", "0".into()),
            ("LC_ALL", "C".into()),
        ] {
            environment.insert(key.into(), Value::Public(value));
        }
        std::fs::write(root.0.path().join("empty-config"), b"").unwrap();
        Self {
            root,
            executable,
            environment,
        }
    }

    fn command(&self, arguments: &[&str]) -> RawBytes {
        let spec = ProcessSpec {
            executable: self.executable.clone(),
            arguments: arguments
                .iter()
                .map(|value| Value::Public((*value).into()))
                .collect(),
            cwd: self.root.0.path().to_owned(),
            environment: self.environment(),
        };
        let limits = Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_millis(100),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = process::supervise_raw(
            &spec,
            &limits,
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: Some(std::num::NonZeroUsize::new(1024 * 1024).unwrap()),
                stderr: None,
            },
        )
        .unwrap();
        assert!(
            report.process.success(),
            "Git fixture command failed: {:?}",
            report.process
        );
        report.stdout.unwrap()
    }
}

#[test]
fn real_git_diff_and_encoded_artifact_match_independent_bytes() {
    let fixture = GitFixture::new();
    let source = fixture.root.0.path();
    std::fs::create_dir_all(source.join("src/models")).unwrap();
    std::fs::write(source.join(".gitattributes"), b"* -text\n").unwrap();
    std::fs::write(source.join("src/models/text.cpp"), b"base\r\n").unwrap();
    std::fs::write(source.join("src/models/binary.cpp"), b"base\0bytes\n").unwrap();
    std::fs::write(source.join("outside.cpp"), b"base\n").unwrap();
    fixture.command(&["init", "--template=", "--initial-branch=fixture"]);
    fixture.command(&["config", "user.name", "Fixture"]);
    fixture.command(&["config", "user.email", "fixture@example.invalid"]);
    fixture.command(&["config", "core.hooksPath", "disabled-hooks"]);
    fixture.command(&["add", ".gitattributes", "src/models", "outside.cpp"]);
    fixture.command(&["commit", "-m", "base", "--no-gpg-sign"]);
    let base = fixture.command(&["rev-parse", "HEAD"]);
    let base = std::str::from_utf8(base.as_bytes())
        .unwrap()
        .trim()
        .to_owned();
    std::fs::write(source.join("src/models/text.cpp"), b"committed\r\n").unwrap();
    fixture.command(&["add", "src/models/text.cpp"]);
    fixture.command(&["commit", "-m", "second", "--no-gpg-sign"]);
    std::fs::write(
        source.join("src/models/text.cpp"),
        b"token password secret authorization invite\r\n",
    )
    .unwrap();
    std::fs::write(source.join("src/models/binary.cpp"), b"changed\0binary\r\n").unwrap();
    std::fs::write(source.join("outside.cpp"), b"excluded change\n").unwrap();
    let expected = fixture.command(&[
        "diff",
        "--no-ext-diff",
        "--binary",
        "--full-index",
        &base,
        "--",
        "src/models",
    ]);
    let text = std::str::from_utf8(expected.as_bytes()).unwrap();
    assert!(text.contains("GIT binary patch"));
    assert!(text.contains("+token password secret authorization invite\r\n"));
    assert!(text.contains("-base\r\n"));
    assert!(!text.contains("outside.cpp"));
    let options = args::Options {
        git: GitDiff {
            executable: fixture.executable.clone(),
            source_root: source.to_owned(),
            base,
            environment: fixture.environment(),
            max_bytes: std::num::NonZeroUsize::new(1024 * 1024).unwrap(),
            timeout: Duration::from_secs(5),
        },
        output: source.join("combined.patch"),
        shards: None,
    };
    let mail = encode_mail_patch(
        "skippy: generate model-family stage controls",
        expected.as_bytes(),
    )
    .unwrap();

    let actual = capture_diff(&options.git, &Cancellation::default()).unwrap();
    publish(actual.as_bytes(), &options).unwrap();

    assert_eq!(actual.as_bytes(), expected.as_bytes());
    assert_eq!(std::fs::read(options.output).unwrap(), mail.bytes);
}
