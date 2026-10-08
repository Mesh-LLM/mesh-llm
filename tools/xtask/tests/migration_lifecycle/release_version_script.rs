//! Actual release version updater in tracked, isolated fixtures. No publication.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/release-version.sh"
));

fn function(name: &str) -> String {
    let marker = format!("{name}() {{\n");
    let (_, remainder) = SOURCE
        .split_once(&marker)
        .unwrap_or_else(|| panic!("maintained {name}"));
    let (body, _) = remainder.split_once("\n}\n").expect("function boundary");
    format!("{marker}{body}\n}}\n")
}

fn tool(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|p| p.join(name))
        .find(|p| p.is_file() && fs::metadata(p).unwrap().permissions().mode() & 0o111 != 0)
        .unwrap_or_else(|| panic!("version fixture requires {name}"))
        .canonicalize()
        .unwrap()
}

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("version fixture with spaces");
        fs::create_dir_all(root.join("bin")).unwrap();
        for name in ["bash", "perl", "cat", "grep", "sort", "node", "dirname"] {
            std::os::unix::fs::symlink(tool(name), root.join("bin").join(name)).unwrap();
        }
        let git = std::env::var_os("MIGRATION_TEST_GIT")
            .map(PathBuf::from)
            .unwrap_or_else(|| tool("git"));
        assert!(git.is_absolute() && git.is_file());
        std::os::unix::fs::symlink(git, root.join("bin/git")).unwrap();
        let fixture = Self {
            _temporary: temporary,
            root,
        };
        fixture.write("bin/cargo", "#!/bin/sh\nset -eu\n[ \"$#\" = 3 ] && [ \"$1\" = metadata ] && [ \"$2\" = --format-version ] && [ \"$3\" = 1 ] || exit 91\nprintf '%s\\n' \"$*\" >> \"$FIXTURE_ROOT/cargo.calls\"\n");
        fs::set_permissions(
            fixture.root.join("bin/cargo"),
            fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        fixture
    }

    fn write(&self, relative: &str, source: &str) {
        let path = self.root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, source).unwrap();
    }

    fn invoke(&self, executable: &str, args: &[&str]) -> (bool, String, String) {
        let environment = BTreeMap::from([
            ("PATH".into(), Value::Public(self.root.join("bin").into())),
            ("HOME".into(), Value::Public(self.root.clone().into())),
            (
                "FIXTURE_ROOT".into(),
                Value::Public(self.root.clone().into()),
            ),
            ("GIT_CONFIG_NOSYSTEM".into(), Value::Public("1".into())),
            ("GIT_MASTER".into(), Value::Public("1".into())),
            ("GIT_OPTIONAL_LOCKS".into(), Value::Public("0".into())),
        ]);
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: self.root.join("bin").join(executable),
                cwd: self.root.clone(),
                environment,
                arguments: args
                    .iter()
                    .map(|arg| Value::Public((*arg).into()))
                    .collect(),
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 32768,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(32768),
                stderr: NonZeroUsize::new(32768),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(
            report.process.failure.is_none() && report.process.cleanup.complete,
            "{report:?}"
        );
        (
            report.process.status.unwrap().success(),
            String::from_utf8(report.stdout.unwrap().as_bytes().to_vec()).unwrap(),
            String::from_utf8(report.stderr.unwrap().as_bytes().to_vec()).unwrap(),
        )
    }

    fn bash(&self, script: &str) -> (bool, String, String) {
        self.invoke("bash", &["-euo", "pipefail", "-c", script])
    }

    fn track(&self) {
        for args in [&["init", "-q"][..], &["add", "."][..]] {
            let (ok, _, error) = self.invoke("git", args);
            assert!(ok, "{error}");
        }
    }

    fn resolve(&self, logical: &str) -> (bool, String, String) {
        let script = format!(
            "{}REPO_ROOT=\"$FIXTURE_ROOT\"\nresolve_product_path '{logical}'\n",
            function("resolve_product_path")
        );
        self.bash(&script)
    }

    fn known_versions(&self, file: &str, version: &str) -> (bool, String, String) {
        let script = format!(
            "{}update_known_mesh_versions '{file}' '{version}'\n",
            function("update_known_mesh_versions")
        );
        self.bash(&script)
    }
}

fn known_source() -> String {
    let logical = SOURCE
        .lines()
        .find_map(|line| {
            line.strip_prefix("known_versions_logical=\"")
                .and_then(|s| s.strip_suffix('"'))
        })
        .unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let present = [
        logical.to_owned(),
        format!("mesh/{logical}"),
        format!("skippy/{logical}"),
    ]
    .into_iter()
    .map(|p| root.join(p))
    .filter(|p| p.is_file())
    .collect::<Vec<_>>();
    assert_eq!(present.len(), 1, "unambiguous known versions owner");
    fs::read_to_string(&present[0]).unwrap()
}

#[test]
fn discovered_tracked_manifests_update_only_versioned_path_dependencies_in_both_layouts() {
    let discovery = SOURCE
        .split_once("manifests=()")
        .unwrap()
        .1
        .split_once("versioned_files=()")
        .unwrap()
        .0;
    for owners in [vec!["crates"], vec!["mesh/crates", "skippy/crates"]] {
        let fixture = Fixture::new();
        for owner in owners.iter().copied().chain(["tools"]) {
            fixture.write(&format!("{owner}/fixture/Cargo.toml"), "[dependencies]\nlocal = { path = \"../local\", version = \"0.76.1\" }\nexternal = \"0.76.1\"\n");
        }
        fixture.track();
        fixture.write("crates/untracked/Cargo.toml", "untracked sentinel 0.76.1\n");
        let script = format!(
            "{}REPO_ROOT=\"$FIXTURE_ROOT\"\nmanifests=(){discovery}\nfor manifest in \"${{manifests[@]}}\"; do update_versioned_path_dependency_versions \"$REPO_ROOT/$manifest\" 0.77.0; done\n",
            function("update_versioned_path_dependency_versions")
        );
        let (ok, _, error) = fixture.bash(&script);
        assert!(ok, "{error}");
        for owner in owners.iter().copied().chain(["tools"]) {
            let updated =
                fs::read_to_string(fixture.root.join(format!("{owner}/fixture/Cargo.toml")))
                    .unwrap();
            assert!(updated.contains("path = \"../local\", version = \"0.77.0\""));
            assert!(updated.contains("external = \"0.76.1\""));
        }
        assert_eq!(
            fs::read_to_string(fixture.root.join("crates/untracked/Cargo.toml")).unwrap(),
            "untracked sentinel 0.76.1\n"
        );
    }
}

#[test]
fn maintained_known_version_owner_defines_the_function_the_updater_edits() {
    assert!(known_source().contains("fn known_mesh_llm_versions()"));
}

#[test]
fn new_known_version_is_prepended_once_and_a_second_update_is_byte_preserving() {
    let fixture = Fixture::new();
    fixture.write("setting_schema.rs", &known_source());
    let (ok, _, error) = fixture.known_versions("setting_schema.rs", "99.99.99-rc1");
    assert!(ok, "{error}");
    let once = fs::read(fixture.root.join("setting_schema.rs")).unwrap();
    assert!(String::from_utf8_lossy(&once).contains("\"99.99.99-rc1\","));
    assert!(
        fixture
            .known_versions("setting_schema.rs", "99.99.99-rc1")
            .0
    );
    assert_eq!(
        fs::read(fixture.root.join("setting_schema.rs")).unwrap(),
        once
    );
}

#[test]
fn wrong_known_version_file_is_rejected_without_writing() {
    let fixture = Fixture::new();
    fixture.write("elsewhere.rs", "// no version list here\n");
    let (ok, _, error) = fixture.known_versions("elsewhere.rs", "99.99.99-rc1");
    assert!(
        !ok && error.contains("known_mesh_llm_versions()"),
        "{error}"
    );
    assert_eq!(
        fs::read_to_string(fixture.root.join("elsewhere.rs")).unwrap(),
        "// no version list here\n"
    );
}

#[test]
fn release_sidecars_resolve_each_supported_layout() {
    let logical = "sdk/kotlin/build.gradle.kts";
    for prefix in ["", "mesh/", "skippy/"] {
        let fixture = Fixture::new();
        let relative = format!("{prefix}{logical}");
        fixture.write(&relative, "sidecar sentinel\n");
        let (ok, output, error) = fixture.resolve(logical);
        assert!(ok, "{error}");
        assert_eq!(output.trim(), relative);
        assert_eq!(
            fs::read_to_string(fixture.root.join(relative)).unwrap(),
            "sidecar sentinel\n"
        );
    }
}

#[test]
fn ambiguous_and_missing_release_sidecars_fail_before_writing() {
    let logical = "sdk/kotlin/build.gradle.kts";
    let fixture = Fixture::new();
    let (ok, _, error) = fixture.resolve(logical);
    assert!(!ok && error.contains("missing required file"), "{error}");
    assert!(!fixture.root.join(logical).exists());
    for prefix in ["", "mesh/"] {
        fixture.write(&format!("{prefix}{logical}"), "unchanged\n");
    }
    let (ok, _, error) = fixture.resolve(logical);
    assert!(!ok && error.contains("ambiguous source layout"), "{error}");
    for prefix in ["", "mesh/"] {
        assert_eq!(
            fs::read_to_string(fixture.root.join(format!("{prefix}{logical}"))).unwrap(),
            "unchanged\n"
        );
    }
}

#[path = "release_version_script/relocated.rs"]
mod relocated;
