//! Finite real-Git revision fixtures for the actual maintained Bash/AWK policy.
use crate::{
    cleanup_owner as ownership,
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

struct Fixture {
    directory: tempfile::TempDir,
    repository: PathBuf,
    scratch: PathBuf,
    git: PathBuf,
    environment: BTreeMap<OsString, OsString>,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        ownership::register(&directory);
        let root = directory.path().canonicalize().unwrap();
        let repository = root.join("repository with spaces");
        let scratch = root.join("private scratch");
        let bin = root.join("bin");
        for child in [&repository, &scratch, &bin] {
            ownership::check(&directory, child).unwrap();
            fs::create_dir(child).unwrap();
        }
        let git = PathBuf::from(
            std::env::var_os("MIGRATION_TEST_GIT")
                .expect("set MIGRATION_TEST_GIT to an absolute Git executable"),
        );
        assert!(git.is_absolute() && git.is_file());
        std::os::unix::fs::symlink(&git, bin.join("git")).unwrap();
        let environment = [
            ("PATH", format!("{}:/usr/bin:/bin", bin.display())),
            ("HOME", root.display().to_string()),
            ("TMPDIR", scratch.display().to_string()),
            ("LC_ALL", "C".into()),
            ("GIT_CONFIG_NOSYSTEM", "1".into()),
            ("GIT_CONFIG_GLOBAL", "/dev/null".into()),
            ("GIT_TERMINAL_PROMPT", "0".into()),
            ("GIT_ALLOW_PROTOCOL", "file".into()),
            ("GIT_CONFIG_COUNT", "0".into()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
        let fixture = Self {
            directory,
            repository,
            scratch,
            git,
            environment,
        };
        fixture.git(&["init", "--quiet", "--template="]);
        fixture
    }
    fn execute(
        &self,
        executable: PathBuf,
        arguments: Vec<String>,
        extra: &[(&str, &str)],
    ) -> Vec<u8> {
        ownership::check(&self.directory, &self.repository).unwrap();
        ownership::check(&self.directory, &self.scratch).unwrap();
        let mut environment: BTreeMap<OsString, Value> = self
            .environment
            .iter()
            .map(|(key, value)| (key.clone(), Value::Public(value.clone())))
            .collect();
        for (key, value) in extra {
            environment.insert((*key).into(), Value::Public((*value).into()));
        }
        let result = process::supervise_raw(
            &ProcessSpec {
                executable,
                cwd: self.repository.clone(),
                environment,
                arguments: arguments
                    .into_iter()
                    .map(|a| Value::Public(a.into()))
                    .collect(),
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert_eq!(
            result.process.outcome,
            process::Outcome::Exited,
            "{:?}",
            result.process
        );
        assert!(
            result.process.failure.is_none()
                && result.process.success()
                && result.process.cleanup.complete,
            "{:?}; stderr={:?}",
            result.process,
            result
                .stderr
                .as_ref()
                .map(|s| String::from_utf8_lossy(s.as_bytes()))
        );
        assert!(
            !result.process.stdout.truncated
                && !result.process.stderr.truncated
                && result.process.stdout.suppressed_lines == 0
                && result.process.stderr.suppressed_lines == 0
        );
        result.stdout.unwrap().as_bytes().to_vec()
    }
    fn git(&self, args: &[&str]) -> String {
        let mut arguments = vec![
            "-c".into(),
            "core.hooksPath=/dev/null".into(),
            "-c".into(),
            "commit.gpgsign=false".into(),
        ];
        arguments.extend(args.iter().map(|s| (*s).into()));
        String::from_utf8(self.execute(self.git.clone(), arguments, &[]))
            .unwrap()
            .trim()
            .into()
    }
    fn write(&self, path: &str, source: &str) {
        let destination = self.repository.join(path);
        ownership::check(&self.directory, &destination).unwrap();
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::write(destination, source).unwrap();
    }
    fn remove(&self, path: &str) {
        let destination = self.repository.join(path);
        ownership::check(&self.directory, &destination).unwrap();
        fs::remove_file(destination).unwrap();
    }
    fn commit(&self) -> String {
        self.git(&["add", "-A"]);
        self.git(&[
            "-c",
            "user.name=Just Policy Fixture",
            "-c",
            "user.email=just-policy@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "finite fixture",
        ]);
        let sha = self.git(&["rev-parse", "HEAD"]);
        assert_eq!(sha.len(), 40);
        assert!(sha.bytes().all(|b| b.is_ascii_hexdigit()));
        sha
    }
    fn classify(&self, base: &str, head: &str, changed: &str, event: &str) -> bool {
        let source = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../.github/actions/compute-changes/derive-outputs.sh"),
        )
        .unwrap();
        let start = "JUSTFILE_RECIPE_AWK='";
        let end = "# Backend/platform lanes rebuild";
        assert_eq!(source.matches(start).count(), 1);
        assert_eq!(source.matches(end).count(), 1);
        let region = source
            .split_once(start)
            .unwrap()
            .1
            .split_once(end)
            .unwrap()
            .0;
        // Preserve the actual action's shell error policy. Values remain process environment,
        // never interpolation into shell source or substitutions of maintained policy text.
        let script = format!(
            "set -e -o pipefail\n{start}{region}\nprintf '%s\\n' \"$BACKEND_RECIPE_CHANGED\"\n"
        );
        let output = self.execute(
            "/bin/bash".into(),
            vec!["-c".into(), script],
            &[
                ("BASE_SHA", base),
                ("HEAD_SHA", head),
                ("CHANGED_FILES", changed),
                ("EVENT_NAME", event),
            ],
        );
        match String::from_utf8(output).unwrap().trim() {
            "true" => true,
            "false" => false,
            other => panic!("unexpected classifier output: {other:?}"),
        }
    }
}

#[test]
fn compute_changes_justfiles_light_assignment_after_backend_recipe_stays_light() {
    let f = Fixture::new();
    let original = "bundle:\n    printf backend\n\nlight_setting := \"base\"\n\nwebsite-build:\n    printf light\n";
    f.write("Justfile", original);
    let base = f.commit();
    f.write("Justfile", &original.replace("\"base\"", "\"changed\""));
    let head = f.commit();
    assert!(!f.classify(&base, &head, "Justfile", "push"));
}

#[test]
fn compute_changes_justfiles_recipe_free_settings_exports_and_unknown_sources_fail_open() {
    for source in [
        "set shell := [\"bash\", \"-uc\"]\n",
        "export mesh_bin := \"target/release/mesh-llm\"\n",
        "# no recipes\n",
        "",
    ] {
        let f = Fixture::new();
        f.write("Justfile", "build:\n    true\n");
        let base = f.commit();
        f.write("just/settings.just", source);
        let head = f.commit();
        assert!(
            f.classify(&base, &head, "just/settings.just", "push"),
            "{source:?}"
        );
    }
}

#[test]
fn compute_changes_justfiles_backend_attributes_and_two_sided_recipe_hunks_are_backend_inputs() {
    let f = Fixture::new();
    f.write(
        "Justfile",
        "[unix]\nbundle:\n    printf backend\n\nwebsite-build:\n    true\n",
    );
    let base = f.commit();
    f.write(
        "Justfile",
        "[windows]\nbundle:\n    printf backend\n\nwebsite-build:\n    true\n",
    );
    let attribute = f.commit();
    assert!(f.classify(&base, &attribute, "Justfile", "push"));
    f.write("Justfile", "website-build:\n    true\n");
    let deleted = f.commit();
    assert!(
        f.classify(&attribute, &deleted, "Justfile", "push"),
        "old-side removed backend lines must still classify"
    );
    f.write(
        "Justfile",
        "website-build:\n    true\n\nbundle:\n    printf backend\n",
    );
    let added = f.commit();
    assert!(
        f.classify(&deleted, &added, "Justfile", "push"),
        "new-side backend lines must classify"
    );
}

#[test]
fn compute_changes_justfiles_pr_old_side_uses_merge_base_after_base_branch_advances() {
    let f = Fixture::new();
    f.write(
        "Justfile",
        "light:\n    printf light\n\nbundle:\n    printf backend\n",
    );
    let common = f.commit();
    f.git(&["checkout", "--quiet", "-b", "feature"]);
    f.write("Justfile", "light:\n    printf light\n");
    let head = f.commit();
    f.git(&["checkout", "--quiet", "-b", "base", &common]);
    f.write(
        "Justfile",
        "setting := \"base\"\n\nlight:\n    printf light\n\nbundle:\n    printf backend\n",
    );
    let base = f.commit();
    assert_eq!(f.git(&["merge-base", &base, &head]), common);
    assert!(f.classify(&base, &head, "Justfile", "pull_request"));
}

#[test]
fn compute_changes_justfiles_standalone_quantize_build_and_release_select_backend() {
    for recipe in [
        "skippy-quantize-standalone-build",
        "skippy-quantize-standalone-release-build",
    ] {
        let f = Fixture::new();
        f.write("Justfile", "import 'just/skippy.just'\n");
        let source = format!("{recipe} backend=\"cpu\":\n    printf base\n");
        f.write("just/skippy.just", &source);
        let base = f.commit();
        f.write(
            "just/skippy.just",
            &source.replace("printf base", "printf changed"),
        );
        let head = f.commit();
        assert!(f.classify(&base, &head, "just/skippy.just", "push"));
    }
}

#[test]
fn compute_changes_justfiles_global_assignments_follow_cross_import_backend_use() {
    let f = Fixture::new();
    let root = "website_dir := \"website\"\n\ndefault: build\n\nimport 'just/mesh.just'\n\nimport 'just/website-ui.just'\n";
    f.write("Justfile", root);
    let mesh = "mesh_bin := env(\"MESH_LLM_BIN\", \"target/release/mesh-llm\")\n\nbundle:\n    \"{{ mesh_bin }}\" --version\n";
    f.write("just/mesh.just", mesh);
    f.write(
        "just/website-ui.just",
        "website-build:\n    printf \"{{ website_dir }}\"\n",
    );
    let base = f.commit();
    f.write(
        "just/mesh.just",
        &mesh.replace("target/release/mesh-llm", "target/debug/mesh-llm"),
    );
    let backend = f.commit();
    assert!(f.classify(&base, &backend, "just/mesh.just", "push"));
    f.write("Justfile", &root.replace("\"website\"", "\"site\""));
    let light = f.commit();
    assert!(!f.classify(&backend, &light, "Justfile", "push"));
}

#[test]
fn compute_changes_justfiles_exported_backend_assignment_is_an_input() {
    let f = Fixture::new();
    let source = "export mesh_bin := \"target/release/mesh-llm\"\n\nbundle:\n    \"{{ mesh_bin }}\" --version\n";
    f.write("Justfile", source);
    let base = f.commit();
    f.write(
        "Justfile",
        &source.replace("target/release/mesh-llm", "target/debug/mesh-llm"),
    );
    let head = f.commit();
    assert!(f.classify(&base, &head, "Justfile", "push"));
}

#[test]
fn compute_changes_justfiles_nested_added_deleted_import_and_root_changes_preserve_policy() {
    let f = Fixture::new();
    let root = "default: build\n\nimport 'just/build.just'\n\nimport 'just/website-ui.just'\n";
    f.write("Justfile", root);
    f.write("just/build.just", "build:\n    true\n");
    f.write("just/website-ui.just", "website-build:\n    true\n");
    let base = f.commit();
    f.write("just/website-ui.just", "website-build:\n    printf light\n");
    let light = f.commit();
    assert!(!f.classify(&base, &light, "just/website-ui.just", "push"));
    f.write("just/build.just", "build:\n    printf backend\n");
    let backend = f.commit();
    assert!(f.classify(&light, &backend, "just/build.just", "push"));
    f.write(
        "just/nested/runtime.just",
        "release-runtime-build:\n    printf backend\n",
    );
    let nested = f.commit();
    f.write(
        "just/nested/runtime.just",
        "release-runtime-build:\n    printf changed\n",
    );
    let nested_head = f.commit();
    assert!(f.classify(&nested, &nested_head, "just/nested/runtime.just", "push"));
    f.write(
        "just/website-ui.just",
        "import 'nested/runtime.just'\n\nwebsite-build:\n    true\n",
    );
    let imported = f.commit();
    assert!(f.classify(&nested_head, &imported, "just/website-ui.just", "push"));
    f.write(
        "just/release-extra.just",
        "release-build-extra:\n    true\n",
    );
    let added = f.commit();
    assert!(f.classify(&imported, &added, "just/release-extra.just", "push"));
    f.remove("just/release-extra.just");
    let deleted = f.commit();
    assert!(f.classify(&added, &deleted, "just/release-extra.just", "push"));
    f.write(
        "Justfile",
        &(root.to_owned() + "\nimport 'just/missing-extra.just'\n"),
    );
    let root_head = f.commit();
    assert!(f.classify(&deleted, &root_head, "Justfile", "push"));
}

#[test]
fn compute_changes_justfiles_unreadable_revision_and_unknown_change_status_fail_open() {
    let f = Fixture::new();
    f.write("Justfile", "website-build:\n    true\n");
    let base = f.commit();
    f.write("Justfile", "website-build:\n    printf light\n");
    let head = f.commit();
    assert!(!f.classify(&base, &head, "Justfile", "push"));
    assert!(f.classify(&"0".repeat(40), &head, "Justfile", "push"));
    assert!(f.classify(&base, &head, "just/missing.just", "push"));
    assert!(f.classify(&head, &head, "Justfile", "push"));
}

#[path = "compute_changes_event_range.rs"]
mod event_range;
