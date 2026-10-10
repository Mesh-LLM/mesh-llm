//! Actual typed owner and its thin caller use metadata directory ownership.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

struct Fixture {
    state: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let state = tempfile::tempdir().unwrap();
        let root = state.path().join("workspace with spaces");
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::create_dir_all(root.join("scripts/lib")).unwrap();
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for name in ["affected-crates.sh", "lib/automation.sh"] {
            fs::copy(
                repo.join("scripts").join(name),
                root.join("scripts").join(name),
            )
            .unwrap();
        }
        let rows = [
            ("mesh-owner", "mesh/crates/mesh-owner", vec![]),
            (
                "skippy-owner",
                "skippy/crates/skippy-owner",
                vec!["mesh-owner"],
            ),
            ("legacy-owner", "crates/legacy-owner", vec![]),
            ("mesh-ui", "mesh/crates/mesh-llm-ui", vec!["legacy-owner"]),
            ("metadata-owned", "unusual/product-root", vec![]),
        ];
        let metadata = json!({"workspace_root":root,"packages":rows.into_iter().map(|(name,dir,deps)| json!({"name":name,"manifest_path":root.join(dir).join("Cargo.toml"),"dependencies":deps.into_iter().map(|name|json!({"name":name})).collect::<Vec<_>>()})).collect::<Vec<_>>()});
        fs::write(
            root.join("metadata.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        // The finite Cargo substitute emits metadata only; no build runs.
        let cargo = root.join("bin/cargo");
        fs::write(&cargo, "#!/bin/sh\n[ \"$*\" = 'metadata --format-version=1 --no-deps' ] || exit 71\n/bin/cat \"$(dirname \"$0\")/../metadata.json\"\n").unwrap();
        fs::set_permissions(cargo, fs::Permissions::from_mode(0o700)).unwrap();
        Self { state, root }
    }
    fn run(&self, mode: u8, changed: &[&str]) -> Json {
        let (executable, arguments) = if mode == 2 {
            let input = self.root.join("changed paths.txt");
            fs::write(
                &input,
                changed
                    .iter()
                    .map(|path| format!("{path}\n"))
                    .collect::<String>(),
            )
            .unwrap();
            (
                PathBuf::from("/bin/bash"),
                vec![
                    "-c".into(),
                    "exec /bin/bash \"$1\" --stdin < \"$2\"".into(),
                    "affected-input".into(),
                    self.root
                        .join("scripts/affected-crates.sh")
                        .into_os_string(),
                    input.into_os_string(),
                ],
            )
        } else if mode == 1 {
            (
                PathBuf::from("/bin/bash"),
                std::iter::once(
                    self.root
                        .join("scripts/affected-crates.sh")
                        .into_os_string(),
                )
                .chain(changed.iter().copied().map(std::ffi::OsString::from))
                .collect::<Vec<_>>(),
            )
        } else {
            (
                PathBuf::from(env!("CARGO_BIN_EXE_xtask")),
                ["repository", "affected-crates"]
                    .into_iter()
                    .chain(changed.iter().copied())
                    .map(std::ffi::OsString::from)
                    .collect(),
            )
        };
        let environment = BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(format!("{}:/usr/bin:/bin", self.root.join("bin").display()).into()),
            ),
            (
                "HOME".into(),
                Value::Public(self.state.path().as_os_str().to_owned()),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
        ]);
        let spec = ProcessSpec {
            executable,
            arguments: arguments.into_iter().map(Value::Public).collect(),
            cwd: self.root.clone(),
            environment,
        };
        let limits = Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = process::supervise_raw(
            &spec,
            &limits,
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(1024 * 1024),
                stderr: None,
            },
        )
        .unwrap();
        assert!(
            report.process.success()
                && report.process.cleanup.complete
                && report.process.failure.is_none(),
            "{report:?}"
        );
        serde_json::from_slice(report.stdout.unwrap().as_bytes()).unwrap()
    }
}
#[test]
fn relocated_metadata_selects_direct_packages_and_reverse_dependents_through_real_caller() {
    let f = Fixture::new();
    for mode in [0, 1, 2] {
        for (path, direct, affected) in [
            (
                "mesh/crates/mesh-owner/src/lib.rs",
                "mesh-owner",
                vec!["mesh-owner", "skippy-owner"],
            ),
            (
                "skippy/crates/skippy-owner/src/lib.rs",
                "skippy-owner",
                vec!["skippy-owner"],
            ),
            (
                "unusual/product-root/src/lib.rs",
                "metadata-owned",
                vec!["metadata-owned"],
            ),
        ] {
            let result = f.run(mode, &[path]);
            assert_eq!(result["test_crates"], json!([direct]));
            assert_eq!(result["affected"], json!(affected));
            assert_eq!(result["all_rust"], false);
            assert_eq!(result["ui_changed"], false);
        }
    }
}
#[test]
fn legacy_crate_and_relocated_ui_keep_distinct_selection_through_real_caller() {
    let f = Fixture::new();
    for mode in [0, 1, 2] {
        let result = f.run(mode, &["crates/legacy-owner/src/lib.rs"]);
        assert_eq!(result["affected"], json!(["legacy-owner", "mesh-ui"]));
        assert_eq!(result["test_crates"], json!(["legacy-owner"]));
        assert_eq!(result["ui_changed"], false);
        assert_eq!(result["all_rust"], false);
        let result = f.run(mode, &["mesh/crates/mesh-llm-ui/src/App.tsx"]);
        assert_eq!(result["ui_changed"], true);
        assert_eq!(result["test_crates"], json!([]));
        assert_eq!(result["affected"], json!([]));
    }
}
#[test]
fn caller_cutover_preserves_relocated_website_and_native_escalation_policy() {
    let f = Fixture::new();
    for mode in [0, 1, 2] {
        for path in ["mesh/website/src/index.njk", "mesh/docs/assets/app.js"] {
            let result = f.run(mode, &[path]);
            assert_eq!(result["website_changed"], true);
            assert_eq!(result["all_rust"], false);
        }
        for path in [
            "skippy/third_party/llama.cpp/upstream.txt",
            "skippy/third_party/llama.cpp/patches/0001.patch",
            "mesh/scripts/build-linux.sh",
            "skippy/scripts/prepare-llama.sh",
        ] {
            assert_eq!(f.run(mode, &[path])["all_rust"], true);
        }
        assert_eq!(
            f.run(mode, &["mesh/scripts/build-linuxx.sh"])["all_rust"],
            false
        );
    }
}
