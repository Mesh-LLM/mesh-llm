//! UI package policy and actual build-script/shell behavior, without installation.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use crate::workflow_yaml::{self, Node};
use semver::{Version, VersionReq};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

mod actual_console_build_script {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../mesh/crates/mesh-llm-ui/build.rs"
    ));
    pub(super) fn run() {
        main();
    }
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn satisfies(pin: &str, range: &str) -> bool {
    let requirement = range.split_whitespace().collect::<Vec<_>>().join(", ");
    assert!(!requirement.is_empty(), "pnpm engine range is required");
    VersionReq::parse(&requirement)
        .unwrap()
        .matches(&Version::parse(pin).unwrap())
}
#[test]
fn ui_pnpm_pin_satisfies_lower_bound() {
    assert!(satisfies("10.30.3", ">=10"));
}
#[test]
fn ui_pnpm_pin_refuses_exclusive_upper_bound() {
    assert!(!satisfies("10.30.3", "<10"));
}
#[test]
fn ui_pnpm_pin_refuses_narrow_conjunction() {
    assert!(!satisfies("10.30.3", ">=10 <10.30.0"));
}
#[test]
fn ui_pnpm_pin_satisfies_wide_conjunction() {
    assert!(satisfies("10.30.3", ">=10 <11"));
}

fn execute(
    executable: PathBuf,
    cwd: &Path,
    arguments: Vec<Value>,
    environment: BTreeMap<std::ffi::OsString, Value>,
) -> String {
    let report = process::supervise(
        &ProcessSpec {
            executable,
            cwd: cwd.to_path_buf(),
            arguments,
            environment,
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
    assert!(
        report.success(),
        "{}",
        String::from_utf8_lossy(&report.stderr.bytes_retained)
    );
    String::from_utf8(report.stdout.bytes_retained).unwrap()
}

#[test]
fn ui_build_script_uses_fallback_without_missing_directory_watch() {
    if std::env::var_os("MESH_UI_BUILD_SCRIPT_FIXTURE_CHILD").is_some() {
        actual_console_build_script::run();
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let ui = root.join("ui");
    let out = root.join("out");
    fs::create_dir_all(&ui).unwrap();
    fs::create_dir_all(&out).unwrap();
    for present in [false, true] {
        if present {
            fs::create_dir(ui.join("dist")).unwrap();
        }
        let expected = if present {
            ui.join("dist")
        } else {
            out.join("empty-ui-dist")
        };
        let stdout = execute(std::env::current_exe().unwrap().canonicalize().unwrap(), &root,
            ["--exact", "ui_package_contracts::ui_build_script_uses_fallback_without_missing_directory_watch", "--nocapture"].into_iter().map(|s| Value::Public(s.into())).collect(),
            BTreeMap::from([("MESH_UI_BUILD_SCRIPT_FIXTURE_CHILD".into(), Value::Public("1".into())), ("CARGO_MANIFEST_DIR".into(), Value::Public(ui.clone().into())), ("OUT_DIR".into(), Value::Public(out.clone().into()))]));
        assert!(
            stdout.contains(&format!(
                "cargo:rustc-env=MESH_LLM_UI_DIST={}",
                expected.display()
            )),
            "{stdout}"
        );
        assert!(
            !stdout.contains("cargo:rerun-if-changed=dist"),
            "missing dist must not force recurring rebuilds"
        );
        assert!(expected.is_dir());
    }
}

#[test]
fn ui_workspace_contains_root_package() {
    let workspace = workflow_yaml::parse(
        &fs::read_to_string(repository().join("mesh/crates/mesh-llm-ui/pnpm-workspace.yaml"))
            .unwrap(),
    )
    .unwrap();
    assert!(workspace.get("packages").unwrap().list().contains(&"."));
}

fn pnpm_versions(node: &Node, versions: &mut Vec<String>) {
    if node
        .get("uses")
        .and_then(Node::text)
        .is_some_and(|v| v.starts_with("pnpm/action-setup@"))
        && let Some(version) = node
            .get("with")
            .and_then(|v| v.get("version"))
            .and_then(Node::text)
    {
        versions.push(version.to_owned());
    }
    match node {
        Node::Map(entries) => {
            for (_, child) in entries {
                pnpm_versions(child, versions);
            }
        }
        Node::Seq(children) => {
            for child in children {
                pnpm_versions(child, versions);
            }
        }
        Node::Scalar(_) => {}
    }
}
#[test]
fn ui_pnpm_engine_and_exact_pin_match_all_ci_setup_majors() {
    let root = repository();
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("mesh/crates/mesh-llm-ui/package.json")).unwrap(),
    )
    .unwrap();
    let range = manifest["engines"]["pnpm"].as_str().unwrap();
    let pin = manifest["packageManager"]
        .as_str()
        .unwrap()
        .strip_prefix("pnpm@")
        .unwrap();
    let version = Version::parse(pin).unwrap();
    assert!(version.pre.is_empty() && version.build.is_empty());
    assert_eq!(
        pin,
        version.to_string(),
        "packageManager must pin a canonical exact version"
    );
    assert!(satisfies(pin, range));
    let engine =
        VersionReq::parse(&range.split_whitespace().collect::<Vec<_>>().join(", ")).unwrap();
    let declared_major = engine.comparators.first().unwrap().major;
    assert_eq!(
        version.major, declared_major,
        "engine floor and packageManager major must agree"
    );
    let mut versions = Vec::new();
    for entry in fs::read_dir(root.join(".github/workflows")).unwrap() {
        let path = entry.unwrap().path();
        if matches!(
            path.extension().and_then(|v| v.to_str()),
            Some("yml" | "yaml")
        ) {
            pnpm_versions(
                &workflow_yaml::parse(&fs::read_to_string(path).unwrap()).unwrap(),
                &mut versions,
            );
        }
    }
    assert!(
        !versions.is_empty(),
        "CI must declare at least one explicit pnpm setup version"
    );
    for ci in versions {
        assert_eq!(
            ci.split('.').next().unwrap().parse::<u64>().unwrap(),
            version.major,
            "CI pnpm version {ci}"
        );
    }
}
#[test]
fn ui_npmrc_enforces_engine_strictly() {
    let source = fs::read_to_string(repository().join("mesh/crates/mesh-llm-ui/.npmrc")).unwrap();
    let mut strict = None;
    for line in source.lines() {
        let line = line.split('#').next().unwrap().trim();
        if let Some((name, value)) = line.split_once('=')
            && name.trim() == "engine-strict"
        {
            assert!(
                strict.replace(value.trim()).is_none(),
                "duplicate engine-strict setting"
            );
        }
    }
    assert_eq!(strict, Some("true"));
}

#[test]
fn ui_actual_shell_normalizes_mixed_case_release_profile() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let ui = root.join("ui");
    fs::create_dir(&ui).unwrap();
    for name in [
        "package.json",
        "pnpm-lock.yaml",
        "vite.config.ts",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.node.json",
        "biome.json",
        "index.html",
    ] {
        fs::write(ui.join(name), "{}\n").unwrap();
    }
    for name in ["src", "public", "dist"] {
        fs::create_dir(ui.join(name)).unwrap();
    }
    fs::write(ui.join("dist/asset.js"), "// built\n").unwrap();
    fs::write(ui.join("dist/.mesh-llm-ui-build-env"), "MESH_LLM_BUILD_PROFILE=release\nVITE_MESH_LLM_DEBUG_UI=false\nVITE_BASE_PATH=\nVITE_ROUTER_BASE_PATH=\nVITE_STORAGE_NAMESPACE=\n").unwrap();
    let search = std::env::var_os("PATH").unwrap();
    let bash = std::env::split_paths(&search)
        .map(|p| p.join("bash"))
        .find(|p| p.is_file())
        .unwrap()
        .canonicalize()
        .unwrap();
    let stdout = execute(
        bash,
        &root,
        vec![
            Value::Public(repository().join("mesh/scripts/build-ui.sh").into()),
            Value::Public(ui.into()),
        ],
        BTreeMap::from([
            ("PATH".into(), Value::Public(search)),
            (
                "MESH_LLM_BUILD_PROFILE".into(),
                Value::Public("ReLeAsE".into()),
            ),
        ]),
    );
    assert!(
        stdout.contains("Skipping mesh-llm UI build"),
        "must reuse the already-current fixture without package installation: {stdout}"
    );
    assert!(stdout.contains("profile: release") && stdout.contains("debug UI: false"));
}
