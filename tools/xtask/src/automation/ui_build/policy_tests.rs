//! UI profile and filesystem freshness policy fixtures.
use super::policy::{Decision, Environment, Profile, decide};
use std::{
    collections::BTreeMap,
    fs::{self, File, FileTimes},
    path::Path,
    time::{Duration, SystemTime},
};

fn environment(profile: Profile) -> Environment {
    Environment {
        profile,
        variables: BTreeMap::from([("VITE_MESH_LLM_DEBUG_UI".into(), "true".into())]),
    }
}
fn time(path: &Path, value: SystemTime) {
    File::open(path)
        .unwrap()
        .set_times(FileTimes::new().set_modified(value))
        .unwrap();
}
fn fixture(root: &Path, build: &Environment) {
    fs::create_dir(root.join("dist")).unwrap();
    fs::create_dir(root.join("node_modules")).unwrap();
    fs::create_dir(root.join("src")).unwrap();
    fs::write(root.join("package.json"), "{}\n").unwrap();
    fs::write(root.join("pnpm-lock.yaml"), "lock\n").unwrap();
    fs::write(root.join("src/app.ts"), "// source\n").unwrap();
    fs::write(root.join("dist/index.html"), "<html></html>\n").unwrap();
    fs::write(
        root.join("dist/.mesh-llm-ui-build-env"),
        build.stamp_for(root).unwrap(),
    )
    .unwrap();
    let base = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
    for name in ["package.json", "pnpm-lock.yaml", "src/app.ts", "src"] {
        time(&root.join(name), base);
    }
    time(&root.join("node_modules"), base + Duration::from_secs(1));
    time(&root.join("dist"), base + Duration::from_secs(2));
}
#[test]
fn profile_normalizes_and_release_forces_debug_off() {
    assert_eq!(Profile::parse("ReLeAsE").unwrap(), Profile::Release);
    assert_eq!(environment(Profile::Release).debug_ui(), "false");
    assert_eq!(Profile::parse("").unwrap(), Profile::Debug);
    assert!(Profile::parse("production").is_err());
}
#[test]
fn exact_stamp_retains_all_projection_values() {
    let mut build = environment(Profile::Dev);
    build.variables.extend([
        ("VITE_MESH_LLM_DEBUG_UI".into(), "".into()),
        ("VITE_BASE_PATH".into(), "/console/".into()),
        ("VITE_ROUTER_BASE_PATH".into(), "/console".into()),
        ("VITE_STORAGE_NAMESPACE".into(), "sdk".into()),
    ]);
    let stamp: serde_json::Value = serde_json::from_str(&build.stamp()).unwrap();
    assert_eq!(stamp["schema"], 2);
    assert_eq!(stamp["profile"], "dev");
    assert_eq!(stamp["variables"]["VITE_BASE_PATH"], "/console/");
    assert_eq!(stamp["variables"]["VITE_ROUTER_BASE_PATH"], "/console");
    assert_eq!(stamp["variables"]["VITE_STORAGE_NAMESPACE"], "sdk");
    assert_eq!(stamp["variables"]["VITE_MESH_LLM_DEBUG_UI"], "true");
}
#[test]
fn current_output_and_dependencies_reuse() {
    let temp = tempfile::tempdir().unwrap();
    let build = environment(Profile::Debug);
    fixture(temp.path(), &build);
    assert_eq!(decide(temp.path(), &build).unwrap(), Decision::Reuse);
}
#[test]
fn stamp_alone_cannot_certify_output() {
    let temp = tempfile::tempdir().unwrap();
    let build = environment(Profile::Debug);
    fixture(temp.path(), &build);
    fs::remove_file(temp.path().join("dist/index.html")).unwrap();
    assert_eq!(decide(temp.path(), &build).unwrap(), Decision::Build);
}
#[test]
fn profile_change_builds_without_reinstalling() {
    let temp = tempfile::tempdir().unwrap();
    fixture(temp.path(), &environment(Profile::Debug));
    assert_eq!(
        decide(temp.path(), &environment(Profile::Release)).unwrap(),
        Decision::Build
    );
}
#[test]
fn newer_source_requires_build() {
    let temp = tempfile::tempdir().unwrap();
    let build = environment(Profile::Debug);
    fixture(temp.path(), &build);
    time(&temp.path().join("src/app.ts"), SystemTime::now());
    assert_eq!(decide(temp.path(), &build).unwrap(), Decision::Build);
}
#[test]
fn newer_lockfile_requires_install_and_build() {
    let temp = tempfile::tempdir().unwrap();
    let build = environment(Profile::Debug);
    fixture(temp.path(), &build);
    time(&temp.path().join("pnpm-lock.yaml"), SystemTime::now());
    assert_eq!(
        decide(temp.path(), &build).unwrap(),
        Decision::InstallAndBuild
    );
}
#[test]
fn missing_optional_input_preserves_current_build() {
    let temp = tempfile::tempdir().unwrap();
    let build = environment(Profile::Debug);
    fixture(temp.path(), &build);
    assert!(!temp.path().join("tsconfig.node.json").exists());
    assert_eq!(decide(temp.path(), &build).unwrap(), Decision::Reuse);
}

#[test]
fn source_removal_requires_rebuild_from_changed_directory() {
    let temp = tempfile::tempdir().unwrap();
    let build = environment(Profile::Debug);
    fixture(temp.path(), &build);
    fs::remove_file(temp.path().join("src/app.ts")).unwrap();
    assert_eq!(decide(temp.path(), &build).unwrap(), Decision::Build);
}

#[test]
fn cache_identity_preserves_absent_empty_and_multiline_values() {
    let absent = environment(Profile::Debug);
    let mut empty = environment(Profile::Debug);
    empty.variables.insert("VITE_BASE_PATH".into(), "".into());
    assert_ne!(absent.stamp(), empty.stamp());
    let mut multiline = environment(Profile::Debug);
    multiline.variables.insert(
        "VITE_STORAGE_NAMESPACE".into(),
        "first\nsecond\rthird".into(),
    );
    let parsed: serde_json::Value = serde_json::from_str(&multiline.stamp()).unwrap();
    assert_eq!(
        parsed["variables"]["VITE_STORAGE_NAMESPACE"],
        "first\nsecond\rthird"
    );
}

#[test]
fn changed_vite_settings_and_file_router_invalidate_cache() {
    for name in [
        "VITE_APP_VERSION",
        "VITE_API_URL",
        "VITE_MANAGEMENT_API_URL",
        "VITE_ENABLE_PERF_ROUTE",
        "VITE_FUTURE_SETTING",
        "TANSTACK_FILE_ROUTER",
        "NODE_ENV",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let mut build = environment(Profile::Debug);
        fixture(temp.path(), &build);
        build.variables.insert(name.into(), "changed".into());
        assert_eq!(
            decide(temp.path(), &build).unwrap(),
            Decision::Build,
            "{name}"
        );
    }
}

#[test]
fn dotenv_build_inputs_invalidate_cached_output() {
    for name in [
        ".env",
        ".env.local",
        ".env.production",
        ".env.production.local",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let build = environment(Profile::Debug);
        fixture(temp.path(), &build);
        fs::write(temp.path().join(name), "VITE_APP_VERSION=changed\n").unwrap();
        assert_eq!(
            decide(temp.path(), &build).unwrap(),
            Decision::Build,
            "{name}"
        );
    }
}

#[test]
fn removed_dotenv_settings_invalidate_even_with_old_directory_timestamps() {
    let temp = tempfile::tempdir().unwrap();
    let build = environment(Profile::Debug);
    fs::write(temp.path().join(".env.local"), "VITE_APP_VERSION=old\n").unwrap();
    fixture(temp.path(), &build);
    fs::remove_file(temp.path().join(".env.local")).unwrap();
    assert_eq!(decide(temp.path(), &build).unwrap(), Decision::Build);
}
