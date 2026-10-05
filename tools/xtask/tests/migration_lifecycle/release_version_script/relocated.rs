//! Complete relocated version bump, with metadata invocation recorded by inert Cargo.
use super::{Fixture, SOURCE};
use std::fs;

fn relocated(logical: &str) -> String {
    let prefix = if logical.starts_with("crates/skippy-") || logical.starts_with("third_party/") {
        "skippy"
    } else {
        "mesh"
    };
    format!("{prefix}/{logical}")
}

fn content(relative: &str) -> String {
    if relative.ends_with("Cargo.toml") {
        "[package]\nname = \"fixture\"\nversion = \"0.76.1\"\n".into()
    } else if relative.ends_with("build.gradle.kts") {
        "plugins { }\nversion = \"0.76.1\"\n".into()
    } else if relative.ends_with("package.json") || relative.ends_with("package-lock.json") {
        serde_json::json!({"name":"fixture","version":"0.76.1","packages":{"":{"version":"0.76.1"}}}).to_string()
    } else if relative.ends_with("setting_schema.rs") {
        "fn known_mesh_llm_versions() -> &'static [&'static str] {\n    &[\n    ]\n}\n".into()
    } else if relative.ends_with(".rs") || relative.ends_with(".json") {
        "release 0.76.1\n".into()
    } else {
        "release v0.76.1\n".into()
    }
}

#[test]
fn relocated_only_release_bumps_every_declared_sidecar_without_legacy_aliases() {
    let body = SOURCE
        .split_once("literal_version_files=(\n")
        .unwrap()
        .1
        .split_once("\n)")
        .unwrap()
        .0;
    let mut paths = body
        .lines()
        .map(|line| {
            line.trim()
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .expect("literal version path")
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert!(paths.len() > 15);
    paths.extend([
        "sdk/kotlin/build.gradle.kts".into(),
        "crates/mesh-llm-config/src/model/built_in_schema/setting_schema.rs".into(),
    ]);
    let mut paths = paths
        .iter()
        .map(|logical| relocated(logical))
        .collect::<Vec<_>>();
    paths.extend([
        "mesh/crates/mesh-llm-config/Cargo.toml".into(),
        "skippy/crates/skippy-runtime/Cargo.toml".into(),
    ]);
    paths.sort();
    paths.dedup();
    let fixture = Fixture::new();
    fixture.write("scripts/release-version.sh", SOURCE);
    fixture.write("Cargo.toml", "[workspace]\nmembers = []\nresolver = \"2\"\n\n[workspace.package]\nversion = \"0.76.1\"\n");
    for relative in &paths {
        fixture.write(relative, &content(relative));
    }
    fixture.track();
    let (ok, stdout, error) = fixture.invoke("bash", &["scripts/release-version.sh", "0.77.0"]);
    assert!(ok, "{stdout}\n{error}");
    for relative in &paths {
        let updated = fs::read_to_string(fixture.root.join(relative)).unwrap();
        assert!(
            updated.contains("0.77.0") && !updated.contains("0.76.1"),
            "{relative}"
        );
        if relative.ends_with("package.json") || relative.ends_with("package-lock.json") {
            let json: serde_json::Value = serde_json::from_str(&updated).unwrap();
            assert_eq!(json["version"], "0.77.0");
            assert_eq!(json["packages"][""]["version"], "0.77.0");
        }
    }
    assert!(
        fs::read_to_string(fixture.root.join("Cargo.toml"))
            .unwrap()
            .contains("version = \"0.77.0\"")
    );
    assert!(stdout.contains("mesh/sdk/kotlin/build.gradle.kts"));
    assert!(!fixture.root.join("crates").exists() && !fixture.root.join("website").exists());
    assert_eq!(
        fs::read_to_string(fixture.root.join("cargo.calls")).unwrap(),
        "metadata --format-version 1\n"
    );
}
