//! Actual copied Unix developer bootstrap; inert package-manager and Cargo boundaries.
use super::{Fixture, executable, fs, program, repo};
fn fixture() -> Fixture {
    let fixture = Fixture::new();
    fs::copy(
        repo().join("scripts/bootstrap-build-tools"),
        fixture.root.join("scripts/bootstrap-build-tools"),
    )
    .unwrap();
    executable(&fixture.root.join("bin/apt-get"), "#!/bin/bash\nexit 93\n");
    executable(
        &fixture.root.join("bin/sudo"),
        "#!/bin/bash\nprintf '%s\\0' \"$@\" >> \"$FIXTURE_ROOT/packages.arguments\"\n",
    );
    executable(
        &fixture.root.join("bin/brew"),
        "#!/bin/bash\nprintf '%s\\0' \"$@\" >> \"$FIXTURE_ROOT/packages.arguments\"\n",
    );
    executable(
        &fixture.root.join("bin/sccache"),
        r#"#!/bin/bash
set -euo pipefail
if [[ -f "$FIXTURE_ROOT/installed-version" ]]; then
 printf 'sccache %s\n' "$(cat "$FIXTURE_ROOT/installed-version")"
else printf 'sccache %s\n' "${SCCACHE_INSTALLED_VERSION:-0.16.0}"; fi
"#,
    );
    executable(
        &fixture.root.join("bin/cargo"),
        r#"#!/bin/bash
set -euo pipefail
printf '%s\0' "$@" > "$FIXTURE_ROOT/install.arguments"
printf '[%s]\n' "${RUSTC_WRAPPER-unset}" > "$FIXTURE_ROOT/install.wrapper"
for ((i=1; i<=$#; i++)); do
 if [[ "${!i}" == --version ]]; then ((i+=1)); printf '%s' "${!i}" > "$FIXTURE_ROOT/installed-version"; exit 0; fi
done
exit 91
"#,
    );
    fixture
}
fn arguments(fixture: &Fixture, file: &str) -> Vec<String> {
    fs::read(fixture.root.join(file))
        .unwrap()
        .split(|b| *b == 0)
        .filter(|b| !b.is_empty())
        .map(|b| String::from_utf8(b.to_vec()).unwrap())
        .collect()
}
#[test]
fn developer_bootstrap_linux_installs_linkers_and_preserves_matching_pinned_cache() {
    let fixture = fixture();
    let result = fixture.run(program("bash"), &["scripts/bootstrap-build-tools"], &[]);
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert_eq!(
        arguments(&fixture, "packages.arguments"),
        [
            "apt-get", "update", "apt-get", "install", "-y", "mold", "lld"
        ]
    );
    assert!(!fixture.root.join("install.arguments").exists());
    assert!(result.stdout.contains("sccache 0.16.0"));
    assert_eq!(fixture.text("compiler.events"), "probe\n");
}
#[test]
fn developer_bootstrap_cache_install_is_pinned_or_overridden_and_cache_bypass_is_child_local() {
    for desired in ["0.16.0", "finite-custom-version"] {
        let fixture = fixture();
        let result = fixture.run(
            program("bash"),
            &["scripts/bootstrap-build-tools"],
            &[
                ("SCCACHE_INSTALLED_VERSION", "old-version"),
                ("MESH_LLM_SCCACHE_VERSION", desired),
                ("RUSTC_WRAPPER", "caller-cache"),
            ],
        );
        assert_eq!(result.code, 0, "{}", result.stderr);
        assert_eq!(
            arguments(&fixture, "install.arguments"),
            [
                "install",
                "sccache",
                "--version",
                desired,
                "--locked",
                "--force"
            ]
        );
        assert_eq!(fixture.text("install.wrapper"), "[]\n");
        assert!(result.stdout.contains(&format!("sccache {desired}")));
        assert_eq!(fixture.text("compiler.environment"), "\ncaller-cache\n\n");
        assert_eq!(fixture.text("compiler.events"), "probe\n");
    }
}
#[test]
fn developer_bootstrap_darwin_installs_lld_and_probes_the_actual_driver() {
    let fixture = fixture();
    let result = fixture.run(
        "/bin/bash".into(),
        &["scripts/bootstrap-build-tools"],
        &[("FAKE_OS", "Darwin"), ("FAKE_ARCH", "arm64")],
    );
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert_eq!(
        arguments(&fixture, "packages.arguments"),
        ["install", "lld"]
    );
    assert!(!fixture.root.join("install.arguments").exists());
    assert!(
        result
            .stdout
            .contains(fixture.root.join("bin/ld64.lld").to_str().unwrap())
    );
    assert_eq!(fixture.text("compiler.events"), "probe\n");
}
