#![cfg(unix)]
use serde_json::Value;
use std::{
    fs,
    os::unix::fs::PermissionsExt as _,
    path::PathBuf,
    process::{Command, Output},
};

const FIXTURE_MODEL: &str = "literal\"\\n雪";

struct Fixture {
    state: tempfile::TempDir,
    script: String,
}

impl Fixture {
    fn new() -> Self {
        let source = fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/ci-opencode-smoke.sh"),
        )
        .unwrap();
        let selector = source
            .split_once("OPENCODE_AUTOMATION_HOME=")
            .unwrap()
            .1
            .split_once("# Frozen automation selection ends.")
            .unwrap()
            .0;
        let function = source
            .split_once("prepare_opencode_config() {\n")
            .unwrap()
            .1
            .split_once("\n}\n")
            .unwrap()
            .0;
        let identity = source
            .split_once("resolve_opencode_model_identity() {\n")
            .unwrap()
            .1
            .split_once("\n}\n")
            .unwrap()
            .0;
        let script = format!(
            "set -euo pipefail\nOPENCODE_AUTOMATION_HOME={selector}\nHOME=\"$CLIENT_HOME\"\nprepare_opencode_config() {{\n{function}\n}}\nresolve_opencode_model_identity() {{\n{identity}\n}}\nresolve_opencode_model_identity\nprepare_opencode_config\nprintf '%s' \"$OPENCODE_CONFIG_CONTENT\"\n"
        );
        let state = tempfile::tempdir().unwrap();
        fs::write(state.path().join("Justfile"), "# finite fixture facade\n").unwrap();
        let just = state.path().join("just");
        fs::write(&just, concat!(
            "#!/bin/bash\nprintf '%s\\n' \"$HOME\" \"$@\" > \"$JUST_TRACE\"\n",
            "[[ \"$1\" == --justfile && \"$2\" == \"$ROOT/Justfile\" && \"$3\" == automation-run ]] || exit 91\n",
            "shift 3\nexec \"$ACTUAL_AUTOMATION\" \"$@\"\n"
        )).unwrap();
        fs::set_permissions(just, fs::Permissions::from_mode(0o700)).unwrap();
        Self { state, script }
    }

    fn run(&self, model: &str, configured: Option<&str>, override_config: &str) -> Output {
        self.run_identity(model, FIXTURE_MODEL, configured, override_config)
    }

    fn run_identity(
        &self,
        model: &str,
        mesh_identity: &str,
        configured: Option<&str>,
        override_config: &str,
    ) -> Output {
        let mut command = Command::new("/bin/bash");
        command
            .args(["-c", &self.script])
            .env("ROOT", self.state.path())
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.state.path().display()),
            )
            .env("HOME", "/original toolchain home")
            .env("CLIENT_HOME", "/isolated client home")
            .env("JUST_TRACE", self.state.path().join("just-trace"))
            .env("ACTUAL_AUTOMATION", env!("CARGO_BIN_EXE_xtask"))
            .env("MODEL", model)
            .env("OPENCODE_SMOKE_MODEL", model)
            .env("MESH_MODEL", mesh_identity)
            .env("CONFIG_BASE_URL", "http://127.0.0.1:9337/v1///")
            .env("OPENCODE_CONFIG_CONTENT", override_config)
            .env("OPENAI_API_KEY", "fixture-private-key")
            .env_remove("MESH_LLM_AUTOMATION_BIN");
        if let Some(configured) = configured {
            command.env("MESH_LLM_AUTOMATION_BIN", configured);
        }
        command.output().unwrap()
    }
}

#[test]
fn actual_opencode_config_adapter_selects_protected_owner_and_preserves_overrides() {
    let fixture = Fixture::new();
    let result = fixture.run("mesh/selected", Some(env!("CARGO_BIN_EXE_xtask")), "");
    assert!(result.status.success(), "{result:?}");
    let config: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        config["provider"]["mesh"]["models"][FIXTURE_MODEL]["name"],
        FIXTURE_MODEL
    );
    assert_eq!(
        config["provider"]["mesh"]["options"]["baseURL"],
        "http://127.0.0.1:9337/v1"
    );
    assert!(
        !String::from_utf8(result.stdout)
            .unwrap()
            .contains("fixture-private-key")
    );
    assert!(!fixture.state.path().join("just-trace").exists());
    let custom = "{\"custom\":\"provider preserved\"}";
    let result = fixture.run("mesh/selected", Some(env!("CARGO_BIN_EXE_xtask")), custom);
    assert!(result.status.success());
    assert_eq!(result.stdout, custom.as_bytes());
    assert!(!fixture.state.path().join("just-trace").exists());
}

#[test]
fn actual_opencode_config_adapter_absent_owner_uses_original_home_and_just_facade() {
    let fixture = Fixture::new();
    let result = fixture.run("third-party/model", None, "");
    assert!(result.status.success(), "{result:?}");
    let config: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert!(config.get("provider").is_none());
    assert_eq!(config["permission"]["edit"], "allow");
    assert_eq!(config["permission"]["webfetch"], "deny");
    assert_eq!(
        fs::read_to_string(fixture.state.path().join("just-trace")).unwrap(),
        format!(
            "/original toolchain home\n--justfile\n{}\nautomation-run\nautomation\nagent-client-config\nopencode\n",
            fixture.state.path().join("Justfile").display()
        )
    );
}

#[test]
fn actual_opencode_config_adapter_invalid_configured_owner_never_falls_back() {
    let fixture = Fixture::new();
    let missing = fixture.state.path().join("missing");
    let directory = fixture.state.path().to_str().unwrap();
    let non_executable = fixture.state.path().join("not executable");
    fs::write(&non_executable, "not executable").unwrap();
    for configured in [
        "",
        "relative",
        missing.to_str().unwrap(),
        directory,
        non_executable.to_str().unwrap(),
    ] {
        let result = fixture.run("mesh/selected", Some(configured), "");
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
        assert_eq!(
            result.stderr,
            b"MESH_LLM_AUTOMATION_BIN must be an absolute executable\n"
        );
        assert!(!fixture.state.path().join("just-trace").exists());
    }
}

#[test]
fn explicit_mesh_prefix_derives_nested_identity_once_and_preserves_explicit_identity() {
    let fixture = Fixture::new();
    let owner = Some(env!("CARGO_BIN_EXE_xtask"));
    let result = fixture.run_identity("mesh/org/nested/model", "", owner, "");
    assert!(result.status.success(), "{result:?}");
    let config: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        config["provider"]["mesh"]["models"]["org/nested/model"]["name"],
        "org/nested/model"
    );
    let result = fixture.run_identity("mesh/mesh/org/model", "", owner, "");
    assert!(result.status.success());
    let config: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        config["provider"]["mesh"]["models"]["mesh/org/model"]["name"],
        "mesh/org/model"
    );
    let result = fixture.run_identity(
        "mesh/different-client-label",
        "explicit selected identity",
        owner,
        "",
    );
    assert!(result.status.success());
    let config: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        config["provider"]["mesh"]["models"]["explicit selected identity"]["name"],
        "explicit selected identity"
    );
}

#[test]
fn empty_mesh_remainder_or_other_provider_without_identity_fails_before_config() {
    let fixture = Fixture::new();
    for model in ["mesh/", "other/model", "mesh-like/model"] {
        let result = fixture.run_identity(model, "", None, "");
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
        assert!(!fixture.state.path().join("just-trace").exists());
    }
}
