//! Typed UI build option admission fixtures.
use super::options::Options;
use std::time::Duration;
fn parse(values: &[&str]) -> crate::command::DynResult<Options> {
    Options::parse(
        &values
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>(),
    )
}
#[test]
fn ui_requires_directory_and_refuses_unknown_repeated_or_missing_values() {
    for args in [
        vec![],
        vec!["--ui-dir"],
        vec!["--ui-dir", ""],
        vec!["--foreign", "value"],
        vec!["--ui-dir", "first", "--ui-dir", "second"],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
}
#[test]
fn ui_default_timeout_and_explicit_timeout_are_bounded() {
    assert_eq!(
        parse(&["--ui-dir", "ui"]).unwrap().timeout,
        Duration::from_secs(1800)
    );
    assert_eq!(
        parse(&["--ui-dir", "ui", "--timeout-secs", "1"])
            .unwrap()
            .timeout,
        Duration::from_secs(1)
    );
    assert_eq!(
        parse(&["--ui-dir", "ui", "--timeout-secs", "3600"])
            .unwrap()
            .timeout,
        Duration::from_secs(3600)
    );
    for value in ["0", "3601", "-1", "invalid", "18446744073709551616"] {
        assert!(parse(&["--ui-dir", "ui", "--timeout-secs", value]).is_err());
    }
}
#[test]
fn ui_native_node_script_preserves_spaces_without_shell_interpolation() {
    let options = parse(&[
        "--ui-dir",
        "UI with spaces",
        "--pnpm-command",
        "Node with spaces.exe",
        "--pnpm-script",
        "pnpm path/pnpm.cjs",
        "--logs-dir",
        "output logs",
    ])
    .unwrap();
    assert_eq!(options.ui.to_str().unwrap(), "UI with spaces");
    assert_eq!(
        options.executable.unwrap().to_str().unwrap(),
        "Node with spaces.exe"
    );
    assert_eq!(
        options.pnpm_script.unwrap().to_str().unwrap(),
        "pnpm path/pnpm.cjs"
    );
    assert_eq!(options.logs.unwrap().to_str().unwrap(), "output logs");
}
#[test]
fn ui_script_requires_explicit_native_launcher() {
    assert!(parse(&["--ui-dir", "ui", "--pnpm-script", "pnpm.cjs"]).is_err());
}

#[test]
fn explicit_profile_is_typed_and_validated() {
    assert_eq!(
        parse(&["--ui-dir", "ui", "--profile", "ReLeAsE"])
            .unwrap()
            .profile,
        Some(super::policy::Profile::Release)
    );
    assert!(parse(&["--ui-dir", "ui", "--profile", "production"]).is_err());
    assert!(
        parse(&[
            "--ui-dir",
            "ui",
            "--profile",
            "debug",
            "--profile",
            "release"
        ])
        .is_err()
    );
}

#[test]
fn build_settings_include_vite_namespace_and_actual_vite_configuration() {
    for name in [
        "VITE_APP_VERSION",
        "VITE_FUTURE_SETTING",
        "TANSTACK_FILE_ROUTER",
        "MESH_UI_API_ORIGIN",
        "NODE_ENV",
    ] {
        assert!(super::options::is_build_setting(name));
    }
    for name in ["NPM_TOKEN", "MESH_LLM_AUTOMATION_BIN", "UI_FIXTURE_MODE"] {
        assert!(!super::options::is_build_setting(name));
    }
}
