use super::*;
use tempfile::TempDir;

fn config_in(dir: &TempDir) -> std::path::PathBuf {
    dir.path().join("config.toml")
}

#[test]
fn disable_writes_a_durable_opt_out() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);

    dispatch_analytics_command(&AnalyticsCommand::Disable, Some(&path)).expect("disable");

    let raw = std::fs::read_to_string(&path).expect("config written");
    assert!(raw.contains("[analytics]"), "{raw}");
    assert!(raw.contains("enabled = false"), "{raw}");
    assert_eq!(configured_enabled(Some(&path)), Some(false));
}

#[test]
fn enable_and_disable_round_trip() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);

    dispatch_analytics_command(&AnalyticsCommand::Disable, Some(&path)).expect("disable");
    assert_eq!(configured_enabled(Some(&path)), Some(false));

    dispatch_analytics_command(&AnalyticsCommand::Enable, Some(&path)).expect("enable");
    assert_eq!(configured_enabled(Some(&path)), Some(true));
}

#[test]
fn opting_out_preserves_the_rest_of_a_hand_edited_config() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    std::fs::write(
        &path,
        "# keep this comment\n[telemetry]\nenabled = true\nservice_name = \"mesh-llm\"\n",
    )
    .expect("seed config");

    dispatch_analytics_command(&AnalyticsCommand::Disable, Some(&path)).expect("disable");

    let raw = std::fs::read_to_string(&path).expect("config");
    assert!(raw.contains("# keep this comment"), "comment lost: {raw}");
    assert!(
        raw.contains("service_name = \"mesh-llm\""),
        "setting lost: {raw}"
    );
    assert!(raw.contains("enabled = false"), "opt-out missing: {raw}");

    // The operator-facing OTLP section is untouched by the analytics opt-out.
    let config = load_config(Some(&path)).expect("reload");
    assert_eq!(config.telemetry.enabled, Some(true));
    assert_eq!(config.analytics.enabled, Some(false));
}

#[test]
fn status_reports_without_a_config_file_present() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    assert!(!path.exists());

    // No config is an ordinary first-run state, not an error.
    dispatch_analytics_command(&AnalyticsCommand::Status { json: true }, Some(&path))
        .expect("status without config");
    assert_eq!(configured_enabled(Some(&path)), None);
}

#[test]
fn status_tolerates_an_unreadable_config() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    std::fs::write(&path, "this is not valid toml {{{").expect("seed");

    // Finding out what is reported must not depend on the config parsing.
    assert_eq!(configured_enabled(Some(&path)), None);
    dispatch_analytics_command(&AnalyticsCommand::Status { json: true }, Some(&path))
        .expect("status with broken config");
}
