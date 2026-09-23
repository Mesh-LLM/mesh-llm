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
    assert_eq!(
        config_preference(Some(&path)),
        ConfigPreference::Stated(false)
    );
}

#[test]
fn enable_and_disable_round_trip() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);

    dispatch_analytics_command(&AnalyticsCommand::Disable, Some(&path)).expect("disable");
    assert_eq!(
        config_preference(Some(&path)),
        ConfigPreference::Stated(false)
    );

    dispatch_analytics_command(&AnalyticsCommand::Enable, Some(&path)).expect("enable");
    assert_eq!(
        config_preference(Some(&path)),
        ConfigPreference::Stated(true)
    );
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
    assert_eq!(config_preference(Some(&path)), ConfigPreference::Unstated);
}

#[test]
fn an_unreadable_config_fails_closed_rather_than_reading_as_consent() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    std::fs::write(&path, "this is not valid toml {{{").expect("seed");

    // One mistyped key anywhere fails the whole document. That must not
    // silently re-enable reporting for someone who had opted out.
    assert_eq!(config_preference(Some(&path)), ConfigPreference::Unreadable);
    assert!(
        !ConsentInputs::from_env(config_preference(Some(&path)))
            .resolve()
            .is_enabled()
    );

    // Status still works: finding out what is reported must not depend on
    // the config parsing.
    dispatch_analytics_command(&AnalyticsCommand::Status { json: true }, Some(&path))
        .expect("status with broken config");
}

#[test]
fn status_does_not_create_an_install_identifier() {
    let dir = TempDir::new().expect("tempdir");
    let path = config_in(&dir);
    dispatch_analytics_command(&AnalyticsCommand::Status { json: true }, Some(&path))
        .expect("status");
    // Creating it here would consume the first-run signal, so the real first
    // run would never report as one.
    assert!(
        !dir.path()
            .join(mesh_llm_analytics::INSTALL_ID_FILE)
            .exists()
    );
}
