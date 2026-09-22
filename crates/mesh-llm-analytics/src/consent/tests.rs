use super::*;

fn inputs() -> ConsentInputs {
    ConsentInputs {
        env_override: None,
        do_not_track: false,
        config_enabled: None,
        has_key: true,
        in_ci: false,
    }
}

#[test]
fn defaults_to_enabled_when_a_key_exists() {
    assert_eq!(inputs().resolve(), Disposition::EnabledByDefault);
    assert!(inputs().resolve().is_enabled());
}

#[test]
fn a_build_without_a_key_never_reports() {
    let mut without_key = inputs();
    without_key.has_key = false;
    // Even an explicit opt-in cannot turn reporting on with nowhere to send it.
    without_key.env_override = Some("1".to_owned());
    assert_eq!(without_key.resolve(), Disposition::DisabledNoKey);
    assert!(!without_key.resolve().is_enabled());
}

#[test]
fn do_not_track_disables_reporting() {
    let mut dnt = inputs();
    dnt.do_not_track = true;
    assert_eq!(dnt.resolve(), Disposition::DisabledByDoNotTrack);
}

#[test]
fn config_opt_out_disables_reporting() {
    let mut configured = inputs();
    configured.config_enabled = Some(false);
    assert_eq!(configured.resolve(), Disposition::DisabledByConfig);
}

#[test]
fn ci_is_excluded_by_default() {
    let mut ci = inputs();
    ci.in_ci = true;
    assert_eq!(ci.resolve(), Disposition::DisabledInCi);
}

#[test]
fn explicit_env_override_beats_every_other_signal() {
    let mut forced_off = inputs();
    forced_off.env_override = Some("0".to_owned());
    assert_eq!(forced_off.resolve(), Disposition::DisabledByEnv);

    // An explicit opt-in is honored in CI and under DO_NOT_TRACK, so a
    // deliberate end-to-end test of this path is still possible.
    let mut forced_on = inputs();
    forced_on.env_override = Some("true".to_owned());
    forced_on.in_ci = true;
    forced_on.do_not_track = true;
    forced_on.config_enabled = Some(false);
    assert_eq!(forced_on.resolve(), Disposition::EnabledByEnv);
}

#[test]
fn truthiness_accepts_the_usual_spellings() {
    for value in ["1", "true", "TRUE", "yes", "on", " on "] {
        assert!(is_truthy(value), "{value:?} should be truthy");
    }
    for value in ["0", "false", "no", "off", "", "maybe"] {
        assert!(!is_truthy(value), "{value:?} should not be truthy");
    }
}

#[test]
fn every_disposition_explains_itself() {
    for disposition in [
        Disposition::EnabledByDefault,
        Disposition::EnabledByEnv,
        Disposition::DisabledByEnv,
        Disposition::DisabledByDoNotTrack,
        Disposition::DisabledByConfig,
        Disposition::DisabledNoKey,
        Disposition::DisabledInCi,
    ] {
        assert!(!disposition.explain().is_empty());
    }
}

#[test]
fn ingestion_host_strips_trailing_slashes() {
    // Guards the URL join in the client, which appends `/batch/`.
    assert!(!DEFAULT_POSTHOG_HOST.ends_with('/'));
}
