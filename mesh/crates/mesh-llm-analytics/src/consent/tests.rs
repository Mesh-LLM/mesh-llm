use super::*;

fn inputs() -> ConsentInputs {
    ConsentInputs {
        env_override: None,
        do_not_track: false,
        config: ConfigPreference::Unstated,
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
    configured.config = ConfigPreference::Stated(false);
    assert_eq!(configured.resolve(), Disposition::DisabledByConfig);
}

#[test]
fn an_unreadable_config_fails_closed() {
    // A config with one mistyped key anywhere fails to deserialize as a
    // whole. Treating that as "no preference" would silently re-enable
    // reporting for someone who had opted out.
    let mut broken = inputs();
    broken.config = ConfigPreference::Unreadable;
    assert_eq!(broken.resolve(), Disposition::DisabledConfigUnreadable);
    assert!(!broken.resolve().is_enabled());
}

#[test]
fn an_absent_config_is_not_treated_as_unreadable() {
    // No config at all is an ordinary first run, not a failure.
    let mut absent = inputs();
    absent.config = ConfigPreference::Unstated;
    assert_eq!(absent.resolve(), Disposition::EnabledByDefault);
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
    forced_on.config = ConfigPreference::Stated(false);
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
        Disposition::DisabledConfigUnreadable,
        Disposition::DisabledNoKey,
        Disposition::DisabledInCi,
    ] {
        assert!(!disposition.explain().is_empty());
    }
}

#[test]
fn normalize_host_trims_and_strips_trailing_slashes() {
    // Guards the URL join in the client, which appends `/batch/`: a host that
    // keeps its trailing slash would produce `//batch/`.
    assert_eq!(
        normalize_host(" https://self.example/ ").as_deref(),
        Some("https://self.example")
    );
    assert_eq!(
        normalize_host("https://self.example//").as_deref(),
        Some("https://self.example")
    );
    // An override that is only whitespace or slashes names no host, so it must
    // fall back to the default rather than reach the client as an empty one.
    assert_eq!(normalize_host("   "), None);
    assert_eq!(normalize_host("/"), None);
    // The default host must survive normalization unchanged, or the fallback
    // would silently differ from the documented endpoint.
    assert_eq!(
        normalize_host(DEFAULT_POSTHOG_HOST).as_deref(),
        Some(DEFAULT_POSTHOG_HOST)
    );
}
