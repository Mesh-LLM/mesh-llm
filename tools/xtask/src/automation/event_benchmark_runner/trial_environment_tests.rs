use super::*;
use serde_json::json;

fn input(pairs: &[(&str, &str)]) -> BTreeMap<OsString, OsString> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).into(), (*value).into()))
        .collect()
}

#[test]
fn child_overrides_and_snapshot_agree_for_every_mode() {
    for mode in [Mode::Production, Mode::EventDisabled, Mode::Off] {
        let child = effective(input(&[(GATE, "0"), (SELECTOR, "stale")]), mode);
        assert_eq!(child.get(&OsString::from(GATE)), Some(&OsString::from("1")));
        assert_eq!(
            child.get(&OsString::from(SELECTOR)),
            Some(&OsString::from(mode.label()))
        );
        let report = snapshot(&child);
        assert_eq!(report[GATE].value, true);
        assert_eq!(report[SELECTOR].value, mode.label());
        assert!(!report[GATE].redacted && !report[SELECTOR].redacted);
    }
}

#[test]
fn caller_environment_is_preserved_and_only_mesh_names_are_persisted() {
    let original = input(&[("PATH", "/bin"), ("API_TOKEN", "private"), (PARSER, "json")]);
    let child = effective(original.clone(), Mode::Production);
    assert_eq!(original.len(), 3);
    assert_eq!(
        child.get(&OsString::from("PATH")),
        original.get(&OsString::from("PATH"))
    );
    let report = snapshot(&child);
    assert_eq!(report.len(), 3);
    assert_eq!(report[PARSER].value, "json");
    let text = serde_json::to_string(&report).unwrap();
    assert!(!text.contains("private") && !text.contains("/bin"));
}

#[test]
fn unknown_empty_and_sensitive_values_retain_presence_without_raw_values() {
    let env = input(&[
        ("MESH_LLM_FUTURE", "hidden-future"),
        ("MESH_LLM_EMPTY", ""),
        ("MESH_LLM_AUTH_URL", "hidden-url"),
    ]);
    let report = snapshot(&env);
    for name in ["MESH_LLM_FUTURE", "MESH_LLM_EMPTY", "MESH_LLM_AUTH_URL"] {
        assert!(report[name].redacted);
        assert_eq!(report[name].value, REDACTED);
    }
    let text = serde_json::to_string(&report).unwrap();
    assert!(!text.contains("hidden-future") && !text.contains("hidden-url"));
}

#[test]
fn sensitive_name_check_cannot_be_bypassed_by_case() {
    for part in SENSITIVE {
        assert!(sensitive(&format!("MESH_LLM_{}", part.to_lowercase())));
    }
    assert!(!sensitive(PARSER));
}

#[test]
fn gate_boolean_requires_the_exact_runtime_wire_value() {
    for raw in ["0", "true", "", "01", " 1", "1"] {
        assert_eq!(
            snapshot(&input(&[(GATE, raw)]))[GATE].value,
            json!(raw == "1")
        );
    }
}

#[cfg(unix)]
#[test]
fn nonunicode_allowlisted_values_are_redacted_and_nonunicode_names_are_omitted() {
    use std::os::unix::ffi::OsStringExt;
    let invalid = OsString::from_vec(vec![0xff]);
    let env = BTreeMap::from([(PARSER.into(), invalid.clone()), (invalid, "hidden".into())]);
    let report = snapshot(&env);
    assert_eq!(report.len(), 1);
    assert!(report[PARSER].redacted);
    assert_eq!(report[PARSER].value, REDACTED);
}
