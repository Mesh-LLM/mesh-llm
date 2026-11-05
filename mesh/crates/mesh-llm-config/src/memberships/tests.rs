use super::*;

const VALID: &str = r#"
version = 1
model_id = "shared"
state_dir = "state"
[[memberships]]
name = "work.mesh"
join_token_file = "work.token"
trust_store = "work-trust.json"
owner_key = "owner.json"
"#;

#[test]
fn documented_manifest_matches_the_local_schema() {
    let documentation = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/SHARED_SERVING.md"
    ));
    let raw = documentation
        .split_once("```toml\n")
        .unwrap()
        .1
        .split_once("\n```")
        .unwrap()
        .0;
    let config = load(raw).1.unwrap();
    assert_eq!(config.memberships.len(), 2);
    assert_eq!(config.concurrency, 2);
}

fn load(raw: &str) -> (tempfile::TempDir, Result<MembershipsConfig>) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("memberships.toml");
    std::fs::write(&path, raw).unwrap();
    let result = MembershipsConfig::load(&path);
    (directory, result)
}

#[test]
fn paths_are_relative_to_manifest_without_reading_credentials_or_creating_state() {
    let (directory, result) = load(VALID);
    let config = result.unwrap();
    assert_eq!(config.state_dir, directory.path().join("state"));
    assert!(!config.state_dir.exists());
    let member = &config.memberships[0];
    assert_eq!(member.join_token_file, directory.path().join("work.token"));
    assert_eq!(
        member.trust_store,
        Some(directory.path().join("work-trust.json"))
    );
    assert_eq!(member.owner_key, Some(directory.path().join("owner.json")));
    assert_eq!(
        (
            config.concurrency,
            config.queue_capacity,
            config.membership_queue_capacity
        ),
        (1, 32, 16)
    );
    assert_eq!(
        (
            config.context_size,
            config.gpu_layers,
            config.management_port
        ),
        (4096, -1, 0)
    );
    assert!(!config.lan_only);
}

#[test]
fn absolute_paths_and_serialization_round_trip() {
    let (directory, result) = load(VALID);
    let expected = result.unwrap();
    let path = directory.path().join("again.toml");
    std::fs::write(&path, toml::to_string(&expected).unwrap()).unwrap();
    let actual = MembershipsConfig::load(&path).unwrap();
    assert_eq!(actual.state_dir, expected.state_dir);
    assert_eq!(
        actual.memberships[0].join_token_file,
        expected.memberships[0].join_token_file
    );
}

#[test]
fn rejects_unknown_fields_inline_tokens_and_malformed_input_without_echoing_values() {
    for raw in [
        format!("{VALID}\njoin_token = 'SECRET'"),
        format!("unknown = 'SECRET'\n{VALID}"),
        VALID.replace("version = 1", "version = 'SECRET'"),
        "broken = SECRET".into(),
    ] {
        let error = load(&raw).1.unwrap_err().to_string();
        assert!(error.contains("invalid memberships TOML"));
        assert!(!error.contains("SECRET"));
    }
}

#[test]
fn rejects_invalid_limits_missing_fields_and_duplicate_memberships() {
    for raw in [
        VALID.replace("version = 1", "version = 2"),
        VALID.replace("version = 1", ""),
        VALID.replace("model_id = \"shared\"", "model_id = ''"),
        VALID.replace("state_dir = \"state\"", "state_dir = ''"),
        VALID.replace("work.token", ""),
        VALID.replace("work-trust.json", ""),
        format!("concurrency = 0\n{VALID}"),
        format!("concurrency = 65\n{VALID}"),
        format!("queue_capacity = 4097\n{VALID}"),
        format!("membership_queue_capacity = 4097\n{VALID}"),
        format!("context_size = 511\n{VALID}"),
        format!("gpu_layers = -2\n{VALID}"),
        format!("management_port = 65536\n{VALID}"),
        format!("{VALID}\n[[memberships]]\nname = 'work.mesh'\njoin_token_file = 'other.token'"),
        "version=1\nmodel_id='x'\nstate_dir='state'\nmemberships=[]".into(),
        format!("{VALID}{}", " ".repeat(65_536)),
    ] {
        assert!(load(&raw).1.is_err(), "accepted invalid manifest");
    }
}

#[test]
fn names_match_dynamic_membership_contract() {
    for name in ["work", "work.mesh", "mesh-1", "mesh_a"] {
        assert!(valid_membership_name(name));
    }
    for name in ["", ".hidden", "..", "../work", "Work", "a/b", "a\\b", "é"] {
        assert!(!valid_membership_name(name));
    }
    assert!(!valid_membership_name(&"a".repeat(65)));
}

#[test]
fn manifests_reject_windows_directory_aliases_without_changing_sdk_label_syntax() {
    for name in [
        "work.",
        "host.lock",
        "management.token",
        "con",
        "prn.txt",
        "aux",
        "nul.tar.gz",
        "com1",
        "com9.log",
        "lpt1",
        "lpt9.txt",
    ] {
        assert!(
            valid_membership_name(name),
            "SDK label syntax changed for {name}"
        );
        assert!(
            !valid_membership_directory_name(name),
            "unsafe directory {name}"
        );
        let error = load(&VALID.replace("work.mesh", name)).1.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid membership directory name")
        );
    }
    for name in ["work", "work.mesh", "conifer", "com10", "lpt0", "mesh-1"] {
        assert!(
            valid_membership_directory_name(name),
            "safe directory {name}"
        );
    }
}
