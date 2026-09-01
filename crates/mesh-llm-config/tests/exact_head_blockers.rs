use mesh_llm_config::{
    ConfigDiagnosticSeverity, MeshConfig, WIRING_MANIFEST, WiringBehavior,
    validate_config_diagnostics,
};
use std::collections::BTreeSet;

fn rejected_paths(raw: &str) -> BTreeSet<String> {
    let config: MeshConfig = toml::from_str(raw).expect("config parses before validation");
    validate_config_diagnostics(&config)
        .into_iter()
        .filter(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
        .filter_map(|diagnostic| diagnostic.path.map(|path| path.render()))
        .collect()
}

#[test]
fn tcp_plugin_control_is_rejected_case_insensitively_without_echoing_secrets() {
    for url in [
        "tcp://127.0.0.1:19091",
        "TcP://127.0.0.1:19091",
        "tcp://user:secret@127.0.0.1:19091/control?token=private#fragment",
    ] {
        let raw = format!("[[plugin]]\nname = \"demo\"\nurl = {url:?}\n");
        let config: MeshConfig = toml::from_str(&raw).expect("plugin config parses");
        let diagnostics = validate_config_diagnostics(&config);
        let diagnostic = diagnostics
            .iter()
            .find(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
            .expect("tcp plugin control must fail static validation");
        let rendered = format!("{diagnostic:?}");

        assert_eq!(
            diagnostic
                .path
                .as_ref()
                .map(|path| path.render())
                .as_deref(),
            Some("plugin[0].url")
        );
        assert!(rendered.contains("authenticated capability handshake"));
        assert!(!rendered.contains("user"));
        assert!(!rendered.contains("secret"));
        assert!(!rendered.contains("token=private"));
    }
}

#[test]
fn every_silent_no_op_and_false_mmap_control_fails_static_validation() {
    let actual = rejected_paths(
        r#"
[defaults.hardware]
model_runtime = "cuda"
fit_target_mib = 8192
split_mode = "row"
main_gpu = 0
fit_context = true
lora_adapters = ["adapter.gguf"]
control_vectors = ["control.gguf"]
check_tensors = true
direct_io = true
repack = true
op_offload = true
no_host_buffer = true
use_mmap_prefetch = true
use_mmap_buffer = true
warmup = true
"#,
    );
    let expected = WIRING_MANIFEST
        .iter()
        .filter(|entry| entry.behavior == WiringBehavior::SilentNoOp)
        .map(|entry| format!("defaults.{}", entry.path))
        .collect();

    assert_eq!(actual, expected);
}

#[test]
fn fit_target_mib_is_accepted_for_tuning_and_live_fit_planning() {
    let config: MeshConfig = toml::from_str(
        r#"
[defaults.hardware]
fit_target_mib = 8192
"#,
    )
    .expect("fit target config parses before validation");
    let diagnostics = validate_config_diagnostics(&config);

    assert!(
        diagnostics.is_empty(),
        "the GPU tuner authors fit_target_mib and the Skippy resolver consumes it: {diagnostics:?}"
    );
}

#[test]
fn closeout_audit_has_structured_forward_evidence_for_every_pr8_row() {
    let audit_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/CONFIGURATION_PR8_CLOSEOUT_AUDIT.md");
    let audit = std::fs::read_to_string(&audit_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", audit_path.display()));
    let forward = audit
        .split_once("## Forward audit")
        .and_then(|(_, suffix)| suffix.split_once("## Reverse audit"))
        .map(|(section, _)| section)
        .expect("audit has a bounded forward section");

    for entry in WIRING_MANIFEST.iter().filter(|entry| entry.owner == "PR8") {
        let prefix = format!("| `{}` |", entry.path);
        let row = forward
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("forward audit is missing {}", entry.path));
        assert!(
            row.matches('|').count() >= 6,
            "incomplete forward row: {row}"
        );
        assert!(row.contains("::"), "row lacks executable evidence: {row}");
    }
}
