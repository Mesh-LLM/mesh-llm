use super::descriptors::{DESCRIPTOR_GROUPS, RawKind};
use super::is_safe_summary;
use super::raw_options::raw_option_is_allowed;
use super::vocabulary::{is_boolean_option, is_redacted_marker, is_static_summary_token};

#[test]
fn vocabulary_classifies_static_boolean_and_redacted_tokens() {
    assert!(is_static_summary_token("mesh-llm"));
    assert!(is_static_summary_token("run-benchmark"));
    assert!(!is_static_summary_token("private-model"));

    assert!(is_boolean_option("--json"));
    assert!(is_boolean_option("--no-speculative-tune"));
    assert!(!is_boolean_option("--backend"));

    assert!(is_redacted_marker("model"));
    assert!(is_redacted_marker("--relay-auth"));
    assert!(!is_redacted_marker("[REDACTED]"));
}

#[test]
fn raw_options_are_limited_to_candidate_producer_contexts() {
    let backend = ["mesh-llm", "gpus", "run-benchmark", "--backend", "cuda"];
    let mode = ["mesh-llm", "runtime", "guardrails", "--mode", "metrics"];
    let port = ["mesh-llm", "doctor", "split", "--json", "--port", "41731"];

    assert!(raw_option_is_allowed(&backend, 3, "--backend"));
    assert!(raw_option_is_allowed(&mode, 3, "--mode"));
    assert!(raw_option_is_allowed(&port, 4, "--port"));
    assert!(!raw_option_is_allowed(&backend, 3, "--mode"));
}

#[test]
fn benchmark_tune_rejects_each_speculative_option_when_tuning_is_disabled() {
    let speculative_options = [
        "--speculative-types",
        "--spec-draft-models",
        "--spec-draft-max-tokens",
        "--spec-draft-min-tokens",
        "--spec-draft-acceptance-threshold",
        "--spec-draft-split-probability",
        "--spec-ngram-min",
        "--spec-ngram-max",
    ];

    for option in speculative_options {
        let summary = format!("mesh-llm benchmark tune --no-speculative-tune {option} [REDACTED]");
        assert!(!is_safe_summary(&summary), "accepted conflict: {option}");
    }
}

#[test]
fn ports_accept_only_ascii_decimal_u16_values() {
    for port in ["0", "1", "65535"] {
        let summary = format!("mesh-llm status --port {port}");
        assert!(is_safe_summary(&summary), "rejected valid port: {port}");
    }

    for port in ["+1", "-1", "65536", "1.0", "١"] {
        let summary = format!("mesh-llm status --port {port}");
        assert!(!is_safe_summary(&summary), "accepted invalid port: {port}");
    }
}

#[test]
fn every_descriptor_accepts_its_complete_canonical_shape() {
    for descriptor in DESCRIPTOR_GROUPS.iter().flat_map(|group| group.iter()) {
        let mut prefix = descriptor.path.to_vec();
        match descriptor.raw {
            RawKind::Backend => prefix.extend_from_slice(&["--backend", "cuda"]),
            RawKind::Mode => prefix.extend_from_slice(&["--mode", "metrics"]),
            RawKind::None => {}
        }
        let mut tokens = prefix.clone();
        tokens.extend_from_slice(descriptor.booleans);
        if descriptor.has_port {
            tokens.extend_from_slice(&["--port", "41731"]);
        }
        for marker in descriptor.redacted {
            tokens.extend_from_slice(&[*marker, "[REDACTED]"]);
        }
        if descriptor
            .conflicts
            .iter()
            .any(|pair| pair.iter().all(|flag| tokens.contains(flag)))
        {
            assert!(!is_safe_summary(&tokens.join(" ")));
            continue;
        }
        let summary = tokens.join(" ");
        if tokens.len() <= 32 && summary.chars().count() <= 256 {
            assert!(is_safe_summary(&summary), "descriptor rejected: {summary}");
        } else {
            for marker in descriptor.booleans.iter().chain(descriptor.redacted) {
                let mut single = prefix.clone();
                single.push(*marker);
                if descriptor.redacted.contains(marker) {
                    single.push("[REDACTED]");
                }
                assert!(
                    is_safe_summary(&single.join(" ")),
                    "descriptor marker rejected: {marker}"
                );
            }
        }
    }
}
