use super::*;

#[test]
fn accepts_catalog_style_model_names() {
    for name in [
        "Qwen2.5-32B-Instruct-Q4_K_M",
        "Qwen/Qwen2.5-7B",
        "llama-3.1-8b",
        "gpt-oss-120b",
    ] {
        assert!(Label::sanitize(name).is_some(), "rejected {name}");
    }
}

#[test]
fn accepts_the_build_version_strings_this_crate_reports() {
    // Regression: `+` build metadata was rejected, which silently redacted
    // the version property on every event from a development build.
    for version in [
        "0.76.0",
        "0.76.0-rc8",
        "0.76.0+gABCDEF",
        "0.76.0-rc8+gABCDEF.dirty",
    ] {
        assert!(Label::sanitize(version).is_some(), "rejected {version}");
    }
}

#[test]
fn rejects_filesystem_paths() {
    for path in [
        "/Users/dan/models/private.gguf",
        "../../etc/passwd",
        "~/models/secret.gguf",
        "C:\\models\\secret.gguf",
        "models/nested/deep.gguf",
        "./relative.gguf",
    ] {
        assert!(Label::sanitize(path).is_none(), "accepted {path}");
    }
}

#[test]
fn rejects_prose_and_control_characters() {
    for value in [
        "summarize this document for me",
        "token sk-abc\ndef",
        "has\ttab",
        "",
    ] {
        assert!(Label::sanitize(value).is_none(), "accepted {value:?}");
    }
}

#[test]
fn rejects_overlong_values() {
    let long = "a".repeat(MAX_LABEL_LEN + 1);
    assert!(Label::sanitize(&long).is_none());
    let at_limit = "a".repeat(MAX_LABEL_LEN);
    assert!(Label::sanitize(&at_limit).is_some());
}

#[test]
fn unsanitizable_values_redact_rather_than_leak() {
    let label = Label::sanitize_or_redact("/Users/dan/models/private.gguf");
    assert_eq!(label.as_str(), "redacted");
}

#[test]
fn event_names_are_unique_and_snake_case() {
    let events = [
        Event::InstallFirstRun,
        Event::CliCommand,
        Event::ServeStarted,
        Event::ServeStopped,
        Event::ModelLoaded,
        Event::ModelDownload,
        Event::HardwareProfile,
    ];
    let mut names: Vec<_> = events.iter().map(|event| event.name()).collect();
    names.sort_unstable();
    let count = names.len();
    names.dedup();
    assert_eq!(names.len(), count, "duplicate event name");
    assert!(
        names
            .iter()
            .all(|name| name.chars().all(|c| c.is_ascii_lowercase() || c == '_')),
        "event names must be snake_case",
    );
}

#[test]
fn buckets_cover_their_ranges() {
    assert_eq!(bucket_count(0), "0");
    assert_eq!(bucket_count(4), "3-4");
    assert_eq!(bucket_count(1_000), "33+");

    const GB: u64 = 1024 * 1024 * 1024;
    assert_eq!(bucket_gigabytes(0), "0-8");
    assert_eq!(bucket_gigabytes(36 * GB), "32-64");
    assert_eq!(bucket_gigabytes(512 * GB), "256+");

    assert_eq!(bucket_duration_secs(30), "under_1m");
    assert_eq!(bucket_duration_secs(3_600), "1h-6h");
    assert_eq!(bucket_duration_secs(90_000), "over_24h");
}

#[test]
fn properties_keep_last_write_per_key() {
    let properties = Properties::new()
        .with("backend", "metal")
        .with("backend", "cuda");
    let collected: Vec<_> = properties.entries().collect();
    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0].1, &Value::Static("cuda"));
}

#[test]
fn slugs_real_device_names() {
    for (raw, expected) in [
        ("Apple M4 Max", "apple-m4-max"),
        ("NVIDIA GeForce RTX 4090", "nvidia-geforce-rtx-4090"),
        ("AMD Radeon RX 7900 XTX", "amd-radeon-rx-7900-xtx"),
        ("Intel(R) Arc(TM) A770", "intel-r-arc-tm-a770"),
        ("  Apple   M1   ", "apple-m1"),
    ] {
        assert_eq!(
            Label::slug(raw).map(|label| label.as_str().to_owned()),
            Some(expected.to_owned()),
            "slug mismatch for {raw}",
        );
    }
}

#[test]
fn slugging_does_not_launder_a_path_into_a_valid_label() {
    // Slugging must widen what is expressible, never what is permitted.
    for path in [
        "/Users/dan/models/private.gguf",
        "../../etc/passwd",
        "C:\\Users\\dan\\secret.gguf",
        "/home/dan/my model.gguf",
    ] {
        assert!(Label::slug(path).is_none(), "slug accepted {path}");
        assert_eq!(Label::slug_or_redact(path).as_str(), "redacted");
    }
}

#[test]
fn slugging_rejects_prose() {
    // A sentence slugs into a long hyphenated run; the length cap stops it.
    let prose = "please summarize this confidential document about our merger plans in detail";
    assert!(Label::slug(prose).is_none());
}
