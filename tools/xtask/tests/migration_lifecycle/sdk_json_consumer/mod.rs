mod fixture;
mod routing;
use crate::workflow_yaml;
use fixture::Fixture;
use serde_json::json;
use std::fs;

#[test]
fn sdk_json_consumer_composition_orders_reader_readiness_and_publication_without_fallback() {
    let rows = json!([{"id":"runtime","backend":"cpu","supported":true}]);
    for report in [json!({"catalogs":{},"runtimes":rows}), rows] {
        let fixture = Fixture::new(&report);
        let result = fixture.run();
        assert!(
            result.process.success(),
            "{:?}: {}",
            result.process,
            String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes())
        );
        assert_eq!(
            fs::read_to_string(fixture.root.join("events")).unwrap(),
            "sdk\nreadiness\n"
        );
        assert!(fixture.root.join("product.tar.gz").is_file());
        let outputs = fs::read_to_string(fixture.root.join("outputs")).unwrap();
        assert!(
            outputs
                .lines()
                .any(|line| line.starts_with("archive_path="))
        );
        assert!(
            String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes())
                .contains("Reusing compatible native runtime")
        );
        assert!(!fixture.root.join("fallback-ran").exists());
        assert!(!fixture.root.join("forbidden").exists());
    }
}

#[test]
fn sdk_json_consumer_bad_report_blocks_readiness_archive_and_workflow_outputs() {
    for (report, diagnostic) in [
        (
            json!({"runtimes":{}}),
            "native runtime compatibility output must be",
        ),
        (
            json!({"runtimes":[{"id":"runtime","backend":"cpu","supported":false}]}),
            "expected exactly one compatible adjacent native runtime",
        ),
    ] {
        let fixture = Fixture::new(&report);
        let result = fixture.run();
        assert!(!result.process.success());
        assert!(
            String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes())
                .contains(diagnostic)
        );
        assert_eq!(
            fs::read_to_string(fixture.root.join("events")).unwrap(),
            "sdk\n"
        );
        for absent in ["product.tar.gz", "outputs", "fallback-ran", "forbidden"] {
            assert!(!fixture.root.join(absent).exists(), "unexpected {absent}");
        }
    }
}

#[test]
fn sdk_json_consumer_rejects_ambient_configuration_leaks_before_reader_and_publication() {
    for variable in [
        "MESH_LLM_CONFIG",
        "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
        "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
    ] {
        let fixture = Fixture::new(&json!([{"id":"runtime","backend":"cpu","supported":true}]));
        let path = fixture.root.join("scripts/ci-prepare-native-runtime.sh");
        let source = fs::read_to_string(&path).unwrap();
        let removal = format!("-u {variable} \\");
        assert!(source.lines().any(|line| line.trim() == removal));
        let edited = source
            .lines()
            .filter(|line| line.trim() != removal)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(&path, edited).unwrap();
        let report = fixture.run();
        assert!(!report.process.success());
        for absent in [
            "sdk-reader-ran",
            "events",
            "outputs",
            "product.tar.gz",
            "fallback-ran",
        ] {
            assert!(!fixture.root.join(absent).exists(), "unexpected {absent}");
        }
    }
}
