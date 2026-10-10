use super::*;
use crate::automation::sdk_advisory::selection::select;

#[test]
fn linux_selection_uses_exact_ids_and_digests_not_the_prefix_decoy() {
    let producer =
        admit(&CONTROLLER, Trigger::WorkflowRun(LINUX_EVENT), LINUX_RUN).expect("producer");

    let selection = select(&producer, &catalog(), LINUX_ARTIFACTS).expect("exact artifacts");

    let selected: Vec<_> = selection
        .products
        .iter()
        .map(|product| (product.artifact_id.get(), product.artifact_name.as_str()))
        .collect();
    assert_eq!(
        selected,
        [
            (701, "ci-product-linux-amd64-cpu"),
            (702, "ci-product-linux-amd64-cuda")
        ]
    );
    let report = serde_json::to_value(&selection).expect("selection JSON");
    assert_eq!(
        report["source_sha"],
        "1111111111111111111111111111111111111111"
    );
    assert_eq!(
        report["products"][0]["artifact_digest"],
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    assert_eq!(
        report["products"][1]["artifact_digest"],
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    );
    assert_eq!(report["payload_verification"], "required_before_execution");
}

#[test]
fn macos_selection_binds_its_independent_run_and_source() {
    let producer =
        admit(&CONTROLLER, Trigger::WorkflowRun(MACOS_EVENT), MACOS_RUN).expect("producer");

    let selection = select(&producer, &catalog(), MACOS_ARTIFACTS).expect("Metal artifact");

    let report = serde_json::to_value(selection).expect("selection JSON");
    assert_eq!(report["producer_run_id"], 202);
    assert_eq!(
        report["source_sha"],
        "2222222222222222222222222222222222222222"
    );
    assert_eq!(report["products"][0]["artifact_id"], 801);
    assert_eq!(
        report["products"][0]["artifact_name"],
        "ci-product-macos-arm64-metal"
    );
}

#[test]
fn manual_selection_never_expands_to_the_producers_other_rows() {
    let producer =
        admit(&CONTROLLER, Trigger::Manual(MANUAL_EVENT), LINUX_RUN).expect("manual producer");

    let selection = select(&producer, &catalog(), LINUX_ARTIFACTS).expect("CUDA artifact");

    assert_eq!(selection.products.len(), 1);
    assert_eq!(selection.products[0].artifact_id.get(), 702);
}

#[test]
fn denied_artifact_fixtures_distinguish_missing_expired_and_mismatched() {
    let producer =
        admit(&CONTROLLER, Trigger::WorkflowRun(LINUX_EVENT), LINUX_RUN).expect("producer");
    let catalog = catalog();
    let mutations: Vec<Mutation> =
        serde_json::from_slice(include_bytes!("fixtures/denied-artifacts.json"))
            .expect("negative fixtures");
    assert_eq!(mutations.len(), 17);

    for mutation in mutations {
        let artifacts = mutated(LINUX_ARTIFACTS, &mutation);

        let error = select(&producer, &catalog, &artifacts).expect_err(&mutation.name);

        assert!(
            mutation.reason.matches(&error),
            "{}: {error:?}",
            mutation.name
        );
    }
}

#[test]
fn failed_exact_candidate_never_falls_back_to_a_second_match() {
    let producer =
        admit(&CONTROLLER, Trigger::WorkflowRun(LINUX_EVENT), LINUX_RUN).expect("producer");
    let mut artifacts = value(LINUX_ARTIFACTS);
    artifacts["artifacts"][0]["name"] = Value::from("ci-product-linux-amd64-cpu");
    artifacts["artifacts"][1]["expired"] = Value::from(true);

    let result = select(&producer, &catalog(), &bytes(&artifacts));

    assert!(matches!(result, Err(Rejected::AmbiguousArtifact)));
}

#[test]
fn complete_empty_inventory_fails_instead_of_selecting_no_work() {
    let producer =
        admit(&CONTROLLER, Trigger::WorkflowRun(LINUX_EVENT), LINUX_RUN).expect("producer");

    let result = select(
        &producer,
        &catalog(),
        br#"{"total_count":0,"artifacts":[]}"#,
    );

    assert!(matches!(result, Err(Rejected::MissingArtifact)));
}

#[test]
fn one_missing_required_row_does_not_return_partial_success() {
    let producer =
        admit(&CONTROLLER, Trigger::WorkflowRun(LINUX_EVENT), LINUX_RUN).expect("producer");
    let mut artifacts = value(LINUX_ARTIFACTS);
    artifacts["artifacts"]
        .as_array_mut()
        .expect("array")
        .pop()
        .expect("CUDA fixture");
    artifacts["total_count"] = Value::from(2);

    let result = select(&producer, &catalog(), &bytes(&artifacts));

    assert!(matches!(result, Err(Rejected::MissingArtifact)));
}
