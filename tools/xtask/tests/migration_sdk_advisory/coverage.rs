use super::*;
use crate::automation::sdk_advisory::selection::select;
use std::collections::BTreeSet;

#[test]
fn every_eligible_product_enumerates_both_models_and_all_three_clients() {
    let expected = BTreeSet::from([
        ("smollm2-q8-inference", "scripts/ci-openai-python-smoke.py"),
        ("smollm2-q8-inference", "scripts/ci-litellm-smoke.py"),
        (
            "smollm2-q8-inference",
            "scripts/ci-langchain-openai-smoke.py",
        ),
        ("family-granite-hybrid", "scripts/ci-openai-python-smoke.py"),
        ("family-granite-hybrid", "scripts/ci-litellm-smoke.py"),
        (
            "family-granite-hybrid",
            "scripts/ci-langchain-openai-smoke.py",
        ),
    ]);
    for (event, run, artifacts) in [
        (LINUX_EVENT, LINUX_RUN, LINUX_ARTIFACTS),
        (MACOS_EVENT, MACOS_RUN, MACOS_ARTIFACTS),
    ] {
        let producer = admit(&CONTROLLER, Trigger::WorkflowRun(event), run).expect("producer");

        let selection = select(&producer, &catalog(), artifacts).expect("selected products");

        for product in selection.products {
            let cases = serde_json::to_value(product.cases).expect("cases JSON");
            let cases = cases.as_array().expect("cases array");
            let pairs: BTreeSet<_> = cases
                .iter()
                .map(|case| {
                    (
                        case["model"].as_str().expect("model ID"),
                        case["client"].as_str().expect("client script"),
                    )
                })
                .collect();
            assert_eq!(cases.len(), 6);
            assert_eq!(pairs, expected);
        }
    }
}

#[test]
fn selected_model_ids_resolve_to_frozen_source_owned_pins() {
    let manifest = value(include_bytes!(
        "../../../../ci/model-artifacts/manifests/product-smoke.json"
    ));
    let expected = [
        (
            "smollm2-q8-inference",
            "9e6855bc4be717fca1ef21360a1db4b29d5c559a",
            "c4a3dd037301b6ecea31d6da37f5cd793ead920dd5ddfe6d589294628d6ce66a",
        ),
        (
            "family-granite-hybrid",
            "a864f823cce6e6048b5752e2816fe7a23987d790",
            "0a8d6a7373602fadfba274a640ba784b86cc6847f1c67f1b0a90fa2ec266b7fb",
        ),
    ];
    let producer = admit(&CONTROLLER, Trigger::Manual(MANUAL_EVENT), LINUX_RUN).expect("producer");

    let selected = serde_json::to_value(
        select(&producer, &catalog(), LINUX_ARTIFACTS).expect("selected CUDA artifact"),
    )
    .expect("selection JSON");

    assert_eq!(selected["model_cadence"], "main");
    for (id, revision, sha256) in expected {
        assert!(
            selected["products"][0]["cases"]
                .as_array()
                .expect("cases")
                .iter()
                .any(|case| case["model"] == id)
        );
        let artifact = manifest["artifacts"]
            .as_array()
            .expect("models")
            .iter()
            .find(|model| model["id"] == id)
            .expect("source-owned model");
        assert_eq!(artifact["revision"], revision);
        assert_eq!(artifact["sha256"], sha256);
        assert!(
            artifact["cadences"]
                .as_array()
                .expect("cadences")
                .contains(&Value::from("main"))
        );
    }
}
