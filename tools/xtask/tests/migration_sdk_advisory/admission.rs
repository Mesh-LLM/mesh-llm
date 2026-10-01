use super::*;
use crate::automation::sdk_advisory::rows::ProductRow;

#[test]
fn main_linux_admits_both_eligible_rows() {
    let producer = admit(&CONTROLLER, Trigger::WorkflowRun(LINUX_EVENT), LINUX_RUN)
        .expect("authorized producer");

    assert_eq!(
        producer.rows(),
        [ProductRow::LinuxCpu, ProductRow::LinuxCuda]
    );
}

#[test]
fn main_macos_admits_only_metal() {
    let producer = admit(&CONTROLLER, Trigger::WorkflowRun(MACOS_EVENT), MACOS_RUN)
        .expect("authorized producer");

    assert_eq!(producer.rows(), [ProductRow::MacosMetal]);
}

#[test]
fn manual_selection_is_one_row_from_a_successful_main_producer() {
    let producer = admit(&CONTROLLER, Trigger::Manual(MANUAL_EVENT), LINUX_RUN)
        .expect("explicit eligible row");

    assert_eq!(producer.rows(), [ProductRow::LinuxCuda]);
}

#[test]
fn denied_event_source_and_workflow_fixtures_have_distinct_reasons() {
    let mutations: Vec<Mutation> =
        serde_json::from_slice(include_bytes!("fixtures/denied-producers.json"))
            .expect("negative fixtures");
    assert_eq!(mutations.len(), 27);

    for mutation in mutations {
        let run = mutated(LINUX_RUN, &mutation);

        let error =
            admit(&CONTROLLER, Trigger::WorkflowRun(LINUX_EVENT), &run).expect_err(&mutation.name);

        assert!(
            mutation.reason.matches(&error),
            "{}: {error:?}",
            mutation.name
        );
    }
}

#[test]
fn event_must_match_completed_action_and_repository() {
    for (pointer, replacement, expected) in [
        ("/action", Value::from("requested"), Rejected::Event),
        ("/repository/id", Value::from(999), Rejected::Repository),
        ("/workflow_run/id", Value::from(999), Rejected::RunIdentity),
        (
            "/workflow_run/head_sha",
            Value::from("4444444444444444444444444444444444444444"),
            Rejected::RunIdentity,
        ),
    ] {
        let mut event = value(LINUX_EVENT);
        *event.pointer_mut(pointer).expect("event pointer") = replacement;

        let actual = admit(&CONTROLLER, Trigger::WorkflowRun(&bytes(&event)), LINUX_RUN)
            .expect_err("event mismatch");

        assert_eq!(
            std::mem::discriminant(&actual),
            std::mem::discriminant(&expected)
        );
    }
}

#[test]
fn both_trigger_modes_refuse_a_foreign_or_feature_controller() {
    for controller in [
        Controller {
            repository: "Contributor/mesh-llm",
            reference: "refs/heads/main",
        },
        Controller {
            repository: "Mesh-LLM/mesh-llm",
            reference: "refs/heads/feature",
        },
    ] {
        for trigger in [
            Trigger::Manual(MANUAL_EVENT),
            Trigger::WorkflowRun(LINUX_EVENT),
        ] {
            let result = admit(&controller, trigger, LINUX_RUN);

            assert!(matches!(result, Err(Rejected::Controller)));
        }
    }
}

#[test]
fn manual_selector_rejects_foreign_rows_runs_and_free_form_inputs() {
    for (pointer, replacement, expected) in [
        (
            "/inputs/product_row",
            Value::from("macos-metal"),
            Rejected::RowProducer,
        ),
        (
            "/inputs/producer_run_id",
            Value::from("202"),
            Rejected::RunIdentity,
        ),
        (
            "/inputs/producer_run_id",
            Value::from("0"),
            Rejected::RunIdentity,
        ),
        (
            "/inputs/producer_run_id",
            Value::from("1e2"),
            Rejected::RunIdentity,
        ),
        (
            "/inputs/producer_run_id",
            Value::from("+101"),
            Rejected::RunIdentity,
        ),
        (
            "/inputs/producer_run_id",
            Value::from("0101"),
            Rejected::RunIdentity,
        ),
    ] {
        let mut event = value(MANUAL_EVENT);
        *event.pointer_mut(pointer).expect("manual pointer") = replacement;

        let error = admit(&CONTROLLER, Trigger::Manual(&bytes(&event)), LINUX_RUN)
            .expect_err("manual input denied");

        assert_eq!(
            std::mem::discriminant(&error),
            std::mem::discriminant(&expected)
        );
    }
}

#[test]
fn manual_cannot_bypass_producer_event_or_workflow_rules() {
    for (field, replacement) in [
        ("event", "workflow_dispatch"),
        ("path", ".github/workflows/ci-linux-lane.yml"),
    ] {
        let mut run = value(LINUX_RUN);
        run[field] = Value::from(replacement);

        let result = admit(&CONTROLLER, Trigger::Manual(MANUAL_EVENT), &bytes(&run));

        assert!(matches!(
            result,
            Err(Rejected::ProducerEvent | Rejected::Workflow)
        ));
    }
}

#[test]
fn malformed_closed_manual_selectors_fail_at_the_boundary() {
    for selector in [
        "windows-cpu",
        "linux-arm64-cpu",
        "linux-rocm",
        "*",
        "https://example.test/product",
    ] {
        let mut event = value(MANUAL_EVENT);
        event["inputs"]["product_row"] = Value::from(selector);

        let result = admit(&CONTROLLER, Trigger::Manual(&bytes(&event)), LINUX_RUN);

        assert!(matches!(result, Err(Rejected::Input(_))));
    }
}

#[test]
fn free_form_artifact_paths_do_not_become_manual_authority() {
    let mut event = value(MANUAL_EVENT);
    event["inputs"]["artifact_path"] = Value::from("/tmp/product");

    let result = admit(&CONTROLLER, Trigger::Manual(&bytes(&event)), LINUX_RUN);

    assert!(matches!(result, Err(Rejected::Input(_))));
}

#[test]
fn duplicate_known_json_fields_cannot_override_a_denial() {
    let run = String::from_utf8(LINUX_RUN.to_vec())
        .expect("UTF-8 fixture")
        .replacen(
            "\"event\": \"push\"",
            "\"event\": \"pull_request\", \"event\": \"push\"",
            1,
        );

    let result = admit(
        &CONTROLLER,
        Trigger::WorkflowRun(LINUX_EVENT),
        run.as_bytes(),
    );

    assert!(matches!(result, Err(Rejected::Input(_))));
}
