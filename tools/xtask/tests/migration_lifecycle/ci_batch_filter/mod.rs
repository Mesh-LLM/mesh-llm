mod fixture;
use crate::workflow_yaml;
use fixture::{Fixture, WORKFLOWS};
use workflow_yaml::Node;
#[test]
fn actual_batch_filter_drops_only_absent_packages_before_cargo_execution() {
    for workflow in WORKFLOWS {
        let fixture = Fixture::new(
            workflow,
            &["absent-owner", "present-owner"],
            &["present-owner"],
            false,
            false,
        );
        let result = fixture.run();
        let stdout = String::from_utf8_lossy(result.stdout.as_ref().unwrap().as_bytes());
        assert!(stdout.contains("::warning::") && stdout.contains("absent-owner"));
        assert_eq!(fixture.packages(), ["present-owner"]);
        assert_eq!(fixture.calls()[0][0], "metadata");
    }
}
#[test]
fn actual_batch_filter_empty_projection_skips_all_cargo_builds() {
    for workflow in WORKFLOWS {
        let fixture = Fixture::new(
            workflow,
            &["absent-owner"],
            &["present-owner"],
            false,
            false,
        );
        fixture.run();
        assert!(fixture.packages().is_empty());
        assert_eq!(fixture.calls().len(), 1);
    }
}
#[test]
fn actual_batch_filter_metadata_failure_preserves_requested_execution() {
    for workflow in WORKFLOWS {
        let fixture = Fixture::new(workflow, &["planned-owner"], &[], true, false);
        let result = fixture.run();
        assert!(
            String::from_utf8_lossy(result.stdout.as_ref().unwrap().as_bytes())
                .contains("cargo metadata failed")
        );
        assert_eq!(fixture.packages(), ["planned-owner"]);
    }
}
#[test]
fn actual_batch_filter_present_packages_execute_without_warning() {
    for workflow in WORKFLOWS {
        let fixture = Fixture::new(
            workflow,
            &["first-owner", "second-owner"],
            &["first-owner", "second-owner"],
            false,
            false,
        );
        let result = fixture.run();
        assert!(
            !String::from_utf8_lossy(result.stdout.as_ref().unwrap().as_bytes())
                .contains("::warning::")
        );
        assert_eq!(fixture.packages(), ["first-owner", "second-owner"]);
    }
}
#[test]
fn parsed_workflow_orders_translation_filter_and_consumers_and_executes_typed_translation() {
    for workflow in WORKFLOWS {
        let steps = fixture::steps(workflow);
        let translate = steps
            .iter()
            .position(|step| step.get("id").and_then(Node::text) == Some("packages"))
            .unwrap();
        let filter = steps
            .iter()
            .position(|step| step.get("id").and_then(Node::text) == Some("resolve_batch_crates"))
            .unwrap();
        let batch = steps
            .iter()
            .position(|step| {
                step.get("name").and_then(Node::text) == Some(fixture::batch_name(workflow))
            })
            .unwrap();
        assert!(translate < filter && filter < batch);
        assert!(
            steps[translate]
                .get("uses")
                .and_then(Node::text)
                .unwrap()
                .contains("resolve-cargo-packages@")
        );
        assert_eq!(
            steps[filter]
                .get("env")
                .unwrap()
                .get("PLANNED_BATCH_CRATES")
                .and_then(Node::text),
            Some("${{ steps.packages.outputs.crates }}")
        );
        let env = steps[batch].get("env").unwrap();
        let variable = if workflow == WORKFLOWS[0] {
            "CLIPPY_CRATES"
        } else {
            "TEST_CRATES"
        };
        assert_eq!(
            env.get(variable).and_then(Node::text),
            Some("${{ steps.resolve_batch_crates.outputs.crates }}")
        );
        let fixture = Fixture::new(
            workflow,
            &["mesh-llm-gpu-bench"],
            &["skippy-gpu-bench", "skippy-package-builder"],
            false,
            true,
        );
        fixture.run();
        assert_eq!(fixture.packages(), ["skippy-gpu-bench"]);
        let calls = fixture.calls();
        assert_eq!(calls[0][0], "metadata");
        assert_eq!(calls[1][0], "metadata");
        assert!(calls.iter().skip(2).all(|call| call[0] != "metadata"));
    }
}
