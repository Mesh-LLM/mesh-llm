use super::*;
use crate::automation::sdk_advisory::rows::{ProducerWorkflow, ProductRow};

#[test]
fn catalog_changes_cannot_expand_an_eligible_rows_identity() {
    for (field, replacement) in [
        ("platform", "windows"),
        ("architecture", "arm64"),
        ("backend", "rocm"),
        ("target", "aarch64-unknown-linux-gnu"),
    ] {
        let mut slices = value(SLICES);
        slices["runtime_rows"][0][field] = Value::from(replacement);
        let catalog = Catalog::parse(OWNERSHIP, &bytes(&slices)).expect("valid planner row");

        let result = catalog.row(ProductRow::LinuxCpu);

        assert!(matches!(result, Err(Rejected::RowIdentity)), "{field}");
    }
}

#[test]
fn existing_catalog_validation_still_rejects_duplicate_rows() {
    let mut slices = value(SLICES);
    let duplicate = slices["runtime_rows"][0].clone();
    slices["runtime_rows"]
        .as_array_mut()
        .expect("rows")
        .push(duplicate);

    let result = Catalog::parse(OWNERSHIP, &bytes(&slices));

    assert!(matches!(result, Err(Rejected::Catalog(_))));
}

#[test]
fn missing_selected_catalog_row_cannot_be_synthesized() {
    let mut slices = value(SLICES);
    slices["runtime_rows"]
        .as_array_mut()
        .expect("rows")
        .retain(|row| row["id"] != "linux-cuda");
    slices["domain_rows"]["backend-cuda"] = serde_json::json!(["windows-cuda"]);
    let catalog = Catalog::parse(OWNERSHIP, &bytes(&slices)).expect("remaining catalog");

    let result = catalog.row(ProductRow::LinuxCuda);

    assert!(matches!(result, Err(Rejected::MissingRow)));
}

#[test]
fn allowed_trigger_names_come_from_workflow_names_not_required_checks() {
    for (workflow, expected) in [
        (
            include_str!("../../../../.github/workflows/main_linux.yml"),
            ProducerWorkflow::Linux,
        ),
        (
            include_str!("../../../../.github/workflows/main_macos.yml"),
            ProducerWorkflow::Macos,
        ),
    ] {
        let name = workflow
            .lines()
            .find_map(|line| line.strip_prefix("name: "))
            .expect("top-level workflow name");
        let path = match expected {
            ProducerWorkflow::Linux => ".github/workflows/main_linux.yml",
            ProducerWorkflow::Macos => ".github/workflows/main_macos.yml",
        };

        let actual = ProducerWorkflow::parse(name, path).expect("actual workflow trigger literal");

        assert_eq!(actual, expected);
    }
}
