use super::*;
use crate::migration_inventory::required_graph::{report, require_census};
use crate::migration_inventory::required_graph_tests::source;
use std::collections::BTreeSet;
fn forward(child: &str) -> String {
    format!("{PREFIX}{child}\" \"$@\"\n")
}
#[test]
fn product_forwarder_admits_only_explicit_repository_product_root_and_exact_arguments() {
    for child in [
        "mesh/scripts/check-sdk-contract.sh",
        "skippy/scripts/owner.sh",
    ] {
        assert_eq!(
            target("scripts/entry.sh", forward(child).trim()).unwrap(),
            Some(child.into())
        );
    }
    for text in [
        forward("foreign/scripts/owner.sh"),
        forward("mesh/scripts/../owner.sh"),
        forward("mesh/scripts/$OWNER.sh"),
        forward("mesh/scripts/owner.sh").replace("/..\"", "/../..\""),
        forward("mesh/scripts/owner.sh").replace("\"$@\"", "$*"),
    ] {
        assert!(target("scripts/entry.sh", text.trim()).is_err());
    }
    assert!(
        target(
            "foreign/scripts/entry.sh",
            forward("mesh/scripts/owner.sh").trim()
        )
        .unwrap()
        .is_none()
    );
}
#[test]
fn product_forwarder_traverses_owner_and_cycle_without_hiding_uncontracted_python() -> DynResult<()>
{
    let root = crate::command::unique_temp_dir("product-forwarder-graph");
    source(&root, "scripts/entry.sh", &forward("mesh/scripts/owner.sh"))?;
    source(
        &root,
        "mesh/scripts/owner.sh",
        "bash scripts/entry.sh\npython3 scripts/child.py\n",
    )?;
    source(&root, "scripts/child.py", "pass\n")?;
    let paths = [
        "scripts/entry.sh",
        "mesh/scripts/owner.sh",
        "scripts/child.py",
    ]
    .map(str::to_owned);
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &["scripts/entry.sh"])?;
    assert_eq!(
        graph
            .edges
            .iter()
            .filter(|edge| edge.parent == "scripts/entry.sh"
                && edge.child.as_deref() == Some("mesh/scripts/owner.sh"))
            .count(),
        1
    );
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "mesh/scripts/owner.sh"
                && edge.child.as_deref() == Some("scripts/child.py")
                && edge.unresolved_reason.is_some())
    );
    assert!(require_census(&graph).is_err());
    std::fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn product_forwarder_refuses_missing_owned_source() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("product-forwarder-missing");
    source(
        &root,
        "scripts/entry.sh",
        &forward("mesh/scripts/missing.sh"),
    )?;
    let error = report(
        &root,
        &["scripts/entry.sh".into()],
        &[],
        &BTreeSet::new(),
        &["scripts/entry.sh"],
    )
    .unwrap_err();
    assert!(error.to_string().contains("missing product owner"));
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn bash_product_forwarder_preserves_closed_owner_and_argument_custody() {
    let valid = forward("skippy/scripts/owner.sh").replacen("exec ", "exec bash ", 1);
    assert_eq!(
        target("scripts/entry.sh", valid.trim()).unwrap(),
        Some("skippy/scripts/owner.sh".into())
    );
    for changed in [
        valid.replace("/..\"", "/../..\""),
        valid.replace("\"$@\"", "$*"),
        valid.replace("skippy/scripts/owner.sh", "skippy/scripts/../owner.sh"),
        valid.replace("skippy/scripts/owner.sh", "foreign/scripts/owner.sh"),
        valid.replace("skippy/scripts/owner.sh", "skippy/scripts/$OWNER.sh"),
    ] {
        assert!(
            target("scripts/entry.sh", changed.trim()).is_err(),
            "{changed}"
        );
    }
    for changed in [
        valid.replace("exec bash ", "exec bash -c "),
        valid.replace("exec bash ", "exec env bash "),
    ] {
        assert!(
            target("scripts/entry.sh", changed.trim())
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn bash_product_forwarder_traverses_cycle_and_refuses_missing_owner() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("bash-product-forwarder");
    let entry = forward("skippy/scripts/owner.sh").replacen("exec ", "exec bash ", 1);
    source(&root, "scripts/entry.sh", &entry)?;
    let missing = report(
        &root,
        &["scripts/entry.sh".into()],
        &[],
        &BTreeSet::new(),
        &["scripts/entry.sh"],
    )
    .unwrap_err();
    assert!(missing.to_string().contains("missing product owner"));
    source(
        &root,
        "skippy/scripts/owner.sh",
        "bash scripts/entry.sh\npython3 scripts/child.py\n",
    )?;
    source(&root, "scripts/child.py", "pass\n")?;
    let paths = [
        "scripts/entry.sh",
        "skippy/scripts/owner.sh",
        "scripts/child.py",
    ]
    .map(str::to_owned);
    let graph = report(&root, &paths, &[], &BTreeSet::new(), &["scripts/entry.sh"])?;
    let forwarded: Vec<_> = graph
        .edges
        .iter()
        .filter(|edge| {
            edge.parent == "scripts/entry.sh"
                && edge.child.as_deref() == Some("skippy/scripts/owner.sh")
        })
        .collect();
    assert_eq!(forwarded.len(), 1);
    assert_eq!(
        forwarded[0].argv.as_deref(),
        Some("bash skippy/scripts/owner.sh $@")
    );
    assert!(
        graph
            .edges
            .iter()
            .any(|edge| edge.parent == "skippy/scripts/owner.sh"
                && edge.child.as_deref() == Some("scripts/child.py")
                && edge.unresolved_reason.is_some())
    );
    assert!(require_census(&graph).is_err());
    std::fs::remove_dir_all(root)?;
    Ok(())
}
