use super::*;
use std::{collections::BTreeMap, fs};

fn fixture(layout: &str) -> (tempfile::TempDir, Vec<super::super::CargoPackage>) {
    let temp = tempfile::tempdir().unwrap();
    let mut packages = Vec::new();
    for (name, directory, leaf) in [
        ("mesh-llm-client", "mesh-client", "src/models/catalog.json"),
        ("mesh-llm-node", "mesh-llm-node", "src/catalog.json"),
    ] {
        let directory = temp.path().join(layout).join(directory);
        fs::create_dir_all(directory.join(leaf).parent().unwrap()).unwrap();
        fs::write(directory.join("Cargo.toml"), "[package]\n").unwrap();
        fs::write(directory.join(leaf), "{\"models\":[]}\n").unwrap();
        packages.push(
            serde_json::from_value(serde_json::json!({
                "id": name, "name": name, "version":"0.76.1",
                "manifest_path": directory.join("Cargo.toml"), "dependencies":[]
            }))
            .unwrap(),
        );
    }
    (temp, packages)
}
fn selected(
    packages: &[super::super::CargoPackage],
) -> BTreeMap<String, &super::super::CargoPackage> {
    packages
        .iter()
        .map(|package| (package.name.clone(), package))
        .collect()
}

#[test]
fn admitted_current_and_historical_packages_bind_their_own_catalogs() {
    for layout in ["mesh/crates", "crates"] {
        let (temp, packages) = fixture(layout);
        check_selected_catalogs(&temp.path().canonicalize().unwrap(), &selected(&packages))
            .unwrap();
    }
}
#[test]
fn no_missing_package_or_file_can_fall_back_to_another_layout() {
    let (temp, packages) = fixture("mesh/crates");
    let root = temp.path().canonicalize().unwrap();
    let mut map = selected(&packages);
    map.remove("mesh-llm-client");
    assert!(check_selected_catalogs(&root, &map).is_err());
    fs::remove_file(
        packages[0]
            .manifest_path
            .parent()
            .unwrap()
            .join("src/models/catalog.json"),
    )
    .unwrap();
    let decoy = root.join("crates/mesh-client/src/models/catalog.json");
    fs::create_dir_all(decoy.parent().unwrap()).unwrap();
    fs::write(decoy, "{\"models\":[]}\n").unwrap();
    assert!(check_selected_catalogs(&root, &selected(&packages)).is_err());
}
#[test]
fn mismatched_and_invalid_catalog_bytes_refuse() {
    let (temp, packages) = fixture("crates");
    let node = packages[1]
        .manifest_path
        .parent()
        .unwrap()
        .join("src/catalog.json");
    for bytes in [b"different".as_slice(), b"\xff".as_slice()] {
        fs::write(&node, bytes).unwrap();
        assert!(
            check_selected_catalogs(&temp.path().canonicalize().unwrap(), &selected(&packages))
                .is_err()
        );
    }
}
#[test]
fn unbound_package_manifest_outside_selected_root_refuses() {
    let (inside, mut packages) = fixture("mesh/crates");
    let (outside, foreign) = fixture("crates");
    packages[0].manifest_path = foreign[0].manifest_path.clone();
    assert!(
        check_selected_catalogs(&inside.path().canonicalize().unwrap(), &selected(&packages))
            .is_err()
    );
    drop(outside);
}
#[cfg(unix)]
#[test]
fn symlink_catalog_escape_from_owning_package_refuses_even_inside_repository() {
    let (temp, packages) = fixture("mesh/crates");
    let client = packages[0]
        .manifest_path
        .parent()
        .unwrap()
        .join("src/models/catalog.json");
    let decoy = temp.path().join("decoy.json");
    fs::write(&decoy, "{\"models\":[]}\n").unwrap();
    fs::remove_file(&client).unwrap();
    std::os::unix::fs::symlink(&decoy, &client).unwrap();
    assert!(
        check_selected_catalogs(&temp.path().canonicalize().unwrap(), &selected(&packages))
            .is_err()
    );
}
