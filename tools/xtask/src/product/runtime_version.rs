//! `product runtime-version <manifest.json>`: the port of the snippet in
//! `scripts/ci-compose-product-input.sh` that prints
//! `json.load(handle)["runtime"]["mesh_version"]` when no version input is
//! given. A failure prints the uncaught exception's last line and exits 1.

use super::manifest_load::load;
use super::python_object::{display, item};
use crate::repository::check_report::CheckReport;
use std::path::Path;

pub(super) fn run(args: &[String]) -> CheckReport {
    match mesh_version(args) {
        Ok(version) => CheckReport::success(format!("{version}\n")),
        Err(line) => CheckReport::failure(String::new(), format!("{line}\n")),
    }
}

fn mesh_version(args: &[String]) -> Result<String, String> {
    let path = args
        .first()
        .ok_or_else(|| "IndexError: list index out of range".to_owned())?;
    let manifest = load(Path::new(path), path)?;
    let version = item(item(&manifest, "runtime")?, "mesh_version")?;
    Ok(display(version))
}
