//! `dependency_order` of `scripts/linux-native-runtime-deps.py`: the
//! packaged libraries in load order (dependencies first) with the primary
//! runtime library last.

use super::linux_deps_elf::{ElfImage, Raised, join};
use super::linux_deps_policy::Package;
use crate::ci_plan::catalog::os_error_text;
use std::collections::{BTreeMap, HashSet};

/// The depth-first walk over packaged libraries.
struct Walk<'a> {
    libraries: BTreeMap<&'a str, &'a ElfImage>,
    visiting: HashSet<&'a str>,
    visited: HashSet<&'a str>,
    ordered: Vec<String>,
}

impl<'a> Walk<'a> {
    fn visit(&mut self, name: &'a str) -> Result<(), Raised> {
        if self.visited.contains(name) {
            return Ok(());
        }
        if self.visiting.contains(name) {
            return Err(format!("cyclic Linux runtime dependency graph at {name}"));
        }
        let Some(image) = self.libraries.get(name).copied() else {
            return Ok(());
        };
        self.visiting.insert(name);
        let mut needed: Vec<&'a str> = image.needed.iter().map(String::as_str).collect();
        needed.sort_unstable();
        for dependency in needed {
            if self.libraries.contains_key(dependency) {
                self.visit(dependency)?;
            }
        }
        self.visiting.remove(name);
        self.visited.insert(name);
        self.ordered.push(image.path.clone());
        Ok(())
    }
}

/// Regular, non-symlink entries of the library directory, by name.
fn library_names(lib_dir: &str) -> Result<Vec<String>, Raised> {
    let entries = std::fs::read_dir(lib_dir).map_err(|error| os_error_text(&error, lib_dir))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| os_error_text(&error, lib_dir))?;
        if entry.file_type().is_ok_and(|kind| kind.is_file()) {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    Ok(names)
}

/// `dependency_order`: the ordered library paths.
pub(super) fn dependency_order(
    package: &Package<'_>,
    primary: &str,
) -> Result<Vec<String>, Raised> {
    package.verify()?;
    let images = package.images()?;
    let lib_dir = package.lib_dir.as_str();
    let libraries: BTreeMap<&str, &ElfImage> = images
        .iter()
        .filter(|image| package.in_lib_dir(image))
        .map(|image| (image.name(), image))
        .collect();
    let names = library_names(lib_dir)?;
    if !names.iter().any(|name| name == primary) {
        return Err(format!(
            "primary Linux runtime library is missing: {primary}"
        ));
    }
    if libraries.is_empty() {
        let mut ordered: Vec<String> = names
            .iter()
            .filter(|name| *name != primary)
            .map(|name| join(lib_dir, name))
            .collect();
        ordered.push(join(lib_dir, primary));
        return Ok(ordered);
    }
    let mut walk = Walk {
        libraries,
        visiting: HashSet::new(),
        visited: HashSet::new(),
        ordered: Vec::new(),
    };
    let others: Vec<&str> = walk
        .libraries
        .keys()
        .copied()
        .filter(|name| *name != primary)
        .collect();
    for name in others {
        walk.visit(name)?;
    }
    let primary_path = match walk.libraries.get(primary).map(|image| image.path.clone()) {
        Some(path) => {
            walk.visit(primary)?;
            if let Some(index) = walk.ordered.iter().position(|known| *known == path) {
                walk.ordered.remove(index);
            }
            path
        }
        None => join(lib_dir, primary),
    };
    let mut ordered = walk.ordered;
    let ordered_names: HashSet<String> = ordered
        .iter()
        .map(|path| super::linux_deps_elf::file_name(path).to_owned())
        .collect();
    ordered.extend(
        names
            .iter()
            .filter(|name| !ordered_names.contains(*name) && *name != primary)
            .map(|name| join(lib_dir, name)),
    );
    ordered.push(primary_path);
    if ordered.len() != names.len() {
        return Err("could not order all Linux runtime libraries".to_owned());
    }
    Ok(ordered)
}
