//! Selection for `ci-ops build-cache prune`: oldest-first incremental
//! directories, then per-package Cargo artifacts, both bounded by the
//! configured size and age. Only these explicitly selected, repo-owned
//! targets are ever removed (and only with `--execute`).

use crate::ci_operations::build_cache_cargo::{Failure, cargo_packages, clean_package};
use crate::ci_operations::build_cache_tree::{artifact_roots, children, remove_tree, tree_metrics};
use crate::ci_operations::python_access::{object, string};
use crate::ci_plan::document::Json;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

pub(crate) fn int(value: i128) -> Json {
    match (i64::try_from(value), u64::try_from(value)) {
        (Ok(small), _) => Json::Number(small.into()),
        (_, Ok(large)) => Json::Number(large.into()),
        _ => Json::String(value.to_string()),
    }
}

pub(crate) fn float(value: f64) -> Json {
    serde_json::Number::from_f64(value).map_or(Json::Null, Json::Number)
}

pub(crate) struct Budget {
    pub(crate) current: i128,
    pub(crate) max_bytes: i128,
    pub(crate) cutoff: f64,
    pub(crate) execute: bool,
}

impl Budget {
    fn satisfied(&self, newest: f64) -> bool {
        newest >= self.cutoff && self.current <= self.max_bytes
    }

    fn spend(&mut self, bytes: i128) {
        self.current = (self.current - bytes).max(0);
    }
}

fn by_float(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

/// `prune_incremental`.
pub(crate) fn prune_incremental(target: &Path, budget: &mut Budget) -> Result<Vec<Json>, Failure> {
    let over = budget.current > budget.max_bytes;
    let mut candidates: Vec<(f64, PathBuf, i128)> = Vec::new();
    for root in artifact_roots(target, "incremental") {
        for child in children(&root) {
            let (bytes, newest) = tree_metrics(&child);
            if newest < budget.cutoff || over {
                candidates.push((newest, child, bytes));
            }
        }
    }
    candidates.sort_by(|left, right| {
        by_float(left.0, right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let mut actions = Vec::new();
    for (newest, path, bytes) in candidates {
        if budget.satisfied(newest) {
            break;
        }
        actions.push(object(&[
            ("kind", string("incremental")),
            ("path", string(&path.to_string_lossy())),
            ("bytes", int(bytes)),
        ]));
        if budget.execute {
            remove_tree(&path, target)?;
        }
        budget.spend(bytes);
    }
    Ok(actions)
}

struct PackageMetrics {
    package: String,
    bytes: i128,
    newest: f64,
}

fn matches_stem(name: &str, stem: &str) -> bool {
    name == stem
        || name.starts_with(&format!("{stem}-"))
        || name.starts_with(&format!("lib{stem}-"))
}

/// `package_metrics`: deps match underscore stems, build dirs the raw names.
fn package_metrics(target: &Path, packages: &[String]) -> Vec<PackageMetrics> {
    let mut totals: Vec<(i128, f64)> = vec![(0, 0.0); packages.len()];
    let mut roots = artifact_roots(target, "deps");
    roots.extend(artifact_roots(target, "build"));
    for root in roots {
        let deps = root.file_name().is_some_and(|name| name == "deps");
        for child in children(&root) {
            let name = child
                .file_name()
                .map(|name| name.to_string_lossy().into_owned());
            let name = name.unwrap_or_default();
            let found = packages.iter().position(|package| {
                let stem = if deps {
                    package.replace('-', "_")
                } else {
                    package.clone()
                };
                matches_stem(&name, &stem)
            });
            if let Some(index) = found {
                let (bytes, newest) = tree_metrics(&child);
                totals[index].0 += bytes;
                totals[index].1 = totals[index].1.max(newest);
            }
        }
    }
    let mut metrics: Vec<PackageMetrics> = packages
        .iter()
        .zip(totals)
        .filter(|(_, (bytes, _))| *bytes != 0)
        .map(|(package, (bytes, newest))| PackageMetrics {
            package: package.clone(),
            bytes,
            newest,
        })
        .collect();
    metrics.sort_by(|left, right| {
        by_float(left.newest, right.newest).then_with(|| right.bytes.cmp(&left.bytes))
    });
    metrics
}

/// `prune_packages`: sizes are estimated rather than re-walked, as the
/// legacy loop only needs them to decide when to stop.
pub(crate) fn prune_packages(
    workspace: &Path,
    target: &Path,
    budget: &mut Budget,
) -> Result<Vec<Json>, Failure> {
    let mut actions = Vec::new();
    for metrics in package_metrics(target, &cargo_packages(workspace)?) {
        if budget.satisfied(metrics.newest) {
            continue;
        }
        actions.push(object(&[
            ("kind", string("cargo-package")),
            ("package", string(&metrics.package)),
            ("estimated_bytes", int(metrics.bytes)),
        ]));
        if budget.execute {
            clean_package(workspace, target, &metrics.package)?;
        }
        budget.spend(metrics.bytes);
        if budget.satisfied(metrics.newest) {
            break;
        }
    }
    Ok(actions)
}
