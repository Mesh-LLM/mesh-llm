//! Bounded job matrices: host/runtime/platform/SDK/smoke rows selected from
//! the catalogs, and deterministic weighted shards for Clippy and tests.

use crate::ci_plan::diagnostics::{PlanResult, fail, repr};
use crate::ci_plan::request::Profile;
use crate::ci_plan::row_catalog::{RowCatalog, RowCatalogs};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;

/// Measured relative cost of the heavy crates; every other crate weighs 1.
const CRATE_WEIGHTS: &[(&str, u32)] = &[
    ("mesh-llm", 10),
    ("mesh-llm-host-runtime", 10),
    ("mesh-llm-embedded-runtime", 8),
    ("mesh-llm-client", 6),
    ("skippy-runtime", 5),
    ("skippy-server", 5),
    ("model-artifact", 4),
    ("model-hf", 4),
    ("openai-frontend", 4),
    ("skippy-correctness", 4),
    ("mesh-llm-api-server", 3),
    ("mesh-llm-system", 3),
    ("skippy-prompt", 3),
];

/// What row selection depends on.
pub(super) struct RowRequest<'a> {
    pub(super) profile: Profile,
    pub(super) domains: &'a [String],
    pub(super) required: &'a [String],
    pub(super) force_all_rows: bool,
}

/// `_select_rows`, returning the five row matrices as a JSON object.
pub(super) fn select_rows(
    catalogs: &RowCatalogs,
    request: &RowRequest<'_>,
) -> PlanResult<Map<String, Value>> {
    let mut matrices = Map::new();
    if request.profile == Profile::PrDraft && !request.force_all_rows {
        for name in [
            "hosts",
            "runtime_products",
            "platform_checks",
            "sdk",
            "smoke",
        ] {
            matrices.insert(name.to_owned(), Value::Array(Vec::new()));
        }
        return Ok(matrices);
    }
    let every = |catalog: &RowCatalog| catalog.ids().map(str::to_owned).collect::<Vec<_>>();
    let is_required = |slice: &str| request.required.iter().any(|id| id == slice);
    let (runtime_ids, platform_ids, sdk_ids, smoke_ids) = if request.force_all_rows {
        (
            every(&catalogs.runtime),
            every(&catalogs.platform),
            every(&catalogs.sdk),
            every(&catalogs.smoke),
        )
    } else {
        let mut runtime = catalogs.runtime.mapped(request.domains);
        if is_required("runtime-product") && runtime.is_empty() {
            runtime.push("linux-cpu".to_owned());
        }
        let mut smoke = catalogs.smoke.mapped(request.domains);
        if is_required("product-smoke") && smoke.is_empty() {
            smoke.push("core".to_owned());
        }
        (
            runtime,
            catalogs.platform.mapped(request.domains),
            catalogs.sdk.mapped(request.domains),
            smoke,
        )
    };
    let runtime = unique_rows(&catalogs.runtime, &runtime_ids)?;
    let macos = runtime
        .iter()
        .filter(|row| row["platform"] == "macos")
        .map(|row| row["architecture"].to_string())
        .collect::<BTreeSet<_>>();
    if macos.len() > 1 {
        return fail(
            "macOS runtime_products must use one architecture until SDK and smoke consumers are row-scoped",
        );
    }
    matrices.insert(
        "hosts".to_owned(),
        Value::Array(hosts(&runtime, request.profile)),
    );
    matrices.insert("runtime_products".to_owned(), Value::Array(runtime));
    matrices.insert(
        "platform_checks".to_owned(),
        Value::Array(unique_rows(&catalogs.platform, &platform_ids)?),
    );
    matrices.insert(
        "sdk".to_owned(),
        Value::Array(unique_rows(&catalogs.sdk, &sdk_ids)?),
    );
    matrices.insert(
        "smoke".to_owned(),
        Value::Array(unique_rows(&catalogs.smoke, &smoke_ids)?),
    );
    Ok(matrices)
}

fn unique_rows(catalog: &RowCatalog, ids: &[String]) -> PlanResult<Vec<Value>> {
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for id in ids {
        if !seen.insert(id.as_str()) {
            continue;
        }
        let Some(row) = catalog.row(id) else {
            return fail(format!(
                "{} references unknown row {}",
                catalog.field,
                repr(id)
            ));
        };
        rows.push(row.to_value());
    }
    Ok(rows)
}

/// One backend-neutral host per runtime platform/architecture.
fn hosts(runtime: &[Value], profile: Profile) -> Vec<Value> {
    let mut seen = BTreeSet::new();
    let mut hosts = Vec::new();
    for row in runtime {
        let (platform, architecture) = (&row["platform"], &row["architecture"]);
        if !seen.insert((platform.to_string(), architecture.to_string())) {
            continue;
        }
        let text = |value: &Value| value.as_str().unwrap_or_default().to_owned();
        let host_profile = row
            .get("profile")
            .cloned()
            .unwrap_or_else(|| Value::String(profile.name().to_owned()));
        hosts.push(json!({
            "id": format!("{}-{}-host", text(platform), text(architecture)),
            "platform": platform,
            "architecture": architecture,
            "runner_role": row["runner_role"],
            "profile": host_profile,
        }));
    }
    hosts
}

/// `_make_batches`: heaviest crates first (ties keep workspace order), each
/// into the currently lightest bucket (ties to the lowest index); empty
/// buckets are dropped.
pub(super) fn make_batches(crates: &[String], bins: usize) -> Vec<Value> {
    let mut unique: Vec<&str> = Vec::new();
    for name in crates {
        if !unique.contains(&name.as_str()) {
            unique.push(name);
        }
    }
    let weight = |name: &str| {
        CRATE_WEIGHTS
            .iter()
            .find(|(heavy, _)| *heavy == name)
            .map_or(1, |(_, weight)| *weight)
    };
    let mut ordered = unique
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, weight(name), index))
        .collect::<Vec<_>>();
    ordered.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(b.0)));
    let mut buckets = (0..bins).map(|_| (0_u32, Vec::new())).collect::<Vec<_>>();
    for (name, crate_weight, _) in ordered {
        let lightest = buckets
            .iter()
            .enumerate()
            .min_by_key(|(index, (total, _))| (*total, *index))
            .map(|(index, _)| index);
        if let Some(bucket) = lightest.and_then(|index| buckets.get_mut(index)) {
            bucket.0 += crate_weight;
            bucket.1.push(name.to_owned());
        }
    }
    buckets
        .into_iter()
        .enumerate()
        .filter(|(_, (_, names))| !names.is_empty())
        .map(|(index, (total, names))| {
            json!({"idx": index, "weight": total, "crates": names, "id": format!("batch-{index}")})
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::make_batches;

    #[test]
    fn migration_ci_plan_batches_balance_weights_deterministically() {
        let crates = ["a", "mesh-llm", "b", "mesh-llm-host-runtime", "a"].map(str::to_owned);
        let batches = make_batches(&crates, 3);
        let summary = batches
            .iter()
            .map(|batch| {
                (
                    batch["id"].clone(),
                    batch["crates"].clone(),
                    batch["weight"].clone(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                ("batch-0".into(), serde_json::json!(["mesh-llm"]), 10.into()),
                (
                    "batch-1".into(),
                    serde_json::json!(["mesh-llm-host-runtime"]),
                    10.into()
                ),
                ("batch-2".into(), serde_json::json!(["a", "b"]), 2.into()),
            ]
        );
        assert!(make_batches(&[], 3).is_empty());
    }
}
