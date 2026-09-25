//! `build_plan`: validate input and catalogs, derive every plan section and
//! re-check the semantic invariants the schema cannot express.

use crate::ci_plan::catalog::{self, Ownership};
use crate::ci_plan::diagnostics::{PlanResult, fail};
use crate::ci_plan::document::Json;
use crate::ci_plan::matrices::{self, RowRequest};
use crate::ci_plan::profile_catalog::BUDGET_KEYS;
use crate::ci_plan::request::{self, Profile};
use crate::ci_plan::selection;
use crate::ci_plan::signals;
use crate::ci_plan::slice_catalog::{self, SliceCatalog, SliceDefinition};
use crate::ci_plan::workspace::{self, Package, Scope};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;
use std::path::Path;

/// The protected checkout and the (possibly source-extracted) catalog root.
pub(super) struct Roots<'a> {
    pub(super) workspace: &'a Path,
    pub(super) manifests: &'a Path,
}

fn load_catalogs(manifests: &Path) -> PlanResult<(Ownership, SliceCatalog)> {
    let ownership = catalog::load(&manifests.join("ci").join("ownership.yml"))?;
    let slices = catalog::load(&manifests.join("ci").join("slices.yml"))?;
    let ownership = catalog::validate_ownership(&ownership)?;
    let slices = slice_catalog::validate(&slices, &ownership)?;
    Ok((ownership, slices))
}

/// A `pr-isolated` slice publishes to trusted scope only on main.
fn planned_cache_mode(slice: &SliceDefinition, profile: Profile) -> &str {
    match (slice.cache_mode.as_str(), profile) {
        ("pr-isolated", Profile::Main) => "trusted-readwrite",
        ("pr-isolated", Profile::ManualFull) => "trusted-readonly",
        (mode, _) => mode,
    }
}

pub(super) fn build(payload: &Json, roots: &Roots<'_>) -> PlanResult<Value> {
    let input = request::parse(payload)?;
    let (ownership, catalog) = load_catalogs(roots.manifests)?;
    let profile = input.profile;
    let changed_files = request::normalise_changed_files(&input.changed_files)?;
    let packages = workspace::packages(roots.workspace, input.workspace_packages.as_ref())?;
    let direct_crates = workspace::direct_crates(&changed_files, &packages);
    let scope = Scope {
        root: roots.workspace,
        changed_files: &changed_files,
        packages: &packages,
        profile,
    };
    let affected_crates = workspace::affected_crates(&scope, input.affected_crates.as_ref())?;
    let domains = selection::matched_domains(&ownership, &changed_files, &direct_crates)?;
    let selected = selection::select_slices(&catalog, profile, &domains)?;
    let required = &selected.required;
    let mut matrices = matrices::select_rows(
        &catalog.rows,
        &RowRequest {
            profile,
            domains: &domains,
            required,
            force_all_rows: selected.force_all_rows,
        },
    )?;
    let all_rust = selected.force_all_rows || profile.exhaustive();
    let rust_scope = if affected_crates.is_empty() && all_rust {
        packages
            .iter()
            .map(|package| package.name.clone())
            .collect()
    } else {
        affected_crates.clone()
    };
    let has_slice = |id: &str| required.iter().any(|slice| slice == id);
    let clippy = has_slice("quality") && (domains.iter().any(|d| d == "rust") || all_rust);
    let batches = |enabled: bool, bins: usize| match enabled {
        true => Value::Array(matrices::make_batches(&rust_scope, bins)),
        false => Value::Array(Vec::new()),
    };
    matrices.insert("clippy".to_owned(), batches(clippy, catalog.clippy_batches));
    matrices.insert(
        "rust_tests".to_owned(),
        batches(has_slice("rust-tests"), catalog.rust_test_batches),
    );
    let per_slice = |derive: &dyn Fn(&SliceDefinition) -> Value| {
        required
            .iter()
            .filter_map(|id| catalog.slice(id))
            .map(|slice| (slice.id.clone(), derive(slice)))
            .collect::<Map<_, _>>()
    };
    let budgets = catalog
        .profiles
        .get(&profile)
        .map(|definition| definition.budgets)
        .unwrap_or_default();
    let plan = json!({
        "schema_version": 1,
        "profile": profile.name(),
        "event_name": input.event_name,
        "source_sha": input.source_sha,
        "base_sha": input.base_sha,
        "direct_crates": direct_crates,
        "affected_crates": affected_crates,
        "domains": domains,
        "required_slices": required,
        "matrices": matrices,
        "reasons": selected.reasons,
        "dependencies": per_slice(&|slice| json!(slice.depends_on)),
        "runner_roles": per_slice(&|slice| json!(slice.runner_role)),
        "cache_modes": per_slice(&|slice| json!(planned_cache_mode(slice, profile))),
        "budgets": BUDGET_KEYS.iter().zip(budgets).map(|(key, value)| ((*key).to_owned(), json!(value))).collect::<Map<_, _>>(),
        "signals": signals::signals(&changed_files, &domains, profile),
    });
    validate(&plan, profile, &packages)?;
    Ok(plan)
}

/// The `_validate_plan` invariants a correct derivation cannot violate but a
/// catalog edit could: unique matrix row IDs and exhaustive test coverage.
fn validate(plan: &Value, profile: Profile, packages: &[Package]) -> PlanResult<()> {
    let empty = Map::new();
    for (name, rows) in plan["matrices"].as_object().unwrap_or(&empty) {
        let ids = rows
            .as_array()
            .into_iter()
            .flatten()
            .map(|row| row["id"].as_str().filter(|id| !id.is_empty()))
            .collect::<Option<Vec<_>>>();
        let Some(ids) = ids else {
            return fail(format!("plan matrix {name} has an invalid row"));
        };
        if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
            return fail(format!("plan matrix {name} contains duplicate rows"));
        }
    }
    let rust_tests_required = plan["required_slices"]
        .as_array()
        .is_some_and(|slices| slices.iter().any(|slice| slice == "rust-tests"));
    if rust_tests_required && profile.exhaustive() {
        let tested = plan["matrices"]["rust_tests"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|batch| batch["crates"].as_array().into_iter().flatten())
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>();
        let workspace = packages
            .iter()
            .map(|package| package.name.as_str())
            .collect();
        if tested != workspace {
            return fail("main rust test matrix does not cover every workspace crate");
        }
    }
    Ok(())
}
