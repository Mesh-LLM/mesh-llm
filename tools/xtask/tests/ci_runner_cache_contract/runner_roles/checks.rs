//! Matrix decisions checked against the placement derived from workflows.
use super::{
    derive::{Inventory, Job, Selector, provider_of},
    matrix::{
        CacheMode, Decision, HostedFallback, Matrix, Profile, Provider, ReadinessState, Role,
    },
};
use std::collections::{BTreeMap, BTreeSet};

const MAIN_GATE: &str = "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}";
const PR_GATE: &str = "${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}";
const SENTINEL_REF: &str = "${{ vars.DEPOT_PR_SENTINEL_REF }}";
const RELEASE_FORCE_HOSTED: &str = "${{ inputs.force_hosted_runners == true }}";
const RELEASE_MANUAL_GATE: &str = "${{ inputs.use_depot == true }}";
const EVIDENCE_PREFIX: &str = "https://github.com/Mesh-LLM/mesh-llm/";

pub(super) fn violations(matrix: &Matrix, inventory: &Inventory) -> Vec<String> {
    let mut errors = Vec::new();
    if matrix.schema_version != 1 {
        errors.push(format!("unsupported schema {}", matrix.schema_version));
    }
    let decisions = selector_decisions(inventory, &mut errors);
    errors.extend(coverage(matrix, inventory));
    for role in &matrix.roles {
        errors.extend(row_shape(role, inventory));
        errors.extend(row_decisions(role, inventory, &decisions));
        errors.extend(row_placement(role, inventory, &decisions));
    }
    errors
}

#[derive(Clone, Copy)]
struct SelectorDecision {
    main: Decision,
    pull_request: Decision,
}

impl SelectorDecision {
    fn depot_declared(self) -> bool {
        self.main == Decision::DepotGated
            || matches!(
                self.pull_request,
                Decision::DepotGated | Decision::SentinelRef
            )
    }

    fn depot_reachable(self) -> bool {
        self.depot_declared() || self.main == Decision::OperatorInputHosted
    }
}

fn input<'a>(selector: &'a Selector, key: &str) -> &'a str {
    selector.inputs.get(key).map_or("false", String::as_str)
}

fn selector_decisions(
    inventory: &Inventory,
    errors: &mut Vec<String>,
) -> BTreeMap<String, SelectorDecision> {
    let mut decisions = BTreeMap::new();
    for (key, selector) in &inventory.selectors {
        match decide(selector) {
            Ok(decision) => {
                decisions.insert(key.clone(), decision);
            }
            Err(reason) => errors.push(format!("{key} ({}): {reason}", selector.workflow)),
        }
    }
    decisions
}

fn decide(selector: &Selector) -> Result<SelectorDecision, String> {
    let force_hosted = input(selector, "force_hosted");
    if force_hosted == "true" {
        return Ok(SelectorDecision {
            main: Decision::NoDepot,
            pull_request: Decision::NoDepot,
        });
    }
    let main = match (
        input(selector, "depot_main_enabled"),
        input(selector, "manual_use_depot"),
    ) {
        ("false", "false") => Decision::NoDepot,
        (MAIN_GATE, RELEASE_MANUAL_GATE) if force_hosted == RELEASE_FORCE_HOSTED => {
            Decision::OperatorInputHosted
        }
        (MAIN_GATE, _) => Decision::DepotGated,
        (main, manual) => {
            return Err(format!(
                "unrecognized main selector inputs {main} / {manual}"
            ));
        }
    };
    let pull_request = match (
        input(selector, "depot_pr_enabled"),
        input(selector, "pr_canary_ref"),
    ) {
        (PR_GATE, _) => Decision::DepotGated,
        ("false", SENTINEL_REF) => Decision::SentinelRef,
        ("false", "false" | "") => Decision::NoDepot,
        (pr, canary) => return Err(format!("unrecognized PR selector inputs {pr} / {canary}")),
    };
    Ok(SelectorDecision { main, pull_request })
}

fn coverage(matrix: &Matrix, inventory: &Inventory) -> Vec<String> {
    let mut errors = Vec::new();
    let mut owner = BTreeMap::new();
    for role in &matrix.roles {
        for member in role.selectors.iter().chain(&role.jobs) {
            if let Some(previous) = owner.insert(member.as_str(), role.id.as_str()) {
                errors.push(format!("{member} is in rows {previous} and {}", role.id));
            }
        }
        for selector in &role.selectors {
            if !inventory.selectors.contains_key(selector) {
                errors.push(format!("row {} names missing selector {selector}", role.id));
            }
        }
        for job in &role.jobs {
            if !inventory.jobs.contains_key(job) {
                errors.push(format!("row {} names missing job {job}", role.id));
            }
        }
    }
    for key in inventory.selectors.keys() {
        if !owner.contains_key(key.as_str()) {
            errors.push(format!("selector {key} has no runner role row"));
        }
    }
    for (key, job) in &inventory.jobs {
        let listed = owner.contains_key(key.as_str());
        if job.selectors.is_empty() && job.needs_row() && !listed {
            errors.push(format!("runner-selecting job {key} has no runner role row"));
        }
        if !job.selectors.is_empty() && listed {
            errors.push(format!(
                "{key} uses a selector; its row is the selector's row"
            ));
        }
    }
    errors
}

fn row_shape(role: &Role, inventory: &Inventory) -> Vec<String> {
    let mut errors = Vec::new();
    let id = &role.id;
    if role.summary.trim().is_empty() || role.trust.is_empty() || role.architecture.is_empty() {
        errors.push(format!("row {id} needs a summary, trust and architecture"));
    }
    if role.selectors.is_empty() && role.jobs.is_empty() {
        errors.push(format!("row {id} owns no selector or job"));
    }
    for dependency in &role.artifact_dependencies {
        if !inventory.jobs.contains_key(dependency) {
            errors.push(format!("row {id} depends on missing producer {dependency}"));
        }
    }
    if !role.readiness.evidence.starts_with(EVIDENCE_PREFIX) {
        errors.push(format!(
            "row {id} readiness evidence must link the repository"
        ));
    }
    let active = role.readiness.state == ReadinessState::Active;
    let main_depot = matches!(role.main, Decision::DepotGated | Decision::ManualDiagnostic);
    if active != main_depot {
        errors.push(format!(
            "row {id}: only rows that run on Depot on main are active"
        ));
    }
    let operator = role.hosted_fallback == HostedFallback::OperatorInput;
    if operator != !role.hosted_required_inputs.is_empty() {
        errors.push(format!(
            "row {id}: operator-input fallback needs hosted_required_inputs"
        ));
    }
    let selected = role.hosted_fallback == HostedFallback::Selector;
    if selected && !role.providers.contains(&Provider::Depot) {
        errors.push(format!(
            "row {id}: selector fallback is only for Depot-capable rows"
        ));
    }
    let hosted_only = role.providers == BTreeSet::from([Provider::GithubHosted]);
    if role.hosted_fallback == HostedFallback::AlwaysHosted && !hosted_only {
        errors.push(format!(
            "row {id}: always-hosted rows allow only GitHub-hosted"
        ));
    }
    if role.cache_mode == CacheMode::ProviderDerived && role.selectors.is_empty() {
        errors.push(format!("row {id}: provider-derived cache needs a selector"));
    }
    errors
}

fn row_decisions(
    role: &Role,
    inventory: &Inventory,
    decisions: &BTreeMap<String, SelectorDecision>,
) -> Vec<String> {
    let mut errors = Vec::new();
    for key in &role.selectors {
        let Some(decision) = decisions.get(key) else {
            continue;
        };
        if decision.main != role.main || decision.pull_request != role.pull_request {
            errors.push(format!(
                "row {}: selector {key} decides main {:?} / PR {:?}, row says {:?} / {:?}",
                role.id, decision.main, decision.pull_request, role.main, role.pull_request
            ));
        }
    }
    if role.selectors.is_empty() {
        errors.extend(fixed_row_decisions(role, inventory));
    }
    errors
}

fn fixed_row_decisions(role: &Role, inventory: &Inventory) -> Vec<String> {
    let jobs = role.jobs.iter().filter_map(|key| inventory.jobs.get(key));
    let depot = jobs
        .clone()
        .any(|job| job.labels.iter().any(|l| provider_of(l) == Provider::Depot));
    let expected_main = if depot {
        Decision::ManualDiagnostic
    } else {
        Decision::NoDepot
    };
    let mut errors = Vec::new();
    if role.main != expected_main || role.pull_request != Decision::NoDepot {
        errors.push(format!(
            "row {}: fixed-label row decides {expected_main:?} / {:?}",
            role.id,
            Decision::NoDepot
        ));
    }
    let manual_only = BTreeSet::from(["workflow_dispatch".to_owned()]);
    for job in jobs.filter(|_| depot) {
        if inventory.triggers.get(&job.workflow) != Some(&manual_only) {
            errors.push(format!(
                "row {}: fixed Depot labels in {} must be manual-dispatch only",
                role.id, job.workflow
            ));
        }
    }
    errors
}

fn row_placement(
    role: &Role,
    inventory: &Inventory,
    decisions: &BTreeMap<String, SelectorDecision>,
) -> Vec<String> {
    let mut errors = Vec::new();
    let mut labels = BTreeSet::new();
    let mut profiles = BTreeSet::new();
    let mut dynamic = false;
    for (key, job) in member_jobs(role, inventory) {
        labels.extend(job_labels(
            role,
            job,
            inventory,
            decisions,
            SelectorDecision::depot_declared,
        ));
        profiles.extend(job.profiles.iter().copied());
        dynamic |= !job.dynamic.is_empty();
        let reachable = job_labels(
            role,
            job,
            inventory,
            decisions,
            SelectorDecision::depot_reachable,
        );
        errors.extend(unsupported_combinations(role, key, job, &reachable));
    }
    let providers: BTreeSet<Provider> = labels.iter().map(|l| provider_of(l)).collect();
    let complete = if dynamic {
        labels.is_subset(&role.labels) && providers.is_subset(&role.providers)
    } else {
        labels == role.labels && providers == role.providers
    };
    if !complete {
        errors.push(format!(
            "row {}: workflows place on labels {labels:?} / providers {providers:?}, row says {:?} / {:?}",
            role.id, role.labels, role.providers
        ));
    }
    if profiles != role.preparation_profiles {
        errors.push(format!(
            "row {}: workflows prepare with {profiles:?}, row says {:?}",
            role.id, role.preparation_profiles
        ));
    }
    errors
}

fn member_jobs<'a>(role: &'a Role, inventory: &'a Inventory) -> Vec<(&'a str, &'a Job)> {
    inventory
        .jobs
        .iter()
        .filter(|(key, job)| {
            role.jobs.contains(key) || job.selectors.keys().any(|s| role.selectors.contains(s))
        })
        .map(|(key, job)| (key.as_str(), job))
        .collect()
}

// A job shared by two rows (the Linux runtime) is placed by each row's own selector.
fn job_labels(
    role: &Role,
    job: &Job,
    inventory: &Inventory,
    decisions: &BTreeMap<String, SelectorDecision>,
    depot: fn(SelectorDecision) -> bool,
) -> BTreeSet<String> {
    let mut labels = job.labels.clone();
    let own = job
        .selectors
        .iter()
        .filter(|(selector, _)| role.selectors.contains(*selector));
    for (selector, outputs) in own {
        let depot = decisions.get(selector).is_some_and(|d| depot(*d));
        for output in outputs {
            let Some((hosted, depot_label)) = inventory.runner_labels.get(output) else {
                continue;
            };
            labels.insert(hosted.clone());
            if depot {
                labels.insert(depot_label.clone());
            }
        }
    }
    labels
}

fn unsupported_combinations(
    role: &Role,
    key: &str,
    job: &Job,
    reachable: &BTreeSet<String>,
) -> Vec<String> {
    let providers: BTreeSet<Provider> = reachable.iter().map(|l| provider_of(l)).collect();
    let off_hosted = providers.iter().any(|p| *p != Provider::GithubHosted);
    let on_shared = providers.iter().any(|p| *p != Provider::SelfHosted);
    let mut errors = Vec::new();
    if job.profiles.contains(&Profile::HostedBare)
        && off_hosted
        && role.hosted_fallback != HostedFallback::OperatorInput
    {
        errors.push(format!(
            "{key}: hosted-bare preparation is reachable on {providers:?}; it requires GitHub-hosted"
        ));
    }
    let preinstalled = [Profile::PreinstalledLinux, Profile::PreinstalledMacos];
    if preinstalled.iter().any(|p| job.profiles.contains(p)) && on_shared {
        errors.push(format!(
            "{key}: preinstalled preparation is reachable on {providers:?}; it requires self-hosted"
        ));
    }
    errors
}
