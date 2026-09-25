//! Compiler-seed and SDK Rust consumer checks of `check` in
//! `scripts/runner-image-identity.py`: one exact seed key expression, one
//! restore action per consumer, a single publisher and the SDK cache key.

use crate::ci_operations::python_access::{Outcome, is_str, item, item_by, require};
use crate::ci_operations::workflow_census::{Workflows, job, read_text};
use crate::ci_operations::workflow_text::{any_line, job_steps, line_is, one_field, seed_keys};
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::display;
use crate::repository::python_text::repr;
use std::path::Path;

type JobId = (String, String);

pub(crate) fn inputs_repr(list: &Json) -> String {
    list.as_array()
        .unwrap_or_default()
        .iter()
        .map(|entry| crate::prepared_input::python_value::repr(Some(entry)))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn seed_expression(catalog: &Json) -> Outcome<String> {
    let seed = item(catalog, "compiler_seed")?;
    let prefix = display(Some(item(seed, "key_prefix")?));
    Ok(format!(
        "{prefix}${{{{ hashFiles({}) }}}}",
        inputs_repr(item(seed, "recipe_inputs")?)
    ))
}

fn binding_ids(roles: &Json, role_id: &Json) -> Outcome<Vec<JobId>> {
    let mut ids = Vec::new();
    for binding in item(item_by(roles, role_id)?, "bindings")?
        .as_array()
        .unwrap_or_default()
    {
        ids.push((
            display(Some(item(binding, "workflow")?)),
            display(Some(item(binding, "job")?)),
        ));
    }
    Ok(ids)
}

fn sorted(mut ids: Vec<JobId>) -> Vec<JobId> {
    ids.sort();
    ids
}

fn has_line(step: &str, test: impl Fn(&str) -> bool) -> bool {
    any_line(step, test)
}

fn is_seed_key_step(step: &str) -> bool {
    has_line(step, |rest| {
        let spaces = rest.len() - rest.trim_start_matches(' ').len();
        spaces > 0
            && rest[spaces..]
                .strip_prefix("id: seed_key")
                .is_some_and(|tail| tail.is_empty() || tail.starts_with('\n'))
    })
}

fn is_restore_step(step: &str) -> bool {
    has_line(step, |rest| {
        let body = rest.trim_start_matches(' ');
        let body = body.strip_prefix("- ").unwrap_or(body);
        line_is(body, "uses: ./.github/actions/restore-sccache-seed")
    })
}

/// Returns the number of restore consumers observed.
pub(crate) fn check_seed(catalog: &Json, workflows: &Workflows, root: &Path) -> Outcome<usize> {
    let (roles, seed) = (
        item(catalog, "consumer_roles")?,
        item(catalog, "compiler_seed")?,
    );
    let runtime = item(seed, "runtime_consumer")?;
    let runtime_id = (
        display(Some(item(runtime, "workflow")?)),
        display(Some(item(runtime, "job")?)),
    );
    let expression = seed_expression(catalog)?;
    let mut expected: Vec<JobId> = binding_ids(roles, item(seed, "publisher_role")?)?;
    for consumer in item(seed, "consumer_roles")?.as_array().unwrap_or_default() {
        expected.extend(binding_ids(roles, consumer)?);
    }
    expected.push(runtime_id.clone());
    let canary_id = ("depot-canary.yml".to_owned(), "runtime_seed".to_owned());
    let canary = roles.get("runtime-seed-canary");
    if let Some(canary) = canary {
        check_canary(canary, seed, workflows, &canary_id, &expression)?;
        expected.push(canary_id.clone());
    }
    let mut observed_seed: Vec<JobId> = Vec::new();
    let mut observed_restore: Vec<JobId> = Vec::new();
    for (workflow, jobs) in workflows {
        for (job_id, body) in jobs {
            let keys = seed_keys(body);
            if !keys.is_empty() {
                require(keys == [expression.as_str()], || {
                    format!("{workflow}:{job_id}: compiler seed key drift")
                })?;
                observed_seed.push((workflow.clone(), job_id.clone()));
            }
            let id = (workflow.clone(), job_id.clone());
            for step in job_steps(body)
                .into_iter()
                .filter(|step| is_restore_step(step))
            {
                let expected_key = if canary.is_some() && id == canary_id {
                    "${{ steps.seed_key.outputs.key }}"
                } else {
                    &expression
                };
                let place = format!("{workflow}:{job_id}");
                require(
                    one_field(step, "cache_key", &place, None)? == expected_key,
                    || format!("{workflow}:{job_id}: compiler seed restore key drift"),
                )?;
                observed_restore.push(id.clone());
            }
        }
    }
    require(sorted(expected.clone()) == sorted(observed_seed), || {
        "compiler seed producer/consumer census drift".to_owned()
    })?;
    let publisher_role = item(seed, "publisher_role")?;
    let publisher_id = binding_ids(roles, publisher_role)?
        .into_iter()
        .next()
        .unwrap_or_default();
    let restorers: Vec<JobId> = expected
        .into_iter()
        .filter(|id| *id != publisher_id)
        .collect();
    let restore_count = observed_restore.len();
    require(sorted(observed_restore) == sorted(restorers), || {
        "compiler seed restore action census drift".to_owned()
    })?;
    check_publisher(workflows, &publisher_id, seed, root)?;
    let runtime_job = workflows
        .get(&runtime_id.0)
        .and_then(|jobs| job(jobs, &runtime_id.1))
        .unwrap_or_default();
    require(
        one_field(runtime_job, "allow_trusted_seed", "runtime consumer", None)? == "false",
        || "runtime seed restore is deliberately disabled".to_owned(),
    )?;
    Ok(restore_count)
}

fn check_canary(
    canary: &Json,
    seed: &Json,
    workflows: &Workflows,
    canary_id: &JobId,
    expression: &str,
) -> Outcome<()> {
    let bindings = item(canary, "bindings")?.as_array().unwrap_or_default();
    let bound =
        crate::ci_operations::python_access::eq(item(canary, "image_id")?, item(seed, "image_id")?)
            && is_str(item(canary, "scope")?, "ordinary")
            && bindings.len() == 1
            && is_str(item(&bindings[0], "workflow")?, &canary_id.0)
            && is_str(item(&bindings[0], "job")?, &canary_id.1);
    require(bound, || "runtime seed canary binding drift".to_owned())?;
    let body = workflows
        .get(&canary_id.0)
        .and_then(|jobs| job(jobs, &canary_id.1));
    let body = body.ok_or_else(|| repr(&canary_id.1))?;
    let steps: Vec<&str> = job_steps(body)
        .into_iter()
        .filter(|step| is_seed_key_step(step))
        .collect();
    let ok = steps.len() == 1
        && one_field(steps[0], "CANARY_KEY", "runtime seed canary", None)? == expression
        && one_field(steps[0], "run", "runtime seed canary", None)?
            == "python3 scripts/runtime-seed-canary.py preflight runtime-seed-evidence";
    require(ok, || "runtime seed canary key resolver drift".to_owned())
}

fn check_publisher(
    workflows: &Workflows,
    publisher_id: &JobId,
    seed: &Json,
    root: &Path,
) -> Outcome<()> {
    let body = workflows
        .get(&publisher_id.0)
        .and_then(|jobs| job(jobs, &publisher_id.1));
    let body = body.ok_or_else(|| repr(&publisher_id.1))?;
    let mut actions: Vec<&str> = Vec::new();
    for step in job_steps(body) {
        let first = ["restore", "save"]
            .into_iter()
            .filter_map(|kind| {
                step.find(&format!("uses: actions/cache/{kind}@"))
                    .map(|at| (at, kind))
            })
            .min()
            .map(|(_, kind)| kind);
        if let Some(kind) = first {
            require(
                one_field(step, "key", "compiler seed publisher", None)?
                    == "${{ steps.seed.outputs.key }}",
                || "compiler seed publisher cache key drift".to_owned(),
            )?;
            actions.push(kind);
        }
    }
    require(actions == ["restore", "save"], || {
        "compiler seed publisher cache action census drift".to_owned()
    })?;
    let recipe = display(Some(item(seed, "recipe")?));
    require(body.contains(&format!("run: just {recipe}\n")), || {
        "compiler seed recipe drift".to_owned()
    })?;
    let just = read_text(&root.join("just/ci.just"))?;
    let declared = any_line(&just, |rest| {
        rest.strip_prefix(recipe.as_str())
            .and_then(|tail| tail.strip_prefix(':'))
            .is_some_and(|tail| {
                crate::ci_operations::workflow_text::blank_to_eol(tail, 0).is_some()
            })
    });
    require(declared, || "compiler seed recipe is missing".to_owned())
}
