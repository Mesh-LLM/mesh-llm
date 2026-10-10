//! Retained embedding SDK preparation belongs to the frozen controller jobs.
use super::{Node, field, handoffs as h};
use crate::command::DynResult;

const ACTION: &str = "./.github/actions/setup-canary-python";
const SDK: &str = "SKIPPY_WORKLOAD_SDK_PYTHON";
// Named selectors affecting this action's tool, dependency source, and SDK slot.
// Ordinary tracing and workflow metadata do not change this admission.
const PREPARATION_SELECTORS: &[&str] = &[
    "PATH",
    "HOME",
    "PYTHONHOME",
    "PYTHONPATH",
    "SDK_KIND",
    "SKIPPY_WORKLOAD_SDK_PYTHON",
    "MESH_REQUIRED_SDK_PYTHON",
    "UV_PYTHON",
    "UV_PYTHON_PREFERENCE",
    "UV_PYTHON_INSTALL_DIR",
    "UV_PYTHON_DOWNLOADS",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PROJECT",
    "UV_WORKING_DIRECTORY",
    "UV_CACHE_DIR",
    "UV_CONFIG_FILE",
    "UV_NO_CONFIG",
    "UV_INDEX",
    "UV_DEFAULT_INDEX",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX_STRATEGY",
    "UV_INSECURE_HOST",
    "UV_FIND_LINKS",
    "UV_NO_INDEX",
    "UV_OFFLINE",
    "UV_LOCKED",
    "UV_FROZEN",
    "UV_NO_SYNC",
    "UV_NO_VERIFY_HASHES",
    "UV_NO_BUILD",
    "UV_NO_BUILD_ISOLATION",
    "PIP_CONFIG_FILE",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_TRUSTED_HOST",
    "PIP_FIND_LINKS",
    "PIP_NO_INDEX",
    "PIP_REQUIRE_VIRTUALENV",
    "PIP_TARGET",
    "PIP_PREFIX",
    "PIP_USER",
];

pub(super) fn check(document: &Node) -> DynResult<()> {
    for (job, consumer) in [
        ("build", "build"),
        ("family", "certify"),
        ("retry_family", "certify"),
    ] {
        admission(h::job(document, job)?, consumer)?;
    }
    Ok(())
}

fn admission(job: &Node, consumer: &str) -> DynResult<()> {
    let steps = h::steps(job)?;
    let controller = h::checkout(steps, "${{ inputs.source }}", None)?;
    let (execute, command) = h::step(steps, "id", consumer)?;
    let preparations = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| field(step, "uses") == Some(ACTION))
        .collect::<Vec<_>>();
    let [(prepare, step)] = preparations.as_slice() else {
        return Err("canary job needs exactly one retained embedding SDK preparation".into());
    };
    h::before(controller, *prepare)?;
    h::before(*prepare, execute)?;
    if step.get("if").is_some()
        || step
            .get("continue-on-error")
            .is_some_and(|value| value.text() != Some("false"))
    {
        return Err(
            "embedding SDK preparation must fail closed without conditional skipping".into(),
        );
    }
    if let Some(inputs) = step.get("with") {
        let Node::Map(entries) = inputs else {
            return Err("SDK preparation inputs must be a map".into());
        };
        if entries
            .iter()
            .any(|(key, value)| key != "sdk-kind" || value.text() != Some("embedding"))
        {
            return Err("canary preparation must select the embedding SDK class".into());
        }
    }
    if let Some(environment) = step.get("env") {
        let Node::Map(entries) = environment else {
            return Err("SDK preparation environment must be a map".into());
        };
        if entries
            .iter()
            .any(|(key, _)| PREPARATION_SELECTORS.contains(&key.as_str()))
        {
            return Err("canary SDK preparation cannot substitute acquisition selectors".into());
        }
    }
    for owner in [job, command] {
        if owner.get("env").is_some_and(|env| env.get(SDK).is_some()) {
            return Err("canary consumer must inherit the prepared SDK interpreter".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn workflow() -> Node {
        super::super::workflow_yaml::parse(include_str!(
            "../../../../../.github/workflows/llama-canary-family-pass.yml"
        ))
        .unwrap()
    }
    fn steps<'a>(document: &'a mut Node, job: &str) -> &'a mut Vec<Node> {
        let job = h::mutable(h::mutable(document, "jobs"), job);
        let Node::Seq(steps) = h::mutable(job, "steps") else {
            panic!("steps required")
        };
        steps
    }
    fn insert(node: &mut Node, key: &str, value: Node) {
        let Node::Map(entries) = node else {
            panic!("mapping required")
        };
        entries.push((key.into(), value));
    }
    #[test]
    fn actual_controller_jobs_prepare_embedding_sdk_before_native_consumers() {
        check(&workflow()).unwrap();
    }
    #[test]
    fn every_job_refuses_missing_duplicate_late_or_untrusted_preparation() {
        for job in ["build", "family", "retry_family"] {
            for mode in ["missing", "duplicate", "late", "before-checkout"] {
                let mut document = workflow();
                let steps = steps(&mut document, job);
                let index = steps
                    .iter()
                    .position(|step| field(step, "uses") == Some(ACTION))
                    .unwrap();
                let step = steps.remove(index);
                match mode {
                    "missing" => {}
                    "duplicate" => {
                        steps.insert(index, step.clone());
                        steps.insert(index, step);
                    }
                    "late" => steps.push(step),
                    "before-checkout" => steps.insert(0, step),
                    _ => unreachable!(),
                }
                assert!(check(&document).is_err(), "{job}/{mode}");
            }
        }
    }
    #[test]
    fn every_job_refuses_skips_failure_bypass_profile_and_environment_substitution() {
        for job in ["build", "family", "retry_family"] {
            for (key, value) in [
                ("if", Node::Scalar("false".into())),
                ("continue-on-error", Node::Scalar("true".into())),
                (
                    "with",
                    Node::Map(vec![(
                        "sdk-kind".into(),
                        Node::Scalar("compatibility".into()),
                    )]),
                ),
                (
                    "with",
                    Node::Map(vec![(
                        "sdk-kind".into(),
                        Node::Scalar("${{ inputs.sdk_kind }}".into()),
                    )]),
                ),
                (
                    "env",
                    Node::Map(vec![(
                        "UV_PROJECT_ENVIRONMENT".into(),
                        Node::Scalar("/tmp/foreign".into()),
                    )]),
                ),
            ] {
                let mut document = workflow();
                let step = steps(&mut document, job)
                    .iter_mut()
                    .find(|step| field(step, "uses") == Some(ACTION))
                    .unwrap();
                insert(step, key, value);
                assert!(check(&document).is_err(), "{job}/{key}");
            }
        }
    }
    #[test]
    fn every_job_refuses_consumer_or_job_interpreter_override() {
        for job in ["build", "family", "retry_family"] {
            for at_job in [true, false] {
                let mut document = workflow();
                let owner = h::mutable(h::mutable(&mut document, "jobs"), job);
                let owner = if at_job {
                    owner
                } else {
                    let Node::Seq(steps) = h::mutable(owner, "steps") else {
                        unreachable!()
                    };
                    steps
                        .iter_mut()
                        .find(|step| {
                            field(step, "id")
                                == Some(if job == "build" { "build" } else { "certify" })
                        })
                        .unwrap()
                };
                if owner.get("env").is_none() {
                    insert(owner, "env", Node::Map(vec![]));
                }
                insert(
                    h::mutable(owner, "env"),
                    SDK,
                    Node::Scalar("/tmp/foreign/python".into()),
                );
                assert!(check(&document).is_err(), "{job}/job={at_job}");
            }
        }
    }
    #[test]
    fn explicit_embedding_and_literal_fail_closed_setting_are_admitted() {
        let mut document = workflow();
        for job in ["build", "family", "retry_family"] {
            let step = steps(&mut document, job)
                .iter_mut()
                .find(|step| field(step, "uses") == Some(ACTION))
                .unwrap();
            insert(
                step,
                "with",
                Node::Map(vec![("sdk-kind".into(), Node::Scalar("embedding".into()))]),
            );
            insert(step, "continue-on-error", Node::Scalar("false".into()));
        }
        check(&document).unwrap();
    }
    #[test]
    fn unrelated_preparation_metadata_is_admitted_in_every_job() {
        let mut document = workflow();
        for job in ["build", "family", "retry_family"] {
            let step = steps(&mut document, job)
                .iter_mut()
                .find(|step| field(step, "uses") == Some(ACTION))
                .unwrap();
            insert(
                step,
                "env",
                Node::Map(vec![
                    ("TRACE_RUN_LABEL".into(), Node::Scalar("SDK smoke".into())),
                    (
                        "OBSERVATION_COMPONENT".into(),
                        Node::Scalar("embedding".into()),
                    ),
                ]),
            );
        }
        check(&document).unwrap();
    }
    #[test]
    fn every_named_acquisition_selector_is_refused_without_rejecting_metadata() {
        for selector in PREPARATION_SELECTORS {
            let mut document = workflow();
            let step = steps(&mut document, "build")
                .iter_mut()
                .find(|step| field(step, "uses") == Some(ACTION))
                .unwrap();
            insert(
                step,
                "env",
                Node::Map(vec![
                    ("TRACE_RUN_LABEL".into(), Node::Scalar("allowed".into())),
                    ((*selector).into(), Node::Scalar("substituted".into())),
                ]),
            );
            assert!(check(&document).is_err(), "selector={selector}");
        }
    }
}
