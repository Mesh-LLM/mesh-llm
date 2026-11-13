//! Backend edges and singleton capacity are an architecture invariant.
use super::{Node, field};
use crate::command::DynResult;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let lane = workflows
        .get("ci-linux-lane.yml")
        .ok_or("Linux lane missing")?;
    verify(lane).map_err(Into::into)
}

fn dependencies(node: &Node) -> BTreeSet<&str> {
    node.get("needs")
        .map(Node::list)
        .unwrap_or_default()
        .into_iter()
        .collect()
}

fn verify(lane: &Node) -> Result<(), String> {
    let jobs = lane.get("jobs").ok_or("Linux jobs missing")?;
    let summary = dependencies(jobs.get("summary").ok_or("Linux summary missing")?);
    for backend in ["cpu", "cuda", "rocm", "vulkan"] {
        let runtime = format!("native_runtime_{backend}");
        let product = format!("product_{backend}");
        let smoke = format!("product_smoke_{backend}");
        for (name, expected, slice) in [
            (
                runtime.as_str(),
                vec!["authority_linux_x64", "product_automation_linux_x64"],
                "runtime",
            ),
            (
                product.as_str(),
                vec![
                    "hosts",
                    runtime.as_str(),
                    "authority_linux_x64",
                    "product_automation_linux_x64",
                ],
                "product",
            ),
            (smoke.as_str(), vec![product.as_str()], "product-smoke"),
        ] {
            let job = jobs
                .get(name)
                .ok_or_else(|| format!("backend call {name} missing"))?;
            if dependencies(job) != expected.into_iter().collect() {
                return Err(format!(
                    "{name} must need only its own inputs; runtimes must not need the host"
                ));
            }
            let inputs = job.get("with").ok_or("backend inputs missing")?;
            if field(inputs, "backend") != Some(backend)
                || field(job, "uses")
                    != Some(format!("./.github/workflows/ci-linux-{slice}-slice.yml").as_str())
                || !summary.contains(name)
            {
                return Err(format!(
                    "{name} backend, slice or summary admission changed"
                ));
            }
            if slice != "product-smoke" && field(inputs, "max_parallel") != Some("1") {
                return Err(format!("{name} must retain singleton capacity"));
            }
        }
    }
    for (name, expected) in [
        ("sdk_rust", vec!["product_cpu"]),
        ("sdk_kotlin", vec!["product_cpu", "kotlin_sdk_input"]),
    ] {
        if jobs.get(name).map(dependencies) != Some(expected.into_iter().collect())
            || !summary.contains(name)
        {
            return Err(format!("{name} must need only CPU and its SDK producer"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_validation::lane_results::workflow_yaml;

    fn source() -> &'static str {
        include_str!("../../../../../.github/workflows/ci-linux-lane.yml")
    }

    #[test]
    fn independent_runtimes_and_matching_consumers_preserve_the_approved_graph() {
        verify(&workflow_yaml::parse(source()).unwrap()).unwrap();
    }

    #[test]
    fn host_first_unrelated_backend_and_multiplied_capacity_are_rejected() {
        for (from, to) in [
            (
                "needs: [authority_linux_x64, product_automation_linux_x64]",
                "needs: [hosts, authority_linux_x64, product_automation_linux_x64]",
            ),
            (
                "needs: [hosts, native_runtime_cpu, authority_linux_x64, product_automation_linux_x64]",
                "needs: [hosts, native_runtime_cpu, native_runtime_cuda, authority_linux_x64, product_automation_linux_x64]",
            ),
            ("max_parallel: 1", "max_parallel: 7"),
            ("needs: [product_cpu]", "needs: [product_cpu, product_cuda]"),
        ] {
            assert!(source().contains(from));
            let mutated = source().replace(from, to);
            assert!(
                verify(&workflow_yaml::parse(&mutated).unwrap()).is_err(),
                "{from}"
            );
        }
    }
}
