//! The lane result rules of `scripts/validate-ci-lane-results.py`: which
//! top-level lane jobs a projection plans, then planned jobs must succeed and
//! every other reported job must be skipped.

use super::Checked;
use crate::ci_plan::document::Json;
use crate::prepared_input::python_value::repr;
use crate::repository::python_text;
use std::collections::BTreeSet;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Lane {
    Quality,
    Website,
    Linux,
    Macos,
    Windows,
}

impl Lane {
    pub(super) const ALL: [Lane; 5] = [
        Lane::Quality,
        Lane::Website,
        Lane::Linux,
        Lane::Macos,
        Lane::Windows,
    ];

    fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|lane| lane.name() == name)
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Lane::Quality => "quality",
            Lane::Website => "website",
            Lane::Linux => "linux",
            Lane::Macos => "macos",
            Lane::Windows => "windows",
        }
    }

    /// The stable check label, as in `CI / macOS` and `PR / macOS`.
    pub(super) fn label(self) -> &'static str {
        match self {
            Lane::Quality => "Quality",
            Lane::Website => "Website",
            Lane::Linux => "Linux",
            Lane::Macos => "macOS",
            Lane::Windows => "Windows",
        }
    }
}

/// A validated projection: its lane and planned jobs in derivation order.
pub(super) struct Outcome {
    pub(super) lane: Lane,
    pub(super) planned: Vec<&'static str>,
}

pub(super) fn load_object(value: &str, label: &str) -> Checked<Json> {
    let parsed = Json::parse(value.as_bytes())
        .map_err(|error| format!("{label} is not valid JSON: {error}"))?;
    match parsed {
        Json::Object(_) => Ok(parsed),
        _ => Err(format!("{label} must be a JSON object")),
    }
}

/// Row IDs of one matrix. A missing matrix is empty, as `dict.get(m, [])`.
/// Duplicate IDs are rejected here; the legacy set collapsed them.
fn ids(plan: &Json, matrix: &str) -> Checked<Vec<String>> {
    let Some(Json::Object(_)) = plan.get("matrices") else {
        return Err("lane plan matrices must be an object".to_owned());
    };
    let rows = match plan
        .get("matrices")
        .and_then(|matrices| matrices.get(matrix))
    {
        None => return Ok(Vec::new()),
        Some(Json::Array(rows)) => rows,
        Some(_) => {
            let name = python_text::repr(matrix);
            return Err(format!("lane plan matrix {name} must be an array"));
        }
    };
    let mut seen = BTreeSet::new();
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let id = row
                .get("id")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("lane plan matrix {matrix}[{index}] needs an ID"))?;
            if !seen.insert(id) {
                let id = python_text::repr(id);
                return Err(format!(
                    "lane plan matrix {matrix} contains duplicate ID {id}"
                ));
            }
            Ok(id.to_owned())
        })
        .collect()
}

struct Planner<'a> {
    plan: &'a Json,
    slices: &'a [Json],
    jobs: Vec<&'static str>,
}

impl Planner<'_> {
    fn selected(&self, slice: &str) -> bool {
        self.slices.iter().any(|item| item.as_str() == Some(slice))
    }

    fn rows(&self, matrix: &str) -> Checked<Vec<String>> {
        ids(self.plan, matrix)
    }

    fn add(&mut self, when: bool, jobs: &[&'static str]) {
        if when {
            for job in jobs {
                if !self.jobs.contains(job) {
                    self.jobs.push(job);
                }
            }
        }
    }

    fn products(&mut self) -> Checked<Vec<String>> {
        let hosts = self.rows("hosts")?;
        let runtimes = self.rows("runtime_products")?;
        self.add(!hosts.is_empty(), &["ui_artifact", "hosts"]);
        Ok(runtimes)
    }

    fn linux(&mut self) -> Checked<()> {
        let runtimes = self.products()?;
        let sdk = self.rows("sdk")?;
        self.add(self.selected("static-abi"), &["static_abi"]);
        let rust_tests = !self.rows("rust_tests")?.is_empty();
        self.add(rust_tests, &["rust_tests"]);
        self.add(
            !runtimes.is_empty(),
            &["native_runtimes", "runtime_product"],
        );
        self.add(sdk.iter().any(|id| id == "kotlin"), &["kotlin_sdk_input"]);
        self.add(!sdk.is_empty(), &["sdk"]);
        let smoke = !self.rows("smoke")?.is_empty();
        self.add(smoke, &["product_smoke"]);
        Ok(())
    }

    fn macos(&mut self) -> Checked<()> {
        let runtimes = self.products()?;
        let sdk = self.rows("sdk")?;
        self.add(
            !runtimes.is_empty(),
            &["native_runtimes", "runtime_product"],
        );
        let checks = !self.rows("platform_checks")?.is_empty();
        self.add(checks, &["platform_checks"]);
        self.add(sdk.iter().any(|id| id == "swift"), &["swift_sdk_input"]);
        self.add(!sdk.is_empty(), &["sdk"]);
        let smoke = !self.rows("smoke")?.is_empty();
        self.add(smoke, &["product_smoke"]);
        let any = !self.jobs.is_empty();
        self.add(any, &["validate_plan"]);
        Ok(())
    }

    fn windows(&mut self) -> Checked<()> {
        let runtimes = self.products()?;
        self.add(
            !runtimes.is_empty(),
            &["native_runtimes", "runtime_product"],
        );
        let checks = !self.rows("platform_checks")?.is_empty();
        self.add(checks, &["platform_checks"]);
        Ok(())
    }
}

fn required_jobs(plan: &Json) -> Checked<Outcome> {
    let lane = plan.get("lane").and_then(Json::as_str);
    let slices = plan.get("required_slices").and_then(Json::as_array);
    let (Some(name), Some(slices)) = (lane, slices) else {
        return Err("lane plan needs lane and required_slices".to_owned());
    };
    // Python's `set(slices)` raises on unhashable items; reject them instead.
    if slices
        .iter()
        .any(|item| matches!(item, Json::Array(_) | Json::Object(_)))
    {
        return Err("lane plan required_slices must contain only scalars".to_owned());
    }
    let lane =
        Lane::parse(name).ok_or_else(|| format!("unknown CI lane {}", python_text::repr(name)))?;
    let mut planner = Planner {
        plan,
        slices,
        jobs: Vec::new(),
    };
    match lane {
        Lane::Quality => {
            planner.add(planner.selected("quality"), &["quality"]);
            planner.add(planner.selected("runner-contract"), &["runner_contract"]);
        }
        Lane::Website => planner.add(planner.selected("web"), &["web"]),
        Lane::Linux => planner.linux()?,
        Lane::Macos => planner.macos()?,
        Lane::Windows => planner.windows()?,
    }
    Ok(Outcome {
        lane,
        planned: planner.jobs,
    })
}

fn result(state: Option<&Json>) -> Option<&Json> {
    state.and_then(|state| state.get("result"))
}

fn is(value: Option<&Json>, expected: &str) -> bool {
    value.and_then(Json::as_str) == Some(expected)
}

pub(super) fn validate(plan: &Json, needs: &Json) -> Checked<Outcome> {
    let Some(Json::Bool(required)) = plan.get("required") else {
        return Err("lane plan required must be a boolean".to_owned());
    };
    let outcome = required_jobs(plan)?;
    if *required && outcome.planned.is_empty() {
        return Err("required lane has no planned jobs".to_owned());
    }
    for job in &outcome.planned {
        let state = result(needs.get(job));
        if !is(state, "success") {
            let (job, state) = (python_text::repr(job), repr(state));
            return Err(format!("planned job {job} finished with {state}"));
        }
    }
    for (job, state) in needs.as_object().unwrap_or_default() {
        let state = result(Some(state));
        if !outcome.planned.contains(&job.as_str()) && !is(state, "skipped") {
            let (job, state) = (python_text::repr(job), repr(state));
            return Err(format!("lane job {job} finished with {state}"));
        }
    }
    Ok(outcome)
}
