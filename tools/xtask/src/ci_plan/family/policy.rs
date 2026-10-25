use super::document::Json;
use super::fields::{PlanResult, choice, exact, object, strings};
use super::text::FamilyString;
use std::collections::BTreeMap;

pub(super) const CORE: [&str; 3] = ["single-step", "chain", "state-handoff"];
pub(super) const NAMES: [&str; 5] = [
    "full",
    "package-oracle",
    "graph-only",
    "workload-smoke",
    "workload-oracle",
];

#[derive(Clone)]
pub(super) struct Profile {
    pub(super) status: String,
    pub(super) oracle: String,
    pub(super) lanes: Vec<FamilyString>,
}

pub(super) fn parse(value: Option<&Json>) -> PlanResult<BTreeMap<String, Profile>> {
    let policy = object(value, "policy")?;
    exact(policy, &["profiles"], "policy")?;
    let profiles = object(policy.get("profiles"), "policy.profiles")?;
    if profiles.as_object().map_or(0, |entries| entries.len()) != NAMES.len()
        || NAMES.iter().any(|name| profiles.get(name).is_none())
    {
        return Err("policy.profiles must define full, package-oracle, graph-only, workload-smoke, and workload-oracle".into());
    }
    let mut result = BTreeMap::new();
    for name in NAMES {
        let field = format!("policy.profiles.{name}");
        let entry = object(profiles.get(name), &field)?;
        exact(entry, &["status", "oracle", "required_lanes"], &field)?;
        let status = choice(
            entry.get("status"),
            &format!("{field}.status"),
            &["certified", "provisional"],
        )?;
        let oracle = choice(
            entry.get("oracle"),
            &format!("{field}.oracle"),
            &["local-monolithic", "independent-trace", "none"],
        )?;
        let lanes = strings(
            entry.get("required_lanes"),
            &format!("{field}.required_lanes"),
        )?;
        let profile = Profile {
            status,
            oracle,
            lanes,
        };
        validate(name, &profile)?;
        result.insert(name.to_owned(), profile);
    }
    if result["full"].oracle != "local-monolithic" {
        return Err("full profile must use the local-monolithic oracle".into());
    }
    if result["package-oracle"].oracle != "independent-trace" {
        return Err("package-oracle must use an independent trace".into());
    }
    Ok(result)
}

fn validate(name: &str, profile: &Profile) -> PlanResult<()> {
    let Profile {
        status,
        oracle,
        lanes,
    } = profile;
    let matches = |expected: &[&str]| {
        lanes.len() == expected.len()
            && lanes
                .iter()
                .zip(expected)
                .all(|(lane, expected)| lane == *expected)
    };
    match name {
        "full" | "package-oracle" => {
            if status != "certified" || !matches(&CORE) {
                return Err(format!(
                    "certified profile {name} must require exactly the three core lanes"
                ));
            }
        }
        "workload-oracle" => {
            if status != "certified"
                || oracle != "local-monolithic"
                || !matches(&["class-specific-smoke", "class-specific-oracle"])
            {
                return Err(
                    "workload-oracle requires certified local-monolithic smoke and oracle lanes"
                        .into(),
                );
            }
        }
        "graph-only" | "workload-smoke" => {
            if status != "provisional" || oracle != "none" {
                return Err(format!("{name} must remain provisional and oracle-free"));
            }
            if name == "graph-only" && !matches(&["graph-parse", "tensor-ownership", "stage-load"])
            {
                return Err("graph-only must require exactly the three graph lanes".into());
            }
            if name == "workload-smoke" && !matches(&["class-specific-smoke"]) {
                return Err(
                    "workload-smoke must require exactly the class-specific smoke lane".into(),
                );
            }
        }
        _ => return Err(format!("unknown profile: {name}")),
    }
    Ok(())
}
