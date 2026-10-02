use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize, PartialEq, Eq)]
#[serde(tag = "engine", rename_all = "kebab-case")]
enum BuildIdentity {
    Mesh {
        commit: String,
        binary_sha256: String,
        runtime_sha256: String,
    },
    External {
        version_sha256: String,
    },
}

#[derive(Deserialize)]
struct Snapshot {
    plan_sha256: String,
    manifest_sha256: String,
    builds: BTreeMap<String, BuildIdentity>,
    completed: Vec<Completed>,
}

#[derive(Deserialize)]
struct Completed {
    pass: u32,
    label: String,
}

#[derive(Deserialize)]
struct Input {
    passes: u32,
    labels: Vec<String>,
    current: Snapshot,
    previous: Option<Snapshot>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct Scheduled {
    pass: u32,
    label: String,
}

fn schedule(input: &Input) -> DynResult<Vec<Scheduled>> {
    if !(1..=1000).contains(&input.passes) || input.labels.is_empty() {
        return Err("pass schedule needs 1..=1000 passes and at least one arm".into());
    }
    let labels: BTreeSet<_> = input.labels.iter().collect();
    if labels.len() != input.labels.len()
        || labels.iter().any(|label| label.is_empty())
        || labels != input.current.builds.keys().collect()
    {
        return Err("arm labels must be unique and cover build identities exactly".into());
    }
    let mut completed = BTreeSet::new();
    if let Some(previous) = &input.previous {
        if previous.plan_sha256 != input.current.plan_sha256 {
            return Err("cannot resume: plan differs".into());
        }
        if previous.manifest_sha256 != input.current.manifest_sha256 {
            return Err("cannot resume: trajectory manifest differs".into());
        }
        if previous.builds != input.current.builds {
            return Err("cannot resume: arm build identity differs".into());
        }
        for row in &previous.completed {
            if !(1..=input.passes).contains(&row.pass)
                || !labels.contains(&row.label)
                || !completed.insert((row.pass, row.label.as_str()))
            {
                return Err("resume results contain a duplicate or foreign arm/pass".into());
            }
        }
    }
    let mut schedule = Vec::new();
    for pass in 1..=input.passes {
        let mut arms = input.labels.iter().collect::<Vec<_>>();
        if pass % 2 == 0 {
            arms.reverse();
        }
        for label in arms {
            if !completed.contains(&(pass, label.as_str())) {
                schedule.push(Scheduled {
                    pass,
                    label: label.clone(),
                });
            }
        }
    }
    Ok(schedule)
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix pass-schedule --input PATH --output PATH",
        values: &["--input", "--output"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input = parsed.last("--input").ok_or("missing --input")?;
    let output = parsed.last("--output").ok_or("missing --output")?;
    let input = serde_json::from_slice(&std::fs::read(input)?)?;
    crate::command::write_json_file(std::path::Path::new(output), &schedule(&input)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> serde_json::Value {
        serde_json::json!({"passes":3,"labels":["baseline","candidate"],"current":{
        "plan_sha256":"plan","manifest_sha256":"manifest","completed":[],"builds":{
            "baseline":{"engine":"mesh","commit":"base","binary_sha256":"binary-a","runtime_sha256":"runtime-a"},
            "candidate":{"engine":"mesh","commit":"head","binary_sha256":"binary-b","runtime_sha256":"runtime-b"}
        }}})
    }

    #[test]
    fn passes_reverse_arm_order_without_sorting_labels() {
        let input: Input = serde_json::from_value(input()).unwrap();
        let order = schedule(&input).unwrap();
        assert_eq!(
            order
                .iter()
                .map(|row| (row.pass, row.label.as_str()))
                .collect::<Vec<_>>(),
            [
                (1, "baseline"),
                (1, "candidate"),
                (2, "candidate"),
                (2, "baseline"),
                (3, "baseline"),
                (3, "candidate")
            ]
        );
    }

    #[test]
    fn resume_skips_only_matching_completed_cells() {
        let mut document = input();
        document["previous"] = document["current"].clone();
        document["previous"]["completed"] = serde_json::json!([{"pass":1,"label":"baseline"}]);
        let input: Input = serde_json::from_value(document).unwrap();
        let order = schedule(&input).unwrap();
        assert_eq!(order.len(), 5);
        assert_eq!(
            order[0],
            Scheduled {
                pass: 1,
                label: "candidate".into()
            }
        );
    }

    #[test]
    fn changed_inputs_and_duplicate_results_reject_resume() {
        for field in ["plan_sha256", "manifest_sha256"] {
            let mut document = input();
            document["previous"] = document["current"].clone();
            document["previous"][field] = "changed".into();
            assert!(schedule(&serde_json::from_value(document).unwrap()).is_err());
        }
        let mut document = input();
        document["previous"] = document["current"].clone();
        document["previous"]["builds"]["candidate"]["binary_sha256"] = "changed".into();
        assert!(schedule(&serde_json::from_value(document).unwrap()).is_err());
        let mut document = input();
        document["previous"] = document["current"].clone();
        document["previous"]["completed"] =
            serde_json::json!([{"pass":1,"label":"baseline"},{"pass":1,"label":"baseline"}]);
        assert!(schedule(&serde_json::from_value(document).unwrap()).is_err());
    }
}
