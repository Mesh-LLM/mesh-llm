//! Complete configured rosters and c1 parity gate promotion; no model or cross-platform proof.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Deserialize, Eq, PartialEq, Ord, PartialOrd)]
struct Cell {
    platform: String,
    model: String,
    workload: String,
    arm: String,
    concurrency: u64,
    output_tokens: u64,
}
#[derive(Clone, Deserialize, Serialize, PartialEq)]
struct Capacity {
    mode: String,
    comparison_kv_matched: bool,
}
#[derive(Deserialize)]
struct Row {
    cell: Cell,
    throughput: f64,
    complete: bool,
    capacity_policy: Capacity,
}
#[derive(Deserialize)]
struct Gate {
    cell: Cell,
    passed: bool,
}
#[derive(Deserialize)]
struct Config {
    concurrency: Vec<u64>,
    synthetic: Synthetic,
}
#[derive(Deserialize)]
struct Synthetic {
    output_tokens: Vec<u64>,
}
#[derive(Serialize)]
pub(super) struct Group {
    platform: String,
    model: String,
    candidates: Vec<Candidate>,
    winner: Option<String>,
}
#[derive(Serialize)]
struct Candidate {
    arm: String,
    complete: bool,
    c1_parity: bool,
    capacity_consistent: bool,
    synthetic_mean_percent: Option<f64>,
    thoughtworks_mean_percent: Option<f64>,
    eligible: bool,
    capacity_policy: Option<Capacity>,
    baseline_capacity_policy: Option<Capacity>,
}
type Key = (u64, Option<u64>);
impl Cell {
    fn key(&self) -> DynResult<Key> {
        if self.concurrency == 0
            || self.output_tokens == 0
            || self.platform.is_empty()
            || self.model.is_empty()
            || self.arm.is_empty()
        {
            return Err("promotion cell identity refused".into());
        }
        match self.workload.as_str() {
            "synthetic" => Ok((self.concurrency, Some(self.output_tokens))),
            "thoughtworks" => Ok((self.concurrency, None)),
            _ => Err("promotion workload refused".into()),
        }
    }
}
pub(super) fn evaluate(
    source: &Value,
    planned: &[Value],
    observed: &[Value],
    gates: &[Value],
) -> DynResult<Vec<Group>> {
    let config: Config = serde_json::from_value(source.clone())?;
    let expected = expected(&config)?;
    let planned: Vec<Cell> = planned
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()?;
    let rows: Vec<Row> = observed
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()?;
    let gates: Vec<Gate> = gates
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()?;
    let mut plan = BTreeSet::new();
    for cell in &planned {
        cell.key()?;
        if !plan.insert(cell.clone()) {
            return Err("duplicate promotion planned cell".into());
        }
    }
    let mut unique = BTreeSet::new();
    for row in &rows {
        row.cell.key()?;
        if !plan.contains(&row.cell)
            || !unique.insert(row.cell.clone())
            || !row.throughput.is_finite()
            || row.throughput <= 0.0
        {
            return Err("promotion observed row identity/throughput refused".into());
        }
    }
    let groups: BTreeSet<_> = planned
        .iter()
        .map(|c| (c.platform.clone(), c.model.clone()))
        .collect();
    let mut result = Vec::new();
    for (platform, model) in groups {
        let group_rows: Vec<_> = rows
            .iter()
            .filter(|r| r.cell.platform == platform && r.cell.model == model)
            .collect();
        let arms: BTreeSet<_> = planned
            .iter()
            .filter(|c| {
                c.platform == platform && c.model == model && c.arm != "llama" && c.arm != "mesh"
            })
            .map(|c| c.arm.clone())
            .collect();
        let mut candidates = Vec::new();
        for arm in arms {
            candidates.push(candidate(
                &arm,
                &platform,
                &model,
                &group_rows,
                &gates,
                &expected,
            )?);
        }
        let mut winner = None;
        let mut best = f64::NEG_INFINITY;
        // Sorted arm order plus strict greater-than preserves the first lexical arm on equal gain.
        for candidate in &candidates {
            if candidate.eligible {
                let gain = candidate
                    .synthetic_mean_percent
                    .ok_or("synthetic promotion mean")?
                    + candidate
                        .thoughtworks_mean_percent
                        .ok_or("trace promotion mean")?;
                if !gain.is_finite() {
                    return Err("promotion combined gain overflow".into());
                }
                if gain > best {
                    best = gain;
                    winner = Some(candidate.arm.clone());
                }
            }
        }
        result.push(Group {
            platform,
            model,
            candidates,
            winner,
        });
    }
    Ok(result)
}
fn expected(config: &Config) -> DynResult<BTreeMap<String, BTreeSet<Key>>> {
    let concurrency: BTreeSet<_> = config.concurrency.iter().copied().collect();
    let outputs: BTreeSet<_> = config.synthetic.output_tokens.iter().copied().collect();
    if concurrency.is_empty()
        || outputs.is_empty()
        || concurrency.contains(&0)
        || outputs.contains(&0)
        || concurrency.len() != config.concurrency.len()
        || outputs.len() != config.synthetic.output_tokens.len()
        || concurrency
            .len()
            .checked_mul(outputs.len())
            .is_none_or(|n| n > 100000)
    {
        return Err("promotion configured roster refused".into());
    }
    Ok(BTreeMap::from([
        (
            "synthetic".into(),
            concurrency
                .iter()
                .flat_map(|c| outputs.iter().map(move |o| (*c, Some(*o))))
                .collect(),
        ),
        (
            "thoughtworks".into(),
            concurrency.iter().map(|c| (*c, None)).collect(),
        ),
    ]))
}
fn candidate(
    arm: &str,
    platform: &str,
    model: &str,
    rows: &[&Row],
    gates: &[Gate],
    expected: &BTreeMap<String, BTreeSet<Key>>,
) -> DynResult<Candidate> {
    let mut complete = true;
    let mut means = BTreeMap::new();
    for (workload, keys) in expected {
        let (mean, all) = mean_gain(rows, arm, workload, keys)?;
        complete &= all;
        means.insert(workload.clone(), mean);
    }
    let capacity = policy(rows, arm);
    let baseline = policy(rows, "mesh");
    let capacity_consistent = capacity.is_some() && baseline.is_some();
    let c1: Vec<_> = gates
        .iter()
        .filter(|g| {
            g.cell.platform == platform
                && g.cell.model == model
                && g.cell.arm == arm
                && g.cell.workload == "synthetic"
                && g.cell.concurrency == 1
        })
        .collect();
    let required: BTreeSet<_> = expected["synthetic"]
        .iter()
        .filter(|(c, _)| *c == 1)
        .map(|(_, o)| *o)
        .collect();
    let c1_keys: BTreeSet<_> = c1.iter().map(|g| Some(g.cell.output_tokens)).collect();
    let c1_parity = !required.is_empty()
        && c1.len() == required.len()
        && c1_keys == required
        && c1.iter().all(|g| g.passed);
    let synthetic_mean_percent = means["synthetic"];
    let thoughtworks_mean_percent = means["thoughtworks"];
    let eligible = complete
        && c1_parity
        && capacity_consistent
        && synthetic_mean_percent.is_some_and(|n| n > 0.0)
        && thoughtworks_mean_percent.is_some_and(|n| n > 0.0);
    Ok(Candidate {
        arm: arm.into(),
        complete,
        c1_parity,
        capacity_consistent,
        synthetic_mean_percent,
        thoughtworks_mean_percent,
        eligible,
        capacity_policy: capacity,
        baseline_capacity_policy: baseline,
    })
}
fn mean_gain(
    rows: &[&Row],
    arm: &str,
    workload: &str,
    expected: &BTreeSet<Key>,
) -> DynResult<(Option<f64>, bool)> {
    let indexed = |name: &str| -> DynResult<BTreeMap<Key, &Row>> {
        let mut index = BTreeMap::new();
        for row in rows
            .iter()
            .copied()
            .filter(|r| r.cell.arm == name && r.cell.workload == workload)
        {
            if index.insert(row.cell.key()?, row).is_some() {
                return Err("duplicate promotion workload key".into());
            }
        }
        Ok(index)
    };
    let candidate = indexed(arm)?;
    let baseline = indexed("mesh")?;
    let mut complete = candidate.keys().copied().collect::<BTreeSet<_>>() == *expected
        && baseline.keys().copied().collect::<BTreeSet<_>>() == *expected;
    let mut sum = 0.0;
    let mut count = 0;
    for (key, row) in candidate {
        let Some(base) = baseline.get(&key) else {
            complete = false;
            continue;
        };
        if !base.complete || !row.complete {
            complete = false;
            continue;
        }
        let delta = 100.0 * (row.throughput / base.throughput - 1.0);
        sum += delta;
        count += 1;
        if !delta.is_finite() || !sum.is_finite() {
            return Err("promotion percentage/mean overflow".into());
        }
    }
    Ok((
        if count == 0 {
            None
        } else {
            Some(sum / f64::from(count))
        },
        complete,
    ))
}
fn policy(rows: &[&Row], arm: &str) -> Option<Capacity> {
    let mut selected = rows.iter().filter(|r| r.cell.arm == arm);
    let first = selected.next()?.capacity_policy.clone();
    selected
        .all(|r| r.capacity_policy == first)
        .then_some(first)
}
pub(super) fn markdown(groups: &[Group]) -> String {
    let mut text = String::from(
        "\n## Promotion gate\n\nCandidates require the complete configured synthetic and Thoughtworks roster, c1 paired exact continuation parity, and positive mean throughput gains over fixed Mesh in both workloads. Policies are retained separately; no cross-platform/model aggregation or automatic promotion occurs.\n\n| Platform | Model | Candidate | Synthetic vs Mesh (%) | Thoughtworks vs Mesh (%) | c1 parity | Complete | Decision |\n| --- | --- | --- | ---: | ---: | --- | --- | --- |\n",
    );
    for group in groups {
        for row in &group.candidates {
            let number = |v: Option<f64>| v.map_or_else(|| "n/a".into(), |v| format!("{v:+.2}"));
            text.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                super::super::report_escape::cell(&group.platform),
                super::super::report_escape::cell(&group.model),
                super::super::report_escape::cell(&row.arm),
                number(row.synthetic_mean_percent),
                number(row.thoughtworks_mean_percent),
                row.c1_parity,
                row.complete,
                if group.winner.as_deref() == Some(row.arm.as_str()) {
                    "PROMOTION CANDIDATE"
                } else {
                    "hold"
                }
            ));
        }
    }
    text
}
#[cfg(test)]
#[path = "competitive_report_promotion_tests.rs"]
mod tests;
