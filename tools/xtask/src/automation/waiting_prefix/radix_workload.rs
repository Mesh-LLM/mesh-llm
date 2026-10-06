//! Original cold/warm exact, divergent and growing coding prefix cohorts.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub(super) enum Scenario {
    Exact,
    Divergent,
    Coding,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Shape {
    pub rounds: u32,
    pub requests: u32,
    pub levels: Vec<u32>,
    pub prefix_blocks: u32,
    pub output_tokens: u32,
    pub lanes: u32,
    pub n_gpu_layers: i32,
}
impl Shape {
    pub fn validate(&self) -> DynResult<()> {
        if !(1..=16).contains(&self.rounds)
            || !(1..=128).contains(&self.requests)
            || self.levels.is_empty()
            || self.levels.len() > 8
            || self.levels.iter().any(|n| *n == 0 || *n > self.lanes)
            || self.levels.windows(2).any(|n| n[0] >= n[1])
            || !(1..=16).contains(&self.lanes)
            || !(1..=2048).contains(&self.prefix_blocks)
            || !(1..=4096).contains(&self.output_tokens)
            || self.n_gpu_layers < -1
        {
            return Err("radix bounded levels/rounds/profile invalid".into());
        }
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Batch {
    pub scenario: Scenario,
    pub concurrency: u32,
    pub warmup: bool,
    pub prompts: Vec<String>,
}
fn prefix(blocks: u32) -> String {
    let mut rows=vec!["You are a deterministic coding assistant. Preserve the repository rules, tool schema, file inventory, and conversation facts below.".into()];
    for i in 0..blocks {
        rows.push(format!("context-block-{i:04}: src/module_{}.rs owns invariant {i}; edits require tests and exact output parity.",i%37));
    }
    rows.join("\n")
}
pub(super) fn pair(scenario: Scenario, blocks: u32) -> (String, String) {
    let stable = prefix(blocks);
    match scenario {
        Scenario::Exact => {
            let prompt = format!(
                "Exact-prefix workload.\n{stable}\nTask: identify the owner of invariant 17."
            );
            (prompt.clone(), prompt)
        }
        Scenario::Divergent => (
            format!(
                "Divergent-prefix workload.\n{stable}\nTool result alpha: inspect module_17 and return its invariant."
            ),
            format!(
                "Divergent-prefix workload.\n{stable}\nTool result beta: inspect module_18 and return its invariant."
            ),
        ),
        Scenario::Coding => {
            let prompt = format!(
                "Coding-agent tool loop.\n{stable}\nTurn 0 user: inspect the repository root.\nTurn 0 tool: Cargo.toml and crates/ were found."
            );
            (prompt.clone(), prompt)
        }
    }
}
pub(super) fn divergent(base: &str, n: u32, id: u32) -> String {
    let markers = [
        "amber", "birch", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo",
        "juniper", "kelp", "lilac", "marble", "nectar", "onyx", "pearl",
    ];
    format!(
        "{base}\nUnique branch {}-{n}-{id}: return only the requested invariant.",
        markers[id as usize % markers.len()]
    )
}
pub(super) fn coding(base: &str, n: u32, id: u32) -> String {
    let mut parts = vec![base.to_owned()];
    for turn in 0..=id {
        parts.push(format!(
            "Turn {} user: inspect module_{}.rs for invariant {turn}.",
            turn + 1,
            turn % 37
        ));
        parts.push(format!(
            "Turn {} tool: module_{}.rs preserves invariant {turn}; trace lane {n}.",
            turn + 1,
            turn % 37
        ));
    }
    parts.push("Assistant: return the latest invariant only.".into());
    parts.join("\n")
}
pub(super) fn batches(shape: &Shape, warm: bool) -> DynResult<Vec<Batch>> {
    shape.validate()?;
    let mut result = vec![];
    let mut total_bytes = 0_usize;
    for scenario in [Scenario::Exact, Scenario::Divergent, Scenario::Coding] {
        let (first, base) = pair(scenario, shape.prefix_blocks);
        for &n in &shape.levels {
            if warm {
                result.push(Batch {
                    scenario,
                    concurrency: 1,
                    warmup: true,
                    prompts: vec![first.clone()],
                });
            }
            let prompts: Vec<String> = (0..shape.requests.max(n))
                .map(|id| match scenario {
                    Scenario::Exact => base.clone(),
                    Scenario::Divergent => divergent(&base, n, id),
                    Scenario::Coding => coding(&base, n, id),
                })
                .collect();
            for prompt in &prompts {
                total_bytes = total_bytes
                    .checked_add(prompt.len())
                    .ok_or("radix scaffold byte budget overflow")?;
                if total_bytes > 32 * 1024 * 1024 {
                    return Err("radix complete prompt scaffold exceeds 32MiB".into());
                }
            }
            result.push(Batch {
                scenario,
                concurrency: n,
                warmup: false,
                prompts,
            });
        }
    }
    Ok(result)
}
