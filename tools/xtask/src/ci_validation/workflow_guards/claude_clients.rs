//! Required pinned Claude client execution and explicitly authorized manual live gate.
use super::{Node, field, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;

const CLIENT: &str = "@anthropic-ai/claude-code@2.1.273";
const AFFECTED: &str = "${{ contains(steps.changes.outputs.affected_crates, 'openai-frontend') || contains(steps.changes.outputs.affected_crates, 'mesh-llm-host-runtime') }}";
const PR_TEST: &str = "claude_cli_executes_read_tool_through_host_ingress";
const LIVE_TEST: &str = "claude_cli_round_trips_through_host_ingress_and_live_claude_model";

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    required(
        workflows
            .get("pr_linux.yml")
            .ok_or("missing Claude PR workflow")?,
    )?;
    manual(
        workflows
            .get("claude-live-model-gate.yml")
            .ok_or("missing Claude live workflow")?,
    )
}

fn not_optional(step: &Node) -> DynResult<()> {
    if field(step, "continue-on-error").is_some_and(|value| value != "false") {
        return Err("Claude execution must fail its owning job on failure".into());
    }
    Ok(())
}

fn pinned_client(steps: &[Node]) -> DynResult<(usize, &Node)> {
    let (index, install) = h::step(steps, "name", "Install pinned Claude Code client")?;
    not_optional(install)?;
    h::command(install, &["npm", "install", "--global", CLIENT], &[])?;
    Ok((index, install))
}

// Bind this small source-owned execution shape. Accept comments, indentation,
// and line continuations, but never certify echoed, conditional or ignored tests.
fn execution(step: &Node, feature: &str, filter: &str, protocol: bool) -> DynResult<()> {
    not_optional(step)?;
    let source = field(step, "run").ok_or("missing Claude execution scalar")?;
    let normalized = source
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("\\\n", " ");
    let observed = normalized
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| invocation(&line.split_whitespace().collect::<Vec<_>>()))
        .collect::<Result<Vec<_>, _>>()?;
    let client = observed
        .iter()
        .filter(|call| {
            call.package.as_deref() == Some("mesh-llm-host-runtime")
                && call.no_default
                && call.features.iter().any(|value| value == feature)
                && call.filter.as_deref() == Some(filter)
        })
        .count();
    let protocols = observed
        .iter()
        .filter(|call| call.package.as_deref() == Some("openai-frontend") && call.filter.is_none())
        .count();
    if client != 1
        || protocols != usize::from(protocol)
        || observed.len() != 1 + usize::from(protocol)
    {
        return Err(
            "Claude gate must directly execute its required Cargo package, feature and test filter"
                .into(),
        );
    }
    Ok(())
}

fn required(workflow: &Node) -> DynResult<()> {
    let job = h::job(workflow, "plan")?;
    h::binding(job, "timeout-minutes", "45")?;
    not_optional(job)?;
    let steps = h::steps(job)?;
    let (install_index, install) = pinned_client(steps)?;
    h::condition(install, AFFECTED)?;
    let (index, gate) = h::step(
        steps,
        "name",
        "Run Claude Code protocol and real-client integration",
    )?;
    h::before(install_index, index)?;
    h::condition(gate, AFFECTED)?;
    let env = h::member(gate, "env")?;
    for key in ["CARGO_INCREMENTAL", "CARGO_PROFILE_TEST_DEBUG"] {
        h::binding(env, key, "0")?;
    }
    execution(gate, "claude-code-integration", PR_TEST, true)
}

fn manual(workflow: &Node) -> DynResult<()> {
    let events = h::member(workflow, "on")?;
    if events.entries().len() != 1 || events.get("workflow_dispatch").is_none() {
        return Err("live Claude credential gate must remain manual-only".into());
    }
    let job = h::job(workflow, "live_claude")?;
    reject_unconditional_skip(job)?;
    h::binding(job, "environment", "claude-live-model")?;
    not_optional(job)?;
    let env = h::member(job, "env")?;
    h::binding(env, "ANTHROPIC_API_KEY", "${{ secrets.ANTHROPIC_API_KEY }}")?;
    h::binding(env, "CARGO_PROFILE_TEST_DEBUG", "0")?;
    let steps = h::steps(job)?;
    let (install_index, install) = pinned_client(steps)?;
    if install.get("if").is_some() {
        return Err("manual Claude gate requires its pinned client install".into());
    }
    let (accelerator_index, accelerator) =
        h::step(steps, "name", "Install repository build accelerator")?;
    if !field(accelerator, "uses")
        .is_some_and(|value| value.starts_with("mozilla-actions/sccache-action@"))
    {
        return Err("manual Claude compiler gate requires its repository build accelerator".into());
    }
    not_optional(accelerator)?;
    if accelerator.get("if").is_some() {
        return Err("manual Claude compiler accelerator must precede execution".into());
    }
    let (gate_index, gate) = h::step(
        steps,
        "name",
        "Run live Claude model round trip through Mesh",
    )?;
    h::before(install_index, gate_index)?;
    h::before(accelerator_index, gate_index)?;
    if gate.get("if").is_some() {
        return Err("authorized manual Claude test must not be silently skipped".into());
    }
    execution(gate, "claude-live-model-integration", LIVE_TEST, false)
}

// Literal unconditional skips cannot qualify a required manual gate. Other
// authorization expressions belong to the workflow's protected environment.
fn reject_unconditional_skip(job: &Node) -> DynResult<()> {
    let Some(condition) = field(job, "if") else {
        return Ok(());
    };
    let compact = condition.split_whitespace().collect::<String>();
    let literal = compact.trim_start_matches("${{").trim_end_matches("}}");
    if literal.eq_ignore_ascii_case("false") {
        return Err("manual Claude job cannot be permanently disabled".into());
    }
    Ok(())
}

#[derive(Default)]
struct Invocation {
    package: Option<String>,
    features: Vec<String>,
    filter: Option<String>,
    no_default: bool,
}

fn invocation(words: &[&str]) -> DynResult<Invocation> {
    let words = words.strip_prefix(&["just", "with-lld"]).unwrap_or(words);
    let words = words
        .strip_prefix(&["cargo", "test"])
        .ok_or("Claude gate requires literal Cargo test execution")?;
    let mut result = Invocation::default();
    let mut locked = false;
    let mut index = 0;
    while let Some(word) = words.get(index) {
        match *word {
            "--locked" => locked = true,
            "--no-default-features" => result.no_default = true,
            "--quiet" | "-q" | "--offline" | "--release" => {}
            "-p" | "--package" | "--features" => {
                let value = words
                    .get(index + 1)
                    .ok_or("missing Claude Cargo option value")?;
                if *word == "--features" {
                    result.features.extend(value.split(',').map(str::to_owned));
                } else if result.package.replace((*value).into()).is_some() {
                    return Err("Claude Cargo gate must select one package".into());
                }
                index += 1;
            }
            "--" => {
                if !words[index + 1..]
                    .iter()
                    .all(|value| *value == "--nocapture")
                {
                    return Err("Claude test harness cannot replace execution with ignored-only or list mode".into());
                }
                break;
            }
            value if value.starts_with('-') => {
                return Err("unsupported Claude Cargo execution option".into());
            }
            value => {
                if result.filter.replace(value.into()).is_some() {
                    return Err("Claude Cargo gate contains extra command or filter tokens".into());
                }
            }
        }
        index += 1;
    }
    if !locked {
        return Err("Claude Cargo invocation must retain --locked".into());
    }
    Ok(result)
}

#[cfg(test)]
#[path = "claude_clients_tests.rs"]
mod tests;
