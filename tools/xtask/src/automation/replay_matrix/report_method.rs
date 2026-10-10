use super::report_escape::code;
use super::report_input::Document;
use crate::command::DynResult;
use std::io::Write;

pub(super) fn write(output: &mut impl Write, document: &Document) -> DynResult<()> {
    let measured = document
        .config
        .concurrency
        .iter()
        .map(|concurrency| {
            document
                .inputs
                .cohorts
                .get(&concurrency.to_string())
                .ok_or("missing measured cohort")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let trajectories = measured.iter().try_fold(0_u64, |total, cohort| {
        total
            .checked_add(cohort.trajectory_count)
            .ok_or("trajectory count overflow")
    })?;
    let turns = measured.iter().try_fold(0_u64, |total, cohort| {
        total
            .checked_add(cohort.assistant_turns)
            .ok_or("turn count overflow")
    })?;
    let warmup = document
        .inputs
        .cohorts
        .get("warmup")
        .ok_or("missing warmup cohort")?;
    let order = if document.order.is_empty() {
        document
            .results
            .iter()
            .map(|result| result.label.as_str())
            .collect::<Vec<_>>()
    } else {
        document
            .order
            .iter()
            .map(|item| item.label.as_str())
            .collect()
    }
    .join(" \u{2192} ");
    writeln!(
        output,
        "\n## Charts\n\n![Decode throughput](charts/decode-throughput.svg)\n\n![End-to-end workload output throughput](charts/workload-output-throughput.svg)\n\n![Median TTFT](charts/ttft-p50.svg)\n\n## Method\n\n- Model: {}",
        code(&document.config.model)
    )?;
    match &document.config.engine_config {
        None => writeln!(
            output,
            "- Mesh startup: `mesh-llm serve --model <model> --log-format json`.\n- Mesh chooses context size, execution lanes, KV budget, and backend tuning."
        )?,
        Some(_) => writeln!(
            output,
            "- Mesh refs use product-default startup; external engines use the exact commands recorded in `plan.json` and each pass result.\n- Each external engine reports its version before launch; the report records that string and its SHA-256 identity. Model/tokenizer paths are provenance, not file-hash gates."
        )?,
    }
    let concurrency = document
        .config
        .concurrency
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    if document.config.context_qualification.as_deref() == Some("captured") {
        writeln!(
            output,
            "- Context qualification: not requested for this captured profile. No Mesh runtime context or long-context eligibility is certified."
        )?;
    }
    writeln!(
        output,
        "- Client concurrency: {}.\n- Pass order: {}.\n- Warm-up: `{}` discarded turns from a disjoint cohort after every model-ready event.\n- Warm-up cohort: `{}` whole trajectories, disjoint from measured cohorts.\n- Dataset revision: {}.\n- Selected trajectories: `{trajectories}` unique whole sessions across disjoint concurrency cohorts.\n- Recorded source steps: `{turns}` assistant turns are represented across the selected trajectories.\n- Replay mode: {}. Selected requests use complete recorded history at their original assistant-turn positions.",
        code(&concurrency),
        code(&order),
        document.config.warmup_turns,
        warmup.trajectory_count,
        code(
            document
                .inputs
                .dataset
                .as_ref()
                .map_or("not applicable (captured manifest)", |dataset| dataset
                    .revision
                    .as_str())
        ),
        code(&document.config.replay_mode)
    )?;
    for line in [
        "Each selected session runs its measured requests sequentially; different sessions overlap up to the offered client concurrency. Cache namespaces are stable per session.",
        "Realized concurrency is the time-weighted mean number of in-flight requests. Slot use makes cohort tail drain explicit; do not interpret offered-concurrency scaling as steady-state when utilization is low.",
        "Each next request uses the recorded conversation history, so experiment arms receive identical growing prefixes and tool observations.",
        "Per-turn output budgets approximate each recorded assistant action from its character length, capped by the configured maximum; generated output is measured but never fed into the next turn.",
        "Decode tok/s is token-weighted generation throughput after first content. E2E output tok/s includes prompt ingestion and scheduling.",
        "Percent deltas are suppressed unless request counts, failed request identities, and available stable generated-content identities match. Pass ranges expose run-to-run spread.",
    ] {
        writeln!(output, "- {line}")?;
    }
    match document.inputs.kind.as_deref() {
        Some("captured") => writeln!(
            output,
            "- Tool definitions and message fields come from the supplied captured manifest unchanged, so captured runs retain their exact reusable-prefix identity."
        )?,
        Some(_) | None => writeln!(
            output,
            "- Tool definitions preserve recorded names but use permissive synthetic schemas. This benchmark measures serving performance on reconstructed prompts, not answer quality or byte-identical production prompts."
        )?,
    }
    let identities = if document.config.engine_config.is_some() {
        "Raw request records, server logs, exact commands and reported external version SHA-256 identities are retained beside this report; model/tokenizer locations are provenance. Mesh arms retain their binary/runtime hashes."
    } else {
        "Raw request records, server logs, build logs, commands, and exact binary/runtime hashes are retained beside this report."
    };
    writeln!(
        output,
        "- Selection is deterministic within the recorded-length window; runtime-formatted per-session lengths are reported separately.\n- Trajectory manifest SHA-256: {}.\n- {identities}\n\nGates are opt-in. When configured, the run command exits non-zero after preserving the complete artifact if any gate fails.\n",
        code(&document.inputs.manifest_sha256)
    )?;
    Ok(())
}
