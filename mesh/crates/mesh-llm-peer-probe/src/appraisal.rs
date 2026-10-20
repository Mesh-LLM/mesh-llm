use crate::{
    model::{ModelClient, Stage},
    protocol::{
        Challenge, Intent, ProbeResponse, ProbeStatus, Reference, ToolRecord, process_binding,
    },
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    PartialConsistent,
    Inconclusive,
    Inconsistent,
    InvalidEvidence,
}

#[derive(Clone, Debug, Serialize)]
pub struct Appraisal {
    pub outcome: Outcome,
    pub assurance: &'static str,
    pub checked_matches: Vec<String>,
    pub missing: Vec<String>,
    pub contradictions: Vec<String>,
    pub judge_note: String,
    pub judge_unsupported_claims: Vec<String>,
    pub judge_model_calls: u32,
}

#[derive(Deserialize)]
struct JudgeOutput {
    note: String,
    #[serde(default)]
    unsupported_claims: Vec<String>,
}

fn relevant_question(challenge: &Challenge, question: &str) -> bool {
    let lower = question.to_ascii_lowercase();
    let has = |terms: &[&str]| terms.iter().any(|term| lower.contains(term));
    (!challenge.intents.contains(&Intent::Process)
        || (has(&["process", "pid"])
            && has(&[
                "identify",
                "show",
                "report",
                "observe",
                "find",
                "provide",
                "tell",
                "what process",
                "which process",
            ])))
        && (!challenge.intents.contains(&Intent::Executable) || has(&["hash", "digest", "sample"]))
        && (!challenge.intents.contains(&Intent::Model)
            || (has(&["model"]) && has(&["metadata", "architecture", "tensor", "active"])))
}

pub fn compose_question(challenge: &Challenge, model: &dyn ModelClient) -> Result<(String, bool)> {
    let mut tasks = Vec::new();
    if challenge.intents.contains(&Intent::Process) {
        tasks.push("Identify the probe component process, its PID and start time, executable handle, and observation source; say unavailable if it cannot be observed.");
    }
    if challenge.intents.contains(&Intent::Executable) {
        tasks.push("Sample the executable specified in the structured challenge and report the nonce-bound hash of its byte range; say if it is no longer the same process or is unavailable.");
    }
    if challenge.intents.contains(&Intent::Model) {
        tasks.push("Report active model metadata: architecture, tensor count, file size, and observation source; say unavailable if this cannot be observed.");
    }
    let output = model.complete(Stage::Writer, &tasks.join(" "));
    if let Ok(text) = &output {
        let question = text.trim();
        if !question.is_empty() && question.len() <= 800 && relevant_question(challenge, question) {
            // This lexical guard is only for useful prose. The structured
            // challenge, not A's model text, defines B's obligations.
            return Ok((question.to_owned(), false));
        }
    }
    let mut question = format!(
        "Challenge {} for peer {} with nonce {}.",
        challenge.id, challenge.peer, challenge.nonce
    );
    if challenge.intents.contains(&Intent::Process) {
        question.push_str(" Identify the probe component process and source.");
    }
    if challenge.intents.contains(&Intent::Executable) {
        question.push_str(&format!(
            " For executable {}, sample offset {} for {} bytes.",
            challenge
                .process_binding
                .as_ref()
                .map_or("the observed handle", |binding| binding
                    .executable_handle
                    .as_str()),
            challenge.offset,
            challenge.length
        ));
    }
    if challenge.intents.contains(&Intent::Model) {
        question.push_str(" Report model metadata or explicit unsupported status.");
    }
    question.push_str(" Return the ordered read-only trace or say unavailable.");
    Ok((question, true))
}

fn find(response: &ProbeResponse, intent: Intent) -> Option<&ToolRecord> {
    response.trace.iter().find(|record| record.tool == intent)
}

/// Appraise either a bundled challenge or two fresh, narrowly scoped rounds.
/// The combined view exists only inside A; B never receives a growing history.
pub fn appraise_rounds(
    rounds: &[(&Challenge, &ProbeResponse)],
    reference: &Reference,
    model: &dyn ModelClient,
    require_wasm: bool,
) -> Appraisal {
    assert!(!rounds.is_empty() && rounds.len() <= 2);
    if rounds.len() == 1 {
        return appraise_inner(rounds[0].0, reference, rounds[0].1, model, require_wasm);
    }
    let (first, first_response) = rounds[0];
    let (second, second_response) = rounds[1];
    let mut combined_challenge = second.clone();
    combined_challenge.intents.insert(0, Intent::Process);
    combined_challenge.process_binding = None;
    combined_challenge.round = 1;
    let mut combined_response = second_response.clone();
    combined_response
        .trace
        .splice(0..0, first_response.trace.clone());
    combined_response.tool_calls += first_response.tool_calls;
    combined_response.model_calls += first_response.model_calls;
    combined_response
        .model_actions
        .splice(0..0, first_response.model_actions.clone());
    combined_response.wasm_executed &= first_response.wasm_executed;
    if combined_response.error.is_none() {
        combined_response.error.clone_from(&first_response.error);
    }
    let mut result = appraise_inner(
        &combined_challenge,
        reference,
        &combined_response,
        model,
        require_wasm,
    );
    let expected_binding = process_binding(first_response);
    let bad_sequence = first.round != 1
        || first.intents != [Intent::Process]
        || second.round != 2
        || second.intents.len() != 1
        || first.id == second.id
        || first.nonce == second.nonce
        || first.peer != second.peer
        || first_response.challenge_id != first.id
        || first_response.nonce != first.nonce
        || first_response.peer != first.peer
        || second_response.challenge_id != second.id
        || second_response.nonce != second.nonce
        || second_response.peer != second.peer
        || second.process_binding != expected_binding
        || matches!(
            first_response.status,
            ProbeStatus::InvalidModelOutput | ProbeStatus::Failed
        );
    if bad_sequence {
        result.outcome = Outcome::InvalidEvidence;
        result
            .contradictions
            .push("round order, identity, nonce, or prior handle binding mismatch".into());
    }
    result
}

pub fn appraise(
    challenge: &Challenge,
    reference: &Reference,
    response: &ProbeResponse,
    model: &dyn ModelClient,
) -> Appraisal {
    appraise_inner(challenge, reference, response, model, true)
}

pub fn appraise_fixed(
    challenge: &Challenge,
    reference: &Reference,
    response: &ProbeResponse,
    model: &dyn ModelClient,
) -> Appraisal {
    appraise_inner(challenge, reference, response, model, false)
}

#[derive(Default)]
struct Checks {
    invalid: bool,
    unavailable: bool,
    checked_matches: Vec<String>,
    missing: Vec<String>,
    contradictions: Vec<String>,
}

impl Checks {
    fn outcome(&self) -> Outcome {
        if self.invalid {
            Outcome::InvalidEvidence
        } else if !self.contradictions.is_empty() {
            Outcome::Inconsistent
        } else if self.unavailable {
            Outcome::Inconclusive
        } else if !self.checked_matches.is_empty() {
            Outcome::PartialConsistent
        } else {
            Outcome::Inconclusive
        }
    }
}

fn check_envelope(
    challenge: &Challenge,
    response: &ProbeResponse,
    require_wasm: bool,
    checks: &mut Checks,
) {
    let binding_bad = response.challenge_id != challenge.id
        || response.peer != challenge.peer
        || response.nonce != challenge.nonce;
    checks.invalid = binding_bad
        || (require_wasm && !response.wasm_executed)
        || response.error.is_some()
        || matches!(
            response.status,
            ProbeStatus::InvalidModelOutput | ProbeStatus::Failed
        );
    checks.unavailable = response.status == ProbeStatus::Unavailable;
    if checks.unavailable {
        checks.missing.push("B reported unavailable".into());
    }
    if binding_bad {
        checks
            .contradictions
            .push("challenge ID, peer, or nonce binding mismatch".into());
    }
    if require_wasm && !response.wasm_executed {
        checks
            .contradictions
            .push("WASM execution was not observed".into());
    }
    if let Some(error) = &response.error {
        checks.contradictions.push(format!("probe error: {error}"));
    }
    if response.trace.len() > 6 {
        checks.invalid = true;
        checks
            .contradictions
            .push("trace exceeds maximum tool calls".into());
    }
    if response.tool_calls != response.trace.len() as u32
        || response.model_calls > 16
        || response.model_actions.len() > 16
        || response
            .trace
            .iter()
            .any(|record| !challenge.intents.contains(&record.tool))
    {
        checks.invalid = true;
        checks
            .contradictions
            .push("trace counters, actions, or tool scope are invalid".into());
    }
    if response
        .trace
        .iter()
        .map(|record| record.tool)
        .collect::<std::collections::HashSet<_>>()
        .len()
        != response.trace.len()
    {
        checks.invalid = true;
        checks
            .contradictions
            .push("duplicate tool observation".into());
    }
}

fn check_obligations(
    challenge: &Challenge,
    reference: &Reference,
    response: &ProbeResponse,
    checks: &mut Checks,
) {
    for intent in &challenge.intents {
        match find(response, *intent) {
            None => checks
                .missing
                .push(format!("{} tool absent", intent.as_str())),
            Some(record) if !record.supported => checks.missing.push(format!(
                "{} unsupported: {}",
                intent.as_str(),
                record.data["reason"].as_str().unwrap_or("no reason")
            )),
            _ => {}
        }
    }
    let process = find(response, Intent::Process).filter(|record| record.supported);
    if let Some(record) = process
        && (record.data["pid"].as_u64().is_none()
            || record.data["start_time"].as_u64().is_none()
            || record.data["executable_handle"].as_str().is_none())
    {
        checks.invalid = true;
        checks
            .contradictions
            .push("malformed process observation".into());
    }
    if challenge.intents.contains(&Intent::Executable)
        && let Some(record) = find(response, Intent::Executable).filter(|record| record.supported)
    {
        check_executable(challenge, reference, record, process, checks);
    }
    if challenge.intents.contains(&Intent::Model)
        && let Some(record) = find(response, Intent::Model).filter(|record| record.supported)
    {
        check_model(reference, record, checks);
    }
}

fn check_executable(
    challenge: &Challenge,
    reference: &Reference,
    record: &ToolRecord,
    process: Option<&ToolRecord>,
    checks: &mut Checks,
) {
    let range_ok = record.data["offset"].as_u64() == Some(challenge.offset)
        && record.data["length"].as_u64() == Some(u64::from(challenge.length));
    if !range_ok {
        checks
            .contradictions
            .push("executable sample range differs from challenge".into());
    }
    let mut linked = process.is_some();
    if let Some(process) = process {
        for field in ["pid", "start_time", "executable_handle"] {
            if record.data[field] != process.data[field] {
                linked = false;
                checks.contradictions.push(format!(
                    "executable {field} differs from process observation"
                ));
            }
        }
    } else {
        checks
            .missing
            .push("executable sample lacks process link".into());
    }
    match &reference.executable_digest {
        Some(expected)
            if record.data["sha256"].as_str() == Some(expected) && linked && range_ok =>
        {
            checks
                .checked_matches
                .push("executable range digest matches A-pinned bytes".into())
        }
        Some(expected) if record.data["sha256"].as_str() != Some(expected) => checks
            .contradictions
            .push("executable range digest differs from A-pinned bytes".into()),
        Some(_) => checks
            .missing
            .push("executable sample lacks valid range or process link".into()),
        None => checks
            .missing
            .push("no A-pinned executable reference".into()),
    }
}

fn check_model(reference: &Reference, record: &ToolRecord, checks: &mut Checks) {
    match &reference.model_metadata {
        Some(expected)
            if expected["architecture"].as_str().is_some()
                && expected["tensor_count"].as_u64().is_some()
                && expected["file_size"].as_u64().is_some()
                && ["architecture", "tensor_count", "file_size"]
                    .iter()
                    .all(|field| record.data[*field] == expected[*field]) =>
        {
            checks
                .checked_matches
                .push("model metadata matches A-pinned reference".into())
        }
        Some(_) => checks
            .contradictions
            .push("model metadata differs from A-pinned reference".into()),
        None => checks
            .missing
            .push("model metadata is B-reported without A reference".into()),
    }
}

fn judge(
    model: &dyn ModelClient,
    challenge: &Challenge,
    response: &ProbeResponse,
    checks: &Checks,
) -> (String, Vec<String>, u32) {
    let judge_input = json!({"challenge":challenge,"answer":response.answer,"trace":response.trace,
        "deterministic_matches":checks.checked_matches,"deterministic_missing":checks.missing,
        "deterministic_contradictions":checks.contradictions})
    .to_string();
    if judge_input.len() > 16 * 1024 {
        return ("judge input exceeded byte budget".into(), vec![], 0);
    }
    match model.complete(Stage::Judge, &judge_input) {
        Ok(text) if text.len() <= 4096 => match serde_json::from_str::<JudgeOutput>(&text) {
            Ok(judge) => (judge.note, judge.unsupported_claims, 1),
            Err(_) => (
                "judge returned malformed JSON; deterministic checks alone were retained".into(),
                vec![],
                1,
            ),
        },
        Ok(_) => ("judge output exceeded byte budget".into(), vec![], 1),
        Err(error) => (format!("judge unavailable: {error}"), vec![], 1),
    }
}

fn appraise_inner(
    challenge: &Challenge,
    reference: &Reference,
    response: &ProbeResponse,
    model: &dyn ModelClient,
    require_wasm: bool,
) -> Appraisal {
    let mut checks = Checks::default();
    check_envelope(challenge, response, require_wasm, &mut checks);
    check_obligations(challenge, reference, response, &mut checks);
    let (judge_note, judge_unsupported_claims, judge_model_calls) =
        judge(model, challenge, response, &checks);
    Appraisal {
        outcome: checks.outcome(),
        assurance: "software_observation_artifact_only",
        checked_matches: checks.checked_matches,
        missing: checks.missing,
        contradictions: checks.contradictions,
        judge_note,
        judge_unsupported_claims,
        judge_model_calls,
    }
}
