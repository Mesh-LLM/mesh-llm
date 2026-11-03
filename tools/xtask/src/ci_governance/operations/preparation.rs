//! Main-only bounded preparation. No credential release or executable mutations.
use super::environment_readback::{self, AccountWire, ApprovalReadback, EnvironmentReadback};
use super::intent::ReviewedIntent;
use super::record::{Control, ExecutionContext};
use super::record_source::{RecordSelector, RecordSource, ResolvedIntent};
use super::{digest, require};
use crate::ci_governance::{
    activation::Anchors,
    app_auth,
    collector::{GithubRead, Request, SystemClock},
    policy::valid_sha,
};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

const REPOSITORY: &str = "Mesh-LLM/mesh-llm";
const WORKFLOW: &str = ".github/workflows/ci-governance-administration.yml";

pub(super) fn run(root: &Path, args: &[String]) -> DynResult<Value> {
    let selector = selector(args)?;
    let locator = RuntimeLocator::from_lookup(|name| std::env::var(name).ok())?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = (|| {
        let source = RecordSource::new(root, interrupt.cancellation())?;
        let anchors = source.activation_at(&locator.sha)?.anchors()?;
        let mut client =
            app_auth::observer_client(root, &anchors, &interrupt.cancellation(), &SystemClock)?;
        let runtime = collect_runtime(&mut client, &anchors, &locator)?;
        let resolved = source.resolve(&selector, &runtime.protected_main_revision)?;
        prepare(
            &mut client,
            &anchors,
            &locator,
            runtime,
            resolved,
            super::now()?,
        )
    })();
    interrupt.finish()?;
    result
}

fn selector(args: &[String]) -> DynResult<RecordSelector> {
    require(
        args.len() == 4,
        "prepare-administration requires only --record-id UUID --record-sha256 DIGEST",
    )?;
    let mut id = None;
    let mut sha256 = None;
    for pair in args.as_chunks::<2>().0 {
        match pair[0].as_str() {
            "--record-id" if id.is_none() => id = Some(pair[1].clone()),
            "--record-sha256" if sha256.is_none() => sha256 = Some(pair[1].clone()),
            _ => return Err("unsupported/duplicate administration preparation option".into()),
        }
    }
    let selector = RecordSelector {
        id: id.ok_or("record UUID missing")?,
        raw_sha256: sha256.ok_or("record digest missing")?,
    };
    selector.validate()?;
    Ok(selector)
}

struct RuntimeLocator {
    run_id: u64,
    attempt: u64,
    sha: String,
    actor_id: u64,
    triggering_actor: String,
}

impl RuntimeLocator {
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> DynResult<Self> {
        let mut get = |name: &str| {
            lookup(name)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| format!("required protected runtime field missing: {name}"))
        };
        require(
            get("GITHUB_REPOSITORY")? == REPOSITORY
                && get("GITHUB_REF")? == "refs/heads/main"
                && get("GITHUB_EVENT_NAME")? == "workflow_dispatch"
                && get("GITHUB_WORKFLOW_REF")?
                    == format!("{REPOSITORY}/{WORKFLOW}@refs/heads/main"),
            "administration preparation requires the fixed repository main-only manual workflow",
        )?;
        let sha = get("GITHUB_SHA")?;
        require(
            valid_sha(&sha) && get("GITHUB_WORKFLOW_SHA")? == sha,
            "runtime source/workflow must be the same exact protected main revision",
        )?;
        let locator = Self {
            run_id: get("GITHUB_RUN_ID")?.parse()?,
            attempt: get("GITHUB_RUN_ATTEMPT")?.parse()?,
            sha,
            actor_id: get("GITHUB_ACTOR_ID")?.parse()?,
            triggering_actor: get("GITHUB_TRIGGERING_ACTOR")?,
        };
        require(
            locator.run_id > 0 && locator.attempt > 0 && locator.actor_id > 0,
            "runtime run/attempt/actor identities must be positive",
        )?;
        Ok(locator)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct RuntimeMetadata {
    repository_id: u64,
    protected_main_revision: String,
    execution: ExecutionContext,
}

#[derive(Deserialize)]
struct RepositoryWire {
    id: u64,
    full_name: String,
    default_branch: String,
}
#[derive(Deserialize)]
struct MainWire {
    name: String,
    protected: bool,
    commit: CommitWire,
}
#[derive(Deserialize)]
struct CommitWire {
    sha: String,
}
#[derive(Deserialize)]
struct RunWire {
    id: u64,
    run_attempt: u64,
    head_branch: String,
    head_sha: String,
    event: String,
    path: String,
    status: String,
    repository: RunRepository,
    actor: AccountWire,
    triggering_actor: AccountWire,
}

#[derive(Deserialize)]
struct RunRepository {
    id: u64,
    full_name: String,
}

fn collect_runtime(
    client: &mut impl GithubRead,
    anchors: &Anchors,
    locator: &RuntimeLocator,
) -> DynResult<RuntimeMetadata> {
    let repository: RepositoryWire = environment_readback::get(client, Request::Repository)?;
    require(
        repository.id == anchors.repository_id
            && repository.full_name == REPOSITORY
            && repository.default_branch == "main",
        "authenticated repository identity/default branch does not match activation anchors",
    )?;
    let main: MainWire = environment_readback::get(client, Request::CurrentMain)?;
    require(
        main.name == "main"
            && main.protected
            && valid_sha(&main.commit.sha)
            && main.commit.sha == locator.sha,
        "workflow must use the exact current protected main, not a historical ancestor",
    )?;
    let run: RunWire = environment_readback::get(
        client,
        Request::WorkflowRun {
            run_id: locator.run_id,
        },
    )?;
    require(
        run.id == locator.run_id
            && run.run_attempt == locator.attempt
            && run.head_branch == "main"
            && run.head_sha == main.commit.sha
            && run.event == "workflow_dispatch"
            && workflow_path(&run.path)
            && run.status == "in_progress"
            && run.repository.id == repository.id
            && run.repository.full_name == REPOSITORY,
        "authenticated workflow run does not match the fixed main-only runtime/attempt",
    )?;
    let actor = run.actor.human()?;
    let triggerer = run.triggering_actor.human()?;
    require(
        actor.id == locator.actor_id && triggerer.login == locator.triggering_actor,
        "authenticated workflow actors do not match current runtime",
    )?;
    let execution = ExecutionContext {
        repository: REPOSITORY.into(),
        run_id: run.id,
        run_attempt: run.run_attempt,
        ref_name: "refs/heads/main".into(),
        protected_revision: main.commit.sha.clone(),
        operator: triggerer.clone(),
        triggerer: actor,
        rerun_actor: triggerer,
    };
    Ok(RuntimeMetadata {
        repository_id: repository.id,
        protected_main_revision: main.commit.sha,
        execution,
    })
}

fn workflow_path(value: &str) -> bool {
    value == WORKFLOW
        || ["main", "refs/heads/main"].into_iter().any(|reference| {
            value == format!("{WORKFLOW}@{reference}")
                || value == format!("{REPOSITORY}/{WORKFLOW}@{reference}")
        })
}

#[derive(Serialize)]
struct PreparedPacket {
    schema_version: u32,
    operation_id: String,
    record_bytes_sha256: String,
    intent_sha256: String,
    policy_sha256: String,
    declared_before_state_sha256: String,
    desired_state_sha256: String,
    runtime: RuntimeMetadata,
    prepared_at_unix: u64,
    expires_at_unix: u64,
    intent: ReviewedIntent,
    environment_configuration: EnvironmentReadback,
    approval_history: ApprovalReadback,
    manual_instructions: Vec<ManualInstruction>,
}

#[derive(Serialize)]
struct ManualInstruction {
    control: Control,
    instruction: &'static str,
    reviewed_desired_value: Value,
}

fn prepare(
    client: &mut impl GithubRead,
    anchors: &Anchors,
    locator: &RuntimeLocator,
    runtime: RuntimeMetadata,
    resolved: ResolvedIntent,
    now: u64,
) -> DynResult<Value> {
    resolved
        .intent
        .validate(&resolved.policy, &resolved.expected, now)?;
    require(
        resolved.protected_main_revision == runtime.protected_main_revision,
        "resolved record source does not match authenticated current main",
    )?;
    let environment_configuration =
        environment_readback::collect_environment(client, anchors, &resolved.policy)?;
    let approval_history =
        environment_readback::collect_approval_history(client, anchors, locator.run_id)?;
    let repeated = collect_runtime(client, anchors, locator)?;
    require(
        repeated == runtime,
        "main/run/attempt/actors changed during administration preparation",
    )?;
    let packet = PreparedPacket {
        schema_version: 1,
        operation_id: resolved.intent.operation_id.clone(),
        record_bytes_sha256: resolved.record_sha256,
        intent_sha256: resolved.intent.canonical_digest()?,
        policy_sha256: digest(&resolved.policy)?,
        declared_before_state_sha256: resolved.intent.expected_before_sha256.clone(),
        desired_state_sha256: resolved.expected.state.canonical_digest()?,
        runtime,
        prepared_at_unix: now,
        expires_at_unix: resolved
            .intent
            .expires_at_unix
            .min(now.checked_add(3600).ok_or("preparation expiry overflow")?),
        manual_instructions: instructions(&resolved.intent),
        intent: resolved.intent,
        environment_configuration,
        approval_history,
    };
    Ok(
        json!({ "schema_version":1,"status":"PREPARED_PENDING_AUTHORIZATION", "packet_sha256":digest(&packet)?, "packet":packet,
        "authorization":{"status":"blocked","exact_attempt_and_packet_approval_verified":false,
            "live_state_precondition_verified":false,"replay_consumption_verified":false,
            "runtime_attested":false,
            "reason":"Supported fixed API metadata/configuration were read, but exact-attempt packet approval, post-approval full live-state/replay precondition and protected runtime attestation are not established."},
        "execution_authorized":false,"credential_release_authorized":false }),
    )
}

fn instructions(intent: &ReviewedIntent) -> Vec<ManualInstruction> {
    intent.controls.iter().map(|control| {
        let (instruction, value) = match control {
            Control::ReviewerTeam => ("After separate verified independent authorization, the human operator reconciles only the reviewed reviewer-team membership, visibility and repository permission; retain actual readback.", json!(&intent.desired_after.reviewer_team)),
            Control::CiRuleset => ("After separate verified independent authorization, the human operator reconciles only the reviewed native CI ruleset; retain effective rule and bypass readback.", json!(&intent.desired_after.native_rule)),
            Control::AdministrationEnvironment => ("After separate verified independent authorization, the human operator reconciles only the reviewed administration environment; retain actual protection and main-branch policy readback.", json!(&intent.desired_after.administration_environment)),
            Control::CredentialBoundaries => ("After separate verified independent authorization, the human credential custodian reconciles only the reviewed credential boundaries; retain names, scope and custody evidence without secret values.", json!(&intent.desired_after.credentials)),
        };
        ManualInstruction { control: *control, instruction, reviewed_desired_value: value }
    }).collect()
}

#[cfg(test)]
#[path = "preparation_tests.rs"]
mod tests;
