//! Fixed-name publication through a verified App; never consumes supplied verdicts.
mod event;

use super::{
    activation, app_auth, approval,
    collector::{self, Clock, GithubRead, Request},
};
use crate::{
    command::{DynResult, print_json},
    command_interrupt::Interrupt,
    process::Cancellation,
};
use collector::request::{CHECK_NAME, CheckConclusion};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    let number = pr_number(args)?;
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let (anchors, head) = protected_activation(root, &cancellation)?;
    let clock = collector::SystemClock;
    let mut clients = app_auth::publisher_clients(root, &anchors, &cancellation, &clock)?;
    let result = publish(
        &mut clients.observer,
        &mut clients.publisher,
        &anchors,
        &head,
        number,
        &clock,
    )?;
    interrupt.finish()?;
    finish(result)
}

pub(crate) fn collect_pr(root: &Path, args: &[String]) -> DynResult<()> {
    let number = pr_number(args)?;
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let (anchors, head) = protected_activation(root, &cancellation)?;
    let clock = collector::SystemClock;
    let mut observer = app_auth::observer_client(root, &anchors, &cancellation, &clock)?;
    let collected = collector::collect(&mut observer, &head, number, &anchors, &clock)?;
    interrupt.finish()?;
    print_json(
        &serde_json::json!({"schema_version":1,"protected_policy":String::from_utf8(collected.policy_bytes)?,
        "policy_sha256":collected.policy_sha256,"snapshot":collected.snapshot,"native_enforcement_verified":false}),
    )
}

pub(crate) fn resolve_event(root: &Path, args: &[String]) -> DynResult<()> {
    if !args.is_empty() {
        return Err("resolve-publisher-event accepts only the authenticated workflow event".into());
    }
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let (anchors, head) = protected_activation(root, &cancellation)?;
    let trigger = event::Trigger::load(root, &head)?;
    let mut observer =
        app_auth::observer_client(root, &anchors, &cancellation, &collector::SystemClock)?;
    let numbers = trigger.resolve_all(&mut observer, &anchors, &head)?;
    interrupt.finish()?;
    print_json(
        &serde_json::json!({"schema_version":1,"repository_id":anchors.repository_id,"pr_numbers":numbers}),
    )
}

fn pr_number(args: &[String]) -> DynResult<u64> {
    let [number] = args else {
        return Err("governance collector/publisher accepts exactly one positive PR number".into());
    };
    number
        .parse::<u64>()
        .ok()
        .filter(|value| *value > 0 && value.to_string() == *number)
        .ok_or_else(|| {
            "governance collector/publisher accepts only one canonical positive PR number".into()
        })
}

pub(crate) fn publish_event(root: &Path, args: &[String]) -> DynResult<()> {
    let [flag, expected] = args else {
        return Err("publish-event requires --expected-pr from its authenticated resolver".into());
    };
    if flag != "--expected-pr" {
        return Err("publish-event requires only --expected-pr".into());
    }
    let expected = pr_number(std::slice::from_ref(expected))?;
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let (anchors, head) = protected_activation(root, &cancellation)?;
    let trigger = event::Trigger::load(root, &head)?;
    let clock = collector::SystemClock;
    let mut clients = app_auth::publisher_clients(root, &anchors, &cancellation, &clock)?;
    trigger.resolve_expected(&mut clients.observer, &anchors, &head, expected)?;
    let result = publish(
        &mut clients.observer,
        &mut clients.publisher,
        &anchors,
        &head,
        expected,
        &clock,
    )?;
    interrupt.finish()?;
    finish(result)
}

pub(crate) fn activation_status(root: &Path, args: &[String]) -> DynResult<()> {
    if !args.is_empty() {
        return Err("activation-status accepts no inputs".into());
    }
    let cancellation = Cancellation::default();
    let head = collector::checkout_head(root, &cancellation)?;
    let bytes = collector::checkout_activation(root, &head, &cancellation)?;
    let activation = activation::from_bytes(&bytes)?;
    if activation.active {
        activation.anchors()?;
    }
    print_json(
        &serde_json::json!({"schema_version":1,"active":activation.active,
        "source_revision":head,"external_protections_verified":false}),
    )
}

fn protected_activation(
    root: &Path,
    cancellation: &Cancellation,
) -> DynResult<(activation::Anchors, String)> {
    let head = collector::checkout_head(root, cancellation)?;
    let bytes = collector::checkout_activation(root, &head, cancellation)?;
    let anchors = activation::from_bytes(&bytes)?.anchors()?;
    Ok((anchors, head))
}

#[derive(Serialize)]
struct Published {
    schema_version: u32,
    status: &'static str,
    check_run_id: u64,
    app_id: u64,
    installation_id: u64,
    head_sha: String,
    protected_revision: String,
    conclusion: &'static str,
    native_enforcement_verified: bool,
    administration_authorized: bool,
    configured_native_ruleset_ids: Vec<u64>,
}

fn finish(result: Published) -> DynResult<()> {
    let success = result.conclusion == "success";
    print_json(&result)?;
    if success {
        Ok(())
    } else {
        Err("current-revision CI policy evidence was denied; App check published failure".into())
    }
}

fn publish(
    observer: &mut impl GithubRead,
    publisher: &mut impl GithubRead,
    anchors: &activation::Anchors,
    controller_head: &str,
    number: u64,
    clock: &impl Clock,
) -> DynResult<Published> {
    let identity = collector::read_identity(observer, number, anchors)?;
    if identity.main_sha != controller_head {
        return Err("publisher checkout is not exact current protected main".into());
    }
    let external_id = format!(
        "ci-governance:{}:{}:{}:{}",
        anchors.repository_id, number, identity.head_sha, controller_head
    );
    let created: CheckReply = app_auth::response(
        publisher.send(&Request::CreateGovernanceCheck {
            head_sha: identity.head_sha.clone(),
            external_id: external_id.clone(),
        })?,
        201,
    )?;
    let binding = CheckBinding {
        id: created.id,
        head_sha: identity.head_sha,
        external_id,
        app_id: anchors.publisher_app_id,
    };
    binding.verify(&created, None)?;
    let allowed = current_approval(
        observer,
        anchors,
        controller_head,
        number,
        &binding.head_sha,
        clock,
    )
    .unwrap_or(false);
    // A fresh complete collection owns success. A previous denial cannot be
    // promoted by a caller-supplied result or by merely rechecking a check name.
    let allowed = allowed
        && current_approval(
            observer,
            anchors,
            controller_head,
            number,
            &binding.head_sha,
            clock,
        )
        .unwrap_or(false);
    let current = collector::read_identity(observer, number, anchors)?;
    if current.main_sha != controller_head || current.head_sha != binding.head_sha {
        return Err("PR head or protected main changed before App check completion".into());
    }
    let pending: CheckReply = app_auth::response(
        publisher.send(&Request::GetGovernanceCheck { id: binding.id })?,
        200,
    )?;
    binding.verify(&pending, None)?;
    let conclusion = if allowed {
        CheckConclusion::Success
    } else {
        CheckConclusion::Failure
    };
    let completed: CheckReply = app_auth::response(
        publisher.send(&Request::CompleteGovernanceCheck {
            id: binding.id,
            conclusion,
        })?,
        200,
    )?;
    let text = if allowed { "success" } else { "failure" };
    binding.verify(&completed, Some(text))?;
    Ok(Published {
        schema_version: 1,
        status: "PUBLISHED_CHECK_READBACK",
        check_run_id: binding.id,
        app_id: anchors.publisher_app_id,
        installation_id: anchors.publisher_installation_id,
        head_sha: binding.head_sha,
        protected_revision: controller_head.into(),
        conclusion: text,
        native_enforcement_verified: false,
        administration_authorized: false,
        configured_native_ruleset_ids: anchors.native_ruleset_ids.clone(),
    })
}

fn current_approval(
    client: &mut impl GithubRead,
    anchors: &activation::Anchors,
    head: &str,
    number: u64,
    expected_pr_head: &str,
    clock: &impl Clock,
) -> DynResult<bool> {
    let collected = collector::collect(client, head, number, anchors, clock)?;
    if collected.snapshot.head_sha != expected_pr_head {
        return Err("PR revision changed during governance collection".into());
    }
    Ok(approval::evaluate_at(
        &collected.policy,
        &collected.snapshot,
        &collected.policy_sha256,
        clock.now()?,
    )
    .is_ok())
}

#[derive(Deserialize)]
struct CheckApp {
    id: u64,
}
#[derive(Deserialize)]
struct CheckReply {
    id: u64,
    name: String,
    head_sha: String,
    external_id: String,
    app: CheckApp,
    status: String,
    conclusion: Option<String>,
}
struct CheckBinding {
    id: u64,
    head_sha: String,
    external_id: String,
    app_id: u64,
}
impl CheckBinding {
    fn verify(&self, reply: &CheckReply, conclusion: Option<&str>) -> DynResult<()> {
        let expected_status = if conclusion.is_some() {
            "completed"
        } else {
            "in_progress"
        };
        if self.id == 0
            || reply.id != self.id
            || reply.name != CHECK_NAME
            || reply.head_sha != self.head_sha
            || reply.external_id != self.external_id
            || reply.app.id != self.app_id
            || reply.status != expected_status
            || reply.conclusion.as_deref() != conclusion
        {
            return Err("GitHub check readback does not bind the exact App, name, head, operation and conclusion".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
