//! Collect frozen protected policy and complete, repeatedly checked PR evidence.
use super::{Request, pagination, transport::GithubRead, wire};
use crate::{
    ci_governance::{
        activation::Anchors,
        approval::{Account, Collection, EvidenceSource, PolicyAuthority, PrSnapshot, Team},
        policy::{Policy, valid_sha},
        scope::policy_changes,
    },
    command::DynResult,
};
use sha2::{Digest, Sha256};

pub(crate) trait Clock {
    fn now(&self) -> DynResult<u64>;
}
pub(crate) struct SystemClock;
impl Clock for SystemClock {
    fn now(&self) -> DynResult<u64> {
        Ok(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrIdentity {
    pub repository_id: u64,
    pub pr_number: u64,
    pub main_sha: String,
    pub base_sha: String,
    pub head_sha: String,
    pub author: Account,
    pub changed_file_count: usize,
}

pub(crate) struct CollectedPr {
    pub policy: Policy,
    pub policy_bytes: Vec<u8>,
    pub policy_sha256: String,
    pub snapshot: PrSnapshot,
}

pub(crate) fn read_identity(
    client: &mut impl GithubRead,
    pr: u64,
    anchors: &Anchors,
) -> DynResult<PrIdentity> {
    if pr == 0 {
        return Err("PR number must be positive".into());
    }
    let repository: wire::Repository =
        wire::json(&pagination::read(client, &Request::Repository)?)?;
    if repository.id != anchors.repository_id
        || repository.full_name != super::request::REPOSITORY
        || repository.default_branch != "main"
    {
        return Err("GitHub repository numeric identity or protected default branch drift".into());
    }
    let main: wire::MainBranch = wire::json(&pagination::read(client, &Request::CurrentMain)?)?;
    if main.name != "main" || !main.protected || !valid_sha(&main.commit.sha) {
        return Err("GitHub main branch is not protected or has invalid identity".into());
    }
    let pull: wire::PullRequest =
        wire::json(&pagination::read(client, &Request::Pull { number: pr })?)?;
    if pull.number != pr
        || pull.state != "open"
        || pull.base.name != "main"
        || pull.base.repo.id != anchors.repository_id
        || pull.base.repo.full_name != super::request::REPOSITORY
        || pull.base.sha != main.commit.sha
        || !valid_sha(&pull.head.sha)
    {
        return Err(
            "GitHub PR is not open against this repository's current protected main".into(),
        );
    }
    Ok(PrIdentity {
        repository_id: repository.id,
        pr_number: pr,
        main_sha: main.commit.sha,
        base_sha: pull.base.sha,
        head_sha: pull.head.sha,
        author: pull.user.account()?,
        changed_file_count: pull.changed_files,
    })
}

/// Returns evidence before approval evaluation so callers can record a denied check.
pub(crate) fn collect(
    client: &mut impl GithubRead,
    controller_head: &str,
    pr: u64,
    anchors: &Anchors,
    clock: &impl Clock,
) -> DynResult<CollectedPr> {
    let started = clock.now()?;
    let initial = read_identity(client, pr, anchors)?;
    if !valid_sha(controller_head) || controller_head != initial.main_sha {
        return Err("controller checkout HEAD must equal freshly observed protected main".into());
    }
    let policy_bytes = pagination::read(
        client,
        &Request::Policy {
            revision: initial.main_sha.clone(),
        },
    )?;
    if policy_bytes.len() > 64 * 1024 {
        return Err("protected governance policy exceeds 64 KiB".into());
    }
    let policy = Policy::from_bytes(&policy_bytes)?;
    let policy_sha256 = hex::encode(Sha256::digest(&policy_bytes));
    let changes = pagination::files(client, pr, initial.changed_file_count)?;
    let controlled = !policy_changes(&policy, &changes)?.is_empty();
    let (team, reviews) = if controlled {
        policy_evidence(client, pr, anchors)?
    } else {
        (None, Vec::new())
    };
    let current = read_identity(client, pr, anchors)?;
    if current != initial {
        return Err(
            "main, PR head/base/author or changed-file count changed during collection".into(),
        );
    }
    let observed_at = clock.now()?;
    if observed_at < started || observed_at - started > 180 {
        return Err("governance collection clock or 180-second budget invalid".into());
    }
    let snapshot = PrSnapshot {
        schema_version: 1,
        collection: Collection {
            source: EvidenceSource::GithubApi,
            observed_at,
            files_complete: true,
            reviews_complete: controlled,
            team_complete: controlled,
            truncated: false,
        },
        repository: policy.repository.clone(),
        base_ref: policy.protected_branch.clone(),
        base_sha: initial.base_sha,
        current_protected_sha: current.main_sha.clone(),
        head_sha: initial.head_sha,
        current_head_sha: current.head_sha,
        policy_authority: PolicyAuthority::ProtectedBase,
        policy_revision: current.main_sha,
        policy_sha256: policy_sha256.clone(),
        pr_number: pr,
        sponsor: initial.author.clone(),
        author: initial.author,
        changed_file_count: initial.changed_file_count,
        changes,
        reviews,
        team,
    };
    Ok(CollectedPr {
        policy,
        policy_bytes,
        policy_sha256,
        snapshot,
    })
}

fn policy_evidence(
    client: &mut impl GithubRead,
    pr: u64,
    anchors: &Anchors,
) -> DynResult<(Option<Team>, Vec<crate::ci_governance::approval::Review>)> {
    let initial_team = team(client, anchors)?;
    let initial_reviews = pagination::reviews(client, pr)?;
    let current_reviews = pagination::reviews(client, pr)?;
    let current_team = team(client, anchors)?;
    if initial_team != current_team || initial_reviews != current_reviews {
        return Err("reviewer membership/access or PR reviews changed during collection".into());
    }
    let reviews = initial_reviews
        .into_values()
        .map(wire::WireReview::review)
        .collect::<DynResult<Vec<_>>>()?;
    for review in &reviews {
        if anchors
            .human_ids
            .get(&review.reviewer.login)
            .is_some_and(|id| *id != review.reviewer.id)
            || (anchors.human_ids.contains_key(&review.reviewer.login)
                && review.reviewer.kind != crate::ci_governance::approval::AccountKind::Human)
        {
            return Err("eligible reviewer numeric human identity drift".into());
        }
    }
    Ok((Some(initial_team.normalized()), reviews))
}

#[derive(PartialEq, Eq)]
struct TeamReadback {
    id: u64,
    visible: bool,
    permission: String,
    members: Vec<Account>,
}
impl TeamReadback {
    fn normalized(self) -> Team {
        Team {
            slug: "Mesh-LLM/ci-policy-reviewers".into(),
            visible: self.visible,
            repository: super::request::REPOSITORY.into(),
            repository_permission: self.permission,
            members: self.members,
        }
    }
}
fn team(client: &mut impl GithubRead, anchors: &Anchors) -> DynResult<TeamReadback> {
    let metadata: wire::TeamMetadata = wire::json(&pagination::read(client, &Request::Team)?)?;
    if !metadata.parent.is_null()
        || metadata.id != anchors.reviewer_team_id
        || metadata.slug != "ci-policy-reviewers"
        || metadata.organization.login != "Mesh-LLM"
    {
        return Err("reviewer team numeric identity drift".into());
    }
    let visible = match metadata.privacy.as_str() {
        "closed" => true,
        "secret" => false,
        _ => return Err("unsupported reviewer team visibility".into()),
    };
    let members = pagination::members(client, anchors.reviewer_team_id)?
        .into_values()
        .map(wire::Actor::account)
        .collect::<DynResult<Vec<_>>>()?;
    if members.len() != anchors.human_ids.len()
        || members.iter().any(|member| {
            anchors.human_ids.get(&member.login) != Some(&member.id)
                || member.kind != crate::ci_governance::approval::AccountKind::Human
        })
    {
        return Err("reviewer team human numeric identity or exact membership drift".into());
    }
    let access: wire::TeamAccess =
        wire::json(&pagination::read(client, &Request::TeamPermission)?)?;
    if access.id != anchors.repository_id || access.full_name != super::request::REPOSITORY {
        return Err("team repository access numeric identity drift".into());
    }
    if access.role_name.as_deref() != Some("write")
        || access.permissions.admin
        || access.permissions.maintain == Some(true)
        || !access.permissions.push
        || !access.permissions.pull
    {
        return Err(
            "team access must affirm exact write role with no admin or maintain privilege".into(),
        );
    }
    Ok(TeamReadback {
        id: metadata.id,
        visible,
        permission: "write".into(),
        members,
    })
}
