//! Pure evaluation of supplied evidence; a trusted collector must authenticate it.
use super::policy::{Policy, valid_sha};
use super::scope::{ChangedPath, policy_changes};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum AccountKind {
    Human,
    Bot,
    Organization,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Account {
    pub login: String,
    pub id: u64,
    pub kind: AccountKind,
}

impl Account {
    fn validate(&self) -> DynResult<()> {
        if self.id == 0
            || self.login.is_empty()
            || self
                .login
                .chars()
                .any(|ch| ch.is_whitespace() || ch.is_control() || ch == '/')
        {
            return Err("missing or malformed account identity".into());
        }
        Ok(())
    }

    fn same_identity(&self, other: &Self) -> bool {
        self.id == other.id || self.login.eq_ignore_ascii_case(&other.login)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum EvidenceSource {
    GithubApi,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Collection {
    pub source: EvidenceSource,
    pub observed_at: u64,
    pub files_complete: bool,
    pub reviews_complete: bool,
    pub team_complete: bool,
    pub truncated: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum PolicyAuthority {
    ProtectedBase,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PrSnapshot {
    pub schema_version: u32,
    pub collection: Collection,
    pub repository: String,
    pub base_ref: String,
    pub base_sha: String,
    pub current_protected_sha: String,
    pub head_sha: String,
    pub current_head_sha: String,
    pub policy_authority: PolicyAuthority,
    pub policy_revision: String,
    pub policy_sha256: String,
    pub pr_number: u64,
    pub author: Account,
    pub sponsor: Account,
    pub changed_file_count: usize,
    pub changes: Vec<ChangedPath>,
    pub reviews: Vec<Review>,
    pub team: Option<Team>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ReviewState {
    Approved,
    ChangesRequested,
    Dismissed,
    Commented,
    Pending,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Review {
    pub id: u64,
    pub reviewer: Account,
    pub state: ReviewState,
    pub commit_sha: Option<String>,
    pub submitted_at: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Team {
    pub slug: String,
    pub visible: bool,
    pub repository: String,
    pub repository_permission: String,
    pub members: Vec<Account>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Decision {
    pub schema_version: u32,
    pub policy_change: bool,
    pub controlled_paths: Vec<String>,
    pub independent_approvers: Vec<String>,
    pub repository: String,
    pub head_sha: String,
    pub policy_revision: String,
    pub evidence_status: &'static str,
}

pub(crate) fn evaluate_at(
    policy: &Policy,
    snapshot: &PrSnapshot,
    trusted_policy_sha256: &str,
    now: u64,
) -> DynResult<Decision> {
    policy.validate()?;
    validate_identity(policy, snapshot, trusted_policy_sha256)?;
    if now < snapshot.collection.observed_at || now - snapshot.collection.observed_at > 300 {
        return Err(
            "PR snapshot is stale or from the future; collect again within 300 seconds".into(),
        );
    }
    let controlled_paths = policy_changes(policy, &snapshot.changes)?;
    let policy_change = !controlled_paths.is_empty();
    let independent_approvers = if policy_change {
        validate_submission(snapshot)?;
        validate_team(policy, snapshot)?;
        eligible_approvers(policy, snapshot)?
    } else {
        Vec::new()
    };
    if policy_change && independent_approvers.len() < policy.required_approvals as usize {
        return Err(
            "CI policy requires one independent eligible human approval of current head".into(),
        );
    }
    Ok(Decision {
        schema_version: 1,
        policy_change,
        controlled_paths,
        independent_approvers,
        repository: snapshot.repository.clone(),
        head_sha: snapshot.head_sha.clone(),
        policy_revision: snapshot.policy_revision.clone(),
        evidence_status: "local-snapshot-validation-only",
    })
}

fn validate_identity(policy: &Policy, snapshot: &PrSnapshot, digest: &str) -> DynResult<()> {
    if snapshot.schema_version != 1
        || snapshot.repository != policy.repository
        || snapshot.base_ref != policy.protected_branch
        || snapshot.pr_number == 0
        || snapshot.collection.observed_at == 0
        || !snapshot.collection.files_complete
        || snapshot.collection.truncated
        || snapshot.changed_file_count != snapshot.changes.len()
    {
        return Err("foreign, incomplete, or truncated PR snapshot".into());
    }
    if !valid_sha(&snapshot.base_sha)
        || !valid_sha(&snapshot.head_sha)
        || snapshot.head_sha != snapshot.current_head_sha
        || snapshot.policy_revision != snapshot.base_sha
        || snapshot.current_protected_sha != snapshot.base_sha
        || digest.len() != 64
        || !digest
            .bytes()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
        || snapshot.policy_sha256 != digest
    {
        return Err("snapshot does not bind protected-base policy and exact current head".into());
    }
    snapshot.author.validate()?;
    snapshot.sponsor.validate()
}

fn validate_submission(snapshot: &PrSnapshot) -> DynResult<()> {
    if snapshot.author.kind != AccountKind::Human
        || snapshot.sponsor.kind != AccountKind::Human
        || snapshot.author != snapshot.sponsor
    {
        return Err(
            "supported CI-policy submission requires human author as authenticated sponsor".into(),
        );
    }
    if !snapshot.collection.reviews_complete || !snapshot.collection.team_complete {
        return Err("CI-policy review/team collection is incomplete".into());
    }
    Ok(())
}

fn validate_team(policy: &Policy, snapshot: &PrSnapshot) -> DynResult<()> {
    let team = snapshot
        .team
        .as_ref()
        .ok_or("missing reviewer team readback")?;
    if team.slug != policy.reviewer_team
        || !team.visible
        || team.repository != policy.repository
        || team.repository_permission != "write"
        || team.members.len() != policy.eligible_humans.len()
    {
        return Err(
            "reviewer team identity, visibility, explicit write access or membership drift".into(),
        );
    }
    let mut ids = BTreeSet::new();
    let mut logins = BTreeSet::new();
    for member in &team.members {
        member.validate()?;
        if member.kind != AccountKind::Human
            || !ids.insert(member.id)
            || !logins.insert(&member.login)
        {
            return Err("reviewer team must contain unique human identities".into());
        }
    }
    if logins != policy.eligible_humans.iter().collect() {
        return Err("reviewer team membership differs from eligible humans".into());
    }
    Ok(())
}

fn eligible_approvers(policy: &Policy, snapshot: &PrSnapshot) -> DynResult<Vec<String>> {
    if snapshot.reviews.len() > 10_000 {
        return Err("review snapshot exceeds supported 10000-review bound".into());
    }
    let mut latest = BTreeMap::<u64, EffectiveReview<'_>>::new();
    let mut review_ids = BTreeSet::new();
    let mut accounts_by_id = BTreeMap::new();
    let mut accounts_by_login = BTreeMap::new();
    for review in &snapshot.reviews {
        validate_review(review, snapshot, &mut review_ids)?;
        record_account(
            &review.reviewer,
            &mut accounts_by_id,
            &mut accounts_by_login,
        )?;
        if matches!(review.state, ReviewState::Commented | ReviewState::Pending) {
            continue;
        }
        latest
            .entry(review.reviewer.id)
            .or_insert(EffectiveReview {
                review,
                ambiguous: false,
            })
            .observe(review);
    }
    let team = snapshot.team.as_ref().ok_or("missing reviewer team")?;
    let approvers: Vec<_> = latest
        .values()
        .filter(|effective| {
            !effective.ambiguous && is_eligible(policy, snapshot, team, effective.review)
        })
        .map(|effective| effective.review.reviewer.login.clone())
        .collect();
    if approvers.is_empty() && latest.values().any(|effective| effective.ambiguous) {
        return Err(
            "ambiguous latest review submission order; collect an unambiguous fresh review".into(),
        );
    }
    Ok(approvers)
}

struct EffectiveReview<'a> {
    review: &'a Review,
    ambiguous: bool,
}

impl<'a> EffectiveReview<'a> {
    fn observe(&mut self, review: &'a Review) {
        match review.submitted_at.cmp(&self.review.submitted_at) {
            std::cmp::Ordering::Greater => {
                self.review = review;
                self.ambiguous = false;
            }
            std::cmp::Ordering::Equal => {
                self.ambiguous |= review.state != self.review.state
                    || review.commit_sha != self.review.commit_sha;
            }
            std::cmp::Ordering::Less => {}
        }
    }
}

fn record_account<'a>(
    account: &'a Account,
    by_id: &mut BTreeMap<u64, &'a Account>,
    by_login: &mut BTreeMap<String, &'a Account>,
) -> DynResult<()> {
    if by_id
        .insert(account.id, account)
        .is_some_and(|prior| prior != account)
        || by_login
            .insert(account.login.to_ascii_lowercase(), account)
            .is_some_and(|prior| prior != account)
    {
        return Err("inconsistent review account identity".into());
    }
    Ok(())
}

fn validate_review(
    review: &Review,
    snapshot: &PrSnapshot,
    ids: &mut BTreeSet<u64>,
) -> DynResult<()> {
    review.reviewer.validate()?;
    if review.id == 0 || !ids.insert(review.id) {
        return Err("duplicate, missing or malformed review metadata".into());
    }
    // GitHub draft reviews have no submitted time or commit yet. They cannot count.
    if review.state == ReviewState::Pending {
        return Ok(());
    }
    let sha = review
        .commit_sha
        .as_deref()
        .ok_or("missing submitted review commit SHA")?;
    let submitted_at = review
        .submitted_at
        .ok_or("missing submitted review timestamp")?;
    if !valid_sha(sha) || submitted_at == 0 || submitted_at > snapshot.collection.observed_at {
        return Err("missing or malformed submitted review metadata".into());
    }
    Ok(())
}

fn is_eligible(policy: &Policy, snapshot: &PrSnapshot, team: &Team, review: &Review) -> bool {
    review.state == ReviewState::Approved
        && review.commit_sha.as_deref() == Some(snapshot.head_sha.as_str())
        && review.reviewer.kind == AccountKind::Human
        && policy.eligible_humans.contains(&review.reviewer.login)
        && !review.reviewer.same_identity(&snapshot.author)
        && !review.reviewer.same_identity(&snapshot.sponsor)
        && team.members.contains(&review.reviewer)
}
