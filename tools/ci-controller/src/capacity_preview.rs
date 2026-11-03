//! A deterministic, non-authoritative preview of shared runner demand.
//!
//! No workflow calls this module and it cannot reserve capacity, cancel work,
//! or authenticate its inputs. A live admission service would additionally
//! need an approved, complete quota and role catalog; authenticated, fresh
//! runner and queue observations; and atomic shared state across all callers.
//! Preview candidates must never be treated as admission authorization.
use std::collections::{BTreeMap, BTreeSet};

pub mod journal;
pub mod reservations;

/// A caller-supplied runner class, not an attested machine identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunnerClass {
    pub provider: String,
    pub os: String,
    pub architecture: String,
    pub label: String,
    pub group: Option<String>,
}

/// All supplied pools charged by one worker in this role.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Role {
    pub runner: RunnerClass,
    pub pools: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkKind {
    PullRequest {
        repository_id: u64,
        number: u64,
    },
    Main {
        repository_id: u64,
        revision: String,
    },
    Release {
        repository_id: u64,
    },
    Warmer {
        repository_id: u64,
    },
    Control {
        repository_id: u64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Work {
    pub id: String,
    pub kind: WorkKind,
    pub role: String,
    pub workers: u64,
}

/// Work remains active while cancellation is requested but not confirmed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Active {
    pub work: Work,
    pub observed_class: Option<RunnerClass>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pending {
    pub work: Work,
    /// An explicitly identified older active or earlier queued PR revision.
    pub supersedes: Option<String>,
}

/// All values, including freshness and limits, are supplied by the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub observed_at: u64,
    pub max_age: u64,
    pub pools: BTreeMap<String, u64>,
    pub roles: BTreeMap<String, Role>,
    pub active: Vec<Active>,
    /// Oldest first. Only explicitly superseded pending PR work is omitted.
    pub pending: Vec<Pending>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preview {
    /// Hypothetical starts only; no reservation or execution is authorized.
    pub candidates: Vec<String>,
    pub deferred: Vec<String>,
    pub superseded_pending: Vec<String>,
    /// Hypothetical cancellations; the active work still consumes capacity.
    pub active_cancellation_candidates: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewError {
    StaleSnapshot,
    InvalidCatalog,
    InvalidWork,
    UnknownRole,
    DuplicateWork,
    MissingRunnerClass,
    RunnerClassMismatch,
    InvalidSupersession,
    ArithmeticOverflow,
    AlreadyOverCapacity,
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:/".contains(&byte))
}

fn runner_class(class: &RunnerClass) -> bool {
    identifier(&class.provider)
        && identifier(&class.os)
        && identifier(&class.architecture)
        && identifier(&class.label)
        && class.group.as_deref().is_none_or(identifier)
}

fn kind_valid(kind: &WorkKind) -> bool {
    match kind {
        WorkKind::PullRequest {
            repository_id,
            number,
        } => *repository_id > 0 && *number > 0,
        WorkKind::Main {
            repository_id,
            revision,
        } => {
            *repository_id > 0
                && revision.len() == 40
                && revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }
        WorkKind::Release { repository_id }
        | WorkKind::Warmer { repository_id }
        | WorkKind::Control { repository_id } => *repository_id > 0,
    }
}

fn same_pr(a: &WorkKind, b: &WorkKind) -> bool {
    matches!(
        (a, b),
        (
            WorkKind::PullRequest { repository_id: ar, number: an },
            WorkKind::PullRequest { repository_id: br, number: bn }
        ) if ar == br && an == bn
    )
}

fn work_role<'a>(work: &Work, snapshot: &'a Snapshot) -> Result<&'a Role, PreviewError> {
    if !identifier(&work.id)
        || !identifier(&work.role)
        || work.workers == 0
        || !kind_valid(&work.kind)
    {
        return Err(PreviewError::InvalidWork);
    }
    snapshot
        .roles
        .get(&work.role)
        .ok_or(PreviewError::UnknownRole)
}

/// Evaluate supplied state in FIFO order without performing any admission.
///
/// A full head-of-line request blocks later work in this conservative preview.
/// Explicit PR supersession may remove an older pending revision from the
/// candidate queue, but a running revision counts until its completion is
/// independently observed. Main, release, warmer and control work is never
/// superseded here. The caller-supplied freshness window is not itself an
/// approved policy or evidence that the snapshot is authentic or complete.
pub fn preview(snapshot: &Snapshot, now: u64) -> Result<Preview, PreviewError> {
    if now < snapshot.observed_at || now - snapshot.observed_at > snapshot.max_age {
        return Err(PreviewError::StaleSnapshot);
    }
    if snapshot.pools.is_empty()
        || snapshot.roles.is_empty()
        || snapshot.pools.keys().any(|id| !identifier(id))
    {
        return Err(PreviewError::InvalidCatalog);
    }
    for (id, role) in &snapshot.roles {
        if !identifier(id)
            || !runner_class(&role.runner)
            || role.pools.is_empty()
            || role
                .pools
                .iter()
                .any(|pool| !snapshot.pools.contains_key(pool))
        {
            return Err(PreviewError::InvalidCatalog);
        }
    }

    let mut used = snapshot
        .pools
        .keys()
        .map(|pool| (pool.as_str(), 0_u64))
        .collect::<BTreeMap<_, _>>();
    let mut known = BTreeMap::<&str, (&WorkKind, bool)>::new();
    for active in &snapshot.active {
        let role = work_role(&active.work, snapshot)?;
        if known
            .insert(&active.work.id, (&active.work.kind, true))
            .is_some()
        {
            return Err(PreviewError::DuplicateWork);
        }
        match &active.observed_class {
            None => return Err(PreviewError::MissingRunnerClass),
            Some(class) if class != &role.runner => return Err(PreviewError::RunnerClassMismatch),
            Some(_) => {}
        }
        for pool in &role.pools {
            let total = used[pool.as_str()]
                .checked_add(active.work.workers)
                .ok_or(PreviewError::ArithmeticOverflow)?;
            if total > snapshot.pools[pool] {
                return Err(PreviewError::AlreadyOverCapacity);
            }
            used.insert(pool, total);
        }
    }

    let mut targeted = BTreeSet::new();
    let mut superseded = BTreeSet::new();
    let mut cancellations = Vec::new();
    for pending in &snapshot.pending {
        work_role(&pending.work, snapshot)?;
        if known.contains_key(pending.work.id.as_str()) {
            return Err(PreviewError::DuplicateWork);
        }
        if let Some(target) = &pending.supersedes {
            let Some((previous, active)) = known.get(target.as_str()) else {
                return Err(PreviewError::InvalidSupersession);
            };
            if !same_pr(&pending.work.kind, previous) || !targeted.insert(target.as_str()) {
                return Err(PreviewError::InvalidSupersession);
            }
            if *active {
                cancellations.push(target.clone());
            } else {
                superseded.insert(target.as_str());
            }
        }
        known.insert(&pending.work.id, (&pending.work.kind, false));
    }

    let mut result = Preview {
        candidates: Vec::new(),
        deferred: Vec::new(),
        superseded_pending: Vec::new(),
        active_cancellation_candidates: cancellations,
    };
    let mut blocked = false;
    for pending in &snapshot.pending {
        let work = &pending.work;
        if superseded.contains(work.id.as_str()) {
            result.superseded_pending.push(work.id.clone());
            continue;
        }
        let role = work_role(work, snapshot)?;
        let mut next = Vec::with_capacity(role.pools.len());
        for pool in &role.pools {
            let value = used[pool.as_str()]
                .checked_add(work.workers)
                .ok_or(PreviewError::ArithmeticOverflow)?;
            if value > snapshot.pools[pool] {
                blocked = true;
            }
            next.push((pool.as_str(), value));
        }
        if blocked {
            result.deferred.push(work.id.clone());
        } else {
            for (pool, value) in next {
                used.insert(pool, value);
            }
            result.candidates.push(work.id.clone());
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "capacity_preview/tests.rs"]
mod tests;
