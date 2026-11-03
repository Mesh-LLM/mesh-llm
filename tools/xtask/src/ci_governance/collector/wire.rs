//! Minimal provider wire types; unsupported account/state/time shapes fail closed.
use crate::{
    ci_governance::approval::{Account, AccountKind, Review, ReviewState},
    command::DynResult,
};
use serde::{Deserialize, Deserializer, de::DeserializeOwned};

pub(super) fn json<T: DeserializeOwned>(bytes: &[u8]) -> DynResult<T> {
    serde_json::from_slice(bytes).map_err(|error| {
        format!(
            "invalid GitHub response schema at line {}, column {}",
            error.line(),
            error.column()
        )
        .into()
    })
}

#[derive(Deserialize)]
pub(super) struct Repository {
    pub id: u64,
    pub full_name: String,
    pub default_branch: String,
}
#[derive(Deserialize)]
pub(super) struct MainBranch {
    pub name: String,
    pub protected: bool,
    pub commit: Commit,
}
#[derive(Clone, Deserialize, PartialEq, Eq)]
pub(super) struct Commit {
    pub sha: String,
}
#[derive(Deserialize)]
pub(super) struct PullRequest {
    pub number: u64,
    pub state: String,
    pub user: RestActor,
    pub base: BaseRef,
    pub head: HeadRef,
    pub changed_files: usize,
}
#[derive(Deserialize)]
pub(super) struct BaseRef {
    #[serde(rename = "ref")]
    pub name: String,
    pub sha: String,
    pub repo: RepositoryRef,
}
#[derive(Deserialize)]
pub(super) struct HeadRef {
    pub sha: String,
}
#[derive(Deserialize)]
pub(super) struct RepositoryRef {
    pub id: u64,
    pub full_name: String,
}
#[derive(Deserialize)]
pub(super) struct RestActor {
    pub id: u64,
    pub login: String,
    #[serde(rename = "type")]
    pub kind: String,
}
impl RestActor {
    pub fn account(self) -> DynResult<Account> {
        account(self.id, self.login, &self.kind)
    }
}
fn account(id: u64, login: String, kind: &str) -> DynResult<Account> {
    let kind = match kind {
        "User" => AccountKind::Human,
        "Bot" => AccountKind::Bot,
        "Organization" => AccountKind::Organization,
        _ => return Err("unsupported GitHub account kind".into()),
    };
    if id == 0
        || login.is_empty()
        || login.len() > 100
        || login
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace() || ch == '/')
    {
        return Err("invalid GitHub account identity".into());
    }
    Ok(Account { id, login, kind })
}

#[derive(Deserialize)]
pub(super) struct File {
    pub filename: String,
    pub status: String,
    pub previous_filename: Option<String>,
}
#[derive(Deserialize)]
pub(super) struct TeamMetadata {
    pub id: u64,
    pub slug: String,
    pub privacy: String,
    pub organization: Organization,
    pub parent: serde_json::Value,
}
#[derive(Deserialize)]
pub(super) struct Organization {
    pub login: String,
}
#[derive(Deserialize)]
pub(super) struct TeamAccess {
    pub id: u64,
    pub full_name: String,
    pub permissions: Permissions,
    pub role_name: Option<String>,
}
#[derive(Deserialize)]
pub(super) struct Permissions {
    pub push: bool,
    pub pull: bool,
    pub admin: bool,
    pub maintain: Option<bool>,
}

#[derive(Clone, Deserialize, PartialEq, Eq)]
pub(super) struct Actor {
    #[serde(rename = "__typename")]
    pub kind: String,
    pub login: String,
    #[serde(rename = "databaseId")]
    pub id: u64,
}
impl Actor {
    pub fn account(self) -> DynResult<Account> {
        account(self.id, self.login, &self.kind)
    }
}
#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(super) struct WireReview {
    #[serde(rename = "fullDatabaseId", deserialize_with = "full_database_id")]
    pub database_id: u64,
    pub state: String,
    pub submitted_at: Option<String>,
    pub commit: Option<ReviewCommit>,
    pub author: Option<Actor>,
}

fn full_database_id<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
    let value = String::deserialize(deserializer)?;
    if value.is_empty() || value.starts_with('0') || !value.bytes().all(|ch| ch.is_ascii_digit()) {
        return Err(serde::de::Error::custom("invalid GitHub fullDatabaseId"));
    }
    value
        .parse::<u64>()
        .map_err(|_| serde::de::Error::custom("GitHub fullDatabaseId exceeds u64"))
}
#[derive(Clone, Deserialize, PartialEq, Eq)]
pub(super) struct ReviewCommit {
    pub oid: String,
}
impl WireReview {
    pub fn review(self) -> DynResult<Review> {
        let state = match self.state.as_str() {
            "APPROVED" => ReviewState::Approved,
            "CHANGES_REQUESTED" => ReviewState::ChangesRequested,
            "DISMISSED" => ReviewState::Dismissed,
            "COMMENTED" => ReviewState::Commented,
            "PENDING" => ReviewState::Pending,
            _ => return Err("unsupported GitHub review state".into()),
        };
        let submitted_at = self.submitted_at.as_deref().map(timestamp).transpose()?;
        let reviewer = self
            .author
            .ok_or("GitHub review author missing")?
            .account()?;
        if self.database_id == 0 {
            return Err("GitHub review ID missing".into());
        }
        Ok(Review {
            id: self.database_id,
            reviewer,
            state,
            submitted_at,
            commit_sha: self.commit.map(|commit| commit.oid),
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Connection<T> {
    pub total_count: usize,
    pub nodes: Vec<T>,
    pub page_info: PageInfo,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PageInfo {
    pub has_next_page: bool,
    pub end_cursor: Option<String>,
}

pub(super) fn pointer<T: DeserializeOwned>(bytes: &[u8], path: &str) -> DynResult<T> {
    let value: serde_json::Value = json(bytes)?;
    if value.get("errors").is_some() {
        return Err("GitHub GraphQL response contains errors; partial data rejected".into());
    }
    let selected = value
        .pointer(path)
        .ok_or("GitHub GraphQL response data missing")?;
    serde_json::from_value(selected.clone())
        .map_err(|_| "invalid GitHub GraphQL response schema".into())
}

/// GitHub's UTC second-resolution DateTime, with calendar validation.
pub(crate) fn timestamp(text: &str) -> DynResult<u64> {
    let template = b"0000-00-00T00:00:00Z";
    if text.len() != 20
        || !text.bytes().zip(template).all(|(byte, expected)| {
            if *expected == b'0' {
                byte.is_ascii_digit()
            } else {
                byte == *expected
            }
        })
    {
        return Err("unsupported GitHub UTC timestamp".into());
    }
    let field = |start, end| {
        text[start..end]
            .parse::<u64>()
            .map_err(|_| "invalid GitHub timestamp field")
    };
    let (year, month, day) = (field(0, 4)?, field(5, 7)?, field(8, 10)?);
    let (hour, minute, second) = (field(11, 13)?, field(14, 16)?, field(17, 19)?);
    if !(1970..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || day == 0
        || day > month_days(year, month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err("invalid GitHub calendar timestamp".into());
    }
    let prior = year - 1;
    let days = 365 * prior + prior / 4 - prior / 100 + prior / 400
        - (365 * 1969 + 1969 / 4 - 1969 / 100 + 1969 / 400)
        + (1..month).map(|month| month_days(year, month)).sum::<u64>()
        + day
        - 1;
    Ok(days * 86400 + hour * 3600 + minute * 60 + second)
}
fn month_days(year: u64, month: u64) -> u64 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) => {
            29
        }
        2 => 28,
        _ => 31,
    }
}
