//! A bounded, non-authoritative REST observation of reviewer-team membership.
use super::{GithubRead, Request, known, parent_state, unknown};
use crate::ci_governance::policy::REVIEWERS;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;

const MAX_PAGES: u32 = 10;
const PAGE_SIZE: usize = 100;
const MAX_BODY: usize = 2 * 1024 * 1024;

#[derive(Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct Member {
    id: u64,
    login: String,
    #[serde(rename = "type")]
    kind: String,
}

pub(super) fn valid_login(login: &str) -> bool {
    let plain = login.strip_suffix("[bot]").unwrap_or(login);
    !plain.is_empty()
        && plain.len() <= 39
        && plain
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

fn read_members(client: &mut impl GithubRead) -> Result<Vec<Member>, &'static str> {
    let mut members = Vec::new();
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    for page in 1..=MAX_PAGES {
        let response = client
            .send(&Request::TeamMembersPage { page })
            .map_err(|_| "transport_failure")?;
        if response.status != 200 {
            return Err(match response.status {
                401 | 403 => "access_denied",
                404 => "not_found_or_inaccessible",
                _ => "http_error",
            });
        }
        if response.body.len() > MAX_BODY {
            return Err("invalid_or_incomplete_response");
        }
        let page: Vec<Member> =
            serde_json::from_slice(&response.body).map_err(|_| "invalid_or_incomplete_response")?;
        if page.len() > PAGE_SIZE {
            return Err("invalid_or_incomplete_response");
        }
        if page.is_empty() {
            members.sort();
            return Ok(members);
        }
        for mut member in page {
            if member.id == 0
                || !valid_login(&member.login)
                || !matches!(member.kind.as_str(), "User" | "Bot" | "Organization")
            {
                return Err("invalid_or_incomplete_response");
            }
            member.login.make_ascii_lowercase();
            if !ids.insert(member.id) || !names.insert(member.login.clone()) {
                return Err("duplicate_member_identity");
            }
            members.push(member);
        }
    }
    Err("pagination_limit_reached")
}

fn same_team(
    client: &mut impl GithubRead,
    team_id: u64,
    no_parent: bool,
) -> Result<(), &'static str> {
    let response = client
        .send(&Request::Team)
        .map_err(|_| "transport_failure")?;
    if response.status != 200 {
        return Err(match response.status {
            401 | 403 => "access_denied",
            404 => "not_found_or_inaccessible",
            _ => "http_error",
        });
    }
    if response.body.len() > MAX_BODY {
        return Err("invalid_or_incomplete_response");
    }
    let team: Value =
        serde_json::from_slice(&response.body).map_err(|_| "invalid_or_incomplete_response")?;
    let current_no_parent = parent_state(team.get("parent"));
    if team.get("id").and_then(Value::as_u64) != Some(team_id)
        || team.get("slug").and_then(Value::as_str) != Some("ci-policy-reviewers")
        || team.pointer("/organization/login").and_then(Value::as_str) != Some("Mesh-LLM")
        || current_no_parent != Some(no_parent)
    {
        return Err("team_identity_changed_or_unverified");
    }
    Ok(())
}

pub(super) fn inspect(client: &mut impl GithubRead, team: Option<&Value>) -> Value {
    let Some(team) = team.filter(|team| {
        team.get("state").and_then(Value::as_str) == Some("observed")
            && team
                .pointer("/fields/expected_slug/value")
                .and_then(Value::as_bool)
                == Some(true)
            && team
                .pointer("/fields/expected_organization/value")
                .and_then(Value::as_bool)
                == Some(true)
    }) else {
        return unknown("team_metadata_unavailable");
    };
    let (Some(team_id), Some(no_parent)) = (
        team.pointer("/fields/numeric_id/value")
            .and_then(Value::as_u64),
        team.pointer("/fields/no_parent/value")
            .and_then(Value::as_bool),
    ) else {
        return unknown("team_identity_unavailable");
    };
    let result = (|| {
        let first = read_members(client)?;
        let second = read_members(client)?;
        if first != second {
            return Err("membership_changed_during_collection");
        }
        same_team(client, team_id, no_parent)?;
        let actual: BTreeSet<_> = first.iter().map(|member| member.login.as_str()).collect();
        let expected: BTreeSet<_> = REVIEWERS.into_iter().collect();
        let matches = actual == expected && first.iter().all(|member| member.kind == "User");
        Ok(json!({"state":"observed", "fields":{
            "count":known(json!(first.len())),
            "expected_human_login_set":known(json!(matches)),
            "repeat_sample_equal":known(json!(true)),
            "numeric_identity_pinned":unknown("activation_anchors_not_verified")
        }}))
    })();
    result.unwrap_or_else(unknown)
}
