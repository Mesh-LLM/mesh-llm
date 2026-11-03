//! Explicit complete pagination; no truncation or partial-result fallback.
use super::{
    Request,
    transport::GithubRead,
    wire::{self, Actor, Connection, WireReview},
};
use crate::{
    ci_governance::scope::{ChangeKind, ChangedPath},
    command::DynResult,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn read(client: &mut impl GithubRead, request: &Request) -> DynResult<Vec<u8>> {
    let response = client.send(request)?;
    if response.status != 200 {
        return Err(format!(
            "GitHub governance read failed with status {}",
            response.status
        )
        .into());
    }
    if response.body.len() > 2 * 1024 * 1024 {
        return Err("GitHub response exceeds 2 MiB bound".into());
    }
    Ok(response.body)
}

pub(super) fn files(
    client: &mut impl GithubRead,
    number: u64,
    expected: usize,
) -> DynResult<Vec<ChangedPath>> {
    if expected > 3000 {
        return Err("PR file count exceeds GitHub REST's complete 3000-file bound".into());
    }
    let mut files = Vec::new();
    let mut names = BTreeSet::new();
    for page in 1..=30 {
        let entries: Vec<wire::File> =
            wire::json(&read(client, &Request::Files { number, page })?)?;
        if entries.len() > 100 {
            return Err("GitHub files page exceeds 100 entries".into());
        }
        let short = entries.len() < 100;
        for entry in entries {
            if !names.insert(entry.filename.clone()) {
                return Err("duplicate file across GitHub pages".into());
            }
            let kind = match entry.status.as_str() {
                "added" => ChangeKind::Added,
                "modified" => ChangeKind::Modified,
                "removed" => ChangeKind::Deleted,
                "renamed" => ChangeKind::Renamed,
                "copied" => ChangeKind::Copied,
                "changed" => ChangeKind::TypeChanged,
                _ => return Err("unsupported GitHub changed-file status".into()),
            };
            files.push(ChangedPath {
                kind,
                path: entry.filename,
                previous_path: entry.previous_filename,
            });
        }
        if files.len() > expected {
            return Err("GitHub files exceed authoritative PR file count".into());
        }
        if short || files.len() == expected {
            if files.len() != expected {
                return Err("GitHub file pagination is incomplete".into());
            }
            return Ok(files);
        }
    }
    Err("GitHub file pagination did not complete".into())
}

pub(super) fn reviews(
    client: &mut impl GithubRead,
    number: u64,
) -> DynResult<BTreeMap<u64, WireReview>> {
    connection(
        client,
        |cursor| Request::Reviews { number, cursor },
        "/data/repository/pullRequest/reviews",
        |review: &WireReview| review.database_id,
    )
}
pub(super) fn members(
    client: &mut impl GithubRead,
    team_id: u64,
) -> DynResult<BTreeMap<u64, Actor>> {
    let mut read = |request: &Request| -> DynResult<Vec<u8>> {
        let bytes = read(client, request)?;
        let id: u64 = wire::pointer(&bytes, "/data/organization/team/databaseId")?;
        if id != team_id {
            return Err("GraphQL reviewer team numeric identity drift".into());
        }
        Ok(bytes)
    };
    connection_with(
        &mut read,
        |cursor| Request::TeamMembers { cursor },
        "/data/organization/team/members",
        |actor: &Actor| actor.id,
    )
}

fn connection<T: serde::de::DeserializeOwned>(
    client: &mut impl GithubRead,
    request: impl Fn(Option<String>) -> Request,
    path: &str,
    id: impl Fn(&T) -> u64,
) -> DynResult<BTreeMap<u64, T>> {
    connection_with(&mut |request| read(client, request), request, path, id)
}
fn connection_with<T: serde::de::DeserializeOwned>(
    read: &mut impl FnMut(&Request) -> DynResult<Vec<u8>>,
    request: impl Fn(Option<String>) -> Request,
    path: &str,
    id: impl Fn(&T) -> u64,
) -> DynResult<BTreeMap<u64, T>> {
    let mut cursor = None;
    let mut cursors = BTreeSet::new();
    let mut entries = BTreeMap::new();
    let mut expected = None;
    for _ in 0..100 {
        let page: Connection<T> = wire::pointer(&read(&request(cursor.clone()))?, path)?;
        if page.total_count > 10_000
            || expected.is_some_and(|count| count != page.total_count)
            || page.nodes.len() > 100
            || (page.nodes.is_empty() && page.page_info.has_next_page)
        {
            return Err("invalid or changing GitHub pagination total/page size".into());
        }
        expected = Some(page.total_count);
        for entry in page.nodes {
            let id = id(&entry);
            if id == 0 || entries.insert(id, entry).is_some() {
                return Err("duplicate or missing identity across GitHub pages".into());
            }
        }
        if entries.len() > page.total_count {
            return Err("GitHub page count exceeds totalCount".into());
        }
        if !page.page_info.has_next_page {
            if entries.len() != page.total_count {
                return Err("GitHub connection is incomplete".into());
            }
            return Ok(entries);
        }
        let next = page
            .page_info
            .end_cursor
            .ok_or("GitHub pagination cursor missing")?;
        if next.is_empty()
            || next.len() > 1024
            || next.chars().any(char::is_control)
            || !cursors.insert(next.clone())
        {
            return Err("invalid or repeated GitHub pagination cursor".into());
        }
        cursor = Some(next);
    }
    Err("GitHub connection exceeds 100-page bound; partial data rejected".into())
}
