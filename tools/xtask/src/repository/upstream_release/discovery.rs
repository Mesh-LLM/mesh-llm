//! Compare publication timestamps across the complete release metadata feed.
use super::{Release, github::Request};
use crate::command::DynResult;
use serde::Deserialize;
use std::collections::BTreeSet;

pub(super) const QUERY: &str = r#"query($cursor:String) {
  repository(owner:"ggml-org",name:"llama.cpp") {
    releases(first:100,after:$cursor,orderBy:{field:CREATED_AT,direction:DESC}) {
      nodes { id:databaseId tag_name:tagName draft:isDraft prerelease:isPrerelease created_at:createdAt published_at:publishedAt }
      pageInfo { hasNextPage endCursor }
    }
  }
}"#;

#[derive(Deserialize)]
struct Page {
    nodes: Vec<Release>,
    #[serde(rename = "pageInfo")]
    page_info: PageInfo,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PageInfo {
    has_next_page: bool,
    end_cursor: Option<String>,
}

pub(super) fn newest(
    read: &mut impl FnMut(&Request) -> DynResult<serde_json::Value>,
) -> DynResult<Release> {
    let mut cursor = None;
    let mut cursors = BTreeSet::new();
    let mut identities = BTreeSet::new();
    let mut newest: Option<Release> = None;
    for _ in 0..1000 {
        let response = read(&Request::ReleasePage(cursor.clone()))?;
        if response.get("errors").is_some() {
            return Err("upstream release query failed; no partial selection admitted".into());
        }
        let page: Page = serde_json::from_value(
            response
                .pointer("/data/repository/releases")
                .ok_or("upstream release page missing")?
                .clone(),
        )?;
        if page.nodes.len() > 100 || (page.nodes.is_empty() && page.page_info.has_next_page) {
            return Err("invalid upstream release page size".into());
        }
        let mut oldest_created_at: Option<String> = None;
        for release in page.nodes {
            if release.id == 0 || !identities.insert(release.id) {
                return Err("invalid or repeated upstream release identity".into());
            }
            let created = release
                .created_at
                .as_deref()
                .ok_or("upstream release has no creation time")?;
            validate_timestamp(created)?;
            if oldest_created_at
                .as_deref()
                .is_none_or(|oldest| created < oldest)
            {
                oldest_created_at = Some(created.to_owned());
            }
            if release.draft {
                continue;
            }
            let published = release
                .published_at
                .as_deref()
                .ok_or("non-draft upstream release has no publication time")?;
            validate_timestamp(published)?;
            if newest.as_ref().is_none_or(|prior| {
                (release.published_at.as_deref(), release.id)
                    > (prior.published_at.as_deref(), prior.id)
            }) {
                newest = Some(release);
            }
        }
        if !page.page_info.has_next_page {
            return newest.ok_or_else(|| "no published upstream releases found".into());
        }
        // The feed is ordered by creation time, so every release on a later page was
        // created no later than this page's oldest creation time. Once the newest
        // publication already found predates that point, no later page can hold a newer
        // publication, and the scan stops instead of walking a feed that only grows. A
        // draft created earlier but published later is the one release such a scan
        // cannot see; the scheduled run re-checks the feed, so the selection stays a
        // published release rather than a partial result.
        if reaches_publication_horizon(newest.as_ref(), oldest_created_at.as_deref()) {
            return newest.ok_or_else(|| "no published upstream releases found".into());
        }
        let next = page
            .page_info
            .end_cursor
            .ok_or("upstream release cursor missing")?;
        if next.is_empty() || next.len() > 1024 || !cursors.insert(next.clone()) {
            return Err("invalid or repeated upstream release cursor".into());
        }
        cursor = Some(next);
    }
    Err("upstream release pagination exceeds 1000-page limit; no partial selection admitted".into())
}

/// True when a later page cannot hold a newer publication. The feed is ordered by creation
/// time, and a release becomes visible on its creation page, so anything on a later page was
/// created at or before `oldest_created_at`. A directly published release is published at its
/// creation time, which is already older than the newest publication found. The one release
/// this cannot rule out is a draft created earlier and published later; the caller re-checks
/// the feed on its next scheduled run instead of widening every scan.
fn reaches_publication_horizon(newest: Option<&Release>, oldest_created_at: Option<&str>) -> bool {
    newest
        .and_then(|release| release.published_at.as_deref())
        .zip(oldest_created_at)
        .is_some_and(|(published, oldest)| published > oldest)
}

fn validate_timestamp(value: &str) -> DynResult<()> {
    let template = b"0000-00-00T00:00:00Z";
    if value.len() != template.len()
        || !value.bytes().zip(template).all(|(byte, expected)| {
            if *expected == b'0' {
                byte.is_ascii_digit()
            } else {
                byte == *expected
            }
        })
    {
        return Err("upstream publication time must be a UTC timestamp".into());
    }
    let field = |start, end| value[start..end].parse::<u32>();
    let (year, month, day) = (field(0, 4)?, field(5, 7)?, field(8, 10)?);
    let days = match month {
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    };
    if year == 0
        || day == 0
        || day > days
        || field(11, 13)? > 23
        || field(14, 16)? > 59
        || field(17, 19)? > 59
    {
        return Err("upstream publication time is invalid".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
