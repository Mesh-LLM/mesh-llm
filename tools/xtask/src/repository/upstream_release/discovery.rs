//! Compare publication timestamps across the complete release metadata feed.
use super::{Release, github::Request};
use crate::command::DynResult;
use serde::Deserialize;
use std::collections::BTreeSet;

pub(super) const QUERY: &str = r#"query($cursor:String) {
  repository(owner:"ggml-org",name:"llama.cpp") {
    releases(first:100,after:$cursor,orderBy:{field:CREATED_AT,direction:DESC}) {
      nodes { id:databaseId tag_name:tagName draft:isDraft prerelease:isPrerelease published_at:publishedAt }
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
        for release in page.nodes {
            if release.id == 0 || !identities.insert(release.id) {
                return Err("invalid or repeated upstream release identity".into());
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
        // Every page is read before selecting: the feed is ordered by creation time, so a
        // draft created long ago and published recently sits on a late page, and stopping
        // early on a publication horizon would hide it on every scheduled run rather than
        // only until the next one. The 180-second lookup budget and the page ceiling are
        // the bounds on a feed that only grows.
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
