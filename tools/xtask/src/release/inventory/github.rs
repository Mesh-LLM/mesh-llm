//! Raw GitHub release/PR evidence, explicitly distinct from release classification.
use super::provenance::{Error, Frozen, Git, Output, Result};
use serde_json::{Map, Value};
const RELEASE_FIELDS: &str =
    "tagName,name,publishedAt,url,isPrerelease,isDraft,body,targetCommitish,assets";
const PR_FIELDS: &str =
    "number,title,url,body,mergedAt,mergeCommit,labels,author,baseRefName,headRefName";
const QUERY_LIMIT: usize = 1000;
pub(crate) trait Gh {
    fn request(&mut self, args: &[&str]) -> Result<Output>;
}
pub(crate) struct Release {
    pub(crate) tag: String,
    pub(crate) published_at: String,
    pub(crate) raw: Map<String, Value>,
}
pub(crate) struct PullRequests {
    pub(crate) rows: Vec<Map<String, Value>>,
    pub(crate) query_limit: usize,
    pub(crate) may_be_truncated: bool,
    pub(crate) query_scope: &'static str,
}
fn argument(value: &str) -> Result<()> {
    if value.is_empty()
        || value.starts_with('-')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        return Err(Error("GitHub evidence argument must be nonempty without whitespace, controls or leading option marker".into()));
    }
    Ok(())
}
fn json(gh: &mut impl Gh, args: &[&str]) -> Result<Value> {
    let out = gh.request(args)?;
    if out.code != 0 {
        return Err(Error("GitHub evidence request failed".into()));
    }
    serde_json::from_slice(&out.stdout).map_err(|_| Error("GitHub evidence JSON invalid".into()))
}
pub(crate) fn release(
    gh: &mut impl Gh,
    repository: &str,
    override_tag: Option<&str>,
) -> Result<Release> {
    argument(repository)?;
    let mut args = vec!["release", "view"];
    if let Some(tag) = override_tag {
        argument(tag)?;
        args.push(tag);
    }
    args.extend(["--repo", repository, "--json", RELEASE_FIELDS]);
    let raw = json(gh, &args)?
        .as_object()
        .cloned()
        .ok_or_else(|| Error("GitHub release evidence must be an object".into()))?;
    let tag = raw
        .get("tagName")
        .and_then(Value::as_str)
        .ok_or_else(|| Error("GitHub release tag absent".into()))?
        .to_owned();
    argument(&tag)?;
    let published_at = raw
        .get("publishedAt")
        .and_then(Value::as_str)
        .ok_or_else(|| Error("GitHub release publication time absent".into()))?
        .to_owned();
    // Only a timestamp is admitted into the fixed merged-after-publication search, not arbitrary search clauses.
    if published_at.len() < 20
        || !published_at
            .bytes()
            .all(|b| b.is_ascii_digit() || b"-:+.TZtz".contains(&b))
        || !published_at.contains(['T', 't'])
    {
        return Err(Error("GitHub release publication timestamp invalid".into()));
    }
    for field in ["isPrerelease", "isDraft"] {
        if !raw.get(field).is_some_and(Value::is_boolean) {
            return Err(Error("GitHub release status metadata invalid".into()));
        }
    }
    if !raw.get("assets").is_some_and(Value::is_array) {
        return Err(Error("GitHub release asset evidence invalid".into()));
    }
    Ok(Release {
        tag,
        published_at,
        raw,
    })
}
enum Membership {
    Known(bool),
    Unknown(&'static str),
}
fn ancestor(git: &mut impl Git, base: &str, head: &str) -> Result<Option<bool>> {
    Ok(
        match git.read(&["merge-base", "--is-ancestor", base, head])?.code {
            0 => Some(true),
            1 => Some(false),
            _ => None,
        },
    )
}
fn membership(git: &mut impl Git, oid: Option<&str>, source: &Frozen) -> Result<Membership> {
    let Some(oid) = oid else {
        return Ok(Membership::Unknown("merge_commit_absent"));
    };
    if oid.len() != 40
        || !oid
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Ok(Membership::Unknown("merge_commit_identity_invalid"));
    }
    if oid == source.previous_sha {
        return Ok(Membership::Known(false));
    }
    let expression = format!("{oid}^{{commit}}");
    match git
        .read(&[
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            &expression,
        ])?
        .code
    {
        0 => {}
        1 => return Ok(Membership::Unknown("merge_commit_unavailable_locally")),
        _ => return Ok(Membership::Unknown("merge_commit_lookup_failed")),
    }
    let Some(in_head) = ancestor(git, oid, &source.candidate_sha)? else {
        return Ok(Membership::Unknown("candidate_ancestry_query_failed"));
    };
    if !in_head {
        return Ok(Membership::Known(false));
    }
    let Some(in_base) = ancestor(git, oid, &source.previous_sha)? else {
        return Ok(Membership::Unknown(
            "previous_release_ancestry_query_failed",
        ));
    };
    // Exact base..head set: reachable from head and NOT reachable from base.
    // The prepared previous tag need not itself be an ancestor of later main commits.
    Ok(Membership::Known(!in_base))
}
pub(crate) fn pull_requests(
    tools: &mut (impl Gh + Git),
    repository: &str,
    release: &Release,
    source: &Frozen,
) -> Result<PullRequests> {
    argument(repository)?;
    if release.tag != source.previous_tag {
        return Err(Error(
            "GitHub release metadata does not match frozen previous tag".into(),
        ));
    }
    let search = format!("merged:>={}", release.published_at);
    let raw = json(
        tools,
        &[
            "pr", "list", "--repo", repository, "--state", "merged", "--search", &search,
            "--limit", "1000", "--json", PR_FIELDS,
        ],
    )?;
    let values = raw
        .as_array()
        .ok_or_else(|| Error("GitHub PR evidence must be an array".into()))?;
    if values.len() > QUERY_LIMIT {
        return Err(Error(
            "GitHub PR evidence exceeds requested query limit".into(),
        ));
    }
    let mut rows = Vec::with_capacity(values.len());
    for value in values {
        let mut row = value
            .as_object()
            .cloned()
            .ok_or_else(|| Error("GitHub PR evidence row must be an object".into()))?;
        let oid = row
            .get("mergeCommit")
            .and_then(|v| v.get("oid"))
            .and_then(Value::as_str);
        let admitted = membership(tools, oid, source)?;
        let (present, reason) = match admitted {
            Membership::Known(v) => (Value::Bool(v), Value::Null),
            Membership::Unknown(reason) => (Value::Null, Value::String(reason.into())),
        };
        row.insert("merge_commit_in_range".into(), present);
        row.insert("merge_commit_range_uncertainty".into(), reason);
        rows.push(row);
    }
    Ok(PullRequests {
        may_be_truncated: rows.len() == QUERY_LIMIT,
        rows,
        query_limit: QUERY_LIMIT,
        query_scope: "Merged since publishedAt, capped at 1000; not a complete enumeration of every PR contributing to the Git range",
    })
}
