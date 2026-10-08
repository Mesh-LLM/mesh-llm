//! Lossless UTF-8 commit and NUL-delimited rename-aware path evidence over frozen Git OIDs.
use super::provenance::{Error, Frozen, Git, Result};
use serde::Serialize;
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct Commit {
    pub(crate) sha: String,
    pub(crate) authored_at: String,
    pub(crate) author: String,
    pub(crate) author_email: String,
    pub(crate) subject: String,
    pub(crate) body: String,
}
#[derive(Debug, Serialize, PartialEq, Eq)]
pub(crate) struct ChangedFile {
    pub(crate) status: String,
    pub(crate) path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) old_path: Option<String>,
}
fn read(git: &mut impl Git, args: &[&str]) -> Result<Vec<u8>> {
    let out = git.read(args)?;
    if out.code != 0 {
        return Err(Error("Git raw evidence command failed".into()));
    }
    Ok(out.stdout)
}
fn fields(bytes: &[u8]) -> Result<Vec<&str>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    if bytes.last() != Some(&0) {
        return Err(Error("Git raw evidence record terminator absent".into()));
    }
    bytes[..bytes.len() - 1]
        .split(|b| *b == 0)
        .map(|v| std::str::from_utf8(v).map_err(|_| Error("Git raw evidence is not UTF-8".into())))
        .collect()
}
pub(crate) fn commits(git: &mut impl Git, source: &Frozen) -> Result<Vec<Commit>> {
    let range = format!("{}..{}", source.previous_sha, source.candidate_sha);
    let bytes = read(
        git,
        &[
            "log",
            "--no-show-signature",
            "--no-notes",
            "--no-decorate",
            "--encoding=UTF-8",
            "-z",
            "--reverse",
            "--format=%H%x00%aI%x00%an%x00%ae%x00%s%x00%b",
            &range,
            "--",
        ],
    )?;
    let values = fields(&bytes)?;
    if !values.len().is_multiple_of(6) {
        return Err(Error("Git commit record has incomplete fields".into()));
    }
    values
        .as_chunks::<6>()
        .0
        .iter()
        .map(|v| {
            if v[0].len() != 40
                || !v[0]
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(Error("Git commit evidence has invalid identity".into()));
            }
            Ok(Commit {
                sha: v[0].into(),
                authored_at: v[1].into(),
                author: v[2].into(),
                author_email: v[3].into(),
                subject: v[4].into(),
                body: v[5].trim_end_matches('\n').into(),
            })
        })
        .collect()
}
pub(crate) fn changed_files(git: &mut impl Git, source: &Frozen) -> Result<Vec<ChangedFile>> {
    let bytes = read(
        git,
        &[
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "--find-renames",
            "-z",
            &source.previous_sha,
            &source.candidate_sha,
            "--",
        ],
    )?;
    let values = fields(&bytes)?;
    let mut rows = Vec::new();
    let mut cursor = 0;
    while cursor < values.len() {
        let status = values[cursor];
        cursor += 1;
        if status.is_empty()
            || !matches!(
                status.as_bytes()[0],
                b'A' | b'C' | b'D' | b'M' | b'R' | b'T' | b'U' | b'X' | b'B'
            )
            || !status.as_bytes()[1..].iter().all(u8::is_ascii_digit)
        {
            return Err(Error("Git changed-path status invalid".into()));
        }
        let renamed = status.starts_with(['R', 'C']);
        let needed = if renamed { 2 } else { 1 };
        let paths = values
            .get(cursor..cursor + needed)
            .ok_or_else(|| Error("Git changed-path record incomplete".into()))?;
        if paths.iter().any(|p| p.is_empty()) {
            return Err(Error("Git changed-path name absent".into()));
        }
        rows.push(ChangedFile {
            status: status.into(),
            path: paths[needed - 1].into(),
            old_path: renamed.then(|| paths[0].into()),
        });
        cursor += needed;
    }
    Ok(rows)
}
