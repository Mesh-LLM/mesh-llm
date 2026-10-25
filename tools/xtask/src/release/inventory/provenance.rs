//! Release inventory source identities, independently of dirty-tree or PR evidence.
//! This owner never fetches, checks out, tags, publishes, or classifies a release.
use std::fmt;
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Error(pub(crate) String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for Error {}
pub(crate) type Result<T> = std::result::Result<T, Error>;
/// Transport errors include cancellation/deadline/output bounds; never map them to false ancestry.
pub(crate) struct Output {
    pub(crate) code: i32,
    pub(crate) stdout: Vec<u8>,
}
pub(crate) trait Git {
    fn read(&mut self, args: &[&str]) -> Result<Output>;
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Frozen {
    pub(crate) requested_head: String,
    pub(crate) candidate_sha: String,
    pub(crate) candidate_tag: Option<String>,
    pub(crate) previous_tag: String,
    pub(crate) previous_sha: String,
    pub(crate) origin_main_sha: String,
    pub(crate) previous_release_base: String,
    pub(crate) candidate_release_base: String,
    /// Dirty evidence in a later layer must explicitly name this distinct diff base.
    pub(crate) working_tree_head: String,
}
fn error(message: &str) -> Error {
    Error(message.to_owned())
}
fn text(output: Output) -> Result<String> {
    if output.code != 0 {
        return Err(error("Git source identity command failed"));
    }
    let text =
        String::from_utf8(output.stdout).map_err(|_| error("Git identity output must be UTF-8"))?;
    Ok(text.trim_end_matches(['\r', '\n']).to_owned())
}
fn checked(git: &mut impl Git, args: &[&str]) -> Result<String> {
    text(git.read(args)?)
}
fn sha(text: &str) -> bool {
    text.len() == 40
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn commit(git: &mut impl Git, reference: &str) -> Result<String> {
    if reference.is_empty() || reference.contains(['\0', '\r', '\n']) {
        return Err(error("source ref must be nonempty single-line text"));
    }
    let expression = format!("{reference}^{{commit}}");
    let oid = checked(
        git,
        &["rev-parse", "--verify", "--end-of-options", &expression],
    )?;
    if !sha(&oid) {
        return Err(error(
            "Git source identity must be a full lowercase SHA-1 commit",
        ));
    }
    Ok(oid)
}
fn tag_name(git: &mut impl Git, tag: &str) -> Result<Option<String>> {
    let reference = format!("refs/tags/{tag}");
    if tag.contains(['\0', '\r', '\n']) {
        return Err(error("tag must be single-line text"));
    }
    match git.read(&["check-ref-format", &reference])?.code {
        0 => Ok(Some(reference)),
        1 => Ok(None),
        _ => Err(error("Git tag syntax operation failed")),
    }
}
fn tag_ref(git: &mut impl Git, tag: &str) -> Result<String> {
    tag_name(git, tag)?.ok_or_else(|| error("release tag has invalid native Git ref syntax"))
}
fn optional_tag(git: &mut impl Git, tag: &str) -> Result<Option<String>> {
    let reference = tag_ref(git, tag)?;
    let present = git.read(&["show-ref", "--verify", "--quiet", &reference])?;
    match present.code {
        0 => commit(git, &reference).map(Some),
        1 => Ok(None),
        _ => Err(error("Git local-tag lookup failed")),
    }
}
fn merge_base(git: &mut impl Git, left: &str, right: &str) -> Result<String> {
    let base = checked(git, &["merge-base", left, right])?;
    if !sha(&base) {
        return Err(error("Git must yield one canonical merge base"));
    }
    Ok(base)
}
fn ancestor(git: &mut impl Git, base: &str, head: &str) -> Result<bool> {
    match git.read(&["merge-base", "--is-ancestor", base, head])?.code {
        0 => Ok(true),
        1 => Ok(false),
        _ => Err(error("Git ancestry operation failed")),
    }
}
fn release_base(git: &mut impl Git, main: &str, tag: &str, head: &str) -> Result<String> {
    let base = merge_base(git, main, head)?;
    if base == head {
        return Ok(base);
    }
    let range = format!("{base}..{head}");
    let commits = checked(git, &["rev-list", "--count", &range])?;
    if commits != "1" {
        return Err(error(
            "off-main release must have exactly one release-prepare commit",
        ));
    }
    let subject = checked(git, &["show", "-s", "--format=%s", head])?;
    if subject != format!("{tag}: prepare release source") {
        return Err(error(
            "release-prepare subject must name its canonical tag exactly",
        ));
    }
    Ok(base)
}
fn canonical_tag(
    git: &mut impl Git,
    reference: &str,
    head: &str,
    base: &str,
) -> Result<Option<String>> {
    if base == head {
        return Ok(None);
    }
    let explicit = reference.strip_prefix("refs/tags/").unwrap_or(reference);
    // A branch/ref expression need not be a valid tag spelling. Native Git owns ref syntax.
    if let Some(tag) = tag_name(git, explicit)? {
        let present = git.read(&["show-ref", "--verify", "--quiet", &tag])?;
        match present.code {
            0 if commit(git, &tag)? == head => return Ok(Some(explicit.to_owned())),
            0 | 1 => {}
            _ => return Err(error("Git explicit candidate-tag lookup failed")),
        }
    }
    let names = checked(git, &["tag", "--points-at", head])?;
    let names: Vec<_> = names.lines().filter(|s| !s.is_empty()).collect();
    let [only] = names.as_slice() else {
        return Err(error(
            "off-main candidate requires explicit release tag or exactly one local tag",
        ));
    };
    Ok(Some((*only).to_owned()))
}
pub(crate) fn freeze(
    git: &mut impl Git,
    previous_tag: &str,
    requested_head: &str,
) -> Result<Frozen> {
    let previous_sha = optional_tag(git, previous_tag)?.ok_or_else(|| {
        error("previous release tag absent locally; verify intended remote and fetch outside inventory")
    })?;
    let candidate_sha = commit(git, requested_head)?;
    let working_tree_head = commit(git, "HEAD")?;
    let origin_main_sha = commit(git, "origin/main")?;
    let previous_release_base = release_base(git, &origin_main_sha, previous_tag, &previous_sha)?;
    let candidate_base = merge_base(git, &origin_main_sha, &candidate_sha)?;
    let candidate_tag = canonical_tag(git, requested_head, &candidate_sha, &candidate_base)?;
    let candidate_release_base = release_base(
        git,
        &origin_main_sha,
        candidate_tag.as_deref().unwrap_or(requested_head),
        &candidate_sha,
    )?;
    if !ancestor(git, &previous_release_base, &candidate_release_base)? {
        return Err(error(
            "previous release base must be an ancestor of candidate release base",
        ));
    }
    Ok(Frozen {
        requested_head: requested_head.to_owned(),
        candidate_sha,
        candidate_tag,
        previous_tag: previous_tag.to_owned(),
        previous_sha,
        origin_main_sha,
        previous_release_base,
        candidate_release_base,
        working_tree_head,
    })
}
/// Re-admit mutable source names before later report publication, without refreshing frozen evidence.
pub(crate) fn revalidate(git: &mut impl Git, frozen: &Frozen) -> Result<()> {
    let current = freeze(git, &frozen.previous_tag, &frozen.requested_head)?;
    if &current != frozen {
        return Err(error(
            "release inventory source identities changed during collection",
        ));
    }
    Ok(())
}
