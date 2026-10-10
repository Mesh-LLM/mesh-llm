//! Assemble unclassified schema1 release evidence; requested candidate is separate from observed workspace.
use super::{
    dirty::{self, Snapshot},
    evidence,
    github::{self, Gh},
    observation::Observation,
    provenance::{self, Error, Frozen, Git, Result},
    publication::Prepared,
    remotes,
};
use serde_json::{Value, json};
use std::path::Path;
pub(crate) struct Report {
    pub(crate) source: Frozen,
    pub(crate) dirty: Snapshot,
    pub(crate) value: Value,
}
fn text(tools: &mut impl Git, args: &[&str], observation: &Observation) -> Result<String> {
    observation.check()?;
    let out = tools.read(args)?;
    if out.code != 0 {
        return Err(Error("Git report evidence unavailable".into()));
    }
    String::from_utf8(out.stdout)
        .map(|s| s.trim_end_matches(['\r', '\n']).into())
        .map_err(|_| Error("Git report evidence is not UTF-8".into()))
}
pub(crate) fn repository_root(
    tools: &mut impl Git,
    observation: &Observation,
) -> Result<std::path::PathBuf> {
    let root = text(tools, &["rev-parse", "--show-toplevel"], observation)?;
    let root = std::path::PathBuf::from(root)
        .canonicalize()
        .map_err(|_| Error("Git working tree root unavailable".into()))?;
    if !root.is_dir() {
        return Err(Error("release inventory requires a working tree".into()));
    }
    Ok(root)
}
pub(crate) fn collect(
    tools: &mut (impl Git + Gh),
    root: &Path,
    repository: &str,
    head: &str,
    override_tag: Option<&str>,
    collected_at: &str,
    observation: &Observation,
) -> Result<Report> {
    observation.check()?;
    let release = github::release(tools, repository, override_tag)?;
    let source = provenance::freeze(tools, &release.tag, head)?;
    let dirty = dirty::capture(tools, root, &source, observation, None)?;
    let branch = text(tools, &["branch", "--show-current"], observation)?;
    let merge_base = text(
        tools,
        &["merge-base", &source.previous_sha, &source.candidate_sha],
        observation,
    )?;
    if merge_base.len() != 40
        || !merge_base
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error(
            "comparison merge base must be a full commit identity".into(),
        ));
    }
    let commits = evidence::commits(tools, &source)?;
    let changed_files = evidence::changed_files(tools, &source)?;
    let prs = github::pull_requests(tools, repository, &release, &source)?;
    let remote_urls = remotes::urls(tools)?;
    let value = json!({"schema_version":1,"collected_at":collected_at,"repository":repository,"remote_urls":remote_urls,
        "candidate":{"ref":source.requested_head,"tag":source.candidate_tag,"sha":source.candidate_sha,"branch":branch,"working_tree_branch":branch,"working_tree_head":source.working_tree_head,"working_tree_root":root,"dirty":dirty},
        "previous_release":release.raw,"previous_release_commit":source.previous_sha,
        "comparison":{"range":format!("{}..{}",source.previous_tag,source.candidate_sha),"merge_base":merge_base,"commits":commits,"changed_files":changed_files,"merged_pr_query_limit":prs.query_limit,"merged_pr_query_may_be_truncated":prs.may_be_truncated,"merged_pr_query_scope":prs.query_scope,"merged_prs_since_release":prs.rows},
        "classification_note":"Raw evidence only. A validator must reconcile and classify atomic release claims; commit and PR counts are not release-item counts."});
    observation.check()?;
    Ok(Report {
        source,
        dirty,
        value,
    })
}
impl Report {
    pub(crate) fn revalidate(
        &self,
        tools: &mut impl Git,
        root: &Path,
        observation: &Observation,
        prepared: Option<&Prepared>,
    ) -> Result<()> {
        observation.check()?;
        provenance::revalidate(tools, &self.source)?;
        dirty::revalidate(
            &self.dirty,
            tools,
            root,
            &self.source,
            observation,
            prepared,
        )?;
        observation.check()
    }
    pub(crate) fn bytes(&self, observation: &Observation) -> Result<Vec<u8>> {
        let mut output = ReportBytes {
            bytes: Vec::new(),
            observation,
            failure: None,
        };
        if serde_json::to_writer_pretty(&mut output, &self.value).is_err() {
            return Err(output
                .failure
                .take()
                .unwrap_or_else(|| Error("release inventory JSON rendering failed".into())));
        }
        if std::io::Write::write_all(&mut output, b"\n").is_err() {
            return Err(output
                .failure
                .take()
                .unwrap_or_else(|| Error("release inventory JSON completion failed".into())));
        }
        Ok(output.bytes)
    }
}
struct ReportBytes<'a> {
    bytes: Vec<u8>,
    observation: &'a Observation,
    failure: Option<Error>,
}
impl std::io::Write for ReportBytes<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Err(error) = self.observation.check() {
            self.failure = Some(error.clone());
            return Err(std::io::Error::other(error));
        }
        if bytes.len() as u64
            > super::publication::REPORT_LIMIT.saturating_sub(self.bytes.len() as u64)
        {
            let error = Error("release inventory report exceeds128MiB".into());
            self.failure = Some(error.clone());
            return Err(std::io::Error::other(error));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.observation.check().map_err(std::io::Error::other)
    }
}
