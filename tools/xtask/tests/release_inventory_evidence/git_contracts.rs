use crate::{
    evidence, github,
    process::Cancellation,
    provenance::{self, Git, Output},
    real_git::Repository,
    remotes,
    transport::Transport,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, time::Duration};
fn transport(repo: &Repository) -> Transport {
    let mut environment: BTreeMap<_, _> = std::env::vars_os().collect();
    environment.insert("GIT_CONFIG_NOSYSTEM".into(), "1".into());
    environment.insert(
        "GIT_CONFIG_GLOBAL".into(),
        repo.root.join("absent-global-config").into(),
    );
    Transport::new(
        &repo.root,
        environment,
        Cancellation::default(),
        Duration::from_secs(30),
    )
    .unwrap()
}
#[cfg(unix)]
#[test]
fn release_inventory_evidence_real_git_metadata_rename_and_unusual_paths() {
    let mut repo = Repository::new();
    let original = "old tab\tand newline\nname";
    let renamed = "new tab\tand newline\nname";
    fs::write(
        repo.root.join(original),
        b"same complete content for rename\n",
    )
    .unwrap();
    repo.ok(&["add", "--", original]);
    repo.main_commit("feature: original path\n\nfirst body paragraph\n\nsecond body paragraph");
    repo.ok(&["tag", "-f", "v1.0.0", "HEAD"]);
    fs::rename(repo.root.join(original), repo.root.join(renamed)).unwrap();
    repo.ok(&["add", "-A"]);
    let last = repo.main_commit("fix: renamed source\n\nmultiline body\nretained continuation");
    let source = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    let mut owned = transport(&repo);
    let rows = evidence::commits(&mut owned, &source).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].sha, last);
    assert_eq!(rows[0].author, "Finite Release Fixture");
    assert_eq!(rows[0].author_email, "release-fixture@example.invalid");
    let native_date = repo.ok(&["show", "--no-patch", "--format=%aI", &last]);
    assert!(matches!(
        native_date.as_str(),
        "2026-10-02T00:00:00Z" | "2026-10-02T00:00:00+00:00"
    ));
    assert_eq!(rows[0].authored_at, native_date);
    assert_eq!(rows[0].subject, "fix: renamed source");
    assert_eq!(rows[0].body, "multiline body\nretained continuation");
    let paths = evidence::changed_files(&mut owned, &source).unwrap();
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].status, "R100");
    assert_eq!(paths[0].path, renamed);
    assert_eq!(paths[0].old_path.as_deref(), Some(original));
    provenance::revalidate(&mut owned, &source).unwrap();
}
#[test]
fn release_inventory_evidence_real_git_order_empty_range_and_add_delete() {
    let mut repo = Repository::new();
    let empty = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    assert!(
        evidence::commits(&mut transport(&repo), &empty)
            .unwrap()
            .is_empty()
    );
    assert!(
        evidence::changed_files(&mut transport(&repo), &empty)
            .unwrap()
            .is_empty()
    );
    fs::write(repo.root.join("removed space"), b"old\n").unwrap();
    repo.ok(&["add", "-A"]);
    repo.main_commit("source for previous release");
    repo.ok(&["tag", "-f", "v1.0.0", "HEAD"]);
    fs::remove_file(repo.root.join("removed space")).unwrap();
    fs::write(repo.root.join("added space"), b"different new data\n").unwrap();
    repo.ok(&["add", "-A"]);
    let first = repo.main_commit("feature: first");
    let second = repo.main_commit("fix: second");
    let source = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    let mut owned = transport(&repo);
    let commits = evidence::commits(&mut owned, &source).unwrap();
    assert_eq!(
        commits.iter().map(|v| v.sha.as_str()).collect::<Vec<_>>(),
        vec![first.as_str(), second.as_str()]
    );
    let paths = evidence::changed_files(&mut owned, &source).unwrap();
    assert!(
        paths
            .iter()
            .any(|v| v.status == "A" && v.path == "added space" && v.old_path.is_none())
    );
    assert!(
        paths
            .iter()
            .any(|v| v.status == "D" && v.path == "removed space" && v.old_path.is_none())
    );
}
struct PrRows(Value);
impl github::Gh for PrRows {
    fn request(&mut self, _: &[&str]) -> provenance::Result<Output> {
        Ok(Output {
            code: 0,
            stdout: serde_json::to_vec(&self.0).unwrap(),
        })
    }
}
struct EvidenceTools<'a, T: Git> {
    gh: &'a mut PrRows,
    git: &'a mut T,
}
impl<T: Git> github::Gh for EvidenceTools<'_, T> {
    fn request(&mut self, args: &[&str]) -> provenance::Result<Output> {
        github::Gh::request(self.gh, args)
    }
}
impl<T: Git> Git for EvidenceTools<'_, T> {
    fn read(&mut self, args: &[&str]) -> provenance::Result<Output> {
        self.git.read(args)
    }
}
fn release(tag: &str) -> github::Release {
    let raw = json!({"tagName":tag,"name":"raw release", "publishedAt":"2026-10-01T00:00:00Z", "url":"https://example.invalid/release", "isDraft":false,"isPrerelease":false,"assets":[],"body":"retained raw body","targetCommitish":"main"});
    let admitted =
        github::release(&mut PrRows(raw.clone()), "Mesh-LLM/mesh-llm", Some(tag)).unwrap();
    assert_eq!(admitted.raw, raw.as_object().unwrap().clone());
    admitted
}
#[test]
fn release_inventory_evidence_real_git_pr_range_excludes_base_and_unknowns() {
    let mut repo = Repository::new();
    let base = repo.ok(&["rev-parse", "HEAD"]);
    let inside = repo.main_commit("feature: actual PR");
    repo.ok(&["checkout", "-b", "other", &base]);
    let outside = repo.commit("outside candidate");
    repo.ok(&["checkout", "main"]);
    let source = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    let mut prs = PrRows(
        json!([{"mergeCommit":{"oid":base}},{"mergeCommit":{"oid":inside},"body":"raw body","labels":[{"name":"feature"}]},{"mergeCommit":{"oid":outside}},{"mergeCommit":null},{"mergeCommit":{"oid":"1111111111111111111111111111111111111111"}},{"mergeCommit":{"oid":"invalid identity"}}]),
    );
    let evidence = github::pull_requests(
        &mut EvidenceTools {
            gh: &mut prs,
            git: &mut transport(&repo),
        },
        "Mesh-LLM/mesh-llm",
        &release(&source.previous_tag),
        &source,
    )
    .unwrap();
    assert_eq!(evidence.rows[0]["merge_commit_in_range"], false);
    assert_eq!(evidence.rows[1]["merge_commit_in_range"], true);
    assert_eq!(evidence.rows[2]["merge_commit_in_range"], false);
    assert_eq!(evidence.rows[1]["body"], "raw body");
    assert_eq!(evidence.rows[1]["labels"][0]["name"], "feature");
    for index in 3..6 {
        assert!(evidence.rows[index]["merge_commit_in_range"].is_null());
        assert!(evidence.rows[index]["merge_commit_range_uncertainty"].is_string());
    }
    assert!(!evidence.may_be_truncated);
    assert_eq!(evidence.query_limit, 1000);
    assert!(evidence.query_scope.contains("not a complete enumeration"));
}
#[test]
fn release_inventory_evidence_real_git_prepared_previous_tag_exact_range() {
    let mut repo = Repository::new();
    let first = repo.ok(&["rev-parse", "HEAD"]);
    repo.prepared(&first, "v1.0.1", false);
    repo.ok(&["checkout", "main"]);
    let merge = repo.main_commit("feature: later main PR");
    let source = provenance::freeze(&mut repo, "v1.0.1", "HEAD").unwrap();
    let commits = evidence::commits(&mut transport(&repo), &source).unwrap();
    assert_eq!(commits.len(), 1);
    assert_eq!(commits[0].sha, merge);
    let mut prs = PrRows(json!([{"mergeCommit":{"oid":merge}},{"mergeCommit":{"oid":first}}]));
    let rows = github::pull_requests(
        &mut EvidenceTools {
            gh: &mut prs,
            git: &mut transport(&repo),
        },
        "Mesh-LLM/mesh-llm",
        &release(&source.previous_tag),
        &source,
    )
    .unwrap();
    assert_eq!(rows.rows[0]["merge_commit_in_range"], true);
    assert_eq!(rows.rows[1]["merge_commit_in_range"], false);
    provenance::revalidate(&mut transport(&repo), &source).unwrap();
}
#[test]
fn release_inventory_evidence_real_git_remote_credentials_never_rendered() {
    let mut repo = Repository::new();
    repo.ok(&["remote","add","origin","https://operator:secret-token@example.invalid/Mesh-LLM/mesh-llm.git?access_token=private-query"]);
    repo.ok(&[
        "remote",
        "add",
        "ssh",
        "operator-private@example.invalid:Mesh-LLM/mesh-llm.git",
    ]);
    let urls = remotes::urls(&mut transport(&repo)).unwrap().join("\n");
    assert!(urls.contains("example.invalid/Mesh-LLM/mesh-llm.git"));
    assert!(urls.contains("example.invalid:Mesh-LLM/mesh-llm.git"));
    for secret in [
        "operator:",
        "secret-token",
        "private-query",
        "operator-private",
    ] {
        assert!(!urls.contains(secret));
    }
}
struct AncestryFailure<'a> {
    owned: &'a mut Transport,
    cancel: bool,
}
impl Git for AncestryFailure<'_> {
    fn read(&mut self, args: &[&str]) -> provenance::Result<Output> {
        if args.first() == Some(&"merge-base") {
            if self.cancel {
                return Err(provenance::Error("finite transport cancellation".into()));
            }
            return Ok(Output {
                code: 128,
                stdout: Vec::new(),
            });
        }
        self.owned.read(args)
    }
}
#[test]
fn release_inventory_evidence_native_failure_unknown_transport_failure_refuses() {
    let mut repo = Repository::new();
    let merge = repo.main_commit("feature: observed merge");
    let source = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    let mut prs = PrRows(json!([{"mergeCommit":{"oid":merge}}]));
    let mut owned = transport(&repo);
    let unknown = github::pull_requests(
        &mut EvidenceTools {
            gh: &mut prs,
            git: &mut AncestryFailure {
                owned: &mut owned,
                cancel: false,
            },
        },
        "Mesh-LLM/mesh-llm",
        &release(&source.previous_tag),
        &source,
    )
    .unwrap();
    assert!(unknown.rows[0]["merge_commit_in_range"].is_null());
    assert_eq!(
        unknown.rows[0]["merge_commit_range_uncertainty"],
        "candidate_ancestry_query_failed"
    );
    let failure = github::pull_requests(
        &mut EvidenceTools {
            gh: &mut prs,
            git: &mut AncestryFailure {
                owned: &mut owned,
                cancel: true,
            },
        },
        "Mesh-LLM/mesh-llm",
        &release(&source.previous_tag),
        &source,
    );
    assert_eq!(failure.err().unwrap().0, "finite transport cancellation");
}
