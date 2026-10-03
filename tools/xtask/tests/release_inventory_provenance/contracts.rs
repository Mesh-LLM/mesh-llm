use super::real_git::Repository;
use crate::provenance::{self, Git, Output, freeze, revalidate};
#[test]
fn release_inventory_provenance_on_main_freezes_requested_and_working_heads_separately() {
    let mut repo = Repository::new();
    let previous = repo.ok(&["rev-parse", "v1.0.0"]);
    let current = repo.main_commit("feature: advance main");
    let old = freeze(&mut repo, "v1.0.0", "v1.0.0").unwrap();
    assert_eq!(old.requested_head, "v1.0.0");
    assert_eq!(old.previous_tag, "v1.0.0");
    assert_eq!(old.candidate_sha, previous);
    assert_eq!(old.previous_sha, previous);
    assert_eq!(old.previous_release_base, previous);
    assert_eq!(old.candidate_release_base, previous);
    assert_eq!(old.working_tree_head, current);
    assert_eq!(old.origin_main_sha, current);
    assert_eq!(old.candidate_tag, None);
    let head = freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    assert_eq!(head.candidate_sha, current);
    assert_eq!(head.candidate_release_base, current);
    revalidate(&mut repo, &head).unwrap();
}
#[test]
fn release_inventory_provenance_prepared_prior_and_annotated_explicit_candidate() {
    let mut repo = Repository::new();
    let first = repo.ok(&["rev-parse", "HEAD"]);
    let second = repo.main_commit("fix: next main source");
    repo.prepared(&first, "v1.0.1", false);
    let candidate = repo.prepared(&second, "v1.0.2", true);
    let frozen = freeze(&mut repo, "v1.0.1", "refs/tags/v1.0.2").unwrap();
    assert_eq!(frozen.candidate_sha, candidate);
    assert_eq!(frozen.candidate_tag.as_deref(), Some("v1.0.2"));
    assert_eq!(frozen.previous_release_base, first);
    assert_eq!(frozen.candidate_release_base, second);
    revalidate(&mut repo, &frozen).unwrap();
}
#[test]
fn release_inventory_provenance_absent_ambiguous_and_explicit_canonical_tags() {
    let mut repo = Repository::new();
    let base = repo.ok(&["rev-parse", "HEAD"]);
    repo.ok(&["checkout", "--detach", &base]);
    let candidate = repo.commit("v1.0.1: prepare release source");
    assert!(freeze(&mut repo, "v1.0.0", "HEAD").is_err());
    repo.ok(&["tag", "v1.0.1"]);
    let single = freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    assert_eq!(single.candidate_tag.as_deref(), Some("v1.0.1"));
    assert_eq!(single.candidate_sha, candidate);
    repo.ok(&["tag", "ambiguous-other"]);
    assert!(freeze(&mut repo, "v1.0.0", "HEAD").is_err());
    let explicit = freeze(&mut repo, "v1.0.0", "v1.0.1").unwrap();
    assert_eq!(explicit.candidate_tag.as_deref(), Some("v1.0.1"));
    assert!(freeze(&mut repo, "v1.0.0", "ambiguous-other").is_err());
}
#[test]
fn release_inventory_provenance_wrong_tag_subject_and_two_commit_tails_refuse() {
    for wrong_subject in [true, false] {
        let mut repo = Repository::new();
        let base = repo.ok(&["rev-parse", "HEAD"]);
        repo.ok(&["checkout", "--detach", &base]);
        if wrong_subject {
            repo.commit("v1.0.9: prepare release source");
        } else {
            repo.commit("fix: off-main change");
            repo.commit("v1.0.1: prepare release source");
        }
        repo.ok(&["tag", "v1.0.1"]);
        assert!(freeze(&mut repo, "v1.0.0", "v1.0.1").is_err());
    }
    let mut prior = Repository::new();
    prior.commit("unapproved release preparation");
    prior.ok(&["tag", "previous-off-main"]);
    assert!(freeze(&mut prior, "previous-off-main", "HEAD").is_err());
}
#[test]
fn release_inventory_provenance_missing_required_local_refs_refuse_without_fetch() {
    let mut repo = Repository::new();
    assert!(freeze(&mut repo, "absent-published-tag", "HEAD").is_err());
    assert!(freeze(&mut repo, "v1.0.0", "absent-candidate").is_err());
    repo.ok(&["update-ref", "-d", "refs/remotes/origin/main"]);
    assert!(freeze(&mut repo, "v1.0.0", "HEAD").is_err());
    // No fixture remote is configured: source admission cannot secretly obtain missing refs.
    assert_eq!(repo.ok(&["remote"]), "");
}
#[test]
fn release_inventory_provenance_reversed_and_divergent_base_order_refuse() {
    let mut repo = Repository::new();
    let first = repo.ok(&["rev-parse", "HEAD"]);
    let second = repo.main_commit("next main source");
    repo.ok(&["tag", "v1.1.0"]);
    assert!(freeze(&mut repo, "v1.1.0", "v1.0.0").is_err());
    // Two independently valid release-prepared sides of a merged main graph have unordered bases.
    repo.ok(&["checkout", "-b", "left", &first]);
    let left = repo.commit("left source");
    repo.ok(&["checkout", "-b", "right", &second]);
    let right = repo.commit("right source");
    repo.ok(&["checkout", "main"]);
    repo.ok(&[
        "-c",
        "core.hooksPath=",
        "merge",
        "--no-gpg-sign",
        "--no-ff",
        "--no-edit",
        "left",
    ]);
    repo.ok(&[
        "-c",
        "core.hooksPath=",
        "merge",
        "--no-gpg-sign",
        "--no-ff",
        "--no-edit",
        "right",
    ]);
    repo.ok(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
    repo.prepared(&left, "v-left", false);
    repo.prepared(&right, "v-right", false);
    assert!(freeze(&mut repo, "v-left", "v-right").is_err());
}
#[test]
fn release_inventory_provenance_revalidation_refuses_changed_source_bindings() {
    for changed in [
        "previous",
        "requested",
        "working",
        "origin",
        "candidate_tag",
    ] {
        let mut repo = Repository::new();
        let first = repo.ok(&["rev-parse", "HEAD"]);
        let second = repo.main_commit("source for candidate");
        let candidate = repo.prepared(&second, "v1.0.1", false);
        let frozen = freeze(&mut repo, "v1.0.0", "v1.0.1").unwrap();
        match changed {
            "previous" => {
                repo.ok(&["tag", "-f", "v1.0.0", &second]);
            }
            "requested" => {
                repo.ok(&["tag", "-f", "v1.0.1", &first]);
            }
            "working" => {
                repo.ok(&["checkout", "--detach", &first]);
            }
            "origin" => {
                repo.ok(&["update-ref", "refs/remotes/origin/main", &candidate]);
            }
            _ => {
                repo.ok(&["tag", "-d", "v1.0.1"]);
            }
        }
        assert!(revalidate(&mut repo, &frozen).is_err(), "{changed}");
    }
}
struct FailSyntax<'a> {
    repo: &'a mut Repository,
    seen: bool,
}
impl Git for FailSyntax<'_> {
    fn read(&mut self, args: &[&str]) -> provenance::Result<Output> {
        if args == ["check-ref-format", "refs/tags/HEAD"] {
            self.seen = true;
            return Err(provenance::Error(
                "finite cancellation from bounded transport".into(),
            ));
        }
        self.repo.read(args)
    }
}
#[test]
fn release_inventory_provenance_transport_error_is_not_missing_tag_or_false_ancestry() {
    let mut repo = Repository::new();
    let base = repo.ok(&["rev-parse", "HEAD"]);
    repo.prepared(&base, "v1.0.1", false);
    let mut failed = FailSyntax {
        repo: &mut repo,
        seen: false,
    };
    let err = freeze(&mut failed, "v1.0.0", "HEAD").unwrap_err();
    assert!(failed.seen);
    assert_eq!(err.0, "finite cancellation from bounded transport");
    // Real Git fatal lookup remains failure, not an invented false ancestry result.
    let result = repo
        .read(&["merge-base", "--is-ancestor", "missing-oid", "HEAD"])
        .unwrap();
    assert!(result.code > 1);
}
