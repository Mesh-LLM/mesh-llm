//! The Hugging Face file a request's model name stands for, as this node
//! knows it without the peer that served the request.
//!
//! A receipt names the model its node says it ran, and the audit reads
//! that model's weights from Hugging Face. Were the audit to open whatever
//! the receipt names, a peer asked for a large model could run a small one
//! and receipt it honestly, and any peer could choose which repo this node
//! resolves with the user's token, fetches, caches and loads. So the audit
//! runs only against an identity this node derives from the name it asked
//! for ([`RequestedModel::of`]), and the receipt must name that file
//! ([`RequestedModel::claimed`]).
//!
//! What ties a name to a file, in this order:
//!
//! - the name is the file: `owner/repo@revision/path/file.gguf`;
//! - this node's own model catalog lists the name, with its repo and file;
//! - the name is a model reference with a selector, `owner/repo:Q4_K_M` or
//!   `owner/repo@revision:Q4_K_M`: the repo, and any GGUF of it the
//!   selector matches.
//!
//! A revision is part of the identity only when it is a commit, which the
//! name or the catalog gives. Otherwise the identity holds the branch the
//! name or the catalog gives, or the default branch `main`, and the commit
//! the receipt names must be in that branch's history: Hugging Face also
//! serves the commits of pull requests, which anyone can open. The audit
//! checks that before it reads the file (`claimed_models`). Any other name, a runtime's own name, an
//! alias, `local-gguf/sha256-…`, a bare `owner/repo`, is tied to a file
//! only by the serving peer's word (its advertisement or its receipt), so
//! this node cannot audit it.

use skippy_receipt_audit::claimed_file::HfFile;
use skippy_receipt_audit::claimed_file::hf_file::is_commit;

use super::peer_text::shown;

/// A model of this node's catalog: its repo, the revision the catalog
/// pins, and the GGUF's path in the repo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CatalogFile {
    pub(super) repo: String,
    pub(super) revision: Option<String>,
    pub(super) file: String,
}

/// How the id a node gives a local GGUF it serves starts.
const LOCAL_FILE_ID: &str = "local-gguf/";

/// Looks a model name up in this node's catalog.
pub(super) type Catalog<'a> = &'a dyn Fn(&str) -> Option<CatalogFile>;

/// This node's loaded model catalog. It reads what is in memory: no
/// request leaves the node for a lookup.
pub(super) fn node_catalog(name: &str) -> Option<CatalogFile> {
    let model = crate::models::find_loaded_remote_catalog_model_exact(name)?;
    Some(CatalogFile {
        repo: model.repo,
        revision: model.revision,
        file: model.source_file,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum FileRule {
    /// This path in the repo.
    Exact(String),
    /// Any GGUF of the repo the selector of a model reference matches.
    Selector(String),
}

/// The Hugging Face file a model name stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RequestedModel {
    repo: String,
    file: FileRule,
    /// The commit, when this node knows it.
    commit: Option<String>,
    /// The name itself gave the repo and a commit: a request for exactly
    /// those bytes, and no name a file of another repo or commit can share.
    pinned: bool,
    /// Without a commit, the branch the receipt's commit must be on.
    branch: String,
}

/// How a receipt's model relates to the one the request asked for.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Claim {
    /// The receipt names the requested model: this file, at a commit.
    /// When the request named no commit, the branch whose history the
    /// commit must be in, which the audit checks before it reads the file.
    Same(HfFile, Option<String>),
    /// The receipt names another model, and why: the peer signed that it
    /// ran something the request did not ask for.
    Another(String),
    /// The receipt names no Hugging Face file at a commit, so nothing
    /// public says what the peer ran. That cannot be audited.
    Unnamed(String),
}

/// The branch a request names by `revision`, when that is not a commit: the
/// default branch when it names none.
fn branch(revision: Option<&str>) -> String {
    match revision {
        Some(revision) if !is_commit(revision) => revision.to_owned(),
        _ => DEFAULT_BRANCH.to_owned(),
    }
}

/// The branch a Hugging Face name without a revision stands for.
const DEFAULT_BRANCH: &str = "main";

impl RequestedModel {
    /// The file `name` stands for, or why this node cannot tell.
    pub(super) fn of(name: &str, catalog: Catalog<'_>) -> Result<Self, String> {
        // a profile suffix picks runtime settings, not weights
        let name = name.trim();
        let name = name.rsplit_once('#').map_or(name, |(model, _)| model);
        let commit = |revision: Option<&str>| revision.filter(|r| is_commit(r)).map(str::to_owned);
        if let Ok(file) = HfFile::parse_checked(name) {
            let commit = commit(Some(&file.revision));
            return Ok(Self {
                pinned: commit.is_some(),
                commit,
                branch: branch(Some(&file.revision)),
                repo: file.repo,
                file: FileRule::Exact(file.file),
            });
        }
        if let Some(listed) = catalog(name) {
            return Ok(Self {
                // the request named a stem: another node's file may share it
                pinned: false,
                commit: commit(listed.revision.as_deref()),
                branch: branch(listed.revision.as_deref()),
                repo: listed.repo,
                file: FileRule::Exact(listed.file),
            });
        }
        // an id a node mints from a local file reads as owner/repo, and is neither
        if name.starts_with(LOCAL_FILE_ID) {
            return Err(format!(
                "this node cannot tie the model name {} to a Hugging Face file: it is the id of a local file, which only the serving peer can say the weights of",
                shown(name)
            ));
        }
        match skippy_model_ref::ModelRef::parse(name) {
            Ok(model) => match model.selector {
                Some(selector) => Ok(Self {
                    pinned: commit(model.revision.as_deref()).is_some(),
                    commit: commit(model.revision.as_deref()),
                    branch: branch(model.revision.as_deref()),
                    repo: model.repo,
                    file: FileRule::Selector(selector),
                }),
                None => Err(format!(
                    "the model name {} names a repo and no file of it, so this node cannot tell which weights it asked for",
                    shown(name)
                )),
            },
            Err(_) => Err(format!(
                "this node cannot tie the model name {} to a Hugging Face file: it is not owner/repo:selector or owner/repo@revision/file.gguf, and this node's catalog does not list it",
                shown(name)
            )),
        }
    }

    /// The file the receipt named, under this node's name of the repo.
    fn file_at(&self, named: HfFile) -> HfFile {
        HfFile {
            repo: self.repo.clone(),
            revision: named.revision,
            file: named.file,
        }
    }

    fn holds(&self, file: &str) -> bool {
        match &self.file {
            FileRule::Exact(path) => path == file,
            FileRule::Selector(selector) => selector_matches(file, selector),
        }
    }

    /// Whether the request named the repo and a commit itself, so that a
    /// receipt of another model is the peer's word that it ran something
    /// else. A stem or a branch may stand for a file of another repo, or
    /// of another commit, on the peer that served it.
    pub(super) fn pinned(&self) -> bool {
        self.pinned
    }

    /// Whether `receipt_model` names a file of another repo than this one.
    pub(super) fn names_another_repo(&self, receipt_model: &str) -> bool {
        HfFile::parse_checked(receipt_model)
            .is_ok_and(|named| !named.repo.eq_ignore_ascii_case(&self.repo))
    }

    /// Whether the model `receipt_model`, a name the peer wrote, is this
    /// one. The file to open is built from this node's repo name, so the
    /// peer chooses nothing of it but a file the request's name covers and
    /// its commit.
    pub(super) fn claimed(&self, receipt_model: &str) -> Claim {
        let named = match HfFile::parse_checked(receipt_model) {
            Ok(named) => named,
            Err(error) => {
                return Claim::Unnamed(format!(
                    "the receipt names the model {}, which is not repo@commit/file on Hugging Face ({})",
                    shown(receipt_model),
                    shown(&format!("{error:#}"))
                ));
            }
        };
        // Hugging Face holds no two repos whose names differ only in case
        if !named.repo.eq_ignore_ascii_case(&self.repo) || !self.holds(&named.file) {
            return Claim::Another(format!(
                "the receipt names the model {}; the request asked for {self}",
                shown(receipt_model)
            ));
        }
        if !named.is_pinned() {
            return Claim::Unnamed(format!(
                "the receipt names {} at the revision {}, which is not a commit and can move to other bytes",
                shown(&named.file),
                shown(&named.revision)
            ));
        }
        match &self.commit {
            Some(commit) if *commit != named.revision => Claim::Another(format!(
                "the receipt names the model {}; the request asked for {self}",
                shown(receipt_model)
            )),
            Some(_) => Claim::Same(self.file_at(named), None),
            None => Claim::Same(self.file_at(named), Some(self.branch.clone())),
        }
    }
}

impl std::fmt::Display for RequestedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.repo)?;
        if let Some(commit) = &self.commit {
            write!(f, "@{commit}")?;
        }
        match &self.file {
            FileRule::Exact(path) => write!(f, "/{path}"),
            FileRule::Selector(selector) => write!(f, ":{selector}"),
        }
    }
}

/// Whether `selector`, of a model reference, picks the GGUF at `file`: its
/// path, its name without `.gguf`, or its quantization, as model
/// resolution matches them.
fn selector_matches(file: &str, selector: &str) -> bool {
    let name = file.rsplit('/').next().unwrap_or(file);
    file.eq_ignore_ascii_case(selector)
        || name.eq_ignore_ascii_case(selector)
        || name
            .strip_suffix(".gguf")
            .is_some_and(|stem| stem.eq_ignore_ascii_case(selector))
        || skippy_model_ref::gguf_matches_quant_selector(file, selector)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const OTHER: &str = "fedcba9876543210fedcba9876543210fedcba98";

    fn no_catalog(_: &str) -> Option<CatalogFile> {
        None
    }

    fn catalog(name: &str) -> Option<CatalogFile> {
        (name == "Qwen3-4B-Q4_K_M").then(|| CatalogFile {
            repo: "unsloth/Qwen3-4B-GGUF".into(),
            revision: None,
            file: "Qwen3-4B-Q4_K_M.gguf".into(),
        })
    }

    fn requested(name: &str) -> RequestedModel {
        RequestedModel::of(name, &catalog).unwrap_or_else(|why| panic!("{name}: {why}"))
    }

    fn receipt(repo: &str, revision: &str, file: &str) -> String {
        format!("{repo}@{revision}/{file}")
    }

    #[test]
    fn the_names_this_node_can_tie_to_a_file() {
        for (name, shown) in [
            (
                "unsloth/Qwen3-4B-GGUF:Q4_K_M",
                "unsloth/Qwen3-4B-GGUF:Q4_K_M",
            ),
            (
                "unsloth/Qwen3-4B-GGUF:Q4_K_M#low-ctx",
                "unsloth/Qwen3-4B-GGUF:Q4_K_M",
            ),
            // a branch is not part of the identity: it names no bytes
            (
                "unsloth/Qwen3-4B-GGUF@main:Q4_K_M",
                "unsloth/Qwen3-4B-GGUF:Q4_K_M",
            ),
            (
                "unsloth/Qwen3-4B-GGUF@main/Q8_0/Qwen3-4B-Q8_0.gguf",
                "unsloth/Qwen3-4B-GGUF/Q8_0/Qwen3-4B-Q8_0.gguf",
            ),
            (
                "Qwen3-4B-Q4_K_M",
                "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
            ),
        ] {
            assert_eq!(requested(name).to_string(), shown, "{name}");
        }
        let pinned = requested(&format!("o/m@{COMMIT}:Q8_0"));
        assert_eq!(pinned.to_string(), format!("o/m@{COMMIT}:Q8_0"));
        let pinned = requested(&receipt("o/m", COMMIT, "m.gguf"));
        assert_eq!(pinned.commit.as_deref(), Some(COMMIT));
    }

    #[test]
    fn the_names_only_the_peer_can_tie_to_a_file() {
        for name in [
            // a runtime's name that no catalog lists, an alias
            "Qwen3-32B-Q4_K_M",
            "my-model",
            // an id minted from a local file
            "local-gguf/sha256-ab12",
            // a repo, and no file of it
            "unsloth/Qwen3-4B-GGUF",
            "unsloth/Qwen3-4B-GGUF@main",
            "auto",
            "",
        ] {
            let why = RequestedModel::of(name, &catalog).expect_err(name);
            assert!(!why.contains('\n'), "{why}");
        }
        // a local file's id is not a repo that lacks a file
        let local = RequestedModel::of("local-gguf/sha256-ab12", &catalog).unwrap_err();
        assert!(local.contains("the id of a local file"), "{local}");
        let repo = RequestedModel::of("unsloth/Qwen3-4B-GGUF", &catalog).unwrap_err();
        assert!(repo.contains("names a repo and no file of it"), "{repo}");
        // a catalog that is not loaded ties nothing
        assert!(RequestedModel::of("Qwen3-4B-Q4_K_M", &no_catalog).is_err());
    }

    #[test]
    fn a_receipt_must_name_the_requested_repo_and_file() {
        let asked = requested("unsloth/Qwen3-32B-GGUF:Q4_K_M");
        let same = receipt("unsloth/Qwen3-32B-GGUF", COMMIT, "Qwen3-32B-Q4_K_M.gguf");
        assert_eq!(
            asked.claimed(&same),
            Claim::Same(HfFile::parse(&same).unwrap(), Some("main".into()))
        );
        // a split is named by its first shard, a quantization by its directory
        for file in [
            "Qwen3-32B-Q4_K_M-00001-of-00003.gguf",
            "Q4_K_M/Qwen3-32B-Q4_K_M.gguf",
        ] {
            let named = receipt("unsloth/Qwen3-32B-GGUF", COMMIT, file);
            assert!(matches!(asked.claimed(&named), Claim::Same(..)), "{file}");
        }
        for (what, named) in [
            (
                "a smaller model of another repo",
                receipt("unsloth/Qwen3-0.6B-GGUF", COMMIT, "Qwen3-0.6B-Q4_K_M.gguf"),
            ),
            (
                "another quantization of the repo",
                receipt("unsloth/Qwen3-32B-GGUF", COMMIT, "Qwen3-32B-Q2_K.gguf"),
            ),
            (
                "a repo the peer owns under the same name",
                receipt("evil/Qwen3-32B-GGUF", COMMIT, "Qwen3-32B-Q4_K_M.gguf"),
            ),
        ] {
            let Claim::Another(why) = asked.claimed(&named) else {
                panic!("{what} passes as the requested model");
            };
            assert!(
                why.contains("unsloth/Qwen3-32B-GGUF:Q4_K_M"),
                "{what}: {why}"
            );
        }
    }

    #[test]
    fn the_file_to_open_carries_this_nodes_repo_name() {
        // the receipt's spelling of the repo is never what is opened
        let asked = requested("unsloth/Qwen3-4B-GGUF:Q4_K_M");
        let named = receipt("UNSLOTH/qwen3-4b-gguf", COMMIT, "Qwen3-4B-Q4_K_M.gguf");
        let Claim::Same(file, _) = asked.claimed(&named) else {
            panic!("the same repo in another case");
        };
        assert_eq!(file.repo, "unsloth/Qwen3-4B-GGUF");
        assert_eq!(file.revision, COMMIT);
    }

    #[test]
    fn a_known_commit_is_part_of_the_identity() {
        let asked = requested(&format!("o/m@{COMMIT}:Q8_0"));
        assert!(matches!(
            asked.claimed(&receipt("o/m", COMMIT, "m-Q8_0.gguf")),
            Claim::Same(..)
        ));
        assert!(matches!(
            asked.claimed(&receipt("o/m", OTHER, "m-Q8_0.gguf")),
            Claim::Another(_)
        ));
        // without one, the receipt's commit is the peer's to state
        let unpinned = requested("o/m:Q8_0");
        assert!(matches!(
            unpinned.claimed(&receipt("o/m", OTHER, "m-Q8_0.gguf")),
            Claim::Same(..)
        ));
    }

    /// Without a commit in the request, the receipt's commit must be in
    /// the history of the branch the request named, or of `main`: a commit
    /// of a pull request on the repo is served too.
    #[test]
    fn a_commit_the_request_did_not_pin_must_be_on_its_branch() {
        let on = |name: &str| match requested(name).claimed(&receipt("o/m", OTHER, "m-Q8_0.gguf")) {
            Claim::Same(_, branch) => branch,
            other => panic!("{name}: {other:?}"),
        };
        assert_eq!(on("o/m:Q8_0").as_deref(), Some("main"));
        assert_eq!(on("o/m@dev:Q8_0").as_deref(), Some("dev"));
        assert_eq!(on("o/m@v1.0/m-Q8_0.gguf").as_deref(), Some("v1.0"));
        // a commit the request named is checked as it is: no branch
        assert_eq!(on(&format!("o/m@{OTHER}:Q8_0")), None);
        // a catalog entry names its branch, or none
        let listed = |revision: Option<&str>| {
            let revision = revision.map(str::to_owned);
            let catalog = move |_: &str| {
                Some(CatalogFile {
                    repo: "o/m".into(),
                    revision: revision.clone(),
                    file: "m-Q8_0.gguf".into(),
                })
            };
            match RequestedModel::of("listed", &catalog)
                .unwrap()
                .claimed(&receipt("o/m", OTHER, "m-Q8_0.gguf"))
            {
                Claim::Same(_, branch) => branch,
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(listed(None).as_deref(), Some("main"));
        assert_eq!(listed(Some("release")).as_deref(), Some("release"));
        assert_eq!(listed(Some(OTHER)), None);
    }

    #[test]
    fn a_receipt_that_names_no_file_at_a_commit_cannot_be_audited() {
        let asked = requested("Qwen3-4B-Q4_K_M");
        for named in [
            "Qwen3-4B-Q4_K_M",
            "/models/Qwen3-4B-Q4_K_M.gguf",
            "local-gguf/sha256-ab12",
            "",
            "unsloth/Qwen3-4B-GGUF@main/../../etc/passwd.gguf",
        ] {
            assert!(matches!(asked.claimed(named), Claim::Unnamed(_)), "{named}");
        }
        // the right file at a branch: its bytes can move
        let branch = receipt("unsloth/Qwen3-4B-GGUF", "main", "Qwen3-4B-Q4_K_M.gguf");
        assert!(matches!(asked.claimed(&branch), Claim::Unnamed(_)));
        // another file at a branch is still another model
        let other = receipt("unsloth/Qwen3-4B-GGUF", "main", "Qwen3-4B-Q2_K.gguf");
        assert!(matches!(asked.claimed(&other), Claim::Another(_)));
    }

    #[test]
    fn a_reason_holds_no_raw_peer_text() {
        let asked = requested("Qwen3-4B-Q4_K_M");
        let forged = format!("x\nWARN audit passed\x1b[2K{}", "y".repeat(4096));
        let Claim::Unnamed(why) = asked.claimed(&forged) else {
            panic!("a forged name is a file");
        };
        assert!(!why.contains('\n') && !why.contains('\x1b'), "{why}");
        assert!(why.len() < 1024, "{}", why.len());
    }

    #[test]
    fn a_selector_matches_as_model_resolution_does() {
        assert!(selector_matches("m-Q4_K_M.gguf", "q4_k_m"));
        assert!(selector_matches("dir/model.gguf", "model.gguf"));
        assert!(selector_matches("dir/model.gguf", "model"));
        assert!(selector_matches("dir/model.gguf", "dir/model.gguf"));
        // a quantization that only starts the same is another file
        assert!(!selector_matches("m-Q4_K_M.gguf", "Q4_K"));
        assert!(!selector_matches("m-Q4_K_M.gguf", "Q8_0"));
    }

    /// Only a name that gives the repo and a commit pins the bytes; a stem,
    /// a branch or a bare selector may stand for another file on the peer.
    #[test]
    fn only_a_repo_at_a_commit_pins_the_request() {
        for name in [format!("o/m@{COMMIT}/m.gguf"), format!("o/m@{COMMIT}:Q8_0")] {
            assert!(requested(&name).pinned(), "{name}");
        }
        for name in [
            "Qwen3-4B-Q4_K_M",
            "o/m:Q8_0",
            "o/m@main:Q8_0",
            "o/m@main/m.gguf",
        ] {
            assert!(!requested(name).pinned(), "{name}");
        }
        let stem = requested("Qwen3-4B-Q4_K_M");
        let elsewhere = receipt("someone/Qwen3-4B-GGUF", COMMIT, "Qwen3-4B-Q4_K_M.gguf");
        assert!(stem.names_another_repo(&elsewhere));
        let same = receipt("Unsloth/qwen3-4b-gguf", OTHER, "Qwen3-4B-Q4_K_M.gguf");
        assert!(!stem.names_another_repo(&same));
        assert!(!stem.names_another_repo("not a file"));
    }
}
