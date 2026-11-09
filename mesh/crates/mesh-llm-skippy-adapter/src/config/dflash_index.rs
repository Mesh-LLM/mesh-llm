//! The DFlash pairing index this release trusts, named by content hash.
//!
//! A release pins `data/dflash-pairings/index.json` by the SHA-256 of its
//! bytes. CI publishes the file as `index/<sha256>.json`; the host fetches it
//! from the Hugging Face cache or `main` and uses it only if its bytes hash
//! to the pin. Choosing a pairing from it is `dflash_catalog`'s job.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use hf_hub::RepoTypeModel;
use sha2::{Digest, Sha256};
use skippy_model_hf::store::local::{huggingface_hub_cache_dir, huggingface_repo_folder_name};
use skippy_package_format::DFlashPairingIndex;

/// The Hugging Face repo that publishes the DFlash pairing index.
pub const DFLASH_PAIRING_INDEX_REPO: &str = "meshllm/dflash-pairings";
/// The SHA-256 of the pairing index this release trusts: the bytes of
/// `data/dflash-pairings/index.json` in this crate. CI publishes that file as
/// `index/<sha256>.json` in [`DFLASH_PAIRING_INDEX_REPO`] when it changes on
/// `main`, so editing the pairings means updating this constant too.
pub const DFLASH_PAIRING_INDEX_SHA256: &str =
    "b30d6fe7e774896bc9748023969fecd600597d5031f82b7c62a590c7f02c94a3";

/// A pairing index named by the SHA-256 of its bytes.
///
/// Published index files are never changed or deleted, so `main` of the repo
/// holds every index any release pins, and a file whose bytes hash to the pin
/// is that index wherever it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DFlashPairingIndexPin {
    pub repo: &'static str,
    pub sha256: &'static str,
}

impl DFlashPairingIndexPin {
    /// The revision the index is fetched at.
    pub const REVISION: &'static str = "main";

    /// The index this release trusts.
    pub fn shipped() -> Self {
        Self {
            repo: DFLASH_PAIRING_INDEX_REPO,
            sha256: DFLASH_PAIRING_INDEX_SHA256,
        }
    }

    /// The index's path in the repo: `index/<sha256>.json`.
    pub fn file(&self) -> String {
        format!("index/{}.json", self.sha256)
    }

    /// Every copy of the index in the Hugging Face cache, one per snapshot
    /// that holds it, in a stable order.
    pub fn cached_paths(&self) -> Vec<PathBuf> {
        self.cached_paths_in(
            &huggingface_hub_cache_dir()
                .join(huggingface_repo_folder_name(self.repo, RepoTypeModel))
                .join("snapshots"),
        )
    }

    /// The copies of the index under `snapshots`, a repo's Hugging Face cache
    /// snapshot directory. Its file name is its hash, so any copy that passes
    /// [`Self::load`] will do.
    fn cached_paths_in(&self, snapshots: &Path) -> Vec<PathBuf> {
        let file = self.file();
        let Ok(entries) = std::fs::read_dir(snapshots) else {
            return Vec::new();
        };
        let mut found: Vec<PathBuf> = entries
            .filter_map(|entry| Some(entry.ok()?.path().join(&file)))
            .filter(|path| path.is_file())
            .collect();
        found.sort();
        found
    }

    /// Reads the index at `path`, refusing it unless its bytes hash to the pin.
    pub fn load(&self, path: &Path) -> Result<DFlashPairingIndex> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let sha256 = sha256_hex(&bytes);
        ensure!(
            sha256 == self.sha256,
            "{} has sha256 {sha256}, expected {}",
            path.display(),
            self.sha256
        );
        let json = std::str::from_utf8(&bytes)
            .with_context(|| format!("{} is not UTF-8", path.display()))?;
        DFlashPairingIndex::from_json(json)
            .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))
    }
}

/// The lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Reads and validates a pairing index file, without checking its hash. Tests
/// only: [`DFlashPairingIndexPin::load`] is the one way to use an index.
#[cfg(test)]
pub(super) fn load_dflash_pairing_index(path: &Path) -> Result<DFlashPairingIndex> {
    let json = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    DFlashPairingIndex::from_json(&json)
        .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))
}

/// The checked-in index this release pins.
#[cfg(test)]
pub(super) const INDEX_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/dflash-pairings/index.json"
);

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_package_format::DFlashBackends;

    const README_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/dflash-pairings/README.md"
    );

    /// The pin is the hash of the checked-in index, so a change to the
    /// pairings has to update [`DFLASH_PAIRING_INDEX_SHA256`] in the same
    /// change, and CI publishes exactly the bytes this release trusts.
    #[test]
    fn the_pin_is_the_hash_of_the_checked_in_index() {
        let bytes = std::fs::read(INDEX_PATH).unwrap();
        assert_eq!(
            DFLASH_PAIRING_INDEX_SHA256,
            sha256_hex(&bytes),
            "set DFLASH_PAIRING_INDEX_SHA256 to the sha256 of {INDEX_PATH}"
        );
        let pin = DFlashPairingIndexPin::shipped();
        assert_eq!(pin.repo, "meshllm/dflash-pairings");
        assert_eq!(
            pin.file(),
            format!("index/{DFLASH_PAIRING_INDEX_SHA256}.json")
        );
        assert_eq!(
            pin.load(Path::new(INDEX_PATH)).unwrap(),
            load_dflash_pairing_index(Path::new(INDEX_PATH)).unwrap()
        );
    }

    #[test]
    fn an_index_whose_bytes_do_not_hash_to_the_pin_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");
        let mut json = std::fs::read_to_string(INDEX_PATH).unwrap();
        // Still a valid index, but not the one the pin names.
        json.push('\n');
        std::fs::write(&path, &json).unwrap();
        assert!(load_dflash_pairing_index(&path).is_ok());
        let error = DFlashPairingIndexPin::shipped()
            .load(&path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected"), "{error}");
        assert!(error.contains(DFLASH_PAIRING_INDEX_SHA256), "{error}");
    }

    #[test]
    fn a_cached_index_is_found_by_its_hash_in_any_snapshot() {
        let pin = DFlashPairingIndexPin::shipped();
        let snapshots = tempfile::tempdir().unwrap();
        assert!(
            pin.cached_paths_in(&snapshots.path().join("missing"))
                .is_empty()
        );
        // Snapshots without the file, then with it in two of them.
        let older = snapshots.path().join("a".repeat(40));
        let newer = snapshots.path().join("b".repeat(40));
        std::fs::create_dir_all(older.join("index")).unwrap();
        std::fs::create_dir_all(newer.join("index")).unwrap();
        assert!(pin.cached_paths_in(snapshots.path()).is_empty());
        let path = newer.join(pin.file());
        std::fs::copy(INDEX_PATH, &path).unwrap();
        assert_eq!(pin.cached_paths_in(snapshots.path()), vec![path.clone()]);
        assert!(pin.load(&path).is_ok());
        std::fs::write(older.join(pin.file()), "corrupt").unwrap();
        assert_eq!(
            pin.cached_paths_in(snapshots.path()),
            vec![older.join(pin.file()), path]
        );
    }

    /// The pairings table in the Hugging Face README, rendered from the index.
    fn readme_table(index: &DFlashPairingIndex) -> String {
        let mut table =
            String::from("| Match | Draft | Cap | Backends |\n| --- | --- | --- | --- |\n");
        for pairing in &index.pairings {
            let target = &pairing.target;
            let mut shape = format!(
                "`{}`, {} layers, {} wide",
                target.architecture, target.layer_count, target.embedding_length
            );
            if target.expert_count > 0 {
                shape.push_str(&format!(", {} experts", target.expert_count));
            }
            if let Some(family) = &target.family {
                shape.push_str(&format!(", name carries `{family}`"));
            }
            let draft = &pairing.draft;
            let backends = match pairing.backends {
                DFlashBackends::All => "all",
                DFlashBackends::Metal => "Metal",
                DFlashBackends::NotMetal => "not Metal",
            };
            table.push_str(&format!(
                "| {shape} | [{repo}](https://huggingface.co/{repo}/blob/{revision}/{file}) `{file}` | {cap} | {backends} |\n",
                repo = draft.repo,
                revision = draft.revision,
                file = draft.file,
                cap = pairing.max_draft_tokens,
            ));
        }
        table
    }

    /// The README CI publishes next to the index lists its pairings.
    /// `UPDATE_DFLASH_PAIRINGS_README=1` rewrites the table from the index.
    #[test]
    fn the_readme_lists_the_checked_in_pairings() {
        const START: &str = "<!-- pairings:start -->\n";
        const END: &str = "<!-- pairings:end -->";
        let index = load_dflash_pairing_index(Path::new(INDEX_PATH)).unwrap();
        let readme = std::fs::read_to_string(README_PATH).unwrap();
        let (head, rest) = readme.split_once(START).expect("README start marker");
        let (_, tail) = rest.split_once(END).expect("README end marker");
        let expected = format!("{head}{START}{}{END}{tail}", readme_table(&index));
        if std::env::var_os("UPDATE_DFLASH_PAIRINGS_README").is_some() {
            std::fs::write(README_PATH, &expected).unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(README_PATH).unwrap(),
            expected,
            "regenerate with UPDATE_DFLASH_PAIRINGS_README=1 cargo test -p mesh-llm-skippy-adapter the_readme_lists"
        );
    }

    #[test]
    fn the_index_is_pinned_by_a_lowercase_sha256() {
        assert_eq!(DFLASH_PAIRING_INDEX_SHA256.len(), 64);
        assert!(
            DFLASH_PAIRING_INDEX_SHA256
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        );
        assert_eq!(DFlashPairingIndexPin::REVISION, "main");
    }
}
