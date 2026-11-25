//! The files the block store is saved in.
//!
//! The operator's and the plugins' blocks are in `peer_blocks.json`. The
//! blocks this node's receipt audits request are in
//! `peer_blocks_receipts.json`: a release without receipts does not know the
//! `receipts` requester, so on a downgrade it would set a file that names it
//! aside as unreadable and start with no blocks, the operator's included.
//! Kept apart, the older release reads the operator's and plugins' blocks
//! as before and never opens the other file, which is still there after the
//! next upgrade.

use std::collections::BTreeMap;
use std::path::Path;

use super::{ActiveBlock, Requester, Store};

pub(super) const FILE_NAME: &str = "peer_blocks.json";
/// The blocks requested by [`Requester::Receipts`].
pub(super) const RECEIPTS_FILE_NAME: &str = "peer_blocks_receipts.json";

/// The blocks of `store` each file holds: the operator's and plugins', and
/// the receipt audits'.
fn by_file(store: &Store) -> (Store, Store) {
    let (receipts, others): (BTreeMap<String, ActiveBlock>, BTreeMap<String, ActiveBlock>) = store
        .blocks
        .clone()
        .into_iter()
        .partition(|(_, block)| block.requested_by == Requester::Receipts);
    (Store { blocks: others }, Store { blocks: receipts })
}

/// The store loaded from `directory`, and why nothing may be saved this
/// run, if anything: a file that cannot be read, or that cannot be decoded
/// and then cannot be set aside, must never be overwritten.
pub(super) struct Loaded {
    pub(super) store: Store,
    pub(super) not_saved: Option<String>,
}

/// Load the store saved in `directory`. A failure of the receipts file
/// keeps the operator's and plugins' blocks already read: the receipts file
/// is auxiliary, and losing it must not make operator-blocked peers
/// routable, even on a node with receipts off.
pub(super) fn load(directory: &Path, now_ms: u64) -> Loaded {
    let mut store = match load_file(&directory.join(FILE_NAME), now_ms) {
        Ok(store) => store,
        Err(problem) => {
            return Loaded {
                store: Store::default(),
                not_saved: Some(problem),
            };
        }
    };
    let receipts = match load_file(&directory.join(RECEIPTS_FILE_NAME), now_ms) {
        Ok(receipts) => receipts,
        Err(problem) => {
            return Loaded {
                store,
                not_saved: Some(problem),
            };
        }
    };
    for (peer, block) in receipts.blocks {
        // A peer has one block. The operator's file wins: it is the one the
        // operator can read and undo on any release.
        store.blocks.entry(peer).or_insert(block);
    }
    Loaded {
        store,
        not_saved: None,
    }
}

/// One file of the store. A missing file is empty; one that cannot be
/// decoded is set aside as `<name>.corrupt-<ms>` and read as empty.
fn load_file(path: &Path, now_ms: u64) -> Result<Store, String> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Store::default()),
        Err(error) => return Err(format!("{} could not be read ({error})", path.display())),
    };
    let error = match serde_json::from_slice(&bytes) {
        Ok(store) => return Ok(store),
        Err(error) => error,
    };
    let aside = path.with_extension(format!("json.corrupt-{now_ms}"));
    match std::fs::rename(path, &aside) {
        Ok(()) => {
            tracing::warn!(
                "peer blocks: {} is unreadable ({error}); moved to {} and starting without its blocks",
                path.display(),
                aside.display()
            );
            Ok(Store::default())
        }
        Err(rename_error) => Err(format!(
            "{} is unreadable ({error}) and could not be moved to {} ({rename_error})",
            path.display(),
            aside.display()
        )),
    }
}

/// Write `store` durably. Each file is replaced whole: the new file is
/// synced before it replaces the old one, and the directory is synced after,
/// so a crash or power loss leaves either the old file or the new one, never
/// a torn one.
///
/// The operator's file is written first. When the operator takes over a
/// block the receipt audits requested, a crash between the two writes then
/// leaves the block in both files, and [`load`] keeps the operator's.
pub(super) fn save(directory: &Path, store: &Store) -> std::io::Result<()> {
    let (others, receipts) = by_file(store);
    std::fs::create_dir_all(directory)?;
    write_file(directory, &directory.join(FILE_NAME), &others)?;
    let receipts_path = directory.join(RECEIPTS_FILE_NAME);
    // A node that never audits never gets the second file.
    if !receipts.blocks.is_empty() || receipts_path.exists() {
        write_file(directory, &receipts_path, &receipts)?;
    }
    Ok(())
}

fn write_file(directory: &Path, path: &Path, store: &Store) -> std::io::Result<()> {
    use std::io::Write as _;

    let temp = path.with_extension("json.tmp");
    {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(&serde_json::to_vec_pretty(store)?)?;
        file.sync_all()?;
    }
    std::fs::rename(&temp, path)?;
    sync_directory(directory)
}

/// Make a rename inside `directory` durable. On Unix that means syncing the
/// directory itself. The standard library cannot open a directory as a file
/// on Windows, so there the file is synced before the rename and the
/// rename's own durability is left to the filesystem.
#[cfg(unix)]
fn sync_directory(directory: &Path) -> std::io::Result<()> {
    std::fs::File::open(directory)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_directory: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{BlockLength, PeerBlocks};
    use super::*;

    fn peer() -> iroh::EndpointId {
        iroh::SecretKey::generate().public()
    }

    /// What a release without receipts makes of `peer_blocks.json`: it
    /// knows the `operator` and `plugin:<id>` requesters only, and a file
    /// naming another is set aside with every block in it (`None`).
    fn read_by_a_release_without_receipts(directory: &Path) -> Option<Vec<String>> {
        let Ok(bytes) = std::fs::read(directory.join(FILE_NAME)) else {
            return Some(Vec::new());
        };
        let file: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        let mut peers = Vec::new();
        for (peer, block) in file["blocks"].as_object()? {
            let requester = block["requested_by"].as_str()?;
            if requester != "operator" && !requester.starts_with("plugin:") {
                return None;
            }
            peers.push(peer.clone());
        }
        Some(peers)
    }

    #[test]
    fn a_downgrade_keeps_the_operators_blocks_beside_a_receipts_block() {
        let directory = tempfile::tempdir().unwrap();
        let (by_operator, by_audit) = (peer(), peer());
        let blocks = PeerBlocks::load(directory.path());
        blocks
            .block(
                &by_operator,
                BlockLength::UntilUndone,
                Requester::Operator,
                None,
                1,
            )
            .unwrap();
        blocks
            .block(
                &by_audit,
                BlockLength::SevenDays,
                Requester::Receipts,
                Some(serde_json::json!({"receipt_audit": {"failures": ["wrong logits"]}})),
                2,
            )
            .unwrap();

        // the older release still reads the operator's block, and only it
        assert_eq!(
            read_by_a_release_without_receipts(directory.path()),
            Some(vec![super::super::peer_key(&by_operator)])
        );

        // and after the upgrade both are in force again
        let reloaded = PeerBlocks::load(directory.path());
        assert!(reloaded.is_blocked(&by_operator, 3));
        assert!(reloaded.is_blocked(&by_audit, 3));
        assert_eq!(
            reloaded.snapshot(3)[&super::super::peer_key(&by_audit)].requested_by,
            Requester::Receipts
        );
    }

    #[test]
    fn a_block_follows_its_requester_between_the_files() {
        let directory = tempfile::tempdir().unwrap();
        let bad = peer();
        let blocks = PeerBlocks::load(directory.path());
        blocks
            .block(&bad, BlockLength::SevenDays, Requester::Receipts, None, 1)
            .unwrap();
        assert_eq!(
            read_by_a_release_without_receipts(directory.path()),
            Some(Vec::new())
        );
        // the operator takes the block over: it is the operator's file's now
        blocks
            .block(&bad, BlockLength::UntilUndone, Requester::Operator, None, 2)
            .unwrap();
        assert_eq!(
            read_by_a_release_without_receipts(directory.path()),
            Some(vec![super::super::peer_key(&bad)])
        );
        let receipts = load_file(&directory.path().join(RECEIPTS_FILE_NAME), 3).unwrap();
        assert!(receipts.blocks.is_empty());
        assert!(PeerBlocks::load(directory.path()).is_blocked(&bad, 3));
    }

    #[test]
    fn receipts_blocks_saved_in_the_operators_file_move_out_on_the_next_change() {
        let directory = tempfile::tempdir().unwrap();
        let (old, new) = (peer(), peer());
        let key = super::super::peer_key(&old);
        std::fs::write(
            directory.path().join(FILE_NAME),
            format!(r#"{{"blocks":{{"{key}":{{"blocked_at_ms":1,"requested_by":"receipts"}}}}}}"#),
        )
        .unwrap();
        let blocks = PeerBlocks::load(directory.path());
        assert!(blocks.is_blocked(&old, 2));
        blocks
            .block(&new, BlockLength::UntilUndone, Requester::Operator, None, 2)
            .unwrap();
        assert_eq!(
            read_by_a_release_without_receipts(directory.path()),
            Some(vec![super::super::peer_key(&new)])
        );
        assert!(PeerBlocks::load(directory.path()).is_blocked(&old, 3));
    }

    #[test]
    fn an_unreadable_receipts_file_is_set_aside_and_keeps_the_operators_blocks() {
        let directory = tempfile::tempdir().unwrap();
        let bad = peer();
        PeerBlocks::load(directory.path())
            .block(&bad, BlockLength::UntilUndone, Requester::Operator, None, 1)
            .unwrap();
        std::fs::write(directory.path().join(RECEIPTS_FILE_NAME), b"not json").unwrap();
        let blocks = PeerBlocks::load(directory.path());
        assert_eq!(blocks.not_saved(), None);
        assert!(blocks.is_blocked(&bad, 2));
        assert!(
            std::fs::read_dir(directory.path())
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    name.starts_with("peer_blocks_receipts") && name.contains(".corrupt-")
                })
        );
    }

    #[test]
    fn a_receipts_file_that_cannot_be_read_keeps_the_operators_blocks_unsaved() {
        let directory = tempfile::tempdir().unwrap();
        let bad = peer();
        PeerBlocks::load(directory.path())
            .block(&bad, BlockLength::UntilUndone, Requester::Operator, None, 1)
            .unwrap();
        // a directory where the file should be: present, but not readable
        std::fs::create_dir(directory.path().join(RECEIPTS_FILE_NAME)).unwrap();
        let blocks = PeerBlocks::load(directory.path());
        assert!(blocks.is_blocked(&bad, 2), "the operator's block holds");
        assert!(blocks.not_saved().is_some(), "and nothing is saved");
        // a change this run is kept in memory and never overwrites a file
        let other = peer();
        blocks
            .block(
                &other,
                BlockLength::UntilUndone,
                Requester::Operator,
                None,
                3,
            )
            .unwrap();
        assert!(blocks.is_blocked(&other, 4));
        assert_eq!(
            read_by_a_release_without_receipts(directory.path()),
            Some(vec![super::super::peer_key(&bad)])
        );
    }
}
