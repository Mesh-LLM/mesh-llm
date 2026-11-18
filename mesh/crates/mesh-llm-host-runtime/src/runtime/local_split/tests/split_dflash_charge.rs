//! What a split's final stage is charged for a DFlash draft it cannot load.

use super::*;

#[cfg(unix)]
#[test]
fn a_configured_draft_is_charged_only_when_it_reaches_the_final_stage() {
    let cache = tempfile::tempdir().unwrap();
    let repo = cache.path().join("models--owner--draft");
    let sha256 = "ab".repeat(32);
    std::fs::create_dir_all(repo.join("blobs")).unwrap();
    std::fs::write(repo.join("blobs").join(&sha256), b"draft bytes").unwrap();
    let snapshot = repo.join("snapshots").join("12".repeat(20));
    std::fs::create_dir_all(&snapshot).unwrap();
    let pinned = snapshot.join("draft.gguf");
    std::os::unix::fs::symlink(format!("../../blobs/{sha256}"), &pinned).unwrap();
    let loose = cache.path().join("draft.gguf");
    std::fs::write(&loose, b"draft bytes").unwrap();

    // The final stage fetches a pinned draft from the cache it names.
    assert_eq!(final_stage_draft_bytes(None, &pinned), Some(11));
    // A draft outside the cache cannot reach another node: the split
    // serves without speculation, so the final stage holds nothing for it.
    assert_eq!(final_stage_draft_bytes(None, &loose), None);
    // A default draft travels by its pin whatever this node has.
    assert_eq!(final_stage_draft_bytes(Some(42), &loose), Some(42));
}
