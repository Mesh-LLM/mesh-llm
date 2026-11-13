use super::{Error, fixtures::Fixture, replay};
use std::os::unix::ffi::OsStrExt;

#[test]
fn nul_paths_when_non_utf8_spaces_and_newlines_are_present_preserve_os_bytes() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};
    let fixture = Fixture::new();
    let root = fixture.temporary.join("agentic-replay-worktrees");
    let owned = root.join(OsString::from_vec(b"space \n\xff".to_vec()));
    let sibling = fixture
        .temporary
        .join("agentic-replay-worktrees-other/keep");
    let mut listing = Vec::new();
    for path in [&root, &owned, &sibling] {
        listing.extend_from_slice(b"worktree ");
        listing.extend_from_slice(path.as_os_str().as_bytes());
        listing.extend_from_slice(b"\0HEAD abc\0\0");
    }
    assert_eq!(
        replay::owned_worktrees(&root, &listing).unwrap(),
        vec![owned]
    );
}

#[test]
fn parent_symlink_when_last_worktree_is_unsafe_preserves_every_sentinel() {
    let fixture = Fixture::new();
    let root = fixture.temporary.join("agentic-replay-worktrees");
    let safe = root.join("safe");
    fixture.seed(&safe);
    let outside = fixture.root.path().join("outside");
    fixture.seed(&outside.join("worktree"));
    let parent = root.join("linked-parent");
    std::os::unix::fs::symlink(&outside, &parent).unwrap();
    let mut listing = Vec::new();
    for path in [&safe, &parent.join("worktree")] {
        listing.extend_from_slice(b"worktree ");
        listing.extend_from_slice(path.as_os_str().as_bytes());
        listing.push(0);
    }
    assert!(matches!(
        replay::owned_worktrees(&root, &listing),
        Err(Error::ParentSymlink(_))
    ));
    assert!(safe.join("payload").exists());
    assert!(outside.join("worktree/payload").exists());
}
