use super::{
    host_lock,
    observation::{self, Host},
};

#[test]
fn vm_stat_admission_counts_only_reclaimable_pages_and_preserves_reserve() {
    let text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 30.\nPages purgeable: 500.\nPages wired down: 1000.\nPages occupied by compressor: 2000.\n";
    assert_eq!(observation::available(text).unwrap(), 60 * 16384);
    assert!(observation::available("page size of 4096 bytes\nPages free: 1.\n").is_err());
    assert!(observation::available(&format!("{text}Pages free: 3.\n")).is_err());
    assert_eq!(
        observation::admission(
            Host {
                total: 1000,
                available: 900
            },
            800
        )
        .unwrap(),
        100
    );
    assert!(
        observation::admission(
            Host {
                total: 1000,
                available: 899
            },
            800
        )
        .is_err()
    );
    assert!(
        observation::admission(
            Host {
                total: 1000,
                available: 1000
            },
            901
        )
        .is_err()
    );
    assert!(
        observation::admission(
            Host {
                total: 0,
                available: 0
            },
            1
        )
        .is_err()
    );
}

#[cfg(unix)]
#[test]
fn stable_lock_rejects_symlink_hardlink_and_runner_writable_directory() {
    use std::{
        fs,
        os::unix::fs::{MetadataExt, PermissionsExt, symlink},
    };
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("locks");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o750)).unwrap();
    let uid = fs::metadata(&root).unwrap().uid();
    let leaf = root.join("mesh-canary-family-host.lock");
    fs::write(&leaf, b"").unwrap();
    fs::set_permissions(&leaf, fs::Permissions::from_mode(0o666)).unwrap();
    assert!(host_lock::open(&root, uid).is_ok());
    let alias = directory.path().join("alias");
    symlink(&root, &alias).unwrap();
    assert!(host_lock::open(&alias, uid).is_err());
    let second = root.join("hard-link");
    fs::hard_link(&leaf, &second).unwrap();
    assert!(host_lock::open(&root, uid).is_err());
    fs::remove_file(second).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o770)).unwrap();
    assert!(host_lock::open(&root, uid).is_err());
    fs::set_permissions(&root, fs::Permissions::from_mode(0o750)).unwrap();
    fs::remove_file(&leaf).unwrap();
    symlink(directory.path().join("external"), &leaf).unwrap();
    assert!(host_lock::open(&root, uid).is_err());
}
