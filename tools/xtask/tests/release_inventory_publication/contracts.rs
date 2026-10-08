use super::{
    dirty, dirty_file,
    observation::Observation,
    provenance,
    publication::{Phase, Prepared},
};
use crate::{process::Cancellation, real_git::Repository};
use std::{fs, io::Write, time::Duration};
fn new_observation() -> Observation {
    Observation::new(Cancellation::default(), Duration::from_secs(60)).unwrap()
}
#[test]
fn release_inventory_publication_real_git_dirty_binary_staged_unstaged_untracked() {
    let mut repo = Repository::new();
    let root = repo.root.clone();
    fs::write(repo.root.join("tracked binary"), b"initial\0binary\xff").unwrap();
    repo.ok(&["add", "--", "tracked binary"]);
    repo.main_commit("source: tracked binary");
    repo.ok(&["tag", "-f", "v1.0.0", "HEAD"]);
    let requested = repo.ok(&["rev-parse", "HEAD"]);
    repo.main_commit("feature: working HEAD advances");
    let source = provenance::freeze(&mut repo, "v1.0.0", "v1.0.0").unwrap();
    assert_ne!(source.candidate_sha, source.working_tree_head);
    assert_eq!(source.candidate_sha, requested);
    fs::write(repo.root.join("tracked binary"), b"staged\0binary\xff").unwrap();
    repo.ok(&["add", "--", "tracked binary"]);
    fs::write(repo.root.join("tracked binary"), b"unstaged\0binary\xfe").unwrap();
    fs::write(repo.root.join("untracked space"), b"untracked\0\xff\xfe").unwrap();
    let observation = new_observation();
    let first = dirty::capture(&mut repo, &root, &source, &observation, None).unwrap();
    assert!(first.is_dirty);
    assert_eq!(first.diff_base, source.working_tree_head);
    assert_ne!(first.diff_base, source.candidate_sha);
    assert!(first.observation_scope.contains("not candidate or build"));
    assert!(
        first.staged_against_head.bytes > 0
            && first.unstaged.bytes > 0
            && first.worktree_against_head.bytes > 0
    );
    assert!(
        first
            .status_entries
            .iter()
            .any(|s| s.index == "M" && s.worktree == "M" && s.path == "tracked binary")
    );
    assert_eq!(first.untracked_files[0].path, "untracked space");
    assert_eq!(first.untracked_files[0].bytes, 12);
    dirty::revalidate(&first, &mut repo, &root, &source, &observation, None).unwrap();
    fs::write(repo.root.join("untracked space"), b"untracked\0\xff\xfd").unwrap();
    assert!(dirty::revalidate(&first, &mut repo, &root, &source, &observation, None).is_err());
}

#[test]
fn release_inventory_publication_real_git_large_untracked_stream_has_no_tiny_file_ceiling() {
    let mut repo = Repository::new();
    let root = repo.root.clone();
    let source = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    let mut file = fs::File::create(repo.root.join("large regular evidence")).unwrap();
    let chunk = vec![0xa5u8; 1024 * 1024];
    for _ in 0..70 {
        file.write_all(&chunk).unwrap();
    }
    file.sync_all().unwrap();
    drop(file);
    let snapshot = dirty::capture(&mut repo, &root, &source, &new_observation(), None).unwrap();
    assert_eq!(snapshot.untracked_files[0].bytes, 70 * 1024 * 1024);
    assert_eq!(snapshot.untracked_files[0].kind, "regular");
    assert_eq!(snapshot.untracked_files[0].sha256.len(), 64);
}
#[cfg(unix)]
#[test]
fn release_inventory_publication_real_git_symlink_targets_and_unusual_status_paths() {
    use std::os::unix::fs::symlink;
    let mut repo = Repository::new();
    let root = repo.root.clone();
    let source = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::write(
        outside.path().join("private"),
        b"outside bytes never consumed",
    )
    .unwrap();
    let name = "link\twith\nnewline";
    symlink(outside.path().join("private"), repo.root.join(name)).unwrap();
    let first = dirty::capture(&mut repo, &root, &source, &new_observation(), None).unwrap();
    assert_eq!(first.untracked_files[0].path, name);
    assert_eq!(first.untracked_files[0].kind, "symlink");
    assert_eq!(first.status_entries[0].path, name);
    fs::write(
        outside.path().join("private"),
        b"outside changes are not link-target evidence",
    )
    .unwrap();
    let second = dirty::capture(&mut repo, &root, &source, &new_observation(), None).unwrap();
    assert_eq!(first.evidence_sha256, second.evidence_sha256);
    fs::remove_file(repo.root.join(name)).unwrap();
    symlink("different target", repo.root.join(name)).unwrap();
    assert!(
        dirty::revalidate(&first, &mut repo, &root, &source, &new_observation(), None).is_err()
    );
}
#[cfg(unix)]
#[test]
fn release_inventory_publication_special_fifo_and_parent_symlink_refuse_without_reading() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink},
        time::Instant,
    };
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("secret"), b"never open outside source").unwrap();
    symlink(outside.path(), root.path().join("escape")).unwrap();
    assert!(dirty_file::content(root.path(), "escape/secret", &new_observation()).is_err());
    let fifo = root.path().join("fifo");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // SAFETY: creates only the finite private fixture's FIFO with a terminated path.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let start = Instant::now();
    assert!(dirty_file::content(root.path(), "fifo", &new_observation()).is_err());
    assert!(start.elapsed() < Duration::from_secs(1));
}
#[test]
fn release_inventory_publication_regular_path_swap_growth_and_cancel_refuse() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source");
    fs::write(&path, vec![1u8; 2 * 1024 * 1024]).unwrap();
    let mut checks = 0;
    let failure = dirty_file::content_checked(root.path(), "source", &mut || {
        checks += 1;
        if checks == 2 {
            fs::rename(&path, root.path().join("original")).unwrap();
            fs::write(&path, vec![2u8; 2 * 1024 * 1024]).unwrap();
        }
        Ok(())
    });
    assert!(failure.is_err());
    let mut checks = 0;
    assert!(
        dirty_file::content_checked(root.path(), "source", &mut || {
            checks += 1;
            if checks == 2 {
                fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .unwrap()
                    .write_all(b"growth")
                    .unwrap();
            }
            Ok(())
        })
        .is_err()
    );
    let observation = new_observation();
    let mut checks = 0;
    assert!(
        dirty_file::content_checked(root.path(), "source", &mut || {
            checks += 1;
            if checks == 2 {
                observation.cancellation.cancel();
            }
            observation.check()
        })
        .is_err()
    );
    assert!(dirty_file::content(root.path(), "../outside", &new_observation()).is_err());
}
#[test]
fn release_inventory_publication_stage_old_output_in_worktree_revalidates_only_owned_scratch() {
    let mut repo = Repository::new();
    let root = repo.root.clone();
    let output = root.join("inventory.json");
    fs::write(&output, b"old operator report").unwrap();
    let source = provenance::freeze(&mut repo, "v1.0.0", "HEAD").unwrap();
    let observation = new_observation();
    let snapshot = dirty::capture(&mut repo, &root, &source, &observation, None).unwrap();
    let prepared = Prepared::stage(&output, b"complete new report\n", &observation).unwrap();
    prepared
        .publish(&observation, |phase, prepared| match phase {
            Phase::BeforeReplacement => dirty::revalidate(
                &snapshot,
                &mut repo,
                &root,
                &source,
                &observation,
                Some(prepared),
            ),
            Phase::AfterReplacement => observation.check(),
        })
        .unwrap();
    assert_eq!(fs::read(&output).unwrap(), b"complete new report\n");
    assert_eq!(
        fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e
                .file_name()
                .to_string_lossy()
                .starts_with(".release-inventory-"))
            .count(),
        0
    );
}
#[test]
fn release_inventory_publication_failure_or_final_cancel_restores_prior_or_absent_output() {
    let root = tempfile::tempdir().unwrap();
    for prior in [Some(b"prior bytes".as_slice()), None] {
        let output = root.path().join("report.json");
        if let Some(bytes) = prior {
            fs::write(&output, bytes).unwrap();
        } else {
            let _ = fs::remove_file(&output);
        }
        let observation = new_observation();
        let prepared = Prepared::stage(&output, b"new complete JSON", &observation).unwrap();
        let result = prepared.publish(&observation, |phase, _| match phase {
            Phase::BeforeReplacement => Ok(()),
            Phase::AfterReplacement => {
                observation.cancellation.cancel();
                observation.check()
            }
        });
        assert!(result.is_err());
        assert_eq!(fs::read(&output).ok().as_deref(), prior);
    }
    let output = root.path().join("unchanged.json");
    fs::write(&output, b"original").unwrap();
    let observation = new_observation();
    let staged = Prepared::stage(&output, b"new", &observation).unwrap();
    assert!(
        staged
            .publish(&observation, |_, _| Err(provenance::Error(
                "source changed".into()
            )))
            .is_err()
    );
    assert_eq!(fs::read(&output).unwrap(), b"original");
}
#[cfg(unix)]
#[test]
fn release_inventory_publication_output_symlink_and_concurrent_writer_preserved() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let victim = root.path().join("victim");
    fs::write(&victim, b"victim original").unwrap();
    let link = root.path().join("report");
    symlink(&victim, &link).unwrap();
    assert!(Prepared::stage(&link, b"new", &new_observation()).is_err());
    assert_eq!(fs::read(&victim).unwrap(), b"victim original");
    fs::remove_file(&link).unwrap();
    fs::write(&link, b"old").unwrap();
    let observation = new_observation();
    let staged = Prepared::stage(&link, b"new", &observation).unwrap();
    fs::write(&link, b"other writer").unwrap();
    assert!(staged.publish(&observation, |_, _| Ok(())).is_err());
    assert_eq!(fs::read(&link).unwrap(), b"other writer");
}
