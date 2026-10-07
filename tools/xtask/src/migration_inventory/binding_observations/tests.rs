use super::*;

fn write(root: &Path, path: &str, text: &str) {
    let file = root.join(path);
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(file, text).unwrap();
}

fn fixture() -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    for args in [
        vec!["init", "--quiet"],
        vec![
            "-c",
            "user.name=Binding Fixture",
            "-c",
            "user.email=binding@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "fixture",
        ],
    ] {
        assert!(
            run_command(Command::new("git").current_dir(root).args(args))
                .unwrap()
                .status
                .success()
        );
    }
    for ledger in LEDGERS {
        // Intentionally invalid ledgers prove observations bypass acceptance,
        // while pinning the exact bytes rather than parsing or fixing them.
        write(
            root,
            &format!("ci/automation-migration/{ledger}"),
            "historical bytes: do not rewrite\n",
        );
    }
    temp
}

#[test]
fn observations_pin_stale_ledgers_and_current_sources_without_mutation() {
    let temp = fixture();
    let root = temp.path();
    let path = "skippy/scripts/tests/test_fixture.py";
    let text = "# source\nsubprocess.run(['python3', 'probe.py'])\nsubprocess.run(['python3', 'probe.py'])\n";
    write(root, path, text);
    write(
        root,
        "mesh/scripts/tests/test_no_calls.py",
        "# no candidates\n",
    );
    let revision = head(root).unwrap();
    let report = observe(root, Some(&revision)).unwrap();
    assert!(!report.acceptance);
    assert_eq!(report.ledger_preimages.len(), LEDGERS.len());
    assert_eq!(report.candidates.len(), 2);
    assert_eq!(report.candidates[0].line, 2);
    assert_eq!(report.candidates[1].line, 3);
    assert_eq!(report.candidates[0].occurrence, 1);
    assert_eq!(report.candidates[1].occurrence, 2);
    assert_eq!(report.candidates[0].path, path);
    let mut digest = Sha256::new();
    add_tree(&mut digest, path, text.as_bytes());
    assert_eq!(
        report.test_candidate_source_tree_sha256,
        hex::encode(digest.finalize())
    );
    assert_eq!(
        report
            .source_files
            .iter()
            .find(|file| file.path == path)
            .unwrap()
            .split_newline_count,
        4
    );
    for ledger in LEDGERS {
        assert_eq!(
            fs::read_to_string(root.join("ci/automation-migration").join(ledger)).unwrap(),
            "historical bytes: do not rewrite\n"
        );
    }
    assert_eq!(fs::read_to_string(root.join(path)).unwrap(), text);
    assert!(
        observe(root, Some(&"0".repeat(40)))
            .unwrap_err()
            .to_string()
            .contains("HEAD differs")
    );
}

#[test]
fn located_github_rows_preserve_context_and_do_not_bind_skipped_heredoc_text() {
    let path = ".github/workflows/fixture.yml";
    let text = "jobs:\n  fixture:\n    steps:\n      - name: Fixture\n        run: |\n          python3 <<'END'\n          python3 child.py\n          END\n          python3 child.py\n          python3 child.py\n";
    let located = scan::scan_source_located(path, text);
    let old = scan::scan_source(path, text);
    assert_eq!(
        located
            .iter()
            .map(|row| row.candidate.id.as_str())
            .collect::<Vec<_>>(),
        old.iter().map(|row| row.id.as_str()).collect::<Vec<_>>()
    );
    let calls = located
        .iter()
        .filter(|row| row.candidate.source_block == "python3 child.py")
        .collect::<Vec<_>>();
    assert_eq!(
        calls.iter().map(|row| row.line).collect::<Vec<_>>(),
        [9, 10]
    );
    assert!(calls[0].candidate.id.ends_with(":1"));
    assert!(calls[1].candidate.id.ends_with(":2"));
}

#[test]
fn observations_keep_product_identities_distinct_and_hash_source_changes() {
    let temp = fixture();
    let root = temp.path();
    for path in [
        "scripts/fixture.sh",
        "mesh/scripts/fixture.sh",
        "skippy/scripts/fixture.sh",
    ] {
        write(root, path, "python3 child.py\n");
    }
    let before = observe(root, None).unwrap();
    assert_eq!(before.candidates.len(), 3);
    assert_eq!(
        before
            .candidates
            .iter()
            .map(|row| &row.id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3
    );
    write(root, "mesh/scripts/fixture.sh", "python3 changed.py\n");
    let after = observe(root, None).unwrap();
    assert_eq!(before.head, after.head);
    assert_ne!(
        before.scanner_source_tree_sha256,
        after.scanner_source_tree_sha256
    );
}

#[cfg(unix)]
#[test]
fn observations_refuse_symlink_sources_instead_of_pinning_outside_bytes() {
    let temp = fixture();
    let outside = tempfile::tempdir().unwrap();
    write(
        outside.path(),
        "outside.py",
        "subprocess.run(['private'])\n",
    );
    fs::create_dir_all(temp.path().join("scripts")).unwrap();
    std::os::unix::fs::symlink(
        outside.path().join("outside.py"),
        temp.path().join("scripts/probe.py"),
    )
    .unwrap();
    assert!(
        observe(temp.path(), None)
            .unwrap_err()
            .to_string()
            .contains("regular file scripts/probe.py")
    );
}

#[cfg(unix)]
#[test]
fn observation_descriptor_refuses_fifo_without_waiting_for_a_writer() {
    use std::os::unix::ffi::OsStrExt;
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("source.py");
    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // The owning reader must open nonblocking and reject the descriptor type.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(
        read(temp.path(), "source.py")
            .unwrap_err()
            .to_string()
            .contains("regular file source.py")
    );
}
