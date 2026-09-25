use crate::support::{Parity, TestResult, Tool, run, write};
use crate::tar_extract::{extract, extract_with, rejected};
use crate::writers::{TarMember, tar_gz};

/// Extracting `members` fails with `message` and leaves `output` empty.
fn refuses(members: &[TarMember], message: &str) -> TestResult {
    let outcome = extract(tar_gz(members))?;
    outcome.assert(1, &rejected(message));
    let written: Vec<_> = outcome
        .tree
        .keys()
        .filter(|path| path.starts_with("output/"))
        .collect();
    assert!(written.is_empty(), "{written:?}");
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_traversal() -> TestResult {
    refuses(
        &[TarMember::file("safe/../../evil", 0o644, b"x")],
        "traversing archive member path is not allowed: safe/../../evil",
    )
}

#[test]
fn migration_archives_tar_rejects_absolute_and_drive_paths() -> TestResult {
    refuses(
        &[TarMember::file("/etc/evil", 0o644, b"x")],
        "absolute archive member path is not allowed: /etc/evil",
    )?;
    refuses(
        &[TarMember::file("C:evil", 0o644, b"x")],
        "absolute archive member path is not allowed: C:evil",
    )
}

#[test]
fn migration_archives_tar_rejects_backslashes_with_repr() -> TestResult {
    refuses(
        &[TarMember::file("dir\\evil", 0o644, b"x")],
        "unsafe archive member path: 'dir\\\\evil'",
    )
}

#[test]
fn migration_archives_tar_rejects_non_directory_root_member() -> TestResult {
    refuses(
        &[TarMember::file("./", 0o644, b"")],
        "empty archive member path is not allowed: './'",
    )
}

#[test]
fn migration_archives_tar_rejects_duplicate_normalized_members() -> TestResult {
    refuses(
        &[
            TarMember::file("dir/file", 0o644, b"one"),
            TarMember::file("./dir//file", 0o644, b"two"),
        ],
        "duplicate archive member path: ./dir//file",
    )
}

#[test]
fn migration_archives_tar_rejects_device_and_fifo_members() -> TestResult {
    refuses(
        &[TarMember::new("dev", b'3', 0o644)],
        "unsupported archive member type for dev: b'3'",
    )?;
    refuses(
        &[TarMember::new("pipe", b'6', 0o644)],
        "unsupported archive member type for pipe: b'6'",
    )
}

#[test]
fn migration_archives_tar_rejects_escaping_links() -> TestResult {
    refuses(
        &[TarMember::symlink("lib/evil", "../../etc/passwd")],
        "traversing link target for lib/evil is not allowed: ../../etc/passwd",
    )?;
    refuses(
        &[TarMember::symlink("evil", "/etc/passwd")],
        "absolute link target for evil is not allowed: /etc/passwd",
    )?;
    refuses(
        &[TarMember::symlink("lib/inner", "./nested/../sibling")],
        "traversing link target for lib/inner is not allowed: ./nested/../sibling",
    )?;
    refuses(
        &[TarMember::hardlink("evil", "../outside")],
        "traversing link target for evil is not allowed: ../outside",
    )
}

#[test]
fn migration_archives_tar_rejects_hardlink_to_directory() -> TestResult {
    let outcome = extract(tar_gz(&[
        TarMember::dir("lib", 0o755),
        TarMember::hardlink("evil", "lib"),
    ]))?;
    outcome.assert(
        1,
        &rejected("hard-link target is not a regular extracted file: lib"),
    );
    assert!(outcome.entry("output/evil").is_none(), "{:?}", outcome.tree);
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_link_over_extracted_path() -> TestResult {
    // Given a file extracted through `link/` before `link` itself is a link.
    let outcome = extract(tar_gz(&[
        TarMember::symlink("link", "target"),
        TarMember::file("link/file", 0o644, b"x"),
        TarMember::dir("target", 0o755),
    ]))?;
    outcome.assert(
        1,
        &rejected("archive link would overwrite an existing path: link"),
    );
    assert!(outcome.path("output/link/file").is_file());
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_nonempty_destination() -> TestResult {
    let outcome = extract_with(tar_gz(&[TarMember::file("a", 0o644, b"a")]), |root| {
        write(root, "output/existing", b"keep")
    })?;
    outcome.assert(
        1,
        &rejected("extraction destination must be empty: <SCRATCH>/output"),
    );
    assert!(outcome.entry("output/a").is_none(), "{:?}", outcome.tree);
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_symlink_destination() -> TestResult {
    let outcome = extract_with(tar_gz(&[TarMember::file("a", 0o644, b"a")]), |root| {
        std::fs::create_dir(root.join("real"))?;
        std::os::unix::fs::symlink("real", root.join("output"))?;
        Ok(())
    })?;
    outcome.assert(
        1,
        &rejected("extraction destination cannot be a symlink: output"),
    );
    assert!(outcome.entry("real/a").is_none(), "{:?}", outcome.tree);
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_dangling_symlink_destination() -> TestResult {
    let outcome = extract_with(tar_gz(&[TarMember::file("a", 0o644, b"a")]), |root| {
        std::os::unix::fs::symlink("missing", root.join("output"))?;
        Ok(())
    })?;
    outcome.assert(1, &rejected("[Errno 17] File exists: 'output'"));
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_destination_under_a_file() -> TestResult {
    let outcome = run(
        Tool::Tar,
        &["bundle.tar.gz", "file/output"],
        Parity::Exact,
        |root| {
            write(root, "bundle.tar.gz", &tar_gz(&[]))?;
            write(root, "file", b"x")
        },
    )?;
    outcome.assert(1, &rejected("[Errno 20] Not a directory: 'file/output'"));
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_member_below_extracted_file() -> TestResult {
    let outcome = extract(tar_gz(&[
        TarMember::file("a", 0o644, b"a"),
        TarMember::file("a/b", 0o644, b"b"),
    ]))?;
    outcome.assert(1, &rejected("[Errno 17] File exists: '<SCRATCH>/output/a'"));
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_unknown_member_type_after_skipping_payload() -> TestResult {
    refuses(
        &[TarMember::new("vendor", b'Z', 0o644).with_data(b"opaque")],
        "unsupported archive member type for vendor: b'Z'",
    )
}
