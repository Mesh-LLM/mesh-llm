use crate::support::{Built, Outcome, Parity, TestResult, Tool, run, write};
use crate::writers::{TarMember, tar, tar_gz};
use std::error::Error;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

const PREFIX: &str = "unsafe or invalid tar archive: ";

/// `extract-tar <archive> output` over archive bytes written into scratch.
pub(crate) fn extract(archive: Vec<u8>) -> Result<Outcome, Box<dyn Error>> {
    extract_with(archive, |_| Ok(()))
}

pub(crate) fn extract_with(
    archive: Vec<u8>,
    prepare: impl Fn(&Path) -> Built,
) -> Result<Outcome, Box<dyn Error>> {
    run(
        Tool::Tar,
        &["bundle.tar.gz", "output"],
        Parity::Exact,
        move |root| {
            write(root, "bundle.tar.gz", &archive)?;
            prepare(root)
        },
    )
}

pub(crate) fn rejected(message: &str) -> String {
    format!("{PREFIX}{message}\n")
}

fn mode(outcome: &Outcome, relative: &str) -> Result<u32, Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(outcome.path(relative))?;
    Ok(metadata.permissions().mode() & 0o777)
}

#[test]
fn migration_archives_tar_extracts_executable_members() -> TestResult {
    // Given the runtime layout package-native-runtime.sh writes with tar -czf.
    let payload = b"#!/bin/sh\nexit 0\n";
    let archive = tar_gz(&[
        TarMember::dir("runtime", 0o750),
        TarMember::file("runtime/tool", 0o755, payload),
        TarMember::file("runtime/README", 0o640, b"docs"),
    ]);
    // When it is extracted.
    let outcome = extract(archive)?;
    // Then bytes and masked permission bits survive.
    outcome.assert(0, "");
    assert_eq!(std::fs::read(outcome.path("output/runtime/tool"))?, payload);
    assert_eq!(mode(&outcome, "output/runtime/tool")?, 0o755);
    assert_eq!(mode(&outcome, "output/runtime/README")?, 0o640);
    assert_eq!(mode(&outcome, "output/runtime")?, 0o750);
    Ok(())
}

#[test]
fn migration_archives_tar_accepts_root_member_from_tar_dot() -> TestResult {
    // Given `tar -C product -czf archive .`, which emits `.` and `./name`.
    let archive = tar_gz(&[
        TarMember::dir(".", 0o755),
        TarMember::dir("./bin/", 0o755),
        TarMember::file("./mesh-llm", 0o755, b"product"),
        TarMember::file("./bin//tool", 0o700, b"tool"),
    ]);
    let outcome = extract(archive)?;
    outcome.assert(0, "");
    assert_eq!(std::fs::read(outcome.path("output/mesh-llm"))?, b"product");
    assert_eq!(mode(&outcome, "output/bin/tool")?, 0o700);
    Ok(())
}

#[test]
fn migration_archives_tar_creates_approved_links() -> TestResult {
    // Given a sibling symlink and a hard link to an earlier regular member.
    let archive = tar_gz(&[
        TarMember::file("lib/libmesh.so.1", 0o644, b"elf"),
        TarMember::symlink("lib/libmesh.so", "libmesh.so.1"),
        TarMember::symlink("lib/current", "./libmesh.so.1"),
        TarMember::hardlink("bin/mesh", "lib/libmesh.so.1"),
    ]);
    let outcome = extract(archive)?;
    outcome.assert(0, "");
    let link = std::fs::read_link(outcome.path("output/lib/libmesh.so"))?;
    assert_eq!(link, Path::new("libmesh.so.1"));
    assert!(
        outcome
            .entry("output/bin/mesh")
            .is_some_and(|entry| entry.contains("nlink=2")),
        "{:?}",
        outcome.tree
    );
    Ok(())
}

#[test]
fn migration_archives_tar_reads_uncompressed_pax_and_old_style_members() -> TestResult {
    // Given a plain tar, a pax long path and a v7 directory spelled `name/`.
    let long = format!("deep/{}/file", "d".repeat(120));
    let record = format!(" path={long}\n");
    let pax = format!("{}{record}", record.len() + 3);
    let archive = tar(&[
        TarMember::new("legacy/", b'\0', 0o755),
        TarMember::new("PaxHeader", b'x', 0o644).with_data(pax.as_bytes()),
        TarMember::file("placeholder", 0o600, b"long"),
    ]);
    let outcome = extract(archive)?;
    outcome.assert(0, "");
    assert_eq!(
        std::fs::read(outcome.path(&format!("output/{long}")))?,
        b"long"
    );
    assert!(
        outcome
            .entry("output/legacy")
            .is_some_and(|e| e.starts_with("dir"))
    );
    Ok(())
}

#[test]
fn migration_archives_tar_accepts_empty_archive_and_new_parents() -> TestResult {
    let outcome = run(
        Tool::Tar,
        &["bundle.tar", "a/b/output"],
        Parity::Exact,
        |root| write(root, "bundle.tar", &tar(&[])),
    )?;
    outcome.assert(0, "");
    assert!(outcome.path("a/b/output").is_dir());
    Ok(())
}

#[test]
fn migration_archives_tar_stops_at_a_corrupt_later_header() -> TestResult {
    // Given a valid member followed by a header with a bad checksum, which
    // Python's tarfile treats as the end of the archive.
    let mut archive = tar(&[TarMember::file("kept", 0o644, b"kept")]);
    archive.truncate(1024);
    let mut garbage = TarMember::file("dropped", 0o644, b"").header_bytes();
    garbage[148] = b'7';
    archive.extend_from_slice(&garbage);
    let outcome = extract(archive)?;
    outcome.assert(0, "");
    assert!(outcome.path("output/kept").is_file());
    assert!(!outcome.path("output/dropped").exists());
    Ok(())
}

#[test]
fn migration_archives_tar_reports_unreadable_containers() -> TestResult {
    let listing = |gz: &str, tar: &str| {
        rejected(&format!(
            "file could not be opened successfully:\n\
             - method gz: ReadError('{gz}')\n\
             - method bz2: ReadError('not a bzip2 file')\n\
             - method xz: ReadError('not an lzma file')\n\
             - method tar: ReadError('{tar}')"
        ))
    };
    let junk = extract(b"hello".to_vec())?;
    junk.assert(1, &listing("not a gzip file", "truncated header"));
    let empty = extract(Vec::new())?;
    empty.assert(1, &listing("empty file", "empty file"));
    let mut bad_sum = tar(&[TarMember::file("x", 0o644, b"x")]);
    bad_sum[148] = b'7';
    extract(bad_sum)?.assert(1, &listing("not a gzip file", "bad checksum"));
    let mut bad_number = tar(&[TarMember::file("x", 0o644, b"x")]);
    bad_number[100] = b'9';
    extract(bad_number)?.assert(1, &listing("not a gzip file", "bad checksum"));
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_truncated_payload() -> TestResult {
    let mut archive = tar(&[TarMember::file("big", 0o644, &[b'x'; 1000])]);
    archive.truncate(600);
    extract(archive)?.assert(1, &rejected("unexpected end of data"));
    Ok(())
}

#[test]
fn migration_archives_tar_reports_missing_archive_after_creating_destination() -> TestResult {
    let outcome = run(
        Tool::Tar,
        &["missing.tar.gz", "output"],
        Parity::Exact,
        |_| Ok(()),
    )?;
    let expected = "[Errno 2] No such file or directory: 'missing.tar.gz'";
    outcome.assert(1, &rejected(expected));
    assert!(outcome.path("output").is_dir());
    Ok(())
}

#[test]
fn migration_archives_tar_rejects_bad_usage() -> TestResult {
    let outcome = run(Tool::Tar, &["only-one"], Parity::Status, |_| Ok(()))?;
    assert_eq!(outcome.code, Some(2), "argparse usage status");
    assert!(outcome.stderr.starts_with("usage: "), "{}", outcome.stderr);
    Ok(())
}
