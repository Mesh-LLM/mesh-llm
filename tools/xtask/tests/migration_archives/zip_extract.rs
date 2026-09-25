use crate::support::{Built, Outcome, Parity, TestResult, Tool, run, write};
use crate::zip_writers::{S_IFDIR, ZipEntry, zip, zip_with};
use std::error::Error;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

const PREFIX: &str = "unsafe ZIP archive: ";

fn extract_as(
    archive: Vec<u8>,
    parity: Parity,
    prepare: impl Fn(&Path) -> Built,
) -> Result<Outcome, Box<dyn Error>> {
    run(Tool::Zip, &["bundle.zip", "output"], parity, move |root| {
        write(root, "bundle.zip", &archive)?;
        prepare(root)
    })
}

fn extract(archive: Vec<u8>) -> Result<Outcome, Box<dyn Error>> {
    extract_as(archive, Parity::Exact, |_| Ok(()))
}

fn rejected(message: &str) -> String {
    format!("{PREFIX}{message}\n")
}

/// Legacy `zipfile` failures are uncaught tracebacks; only the final
/// exception line is compared.
fn raises(archive: Vec<u8>, exception: &str) -> Result<Outcome, Box<dyn Error>> {
    let outcome = extract_as(archive, Parity::LastLine, |_| Ok(()))?;
    outcome.assert(1, &format!("{exception}\n"));
    Ok(outcome)
}

/// Extracting `entries` fails with `message` and writes nothing.
fn refuses(entries: &[ZipEntry], message: &str) -> TestResult {
    let outcome = extract(zip(entries))?;
    outcome.assert(1, &rejected(message));
    let written: Vec<_> = outcome
        .tree
        .keys()
        .filter(|path| path.starts_with("output/"))
        .collect();
    assert!(written.is_empty(), "{written:?}");
    Ok(())
}

fn mode(outcome: &Outcome, relative: &str) -> Result<u32, Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(outcome.path(relative))?;
    Ok(metadata.permissions().mode() & 0o777)
}

/// Rewrites the method (or flag) field of the first local and central header.
fn patch_first(mut archive: Vec<u8>, local: usize, central: usize, value: u16) -> Vec<u8> {
    let start = archive
        .windows(4)
        .position(|window| window == [0x50, 0x4b, 0x01, 0x02])
        .expect("central directory");
    for offset in [local, start + central] {
        archive[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
    archive
}

#[test]
fn migration_archives_zip_extracts_files_dirs_modes_and_links() -> TestResult {
    // Given the framework layout the Swift SDK zip carries.
    let archive = zip(&[
        ZipEntry::raw("Mesh.framework/", (S_IFDIR | 0o700) << 16, b""),
        ZipEntry::file("Mesh.framework/Versions/A/Mesh", 0o755, b"binary").deflated(),
        ZipEntry::file("Mesh.framework/Versions/A/Info.plist", 0o640, b"plist"),
        ZipEntry::raw("plain.txt", 0, b"no mode bits"),
        ZipEntry::symlink("Mesh.framework/Versions/Current", "A"),
        ZipEntry::symlink("Mesh.framework/Mesh", "Versions/Current/Mesh"),
    ]);
    // When it is extracted.
    let outcome = extract(archive)?;
    // Then bytes, masked modes and link targets survive.
    outcome.assert(0, "");
    let binary = "output/Mesh.framework/Versions/A/Mesh";
    assert_eq!(std::fs::read(outcome.path(binary))?, b"binary");
    assert_eq!(mode(&outcome, binary)?, 0o755);
    assert_eq!(
        mode(&outcome, "output/Mesh.framework/Versions/A/Info.plist")?,
        0o640
    );
    assert_eq!(
        outcome.entry("output/Mesh.framework/Mesh"),
        Some("link -> Versions/Current/Mesh")
    );
    assert_eq!(
        std::fs::read(outcome.path("output/Mesh.framework/Mesh"))?,
        b"binary"
    );
    Ok(())
}

#[test]
fn migration_archives_zip_rejects_traversal_and_absolute_names() -> TestResult {
    refuses(
        &[ZipEntry::file("safe/../../evil", 0o644, b"x")],
        "entry escapes the extraction root: 'safe/../../evil'",
    )?;
    refuses(
        &[ZipEntry::file("/etc/evil", 0o644, b"x")],
        "entry is not a portable relative path: '/etc/evil'",
    )?;
    refuses(
        &[ZipEntry::file("C:evil", 0o644, b"x")],
        "entry is not a portable relative path: 'C:evil'",
    )?;
    refuses(
        &[ZipEntry::file("dir\\evil", 0o644, b"x")],
        "entry is not a portable relative path: 'dir\\\\evil'",
    )?;
    refuses(
        &[ZipEntry::raw("./", (S_IFDIR | 0o755) << 16, b"")],
        "entry escapes the extraction root: '.'",
    )
}

#[test]
fn migration_archives_zip_validates_every_entry_before_writing() -> TestResult {
    refuses(
        &[
            ZipEntry::file("good", 0o644, b"x"),
            ZipEntry::file("tab\tname", 0o644, b"x"),
        ],
        "entry is not a portable relative path: 'tab\\tname'",
    )
}

#[test]
fn migration_archives_zip_rejects_duplicates_and_special_entries() -> TestResult {
    refuses(
        &[
            ZipEntry::file("dir/file", 0o644, b"one"),
            ZipEntry::file("dir//./file", 0o644, b"two"),
        ],
        "duplicate entry path: 'dir//./file'",
    )?;
    refuses(
        &[ZipEntry::raw("pipe", 0o010_644 << 16, b"")],
        "unsupported entry type for 'pipe'",
    )
}

#[test]
fn migration_archives_zip_rejects_escaping_or_ancestor_symlinks() -> TestResult {
    refuses(
        &[ZipEntry::symlink("lib/evil", "../../etc/passwd")],
        "symlink target escapes the extraction root: '../../etc/passwd'",
    )?;
    refuses(
        &[ZipEntry::symlink("lib/evil", "/etc/passwd")],
        "symlink target is not portable: '/etc/passwd'",
    )?;
    refuses(
        &[ZipEntry::symlink("lib/root", "..")],
        "symlink target resolves to the extraction root: '..'",
    )?;
    refuses(
        &[
            ZipEntry::symlink("link", "dir"),
            ZipEntry::file("link/file", 0o644, b"x"),
        ],
        "entry is nested beneath an archive symlink: 'link/file'",
    )?;
    refuses(
        &[ZipEntry::raw("bad", 0o120_777 << 16, b"\xff")],
        "symlink target is not UTF-8: 'bad'",
    )
}

#[test]
fn migration_archives_zip_rejects_unsafe_destinations() -> TestResult {
    let archive = zip(&[ZipEntry::file("file", 0o644, b"x")]);
    let outcome = extract_as(archive.clone(), Parity::Exact, |root| {
        write(root, "output/stale", b"old")
    })?;
    outcome.assert(1, &rejected("destination must be empty: output"));
    let outcome = extract_as(archive, Parity::Exact, |root| {
        std::fs::create_dir(root.join("real"))?;
        std::os::unix::fs::symlink("real", root.join("output"))?;
        Ok(())
    })?;
    outcome.assert(1, &rejected("destination cannot be a symlink: output"));
    assert!(outcome.tree.keys().all(|path| !path.starts_with("real/")));
    let archive = zip(&[ZipEntry::file("file", 0o644, b"x")]);
    let outcome = extract_as(archive, Parity::LastLine, |root| {
        write(root, "output", b"not a directory")
    })?;
    outcome.assert(1, "FileExistsError: [Errno 17] File exists: 'output'\n");
    Ok(())
}

#[test]
fn migration_archives_zip_reports_missing_archive_and_usage() -> TestResult {
    let outcome = run(
        Tool::Zip,
        &["./missing.zip", "output"],
        Parity::Exact,
        |_| Ok(()),
    )?;
    outcome.assert(1, &rejected("archive does not exist: missing.zip"));
    assert!(outcome.entry("output").is_none());
    let usage = "usage: scripts/safe-extract-zip.py ARCHIVE.zip DESTINATION\n";
    for args in [&["only-one"][..], &["a", "b", "c"], &["-h"]] {
        run(Tool::Zip, args, Parity::Exact, |_| Ok(()))?.assert(1, usage);
    }
    Ok(())
}

#[test]
fn migration_archives_zip_rejects_non_zip_bytes() -> TestResult {
    let outcome = raises(
        b"hello world".to_vec(),
        "zipfile.BadZipFile: File is not a zip file",
    )?;
    assert_eq!(outcome.entry("output"), Some("dir 755"));
    Ok(())
}

#[test]
fn migration_archives_zip_rejects_bad_crc_after_creating_file() -> TestResult {
    let archive = zip_with(&[ZipEntry::file("a/b.txt", 0o644, b"hi").deflated()], 1);
    let outcome = raises(archive, "zipfile.BadZipFile: Bad CRC-32 for file 'a/b.txt'")?;
    assert!(outcome.entry("output/a/b.txt").is_some());
    let archive = zip_with(&[ZipEntry::symlink("link", "target")], 1);
    let outcome = raises(archive, "zipfile.BadZipFile: Bad CRC-32 for file 'link'")?;
    assert_eq!(outcome.entry("output"), Some("dir 755"));
    Ok(())
}

#[test]
fn migration_archives_zip_rejects_unsupported_method_and_encryption() -> TestResult {
    let archive = zip(&[ZipEntry::file("a/b.txt", 0o644, b"hi")]);
    let outcome = raises(
        patch_first(archive.clone(), 8, 10, 99),
        "NotImplementedError: That compression method is not supported",
    )?;
    assert_eq!(outcome.entry("output/a"), Some("dir 755"));
    assert!(outcome.entry("output/a/b.txt").is_none());
    raises(
        patch_first(archive, 6, 8, 0x0801),
        "RuntimeError: File <ZipInfo filename='a/b.txt' filemode='-rw-r--r--' \
         file_size=2> is encrypted, password required for extraction",
    )?;
    Ok(())
}
