use super::{
    archive::{self, Root},
    git::Git,
    snapshot,
};
use flate2::read::GzDecoder;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    os::unix::{ffi::OsStrExt, fs::symlink},
    path::{Path, PathBuf},
};

struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    base: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap().join("candidate");
        fs::create_dir(&root).unwrap();
        git(&root, &["init", "-q"]);
        fs::write(root.join("tracked.txt"), b"base\n").unwrap();
        fs::write(root.join("binary.dat"), b"\0base\xff").unwrap();
        fs::write(root.join(".gitignore"), b".deps/\n").unwrap();
        git(&root, &["add", "."]);
        commit(&root);
        let base = git(&root, &["rev-parse", "HEAD"]);
        Self {
            _temp: temp,
            root,
            base,
        }
    }
}
fn git(root: &Path, args: &[&str]) -> String {
    Git::new().text(root, args).unwrap()
}
fn commit(root: &Path) {
    git(
        root,
        &[
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
    );
}
fn manifest(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap()
}
fn members(path: &Path) -> BTreeMap<PathBuf, (Vec<u8>, Option<PathBuf>)> {
    let mut archive = tar::Archive::new(GzDecoder::new(fs::File::open(path).unwrap()));
    archive
        .entries()
        .unwrap()
        .map(|entry| {
            let mut entry = entry.unwrap();
            let name = entry.path().unwrap().into_owned();
            let link = entry.link_name().unwrap().map(|p| p.into_owned());
            let mut body = Vec::new();
            entry.read_to_end(&mut body).unwrap();
            (name, (body, link))
        })
        .collect()
}

#[test]
fn recovery_captures_committed_staged_unstaged_binary_and_new_source_without_certification() {
    let fixture = Fixture::new();
    fs::write(fixture.root.join("tracked.txt"), b"committed repair\n").unwrap();
    git(&fixture.root, &["add", "tracked.txt"]);
    commit(&fixture.root);
    fs::write(fixture.root.join("tracked.txt"), b"staged repair\n").unwrap();
    git(&fixture.root, &["add", "tracked.txt"]);
    fs::write(fixture.root.join("tracked.txt"), b"final unstaged repair\n").unwrap();
    fs::write(fixture.root.join("binary.dat"), b"\0repaired\xff").unwrap();
    fs::write(fixture.root.join("new source.txt"), b"new source\n").unwrap();
    let output = fixture.root.join("recovery");
    snapshot::save(&fixture.root, &output, &fixture.base).unwrap();
    let report = manifest(&output);
    assert_eq!(report["base"], fixture.base);
    assert_ne!(report["head"], fixture.base);
    assert_eq!(report["verified"], false);
    assert_eq!(report["untracked_files"], 1);
    assert!(report["tracked_patch_bytes"].as_u64().unwrap() > 0);
    let restored = fixture.root.parent().unwrap().join("restored");
    git(
        fixture.root.parent().unwrap(),
        &[
            "clone",
            "-q",
            fixture.root.to_str().unwrap(),
            restored.to_str().unwrap(),
        ],
    );
    git(&restored, &["reset", "--hard", &fixture.base]);
    git(
        &restored,
        &[
            "apply",
            "--binary",
            output.join("tracked.patch").to_str().unwrap(),
        ],
    );
    assert_eq!(
        fs::read(restored.join("tracked.txt")).unwrap(),
        fs::read(fixture.root.join("tracked.txt")).unwrap()
    );
    assert_eq!(
        fs::read(restored.join("binary.dat")).unwrap(),
        fs::read(fixture.root.join("binary.dat")).unwrap()
    );
    assert_eq!(
        members(&output.join("untracked.tar.gz"))[Path::new("new source.txt")].0,
        b"new source\n"
    );
    assert!(!output.join("identity.json").exists());
}

#[test]
fn recovery_preserves_literal_external_dangling_links_and_platform_filenames_without_target_reads()
{
    let fixture = Fixture::new();
    let outside = fixture.root.parent().unwrap().join("outside");
    fs::write(&outside, b"unrelated private bytes").unwrap();
    symlink(&outside, fixture.root.join("external link")).unwrap();
    symlink("missing", fixture.root.join("dangling")).unwrap();
    #[cfg(target_os = "linux")]
    let raw = std::ffi::OsStr::from_bytes(b"nonutf8-\xff");
    #[cfg(not(target_os = "linux"))]
    let raw = std::ffi::OsStr::new("Unicode source é");
    fs::write(fixture.root.join(raw), b"raw filename").unwrap();
    let output = fixture.root.join("recovery");
    snapshot::save(&fixture.root, &output, &fixture.base).unwrap();
    let entries = members(&output.join("untracked.tar.gz"));
    assert_eq!(
        entries[Path::new("external link")],
        (Vec::new(), Some(outside.clone()))
    );
    assert_eq!(
        entries[Path::new("dangling")],
        (Vec::new(), Some("missing".into()))
    );
    assert_eq!(entries[Path::new(raw)].0, b"raw filename");
    assert_eq!(fs::read(outside).unwrap(), b"unrelated private bytes");
    assert_eq!(manifest(&output)["untracked_files"], 3);
}

#[test]
fn recovery_captures_dirty_prepared_checkout_against_its_own_head() {
    let fixture = Fixture::new();
    let nested = fixture.root.join(".deps/llama.cpp");
    fs::create_dir_all(&nested).unwrap();
    git(&nested, &["init", "-q"]);
    fs::write(nested.join("runtime.cpp"), b"upstream").unwrap();
    git(&nested, &["add", "."]);
    commit(&nested);
    let base = git(&nested, &["rev-parse", "HEAD"]);
    fs::write(nested.join("runtime.cpp"), b"dirty repair").unwrap();
    fs::write(nested.join("new.cpp"), b"new repair").unwrap();
    let output = fixture.root.join("recovery");
    snapshot::save(&fixture.root, &output, &fixture.base).unwrap();
    let report = manifest(&output);
    assert_eq!(report["prepared_llama_cpp"]["base"], base);
    assert_eq!(report["prepared_llama_cpp"]["untracked_files"], 1);
    let patch = fs::read(output.join("prepared-llama-cpp/tracked.patch")).unwrap();
    assert!(String::from_utf8_lossy(&patch).contains("dirty repair"));
    assert_eq!(
        members(&output.join("prepared-llama-cpp/untracked.tar.gz"))[Path::new("new.cpp")].0,
        b"new repair"
    );
}

#[test]
fn recovery_does_not_execute_candidate_git_fsmonitor_external_diff_or_textconv() {
    let fixture = Fixture::new();
    let marker = fixture.root.parent().unwrap().join("must-not-execute");
    let command = format!("touch '{}'", marker.display());
    for key in [
        "core.fsmonitor",
        "diff.external",
        "diff.diagnostic.textconv",
    ] {
        git(&fixture.root, &["config", key, &command]);
    }
    fs::write(
        fixture.root.join(".gitattributes"),
        b"*.txt diff=diagnostic\n",
    )
    .unwrap();
    fs::write(fixture.root.join("tracked.txt"), b"repair").unwrap();
    snapshot::save(&fixture.root, &fixture.root.join("recovery"), &fixture.base).unwrap();
    assert!(!marker.exists());
}

#[test]
fn recovery_refuses_bad_base_and_existing_destination_without_complete_output() {
    for mutation in ["bad-base", "existing"] {
        let fixture = Fixture::new();
        let output = fixture.root.join("recovery");
        let base = if mutation == "bad-base" {
            "0000000000000000000000000000000000000000"
        } else {
            &fixture.base
        };
        if mutation == "existing" {
            fs::create_dir(&output).unwrap();
            fs::write(output.join("sentinel"), b"keep").unwrap();
        }
        assert!(snapshot::save(&fixture.root, &output, base).is_err());
        assert!(!output.join("manifest.json").exists());
        if mutation == "existing" {
            assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"keep");
        } else {
            assert!(!output.exists());
        }
        assert!(!fs::read_dir(&fixture.root).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .as_bytes()
                .starts_with(b".canary-recovery-")
        }));
    }
}

#[test]
fn recovery_archive_refuses_a_selected_nonregular_member() {
    let fixture = Fixture::new();
    let path = std::ffi::CString::new(fixture.root.join("fifo").as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    let root = Root::open(&fixture.root).unwrap();
    let output = fs::File::create(fixture.root.join("archive.gz")).unwrap();
    let error = archive::write(&root, &[b"fifo"], output, &Git::new()).unwrap_err();
    assert!(error.to_string().contains("nonregular untracked source"));
}

#[test]
fn recovery_tar_paths_preserve_raw_bytes_without_filesystem_encoding_assumptions() {
    let name = Path::new(std::ffi::OsStr::from_bytes(b"nonutf8-\xff"));
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(3);
    header.set_mode(0o600);
    builder.append_data(&mut header, name, &b"raw"[..]).unwrap();
    let bytes = builder.into_inner().unwrap();
    let mut archive = tar::Archive::new(&bytes[..]);
    let mut entries = archive.entries().unwrap();
    let mut entry = entries.next().unwrap().unwrap();
    assert_eq!(entry.path_bytes().as_ref(), b"nonutf8-\xff");
    let mut body = Vec::new();
    entry.read_to_end(&mut body).unwrap();
    assert_eq!(body, b"raw");
}

#[test]
fn recovery_descriptor_traversal_refuses_parent_links_and_detects_directory_substitution() {
    let fixture = Fixture::new();
    let admitted = Root::open(&fixture.root).unwrap();
    symlink(
        fixture.root.parent().unwrap(),
        fixture.root.join("redirect"),
    )
    .unwrap();
    assert!(admitted.file(Path::new("redirect/outside")).is_err());
    let moved = fixture.root.with_file_name("old-candidate");
    fs::rename(&fixture.root, &moved).unwrap();
    fs::create_dir(&fixture.root).unwrap();
    assert!(admitted.unchanged(&fixture.root).is_err());
}

#[test]
fn recovery_sparse_oversized_untracked_file_refuses_before_body_copy() {
    let fixture = Fixture::new();
    let file = fs::File::create(fixture.root.join("large")).unwrap();
    file.set_len(1024 * 1024 * 1024 + 1).unwrap();
    let output = fixture.root.join("recovery");
    assert!(snapshot::save(&fixture.root, &output, &fixture.base).is_err());
    assert!(!output.exists());
}

#[test]
fn recovery_archive_rejects_escaping_member_names() {
    let fixture = Fixture::new();
    let root = Root::open(&fixture.root).unwrap();
    assert!(
        archive::write(
            &root,
            &[b"../outside"],
            root.create_file(Path::new("rejected.tar.gz")).unwrap(),
            &Git::new()
        )
        .is_err()
    );
}

#[test]
fn recovery_atomic_publication_refuses_existing_destination_without_clobbering() {
    let fixture = Fixture::new();
    let parent = Root::open(&fixture.root).unwrap();
    fs::create_dir(fixture.root.join("stage")).unwrap();
    fs::create_dir(fixture.root.join("destination")).unwrap();
    assert!(
        parent
            .publish(
                std::ffi::OsStr::new("stage"),
                std::ffi::OsStr::new("destination")
            )
            .is_err()
    );
    assert!(fixture.root.join("stage").is_dir());
    assert!(fixture.root.join("destination").is_dir());
}
