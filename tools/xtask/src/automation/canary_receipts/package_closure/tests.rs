use super::{archive, executable, source, workload};
use crate::automation::canary_receipts::Digest;
use serde_json::json;
use std::{fs, io::Cursor, path::Path};

fn executable_bytes(import: &str) -> Vec<u8> {
    let size = (24 + import.len() + 1).div_ceil(8) * 8;
    let mut bytes = vec![0; 32 + size];
    for (offset, value) in [
        (0, 0xfeedfacf_u32),
        (4, 0x0100000c),
        (12, 2),
        (16, 1),
        (20, u32::try_from(size).unwrap()),
        (32, 0xc),
        (36, u32::try_from(size).unwrap()),
        (40, 24),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[56..56 + import.len()].copy_from_slice(import.as_bytes());
    bytes
}

fn tar_member(output: &mut Vec<u8>, name: &str, bytes: &[u8], mode: u64, modified: u64, kind: u8) {
    let mut header = [0; 512];
    header[..name.len()].copy_from_slice(name.as_bytes());
    for (start, end, value) in [
        (100, 108, mode),
        (124, 136, u64::try_from(bytes.len()).unwrap()),
        (136, 148, modified),
    ] {
        let field = format!("{:0width$o}\0", value, width = end - start - 1);
        header[start..end].copy_from_slice(field.as_bytes());
    }
    header[156] = kind;
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    header[148..156].fill(b' ');
    let checksum: u64 = header.iter().map(|byte| u64::from(*byte)).sum();
    header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
    output.extend_from_slice(&header);
    output.extend_from_slice(bytes);
    output.resize(output.len() + (512 - bytes.len() % 512) % 512, 0);
}

fn write_tar(path: &Path, mut bytes: Vec<u8>) {
    bytes.extend_from_slice(&[0; 1024]);
    fs::write(path, bytes).unwrap();
}

#[test]
fn macho_admits_real_thin_and_single_arch_fat_bytes_and_rejects_foreign_imports() {
    let valid = executable_bytes("/usr/lib/libSystem.B.dylib");
    assert!(
        executable::inspect(
            &mut Cursor::new(&valid),
            0,
            u64::try_from(valid.len()).unwrap()
        )
        .is_ok()
    );
    let mut fat = vec![0; 4096];
    for (at, number) in [
        (0, 0xcafebabe_u32),
        (4, 1),
        (8, 0x0100000c),
        (16, 4096),
        (20, u32::try_from(valid.len()).unwrap()),
    ] {
        fat[at..at + 4].copy_from_slice(&number.to_be_bytes());
    }
    fat.extend_from_slice(&valid);
    assert!(
        executable::inspect(&mut Cursor::new(&fat), 0, u64::try_from(fat.len()).unwrap()).is_ok()
    );
    fat[4..8].copy_from_slice(&2_u32.to_be_bytes());
    assert!(
        executable::inspect(&mut Cursor::new(&fat), 0, u64::try_from(fat.len()).unwrap()).is_err()
    );
    for import in [
        "@rpath/libllama.dylib",
        "/opt/homebrew/lib/libomp.dylib",
        "/usr/lib/../../tmp/foreign.dylib",
    ] {
        let bytes = executable_bytes(import);
        assert!(
            executable::inspect(
                &mut Cursor::new(&bytes),
                0,
                u64::try_from(bytes.len()).unwrap()
            )
            .is_err()
        );
    }
    let mut wrong = valid;
    wrong[4..8].copy_from_slice(&0x01000007_u32.to_le_bytes());
    assert!(
        executable::inspect(
            &mut Cursor::new(&wrong),
            0,
            u64::try_from(wrong.len()).unwrap()
        )
        .is_err()
    );
    wrong[4..8].copy_from_slice(&0x0100000c_u32.to_le_bytes());
    wrong[36..40].copy_from_slice(&0xfffffff8_u32.to_le_bytes());
    assert!(
        executable::inspect(
            &mut Cursor::new(&wrong),
            0,
            u64::try_from(wrong.len()).unwrap()
        )
        .is_err()
    );
}

#[test]
fn binary_archive_requires_exact_regular_executable_member_set() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("binaries.tar");
    let binary = executable_bytes("/System/Library/Frameworks/Metal.framework/Versions/A/Metal");
    let mut bytes = Vec::new();
    for name in archive::BINARIES {
        tar_member(&mut bytes, name, &binary, 0o755, 200, b'0');
    }
    write_tar(&path, bytes.clone());
    assert!(archive::binaries(&path).is_ok());
    for (name, kind, mode) in [
        ("skippy-server", b'0', 0o755),
        ("foreign", b'0', 0o755),
        ("linked", b'2', 0o755),
        ("../escape", b'0', 0o755),
        ("nested//escape", b'0', 0o755),
    ] {
        let mut bad = bytes.clone();
        tar_member(&mut bad, name, &binary, mode, 200, kind);
        write_tar(&path, bad);
        assert!(archive::binaries(&path).is_err());
    }
    let mut no_execute = Vec::new();
    for name in archive::BINARIES {
        tar_member(
            &mut no_execute,
            name,
            &binary,
            if name == "skippy-mm-test" {
                0o644
            } else {
                0o755
            },
            200,
            b'0',
        );
    }
    write_tar(&path, no_execute);
    assert!(archive::binaries(&path).is_err());
    bytes[0] ^= 1;
    write_tar(&path, bytes);
    assert!(archive::binaries(&path).is_err());
}

fn pax_record(key: &str, value: &str) -> Vec<u8> {
    let body = format!(" {key}={value}\n");
    let mut length = body.len() + 1;
    loop {
        let updated = body.len() + length.to_string().len();
        if updated == length {
            break;
        }
        length = updated;
    }
    format!("{length}{body}").into_bytes()
}

#[test]
fn pax_mtime_and_path_are_bound_and_extensions_cannot_escape_or_smuggle_members() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("archive.tar");
    let mut bytes = Vec::new();
    let mut metadata = pax_record("path", "native/bin/llama-server");
    metadata.extend(pax_record("mtime", "100.123456789"));
    tar_member(&mut bytes, "././@PaxHeader", &metadata, 0o644, 0, b'x');
    tar_member(
        &mut bytes,
        "ignored",
        b"real member bytes",
        0o755,
        100,
        b'0',
    );
    write_tar(&path, bytes);
    let members = archive::scan(&mut fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(members["native/bin/llama-server"].modified, 100_123_456_789);
    assert_eq!(
        members["native/bin/llama-server"].sha256,
        Digest::of_bytes(b"real member bytes")
    );
    for metadata in [
        pax_record("path", "../escape"),
        pax_record("SCHILY.xattr.hidden", "unsafe"),
        b"999 mtime=1\n".to_vec(),
    ] {
        let mut bytes = Vec::new();
        tar_member(&mut bytes, "PaxHeader", &metadata, 0o644, 0, b'x');
        tar_member(&mut bytes, "member", b"x", 0o755, 1, b'0');
        write_tar(&path, bytes);
        assert!(archive::scan(&mut fs::File::open(&path).unwrap()).is_err());
    }
}

pub(super) fn workload_tar(
    candidate: &str,
    native: &str,
    candidate_time: u64,
    mutate: impl FnOnce(&mut serde_json::Value),
) -> Vec<u8> {
    let binary = executable_bytes("/usr/lib/libSystem.B.dylib");
    let stamp = format!("stamp-version=3\npatched-sha={native}\nbackend=cpu\nlink-mode=static\n")
        .into_bytes();
    let mut producer = json!({"schema_version":1,"source":{"head":candidate,"worktree_sha256":Digest::of_bytes(b"")},"files":{}});
    let records = [
        ("candidate", "cargo/debug/skippy"),
        ("test_binary", "cargo/debug/deps/skippy_serving-fixture"),
        ("model_package", "cargo/debug/skippy-package-builder"),
        ("correctness", "cargo/debug/skippy-correctness"),
        ("topology_plan", "cargo/debug/skippy-topology-plan"),
        ("native_stamp", "native/.mesh-llm-build-stamp"),
        ("llama-server", "native/bin/llama-server"),
        ("llama-completion", "native/bin/llama-completion"),
        ("llama-tts", "native/bin/llama-tts"),
    ];
    let mut bytes = Vec::new();
    for (key, name) in records {
        let data = if key == "native_stamp" {
            &stamp
        } else {
            &binary
        };
        producer["files"][key] = json!({"path":name,"sha256":Digest::of_bytes(data)});
        tar_member(
            &mut bytes,
            name,
            data,
            0o755,
            if key == "native_stamp" {
                100
            } else {
                candidate_time
            },
            b'0',
        );
    }
    mutate(&mut producer);
    tar_member(
        &mut bytes,
        "producer.json",
        &serde_json::to_vec(&producer).unwrap(),
        0o644,
        200,
        b'0',
    );
    bytes
}

#[test]
fn workload_archive_binds_candidate_native_stamp_full_closure_and_strict_freshness() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("workloads.tar");
    let candidate = "a".repeat(40);
    let native = "b".repeat(40);
    write_tar(&path, workload_tar(&candidate, &native, 200, |_| {}));
    assert!(workload::archive(&path, &candidate, &native).is_ok());
    assert!(workload::archive(&path, &"c".repeat(40), &native).is_err());
    assert!(workload::archive(&path, &candidate, &"d".repeat(40)).is_err());
    write_tar(&path, workload_tar(&candidate, &native, 100, |_| {}));
    assert!(workload::archive(&path, &candidate, &native).is_err());
    for mutate in [0, 1, 2, 3] {
        write_tar(
            &path,
            workload_tar(&candidate, &native, 200, |producer| match mutate {
                0 => {
                    producer["source"]["worktree_sha256"] = json!(Digest::of_bytes(b"dirty source"))
                }
                1 => {
                    producer["files"]["candidate"]["sha256"] =
                        json!(Digest::of_bytes(b"replaced binary"))
                }
                2 => {
                    producer["files"]
                        .as_object_mut()
                        .unwrap()
                        .remove("llama-tts");
                }
                _ => producer["files"]["test_binary"]["path"] = json!("cargo/debug/skippy"),
            }),
        );
        assert!(workload::archive(&path, &candidate, &native).is_err());
    }
}

#[test]
fn prepared_recipe_uses_ordered_lane_names_and_rejects_orphans_and_stale_markers() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let patches = root.join("third_party/llama.cpp/patches");
    fs::create_dir_all(patches.join("model_support")).unwrap();
    fs::create_dir(patches.join("generated")).unwrap();
    fs::write(patches.join("0001-core.patch"), b"core").unwrap();
    fs::write(patches.join("model_support/series"), "0001-qwen.patch\n").unwrap();
    fs::write(patches.join("model_support/0001-qwen.patch"), b"model").unwrap();
    fs::write(patches.join("generated/series"), "0001-family-qwen.patch\n").unwrap();
    fs::write(patches.join("generated/0001-family-qwen.patch"), b"graph").unwrap();
    let transcript = format!(
        "0001-core.patch\n{}\nmodel_support/0001-qwen.patch\n{}\ngenerated/0001-family-qwen.patch\n{}\n",
        Digest::of_bytes(b"core").as_str(),
        Digest::of_bytes(b"model").as_str(),
        Digest::of_bytes(b"graph").as_str()
    );
    let expected = Digest::of_bytes(transcript.as_bytes());
    assert_eq!(source::patch_digest(&patches).unwrap(), expected);
    let upstream = "a".repeat(40);
    let head = "b".repeat(40);
    fs::write(root.join("third_party/llama.cpp/upstream.txt"), &upstream).unwrap();
    let mut provenance = source::Provenance {
        head: head.clone(),
        markers: std::collections::BTreeMap::from([
            (".mesh-llm-upstream-sha".into(), upstream),
            (".mesh-llm-patched-sha".into(), head),
            (".mesh-llm-prepare-schema".into(), "5\n".into()),
            (".mesh-llm-patch-digest".into(), expected.as_str().into()),
        ]),
    };
    assert!(source::validate_recipe(root, &provenance).is_ok());
    provenance
        .markers
        .insert(".mesh-llm-prepare-schema".into(), "4".into());
    assert!(source::validate_recipe(root, &provenance).is_err());
    provenance
        .markers
        .insert(".mesh-llm-prepare-schema".into(), "5".into());
    fs::write(patches.join("0001-core.patch"), b"changed core").unwrap();
    assert!(source::validate_recipe(root, &provenance).is_err());
    fs::write(patches.join("model_support/0002-orphan.patch"), b"orphan").unwrap();
    assert!(source::patch_digest(&patches).is_err());
}

#[test]
fn test_build_stream_requires_one_exact_skippy_server_test_artifact() {
    let artifact = json!({"reason":"compiler-artifact","target":{"name":"skippy_server"},"profile":{"test":true},"executable":"/candidate/target/debug/deps/skippy_server-fixture"});
    let one = serde_json::to_vec(&artifact).unwrap();
    assert_eq!(
        executable::test_binary(&one).unwrap(),
        Path::new("/candidate/target/debug/deps/skippy_server-fixture")
    );
    let mut two = one.clone();
    two.push(b'\n');
    two.extend_from_slice(&one);
    assert!(executable::test_binary(&two).is_err());
    assert!(executable::test_binary(b"{\"reason\":\"build-script-executed\"}\n").is_err());
    assert!(executable::test_binary(b"not json\n").is_err());
}
