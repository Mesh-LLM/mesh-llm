use super::*;
#[cfg(unix)]
use std::path::PathBuf;

// Representative Apple `otool -tvV` archive members, including unrelated text.
const ARM64: &str = "archive.a(api.o):\n(__TEXT,__text) section\n_unrelated_function:\n0000000000000000 ret\n_uniffi_meshllm_ffi_checksum_func_create_client:\n0000000000000010 mov w0, #0xd88e\n0000000000000014 ret\n_uniffi_meshllm_ffi_checksum_method_meshclienthandle_chat:\n0000000000000020 mov w0, #0x8d1b\n0000000000000024 ret\n";
const X86_64: &str = "archive.a(api.o):\n(__TEXT,__text) section\n_uniffi_meshllm_ffi_checksum_func_create_client:\n0000000000000010 movw $0xd88e, %ax\n0000000000000015 retq\n_uniffi_meshllm_ffi_checksum_method_meshclienthandle_chat:\n0000000000000020 movl $0x8d1b, %eax\n0000000000000025 retq\n";
const SWIFT: &str = "// 雪: preserve this UTF-8 text and unrelated numeric literals.\nlet unrelated = 123\r\nif (uniffi_meshllm_ffi_checksum_func_create_client() != 1) {\n    return false\n}\nif (uniffi_meshllm_ffi_checksum_method_meshclienthandle_chat() != 2) {\n    return false\n}\n";

#[test]
fn both_architectures_update_only_api_guard_literals() {
    for disassembly in [ARM64, X86_64] {
        let (actual, count) = rewrite(SWIFT, &constants(disassembly).unwrap()).unwrap();
        assert_eq!(count, 2);
        assert_eq!(
            actual,
            SWIFT
                .replace("() != 1)", "() != 55438)")
                .replace("() != 2)", "() != 36123)")
        );
        assert_eq!(
            rewrite(&actual, &constants(disassembly).unwrap())
                .unwrap()
                .0,
            actual
        );
    }
}

#[test]
fn malformed_missing_duplicate_and_out_of_range_constants_reject() {
    for disassembly in [
        String::new(),
        ARM64.replace("#0xd88e", "#0x10000"),
        ARM64.replace("mov w0", "mov x1"),
        ARM64.replace("0000000000000014 ret", "0000000000000014 b other"),
        format!("{ARM64}{ARM64}"),
        "_uniffi_meshllm_ffi_checksum_func_create_client:\n".into(),
    ] {
        assert!(constants(&disassembly).is_err(), "accepted {disassembly}");
    }
    assert!(return_constant("0001 mov w0, #42", "0005 ret").is_ok());
}

#[test]
fn every_generated_guard_requires_one_native_constant() {
    let values = constants(ARM64).unwrap();
    for swift in [
        "// no checksums".into(),
        SWIFT.replace("checksum_func_create_client", "checksum_func_unknown"),
        SWIFT.replace("() != 1)", "() != 1.0)"),
        SWIFT.replace("() != 1)", "() != 65536)"),
        format!("{SWIFT}{SWIFT}"),
    ] {
        assert!(rewrite(&swift, &values).is_err(), "accepted {swift}");
    }
}

#[test]
fn atomic_update_preserves_permissions_and_refuses_stale_preimage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("generated.swift");
    fs::write(&path, SWIFT).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    }
    let (replacement, _) = rewrite(SWIFT, &constants(ARM64).unwrap()).unwrap();
    atomic_replace(&path, SWIFT, &replacement).unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), replacement);
    let error = atomic_replace(&path, SWIFT, "partial corruption").unwrap_err();
    assert!(error.to_string().contains("changed during"));
    assert_eq!(fs::read_to_string(&path).unwrap(), replacement);
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }
}

#[cfg(unix)]
fn fixture_tool(dir: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    let executable = dir.join("fixture-otool");
    fs::write(&executable, format!("#!/bin/sh\n{body}\n")).unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
    executable
}

#[test]
#[cfg(unix)]
fn real_bounded_process_reads_disassembly_and_checks_actual_tool_arguments() {
    let dir = tempfile::tempdir().unwrap();
    let library = dir.path().join("native library.a");
    fs::write(&library, b"fixture static archive identity").unwrap();
    let tool = fixture_tool(
        dir.path(),
        &format!(
            "test \"$#\" = 2 || exit 11\ntest \"$1\" = -tvV || exit 12\ntest \"${{2##*/}}\" = 'native library.a' || exit 13\ncat <<'ASSEMBLY'\n{ARM64}ASSEMBLY"
        ),
    );
    let actual = disassemble(
        &tool,
        &library,
        &process::Cancellation::default(),
        Duration::from_secs(2),
    )
    .unwrap();
    assert_eq!(actual, ARM64);
    assert_eq!(constants(&actual).unwrap().len(), 2);
    let swift = dir.path().join("generated.swift");
    fs::write(&swift, SWIFT).unwrap();
    assert_eq!(
        synchronize(
            &tool,
            &library,
            &swift,
            &process::Cancellation::default(),
            Duration::from_secs(2)
        )
        .unwrap(),
        2
    );
    assert!(fs::read_to_string(&swift).unwrap().contains("() != 55438)"));
}

#[test]
#[cfg(unix)]
fn nonzero_or_timeout_tool_failure_leaves_swift_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let library = dir.path().join("native.a");
    fs::write(&library, b"archive").unwrap();
    let swift = dir.path().join("generated.swift");
    fs::write(&swift, SWIFT).unwrap();
    for (body, expected) in [("exit 7", "Exited"), ("sleep 10", "Deadline")] {
        let tool = fixture_tool(dir.path(), body);
        let error = synchronize(
            &tool,
            &library,
            &swift,
            &process::Cancellation::default(),
            Duration::from_millis(100),
        )
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(fs::read_to_string(&swift).unwrap(), SWIFT);
    }
}

#[test]
#[cfg(unix)]
fn symlink_destination_is_rejected_without_touching_target() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target.swift");
    let link = dir.path().join("link.swift");
    fs::write(&target, SWIFT).unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    assert!(read_swift(&link).is_err());
    assert_eq!(fs::read_to_string(target).unwrap(), SWIFT);
}

#[test]
#[cfg(unix)]
fn partial_native_map_and_malformed_body_never_publish_a_partial_update() {
    let dir = tempfile::tempdir().unwrap();
    let library = dir.path().join("native.a");
    let swift = dir.path().join("generated.swift");
    fs::write(&library, b"archive").unwrap();
    fs::write(&swift, SWIFT).unwrap();
    let first_only = ARM64
        .split("_uniffi_meshllm_ffi_checksum_method")
        .next()
        .unwrap();
    for body in [
        first_only.to_owned(),
        ARM64.replace("#0x8d1b", "#not_a_number"),
    ] {
        let tool = fixture_tool(dir.path(), &format!("cat <<'ASSEMBLY'\n{body}ASSEMBLY"));
        let result = synchronize(
            &tool,
            &library,
            &swift,
            &process::Cancellation::default(),
            Duration::from_secs(2),
        );
        assert!(result.is_err());
        assert_eq!(fs::read_to_string(&swift).unwrap(), SWIFT);
        assert!(!fs::read_dir(dir.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
    }
}
