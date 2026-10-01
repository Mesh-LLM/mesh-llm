use super::TestResult;
use std::process::Command;

#[test]
fn approved_contract_rejects_when_invalid_suffix_crosses_read_ahead_boundaries() -> TestResult {
    let directory = tempfile::tempdir()?;
    for (kind, prefix) in [
        ("workspace", &b"[workspace.package]\nversion=\"x\"\n"[..]),
        (
            "abi",
            &b"pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_PATCH: u32 = 3;\n"[..],
        ),
    ] {
        for padding in [
            0,
            1,
            4096,
            8191_usize.saturating_sub(prefix.len()),
            8192_usize.saturating_sub(prefix.len()),
            8193_usize.saturating_sub(prefix.len()),
            8192,
            8193,
            16384,
        ] {
            let mut source = prefix.to_vec();
            source.extend(std::iter::repeat_n(b'\n', padding));
            source.push(0xff);
            let path = directory.path().join(format!("{kind}-{padding}.source"));
            std::fs::write(&path, &source)?;

            let actual = Command::new(env!("CARGO_BIN_EXE_xtask"))
                .args(["native", "package-source-version", kind])
                .arg(&path)
                .output()?;

            assert_eq!(actual.status.code(), Some(1), "{kind} padding={padding}");
            assert!(actual.stdout.is_empty(), "{kind} padding={padding}");
        }
    }
    Ok(())
}

const VALID_SOURCES: [(&str, &[u8], &[u8]); 2] = [
    (
        "workspace",
        include_bytes!("../../../../Cargo.toml"),
        b"0.76.1\n",
    ),
    (
        "abi",
        include_bytes!("../../../../crates/skippy-ffi/src/lib.rs"),
        b"0.1.64\n",
    ),
];

#[test]
fn approved_contract_preserves_output_when_repository_source_bytes_are_supplied() -> TestResult {
    let directory = tempfile::tempdir()?;
    for (kind, source, stdout) in VALID_SOURCES {
        let path = directory.path().join(format!("{kind}.source"));
        std::fs::write(&path, source)?;

        let actual = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["native", "package-source-version", kind])
            .arg(&path)
            .output()?;

        assert_eq!(actual.status.code(), Some(0), "{kind}");
        assert_eq!(actual.stdout, stdout, "{kind}");
        assert!(actual.stderr.is_empty(), "{kind}");
    }
    Ok(())
}
