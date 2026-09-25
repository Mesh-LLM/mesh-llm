use crate::support::{Outcome, Parity, TestResult, Tool, run, sha256, write};
use std::error::Error;

const NAME: &str = "runtime.tar.gz";
const PAYLOAD: &[u8] = b"runtime";
const FORMAT: &str = "checksum sidecar must use '<sha256>  <archive-name>' format\n";

fn digest() -> String {
    sha256(PAYLOAD)
}

/// `verify-checksum <artifact>` with the artifact at `path` and the sidecar
/// bytes beside it (`None` leaves the sidecar out).
fn verify_at(path: &str, sidecar: Option<&[u8]>) -> Result<Outcome, Box<dyn Error>> {
    let sidecar = sidecar.map(<[u8]>::to_vec);
    run(Tool::Checksum, &[path], Parity::Exact, move |root| {
        let artifact = root.join(path);
        write(root, path, PAYLOAD)?;
        if let Some(bytes) = &sidecar {
            let name = format!("{}.sha256", artifact.to_string_lossy());
            std::fs::write(name, bytes)?;
        }
        Ok(())
    })
}

fn verify(sidecar: &str) -> Result<Outcome, Box<dyn Error>> {
    verify_at(NAME, Some(sidecar.as_bytes()))
}

#[test]
fn migration_archives_checksum_accepts_canonical_sidecar() -> TestResult {
    // Given the exact `<sha256>  <name>` line package-release.{sh,ps1} write.
    let digest = digest();
    // Then newline-terminated, CRLF and unterminated spellings all verify.
    verify(&format!("{digest}  {NAME}\n"))?.assert(0, "");
    verify(&format!("{digest}  {NAME}\r\n"))?.assert(0, "");
    verify(&format!("{digest}  {NAME}"))?.assert(0, "");
    Ok(())
}

#[test]
fn migration_archives_checksum_leaves_artifact_and_sidecar_untouched() -> TestResult {
    // Given a valid artifact and sidecar.
    let line = format!("{}  {NAME}\n", digest());
    // When the checksum verifies.
    let outcome = verify(&line)?;
    // Then the command only reads: the tree still holds exactly both inputs.
    outcome.assert(0, "");
    assert_eq!(outcome.tree.len(), 2, "{:?}", outcome.tree);
    assert_eq!(std::fs::read(outcome.path(NAME))?, PAYLOAD);
    let sidecar = outcome.entry("runtime.tar.gz.sha256").unwrap_or_default();
    assert!(sidecar.ends_with(&format!("{line:?}")), "{sidecar}");
    Ok(())
}

#[test]
fn migration_archives_checksum_resolves_sidecar_beside_artifact() -> TestResult {
    let line = format!("{}  {NAME}\n", digest());
    verify_at("dist/runtime.tar.gz", Some(line.as_bytes()))?.assert(0, "");
    let missing = verify_at("./dist//runtime.tar.gz", None)?;
    let expected = "archive checksum sidecar is missing or empty: dist/runtime.tar.gz.sha256\n";
    missing.assert(1, expected);
    Ok(())
}

#[test]
fn migration_archives_checksum_rejects_missing_or_empty_sidecar() -> TestResult {
    let expected = format!("archive checksum sidecar is missing or empty: {NAME}.sha256\n");
    verify_at(NAME, None)?.assert(1, &expected);
    verify("")?.assert(1, &expected);
    Ok(())
}

#[test]
fn migration_archives_checksum_rejects_wrong_name_and_extra_lines() -> TestResult {
    let digest = digest();
    let wrong = verify(&format!("{digest}  wrong-name.tar.gz\n"))?;
    wrong.assert(
        1,
        "checksum sidecar names 'wrong-name.tar.gz', expected 'runtime.tar.gz'\n",
    );
    // A trailing space is part of the recorded name, not a format error.
    let trailing = verify(&format!("{digest}  {NAME} \n"))?;
    trailing.assert(
        1,
        "checksum sidecar names 'runtime.tar.gz ', expected 'runtime.tar.gz'\n",
    );
    let one_line = "checksum sidecar must contain exactly one canonical line\n";
    for contents in [
        format!("{digest}  {NAME}\n{digest}  {NAME}\n"),
        format!("\n{digest}  {NAME}\n"),
        format!("{digest}  {NAME}\n\n"),
    ] {
        verify(&contents)?.assert(1, one_line);
    }
    Ok(())
}

#[test]
fn migration_archives_checksum_rejects_noncanonical_lines() -> TestResult {
    let digest = digest();
    for contents in [
        format!("{digest} {NAME}\n"),
        format!("{digest}\t{NAME}\n"),
        format!("{digest} *{NAME}\n"),
        format!(" {digest}  {NAME}\n"),
        format!("{}  {NAME}\n", digest.to_uppercase()),
        format!("{digest}  dist/{NAME}\n"),
        format!("{}  {NAME}\n", &digest[1..]),
    ] {
        verify(&contents)?.assert(1, FORMAT);
    }
    Ok(())
}

#[test]
fn migration_archives_checksum_rejects_mismatch() -> TestResult {
    let zeros = "0".repeat(64);
    let outcome = verify(&format!("{zeros}  {NAME}\n"))?;
    let expected = format!(
        "archive checksum mismatch: {NAME}\n  expected: {zeros}\n  actual:   {}\n",
        digest()
    );
    outcome.assert(1, &expected);
    Ok(())
}

#[test]
fn migration_archives_checksum_reports_decode_and_io_errors() -> TestResult {
    let outcome = verify_at(NAME, Some(b"\xff\n"))?;
    let expected = "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte\n";
    outcome.assert(1, expected);
    // Given a well-formed sidecar whose artifact is a directory.
    let line = format!("{}  dist\n", digest());
    let directory = run(Tool::Checksum, &["dist"], Parity::Exact, |root| {
        std::fs::create_dir(root.join("dist"))?;
        write(root, "dist.sha256", line.as_bytes())
    })?;
    directory.assert(1, "[Errno 21] Is a directory: 'dist'\n");
    Ok(())
}

#[test]
fn migration_archives_checksum_rejects_bad_usage() -> TestResult {
    let outcome = run(Tool::Checksum, &[], Parity::Status, |_| Ok(()))?;
    assert_eq!(outcome.code, Some(2), "argparse usage status");
    assert!(outcome.stderr.starts_with("usage: "), "{}", outcome.stderr);
    Ok(())
}

const USAGE: &str = "usage: verify-checksum-sidecar.py [-h] artifact\n";

fn usage_error(message: &str) -> String {
    format!("{USAGE}verify-checksum-sidecar.py: error: {message}\n")
}

fn run_args(args: &[&str]) -> Result<Outcome, Box<dyn Error>> {
    run(Tool::Checksum, args, Parity::Exact, |_| Ok(()))
}

#[test]
fn migration_archives_checksum_matches_argparse_usage_errors() -> TestResult {
    for (args, message) in [
        (&[][..], "the following arguments are required: artifact"),
        (
            &["--"][..],
            "the following arguments are required: artifact",
        ),
        (
            &["-x"][..],
            "the following arguments are required: artifact",
        ),
        (&["a", "b", "c"][..], "unrecognized arguments: b c"),
        (&["a", "-x", "b"][..], "unrecognized arguments: -x b"),
        (&["-x", "--", "--", "a"][..], "unrecognized arguments: -x a"),
        (&["a", "--", "-h"][..], "unrecognized arguments: -h"),
        (
            &["--help=1"][..],
            "argument -h/--help: ignored explicit argument '1'",
        ),
        (
            &["-h-"][..],
            "argument -h/--help: ignored explicit argument '-'",
        ),
        (&["-=x"][..], "ambiguous option: -=x could match -h, --help"),
    ] {
        run_args(args)?.assert(2, &usage_error(message));
    }
    Ok(())
}

#[test]
fn migration_archives_checksum_prints_argparse_help() -> TestResult {
    let help = format!(
        "{USAGE}\npositional arguments:\n  artifact\n\noptions:\n  -h, --help  show this help message and exit\n"
    );
    for args in [&["-h"][..], &["--he"], &["a", "-hx"], &["-x", "--help"]] {
        let outcome = run_args(args)?;
        assert_eq!(
            (outcome.code, outcome.stdout.as_str()),
            (Some(0), help.as_str())
        );
        assert_eq!(outcome.stderr, "", "{args:?}");
    }
    Ok(())
}

#[test]
fn migration_archives_checksum_treats_dash_values_as_artifacts() -> TestResult {
    for (args, shown) in [
        (&["-1.5"][..], "-1.5"),
        (&["-"][..], "-"),
        (&["-x y"][..], "-x y"),
        (&["--", "-h"][..], "-h"),
        (&["a", "--"][..], "a"),
    ] {
        let expected = format!("archive checksum sidecar is missing or empty: {shown}.sha256\n");
        run_args(args)?.assert(1, &expected);
    }
    run_args(&["/"])?.assert(1, "PosixPath('/') has an empty name\n");
    Ok(())
}
