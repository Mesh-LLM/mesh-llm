use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::Path;

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let grammar = Grammar {
        usage: "cargo xtool automation replay-matrix verify-digest --file <path> --sha256 <digest>",
        values: &["--file", "--sha256"],
        flags: &["--help"],
    };
    let parsed = match grammar.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("usage: {}\n", grammar.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return grammar.error("unexpected positional arguments").emit();
    }
    let path = parsed.last("--file").ok_or("missing --file")?;
    let expected = parsed.last("--sha256").ok_or("missing --sha256")?;
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err("expected SHA-256 must be 64 lowercase hexadecimal characters".into());
    }
    let actual =
        crate::product::digest::file_sha256(Path::new(path)).map_err(|error| error.error)?;
    if actual != expected {
        return Err("replay input SHA-256 mismatch".into());
    }
    Ok(())
}
