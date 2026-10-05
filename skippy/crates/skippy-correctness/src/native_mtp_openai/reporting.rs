use std::path::Path;

use anyhow::{Context, Result};

pub(super) fn emit_report<T: serde::Serialize>(
    report: &T,
    report_out: Option<&Path>,
) -> Result<()> {
    let json = serde_json::to_vec_pretty(report)?;
    if let Some(path) = report_out {
        std::fs::write(path, &json)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}

pub(super) fn status(matches: bool) -> &'static str {
    if matches { "pass" } else { "fail" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_names_match_report_contract() {
        assert_eq!(status(true), "pass");
        assert_eq!(status(false), "fail");
    }
}
