use crate::command::DynResult;
use serde::Deserialize;
use std::io::Read;

#[derive(Deserialize)]
struct Metadata {
    target_directory: String,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if !args.is_empty() {
        return Err(
            "cargo-target-directory accepts Cargo metadata on stdin, with no arguments".into(),
        );
    }
    let mut bytes = Vec::new();
    std::io::stdin()
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("Cargo metadata exceeds 16 MiB".into());
    }
    let path = project(&bytes)?;
    crate::repository::check_report::CheckReport::success(format!("{path}\n")).emit()
}

fn project(bytes: &[u8]) -> DynResult<String> {
    let metadata: Metadata = serde_json::from_slice(bytes)?;
    if metadata.target_directory.is_empty()
        || metadata.target_directory.contains(['\0', '\n', '\r'])
    {
        return Err("Cargo target_directory must be a non-empty single-line path".into());
    }
    Ok(metadata.target_directory)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn l4_projection_preserves_configured_path_when_spaces_are_present() {
        let metadata =
            br#"{"target_directory":"/configured cargo/target directory","packages":[]}"#;
        let path = project(metadata).unwrap();
        assert_eq!(path, "/configured cargo/target directory");
    }
    #[test]
    fn l4_projection_rejects_wrong_metadata_when_target_is_missing_or_wrong_type() {
        for metadata in [
            b"{}".as_slice(),
            br#"{"target_directory":false}"#,
            br#"{"target_directory":""}"#,
        ] {
            let result = project(metadata);
            assert!(result.is_err());
        }
    }
}
