use crate::command::DynResult;
use serde::Deserialize;
use std::{fs, io::Write, path::Path};

#[derive(Deserialize)]
struct Download {
    path: String,
}

fn verify(output: &[u8], root: &Path) -> DynResult<(std::path::PathBuf, u64)> {
    let root = root.canonicalize()?;
    let text = std::str::from_utf8(output)?;
    let payload = text
        .char_indices()
        .filter(|(_, character)| *character == '{')
        .find_map(|(index, _)| {
            serde_json::Deserializer::from_str(&text[index..])
                .into_iter::<Download>()
                .next()
                .and_then(Result::ok)
        })
        .ok_or("download output has no JSON path payload")?;
    let downloaded = Path::new(&payload.path).canonicalize()?;
    let metadata = fs::metadata(&downloaded)?;
    if !downloaded.starts_with(root)
        || !metadata.is_file()
        || !(1..=64 * 1024 * 1024).contains(&metadata.len())
    {
        return Err("download escaped cache or violates fixture size bound".into());
    }
    Ok((downloaded, metadata.len()))
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let [output, root] = args else {
        return Err(
            "usage: automation hf-xet-smoke <download-output> <isolated-cache-root>".into(),
        );
    };
    let (path, size) = verify(&fs::read(output)?, Path::new(root))?;
    writeln!(
        crate::cli_output::stdout(),
        "Xet portability smoke passed: {} ({size} bytes)",
        path.display()
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepts_payload_after_lifecycle_records() -> DynResult<()> {
        let root = tempfile::tempdir()?;
        let artifact = root.path().join("fixture.gguf");
        fs::write(&artifact, b"GGUF")?;
        let output = format!(
            "{{\"event\":\"startup\"}}\n{}",
            serde_json::to_string(&serde_json::json!({"path":artifact}))?
        );
        assert_eq!(verify(output.as_bytes(), root.path())?.1, 4);
        Ok(())
    }
    #[test]
    fn rejects_external_path_and_empty_fixture() -> DynResult<()> {
        let root = tempfile::tempdir()?;
        let outside = tempfile::NamedTempFile::new()?;
        let output = serde_json::to_vec(&serde_json::json!({"path":outside.path()}))?;
        assert!(verify(&output, root.path()).is_err());
        let empty = root.path().join("empty");
        fs::write(&empty, [])?;
        assert!(
            verify(
                &serde_json::to_vec(&serde_json::json!({"path":empty}))?,
                root.path()
            )
            .is_err()
        );
        Ok(())
    }
}
