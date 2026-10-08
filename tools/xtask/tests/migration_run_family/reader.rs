use std::{fs, path::PathBuf, process::Command};

#[test]
fn run_family_reader_fixture() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("REPLAY_RUST_READER").is_none() {
        return Ok(());
    }
    let root = std::env::current_dir()?;
    let path = |name| -> Result<PathBuf, Box<dyn std::error::Error>> {
        Ok(root.join(match name {
            "REPLAY_JSON" => "params.json",
            "REPLAY_ENV" => "github.env",
            "REPLAY_FIXTURE_MARKER" => "reader-started.json",
            "REPLAY_MANIFEST_SOURCE" => "manifest.json",
            "REPLAY_MANIFEST_TARGET" => {
                return Ok(std::env::var_os(name)
                    .map(PathBuf::from)
                    .ok_or("missing target")?);
            }
            _ => return Err("unknown fixture path".into()),
        }))
    };
    assert!(path("REPLAY_JSON")?.is_file());
    assert!(path("REPLAY_ENV")?.is_file());
    let marker = path("REPLAY_FIXTURE_MARKER")?;
    let control: (String, i32) =
        serde_json::from_slice(&fs::read(root.join("reader-control.json"))?)?;
    let (mode, status) = control;
    if mode == "hang" {
        let mut descendant = Command::new("/bin/sleep").arg("60").spawn()?;
        fs::write(
            marker,
            serde_json::to_vec(&serde_json::json!({
                "reader":std::process::id(),"descendant":descendant.id()
            }))?,
        )?;
        let _status = descendant.wait()?;
        return Err("hanging reader descendant exited before owned cleanup".into());
    }
    fs::write(
        marker,
        serde_json::to_vec(&serde_json::json!({"reader":std::process::id()}))?,
    )?;

    if status != 0 {
        std::process::exit(status);
    }
    fs::copy(
        path("REPLAY_MANIFEST_SOURCE")?,
        path("REPLAY_MANIFEST_TARGET")?,
    )?;
    Ok(())
}
