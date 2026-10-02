use std::{fs, path::PathBuf, process::Command};

#[test]
fn run_family_reader_fixture() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("REPLAY_RUST_READER").is_none() {
        return Ok(());
    }
    let path = |name| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .ok_or("missing fixture path")
    };
    assert!(path("REPLAY_JSON")?.is_file());
    assert!(path("REPLAY_ENV")?.is_file());
    let marker = path("REPLAY_FIXTURE_MARKER")?;
    let mode = std::env::var("REPLAY_FIXTURE_MODE")?;
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
    let status = std::env::var("REPLAY_FIXTURE_EXIT")?.parse::<i32>()?;
    if status != 0 {
        std::process::exit(status);
    }
    fs::copy(
        path("REPLAY_MANIFEST_SOURCE")?,
        path("REPLAY_MANIFEST_TARGET")?,
    )?;
    Ok(())
}
