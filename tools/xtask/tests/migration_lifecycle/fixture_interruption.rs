use super::{Behavior, wait_stop};
use std::path::Path;
use std::time::{Duration, Instant};

pub(super) fn branch(root: &Path, behavior: Behavior) -> Result<(), Box<dyn std::error::Error>> {
    let stubborn = matches!(behavior, Behavior::StubbornDescendant);
    if stubborn {
        std::fs::write(root.join("stubborn-leaf"), b"stubborn")?;
    }
    let mut child = std::process::Command::new(std::env::current_exe()?)
        .arg("--leaf")
        .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", root)
        .spawn()?;
    let result = (|| {
        wait_file(root, "leaf.armed")?;
        std::fs::write(root.join("startup.armed"), b"armed")?;
        wait_stop(stubborn)?;
        std::fs::write(root.join("handler"), b"graceful")?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })();
    if result.is_err() {
        child.kill()?;
    }
    child.wait()?;
    result
}

pub(super) fn hold_cleanup(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(root.join("cleanup.armed"), b"armed")?;
    wait_file(root, "cleanup.release")
}

fn wait_file(root: &Path, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let until = Instant::now() + Duration::from_secs(5);
    while !root.join(name).exists() {
        if Instant::now() >= until {
            return Err("fixture interruption handshake deadline".into());
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    Ok(())
}
