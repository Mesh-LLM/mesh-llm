use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;

const SHIM_NAMES: &[&str] = &[
    "llama-quantize",
    "convert_hf_to_gguf.py",
    "convert-hf-to-gguf",
    "hf_to_gguf.py",
    "hf_to_gguff.py",
    "skippy-quantize-llama-quantize",
    "skippy-quantize-convert-hf-to-gguf",
];

#[derive(Debug, Parser)]
pub(crate) struct InstallShimsArgs {
    #[arg(long)]
    pub(crate) dir: PathBuf,
    #[arg(long)]
    pub(crate) binary: Option<PathBuf>,
    #[arg(long)]
    pub(crate) force: bool,
}

pub(crate) fn install_shims(args: InstallShimsArgs) -> Result<()> {
    let binary = args
        .binary
        .map(Ok)
        .unwrap_or_else(std::env::current_exe)
        .context("resolve skippy-quantize binary path")?;
    fs::create_dir_all(&args.dir)
        .with_context(|| format!("create shim directory {}", args.dir.display()))?;

    for name in SHIM_NAMES {
        install_shim(&args.dir, name, &binary, args.force)?;
    }
    Ok(())
}

fn install_shim(dir: &Path, name: &str, binary: &Path, force: bool) -> Result<()> {
    let shim_path = dir.join(name);
    if fs::symlink_metadata(&shim_path).is_ok() {
        if !force {
            bail!(
                "shim {} already exists; pass --force to replace it",
                shim_path.display()
            );
        }
        fs::remove_file(&shim_path)
            .with_context(|| format!("remove existing shim {}", shim_path.display()))?;
    }
    create_link_or_copy(binary, &shim_path)
        .with_context(|| format!("install shim {}", shim_path.display()))?;
    println!(
        "shim_installed name={} path={} target={}",
        name,
        shim_path.display(),
        binary.display()
    );
    Ok(())
}

#[cfg(unix)]
fn create_link_or_copy(binary: &Path, shim_path: &Path) -> Result<()> {
    std::os::unix::fs::symlink(binary, shim_path)?;
    Ok(())
}

#[cfg(not(unix))]
fn create_link_or_copy(binary: &Path, shim_path: &Path) -> Result<()> {
    fs::copy(binary, shim_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn installs_expected_shims() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let binary = root.join("skippy-quantize");
        fs::File::create(&binary)
            .unwrap()
            .write_all(b"fake")
            .unwrap();
        let dir = root.join("bin");

        install_shims(InstallShimsArgs {
            dir: dir.clone(),
            binary: Some(binary),
            force: false,
        })
        .unwrap();

        for name in SHIM_NAMES {
            assert!(fs::symlink_metadata(dir.join(name)).is_ok());
        }
        assert!(fs::symlink_metadata(dir.join("hf_to_gguff.py")).is_ok());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn refuses_to_replace_existing_without_force() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let binary = root.join("skippy-quantize");
        fs::write(&binary, b"fake").unwrap();
        let dir = root.join("bin");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("llama-quantize"), b"existing").unwrap();

        let result = install_shims(InstallShimsArgs {
            dir,
            binary: Some(binary),
            force: false,
        });

        assert!(result.is_err());
        fs::remove_dir_all(root).unwrap();
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("skippy-quantize-shims-{nanos}"))
    }
}
