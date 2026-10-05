use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command as ProcessCommand;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

pub(crate) fn file_sha256(path: &Path) -> Result<String> {
    if let Some(hash) = file_sha256_openssl(path)? {
        return Ok(hash);
    }

    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

pub(crate) fn file_sha256_openssl(path: &Path) -> Result<Option<String>> {
    let output = match ProcessCommand::new("openssl")
        .arg("dgst")
        .arg("-sha256")
        .arg("-r")
        .arg(path)
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("hash {}", path.display())),
    };
    if !output.status.success() {
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(hash) = stdout.split_whitespace().next() else {
        return Ok(None);
    };
    if hash.len() == 64 && hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
        Ok(Some(hash.to_ascii_lowercase()))
    } else {
        Ok(None)
    }
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}
