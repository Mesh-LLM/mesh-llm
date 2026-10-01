use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

pub(super) fn hashes<'a>(
    root: &Path,
    paths: impl Iterator<Item = &'a String>,
) -> DynResult<BTreeMap<String, String>> {
    let root = root.canonicalize()?;
    paths
        .map(|relative| {
            if relative.is_empty()
                || relative.contains(['\\', '\0', ':'])
                || Path::new(relative).is_absolute()
                || relative.split('/').any(|part| part == "..")
            {
                return Err(format!("unsafe artifact path: {relative}").into());
            }
            let path = root.join(relative).canonicalize()?;
            if !path.starts_with(&root) || !path.is_file() {
                return Err(
                    format!("artifact file escapes root or is not a file: {relative}").into(),
                );
            }
            let mut file = std::fs::File::open(path)?;
            let mut digest = Sha256::new();
            let mut buffer = [0_u8; 65536];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
            }
            Ok((relative.clone(), hex::encode(digest.finalize())))
        })
        .collect()
}

pub(super) fn glibc_floor<'a>(
    root: &Path,
    os: &str,
    paths: impl Iterator<Item = &'a String>,
) -> DynResult<Option<String>> {
    if os != "linux" {
        return Ok(None);
    }
    let mut floor = None;
    for relative in paths {
        let path = root.join(relative);
        let mut magic = [0_u8; 4];
        if std::fs::File::open(&path)?.read(&mut magic)? != 4 || magic != *b"\x7fELF" {
            continue;
        }
        let output = super::super::toolchain::run_tool(
            &super::super::toolchain::HostToolchain,
            &[
                "readelf".into(),
                "-V".into(),
                path.to_string_lossy().into_owned(),
            ],
        )?;
        floor = floor.max(version_floor(&output)?);
    }
    Ok(floor.map(|(major, minor)| format!("{major}.{minor}")))
}

fn version_floor(output: &str) -> DynResult<Option<(u32, u32)>> {
    let Some((_, needs)) = output.split_once("Version needs section") else {
        return Ok(None);
    };
    let mut floor = None;
    for suffix in needs.split("GLIBC_").skip(1) {
        let version = suffix
            .split(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
            .next()
            .unwrap_or_default();
        let parsed = if suffix.starts_with("ABI_DT_RELR") {
            Some((2, 36))
        } else if let Some((major, minor)) = version.split_once('.') {
            Some((
                major.parse()?,
                minor
                    .split('.')
                    .next()
                    .ok_or("missing GLIBC minor")?
                    .parse()?,
            ))
        } else {
            None
        };
        floor = floor.max(parsed);
    }
    Ok(floor)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn l4_glibc_floor_when_definitions_and_relr_are_present() {
        let output = "Version definition section GLIBC_9.99\nVersion needs section GLIBC_2.2.5 GLIBC_2.17 GLIBC_ABI_DT_RELR GLIBC_2.35";
        let floor = version_floor(output).unwrap();
        assert_eq!(floor, Some((2, 36)));
    }

    #[test]
    fn l4_glibc_floor_when_only_definitions_are_present() {
        let output = "Version definition section GLIBC_9.99";
        let floor = version_floor(output).unwrap();
        assert_eq!(floor, None);
    }
}
