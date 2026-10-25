//! Synthetic PE images for `native windows-runtime-deps` cases. Each entry
//! of a case's `pes` object is written as a minimal PE32+ file with one
//! `.idata` section whose import descriptors name `imports`, in order.
//! `magic` overrides the optional-header magic (`267` selects PE32),
//! `bad_signature` corrupts the `PE\0\0` signature, `nameless` zeroes the
//! first descriptor's name RVA, `no_imports` zeroes the import directory,
//! `name_rva` overrides the first name RVA, and `truncate` cuts the file.

use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::Path;

const PE_OFFSET: usize = 0x40;
const SECTION_RVA: u32 = 0x1000;
const RAW_OFFSET: usize = 0x200;

fn put_u16(bytes: &mut [u8], at: usize, value: u16) {
    bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// The `.idata` section: descriptors, a zero terminator, then the names.
fn import_section(pe: &Value) -> Result<Vec<u8>, Box<dyn Error>> {
    let imports: Vec<&str> = pe["imports"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|name| name.as_str().unwrap_or_default())
        .collect();
    let mut section = vec![0u8; (imports.len() + 1) * 20];
    for (index, name) in imports.iter().enumerate() {
        let descriptor = index * 20;
        let rva = SECTION_RVA + u32::try_from(section.len())?;
        section.extend_from_slice(name.as_bytes());
        section.push(0);
        put_u32(&mut section, descriptor, 1);
        let name_rva = match (index, pe["nameless"].as_bool(), pe["name_rva"].as_u64()) {
            (0, Some(true), _) => 0,
            (0, _, Some(forced)) => u32::try_from(forced)?,
            _ => rva,
        };
        put_u32(&mut section, descriptor + 12, name_rva);
    }
    Ok(section)
}

fn image(pe: &Value) -> Result<Vec<u8>, Box<dyn Error>> {
    let magic = u16::try_from(pe["magic"].as_u64().unwrap_or(0x20B))?;
    let directories = if magic == 0x10B { 96 } else { 112 };
    let optional_size = directories + 16 * 8;
    let optional_header = PE_OFFSET + 24;
    let section_table = optional_header + optional_size;
    let section = import_section(pe)?;
    let mut bytes = vec![0u8; RAW_OFFSET];
    bytes[..2].copy_from_slice(b"MZ");
    put_u32(&mut bytes, 0x3C, u32::try_from(PE_OFFSET)?);
    let signature: &[u8] = if pe["bad_signature"].as_bool() == Some(true) {
        b"PX\0\0"
    } else {
        b"PE\0\0"
    };
    bytes[PE_OFFSET..PE_OFFSET + 4].copy_from_slice(signature);
    put_u16(&mut bytes, PE_OFFSET + 4, 0x8664);
    put_u16(&mut bytes, PE_OFFSET + 6, 1);
    put_u16(&mut bytes, PE_OFFSET + 20, u16::try_from(optional_size)?);
    put_u16(&mut bytes, optional_header, magic);
    if pe["no_imports"].as_bool() != Some(true) {
        put_u32(&mut bytes, optional_header + directories + 8, SECTION_RVA);
        put_u32(
            &mut bytes,
            optional_header + directories + 12,
            u32::try_from(section.len())?,
        );
    }
    bytes[section_table..section_table + 6].copy_from_slice(b".idata");
    let size = u32::try_from(section.len())?;
    put_u32(&mut bytes, section_table + 8, size);
    put_u32(&mut bytes, section_table + 12, SECTION_RVA);
    put_u32(&mut bytes, section_table + 16, size);
    put_u32(&mut bytes, section_table + 20, u32::try_from(RAW_OFFSET)?);
    bytes.extend_from_slice(&section);
    if let Some(length) = pe["truncate"].as_u64() {
        bytes.truncate(usize::try_from(length)?);
    }
    Ok(bytes)
}

/// Writes the case's synthetic PE files below `root`.
pub(crate) fn build(root: &Path, case: &Value) -> Result<(), Box<dyn Error>> {
    for (relative, pe) in case["pes"].as_object().into_iter().flatten() {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().ok_or("file without parent")?)?;
        fs::write(&path, image(pe)?)?;
    }
    Ok(())
}
