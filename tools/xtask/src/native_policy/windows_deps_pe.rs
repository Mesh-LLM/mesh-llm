//! `imported_dlls` of `scripts/windows-native-runtime-deps.py`: the legacy
//! script's own PE import-table reader (no external tool runs). Every
//! bounds check, error text and `ascii` decode matches the Python original.

use crate::ci_plan::catalog::os_error_text;

/// A failure the legacy `main` printed (or, for an uncaught exception, the
/// final traceback line) before exiting 1.
pub(super) type Raised = String;

/// `_unpack`: `len` little-endian bytes at `offset`, or a truncation error.
fn field(data: &[u8], offset: u64, len: u64) -> Result<&[u8], Raised> {
    let end = offset.saturating_add(len);
    if end > data.len() as u64 {
        return Err("truncated PE image".to_owned());
    }
    // Both bounds are within `data.len()`, so they fit in `usize`.
    Ok(&data[offset as usize..end as usize])
}

fn u16_at(data: &[u8], offset: u64) -> Result<u64, Raised> {
    let bytes = field(data, offset, 2)?;
    Ok(u64::from(u16::from_le_bytes([bytes[0], bytes[1]])))
}

fn u32_at(data: &[u8], offset: u64) -> Result<u64, Raised> {
    let bytes = field(data, offset, 4)?;
    Ok(u64::from(u32::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
    ])))
}

/// `_cstring`: the NUL-terminated ASCII name at `offset`.
fn cstring(data: &[u8], offset: u64) -> Result<String, Raised> {
    if offset >= data.len() as u64 {
        return Err("PE import name points outside the image".to_owned());
    }
    let tail = &data[offset as usize..];
    let Some(end) = tail.iter().position(|byte| *byte == 0) else {
        return Err("unterminated PE import name".to_owned());
    };
    let name = &tail[..end];
    if let Some(position) = name.iter().position(|byte| !byte.is_ascii()) {
        return Err(format!(
            "UnicodeDecodeError: 'ascii' codec can't decode byte 0x{:02x} in position {position}: ordinal not in range(128)",
            name[position]
        ));
    }
    Ok(name.iter().map(|byte| char::from(*byte)).collect())
}

/// One section header: virtual address, mapped size and raw file offset.
struct Section {
    virtual_address: u64,
    size: u64,
    raw_offset: u64,
}

/// The parsed layout `rva_offset` resolves against.
struct Image<'a> {
    data: &'a [u8],
    path: &'a str,
    sections: Vec<Section>,
}

impl Image<'_> {
    fn rva_offset(&self, rva: u64) -> Result<u64, Raised> {
        for section in &self.sections {
            if section.virtual_address <= rva && rva < section.virtual_address + section.size {
                return Ok(section.raw_offset + rva - section.virtual_address);
            }
        }
        if rva < self.data.len() as u64 {
            return Ok(rva);
        }
        Err(format!(
            "PE RVA 0x{rva:x} is outside every section: {}",
            self.path
        ))
    }
}

/// `imported_dlls(path)`: each import descriptor's DLL name, in table order.
pub(super) fn imported_dlls(path: &str) -> Result<Vec<String>, Raised> {
    let data = std::fs::read(path).map_err(|error| os_error_text(&error, path))?;
    if !data.starts_with(b"MZ") {
        return Err(format!("not a PE image: {path}"));
    }
    let pe_offset = u32_at(&data, 0x3C)?;
    if data
        .get(pe_offset as usize..)
        .and_then(|rest| rest.get(..4))
        != Some(b"PE\0\0")
    {
        return Err(format!("invalid PE signature: {path}"));
    }
    let file_header = pe_offset + 4;
    field(&data, file_header, 20)?;
    let section_count = u16_at(&data, file_header + 2)?;
    let optional_size = u16_at(&data, file_header + 16)?;
    let optional_header = file_header + 20;
    let data_directories = match u16_at(&data, optional_header)? {
        0x20B => optional_header + 112,
        0x10B => optional_header + 96,
        magic => {
            return Err(format!(
                "unsupported PE optional-header magic 0x{magic:x}: {path}"
            ));
        }
    };
    field(&data, data_directories + 8, 8)?;
    let import_rva = u32_at(&data, data_directories + 8)?;
    let section_table = optional_header + optional_size;
    let mut sections = Vec::new();
    for index in 0..section_count {
        let header = section_table + index * 40 + 8;
        field(&data, header, 16)?;
        let virtual_size = u32_at(&data, header)?;
        sections.push(Section {
            virtual_address: u32_at(&data, header + 4)?,
            size: virtual_size.max(u32_at(&data, header + 8)?),
            raw_offset: u32_at(&data, header + 12)?,
        });
    }
    let image = Image {
        data: &data,
        path,
        sections,
    };
    if import_rva == 0 {
        return Ok(Vec::new());
    }
    import_names(&image, image.rva_offset(import_rva)?)
}

/// Walks the null-terminated import descriptor array from `descriptor`.
fn import_names(image: &Image<'_>, mut descriptor: u64) -> Result<Vec<String>, Raised> {
    let mut imports = Vec::new();
    loop {
        let fields = field(image.data, descriptor, 20)?;
        if fields.iter().all(|byte| *byte == 0) {
            return Ok(imports);
        }
        let name_rva = u32_at(image.data, descriptor + 12)?;
        if name_rva == 0 {
            return Err(format!(
                "PE import descriptor has no DLL name: {}",
                image.path
            ));
        }
        imports.push(cstring(image.data, image.rva_offset(name_rva)?)?);
        descriptor += 20;
    }
}
