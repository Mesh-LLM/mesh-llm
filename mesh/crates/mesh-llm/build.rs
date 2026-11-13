use std::path::PathBuf;

fn main() {
    // Windows gives a program's main thread 1 MB of stack, macOS and Linux
    // (by default) 8 MB. `--print-build-contract` runs on the main thread,
    // before the 8 MB application thread exists, and parsing the whole CLI
    // there overflows 1 MB in a debug build. Reserve 8 MB on Windows too.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let stack = 8 * 1024 * 1024;
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/STACK:{stack}");
        embed_icon();
    } else {
        println!("cargo:rustc-link-arg-bins=-Wl,--stack,{stack}");
    }
}

/// Give `mesh-llm.exe` the MeshLLM icon, so Explorer and the Start menu entry
/// `install.ps1` creates show it instead of the generic program icon.
///
/// The MSVC linker takes a compiled resource (`.res`) file as an input, so
/// this writes one from `assets/mesh-llm.ico` itself: no resource compiler and
/// no extra build dependency.
fn embed_icon() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let ico = manifest_dir.join("assets").join("mesh-llm.ico");
    println!("cargo:rerun-if-changed={}", ico.display());
    let bytes = std::fs::read(&ico).unwrap_or_else(|err| panic!("read {}: {err}", ico.display()));
    let res = icon_resource(&bytes).unwrap_or_else(|err| panic!("{}: {err}", ico.display()));
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("mesh-llm-icon.res");
    std::fs::write(&out, res).unwrap_or_else(|err| panic!("write {}: {err}", out.display()));
    println!("cargo:rustc-link-arg-bins={}", out.display());
}

const RT_ICON: u16 = 3;
const RT_GROUP_ICON: u16 = 14;
/// MOVEABLE | DISCARDABLE, as rc.exe writes for icon images.
const ICON_FLAGS: u16 = 0x1010;
/// MOVEABLE | PURE | DISCARDABLE, as rc.exe writes for an icon group.
const GROUP_FLAGS: u16 = 0x1030;
const LANG_EN_US: u16 = 0x0409;

/// A `.res` file holding every image of an `.ico` file as one icon group.
///
/// The `.ico` directory (6-byte header, then one 16-byte entry per image)
/// becomes an `RT_GROUP_ICON` resource whose entries point at `RT_ICON`
/// resources by id instead of by file offset. The group gets id 1, so it is
/// the first icon group and the one Windows shows for the executable.
fn icon_resource(ico: &[u8]) -> Result<Vec<u8>, String> {
    let u16_at = |at: usize| -> Result<u16, String> {
        ico.get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .ok_or_else(|| format!("truncated .ico at byte {at}"))
    };
    let u32_at = |at: usize| -> Result<u32, String> {
        ico.get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| format!("truncated .ico at byte {at}"))
    };
    if u16_at(0)? != 0 || u16_at(2)? != 1 {
        return Err("not an .ico file".into());
    }
    let count = u16_at(4)?;
    if count == 0 {
        return Err("the .ico file holds no image".into());
    }

    // An empty resource first marks the file as a 32-bit .res file.
    let mut res = Vec::new();
    push_resource(&mut res, 0, 0, 0, 0, &[]);

    let mut group = Vec::with_capacity(6 + 14 * usize::from(count));
    group.extend_from_slice(&0u16.to_le_bytes());
    group.extend_from_slice(&1u16.to_le_bytes());
    group.extend_from_slice(&count.to_le_bytes());
    for index in 0..count {
        let entry = 6 + 16 * usize::from(index);
        let size = u32_at(entry + 8)?;
        let offset = u32_at(entry + 12)? as usize;
        let image = ico
            .get(offset..offset + size as usize)
            .ok_or_else(|| format!("image {index} lies outside the .ico file"))?;
        let id = index + 1;
        push_resource(&mut res, RT_ICON, id, ICON_FLAGS, LANG_EN_US, image);
        // Width, height, color count, reserved, planes and bit count are the
        // same in both directories; only the 4-byte file offset becomes a
        // 2-byte resource id.
        group.extend_from_slice(&ico[entry..entry + 12]);
        group.extend_from_slice(&id.to_le_bytes());
    }
    push_resource(&mut res, RT_GROUP_ICON, 1, GROUP_FLAGS, LANG_EN_US, &group);
    Ok(res)
}

/// Append one resource with a numeric type and name: a 32-byte header, then
/// the data padded to a 4-byte boundary.
fn push_resource(res: &mut Vec<u8>, kind: u16, name: u16, flags: u16, lang: u16, data: &[u8]) {
    res.extend_from_slice(&(data.len() as u32).to_le_bytes()); // DataSize
    res.extend_from_slice(&32u32.to_le_bytes()); // HeaderSize
    res.extend_from_slice(&0xFFFFu16.to_le_bytes()); // TYPE is an ordinal
    res.extend_from_slice(&kind.to_le_bytes());
    res.extend_from_slice(&0xFFFFu16.to_le_bytes()); // NAME is an ordinal
    res.extend_from_slice(&name.to_le_bytes());
    res.extend_from_slice(&0u32.to_le_bytes()); // DataVersion
    res.extend_from_slice(&flags.to_le_bytes()); // MemoryFlags
    res.extend_from_slice(&lang.to_le_bytes()); // LanguageId
    res.extend_from_slice(&0u32.to_le_bytes()); // Version
    res.extend_from_slice(&0u32.to_le_bytes()); // Characteristics
    res.extend_from_slice(data);
    while !res.len().is_multiple_of(4) {
        res.push(0);
    }
}
