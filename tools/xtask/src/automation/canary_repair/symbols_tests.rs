use super::defined;

fn word(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
fn wide(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}
fn macho(name: &str, imported: bool, code: bool) -> Vec<u8> {
    let mut bytes = vec![0; 512];
    for (at, value) in [
        (0, 0xfeedfacf),
        (4, 0x0100000c),
        (12, 2),
        (16, 2),
        (20, 176),
        (32, 0x19),
        (36, 152),
        (88, 7),
        (92, 5),
        (96, 1),
        (152, 511),
        (184, 2),
        (188, 24),
        (192, 208),
        (196, 1),
        (200, 224),
        (204, u32::try_from(name.len() + 2).unwrap()),
        (208, 1),
    ] {
        word(&mut bytes, at, value);
    }
    wide(&mut bytes, 56, 0x100000000);
    wide(&mut bytes, 64, 4096);
    wide(&mut bytes, 80, 512);
    wide(&mut bytes, 136, 0x100000200);
    wide(&mut bytes, 144, 1);
    wide(&mut bytes, 216, 0x100000200);
    bytes[40..46].copy_from_slice(b"__TEXT");
    bytes[104..110].copy_from_slice(b"__text");
    bytes[120..126].copy_from_slice(b"__TEXT");
    if code {
        word(&mut bytes, 168, 0x80000400);
    }
    bytes[212] = if imported { 1 } else { 0xf };
    bytes[213] = u8::from(!imported);
    bytes[225..225 + name.len()].copy_from_slice(name.as_bytes());
    bytes
}
fn elf(name: &str, imported: bool, function: bool) -> Vec<u8> {
    let mut bytes = vec![0; 600];
    bytes[..16].copy_from_slice(b"\x7fELF\x02\x01\x01\0\0\0\0\0\0\0\0\0");
    bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
    bytes[18..20].copy_from_slice(&62_u16.to_le_bytes());
    word(&mut bytes, 20, 1);
    wide(&mut bytes, 40, 64);
    for (at, value) in [(52, 64_u16), (58, 64), (60, 4)] {
        bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }
    for (at, value) in [(132, 1), (196, 2), (232, 3), (236, 1), (260, 3), (344, 1)] {
        word(&mut bytes, at, value);
    }
    for (at, value) in [
        (136, 6),
        (144, 0x400000),
        (152, 512),
        (160, 1),
        (176, 1),
        (216, 320),
        (224, 48),
        (240, 8),
        (248, 24),
        (280, 368),
        (288, u64::try_from(name.len() + 2).unwrap()),
        (304, 1),
        (352, 0x400000),
        (360, 1),
    ] {
        wide(&mut bytes, at, value);
    }
    bytes[348] = if function { 0x12 } else { 0x11 };
    bytes[350..352].copy_from_slice(&u16::from(!imported).to_le_bytes());
    bytes[369..369 + name.len()].copy_from_slice(name.as_bytes());
    bytes
}

#[test]
fn macho_requires_defined_code_instead_of_import_data_or_rust_wrapper() {
    assert!(defined(&macho("_skippy_model_open", false, true)).unwrap());
    for bytes in [
        macho("_skippy_model_open", true, true),
        macho("_skippy_model_open", false, false),
        macho("_rust_skippy_model_open", false, true),
    ] {
        assert!(!defined(&bytes).unwrap());
    }
}
#[test]
fn elf_requires_defined_function_instead_of_dynamic_import_or_data() {
    assert!(defined(&elf("skippy_model_open", false, true)).unwrap());
    for bytes in [
        elf("skippy_model_open", true, true),
        elf("skippy_model_open", false, false),
        elf("rust_skippy_model_open", false, true),
    ] {
        assert!(!defined(&bytes).unwrap());
    }
}
#[test]
fn malformed_binary_cannot_supply_native_identity() {
    assert!(defined(&macho("_skippy_model_open", false, true)[..30]).is_err());
    assert!(!defined(b"not an executable").unwrap_or(false));
}
