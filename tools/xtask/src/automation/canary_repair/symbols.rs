//! Inspect executable code symbols directly; dumping Rust's full symbol table is unbounded diagnostics.
use crate::command::DynResult;
use std::{collections::BTreeSet, fs, path::Path};

pub(super) fn require(path: &Path) -> DynResult<()> {
    if !defined(&fs::read(path)?)? {
        return Err(format!(
            "executable lacks a defined static skippy_model_open: {}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn defined(bytes: &[u8]) -> DynResult<bool> {
    match goblin::Object::parse(bytes)? {
        goblin::Object::Mach(goblin::mach::Mach::Binary(binary)) => {
            let mut code = BTreeSet::new();
            for (index, section) in binary.segments.sections().flatten().enumerate() {
                let (section, _) = section?;
                if section.flags
                    & (goblin::mach::constants::S_ATTR_PURE_INSTRUCTIONS
                        | goblin::mach::constants::S_ATTR_SOME_INSTRUCTIONS)
                    != 0
                {
                    code.insert(index + 1);
                }
            }
            for symbol in binary.symbols() {
                let (name, symbol) = symbol?;
                if name == "_skippy_model_open"
                    && symbol.is_global()
                    && !symbol.is_stab()
                    && symbol.get_type() == goblin::mach::symbols::N_SECT
                    && code.contains(&symbol.n_sect)
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        goblin::Object::Elf(binary) => {
            for (symbols, strings) in [
                (&binary.syms, &binary.strtab),
                (&binary.dynsyms, &binary.dynstrtab),
            ] {
                if symbols.iter().any(|symbol| {
                    symbol.is_function()
                        && symbol.st_shndx != goblin::elf::section_header::SHN_UNDEF as usize
                        && strings.get_at(symbol.st_name) == Some("skippy_model_open")
                        && binary
                            .section_headers
                            .get(symbol.st_shndx)
                            .is_some_and(|section| {
                                section.sh_flags
                                    & u64::from(goblin::elf::section_header::SHF_EXECINSTR)
                                    != 0
                            })
                }) {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

#[cfg(test)]
#[path = "symbols_tests.rs"]
mod tests;
