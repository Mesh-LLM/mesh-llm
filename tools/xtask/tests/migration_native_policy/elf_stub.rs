//! A `readelf` stand-in for `native linux-runtime-deps` cases. Each entry of
//! a case's `elves` object is written as an ELF-magic file whose remaining
//! bytes are its `content` key (default: its path with `/` as `%`, so
//! digests differ), and the canned `readelf -h`/`-d` output for that key is
//! stored beside the stub, so copies inspect alike. `class` defaults to
//! ELF64 and `machine` to x86-64; an empty string omits the line. The stub
//! logs each call and fails like readelf for a key it has no output for
//! (`broken`). `no_readelf` leaves the stub out; `links` adds symlinks.

use serde_json::Value;
use std::error::Error;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

fn text(value: &Value) -> &str {
    value.as_str().unwrap_or_default()
}

fn header(elf: &Value) -> String {
    let mut output = String::from("ELF Header:\n  Magic:   7f 45 4c 46\n");
    let class = elf["class"].as_str().unwrap_or("ELF64");
    if !class.is_empty() {
        output.push_str(&format!("  Class:                             {class}\n"));
    }
    output.push_str("  Data:                              2's complement, little endian\n");
    let machine = elf["machine"]
        .as_str()
        .unwrap_or("Advanced Micro Devices X86-64");
    if !machine.is_empty() {
        output.push_str(&format!("  Machine:                           {machine}\n"));
    }
    output
}

fn dynamic(elf: &Value) -> String {
    let mut output = String::from("\nDynamic section at offset 0x2de0 contains 4 entries:\n");
    output.push_str("  Tag        Type                         Name/Value\n");
    for needed in elf["needed"].as_array().into_iter().flatten() {
        output.push_str(&format!(
            " 0x0000000000000001 (NEEDED)             Shared library: [{}]\n",
            text(needed)
        ));
    }
    if let Some(soname) = elf["soname"].as_str() {
        output.push_str(&format!(
            " 0x000000000000000e (SONAME)             Library soname: [{soname}]\n"
        ));
    }
    output.push_str(" 0x0000000000000000 (NULL)               0x0\n");
    output
}

/// Writes the case's ELF files and, when it has an `elves` object, the
/// `readelf` stub into `bin`.
pub(crate) fn build(root: &Path, bin: &Path, case: &Value) -> Result<(), Box<dyn Error>> {
    for (relative, target) in case["links"].as_object().into_iter().flatten() {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().ok_or("link without parent")?)?;
        std::os::unix::fs::symlink(text(target), path)?;
    }
    let Some(elves) = case["elves"].as_object() else {
        return Ok(());
    };
    let install = case["no_readelf"].as_bool() != Some(true);
    let outputs = bin.join("readelf.d");
    fs::create_dir_all(&outputs)?;
    for (relative, elf) in elves {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().ok_or("file without parent")?)?;
        let key = elf["content"]
            .as_str()
            .map_or_else(|| relative.replace('/', "%"), str::to_owned);
        let mut bytes = b"\x7fELF".to_vec();
        bytes.extend_from_slice(key.as_bytes());
        fs::write(&path, bytes)?;
        if elf["broken"].as_bool() == Some(true) {
            continue;
        }
        fs::write(outputs.join(format!("{key}.h")), header(elf))?;
        fs::write(outputs.join(format!("{key}.d")), dynamic(elf))?;
    }
    let script = format!(
        "#!/bin/sh\nprintf 'readelf %s\\n' \"$*\" >> '{log}'\n\
key=$(/usr/bin/tail -c +5 \"$2\")\n\
out='{outputs}'/\"$key\".${{1#-}}\n\
if [ -f \"$out\" ]; then /bin/cat \"$out\"; exit 0; fi\n\
printf 'readelf: Error: %s: Failed to read file header\\n' \"$2\" >&2\nexit 1\n",
        log = bin.join("calls.log").display(),
        outputs = outputs.display(),
    );
    if install {
        let stub = bin.join("readelf");
        fs::write(&stub, script)?;
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
}
