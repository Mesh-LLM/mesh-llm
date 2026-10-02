//! Synchronize generated UniFFI API guards with constants in the built static library.
use crate::{command::DynResult, process};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read, Write},
    num::NonZeroUsize,
    path::Path,
    time::Duration,
};

const SYMBOL: &str = "uniffi_meshllm_ffi_checksum_";
const MAX_SWIFT: usize = 8 * 1024 * 1024;
const MAX_DISASSEMBLY: usize = 128 * 1024 * 1024;
static TEMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let [library, swift] = args else {
        return Err("usage: prepared-input swift-api-checksum LIBRARY GENERATED_SWIFT".into());
    };
    let library = Path::new(library).canonicalize()?;
    if !library.is_file() {
        return Err("Swift checksum input must be a static library file".into());
    }
    let swift = Path::new(swift);
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = synchronize(
        Path::new("/usr/bin/otool"),
        &library,
        swift,
        &interrupt.cancellation(),
        Duration::from_secs(120),
    );
    interrupt.finish()?;
    let count = result?;
    println!("Synchronized {count} UniFFI API checksum guards");
    Ok(())
}

fn synchronize(
    executable: &Path,
    library: &Path,
    swift: &Path,
    cancellation: &process::Cancellation,
    budget: Duration,
) -> DynResult<usize> {
    let original = read_swift(swift)?;
    let disassembly = disassemble(executable, library, cancellation, budget)?;
    let values = constants(&disassembly)?;
    let (replacement, count) = rewrite(&original, &values)?;
    if cancellation.is_cancelled() {
        return Err("Swift checksum synchronization cancelled before write".into());
    }
    atomic_replace(swift, &original, &replacement)?;
    Ok(count)
}

fn disassemble(
    executable: &Path,
    library: &Path,
    cancellation: &process::Cancellation,
    budget: Duration,
) -> DynResult<String> {
    let report = process::supervise_raw(
        &process::ProcessSpec {
            executable: executable.to_path_buf(),
            arguments: vec![
                process::Value::Public("-tvV".into()),
                process::Value::Public(library.as_os_str().to_owned()),
            ],
            cwd: library
                .parent()
                .ok_or("library has no parent")?
                .to_path_buf(),
            environment: [("PATH", "/usr/bin:/bin"), ("LC_ALL", "C")]
                .into_iter()
                .map(|(key, value)| (key.into(), process::Value::Public(value.into())))
                .collect(),
        },
        &process::Limits {
            execution: budget,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        cancellation,
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(MAX_DISASSEMBLY),
            stderr: None,
        },
    )?;
    if !report.process.success() {
        return Err(format!(
            "otool checksum disassembly failed: {:?}",
            report.process.outcome
        )
        .into());
    }
    let bytes = report
        .stdout
        .ok_or("otool did not return complete disassembly")?;
    Ok(std::str::from_utf8(bytes.as_bytes())?.to_owned())
}

fn instruction(line: &str) -> DynResult<(&str, &str)> {
    let line = line.trim();
    let (address, text) = line
        .split_once(char::is_whitespace)
        .ok_or("missing instruction address")?;
    if address.is_empty() || !address.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("invalid instruction address".into());
    }
    let text = text.trim();
    Ok(text
        .split_once(char::is_whitespace)
        .map_or((text, ""), |(op, args)| (op, args.trim())))
}

fn return_constant(load: &str, exit: &str) -> DynResult<u16> {
    let (operation, arguments) = instruction(load)?;
    let compact: String = arguments.chars().filter(|ch| !ch.is_whitespace()).collect();
    let immediate = if operation == "mov" {
        compact.strip_prefix("w0,#")
    } else if matches!(operation, "movw" | "movl") {
        let register = if operation == "movw" { ",%ax" } else { ",%eax" };
        compact
            .strip_prefix('$')
            .and_then(|value| value.strip_suffix(register))
    } else {
        None
    }
    .ok_or("checksum symbol must return a constant in w0, ax or eax")?;
    let (ret, operands) = instruction(exit)?;
    if !matches!(ret, "ret" | "retq") || !operands.is_empty() {
        return Err("checksum constant must be followed by return".into());
    }
    let value = match immediate.strip_prefix("0x") {
        Some(hex) => u16::from_str_radix(hex, 16),
        None => immediate.parse::<u16>(),
    }?;
    Ok(value)
}

fn constants(disassembly: &str) -> DynResult<BTreeMap<String, u16>> {
    if disassembly.len() > MAX_DISASSEMBLY {
        return Err("checksum disassembly exceeds 128 MiB".into());
    }
    let mut lines = disassembly.lines();
    let mut values = BTreeMap::new();
    while let Some(line) = lines.next() {
        let Some(name) = line
            .trim()
            .strip_prefix('_')
            .and_then(|name| name.strip_suffix(':'))
        else {
            continue;
        };
        if !name.starts_with(SYMBOL) {
            continue;
        }
        if name.len() > 512
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err("invalid UniFFI checksum symbol".into());
        }
        let load = lines.next().ok_or("missing checksum constant")?;
        let exit = lines.next().ok_or("missing checksum return")?;
        let value = return_constant(load, exit)?;
        if values.insert(name.to_owned(), value).is_some() {
            return Err(format!("ambiguous duplicate checksum symbol: {name}").into());
        }
    }
    if values.is_empty() {
        return Err("otool output contains no UniFFI API checksum constants".into());
    }
    Ok(values)
}

fn rewrite(swift: &str, values: &BTreeMap<String, u16>) -> DynResult<(String, usize)> {
    let mut output = String::with_capacity(swift.len());
    let mut guards = BTreeMap::new();
    for line in swift.split_inclusive('\n') {
        let Some(start) = line.find(SYMBOL) else {
            output.push_str(line);
            continue;
        };
        let call = &line[start..];
        let (name, remaining) = call
            .split_once("() != ")
            .ok_or("malformed generated checksum guard")?;
        let digits = remaining.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0
            || !remaining[digits..].starts_with(')')
            || remaining[digits..].contains(SYMBOL)
        {
            return Err("malformed generated checksum literal".into());
        }
        remaining[..digits].parse::<u16>()?;
        let value = values
            .get(name)
            .ok_or_else(|| format!("native checksum missing for {name}"))?;
        if guards.insert(name, ()).is_some() {
            return Err(format!("duplicate generated checksum guard: {name}").into());
        }
        output.push_str(&line[..start]);
        output.push_str(name);
        output.push_str("() != ");
        output.push_str(&value.to_string());
        output.push_str(&remaining[digits..]);
    }
    if guards.is_empty() {
        return Err("Swift bindings contain no UniFFI API checksum guards".into());
    }
    Ok((output, guards.len()))
}

fn read_swift(path: &Path) -> DynResult<String> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_SWIFT as u64 {
        return Err("Swift bindings must be a regular file of at most 8 MiB".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take((MAX_SWIFT + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_SWIFT {
        return Err("Swift bindings exceed 8 MiB".into());
    }
    Ok(String::from_utf8(bytes)?)
}

fn atomic_replace(path: &Path, original: &str, replacement: &str) -> DynResult<()> {
    let directory = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let sequence = TEMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = directory.join(format!(
        ".swift-api-checksum-{}-{timestamp}-{sequence}.tmp",
        std::process::id()
    ));
    let result = (|| -> DynResult<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.set_permissions(fs::metadata(path)?.permissions())?;
        file.write_all(replacement.as_bytes())?;
        file.sync_all()?;
        if read_swift(path)? != original {
            return Err("Swift bindings changed during checksum synchronization".into());
        }
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _cleanup = fs::remove_file(temporary);
    }
    result
}

#[cfg(test)]
#[path = "swift_api_checksum_tests.rs"]
mod tests;
