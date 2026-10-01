use super::{Checked, positional, python_io};
use std::path::PathBuf;

struct CacheOperands {
    source: PathBuf,
    destination: PathBuf,
}

pub(super) fn run(args: &[String]) -> Checked<String> {
    let [source, destination] = positional(args, "SOURCE DESTINATION")?;
    let operands = CacheOperands {
        source: PathBuf::from(source),
        destination: PathBuf::from(destination),
    };
    filter(&operands)?;
    Ok(String::new())
}

fn filter(operands: &CacheOperands) -> Checked<()> {
    let source = python_io::read_text(&operands.source)?;
    let normalized = source.replace("\r\n", "\n").replace('\r', "\n");
    let mut output = String::from("# Portable MeshLLM static ABI link metadata\n");
    for line in normalized
        .split_inclusive('\n')
        .filter(|line| retained(line))
    {
        output.push_str(line);
    }
    std::fs::write(&operands.destination, output)
        .map_err(|error| python_io::os_error(&operands.destination, &error).into())
}

fn retained(line: &str) -> bool {
    let Some((key, typed_value)) = line.split_once(':') else {
        return false;
    };
    let Some((kind, _)) = typed_value.split_once('=') else {
        return false;
    };
    if kind.is_empty() {
        return false;
    }
    match key {
        "GGML_OPENMP_ENABLED" | "OpenMP_C_LIB_NAMES" | "OpenMP_CXX_LIB_NAMES" => true,
        other => other
            .strip_prefix("OpenMP_")
            .and_then(|suffix| suffix.strip_suffix("_LIBRARY"))
            .is_some_and(|name| {
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            }),
    }
}

#[cfg(test)]
#[path = "abi_cache_filter_tests.rs"]
mod tests;
