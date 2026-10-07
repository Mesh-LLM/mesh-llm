//! `native verify-host-dependencies`: the Rust owner of
//! `scripts/verify-host-dependencies.py`. Inspects a binary's imports with
//! the platform tool, rejects backend libraries unless `--no-import-policy`,
//! checks an ELF binary's GLIBC floor against `--max-glibc`, prints the
//! sorted-key JSON report and optionally writes it indented to `--report`.
//! Handled failures print `str(error)` with status 2, like the legacy script.

use super::argv::{Grammar, Opt, Parsed};
use super::forbidden::forbidden;
use super::glibc::{Version, declared_floor, elf_floor, parse_version};
use super::imports::{elf_imports, macho_imports, pe_imports};
use super::toolchain::{Toolchain, run_tool};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use crate::model_registry::json_bytes::{ASCII, Style, dumps};
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};

#[path = "host_dependencies/binary_identity.rs"]
mod binary_identity;

pub(super) fn binary_sha256(path: &Path) -> Result<String, String> {
    binary_identity::digest(path)
}

const GRAMMAR: Grammar = Grammar {
    prog: "verify-host-dependencies.py",
    usage: "\
usage: verify-host-dependencies.py [-h] [--format {elf,macho,pe}]
                                   [--report REPORT] [--no-import-policy]
                                   [--max-glibc MAX_GLIBC]
                                   binary
",
    help: "\
usage: verify-host-dependencies.py [-h] [--format {elf,macho,pe}]
                                   [--report REPORT] [--no-import-policy]
                                   [--max-glibc MAX_GLIBC]
                                   binary

positional arguments:
  binary

options:
  -h, --help            show this help message and exit
  --format {elf,macho,pe}
  --report REPORT
  --no-import-policy    Report imports without enforcing the host policy.
                        Native runtime libraries legitimately import each
                        other, so only the glibc floor applies to them.
  --max-glibc MAX_GLIBC
                        Reject an ELF binary that needs a GLIBC symbol version
                        above this one, or the literal 'declared' to read
                        scripts/linux-glibc-floor.txt.
",
    options: &[
        Opt::choice("--format", &["elf", "macho", "pe"]),
        Opt::value("--report"),
        Opt::flag("--no-import-policy"),
        Opt::flag("--bind-sha256"),
        Opt::value("--max-glibc"),
    ],
    positional: Some("binary"),
};

/// `json.dumps(report, indent=2, sort_keys=True)`.
const INDENTED: Style = Style {
    indent: Some(2),
    item_separator: ",",
    ..ASCII
};

/// Where `--max-glibc declared` reads the floor: the legacy script's
/// directory, `scripts/` of the checkout.
pub(super) type FloorFile<'a> = &'a mut dyn FnMut() -> Result<PathBuf, String>;

pub(super) fn run(
    args: &[String],
    tools: &dyn Toolchain,
    floor_file: FloorFile<'_>,
) -> CheckReport {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    verify(&parsed, tools, floor_file).unwrap_or_else(|message| CheckReport {
        stdout: String::new(),
        stderr: format!("{message}\n"),
        code: 2,
    })
}

/// `Path.name` of a displayed path.
fn file_name(shown: &str) -> &str {
    match shown {
        "." | "/" => "",
        _ => shown.rsplit('/').next().unwrap_or(shown),
    }
}

fn verify(
    parsed: &Parsed,
    tools: &dyn Toolchain,
    floor_file: FloorFile<'_>,
) -> Result<CheckReport, String> {
    let binary = python_path_display(Path::new(parsed.positional.as_deref().unwrap_or_default()));
    let identity = parsed
        .flag("--bind-sha256")
        .then(|| binary_identity::digest(Path::new(&binary)))
        .transpose()?;
    let no_policy = parsed.flag("--no-import-policy");
    let (format, imports) = inspect(&binary, parsed.value("--format"), tools)?;
    let rejected = if no_policy {
        Vec::new()
    } else {
        forbidden(&imports)
    };
    let max_glibc = resolve_max_glibc(parsed.value("--max-glibc"), floor_file)?;
    let floor = if format == "elf" {
        let output = run_tool(tools, &argv(["readelf", "-V", &binary]))?;
        elf_floor(&output)
    } else {
        None
    };
    let name = file_name(&binary);
    let strings = |items: &[String]| Json::Array(items.iter().cloned().map(Json::String).collect());
    let mut fields = vec![
        ("binary".to_owned(), Json::String(name.to_owned())),
        ("format".to_owned(), Json::String(format.to_owned())),
        (
            "glibc_floor".to_owned(),
            floor
                .as_ref()
                .map_or(Json::Null, |floor| Json::String(floor.render())),
        ),
        ("imports".to_owned(), strings(&imports)),
        (
            "policy".to_owned(),
            Json::String(
                if no_policy {
                    "none"
                } else {
                    "mesh-llm-dynamic-host-v2"
                }
                .to_owned(),
            ),
        ),
        ("rejected_imports".to_owned(), strings(&rejected)),
    ];
    if let Some(identity) = identity {
        if binary_identity::digest(Path::new(&binary))? != identity {
            return Err("host binary changed during dependency inspection".to_owned());
        }
        fields.push(("binary_sha256".to_owned(), Json::String(identity)));
    }
    let report = Json::Object(fields);
    if let Some(path) = parsed.value("--report") {
        write_report(path, &report)?;
    }
    let stdout = format!("{}\n", dumps(&report, ASCII));
    if !rejected.is_empty() {
        let stderr = format!("host dependency policy rejected: {}\n", rejected.join(", "));
        return Ok(CheckReport::failure(stdout, stderr));
    }
    if let (Some(max), Some(floor)) = (max_glibc, floor)
        && floor > max
    {
        let stderr = format!(
            "{name} needs GLIBC_{} but the declared floor is {}. Raising the floor drops \
             Linux distributions that were supported before; see mesh-llm#1522.\n",
            floor.render(),
            max.render()
        );
        return Ok(CheckReport::failure(stdout, stderr));
    }
    Ok(CheckReport::success(stdout))
}

fn argv<const N: usize>(words: [&str; N]) -> Vec<String> {
    words.map(str::to_owned).to_vec()
}

/// `inspect_dependencies(path, format_name)`.
fn inspect(
    binary: &str,
    format: Option<&str>,
    tools: &dyn Toolchain,
) -> Result<(&'static str, Vec<String>), String> {
    let format = match format {
        Some("elf") => "elf",
        Some("macho") => "macho",
        Some(_) => "pe",
        None => binary_format(binary)?,
    };
    let imports = match format {
        "elf" => elf_imports(&run_tool(tools, &argv(["readelf", "-d", binary]))?),
        "macho" => macho_imports(&run_tool(tools, &argv(["otool", "-L", binary]))?),
        _ if tools.which("llvm-readobj") => pe_imports(&run_tool(
            tools,
            &argv(["llvm-readobj", "--coff-imports", binary]),
        )?),
        _ => pe_imports(&run_tool(tools, &argv(["objdump", "-p", binary]))?),
    };
    Ok((format, imports))
}

/// `binary_format(path)` from the first four bytes.
fn binary_format(binary: &str) -> Result<&'static str, String> {
    let bytes = std::fs::read(binary).map_err(|error| os_error_text(&error, binary))?;
    let header = &bytes[..bytes.len().min(4)];
    const MACHO: [&[u8]; 6] = [
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    ];
    if header == b"\x7fELF" {
        Ok("elf")
    } else if header.starts_with(b"MZ") {
        Ok("pe")
    } else if MACHO.contains(&header) {
        Ok("macho")
    } else {
        Err(format!("unsupported host executable format: {binary}"))
    }
}

fn resolve_max_glibc(
    value: Option<&str>,
    floor_file: FloorFile<'_>,
) -> Result<Option<Version>, String> {
    match value {
        None => Ok(None),
        Some("declared") => {
            let path = floor_file()?;
            let shown = path.to_string_lossy().into_owned();
            declared_floor(&path, &shown).map(Some)
        }
        Some(text) => parse_version(text).map(Some),
    }
}

/// `report.parent.mkdir(parents=True, exist_ok=True)` then `write_text`.
fn write_report(path: &str, report: &Json) -> Result<(), String> {
    let shown = python_path_display(Path::new(path));
    let parent = match shown.rsplit_once('/') {
        Some(("", _)) => "/".to_owned(),
        Some((parent, _)) => parent.to_owned(),
        None => ".".to_owned(),
    };
    std::fs::create_dir_all(&parent).map_err(|error| os_error_text(&error, &parent))?;
    std::fs::write(&shown, format!("{}\n", dumps(report, INDENTED)))
        .map_err(|error| os_error_text(&error, &shown))
}

#[cfg(test)]
#[path = "host_dependencies_tests.rs"]
mod tests;
