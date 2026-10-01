use super::glibc::{Version, elf_floor, parse_version};
use super::runtime_package_manifest::Package;
use super::toolchain::{Exit, Toolchain, decode};
use std::io::Read;
use std::path::Path;

const GLIBC_POLICY: &str = include_str!("../../../../scripts/linux-glibc-floor.txt");

fn embedded_floor() -> Result<Version, String> {
    GLIBC_POLICY
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .ok_or_else(|| "embedded GLIBC policy has no floor".to_owned())
        .and_then(parse_version)
}

fn inspect(tools: &dyn Toolchain, flag: &str, path: &Path) -> Result<String, String> {
    let args = [
        "readelf".to_owned(),
        flag.to_owned(),
        path.to_string_lossy().into_owned(),
    ];
    let captured = tools.output(&args).map_err(|error| error.to_string())?;
    match captured.exit {
        Exit::Code(0) => decode(&captured.stdout),
        Exit::Code(code) => Err(format!(
            "Command '['readelf', '{flag}', '{}']' returned non-zero exit status {code}.",
            path.display()
        )),
        Exit::Signal(signal) => Err(format!("readelf terminated with signal {signal}")),
    }
}

fn actual_floor(output: &str) -> Option<Version> {
    let mut floor = elf_floor(output);
    if output
        .split_once("Version needs section")
        .is_some_and(|(_, needs)| needs.contains("GLIBC_ABI_DT_RELR"))
    {
        floor = floor.max(parse_version("2.36").ok());
    }
    floor
}

fn dynamic(output: &str) -> (Vec<String>, Vec<String>) {
    let mut needed = Vec::new();
    let mut search = Vec::new();
    for line in output.lines() {
        let Some(start) = line.rfind('[') else {
            continue;
        };
        let Some(end) = line.rfind(']') else {
            continue;
        };
        let value = &line[start + 1..end];
        if line.contains("(NEEDED)") {
            needed.push(value.to_owned());
        } else if line.contains("(RPATH)") || line.contains("(RUNPATH)") {
            search.extend(
                value
                    .split(':')
                    .filter(|part| !part.is_empty())
                    .map(str::to_owned),
            );
        }
    }
    (needed, search)
}

fn ldd_resolves(
    package: &Package,
    tools: &dyn Toolchain,
    relative: &str,
    needed: &[String],
) -> Result<(), String> {
    if !cfg!(target_os = "linux") || needed.is_empty() {
        return Ok(());
    }
    if !tools.which("ldd") {
        return Err("ldd is required to verify Linux packaged dependency resolution".to_owned());
    }
    let args = [
        "ldd".to_owned(),
        package.root.join(relative).to_string_lossy().into_owned(),
    ];
    let captured = tools
        .capture_without_ld_library_path(&args)
        .map_err(|error| error.to_string())?;
    if !matches!(captured.exit, Exit::Code(0)) {
        return Err(format!("ldd failed for {relative}"));
    }
    let output = decode(&captured.output)?;
    let root = package
        .root
        .canonicalize()
        .map_err(|error| error.to_string())?;
    for name in needed {
        let resolved = output
            .lines()
            .filter_map(|line| line.trim().strip_prefix(name))
            .find_map(|rest| {
                rest.strip_prefix(' ')
                    .and_then(|rest| rest.trim_start().strip_prefix("=>"))
                    .and_then(|value| value.split_whitespace().next())
            });
        let Some(path) = resolved else {
            return Err(format!(
                "{relative} ldd output is missing packaged dependency {name}"
            ));
        };
        if path == "not" {
            return Err(format!(
                "{relative} does not resolve packaged dependency {name} without LD_LIBRARY_PATH"
            ));
        }
        if !Path::new(path)
            .canonicalize()
            .is_ok_and(|path| path.starts_with(&root) && path != root)
        {
            return Err(format!(
                "{relative} resolves packaged dependency {name} outside artifact: {path}"
            ));
        }
    }
    Ok(())
}

fn version_consistency(
    package: &Package,
    tools: &dyn Toolchain,
    entries: &[String],
) -> Result<(), String> {
    let mut actual: Option<Version> = None;
    for relative in entries {
        actual = actual.max(actual_floor(&inspect(
            tools,
            "-V",
            &package.root.join(relative),
        )?));
    }
    let Some(declared) = &package.min_glibc else {
        return Ok(());
    };
    if actual.as_ref().map(Version::render).as_ref() != Some(declared) {
        let observed = actual
            .as_ref()
            .map(Version::render)
            .map_or("None".to_owned(), |text| format!("'{text}'"));
        return Err(format!(
            "runtime platform min_glibc '{declared}' does not match packaged ELF requirement {observed}; refusing to publish a misleading floor"
        ));
    }
    Ok(())
}

fn verify_cuda(package: &Package, tools: &dyn Toolchain) -> Result<(), String> {
    if package.backend != "cuda" {
        return Ok(());
    }
    let lib = package.root.join("lib").to_string_lossy().into_owned();
    let scan = package.root.join("tools").to_string_lossy().into_owned();
    let args: Vec<String> = [
        "verify",
        "--lib-dir",
        &lib,
        "--scan-dir",
        &scan,
        "--arch",
        &package.arch,
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    let report = super::linux_deps::run(&args, tools);
    if report.code != 0 {
        return Err(report.stderr.trim_end().to_owned());
    }
    let primary = package
        .primary
        .rsplit('/')
        .next()
        .unwrap_or(&package.primary);
    let args: Vec<String> = [
        "order",
        "--lib-dir",
        &lib,
        "--scan-dir",
        &scan,
        "--arch",
        &package.arch,
        "--primary",
        primary,
    ]
    .iter()
    .map(|word| (*word).to_owned())
    .collect();
    let report = super::linux_deps::run(&args, tools);
    if report.code != 0 {
        return Err(report.stderr.trim_end().to_owned());
    }
    let expected = report.stdout.trim_end_matches('\n');
    let actual = package
        .libraries
        .iter()
        .filter_map(|path| path.rsplit('/').next())
        .collect::<Vec<_>>()
        .join("\n");
    if expected != actual {
        return Err("Linux CUDA runtime libraries are not in dependency-first order".to_owned());
    }
    Ok(())
}

fn verify_search_paths(package: &Package, tools: &dyn Toolchain) -> Result<(), String> {
    let names: Vec<&str> = package
        .libraries
        .iter()
        .filter_map(|path| path.rsplit('/').next())
        .collect();
    for relative in package.entries() {
        if !relative
            .rsplit('/')
            .next()
            .unwrap_or(relative)
            .contains(".so")
            && !package.tools.iter().any(|tool| tool == relative)
        {
            continue;
        }
        let path = package.root.join(relative);
        let (needed, search) = dynamic(&inspect(tools, "-d", &path)?);
        let packaged: Vec<String> = needed
            .iter()
            .filter(|name| names.contains(&name.rsplit('/').next().unwrap_or(name)))
            .cloned()
            .collect();
        for entry in &search {
            if ["/home/runner/work", ".deps/llama-build", "build-stage-abi"]
                .iter()
                .any(|token| entry.contains(token))
            {
                return Err(format!(
                    "{relative} contains build-directory runtime search path: {entry}"
                ));
            }
            if entry.starts_with('/') {
                return Err(format!(
                    "{relative} contains absolute runtime search path: {entry}"
                ));
            }
        }
        let tool = package.tools.iter().any(|tool| tool == relative);
        let expected = if tool { "$ORIGIN/../lib" } else { "$ORIGIN" };
        if !packaged.is_empty()
            && (package.backend != "cuda"
                || package.relocatable.iter().any(|name| name == relative)
                || tool)
            && !search.iter().any(|part| part == expected)
        {
            return Err(format!(
                "{relative} needs packaged libraries ({}) but is missing {expected} RPATH/RUNPATH",
                packaged.join(", ")
            ));
        }
        if package.backend != "cuda" {
            ldd_resolves(package, tools, relative, &packaged)?;
        }
    }
    Ok(())
}

pub(super) fn verify(package: &Package, tools: &dyn Toolchain) -> Result<(), String> {
    if !tools.which("readelf") {
        return Err(
            "readelf is required to verify Linux native runtime shared libraries".to_owned(),
        );
    }
    let floor = embedded_floor()?.render();
    let mut elf_entries = Vec::new();
    for relative in package.entries() {
        let path = package.root.join(relative);
        let mut header = [0_u8; 4];
        let mut file = std::fs::File::open(&path).map_err(|error| error.to_string())?;
        if file.read(&mut header).map_err(|error| error.to_string())? == 4 && header == *b"\x7fELF"
        {
            let args = [
                path.to_string_lossy().into_owned(),
                "--no-import-policy".to_owned(),
                "--max-glibc".to_owned(),
                floor.clone(),
            ];
            let mut floor_path = || Err("embedded floor does not require a file".to_owned());
            let report = super::host_dependencies::run(&args, tools, &mut floor_path);
            if report.code != 0 {
                return Err(report.stderr.trim_end_matches('\n').to_owned());
            }
            elf_entries.push(relative.to_owned());
        }
    }
    version_consistency(package, tools, &elf_entries)?;
    verify_cuda(package, tools)?;
    verify_search_paths(package, tools)
}

#[cfg(test)]
mod tests {
    #[test]
    fn embedded_policy_matches_checked_in_policy() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/linux-glibc-floor.txt");
        assert_eq!(super::GLIBC_POLICY, std::fs::read_to_string(path).unwrap());
        assert!(super::embedded_floor().is_ok());
    }
}
