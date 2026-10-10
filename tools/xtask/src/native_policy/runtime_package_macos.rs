use super::runtime_package_manifest::Package;
use super::toolchain::{Exit, Toolchain, decode};
use std::path::{Path, PathBuf};

fn inspect(tools: &dyn Toolchain, flag: &str, file: &Path) -> Result<String, String> {
    let args = [
        "otool".to_owned(),
        flag.to_owned(),
        file.to_string_lossy().into_owned(),
    ];
    let captured = tools.output(&args).map_err(|error| error.to_string())?;
    match captured.exit {
        Exit::Code(0) => decode(&captured.stdout),
        Exit::Code(code) => Err(format!(
            "Command '['otool', '{flag}', '{}']' returned non-zero exit status {code}.",
            file.display()
        )),
        Exit::Signal(signal) => Err(format!("otool terminated with signal {signal}")),
    }
}

pub(super) fn verify(package: &Package, tools: &dyn Toolchain) -> Result<(), String> {
    let mut dylib_found = false;
    let mut pending = vec![(package.root.clone(), Vec::<PathBuf>::new())];
    while let Some((directory, ancestors)) = pending.pop() {
        let resolved = directory
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if ancestors.contains(&resolved) {
            return Err(format!(
                "native runtime directory cycle: {}",
                directory.display()
            ));
        }
        let mut descendants = ancestors;
        descendants.push(resolved);
        for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                pending.push((path, descendants.clone()));
            } else if path
                .extension()
                .is_some_and(|extension| extension == "dylib")
            {
                dylib_found = true;
            }
        }
    }
    if !dylib_found {
        return Ok(());
    }
    if !tools.which("otool") {
        return Err("otool is required to verify macOS native runtime dylibs".to_owned());
    }
    let names: Vec<&str> = package
        .libraries
        .iter()
        .filter_map(|path| path.rsplit('/').next())
        .collect();
    for relative in package.entries() {
        if !relative.ends_with(".dylib") && !package.tools.iter().any(|tool| tool == relative) {
            continue;
        }
        let file = package.root.join(relative);
        let dependencies = inspect(tools, "-L", &file)?;
        for dependency in dependencies
            .lines()
            .skip(1)
            .filter_map(|line| line.split_whitespace().next())
        {
            if dependency.starts_with('/')
                && names.contains(&dependency.rsplit('/').next().unwrap_or(dependency))
            {
                return Err(format!(
                    "{relative} depends on absolute packaged dylib path: {dependency}"
                ));
            }
        }
        let expected = if package.tools.iter().any(|tool| tool == relative) {
            "@loader_path/../lib"
        } else {
            "@loader_path"
        };
        if !inspect(tools, "-l", &file)?
            .lines()
            .filter_map(|line| {
                let fields: Vec<&str> = line.split_whitespace().collect();
                (fields.first() == Some(&"path"))
                    .then(|| fields.get(1).copied())
                    .flatten()
            })
            .any(|path| path == expected)
        {
            return Err(format!("{relative} is missing {expected} LC_RPATH"));
        }
    }
    Ok(())
}
