use super::{Error, contract::Component};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component as PathComponent, Path, PathBuf},
};

pub(super) fn framework(root: &Path, location: &(Component, Component)) -> Result<PathBuf, Error> {
    let framework = root.join(location.0.as_str()).join(location.1.as_str());
    let resolved_root = resolve(root, &mut BTreeSet::new())?;
    let resolved = resolve(&framework, &mut BTreeSet::new())?;
    if resolved == resolved_root || !resolved.starts_with(&resolved_root) {
        return Err(Error::Contract(format!(
            "XCFramework library escapes its root: {}",
            framework.display()
        )));
    }
    if !framework.is_dir() {
        return Err(Error::Contract(format!(
            "XCFramework library is missing: {}",
            framework.display()
        )));
    }
    Ok(framework)
}

fn resolve(path: &Path, links: &mut BTreeSet<PathBuf>) -> Result<PathBuf, Error> {
    let absolute = std::path::absolute(path).map_err(|error| Error::io(path, error))?;
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            PathComponent::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            PathComponent::RootDir => resolved.push(component.as_os_str()),
            PathComponent::CurDir => {}
            PathComponent::ParentDir => {
                resolved.pop();
            }
            PathComponent::Normal(name) => {
                resolved.push(name);
                match fs::symlink_metadata(&resolved) {
                    Ok(metadata) if metadata.file_type().is_symlink() => {
                        if !links.insert(resolved.clone()) {
                            return Err(Error::Contract(format!(
                                "symlink loop resolving {}",
                                resolved.display()
                            )));
                        }
                        let link = resolved.clone();
                        let target =
                            fs::read_link(&link).map_err(|error| Error::io(&link, error))?;
                        resolved.pop();
                        resolved = resolve(&resolved.join(target), links)?;
                        links.remove(&link);
                    }
                    Ok(_) => {}
                    Err(error)
                        if error.kind() == std::io::ErrorKind::NotFound
                            || error.kind() == std::io::ErrorKind::NotADirectory => {}
                    Err(error) => return Err(Error::io(&resolved, error)),
                }
            }
        }
    }
    Ok(resolved)
}

pub(super) fn name(framework: &Path) -> Result<&std::ffi::OsStr, Error> {
    framework
        .file_stem()
        .ok_or_else(|| Error::Contract("framework filename has no stem".into()))
}

pub(super) fn binary(framework: &Path) -> Result<PathBuf, Error> {
    let binary = framework.join(name(framework)?);
    if !binary.exists() || !binary.is_file() {
        return Err(Error::Contract(format!(
            "XCFramework binary is missing: {}",
            binary.display()
        )));
    }
    Ok(binary)
}

pub(super) fn macos(framework: &Path) -> Result<(), Error> {
    let name = name(framework)?;
    let binary_target = Path::new("Versions/Current").join(name);
    let links = [
        (Path::new("Versions/Current"), Path::new("A")),
        (Path::new(name), binary_target.as_path()),
        (Path::new("Headers"), Path::new("Versions/Current/Headers")),
        (Path::new("Modules"), Path::new("Versions/Current/Modules")),
        (
            Path::new("Resources"),
            Path::new("Versions/Current/Resources"),
        ),
    ];
    for (relative, target) in links {
        let path = framework.join(relative);
        if !path.is_symlink() {
            return Err(Error::Contract(format!(
                "macOS framework is not versioned; missing symlink: {}",
                path.display()
            )));
        }
        let actual = fs::read_link(&path).map_err(|error| Error::io(&path, error))?;
        if actual.as_os_str() != target.as_os_str() {
            return Err(Error::Contract(format!(
                "unexpected symlink target for {}: {actual:?} != {target:?}",
                path.display()
            )));
        }
    }
    let version = framework.join("Versions/A");
    let required = [
        version.join(name),
        version.join("Headers"),
        version.join("Modules/module.modulemap"),
        version.join("Resources/Info.plist"),
        version.join("Resources/PrivacyInfo.xcprivacy"),
    ];
    for path in required {
        if !path.exists() {
            return Err(Error::Contract(format!(
                "macOS framework versioned layout is incomplete: {}",
                path.display()
            )));
        }
    }
    Ok(())
}
