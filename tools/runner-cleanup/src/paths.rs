use super::Error;
use std::{
    collections::VecDeque,
    ffi::OsString,
    path::{Component, Path, PathBuf},
};

pub(super) fn absolute(path: &Path) -> Result<PathBuf, Error> {
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    })
}

pub(super) fn resolve(path: &Path) -> Result<PathBuf, Error> {
    let absolute = absolute(path)?;
    let mut resolved = PathBuf::new();
    let mut pending: VecDeque<OsString> = absolute
        .components()
        .map(|part| part.as_os_str().to_owned())
        .collect();
    let mut hops = 0;
    while let Some(part) = pending.pop_front() {
        let component = Path::new(&part).components().next();
        match component {
            Some(Component::ParentDir) => {
                resolved.pop();
            }
            Some(Component::CurDir) | None => {}
            Some(Component::RootDir | Component::Prefix(_)) => resolved.push(&part),
            Some(Component::Normal(_)) => {
                let next = resolved.join(&part);
                match std::fs::read_link(&next) {
                    Ok(target) => {
                        hops += 1;
                        if hops > 40 {
                            return Err(Error::Input("symlink resolution limit exceeded"));
                        }
                        if target.is_absolute() {
                            resolved.clear();
                        }
                        for component in target.components().rev() {
                            pending.push_front(component.as_os_str().to_owned());
                        }
                    }
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::NotFound | std::io::ErrorKind::InvalidInput
                        ) =>
                    {
                        resolved = next
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        }
    }
    Ok(resolved)
}
