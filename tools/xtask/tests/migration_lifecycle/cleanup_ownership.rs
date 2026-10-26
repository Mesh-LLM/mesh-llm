use std::{
    fs, io,
    path::{Component, Path},
};

const MARKER: &str = ".runner-cleanup-fixture-owner";

pub(crate) fn validate(root: &Path, path: &Path) -> io::Result<()> {
    let refusal = || {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "unowned cleanup fixture path",
        )
    };
    if !root.is_absolute()
        || root.canonicalize()? != root
        || root.parent().is_none()
        || path == root
        || !path.starts_with(root)
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err(refusal());
    }
    let marker = root.join(MARKER);
    if !fs::symlink_metadata(&marker)?.is_file()
        || fs::read(marker)? != root.as_os_str().as_encoded_bytes()
    {
        return Err(refusal());
    }
    for parent in path.ancestors().skip(1) {
        match fs::symlink_metadata(parent) {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => return Err(refusal()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}
