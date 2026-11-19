use super::{Error, Plan};
use std::path::{Component, Path};

pub(super) struct Admitted<'a>(&'a Plan);

impl<'a> Admitted<'a> {
    pub(super) fn plan(&self) -> &'a Plan {
        self.0
    }
}

pub(crate) fn validate_path(base: &Path, path: &Path) -> Result<(), Error> {
    if path == base
        || !path.starts_with(base)
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(Error::Escape(path.to_owned()));
    }
    for parent in path.ancestors().skip(1) {
        if parent == base {
            break;
        }
        if parent.is_symlink() {
            return Err(Error::ParentSymlink(parent.to_owned()));
        }
    }
    Ok(())
}

pub(super) fn admit(plan: &Plan) -> Result<Admitted<'_>, Error> {
    for target in &plan.targets {
        validate_path(&target.base, &target.path)?;
    }
    Ok(Admitted(plan))
}
