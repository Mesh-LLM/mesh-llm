//! Putting a staged plugin package in place: checking that its links stay
//! inside it, replacing the installed copy with a rollback, and moving a tree
//! across filesystems without following links.

use std::{
    ffi::OsString,
    fs,
    path::{Component, Path},
};

use anyhow::{Context, Result, bail};

/// The most links followed while resolving one package link, as Linux's
/// `MAXSYMLINKS`.
const MAX_LINK_HOPS: usize = 40;

/// Refuse a staged package whose symlinks point outside it. Links survive the
/// move into place, so an absolute link, or one that resolves above the
/// package root directly or through other links, would let the installed
/// plugin's files stand for files elsewhere: the web UI asset route and the
/// skill installer read through them. Links that stay inside are kept.
pub(crate) fn reject_escaping_links(plugin_root: &Path) -> Result<()> {
    let root_type = fs::symlink_metadata(plugin_root)
        .with_context(|| format!("inspect plugin package root {}", plugin_root.display()))?
        .file_type();
    if !root_type.is_dir() {
        bail!(
            "plugin package root {} is not a directory",
            plugin_root.display()
        );
    }
    reject_escaping_links_in(plugin_root, plugin_root, &mut Vec::new())
}

/// Walk `dir`, which is `relative` below `root`, checking every link in it.
fn reject_escaping_links_in(root: &Path, dir: &Path, relative: &mut Vec<OsString>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read directory {}", dir.display()))? {
        let entry = entry.with_context(|| format!("read directory entry in {}", dir.display()))?;
        let path = entry.path();
        // `DirEntry::file_type` describes the entry itself, not a link target,
        // so the walk never leaves the package through a linked directory.
        let file_type = entry
            .file_type()
            .with_context(|| format!("read file type for {}", path.display()))?;
        if file_type.is_symlink() {
            reject_escaping_link(root, relative, &path)?;
        } else if file_type.is_dir() {
            relative.push(entry.file_name());
            reject_escaping_links_in(root, &path, relative)?;
            relative.pop();
        }
    }
    Ok(())
}

enum LinkProblem {
    Absolute,
    Escapes,
    Missing,
    TooManyHops,
    Unresolvable,
}

fn reject_escaping_link(root: &Path, parent: &[OsString], link: &Path) -> Result<()> {
    let target = fs::read_link(link).with_context(|| format!("read symlink {}", link.display()))?;
    // `link` itself is the first link followed.
    let problem = match resolve_inside(root, parent.to_vec(), &target, &mut 1) {
        // Inside the package, but the system must also be able to open it:
        // the walk above reads `index.js/` as `index.js`, for one.
        Ok(_) if fs::metadata(link).is_err() => LinkProblem::Unresolvable,
        Ok(_) => return Ok(()),
        Err(problem) => problem,
    };
    let (link, target) = (link.display(), target.display());
    match problem {
        LinkProblem::Absolute => {
            bail!("plugin package link {link} points to absolute path {target}")
        }
        LinkProblem::Escapes => {
            bail!("plugin package link {link} -> {target} escapes the plugin package")
        }
        LinkProblem::Missing => {
            bail!("plugin package link {link} points to {target}, which is not in the package")
        }
        LinkProblem::TooManyHops => {
            bail!("plugin package link {link} -> {target} goes through too many links")
        }
        LinkProblem::Unresolvable => {
            bail!("plugin package link {link} -> {target} does not resolve")
        }
    }
}

/// Resolve `target` from the package directory `current` (a path below
/// `root`) one component at a time, following links on the way, and refuse
/// any step above the root. Canonicalizing would also accept a path that
/// leaves the root and comes back in through the root's own name; that path
/// breaks once the package is installed under another name, so resolution
/// here never looks above the root.
fn resolve_inside(
    root: &Path,
    mut current: Vec<OsString>,
    target: &Path,
    hops: &mut usize,
) -> std::result::Result<Vec<OsString>, LinkProblem> {
    for component in target.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => return Err(LinkProblem::Absolute),
            Component::CurDir => {}
            Component::ParentDir => {
                // `file/..` does not resolve, so `..` must follow a directory.
                // Links met on the way are already resolved, so this is the
                // entry itself.
                if !current.is_empty() && !package_path(root, &current).is_dir() {
                    return Err(LinkProblem::Missing);
                }
                current.pop().ok_or(LinkProblem::Escapes)?;
            }
            Component::Normal(name) => {
                current.push(name.to_os_string());
                let path = package_path(root, &current);
                let metadata = fs::symlink_metadata(&path).map_err(|_| LinkProblem::Missing)?;
                if metadata.file_type().is_symlink() {
                    *hops += 1;
                    if *hops > MAX_LINK_HOPS {
                        return Err(LinkProblem::TooManyHops);
                    }
                    let next = fs::read_link(&path).map_err(|_| LinkProblem::Missing)?;
                    current.pop();
                    current = resolve_inside(root, current, &next, hops)?;
                }
            }
        }
    }
    Ok(current)
}

fn package_path(root: &Path, relative: &[OsString]) -> std::path::PathBuf {
    relative
        .iter()
        .fold(root.to_path_buf(), |path, name| path.join(name))
}

/// Put the staged tree `from` at `to`. An installed copy at `to` is moved
/// aside first and put back if the new one cannot be placed; whenever the
/// copy moved aside is the only complete one left, it is kept and the error
/// says where.
pub(crate) fn replace_plugin_dir(from: &Path, to: &Path, plugin_name: &str) -> Result<()> {
    let install_parent = to.parent().unwrap_or_else(|| Path::new("."));
    remove_stale_backups(install_parent, plugin_name);
    if !to.exists() {
        return place_staged_tree(from, to)
            .with_context(|| format!("install plugin to {}", to.display()));
    }
    // A fixed prefix: a long plugin name plus a prefix and random suffix
    // could exceed the file name limit.
    let backup_parent = tempfile::Builder::new()
        .prefix(BACKUP_PREFIX)
        .tempdir_in(install_parent)
        .with_context(|| format!("create plugin install backup for {}", to.display()))?;
    let backup_dir = backup_parent.path().join(plugin_name);
    let moved = place_tree(to, &backup_dir)
        .with_context(|| format!("backup previous plugin install {}", to.display()))?;
    if moved == Placed::Copied
        && let Err(error) = remove_tree(to)
    {
        // `to` may be partly removed, so the backup is the only complete copy.
        let kept = backup_parent.keep().join(plugin_name);
        return Err(anyhow::Error::new(error).context(format!(
            "backup previous plugin install {}: it was copied but could not be removed, and \
             the previous install is kept at {}",
            to.display(),
            kept.display()
        )));
    }
    if let Err(error) = place_staged_tree(from, to) {
        let error = error.context(format!("replace plugin install {}", to.display()));
        if let Err(restore_error) = restore_previous_install(&backup_dir, to) {
            let kept = backup_parent.keep().join(plugin_name);
            return Err(error.context(format!(
                "restoring the previous install also failed ({restore_error:#}); it is kept at {}",
                kept.display()
            )));
        }
        let _ = remove_tree(backup_parent.path());
        return Err(error);
    }
    // The previous install is no longer needed. `TempDir` cannot remove the
    // read-only directories it may hold, so remove it here; if that fails
    // too, only its files are left behind.
    let _ = remove_tree(backup_parent.path());
    Ok(())
}

const BACKUP_PREFIX: &str = ".previous-";

/// Remove the backups an earlier replace of `plugin_name` could not delete:
/// on Windows a running executable cannot be deleted, so updating a plugin
/// while the node still runs it leaves its backup behind. Only `.previous-*`
/// directories holding this plugin's directory are touched, so another
/// plugin's replace in progress keeps its backup; one that still cannot be
/// removed is left for a later replace.
fn remove_stale_backups(install_parent: &Path, plugin_name: &str) {
    let Ok(entries) = fs::read_dir(install_parent) else {
        return;
    };
    for entry in entries.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with(BACKUP_PREFIX)
        {
            continue;
        }
        let backup = entry.path();
        let is_dir = fs::symlink_metadata(&backup).is_ok_and(|metadata| metadata.is_dir());
        if is_dir && fs::symlink_metadata(backup.join(plugin_name)).is_ok() {
            let _ = remove_tree(&backup);
        }
    }
}

/// Put the previous install back after a failed replacement. The failed move
/// can leave the new tree at `to`; that is removed first so the two trees are
/// not merged.
fn restore_previous_install(backup_dir: &Path, to: &Path) -> Result<()> {
    if fs::symlink_metadata(to).is_ok() {
        remove_tree(to)
            .with_context(|| format!("remove partial plugin install {}", to.display()))?;
    }
    // A copied backup is left for the caller to remove with its directory.
    place_tree(backup_dir, to)
        .map(drop)
        .with_context(|| format!("restore {}", to.display()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placed {
    Renamed,
    Copied,
}

/// Put the tree `from` at `to`, which must not exist yet: by a rename, or
/// across filesystems by a copy, which leaves `from` for the caller to
/// remove. A copy that fails is removed again and `from` is untouched; if
/// even that removal fails, the error says so.
fn place_tree(from: &Path, to: &Path) -> Result<Placed> {
    if fs::rename(from, to).is_ok() {
        return Ok(Placed::Renamed);
    }
    // If `to` exists, for example because another install of the same plugin
    // just finished, `create_private_dir` fails: stop rather than merge into
    // it or remove it afterwards.
    create_private_dir(to)?;
    let Err(error) = copy_dir_contents(from, to).and_then(|()| copy_dir_mode(from, to)) else {
        return Ok(Placed::Copied);
    };
    if let Err(cleanup_error) = remove_tree(to) {
        return Err(error.context(format!(
            "the partial copy at {} could not be removed ({cleanup_error})",
            to.display()
        )));
    }
    Err(error)
}

/// Place a staged package. After a copy its source is still in the staging
/// directory, which is removed afterwards anyway, so failing to remove the
/// source here does not fail an install whose files are all in place.
fn place_staged_tree(from: &Path, to: &Path) -> Result<()> {
    if place_tree(from, to)? == Placed::Copied {
        let _ = remove_tree(from);
    }
    Ok(())
}

/// Remove a tree this module is discarding: a moved source or a partial copy.
/// A package may hold read-only directories, whose entries cannot be removed,
/// so when the first attempt fails, the tree's directories are made writable
/// by their owner and it is tried again. Installed trees are never passed
/// here, so their modes are not changed.
fn remove_tree(path: &Path) -> std::io::Result<()> {
    let Err(error) = fs::remove_dir_all(path) else {
        return Ok(());
    };
    #[cfg(unix)]
    if make_dirs_writable(path).is_ok() {
        return fs::remove_dir_all(path);
    }
    Err(error)
}

#[cfg(unix)]
fn make_dirs_writable(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    // Never follow a link out of the tree.
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() {
        return Ok(());
    }
    let mode = metadata.permissions().mode();
    fs::set_permissions(path, fs::Permissions::from_mode(mode | 0o700))?;
    for entry in fs::read_dir(path)? {
        make_dirs_writable(&entry?.path())?;
    }
    Ok(())
}

/// Copy the contents of a plugin tree into the empty directory `to` the way
/// `fs::rename` would move them: symlinks stay symlinks. Following them would
/// copy whatever they point at, such as a key outside the package, into the
/// plugin directory, where the web UI asset route's containment check no
/// longer sees a link to refuse.
fn copy_dir_contents(from: &Path, to: &Path) -> Result<()> {
    for entry in fs::read_dir(from).with_context(|| format!("read directory {}", from.display()))? {
        let entry = entry?;
        let from_path = entry.path();
        let to_path = to.join(entry.file_name());
        // `DirEntry::file_type` describes the entry itself, not a link target.
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            copy_symlink(&from_path, &to_path)?;
        } else if file_type.is_dir() {
            create_private_dir(&to_path)?;
            copy_dir_contents(&from_path, &to_path)?;
            copy_dir_mode(&from_path, &to_path)?;
        } else if file_type.is_file() {
            fs::copy(&from_path, &to_path).with_context(|| {
                format!("copy {} to {}", from_path.display(), to_path.display())
            })?;
        } else {
            bail!(
                "plugin package contains an unsupported file type at {}",
                from_path.display()
            );
        }
    }
    Ok(())
}

/// Create the directory `path`, which must not exist, readable only by its
/// owner until `copy_dir_mode` gives it the source directory's mode: the
/// umask's default would let other users read files copied into a
/// directory that was private in the package.
fn create_private_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    let builder = {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700);
        builder
    };
    #[cfg(not(unix))]
    let builder = fs::DirBuilder::new();
    builder
        .create(path)
        .with_context(|| format!("create directory {}", path.display()))
}

/// Give `to` the mode of `from`, as a rename would keep it. Applied after
/// the contents are copied, so a read-only directory can still be filled.
fn copy_dir_mode(from: &Path, to: &Path) -> Result<()> {
    let permissions = fs::metadata(from)
        .with_context(|| format!("read permissions of {}", from.display()))?
        .permissions();
    fs::set_permissions(to, permissions)
        .with_context(|| format!("set permissions of {}", to.display()))
}

#[cfg(unix)]
fn copy_symlink(from: &Path, to: &Path) -> Result<()> {
    let target = fs::read_link(from).with_context(|| format!("read symlink {}", from.display()))?;
    std::os::unix::fs::symlink(&target, to)
        .with_context(|| format!("create symlink {} -> {}", to.display(), target.display()))
}

#[cfg(not(unix))]
fn copy_symlink(from: &Path, _to: &Path) -> Result<()> {
    bail!(
        "plugin package contains a symlink at {}, which cannot be copied to another filesystem here",
        from.display()
    )
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn a_failed_replacement_restores_the_previous_install_without_merging() -> Result<()> {
        let temp = TempDir::new()?;
        let to = temp.path().join("demo");
        fs::create_dir_all(to.join("lib"))?;
        fs::write(to.join("lib/old.so"), b"old")?;
        fs::write(to.join("demo"), b"old binary")?;
        let backup = temp.path().join("backup").join("demo");
        fs::create_dir_all(backup.parent().unwrap())?;
        fs::rename(&to, &backup)?;
        // What a copy that stopped partway leaves behind: `lib` is a file
        // here but a directory in the backup, so a merge would fail.
        fs::create_dir_all(&to)?;
        fs::write(to.join("lib"), b"new")?;
        fs::write(to.join("new-only"), b"new")?;

        restore_previous_install(&backup, &to)?;

        assert_eq!(fs::read(to.join("lib/old.so"))?, b"old");
        assert_eq!(fs::read(to.join("demo"))?, b"old binary");
        assert!(!to.join("new-only").exists());
        assert!(!backup.exists());
        Ok(())
    }

    #[test]
    fn a_failed_fresh_install_leaves_nothing_behind() -> Result<()> {
        let temp = TempDir::new()?;
        let to = temp.path().join("installed").join("demo");
        fs::create_dir_all(to.parent().unwrap())?;

        let missing = temp.path().join("missing");
        assert!(replace_plugin_dir(&missing, &to, "demo").is_err());
        assert!(fs::symlink_metadata(&to).is_err());
        Ok(())
    }

    #[test]
    fn a_failed_replacement_keeps_the_previous_install() -> Result<()> {
        let temp = TempDir::new()?;
        let install_dir = temp.path().join("installed");
        let to = install_dir.join("demo");
        fs::create_dir_all(&to)?;
        fs::write(to.join("demo"), b"old binary")?;

        let missing = temp.path().join("missing");
        assert!(replace_plugin_dir(&missing, &to, "demo").is_err());
        assert_eq!(fs::read(to.join("demo"))?, b"old binary");
        let leftovers: Vec<_> = fs::read_dir(&install_dir)?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<std::io::Result<_>>()?;
        assert_eq!(leftovers, vec![std::ffi::OsString::from("demo")]);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn copying_a_plugin_tree_keeps_symlinks_as_links() -> Result<()> {
        let temp = TempDir::new()?;
        let secret = temp.path().join("id_ed25519");
        fs::write(&secret, b"private key")?;
        let from = temp.path().join("from");
        fs::create_dir_all(from.join("ui/dist"))?;
        fs::write(from.join("ui/dist/index.js"), b"export {}")?;
        std::os::unix::fs::symlink(&secret, from.join("ui/dist/k"))?;
        std::os::unix::fs::symlink("index.js", from.join("ui/dist/alias.js"))?;
        let outside_dir = temp.path().join("outside");
        fs::create_dir_all(&outside_dir)?;
        fs::write(outside_dir.join("other"), b"outside")?;
        std::os::unix::fs::symlink(&outside_dir, from.join("ui/linked-dir"))?;

        let to = temp.path().join("to");
        create_private_dir(&to)?;
        copy_dir_contents(&from, &to)?;

        let link = to.join("ui/dist/k");
        assert!(fs::symlink_metadata(&link)?.file_type().is_symlink());
        assert_eq!(fs::read_link(&link)?, secret);
        assert_eq!(
            fs::read_link(to.join("ui/dist/alias.js"))?,
            Path::new("index.js")
        );
        let linked_dir = to.join("ui/linked-dir");
        assert!(fs::symlink_metadata(&linked_dir)?.file_type().is_symlink());
        assert_eq!(fs::read(to.join("ui/dist/index.js"))?, b"export {}");
        Ok(())
    }

    #[cfg(unix)]
    fn package_root(temp: &TempDir) -> Result<std::path::PathBuf> {
        let root = temp.path().join("staging").join("demo");
        fs::create_dir_all(root.join("skills/demo-skill"))?;
        fs::write(root.join("skills/demo-skill/SKILL.md"), b"---\n---\n")?;
        fs::write(root.join("demo"), b"")?;
        Ok(root)
    }

    #[cfg(unix)]
    fn link(root: &Path, link: &str, target: impl AsRef<Path>) -> Result<()> {
        let link_path = root.join(link);
        fs::create_dir_all(link_path.parent().unwrap())?;
        std::os::unix::fs::symlink(target, link_path)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn links_that_stay_inside_the_package_are_accepted() -> Result<()> {
        let temp = TempDir::new()?;
        let root = package_root(&temp)?;
        link(&root, "web/skills", "../skills")?;
        link(&root, "demo-alias", "demo")?;
        link(&root, "self", ".")?;
        reject_escaping_links(&root)
    }

    #[cfg(unix)]
    #[test]
    fn links_that_leave_the_package_are_refused() -> Result<()> {
        let outside = TempDir::new()?;
        fs::create_dir_all(outside.path().join("other-skill"))?;
        fs::write(outside.path().join("other-skill/SKILL.md"), b"---\n---\n")?;
        let cases: [(&str, &Path, &str); 3] = [
            // An absolute target, even an existing directory.
            ("skills/outside", outside.path(), "absolute path"),
            // A relative target above the package root.
            ("web/skills", Path::new("../../.."), "escapes"),
            // A target that does not exist in the package.
            ("ui/k", Path::new("../missing"), "not in the package"),
        ];
        for (name, target, expected) in cases {
            let temp = TempDir::new()?;
            let root = package_root(&temp)?;
            link(&root, name, target)?;
            let error = reject_escaping_links(&root).unwrap_err();
            assert!(error.to_string().contains(expected), "{name}: {error}");
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_chain_of_links_is_judged_by_where_it_resolves() -> Result<()> {
        let temp = TempDir::new()?;
        let root = package_root(&temp)?;
        // Read as text, `here/../neighbour` stays inside the package. But
        // `here` is the package root, so `..` leaves it for the staging
        // directory, where `neighbour` exists.
        fs::create_dir_all(temp.path().join("staging/neighbour"))?;
        link(&root, "here", ".")?;
        link(&root, "skills/next", "../here/../neighbour")?;
        let error = reject_escaping_links(&root).unwrap_err();
        assert!(error.to_string().contains("escapes"), "{error}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_package_root_that_is_a_link_is_refused() -> Result<()> {
        let temp = TempDir::new()?;
        let real = temp.path().join("real");
        fs::create_dir_all(&real)?;
        let root = temp.path().join("demo");
        std::os::unix::fs::symlink(&real, &root)?;
        let error = reject_escaping_links(&root).unwrap_err();
        assert!(error.to_string().contains("not a directory"), "{error}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_link_that_only_stays_inside_under_the_staging_name_is_refused() -> Result<()> {
        // The archive's root directory is `release`, but the plugin installs
        // as `demo`. Seen from staging, `../release/file` and
        // `here/../release/file` (with `here -> .`) both land back inside;
        // once installed they point at a sibling plugin.
        for target in ["../release/file", "here/../release/file"] {
            let temp = TempDir::new()?;
            let root = temp.path().join("staging").join("release");
            fs::create_dir_all(&root)?;
            fs::write(root.join("file"), b"")?;
            link(&root, "here", ".")?;
            link(&root, "alias", target)?;
            let error = reject_escaping_links(&root).unwrap_err();
            assert!(error.to_string().contains("escapes"), "{target}: {error}");
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_link_loop_is_refused() -> Result<()> {
        let temp = TempDir::new()?;
        let root = package_root(&temp)?;
        link(&root, "a", "b")?;
        link(&root, "b", "a")?;
        let error = reject_escaping_links(&root).unwrap_err();
        assert!(error.to_string().contains("too many links"), "{error}");
        Ok(())
    }

    #[test]
    fn moving_onto_an_existing_tree_neither_merges_nor_removes_it() -> Result<()> {
        let temp = TempDir::new()?;
        let from = temp.path().join("from");
        fs::create_dir_all(&from)?;
        fs::write(from.join("new"), b"new")?;
        // Another install of the same plugin finished first.
        let to = temp.path().join("to");
        fs::create_dir_all(&to)?;
        fs::write(to.join("keep"), b"keep")?;

        assert!(place_tree(&from, &to).is_err());
        assert_eq!(fs::read(to.join("keep"))?, b"keep");
        assert!(!to.join("new").exists());
        assert_eq!(fs::read(from.join("new"))?, b"new");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn copying_a_plugin_tree_keeps_directory_modes() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let temp = TempDir::new()?;
        let from = temp.path().join("from");
        fs::create_dir_all(from.join("private"))?;
        fs::create_dir_all(from.join("shared"))?;
        fs::write(from.join("private/secret"), b"secret")?;
        fs::set_permissions(
            from.join("private/secret"),
            fs::Permissions::from_mode(0o644),
        )?;
        fs::set_permissions(from.join("private"), fs::Permissions::from_mode(0o700))?;
        fs::set_permissions(from.join("shared"), fs::Permissions::from_mode(0o755))?;
        fs::set_permissions(&from, fs::Permissions::from_mode(0o750))?;

        let to = temp.path().join("to");
        create_private_dir(&to)?;
        copy_dir_contents(&from, &to)?;
        copy_dir_mode(&from, &to)?;

        let mode =
            |path: &Path| -> Result<u32> { Ok(fs::metadata(path)?.permissions().mode() & 0o777) };
        assert_eq!(mode(&to)?, 0o750);
        assert_eq!(mode(&to.join("private"))?, 0o700);
        assert_eq!(mode(&to.join("shared"))?, 0o755);
        assert_eq!(fs::read(to.join("private/secret"))?, b"secret");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_link_through_a_file_is_refused() -> Result<()> {
        let temp = TempDir::new()?;
        let root = package_root(&temp)?;
        // `demo` is the plugin's executable, a regular file.
        link(&root, "alias", "demo/../skills")?;
        let error = reject_escaping_links(&root).unwrap_err();
        assert!(error.to_string().contains("not in the package"), "{error}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_link_the_system_cannot_open_is_refused() -> Result<()> {
        // Read component by component, `index.js/` is `index.js`, but a
        // trailing slash asks for a directory and `index.js` is a file.
        let temp = TempDir::new()?;
        let root = package_root(&temp)?;
        link(&root, "alias.js", "demo/")?;
        let error = reject_escaping_links(&root).unwrap_err();
        assert!(error.to_string().contains("does not resolve"), "{error}");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_chain_counts_its_first_link() -> Result<()> {
        // The system follows at most 40 links for one path, counting the
        // first, so a chain of 40 opens and a chain of 41 does not.
        for (length, accepted) in [(40, true), (41, false)] {
            let temp = TempDir::new()?;
            let root = package_root(&temp)?;
            for index in 1..length {
                link(&root, &format!("l{index}"), format!("l{}", index + 1))?;
            }
            link(&root, &format!("l{length}"), "demo")?;
            let result = reject_escaping_links(&root);
            assert_eq!(result.is_ok(), accepted, "{length}: {result:?}");
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn read_only_directories_do_not_block_removing_a_discarded_tree() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let temp = TempDir::new()?;
        let tree = temp.path().join("tree");
        fs::create_dir_all(tree.join("locked/inner"))?;
        fs::write(tree.join("locked/inner/file"), b"")?;
        fs::set_permissions(tree.join("locked/inner"), fs::Permissions::from_mode(0o555))?;
        fs::set_permissions(tree.join("locked"), fs::Permissions::from_mode(0o555))?;

        remove_tree(&tree)?;
        assert!(fs::symlink_metadata(&tree).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_partial_tree_does_not_block_restoring_the_previous_install() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let temp = TempDir::new()?;
        let backup = temp.path().join("backup").join("demo");
        fs::create_dir_all(&backup)?;
        fs::write(backup.join("demo"), b"old binary")?;
        let to = temp.path().join("demo");
        fs::create_dir_all(to.join("assets"))?;
        fs::write(to.join("assets/new.js"), b"")?;
        fs::set_permissions(to.join("assets"), fs::Permissions::from_mode(0o555))?;

        restore_previous_install(&backup, &to)?;
        assert_eq!(fs::read(to.join("demo"))?, b"old binary");
        assert!(!to.join("assets").exists());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_replaced_install_with_read_only_directories_leaves_no_backup() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let temp = TempDir::new()?;
        let install_dir = temp.path().join("installed");
        let to = install_dir.join("demo");
        fs::create_dir_all(to.join("assets"))?;
        fs::write(to.join("assets/old.js"), b"")?;
        fs::set_permissions(to.join("assets"), fs::Permissions::from_mode(0o555))?;
        let from = temp.path().join("staging").join("demo");
        fs::create_dir_all(&from)?;
        fs::write(from.join("demo"), b"new binary")?;

        replace_plugin_dir(&from, &to, "demo")?;

        assert_eq!(fs::read(to.join("demo"))?, b"new binary");
        let leftovers: Vec<_> = fs::read_dir(&install_dir)?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<std::io::Result<_>>()?;
        assert_eq!(leftovers, vec![std::ffi::OsString::from("demo")]);
        Ok(())
    }

    fn entry_names(dir: &Path) -> std::io::Result<Vec<std::ffi::OsString>> {
        let mut names: Vec<_> = fs::read_dir(dir)?
            .map(|entry| entry.map(|entry| entry.file_name()))
            .collect::<std::io::Result<_>>()?;
        names.sort();
        Ok(names)
    }

    fn staged_demo(root: &Path, round: u32) -> Result<std::path::PathBuf> {
        let from = root.join(format!("staging-{round}")).join("demo");
        fs::create_dir_all(&from)?;
        fs::write(from.join("demo.exe"), format!("binary {round}"))?;
        Ok(from)
    }

    // Windows keeps a running executable's file from being deleted, so the
    // backup of an install whose plugin still runs (an update while the node
    // is up) cannot be removed by the replace that made it.
    #[cfg(windows)]
    #[test]
    fn a_backup_left_by_a_running_executable_is_removed_by_the_next_replace() -> Result<()> {
        let temp = TempDir::new()?;
        let install_dir = temp.path().join("installed");
        let to = install_dir.join("demo");
        fs::create_dir_all(&to)?;
        let system_root = std::env::var_os("SystemRoot").expect("SystemRoot is set on Windows");
        let exe = to.join("demo.exe");
        fs::copy(Path::new(&system_root).join(r"System32\PING.EXE"), &exe)?;
        let mut running = std::process::Command::new(&exe)
            .args(["-n", "60", "127.0.0.1"])
            .stdout(std::process::Stdio::null())
            .spawn()?;

        replace_plugin_dir(&staged_demo(temp.path(), 1)?, &to, "demo")?;
        let names = entry_names(&install_dir)?;
        assert_eq!(
            names.len(),
            2,
            "the running executable keeps its backup: {names:?}"
        );

        running.kill()?;
        running.wait()?;
        replace_plugin_dir(&staged_demo(temp.path(), 2)?, &to, "demo")?;

        assert_eq!(fs::read(to.join("demo.exe"))?, b"binary 2");
        assert_eq!(entry_names(&install_dir)?, ["demo"]);
        Ok(())
    }

    #[test]
    fn a_replace_leaves_other_plugins_backups_alone() -> Result<()> {
        let temp = TempDir::new()?;
        let install_dir = temp.path().join("installed");
        let to = install_dir.join("demo");
        fs::create_dir_all(&to)?;
        fs::write(to.join("demo.exe"), b"binary 0")?;
        // Another plugin's backup, as a replace of it running now would hold.
        let other = install_dir.join(".previous-other").join("other");
        fs::create_dir_all(&other)?;
        fs::write(other.join("other.exe"), b"")?;

        replace_plugin_dir(&staged_demo(temp.path(), 1)?, &to, "demo")?;

        assert!(other.join("other.exe").exists());
        assert_eq!(entry_names(&install_dir)?, [".previous-other", "demo"]);
        Ok(())
    }
}
