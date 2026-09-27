//! `pathlib.PurePosixPath` as the legacy composer uses it: parsing (a `//`
//! root is distinct from `/`, empty and `.` parts vanish, `..` stays
//! literal), `str()`, the `/` join and Python 3.13's `relative_to` without
//! `walk_up`, including its `ValueError` wording.

use crate::repository::python_text::repr;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct PurePath {
    root: &'static str,
    parts: Vec<String>,
}

impl PurePath {
    pub(super) fn new(text: &str) -> Self {
        let root = if text.starts_with("//") && !text.starts_with("///") {
            "//"
        } else if text.starts_with('/') {
            "/"
        } else {
            ""
        };
        let parts = text
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .map(str::to_owned)
            .collect();
        Self { root, parts }
    }

    /// `str(path)`: the root and the parts, or `.` for an empty relative path.
    pub(super) fn display(&self) -> String {
        if self.root.is_empty() && self.parts.is_empty() {
            return ".".to_owned();
        }
        format!("{}{}", self.root, self.parts.join("/"))
    }

    /// `self / name` for a single plain component.
    pub(super) fn join(&self, name: &str) -> Self {
        let mut parts = self.parts.clone();
        parts.push(name.to_owned());
        Self {
            root: self.root,
            parts,
        }
    }

    /// `self.relative_to(other).as_posix()`: `other` must equal `self` or
    /// be one of its parents; otherwise the legacy `ValueError` text.
    pub(super) fn relative_to(&self, other: &Self) -> Result<String, String> {
        let related = self.root == other.root
            && other.parts.len() <= self.parts.len()
            && self.parts[..other.parts.len()] == other.parts[..];
        if !related {
            return Err(format!(
                "{} is not in the subpath of {}",
                repr(&self.display()),
                repr(&other.display())
            ));
        }
        let rest = &self.parts[other.parts.len()..];
        Ok(if rest.is_empty() {
            ".".to_owned()
        } else {
            rest.join("/")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::PurePath;

    fn relative(path: &str, other: &str) -> Result<String, String> {
        PurePath::new(path).relative_to(&PurePath::new(other))
    }

    #[test]
    fn migration_product_pure_path_matches_python() {
        assert_eq!(PurePath::new("a//./b/").display(), "a/b");
        assert_eq!(PurePath::new("").display(), ".");
        assert_eq!(PurePath::new("//a").display(), "//a");
        assert_eq!(PurePath::new("///a").display(), "/a");
        assert_eq!(PurePath::new(".").join("m").display(), "m");
        assert_eq!(relative("./a", "a").as_deref(), Ok("."));
        assert_eq!(relative("a/b", ".").as_deref(), Ok("a/b"));
        assert_eq!(relative("a/b/../c", "a/b/..").as_deref(), Ok("c"));
        assert_eq!(
            relative("//a/b", "/a"),
            Err("'//a/b' is not in the subpath of '/a'".to_owned())
        );
        assert_eq!(
            relative("/a", "b"),
            Err("'/a' is not in the subpath of 'b'".to_owned())
        );
    }
}
