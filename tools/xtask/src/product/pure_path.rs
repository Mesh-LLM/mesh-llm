use crate::repository::text::repr;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct PurePath {
    root: String,
    parts: Vec<String>,
    /// Whether this path parsed with the Windows flavour (`PureWindowsPath`),
    /// stored separately from the root: drive-less Windows paths keep their
    /// flavour for comparisons and display.
    windows: bool,
}

impl PurePath {
    /// Parses with the host's path flavour: Windows paths (`C:\a\b`)
    /// behave like Python's `PureWindowsPath`, everything else like
    /// `PurePosixPath`. The composer receives native paths from
    /// `package-release.ps1`, so a POSIX-only parse broke Windows releases.
    pub(super) fn new(text: &str) -> Self {
        Self::parse(text, cfg!(windows))
    }

    fn parse(text: &str, windows: bool) -> Self {
        let mut path = Self {
            root: String::new(),
            parts: Vec::new(),
            windows,
        };
        if !windows {
            path.root = if text.starts_with('/') {
                "/".to_owned()
            } else {
                String::new()
            };
            path.parts = path_parts(text);
            return path;
        }
        let normalized = text.replace('\\', "/");
        let mut rest = normalized.as_str();
        let bytes = rest.as_bytes();
        if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
            path.root.push(bytes[0].to_ascii_uppercase() as char);
            path.root.push(':');
            rest = &rest[2..];
        }
        if rest.starts_with('/') {
            path.root.push('/');
        }
        path.parts = path_parts(rest);
        path
    }

    fn windows_flavour(&self) -> bool {
        self.windows
    }

    /// `str(path)`: the root and the parts, or `.` for an empty relative path.
    pub(super) fn display(&self) -> String {
        if self.root.is_empty() && self.parts.is_empty() {
            return ".".to_owned();
        }
        if self.windows_flavour() {
            format!("{}{}", self.root, self.parts.join("\\")).replace('/', "\\")
        } else {
            format!("{}{}", self.root, self.parts.join("/"))
        }
    }

    /// `self / name` for a single plain component.
    pub(super) fn join(&self, name: &str) -> Self {
        let mut parts = self.parts.clone();
        parts.push(name.to_owned());
        Self {
            root: self.root.clone(),
            parts,
            windows: self.windows,
        }
    }

    /// `self.relative_to(other).as_posix()`: `other` must equal `self` or
    /// be one of its parents; otherwise the legacy `ValueError` text.
    pub(super) fn relative_to(&self, other: &Self) -> Result<String, String> {
        // Windows paths compare case-insensitively, like `PureWindowsPath`.
        let fold = self.windows_flavour() || other.windows_flavour();
        let same = |a: &str, b: &str| {
            if fold {
                a.eq_ignore_ascii_case(b)
            } else {
                a == b
            }
        };
        let related = same(&self.root, &other.root)
            && other.parts.len() <= self.parts.len()
            && self.parts[..other.parts.len()]
                .iter()
                .zip(&other.parts)
                .all(|(a, b)| same(a, b));
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

/// `pathlib` parts: the path body without empty and `.` components. Both
/// flavours split bodies on `/`; the Windows parser splits the root off first.
fn path_parts(body: &str) -> Vec<String> {
    body.split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(str::to_owned)
        .collect()
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
        assert_eq!(PurePath::new("//a").display(), "/a");
        assert_eq!(PurePath::new("///a").display(), "/a");
        assert_eq!(PurePath::new(".").join("m").display(), "m");
        assert_eq!(relative("./a", "a").as_deref(), Ok("."));
        assert_eq!(relative("a/b", ".").as_deref(), Ok("a/b"));
        assert_eq!(relative("a/b/../c", "a/b/..").as_deref(), Ok("c"));
        assert_eq!(relative("//a/b", "/a"), Ok("b".to_owned()));
        assert_eq!(
            relative("/a", "b"),
            Err("\"/a\" is not in the subpath of \"b\"".to_owned())
        );
    }

    fn win(path: &str, other: &str) -> Result<String, String> {
        PurePath::parse(path, true).relative_to(&PurePath::parse(other, true))
    }

    /// Regression: release run 40466744770 failed `Compose Windows CPU
    /// product` because backslash paths were parsed as one POSIX component.
    #[test]
    fn windows_native_paths_relate_like_pure_windows_path() {
        let bundle = r"C:\Users\runneradmin\AppData\Local\Temp\rel\mesh-bundle";
        assert_eq!(
            win(&format!(r"{bundle}\mesh-llm.exe"), bundle).as_deref(),
            Ok("mesh-llm.exe")
        );
        assert_eq!(
            win(&format!(r"{bundle}\native-runtimes\cpu"), bundle).as_deref(),
            Ok("native-runtimes/cpu")
        );
        assert_eq!(win(r"c:\A\b", r"C:/a").as_deref(), Ok("b"));
        assert!(win(r"D:\a\b", r"C:\a").is_err());
        assert_eq!(PurePath::parse(r"C:\a/b", true).display(), r"C:\a\b");
        assert_eq!(
            PurePath::parse(r"C:\a", true).join("m.json").display(),
            r"C:\a\m.json"
        );
    }

    /// Regression: the Windows flavour used to be inferred from a `:` in
    /// the root, so drive-less Windows paths lost Windows comparison and
    /// display.
    #[test]
    fn windows_flavour_survives_drive_less_paths() {
        assert_eq!(
            win(r"Bundle\mesh-llm.exe", "bundle").as_deref(),
            Ok("mesh-llm.exe")
        );
        assert_eq!(win(r"\foo\m.exe", r"\FOO").as_deref(), Ok("m.exe"));
        assert_eq!(PurePath::parse(r"\foo", true).display(), r"\foo");
        assert_eq!(PurePath::parse(r"Bundle\x", true).display(), r"Bundle\x");
    }
}
