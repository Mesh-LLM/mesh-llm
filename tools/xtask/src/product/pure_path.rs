use crate::repository::text::repr;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct PurePath {
    root: String,
    parts: Vec<String>,
}

impl PurePath {
    /// Parses with the host's path flavour: Windows paths (`C:\\a\\b`)
    /// behave like Python's `PureWindowsPath`, everything else like
    /// `PurePosixPath`. The composer receives native paths from
    /// `package-release.ps1`, so a POSIX-only parse broke Windows releases.
    pub(super) fn new(text: &str) -> Self {
        Self::parse(text, cfg!(windows))
    }

    fn parse(text: &str, windows: bool) -> Self {
        if !windows {
            let root = if text.starts_with('/') { "/" } else { "" };
            let parts = text
                .split('/')
                .filter(|part| !part.is_empty() && *part != ".")
                .map(str::to_owned)
                .collect();
            return Self {
                root: root.to_owned(),
                parts,
            };
        }
        let normalized = text.replace('\\', "/");
        let mut rest = normalized.as_str();
        let mut root = String::new();
        let bytes = rest.as_bytes();
        if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
            root.push(bytes[0].to_ascii_uppercase() as char);
            root.push(':');
            rest = &rest[2..];
        }
        if rest.starts_with('/') {
            root.push('/');
        }
        let parts = rest
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .map(str::to_owned)
            .collect();
        Self { root, parts }
    }

    fn windows_flavour(&self) -> bool {
        self.root.contains(':')
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
}
