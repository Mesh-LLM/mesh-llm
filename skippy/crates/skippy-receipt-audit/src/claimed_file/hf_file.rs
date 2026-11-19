//! The name of a claimed GGUF on Hugging Face,
//! `owner/name@revision/path/to/file.gguf`.
//!
//! The node that issued a receipt writes this name, and the client turns it
//! into a URL and into a path under its cache. So the name is checked as
//! hostile text: each part holds only characters that mean the same in a
//! URL, in a path on any platform and in a file name, and no part is one a
//! path or a URL would resolve away. What is not plainly a repo, a revision
//! and a GGUF file is refused.

use std::fmt;

use anyhow::{Context, Result, bail};

/// The longest name, in bytes.
const MAX_NAME_BYTES: usize = 1024;
/// The longest owner or repo name, as Hugging Face has it.
const MAX_REPO_PART_BYTES: usize = 96;
const MAX_REVISION_BYTES: usize = 128;
/// The longest component of the file's path: the limit of common file
/// systems.
const MAX_FILE_PART_BYTES: usize = 255;
/// The most components of the file's path.
const MAX_FILE_PARTS: usize = 16;
/// The file is a GGUF. The copy's record and lock sit next to it under
/// other suffixes, which a claimed file therefore never has.
const FILE_SUFFIX: &str = ".gguf";
/// Names Windows gives to devices, with any extension.
const DEVICE_NAMES: [&str; 4] = ["con", "prn", "aux", "nul"];

/// A file of a Hugging Face model repo at a revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfFile {
    pub repo: String,
    pub revision: String,
    pub file: String,
}

impl HfFile {
    /// Parses `owner/name@revision/path/to/file.gguf`, or `None` when the
    /// text is not such a name ([`Self::parse_checked`] says why).
    pub fn parse(text: &str) -> Option<Self> {
        Self::parse_checked(text).ok()
    }

    /// Parses `owner/name@revision/path/to/file.gguf` strictly.
    pub fn parse_checked(text: &str) -> Result<Self> {
        if text.len() > MAX_NAME_BYTES {
            bail!("a model name of {} bytes is too long", text.len());
        }
        let (repo, rest) = text
            .split_once('@')
            .context("it is not repo@revision/file")?;
        let (revision, file) = rest
            .split_once('/')
            .context("it names no file after the revision")?;
        let parsed = Self {
            repo: repo.to_owned(),
            revision: revision.to_owned(),
            file: file.to_owned(),
        };
        parsed.check()?;
        Ok(parsed)
    }

    /// Checks every part, as [`Self::parse_checked`] does: the fields are
    /// public, so a name built from them is checked again before it is a
    /// URL or a path.
    pub fn check(&self) -> Result<()> {
        let (owner, name) = self
            .repo
            .split_once('/')
            .context("the repo is not owner/name")?;
        for (part, what) in [(owner, "the repo's owner"), (name, "the repo's name")] {
            check_word(part, what, MAX_REPO_PART_BYTES, "")?;
            // the cache joins owner and name with `--`
            if part.contains("--") {
                bail!("{what} {part:?} holds `--`");
            }
        }
        check_word(&self.revision, "the revision", MAX_REVISION_BYTES, "")?;
        let parts: Vec<&str> = self.file.split('/').collect();
        if parts.len() > MAX_FILE_PARTS {
            bail!("the file's path has more than {MAX_FILE_PARTS} parts");
        }
        for part in &parts {
            check_word(
                part,
                "a part of the file's path",
                MAX_FILE_PART_BYTES,
                "+=,",
            )?;
            let stem = part.split('.').next().unwrap_or(part).to_ascii_lowercase();
            if is_device_name(&stem) {
                bail!("a part of the file's path, {part:?}, names a device");
            }
        }
        match parts.last() {
            Some(last) if last.len() > FILE_SUFFIX.len() && last.ends_with(FILE_SUFFIX) => Ok(()),
            _ => bail!("the file {:?} is not a {FILE_SUFFIX} file", self.file),
        }
    }

    /// Whether the revision is a commit, which names the same bytes
    /// forever: 40 lowercase hex digits. A branch or a tag can move.
    pub fn is_pinned(&self) -> bool {
        is_commit(&self.revision)
    }

    /// The same file at `revision`.
    pub fn at(&self, revision: &str) -> Self {
        Self {
            repo: self.repo.clone(),
            revision: revision.to_owned(),
            file: self.file.clone(),
        }
    }

    /// The same repo and revision's `file`.
    pub fn sibling(&self, file: &str) -> Self {
        Self {
            repo: self.repo.clone(),
            revision: self.revision.clone(),
            file: file.to_owned(),
        }
    }
}

impl fmt::Display for HfFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}/{}", self.repo, self.revision, self.file)
    }
}

/// Whether `text` is a commit id: 40 lowercase hex digits.
pub fn is_commit(text: &str) -> bool {
    text.len() == 40
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Checks one part of a name: ASCII letters and digits, `-`, `_`, `.` and
/// the characters of `also`; not empty, at most `max` bytes, no `..`, no
/// leading `.` or `-`, no trailing `.`. So it holds no separator of a path
/// or a URL (`/`, `\`, `:`, `?`, `#`, `%`, `@`), no control character and
/// no space, and is no `.` or `..`.
fn check_word(part: &str, what: &str, max: usize, also: &str) -> Result<()> {
    if part.is_empty() {
        bail!("{what} is empty");
    }
    if part.len() > max {
        bail!("{what} is longer than {max} bytes");
    }
    let allowed = |c: char| c.is_ascii_alphanumeric() || "-_.".contains(c) || also.contains(c);
    if let Some(c) = part.chars().find(|&c| !allowed(c)) {
        bail!("{what} {part:?} holds {c:?}");
    }
    if part.contains("..") || part.starts_with(['.', '-']) || part.ends_with('.') {
        bail!("{what} {part:?} starts with `.` or `-`, ends with `.` or holds `..`");
    }
    Ok(())
}

/// Whether `stem`, in lowercase, is a device on Windows: `con`, `nul`,
/// `com1`, `lpt9` and their like.
fn is_device_name(stem: &str) -> bool {
    let numbered = |prefix: &str| {
        stem.strip_prefix(prefix)
            .is_some_and(|n| n.len() == 1 && n.bytes().all(|b| b.is_ascii_digit()))
    };
    DEVICE_NAMES.contains(&stem) || numbered("com") || numbered("lpt")
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn references_parse_as_repo_revision_and_file() {
        assert_eq!(
            HfFile::parse("unsloth/Qwen3-4B-GGUF@main/Q8_0/Qwen3-4B-Q8_0.gguf"),
            Some(HfFile {
                repo: "unsloth/Qwen3-4B-GGUF".into(),
                revision: "main".into(),
                file: "Q8_0/Qwen3-4B-Q8_0.gguf".into(),
            })
        );
        for name in [
            "TheBloke/Mistral-7B-Instruct-v0.2-GGUF@main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "o/m@v1.0/a/b/c/M-00001-of-00003.gguf",
            "o_1/m.x@0123456789abcdef0123456789abcdef01234567/m+lora=1,2.gguf",
        ] {
            let parsed = HfFile::parse_checked(name).unwrap();
            assert_eq!(parsed.to_string(), name);
        }
        assert_eq!(HfFile::parse("Qwen3-4B-Q8_0.gguf"), None);
        assert_eq!(HfFile::parse("owner/model@main"), None);
        assert_eq!(HfFile::parse("owner/model@main/../x.gguf"), None);
        assert_eq!(HfFile::parse("model@main/x.gguf"), None);
    }

    #[test]
    fn only_a_commit_is_pinned() {
        let pinned = |revision: &str| {
            HfFile::parse(&format!("o/m@{revision}/x.gguf"))
                .unwrap()
                .is_pinned()
        };
        assert!(pinned(COMMIT));
        assert!(!pinned("main"));
        assert!(!pinned(&COMMIT[..39]));
        assert!(!pinned(&format!("{COMMIT}0")));
        assert!(!pinned(&COMMIT.to_uppercase()));
        assert!(!pinned(&COMMIT.replace('0', "g")));
        let at = HfFile::parse("o/m@main/x.gguf").unwrap().at(COMMIT);
        assert!(at.is_pinned());
        assert_eq!(at.to_string(), format!("o/m@{COMMIT}/x.gguf"));
    }

    /// Names a node may write to reach outside the repo, the cache or the
    /// resolver. None is a file of a repo.
    #[test]
    fn names_that_are_not_a_file_of_a_repo_are_refused() {
        let long = "a".repeat(300);
        let cases = [
            // absolute paths, with the doubled slash Hugging Face serves
            "evil/m@main//home/u/.ssh/authorized_keys",
            "evil/m@main//etc/cron.d/x.gguf",
            "evil/m@main//x.gguf",
            "evil/m@main/a//x.gguf",
            "evil/m@main/a/x.gguf/",
            "evil/m@main/",
            // `..` and `.` in every part
            "../m@main/x.gguf",
            "evil/..@main/x.gguf",
            "evil/m@../x.gguf",
            "evil/m@main/../x.gguf",
            "evil/m@main/a/../../x.gguf",
            "evil/m@main/a/..x.gguf",
            "evil/m@main/a..b/x.gguf",
            "evil/m@main/./x.gguf",
            "./m@main/x.gguf",
            "evil/.@main/x.gguf",
            "evil/m@./x.gguf",
            "evil/.m@main/x.gguf",
            "evil/m@main/.hidden/x.gguf",
            "evil/m@main/x.gguf.",
            // backslashes and drive letters
            "evil/m@main/..\\..\\x.gguf",
            "evil/m@main/a\\x.gguf",
            "evil\\m/m@main/x.gguf",
            "evil/m@main/C:/x.gguf",
            "evil/m@main/C:x.gguf",
            "evil/m@C:/x.gguf",
            "C:/m@main/x.gguf",
            "evil/m@main/x.gguf:stream",
            // what a URL reads as something else
            "evil/m@main/x.gguf?download=1",
            "evil/m@main/x.gguf#frag",
            "evil/m@main/%2e%2e/x.gguf",
            "evil/m@main/a%2fb.gguf",
            "evil/m@main?x/x.gguf",
            "evil/m@main#x/x.gguf",
            "evil/m@ma%69n/x.gguf",
            "evil/m@main@other/x.gguf",
            "evil/m?x@main/x.gguf",
            "evil%2f/m@main/x.gguf",
            "http://evil/m@main/x.gguf",
            "evil/m@main/x y.gguf",
            "evil/m@main/x.gguf ",
            " evil/m@main/x.gguf",
            // control characters
            "evil/m@main/x\n.gguf",
            "evil/m@main/x\0.gguf",
            "evil/m@main/x\r\nHost: other.gguf",
            "evil/m@ma\tin/x.gguf",
            "evil/m\u{7f}@main/x.gguf",
            // not ASCII
            "evil/m@main/x\u{202e}.gguf",
            "evil/ｍ@main/x.gguf",
            // empty parts and parts in the wrong number
            "/m@main/x.gguf",
            "evil/@main/x.gguf",
            "evil/m@/x.gguf",
            "evil/m/extra@main/x.gguf",
            "@main/x.gguf",
            "evil/m@main",
            "",
            // names the cache would confuse with another repo or an option
            "a--b/c@main/x.gguf",
            "a/b--c@main/x.gguf",
            "-rf/m@main/x.gguf",
            "evil/m@-main/x.gguf",
            "evil/m@main/-x.gguf",
            // not a GGUF: the copy's own files, and files of no model
            "evil/m@main/x.gguf.ranges",
            "evil/m@main/x.gguf.lock",
            "evil/m@main/x.bin",
            "evil/m@main/.gguf",
            "evil/m@main/x.GGUF",
            "evil/m@main/gguf",
            // devices on Windows
            "evil/m@main/NUL.gguf",
            "evil/m@main/con/x.gguf",
            "evil/m@main/COM1.x.gguf",
            "evil/m@main/a/lpt9.gguf",
        ];
        for name in cases {
            let refused = HfFile::parse_checked(name);
            assert!(refused.is_err(), "{name:?} parsed as {refused:?}");
            assert_eq!(HfFile::parse(name), None);
        }
        // overlong names and parts
        for name in [
            format!("{long}/m@main/x.gguf"),
            format!("evil/{long}@main/x.gguf"),
            format!("evil/m@{long}/x.gguf"),
            format!("evil/m@main/{long}.gguf"),
            format!("evil/m@main/{}x.gguf", "a/".repeat(MAX_FILE_PARTS)),
            format!("evil/m@main/{}x.gguf", "abcdefgh/".repeat(200)),
        ] {
            assert!(HfFile::parse_checked(&name).is_err(), "{name:?}");
        }
        assert!(HfFile::parse_checked(&format!("evil/m@main/{}x.gguf", "a/".repeat(15))).is_ok());
    }

    #[test]
    fn a_name_built_from_its_fields_is_checked_too() {
        let good = HfFile::parse("o/m@main/x.gguf").unwrap();
        assert!(good.check().is_ok());
        assert!(good.sibling("/etc/passwd").check().is_err());
        assert!(good.sibling("../x.gguf").check().is_err());
        assert!(good.at("../../main").check().is_err());
        assert!(good.at("").check().is_err());
        let mut repo = good.clone();
        repo.repo = "o/m/../../x".into();
        assert!(repo.check().is_err());
    }
}
