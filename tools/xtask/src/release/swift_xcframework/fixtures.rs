use super::{Error, input::Document};
use plist::{Dictionary, Value};
#[cfg(unix)]
use std::{collections::BTreeSet, path::Path};
use std::{fs, path::PathBuf};

pub(super) struct Fixture {
    pub(super) directory: tempfile::TempDir,
    pub(super) root: PathBuf,
    pub(super) entries: Vec<Value>,
}

pub(super) fn entry(platform: &str, variant: &str, architectures: &[&str]) -> Value {
    let mut fields = Dictionary::new();
    fields.insert(
        "LibraryIdentifier".into(),
        Value::String(format!("{platform}-{variant}")),
    );
    fields.insert(
        "LibraryPath".into(),
        Value::String("MeshLLMFFI.framework".into()),
    );
    fields.insert("SupportedPlatform".into(), Value::String(platform.into()));
    if !variant.is_empty() {
        fields.insert(
            "SupportedPlatformVariant".into(),
            Value::String(variant.into()),
        );
    }
    fields.insert(
        "SupportedArchitectures".into(),
        Value::Array(
            architectures
                .iter()
                .map(|name| Value::String((*name).into()))
                .collect(),
        ),
    );
    Value::Dictionary(fields)
}

pub(super) fn set(entry: &mut Value, key: &str, value: Value) {
    entry.as_dictionary_mut().unwrap().insert(key.into(), value);
}

impl Fixture {
    pub(super) fn new(entries: Vec<Value>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("MeshLLMFFI.xcframework");
        fs::create_dir(&root).unwrap();
        Self {
            directory,
            root,
            entries,
        }
    }

    pub(super) fn host() -> Self {
        Self::new(vec![entry("macos", "", &["arm64"])])
    }

    pub(super) fn full() -> Self {
        Self::new(vec![
            entry("ios", "", &["arm64"]),
            entry("ios", "simulator", &["arm64"]),
            entry("ios", "maccatalyst", &["arm64"]),
            entry("macos", "", &["arm64"]),
        ])
    }

    pub(super) fn write(&self, binary: bool) {
        let mut info = Dictionary::new();
        info.insert(
            "AvailableLibraries".into(),
            Value::Array(self.entries.clone()),
        );
        let value = Value::Dictionary(info);
        let file = fs::File::create(self.root.join("Info.plist")).unwrap();
        if binary {
            value.to_writer_binary(file).unwrap();
        } else {
            value.to_writer_xml(file).unwrap();
        }
    }

    pub(super) fn declarations(&self, mode: Option<super::Mode>) -> Result<usize, Error> {
        let document = Document::read(&self.root)?;
        let entries = document.entries(mode)?;
        for entry in &entries {
            entry.architectures(mode)?;
            entry.location()?;
        }
        Ok(entries.len())
    }

    #[cfg(unix)]
    pub(super) fn framework(&self, index: usize) -> PathBuf {
        let fields = self.entries[index].as_dictionary().unwrap();
        self.root
            .join(fields["LibraryIdentifier"].as_string().unwrap())
            .join(fields["LibraryPath"].as_string().unwrap())
    }

    #[cfg(unix)]
    pub(super) fn materialize(&self) {
        for (index, entry) in self.entries.iter().enumerate() {
            let fields = entry.as_dictionary().unwrap();
            let architectures = fields["SupportedArchitectures"]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| name.as_string().unwrap())
                .collect::<Vec<_>>()
                .join(" ");
            let framework = self.framework(index);
            fs::create_dir_all(&framework).unwrap();
            let name = framework.file_stem().unwrap();
            if fields["SupportedPlatform"].as_string() == Some("macos")
                && fields.get("SupportedPlatformVariant").is_none()
            {
                let version = framework.join("Versions/A");
                for path in ["Headers", "Modules", "Resources"] {
                    fs::create_dir_all(version.join(path)).unwrap();
                }
                fs::write(version.join(name), architectures).unwrap();
                for path in [
                    "Modules/module.modulemap",
                    "Resources/Info.plist",
                    "Resources/PrivacyInfo.xcprivacy",
                ] {
                    fs::write(version.join(path), b"fixture").unwrap();
                }
                std::os::unix::fs::symlink("A", framework.join("Versions/Current")).unwrap();
                std::os::unix::fs::symlink(
                    Path::new("Versions/Current").join(name),
                    framework.join(name),
                )
                .unwrap();
                for path in ["Headers", "Modules", "Resources"] {
                    std::os::unix::fs::symlink(
                        format!("Versions/Current/{path}"),
                        framework.join(path),
                    )
                    .unwrap();
                }
            } else {
                fs::write(framework.join(name), architectures).unwrap();
            }
        }
    }

    #[cfg(unix)]
    pub(super) fn verify(&self, mode: Option<super::Mode>) -> Result<usize, Error> {
        let document = Document::read(&self.root)?;
        let entries = document.entries(mode)?;
        super::verify(
            super::Verification {
                entries: &entries,
                root: &self.root,
                mode,
            },
            |binary| {
                Ok(fs::read_to_string(binary)
                    .unwrap()
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect::<BTreeSet<_>>())
            },
        )
    }
}
