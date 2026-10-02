use super::support::valid;

pub struct ExportCase {
    pub id: String,
    pub input: Option<Vec<u8>>,
    pub args: Vec<String>,
    pub initial: Vec<(String, Vec<u8>)>,
    pub directories: Vec<String>,
}

/// The ordinary export consumed by nightly: JSON, environment and shell fields.
pub fn complete_export() -> ExportCase {
    ExportCase {
        id: "complete-export".into(),
        input: Some(valid().into_bytes()),
        args: [
            "--matrix",
            "matrix.json",
            "--json-output",
            "params.json",
            "--github-env",
            "github.env",
            "--print-shell",
        ]
        .map(str::to_owned)
        .to_vec(),
        initial: Vec::new(),
        directories: Vec::new(),
    }
}
