use std::path::Path;

pub(crate) fn os_error(path: &Path, error: &std::io::Error) -> String {
    format!("{}: {error}", path.display())
}

pub(crate) fn read_text(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|error| os_error(path, &error))
}

pub(crate) fn decode_utf8(bytes: Vec<u8>) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|error| error.to_string())
}
