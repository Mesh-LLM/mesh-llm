use libloading::Library;
use std::error::Error;
use std::io;
use std::path::Path;

fn format_load_error(path: &Path, error: &libloading::Error) -> String {
    let os_error = error
        .source()
        .and_then(|source| source.downcast_ref::<io::Error>());
    match os_error {
        Some(os_error) => format!(
            "failed to load native runtime library {}: {} (OS error {}: {})",
            path.display(),
            error,
            os_error.raw_os_error().unwrap_or(0),
            os_error,
        ),
        None => format!(
            "failed to load native runtime library {}: {} (OS error 0)",
            path.display(),
            error,
        ),
    }
}

#[cfg(target_os = "windows")]
pub(crate) unsafe fn load(path: &Path) -> Result<Library, String> {
    use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library as WindowsLibrary};

    // LoadLibraryExW normally searches the application directory for a
    // dependent DLL, even when the requested DLL has an absolute path. Native
    // runtimes are installed elsewhere, so make the loaded DLL's directory the
    // first dependency search location.
    let result = unsafe { WindowsLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) };
    result
        .map(Into::into)
        .map_err(|error| format_load_error(path, &error))
}

#[cfg(not(target_os = "windows"))]
pub(crate) unsafe fn load(path: &Path) -> Result<Library, String> {
    unsafe { Library::new(path) }.map_err(|error| format_load_error(path, &error))
}

#[cfg(test)]
mod tests {
    use super::format_load_error;
    use libloading::Library;
    use std::path::Path;

    #[test]
    fn load_error_includes_library_path_and_os_error() {
        let path = Path::new("mesh-llm-missing-native-runtime.dll");
        let error = match unsafe { Library::new(path) } {
            Ok(_) => panic!("unexpectedly loaded missing test library"),
            Err(error) => error,
        };

        let message = format_load_error(path, &error);
        assert!(message.contains(path.to_string_lossy().as_ref()));
        assert!(message.contains("OS error"));
    }
}
