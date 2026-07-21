use libloading::Library;
use std::path::Path;

#[cfg(target_os = "windows")]
pub(crate) unsafe fn load(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::windows::{LOAD_WITH_ALTERED_SEARCH_PATH, Library as WindowsLibrary};

    // LoadLibraryExW normally searches the application directory for a
    // dependent DLL, even when the requested DLL has an absolute path. Native
    // runtimes are installed elsewhere, so make the loaded DLL's directory the
    // first dependency search location.
    unsafe { WindowsLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }.map(Into::into)
}

#[cfg(not(target_os = "windows"))]
pub(crate) unsafe fn load(path: &Path) -> Result<Library, libloading::Error> {
    unsafe { Library::new(path) }
}
