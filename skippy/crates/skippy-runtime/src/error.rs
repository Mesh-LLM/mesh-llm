use std::ffi::CStr;

use anyhow::Result;
use skippy_ffi::Error as RawError;

use crate::Status;

pub(crate) fn format_skippy_error(status: Status, message: &str) -> String {
    if message.is_empty() {
        format!("{:?}: native runtime returned no error detail", status)
    } else {
        format!("{:?}: {}", status, message)
    }
}

/// A failed native call, keeping its `Status` so callers can classify the
/// failure without parsing the rendered message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeError {
    pub status: Status,
    pub message: String,
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format_skippy_error(self.status, &self.message))
    }
}

impl std::error::Error for NativeError {}

impl NativeError {
    /// Finds the native failure anywhere in an error chain.
    pub fn find(error: &anyhow::Error) -> Option<&NativeError> {
        error
            .chain()
            .find_map(|cause| cause.downcast_ref::<NativeError>())
    }
}

pub(crate) fn ensure_ok(status: Status, error: *mut RawError) -> Result<()> {
    if status == Status::Ok {
        free_error(error);
        Ok(())
    } else {
        let message = error_message(error);
        free_error(error);
        Err(anyhow::Error::new(NativeError { status, message }))
    }
}

fn error_message(error: *mut RawError) -> String {
    if error.is_null() {
        return String::new();
    }

    let message = unsafe { (*error).message };
    if message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }
}

pub(crate) fn free_error(error: *mut RawError) {
    if !error.is_null() {
        unsafe {
            skippy_ffi::skippy_error_free(error);
        }
    }
}
