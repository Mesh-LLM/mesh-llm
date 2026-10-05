use std::ffi::CStr;

use anyhow::{Result, anyhow};
use skippy_ffi::Error as RawError;

use crate::Status;

pub(crate) fn format_skippy_error(status: Status, message: &str) -> String {
    if message.is_empty() {
        format!("{:?}", status)
    } else {
        format!("{:?}: {}", status, message)
    }
}

pub(crate) fn ensure_ok(status: Status, error: *mut RawError) -> Result<()> {
    if status == Status::Ok {
        free_error(error);
        Ok(())
    } else {
        let message = error_message(error);
        free_error(error);
        Err(anyhow!("{}", format_skippy_error(status, &message)))
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
