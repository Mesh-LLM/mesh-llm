use super::super::{Error, ErrorKind, ReceiptContext};
use super::{FeedbackDraft, PublishedFeedback, contract_error, verify};
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

pub(super) fn publish(
    context: &ReceiptContext,
    draft: FeedbackDraft,
    destination: &Path,
) -> Result<PublishedFeedback, Error> {
    draft.payload.validate(context, draft.payload.state)?;
    let name = destination
        .file_name()
        .ok_or_else(|| contract_error("feedback destination must name a fresh directory"))?;
    super::evidence::safe_relative(Path::new(name))?;
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = parent.canonicalize()?;
    let parent_descriptor = super::evidence::open_directory(&parent)?;
    let destination = parent.join(name);
    if fs::symlink_metadata(&destination).is_ok() {
        return Err(contract_error("feedback destination already exists"));
    }
    let stage = tempfile::Builder::new()
        .prefix(".canary-feedback-")
        .tempdir_in(&parent)?;
    let stage_descriptor = super::evidence::open_directory(stage.path())?;
    for (family, admitted) in &draft.evidence {
        let directory = stage.path().join(family.as_str());
        fs::create_dir(&directory)?;
        admitted.snapshot.copy_into(&directory)?;
    }
    let bytes = serde_json::to_vec_pretty(&draft.payload)?;
    if u64::try_from(bytes.len()).map_or(true, |length| {
        length.saturating_add(1) > super::super::storage::RECEIPT_LIMIT
    }) {
        return Err(Error::new(
            ErrorKind::InputLimit,
            "feedback metadata exceeds limit",
        ));
    }
    let mut file = File::create_new(stage.path().join("feedback.json"))?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    // Verify every staged byte before making the final directory visible.
    super::evidence::same_directory(&stage_descriptor, stage.path())?;
    let _ = verify(context, stage.path(), draft.payload.state)?;
    super::evidence::same_directory(&parent_descriptor, &parent)?;
    publish_owned_directory(
        stage.path(),
        &destination,
        &parent_descriptor,
        &stage_descriptor,
        || {},
    )?;
    super::evidence::same_directory(&parent_descriptor, &parent)?;
    let _ = stage.keep();
    Ok(PublishedFeedback {
        directory: destination,
        state: draft.payload.state,
    })
}

/// Validate the original staged inode on both sides of the atomic rename.
/// The local seam permits deterministic tests of entry replacement after the
/// source guard; production supplies no action at that boundary.
pub(super) fn publish_owned_directory(
    source: &Path,
    destination: &Path,
    parent: &File,
    staged: &File,
    before_rename: impl FnOnce(),
) -> Result<(), Error> {
    super::evidence::same_directory(staged, source)?;
    before_rename();
    publish_directory(source, destination, parent)?;
    super::evidence::same_directory(staged, destination)
}

pub(super) fn publish_directory(
    source: &Path,
    destination: &Path,
    directory: &File,
) -> Result<(), Error> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        use std::{
            ffi::CString,
            os::{fd::AsRawFd, unix::ffi::OsStrExt},
        };
        let left = CString::new(
            source
                .file_name()
                .ok_or_else(|| contract_error("feedback staging name missing"))?
                .as_bytes(),
        )
        .map_err(|_| contract_error("invalid feedback staging name"))?;
        let right = CString::new(
            destination
                .file_name()
                .ok_or_else(|| contract_error("feedback destination name missing"))?
                .as_bytes(),
        )
        .map_err(|_| contract_error("invalid feedback destination name"))?;
        #[cfg(target_os = "linux")]
        // SAFETY: both names are terminated direct children of the owned parent
        // descriptor. NOREPLACE atomically refuses an existing destination.
        let status = unsafe {
            libc::renameat2(
                directory.as_raw_fd(),
                left.as_ptr(),
                directory.as_raw_fd(),
                right.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        #[cfg(target_os = "macos")]
        let status = {
            unsafe extern "C" {
                fn renameatx_np(
                    from_fd: libc::c_int,
                    from: *const libc::c_char,
                    to_fd: libc::c_int,
                    to: *const libc::c_char,
                    flags: libc::c_uint,
                ) -> libc::c_int;
            }
            // SAFETY: owned parent descriptor and terminated direct child names;
            // Darwin RENAME_EXCL (0x4) gives atomic no-clobber publication.
            unsafe {
                renameatx_np(
                    directory.as_raw_fd(),
                    left.as_ptr(),
                    directory.as_raw_fd(),
                    right.as_ptr(),
                    0x00000004,
                )
            }
        };
        if status != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (source, destination, directory);
        Err(contract_error(
            "atomic feedback publication requires macOS or Linux",
        ))
    }
}
