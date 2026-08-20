use std::future::Future;

use crate::SDK_RUNTIME;

pub(super) fn block_on<F>(future: F) -> F::Output
where
    F: Future,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
        Err(_) => SDK_RUNTIME.block_on(future),
    }
}
