//! What the fixture children inherit after `env_clear()`.
//!
//! Hosts without `/bin/bash` or `/usr/bin` (NixOS, Nix build sandboxes, some
//! containers) keep bash and the coreutils the scripts call elsewhere on
//! `PATH`, and the CA bundle may only be reachable through `SSL_CERT_FILE`, so
//! the children get the test process's `PATH` and CA variables instead of
//! fixed system paths. Everything else stays cleared.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

/// The test process's `PATH`, or the usual system directories without one.
pub fn path() -> OsString {
    std::env::var_os("PATH").unwrap_or_else(|| OsString::from("/usr/bin:/bin"))
}

/// `dir` in front of [`path`].
pub fn path_with(dir: &Path) -> OsString {
    let mut dirs = vec![dir.to_path_buf()];
    dirs.extend(std::env::split_paths(&path()));
    std::env::join_paths(dirs).expect("PATH entries are joinable")
}

/// Bash 4 or newer: Homebrew's on macOS (the system one is 3.2), otherwise the
/// first `bash` on [`path`].
pub fn bash() -> PathBuf {
    let homebrew = Path::new("/opt/homebrew/bin/bash");
    if homebrew.exists() {
        return homebrew.to_path_buf();
    }
    std::env::split_paths(&path())
        .map(|dir| dir.join("bash"))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| PathBuf::from("/bin/bash"))
}

/// `script` with its `#!/bin/bash` line pointing at [`bash`].
pub fn bash_script(script: impl AsRef<[u8]>) -> Vec<u8> {
    let script = script.as_ref();
    let body = script.strip_prefix(b"#!/bin/bash").unwrap_or(script);
    let mut out = format!("#!{}", bash().display()).into_bytes();
    out.extend_from_slice(body);
    out
}

/// The CA variables that are set, for children that build HTTPS clients.
pub fn ca_certificates() -> Vec<(&'static str, OsString)> {
    ["SSL_CERT_FILE", "SSL_CERT_DIR", "NIX_SSL_CERT_FILE"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name, value)))
        .collect()
}
