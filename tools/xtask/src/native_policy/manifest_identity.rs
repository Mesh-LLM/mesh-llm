use crate::command::DynResult;

pub(crate) fn verify(target: &str, os: &str, arch: &str) -> DynResult<()> {
    let pair = match target {
        "aarch64-apple-darwin" => ("macos", "aarch64"),
        "x86_64-apple-darwin" => ("macos", "x86_64"),
        "aarch64-unknown-linux-gnu" | "aarch64-linux-android" => ("linux", "aarch64"),
        "x86_64-unknown-linux-gnu" | "x86_64-linux-android" => ("linux", "x86_64"),
        "armv7-unknown-linux-gnueabihf" | "armv7-linux-androideabi" => ("linux", "arm"),
        "x86_64-pc-windows-msvc" => ("windows", "x86_64"),
        _ => return Err(format!("unsupported native target: {target}").into()),
    };
    if pair != (os, arch) {
        return Err(format!(
            "native OS/architecture does not match target: {os}/{arch} != {}/{}, target {target}",
            pair.0, pair.1
        )
        .into());
    }
    Ok(())
}
