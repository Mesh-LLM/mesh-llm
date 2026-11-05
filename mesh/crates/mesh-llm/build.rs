fn main() {
    // Windows gives a program's main thread 1 MB of stack, macOS and Linux
    // (by default) 8 MB. `--print-build-contract` runs on the main thread,
    // before the 8 MB application thread exists, and parsing the whole CLI
    // there overflows 1 MB in a debug build. Reserve 8 MB on Windows too.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let stack = 8 * 1024 * 1024;
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/STACK:{stack}");
    } else {
        println!("cargo:rustc-link-arg-bins=-Wl,--stack,{stack}");
    }
}
