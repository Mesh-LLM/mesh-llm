//! Compiles in this build's pin for the capsules plugin archive it ships
//! with. The release job builds that archive from the same commit before the
//! host and passes its SHA-256 here; a build without it pins no capsules
//! archive and so installs none. The pin is written to a generated source
//! file, so a compiler cache keys on its value.

use std::{env, fs, path::Path};

/// The lowercase hex SHA-256 of `capsules-<version>-<target>.tar.gz`.
const SHA256_ENV: &str = "MESH_LLM_BUNDLED_CAPSULES_SHA256";

fn main() {
    println!("cargo:rerun-if-env-changed={SHA256_ENV}");
    let target = env::var("TARGET").expect("cargo sets TARGET for build scripts");
    let pins = match env::var(SHA256_ENV) {
        Ok(sha256) if !sha256.is_empty() => {
            assert!(
                sha256.len() == 64
                    && sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "{SHA256_ENV} must be 64 lowercase hex characters"
            );
            format!("&[({target:?}, {sha256:?})]")
        }
        _ => "&[]".to_owned(),
    };
    let out = Path::new(&env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("bundled_capsules_pins.rs");
    fs::write(
        out,
        format!(
            "/// `(target triple, SHA-256)` of the capsules archive bundled with this build.\n\
             const BUNDLED_CAPSULES_PINS: &[(&str, &str)] = {pins};\n"
        ),
    )
    .expect("write the bundled capsules pin");
}
