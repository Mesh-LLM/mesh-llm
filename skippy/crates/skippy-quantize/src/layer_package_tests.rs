use crate::{Args, Command, quantize_layer_package};
use clap::Parser;
use std::path::PathBuf;

/// Removes its directory when dropped, also when an assertion fails.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_builder_that_cannot_start_leaves_the_existing_package_in_place() {
    let dir = Scratch(std::env::temp_dir().join(format!(
        "skippy-quantize-replace-package-{}",
        std::process::id()
    )));
    let root = &dir.0;
    let _ = std::fs::remove_dir_all(root);
    let package = root.join("package");
    std::fs::create_dir_all(&package).unwrap();
    let marker = package.join("marker");
    std::fs::write(&marker, b"existing package").unwrap();
    // An empty file is a file, but no platform can start it.
    let builder = root.join(format!(
        "skippy-package-builder{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&builder, b"").unwrap();

    let argv = [
        "skippy-quantize".as_ref(),
        "quantize-layer-package".as_ref(),
        "--source".as_ref(),
        root.join("source").as_os_str(),
        "--source-prefix".as_ref(),
        "model".as_ref(),
        "--target".as_ref(),
        root.join("target").as_os_str(),
        "--target-prefix".as_ref(),
        "model".as_ref(),
        "--output-basename".as_ref(),
        "model".as_ref(),
        "--quant".as_ref(),
        "Q4_K_M".as_ref(),
        "--manifest".as_ref(),
        root.join("manifest.json").as_os_str(),
        "--work-dir".as_ref(),
        root.join("work").as_os_str(),
        "--package-dir".as_ref(),
        package.as_os_str(),
        "--package-model-id".as_ref(),
        "org/model".as_ref(),
        "--package-source-repo".as_ref(),
        "org/model".as_ref(),
        "--package-source-revision".as_ref(),
        "main".as_ref(),
        "--skippy-model-package-bin".as_ref(),
        builder.as_os_str(),
        "--replace-package".as_ref(),
    ]
    .map(std::ffi::OsStr::to_os_string);
    let Command::QuantizeLayerPackage(args) = Args::try_parse_from(argv).unwrap().command else {
        panic!("expected the quantize-layer-package route");
    };

    let error = quantize_layer_package(args).unwrap_err();
    assert!(marker.is_file(), "the existing package was removed");
    assert!(
        format!("{error:#}").contains(&format!("start {}", builder.display())),
        "the builder check stopped the run: {error:#}"
    );
}
