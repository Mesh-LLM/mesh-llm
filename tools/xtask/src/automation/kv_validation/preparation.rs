use super::{
    contract::{CATALOG_SHA, Prepared, Producer, Tool},
    execution::child,
    publish,
};
use crate::{command::DynResult, process::Cancellation};
use std::{ffi::OsString, path::Path, time::Duration};

pub(super) fn pin(mut input: Producer, output: &Path) -> DynResult<()> {
    for tool in [
        &mut input.cli,
        &mut input.correctness,
        &mut input.serving_tests,
    ] {
        tool.sha256 = crate::product::digest::file_sha256(&tool.path).map_err(|e| e.error)?;
    }
    input.runtime_sha256 =
        crate::automation::native_artifact_identity::observe(&input.runtime_bundle)?;
    input.verify()?;
    publish(output, &input)
}

pub(super) fn run(input: Producer, output: &Path) -> DynResult<()> {
    input.verify()?;
    std::fs::create_dir(output)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = prepare(&input, output, &interrupt.cancellation());
    let finished = interrupt.finish();
    finished?;
    result
}
fn prepare(input: &Producer, output: &Path, cancel: &Cancellation) -> DynResult<()> {
    source(input, output, cancel, "source-before")?;
    let native = input.workspace.join("skippy/evals/kv-validation/native");
    let build = output.join("build");
    let args: Vec<OsString> = vec![
        "with-lld".into(),
        input.cmake.clone().into_os_string(),
        "-S".into(),
        native.into_os_string(),
        "-B".into(),
        build.clone().into_os_string(),
        "-DCMAKE_BUILD_TYPE=Release".into(),
        format!("-DNATIVE_SOURCE={}", input.native_source.display()).into(),
        format!(
            "-DNATIVE_LIB_DIR={}",
            input.runtime_bundle.join("lib").display()
        )
        .into(),
    ];
    child(
        &input.just,
        args,
        (&input.workspace, output),
        "configure",
        Duration::from_secs(120),
        cancel,
        &[],
    )?;
    child(
        &input.just,
        vec![
            "with-lld".into(),
            input.cmake.clone().into_os_string(),
            "--build".into(),
            build.clone().into_os_string(),
            "--target".into(),
            "indexed-real-kv-matrix".into(),
            "-j".into(),
            "2".into(),
        ],
        (&input.workspace, output),
        "build",
        Duration::from_secs(180),
        cancel,
        &[],
    )?;
    let path = build.join("indexed-real-kv-matrix");
    let sha256 = crate::product::digest::file_sha256(&path).map_err(|e| e.error)?;
    input.verify()?;
    source(input, output, cancel, "source-after")?;
    publish(
        &output.join("prepared.json"),
        &Prepared {
            schema_version: 1,
            catalog_sha256: CATALOG_SHA.into(),
            producer: input.clone(),
            matrix: Tool { path, sha256 },
        },
    )
}

fn source(input: &Producer, output: &Path, cancel: &Cancellation, label: &str) -> DynResult<()> {
    let args = vec![
        "with-lld".into(),
        "git".into(),
        "-C".into(),
        input.native_source.clone().into_os_string(),
        "rev-parse".into(),
        "HEAD".into(),
    ];
    let observed = child(
        &input.just,
        args,
        (&input.workspace, output),
        label,
        Duration::from_secs(30),
        cancel,
        &[],
    )?;
    if observed.trim() != input.native_source_commit {
        return Err("prepared native source HEAD differs from declared identity".into());
    }
    child(
        &input.just,
        vec![
            "with-lld".into(),
            "git".into(),
            "-C".into(),
            input.native_source.clone().into_os_string(),
            "diff".into(),
            "--quiet".into(),
            "HEAD".into(),
            "--".into(),
        ],
        (&input.workspace, output),
        &format!("{label}-clean"),
        Duration::from_secs(30),
        cancel,
        &[],
    )?;
    Ok(())
}
