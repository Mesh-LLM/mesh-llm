use anyhow::Result;
use skippy_runtime::package::artifact_plan::{
    declared_stage_parts, even_stage_range, read_manifest_declaration,
};
use std::path::Path;

pub(crate) fn write_artifacts(
    manifest: &Path,
    index: u32,
    count: u32,
    start: u32,
    end: u32,
    output: &mut dyn std::io::Write,
) -> Result<()> {
    let contents = read_manifest_declaration(manifest)?;
    let parts = declared_stage_parts(&contents, index, count, start, end)?;
    // Admission finishes before the first row. Artifact files need not exist.
    for part in parts {
        let path = part
            .path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("artifact path is not UTF-8"))?;
        writeln!(output, "{path}\t{}", part.artifact_bytes)?;
    }
    output.flush()?;
    Ok(())
}

pub(crate) fn write_range(
    index: u32,
    count: u32,
    layers: u32,
    output: &mut dyn std::io::Write,
) -> Result<()> {
    let (start, end) = even_stage_range(index, count, layers)?;
    writeln!(output, "{start} {end}")?;
    output.flush()?;
    Ok(())
}
