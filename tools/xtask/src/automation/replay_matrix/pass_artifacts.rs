use super::cell_execution::Workload;
use crate::command::DynResult;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

pub(super) fn prepare(workload: &Workload, outputs: (&Path, &Path), log: &Path) -> DynResult<()> {
    let normalized_log = output_identity(log)?;
    let native = normalized_log
        .parent()
        .ok_or("missing log parent")?
        .join("native-runtime");
    if native.try_exists()? {
        return Err("native-runtime evidence directory already exists".into());
    }
    let mut paths = BTreeSet::new();
    for path in [outputs.0, outputs.1, log, &log.with_extension("stderr.log")] {
        reserve(path, &mut paths)?;
    }
    reserve(
        &log.parent()
            .ok_or("missing log parent")?
            .join("lifecycle.json"),
        &mut paths,
    )?;
    if let Some(pin) = &workload.model_pin {
        reserve(&pin.output, &mut paths)?;
    }
    if let Some(context) = &workload.runtime_context {
        let pin = workload
            .model_pin
            .as_ref()
            .ok_or("runtime-context qualification requires a model pin")?;
        if pin.minimum_context_tokens < context.required_tokens {
            return Err("model pin context must cover runtime context requirement".into());
        }
        reserve(&context.output, &mut paths)?;
    }
    validate_cell(workload, workload.runtime_context.is_some())?;
    for cell in &workload.following_cells {
        if cell.workload.base_url != workload.base_url
            || !cell.workload.following_cells.is_empty()
            || cell.workload.runtime_context.is_some()
            || cell.workload.model_pin.is_some()
        {
            return Err(
                "follow-on cells require the same endpoint and parent qualification".into(),
            );
        }
        validate_cell(&cell.workload, workload.runtime_context.is_some())?;
        reserve(&cell.requests_output, &mut paths)?;
        reserve(&cell.summary_output, &mut paths)?;
    }
    for path in paths {
        if path.starts_with(&native) {
            return Err("cell output overlaps native-runtime evidence".into());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn reserve(path: &Path, paths: &mut BTreeSet<PathBuf>) -> DynResult<()> {
    let path = output_identity(path)?;
    if path.try_exists()? || !paths.insert(path) {
        return Err("replay outputs must be distinct unused paths".into());
    }
    Ok(())
}

fn output_identity(path: &Path) -> DynResult<PathBuf> {
    let absolute = std::path::absolute(path)?;
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => (),
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return Err("output path escapes filesystem root".into());
                }
            }
            std::path::Component::Prefix(_)
            | std::path::Component::RootDir
            | std::path::Component::Normal(_) => {
                normalized.push(component.as_os_str());
                if normalized.try_exists()? {
                    normalized = normalized.canonicalize()?;
                }
            }
        }
    }
    Ok(normalized)
}

fn validate_cell(workload: &Workload, runtime: bool) -> DynResult<()> {
    workload.validate()?;
    if workload.eligibility.is_some() && !runtime {
        return Err("cell eligibility requires parent runtime context".into());
    }
    if workload.minimum_recurrent_restored_tokens == Some(0) {
        return Err("recurrent restore requirement must be positive".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload() -> Workload {
        serde_json::from_value(serde_json::json!({
            "trajectories":[{"session_id":"s","source_dataset":"fixture","agent_framework":"fixture","messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]}], "model":"fixture", "base_url":"http://127.0.0.1:9337/v1",
            "concurrency":1,"max_output_tokens":2048,"request_timeout_seconds":1
        }))
        .unwrap()
    }

    #[test]
    fn prior_native_logs_cannot_certify_a_new_pass() {
        let state = tempfile::tempdir().unwrap();
        std::fs::create_dir(state.path().join("native-runtime")).unwrap();
        let result = prepare(
            &workload(),
            (
                &state.path().join("requests.jsonl"),
                &state.path().join("summary.json"),
            ),
            &state.path().join("server.log"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn colliding_outputs_cannot_overwrite_request_evidence() {
        let state = tempfile::tempdir().unwrap();
        let output = state.path().join("output.json");
        let result = prepare(
            &workload(),
            (&output, &output),
            &state.path().join("server.log"),
        );
        assert!(result.is_err());
        assert!(!output.exists());
    }

    #[test]
    fn existing_evidence_is_preserved_when_rerun_is_rejected() {
        let state = tempfile::tempdir().unwrap();
        let output = state.path().join("requests.jsonl");
        std::fs::write(&output, b"prior evidence").unwrap();
        let result = prepare(
            &workload(),
            (&output, &state.path().join("summary.json")),
            &state.path().join("server.log"),
        );
        assert!(result.is_err());
        assert_eq!(std::fs::read(output).unwrap(), b"prior evidence");
    }

    #[test]
    fn parent_components_cannot_hide_colliding_output_paths() {
        let state = tempfile::tempdir().unwrap();
        let direct = state.path().join("summary.json");
        let alias = state.path().join("sub/../summary.json");
        assert!(
            prepare(
                &workload(),
                (&direct, &alias),
                &state.path().join("server.log")
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_parent_cannot_hide_colliding_output_paths() {
        let state = tempfile::tempdir().unwrap();
        let directory = state.path().join("data");
        std::fs::create_dir(&directory).unwrap();
        std::os::unix::fs::symlink(&directory, state.path().join("alias")).unwrap();
        assert!(
            prepare(
                &workload(),
                (
                    &directory.join("summary.json"),
                    &state.path().join("alias/summary.json")
                ),
                &state.path().join("server.log")
            )
            .is_err()
        );
    }
}
