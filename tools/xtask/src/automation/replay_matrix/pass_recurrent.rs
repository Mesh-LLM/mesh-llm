use super::{cell_execution::Workload, recurrent_evidence, session_evidence::Request};
use crate::command::DynResult;
use std::path::{Path, PathBuf};

pub(super) fn qualify(workload: &Workload, outputs: (&Path, &Path), log: &Path) -> DynResult<()> {
    let mut paths = vec![log.to_path_buf(), log.with_extension("stderr.log")];
    let native = log
        .parent()
        .ok_or("missing log parent")?
        .join("native-runtime");
    if native.is_dir() {
        collect_logs(&native, &mut paths)?;
    }
    let lookups = recurrent_evidence::read_logs(&paths)?;
    let mut passed = apply(workload, outputs, &lookups)?;
    for cell in &workload.following_cells {
        passed &= apply(
            &cell.workload,
            (&cell.requests_output, &cell.summary_output),
            &lookups,
        )?;
    }
    if passed {
        Ok(())
    } else {
        Err("recurrent cell qualification failed; reports retained".into())
    }
}

fn collect_logs(root: &Path, paths: &mut Vec<PathBuf>) -> DynResult<()> {
    let mut entries = std::fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_logs(&entry.path(), paths)?;
        } else if kind.is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "log")
        {
            paths.push(entry.path());
        }
    }
    Ok(())
}

fn apply(
    workload: &Workload,
    outputs: (&Path, &Path),
    lookups: &[recurrent_evidence::Lookup],
) -> DynResult<bool> {
    let Some(minimum) = workload.minimum_recurrent_restored_tokens else {
        return Ok(true);
    };
    use std::io::{BufRead, BufReader};
    let requests = BufReader::new(std::fs::File::open(outputs.0)?)
        .lines()
        .map(|line| -> DynResult<Request> { Ok(serde_json::from_str(&line?)?) })
        .collect::<DynResult<Vec<_>>>()?;
    let recurrent = recurrent_evidence::evaluate(&requests, lookups, minimum);
    let mut summary: serde_json::Value = serde_json::from_slice(&std::fs::read(outputs.1)?)?;
    let passed = summary["acceptance"]["passed"] == true && recurrent.passed;
    summary["acceptance"]["problems"]
        .as_array_mut()
        .ok_or("missing acceptance problems")?
        .extend(
            recurrent
                .problems
                .iter()
                .cloned()
                .map(serde_json::Value::String),
        );
    summary["acceptance"]["passed"] = passed.into();
    summary["recurrent_state"] = serde_json::to_value(recurrent)?;
    crate::command::write_json_file(outputs.1, &summary)?;
    Ok(passed)
}
