use super::run_workload::{Build, BuildJob};
use crate::command::DynResult;
use std::path::Path;

pub(super) fn prepare(
    root: Option<&Path>,
    jobs: &[BuildJob],
    output: &Path,
) -> DynResult<Vec<Build>> {
    let mut labels = std::collections::BTreeSet::new();
    for job in jobs {
        if job.label.is_empty()
            || !job
                .label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
            || !labels.insert(&job.label)
        {
            return Err("invalid or duplicate replay build job label".into());
        }
    }
    let mut builds = Vec::new();
    for job in jobs {
        let request = serde_json::json!({"repo":job.repo,"worktree_root":job.worktree_root,"label":job.label,
            "ref":job.reference,"backend":job.backend,"git":job.git,"just":job.just,
            "timeout_seconds":job.timeout_seconds,"logs":job.logs,"skip_build":job.skip_build});
        let input = output.join(format!("build-{}.input.json", job.label));
        let result = output.join(format!("build-{}.json", job.label));
        if input.try_exists()? || result.try_exists()? {
            return Err("replay build handoff already exists".into());
        }
        crate::command::write_json_file(&input, &request)?;
        super::run_transport::invoke(root, "build-arm", &input, &result)?;
        builds.push(serde_json::from_slice(&std::fs::read(result)?)?);
    }
    Ok(builds)
}
