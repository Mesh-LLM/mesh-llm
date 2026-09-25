use super::scan::Candidate;
use super::shard_rows::{Group, OtherDisposition, OtherMember, OtherShard};
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

fn other_scope(path: &str) -> bool {
    !path.starts_with(".github/")
        && path != "Justfile"
        && !path.starts_with("just/")
        && !path.starts_with("scripts/")
        && path != "tools/skippy-stage-rewriter/CMakeLists.txt"
}

fn fingerprint(line: &str) -> String {
    let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
    hex::encode(Sha256::digest(normalized.as_bytes()))[..16].to_owned()
}

pub(super) fn source_lines(root: &Path, file: &str) -> DynResult<Vec<String>> {
    if !other_scope(file) || file.starts_with('/') || file.split('/').any(|part| part == "..") {
        return Err(format!("other-edges.json: invalid source path {file}").into());
    }
    Ok(fs::read_to_string(root.join(file))?
        .lines()
        .map(str::trim)
        .map(str::to_owned)
        .collect())
}

fn validate_member_target(disposition: &OtherDisposition, target: &str) -> DynResult<()> {
    if target.trim().is_empty() {
        return Err("other-edges.json: empty target".into());
    }
    match disposition {
        OtherDisposition::Conditional => {
            if !target.contains("configured")
                && !target.contains("selected")
                && !target.contains("supplied")
                && !target.contains("command")
                && !target.contains("argv")
                && !target.contains("executable")
                && !target.contains("<operator path>")
            {
                return Err(format!(
                    "other-edges.json: conditional target lacks unresolved boundary: {target}"
                )
                .into());
            }
        }
        OtherDisposition::Execution
        | OtherDisposition::Instruction
        | OtherDisposition::Provisioning => {
            if target.starts_with("unresolved:") || target == "unknown" {
                return Err(format!("other-edges.json: unresolved target: {target}").into());
            }
        }
        OtherDisposition::Selection | OtherDisposition::Prose | OtherDisposition::Data => {}
    }
    Ok(())
}

fn valid_disposition(path: &str, text: &str, disposition: &OtherDisposition) -> bool {
    if path.ends_with(".py") {
        return match disposition {
            OtherDisposition::Execution
            | OtherDisposition::Conditional
            | OtherDisposition::Selection
            | OtherDisposition::Prose => true,
            OtherDisposition::Instruction
            | OtherDisposition::Provisioning
            | OtherDisposition::Data => false,
        };
    }
    if matches!(
        path,
        "ci/ownership.yml"
            | "tools/xtask/src/ci_validation/producers.rs"
            | "tools/xtask/src/ci_validation/windows_runtime.rs"
            | "tools/xtask/src/ci_validation/crate_coverage.rs"
            | "tools/xtask/src/automation_parity/legacy.rs"
    ) {
        return matches!(
            disposition,
            OtherDisposition::Data | OtherDisposition::Selection | OtherDisposition::Execution
        );
    }
    if text.starts_with("python") || text.starts_with("scripts/") || text.starts_with("- `python") {
        return matches!(
            disposition,
            OtherDisposition::Instruction
                | OtherDisposition::Execution
                | OtherDisposition::Provisioning
        );
    }
    true
}

fn validate_members(
    root: &Path,
    groups: Vec<Group<OtherMember>>,
    observed: &BTreeMap<&str, &Candidate>,
    recorded: &mut BTreeSet<String>,
    implementation: bool,
) -> DynResult<()> {
    for group in groups {
        if group.file.ends_with(".py") != implementation {
            return Err(format!("other-edges.json: wrong source group {}", group.file).into());
        }
        let lines = source_lines(root, &group.file)?;
        for (line, hash, occurrence, disposition, target, boundary) in group.members {
            if line == 0
                || occurrence == 0
                || hash.len() != 16
                || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                || boundary.trim().is_empty()
            {
                return Err(
                    format!("other-edges.json: incomplete member in {}", group.file).into(),
                );
            }
            validate_member_target(&disposition, &target)?;
            let kind = if group.file.ends_with(".py") {
                let text = lines.get(line - 1).ok_or("missing Python source line")?;
                if text.contains("spec_from_file_location(")
                    || text.contains("run_path(")
                    || text.contains("import_module(")
                {
                    "dynamic-import"
                } else {
                    "subprocess-or-interpreter"
                }
            } else {
                "candidate"
            };
            let id = format!("{}#{kind}:{hash}:{occurrence}", group.file);
            if !recorded.insert(id.clone()) {
                return Err(format!("other-edges.json: duplicate member {id}").into());
            }
            let actual = observed
                .get(id.as_str())
                .ok_or_else(|| format!("other-edges.json: stale identity {id}"))?;
            let text = lines
                .get(line - 1)
                .ok_or_else(|| format!("other-edges.json: stale source line {id}:{line}"))?;
            if actual.path != group.file
                || actual.source_block != *text
                || fingerprint(text) != hash
            {
                return Err(format!("other-edges.json: stale source line {id}:{line}").into());
            }
            if !valid_disposition(&group.file, text, &disposition) {
                return Err(format!("other-edges.json: incorrect disposition {id}").into());
            }
        }
    }
    Ok(())
}

pub(super) fn check_other_shard(root: &Path, text: &str, observed: &[Candidate]) -> DynResult<()> {
    let shard: OtherShard = serde_json::from_str(text)?;
    if shard.schema_version != 1 {
        return Err("other-edges.json: unsupported shard version".into());
    }
    let source = observed
        .iter()
        .filter(|row| other_scope(&row.path))
        .map(|row| (row.id.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut recorded = BTreeSet::new();
    validate_members(root, shard.groups, &source, &mut recorded, false)?;
    validate_members(
        root,
        shard.python_implementation_edges,
        &source,
        &mut recorded,
        true,
    )?;
    let missing = source
        .keys()
        .filter(|id| !recorded.contains(**id))
        .take(5)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!("other-edges.json: missing source members {missing:?}").into());
    }
    super::manual_calls::validate_manual(
        root,
        shard.outside_scanner_source_calls,
        shard.manual_tsv_commands,
        shard.manual_tsv_rows,
    )
}
